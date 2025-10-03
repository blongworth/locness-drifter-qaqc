"""
Prefect workflow for processing and combining drifter data from multiple sensors.

This workflow integrates data from AquaTROLL sensors, fluorometers, and GPS tracks
to create a comprehensive dataset for each drifter deployment.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner

from aquatroll_parser import parse_aquatroll_folder
from fluorometer_parser import parse_fluorometer_folder
from gpx_parser import parse_gpx_to_dataframe
from logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@task()
def load_drifter_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """
    Load drifter metadata from CSV file.

    Args:
        metadata_path: Path to drifter metadata CSV file

    Returns:
        DataFrame with drifter metadata
    """
    logger.info(f"Loading drifter metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata for {len(df)} drifters")
    return df


@task()
def parse_aquatroll_data(
    folder_path: str | Path, drifter_id: str | None = None
) -> pd.DataFrame:
    """
    Parse AquaTROLL sensor data from a folder.

    Args:
        folder_path: Path to folder containing AquaTROLL HTML files
        drifter_id: Optional drifter identifier to add to the data

    Returns:
        DataFrame with AquaTROLL sensor data
    """
    logger.info(f"Parsing AquaTROLL data from {folder_path}")
    df = parse_aquatroll_folder(folder_path)

    if drifter_id:
        df["drifter_id"] = drifter_id

    logger.info(f"Parsed {len(df)} AquaTROLL records")
    return df


@task()
def parse_fluorometer_data(
    folder_path: str | Path, drifter_id: str | None = None, depth: str | None = None
) -> pd.DataFrame:
    """
    Parse fluorometer sensor data from a folder.

    Args:
        folder_path: Path to folder containing fluorometer data files
        drifter_id: Optional drifter identifier to add to the data
        depth: Optional depth indicator ('top' or 'bottom')

    Returns:
        DataFrame with fluorometer sensor data
    """
    logger.info(f"Parsing fluorometer data from {folder_path}")
    df = parse_fluorometer_folder(folder_path)

    if drifter_id:
        df["drifter_id"] = drifter_id
    if depth:
        df["depth"] = depth

    logger.info(f"Parsed {len(df)} fluorometer records")
    return df


@task()
def parse_gpx_data(file_path: str | Path) -> pd.DataFrame:
    """
    Parse GPS position data from GPX file.

    Args:
        file_path: Path to GPX file

    Returns:
        DataFrame with GPS position data
    """
    logger.info(f"Parsing GPX data from {file_path}")
    df = parse_gpx_to_dataframe(file_path)
    logger.info(f"Parsed {len(df)} GPS position records")
    return df


@task()
def combine_and_resample_fluorometer(
    fluorometer_dfs: list[pd.DataFrame],
    resample_freq: str = "1min",
) -> pd.DataFrame:
    """
    Combine and resample fluorometer data from multiple sensors/depths.

    Args:
        fluorometer_dfs: List of fluorometer DataFrames (for different depths/drifters)
        resample_freq: Pandas frequency string for resampling (default: '1min')

    Returns:
        Combined and resampled fluorometer DataFrame
    """
    if not fluorometer_dfs or all(df is None or df.empty for df in fluorometer_dfs):
        logger.warning("No fluorometer data to process")
        return pd.DataFrame()

    logger.info(f"Combining {len(fluorometer_dfs)} fluorometer datasets")

    # Concatenate all fluorometer data
    all_fluoro_data = pd.concat(
        [df for df in fluorometer_dfs if df is not None and not df.empty],
        ignore_index=True,
    )

    # Ensure timestamp is datetime
    if "timestamp" in all_fluoro_data.columns:
        all_fluoro_data["timestamp"] = pd.to_datetime(all_fluoro_data["timestamp"])
        all_fluoro_data = all_fluoro_data.set_index("timestamp")

        # Group by drifter_id and depth, then resample
        if "drifter_id" in all_fluoro_data.columns and "depth" in all_fluoro_data.columns:
            # Resample within each group
            resampled_groups = []
            for (drifter_id, depth), group in all_fluoro_data.groupby(["drifter_id", "depth"]):
                numeric_cols = group.select_dtypes(include=["number"]).columns
                non_numeric_cols = group.select_dtypes(exclude=["number"]).columns

                resampled_numeric = group[numeric_cols].resample(resample_freq).mean()
                if len(non_numeric_cols) > 0:
                    resampled_non_numeric = group[non_numeric_cols].resample(resample_freq).first()
                    resampled = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
                else:
                    resampled = resampled_numeric

                # Add back the grouping columns
                resampled["drifter_id"] = drifter_id
                resampled["depth"] = depth
                resampled_groups.append(resampled)

            combined_resampled = pd.concat(resampled_groups).sort_index()
        else:
            # Simple resample if no grouping columns
            combined_resampled = _resample_dataframe(
                all_fluoro_data.reset_index(), "timestamp", resample_freq
            )

        logger.info(f"Resampled fluorometer data: {len(combined_resampled)} records")
        return combined_resampled
    else:
        logger.warning("No timestamp column in fluorometer data")
        return pd.DataFrame()


@task()
def resample_aquatroll(
    aquatroll_df: pd.DataFrame | None,
    resample_freq: str = "1min",
) -> pd.DataFrame:
    """
    Resample AquaTROLL data to a common time frequency.

    Args:
        aquatroll_df: AquaTROLL sensor data
        resample_freq: Pandas frequency string for resampling (default: '1min')

    Returns:
        Resampled AquaTROLL DataFrame
    """
    if aquatroll_df is None or aquatroll_df.empty:
        logger.warning("No AquaTROLL data to process")
        return pd.DataFrame()

    logger.info("Resampling AquaTROLL data")

    if "datetime" not in aquatroll_df.columns:
        logger.warning("No datetime column in AquaTROLL data")
        return pd.DataFrame()

    # Ensure datetime is datetime
    aquatroll_df["datetime"] = pd.to_datetime(aquatroll_df["datetime"])
    aquatroll_df = aquatroll_df.set_index("datetime")

    # Group by drifter_id if available
    if "drifter_id" in aquatroll_df.columns:
        resampled_groups = []
        for drifter_id, group in aquatroll_df.groupby("drifter_id"):
            numeric_cols = group.select_dtypes(include=["number"]).columns
            non_numeric_cols = group.select_dtypes(exclude=["number"]).columns

            resampled_numeric = group[numeric_cols].resample(resample_freq).mean()
            if len(non_numeric_cols) > 0:
                resampled_non_numeric = group[non_numeric_cols].resample(resample_freq).first()
                resampled = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
            else:
                resampled = resampled_numeric

            resampled["drifter_id"] = drifter_id
            resampled_groups.append(resampled)

        combined_resampled = pd.concat(resampled_groups).sort_index()
    else:
        # Simple resample if no grouping column
        combined_resampled = _resample_dataframe(
            aquatroll_df.reset_index(), "datetime", resample_freq
        )

    logger.info(f"Resampled AquaTROLL data: {len(combined_resampled)} records")
    return combined_resampled


@task()
def resample_position(
    gpx_df: pd.DataFrame | None,
    resample_freq: str = "1min",
) -> pd.DataFrame:
    """
    Resample GPS position data to a common time frequency.

    Args:
        gpx_df: GPS position data
        resample_freq: Pandas frequency string for resampling (default: '1min')

    Returns:
        Resampled GPS position DataFrame
    """
    if gpx_df is None or gpx_df.empty:
        logger.warning("No GPS position data to process")
        return pd.DataFrame()

    logger.info("Resampling GPS position data")

    if "timestamp" not in gpx_df.columns:
        logger.warning("No timestamp column in GPS data")
        return pd.DataFrame()

    # Ensure timestamp is datetime
    gpx_df["timestamp"] = pd.to_datetime(gpx_df["timestamp"])
    gpx_df = gpx_df.set_index("timestamp")

    # Group by asset_id if available
    if "asset_id" in gpx_df.columns:
        resampled_groups = []
        for asset_id, group in gpx_df.groupby("asset_id"):
            numeric_cols = group.select_dtypes(include=["number"]).columns
            non_numeric_cols = group.select_dtypes(exclude=["number"]).columns

            resampled_numeric = group[numeric_cols].resample(resample_freq).mean()
            if len(non_numeric_cols) > 0:
                resampled_non_numeric = group[non_numeric_cols].resample(resample_freq).first()
                resampled = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
            else:
                resampled = resampled_numeric

            resampled["asset_id"] = asset_id
            resampled_groups.append(resampled)

        combined_resampled = pd.concat(resampled_groups).sort_index()
    else:
        # Simple resample if no grouping column
        combined_resampled = _resample_dataframe(
            gpx_df.reset_index(), "timestamp", resample_freq
        )

    logger.info(f"Resampled GPS position data: {len(combined_resampled)} records")
    return combined_resampled


def _resample_dataframe(
    df: pd.DataFrame, time_col: str, freq: str, prefix: str = ""
) -> pd.DataFrame:
    """
    Resample a dataframe to a specified frequency.

    Args:
        df: Input DataFrame
        time_col: Name of the timestamp column
        freq: Pandas frequency string
        prefix: Prefix to add to column names

    Returns:
        Resampled DataFrame with timestamp as index
    """
    # Create a copy and set timestamp as index
    df_copy = df.copy()

    # Ensure timestamp is datetime
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])

    # Set index
    df_copy = df_copy.set_index(time_col)

    # Separate numeric and non-numeric columns
    numeric_cols = df_copy.select_dtypes(include=["number"]).columns
    non_numeric_cols = df_copy.select_dtypes(exclude=["number"]).columns

    # Resample numeric columns with mean
    resampled_numeric = df_copy[numeric_cols].resample(freq).mean()

    # Resample non-numeric columns with first value
    if len(non_numeric_cols) > 0:
        resampled_non_numeric = df_copy[non_numeric_cols].resample(freq).first()
        resampled = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
    else:
        resampled = resampled_numeric

    # Add prefix to column names
    if prefix:
        resampled.columns = [f"{prefix}_{col}" for col in resampled.columns]

    return resampled


@task()
def save_to_parquet(df: pd.DataFrame, output_path: str | Path, dataset_name: str) -> None:
    """
    Save DataFrame to parquet file.

    Args:
        df: DataFrame to save
        output_path: Path for output parquet file
        dataset_name: Name of the dataset for logging
    """
    if df is None or df.empty:
        logger.warning(f"No {dataset_name} data to save")
        return

    output_path = Path(output_path)
    logger.info(f"Saving {dataset_name} data to {output_path}")

    # Reset index to save timestamp as a column
    df_to_save = df.reset_index()

    # Convert any timezone-aware datetime columns to UTC and remove timezone for parquet compatibility
    for col in df_to_save.columns:
        if pd.api.types.is_datetime64_any_dtype(df_to_save[col]):
            if hasattr(df_to_save[col].dtype, "tz") and df_to_save[col].dtype.tz is not None:
                # Convert to UTC and then remove timezone info
                df_to_save[col] = df_to_save[col].dt.tz_convert("UTC").dt.tz_localize(None)
            elif df_to_save[col].dt.tz is None:
                # Already naive, keep as is
                pass

    # Save to parquet
    df_to_save.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df_to_save)} {dataset_name} records to {output_path}")


@flow(
    name="Process Drifter Data",
    description="Process and combine drifter sensor data from multiple sources",
    task_runner=ConcurrentTaskRunner(),
)
def process_drifter_data(
    data_dir: str | Path = "data",
    output_dir: str | Path = "output",
    resample_freq: str = "1min",
) -> dict[str, pd.DataFrame]:
    """
    Main Prefect flow to process and combine drifter data.

    This flow:
    1. Loads drifter metadata
    2. Parses AquaTROLL sensor data by drifter
    3. Parses fluorometer data by drifter and depth
    4. Parses GPX position data
    5. Resamples each dataset to the nearest minute
    6. Saves separate parquet files for each data type

    Args:
        data_dir: Base directory containing drifter data
        output_dir: Directory for output parquet files
        resample_freq: Pandas frequency string for resampling (default: '1min')

    Returns:
        Dictionary with resampled DataFrames for each data type
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Starting drifter data processing workflow")

    # Load metadata
    metadata_path = data_dir / "drifter_metadata.csv"
    metadata = load_drifter_metadata(metadata_path)

    # Parse GPX data (all drifters in one file)
    gpx_path = data_dir / "LOC02_drifter_tracks.gpx"
    gpx_data = None
    if gpx_path.exists():
        gpx_data = parse_gpx_data(gpx_path)
    else:
        logger.warning(f"GPX file not found: {gpx_path}")

    # Process data for each drifter
    all_aquatroll_data = []
    all_fluorometer_data = []

    drifters_dir = data_dir / "Drifters_LOC02"

    # Parse AquaTROLL data by serial number
    for _, row in metadata.iterrows():
        drifter_id = row["drifter"]
        aquatroll_sn = row["aquatroll_sn"]

        if pd.notna(aquatroll_sn):
            aquatroll_folder = drifters_dir / str(aquatroll_sn)
            if aquatroll_folder.exists():
                aquatroll_df = parse_aquatroll_data(aquatroll_folder, drifter_id)
                all_aquatroll_data.append(aquatroll_df)

    # Parse fluorometer data by drifter and depth
    for _, row in metadata.iterrows():
        drifter_id = row["drifter"]
        pme_top_sn = row["pme_top_sn"]
        pme_bottom_sn = row["pme_bottom_sn"]

        # Top fluorometer
        if pd.notna(pme_top_sn):
            # Look for fluorometer folders
            for folder_pattern in [
                f"Drifter_{drifter_id:02d}_Top_Fluorometer",
                f"Drifter_{drifter_id:02d}_Top_Fluorometer_Final",
                f"Drifter_0{drifter_id}_Top_Fluorometer",
            ]:
                fluoro_folder = drifters_dir / folder_pattern
                if fluoro_folder.exists():
                    fluoro_df = parse_fluorometer_data(
                        fluoro_folder, drifter_id, "top"
                    )
                    all_fluorometer_data.append(fluoro_df)
                    break

        # Bottom fluorometer
        if pd.notna(pme_bottom_sn):
            for folder_pattern in [
                f"Drifter_{drifter_id:02d}_Bottom_Fluorometer",
                f"Drifter_{drifter_id:02d}_Bottom_Fluorometer_Final",
                f"Drifter_{drifter_id:02d}_Fluorometer",
                f"Drifter_0{drifter_id}_Fluorometer",
                f"Drifter_0{drifter_id}_Fluorometer_Final",
            ]:
                fluoro_folder = drifters_dir / folder_pattern
                if fluoro_folder.exists():
                    fluoro_df = parse_fluorometer_data(
                        fluoro_folder, drifter_id, "bottom"
                    )
                    all_fluorometer_data.append(fluoro_df)
                    break

    # Combine all AquaTROLL data
    combined_aquatroll = (
        pd.concat(all_aquatroll_data, ignore_index=True)
        if all_aquatroll_data
        else None
    )

    # Resample each dataset separately
    resampled_aquatroll = resample_aquatroll(combined_aquatroll, resample_freq)
    resampled_fluorometer = combine_and_resample_fluorometer(all_fluorometer_data, resample_freq)
    resampled_position = resample_position(gpx_data, resample_freq)

    # Save each dataset to separate parquet files
    save_to_parquet(resampled_aquatroll, output_dir / "aquatroll_data.parquet", "AquaTROLL")
    save_to_parquet(resampled_fluorometer, output_dir / "fluorometer_data.parquet", "fluorometer")
    save_to_parquet(resampled_position, output_dir / "position_data.parquet", "position")

    logger.info("Drifter data processing workflow completed")

    return {
        "aquatroll": resampled_aquatroll,
        "fluorometer": resampled_fluorometer,
        "position": resampled_position,
    }


def main():
    """Entry point for the drifter data processing workflow."""
    # Run the Prefect flow
    results = process_drifter_data(
        data_dir="data",
        output_dir="output",
        resample_freq="1min",
    )

    print("\nProcessing complete!")
    print("\nGenerated parquet files:")
    for name, df in results.items():
        if df is not None and not df.empty:
            print(f"  - {name}_data.parquet: {len(df)} records, {len(df.columns)} columns")
        else:
            print(f"  - {name}_data.parquet: No data")


if __name__ == "__main__":
    main()
