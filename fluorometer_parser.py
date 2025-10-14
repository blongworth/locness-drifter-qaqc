"""
Fluorometer data parser module for drifter fluorometer data files.

This module provides functionality to parse fluorometer data files from a folder
and return the concatenated data as a pandas DataFrame.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from logging_config import setup_logging


# Setup logging
setup_logging()

logger = logging.getLogger(__name__)


def parse_fluorometer_folder(folder_path: str | Path) -> pd.DataFrame:
    """
    Parse all fluorometer data files in a folder recursively and return a concatenated DataFrame.

    Args:
        folder_path: Path to the folder containing fluorometer data files

    Returns:
        pandas DataFrame with columns: serial_number, time_sec, battery_volts,
        temperature_c, sensor_ppb_rwt, gain, filename

    Raises:
        FileNotFoundError: If the folder doesn't exist
        ValueError: If no valid data files found
        Exception: For other parsing errors
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    logger.info(f"Recursively parsing fluorometer data files in: {folder_path}")

    all_data = []
    files_processed = 0

    # Recursively find all files in the folder
    for file_path in folder_path.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            # Attempt to parse the file
            data = parse_fluorometer_file(file_path)
            if not data.empty:
                all_data.append(data)
                files_processed += 1
                logger.debug(f"Processed {file_path.name}: {len(data)} records")
        except ValueError as e:
            # Log expected parsing errors at a debug level as many files might not be valid
            logger.debug(f"Skipped non-fluorometer file {file_path.name}: {e}")
        except Exception as e:
            # Log other unexpected errors as warnings
            logger.warning(f"Failed to parse {file_path.name}: {e}")

    if not all_data:
        raise ValueError(f"No valid fluorometer data files found in {folder_path}")

    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by time if available
    if "time_sec" in combined_df.columns:
        combined_df = combined_df.sort_values("time_sec").reset_index(drop=True)

    logger.info(
        f"Successfully parsed {files_processed} files with {len(combined_df)} total records"
    )
    return combined_df


def parse_fluorometer_file(file_path: str | Path) -> pd.DataFrame:
    """
    Parse a single fluorometer data file.

    Args:
        file_path: Path to the fluorometer data file

    Returns:
        pandas DataFrame with parsed data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 4:
            raise ValueError("File too short - expected at least 4 lines")

        # Parse header information
        serial_number = lines[0].strip()
        os_rev_line = lines[1].strip()
        header_line = lines[2].strip()

        # Validate header format
        if not header_line.startswith("Time (sec)"):
            raise ValueError("Invalid header format - expected 'Time (sec)' header")

        # Parse data lines (starting from line 3)
        data_rows = []

        for line_num, line in enumerate(lines[3:], start=4):
            line = line.strip()
            if not line:
                continue

            try:
                # Split by comma and clean up spaces
                parts = [part.strip() for part in line.split(",")]

                if len(parts) != 5:
                    logger.warning(
                        f"Line {line_num} in {file_path.name}: expected 5 columns, got {len(parts)}"
                    )
                    continue

                # Parse each field
                time_sec = int(parts[0])
                battery_volts = float(parts[1])
                temperature_c = float(parts[2])
                sensor_ppb_rwt = float(parts[3])
                gain = int(parts[4])

                data_rows.append(
                    {
                        "serial_number": serial_number,
                        "time_sec": time_sec,
                        "battery_volts": battery_volts,
                        "temperature_c": temperature_c,
                        "sensor_ppb_rwt": sensor_ppb_rwt,
                        "gain": gain,
                        "filename": file_path.name,
                        "os_rev": os_rev_line,
                    }
                )

            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Line {line_num} in {file_path.name}: parsing error - {e}"
                )
                continue

        if not data_rows:
            raise ValueError("No valid data rows found")

        df = pd.DataFrame(data_rows)

        # Convert timestamp to datetime in UTC
        if "time_sec" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["time_sec"], unit="s", utc=True)
            except Exception as e:
                logger.debug(f"Could not convert timestamps in {file_path.name}: {e}")

        return df

    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, encoding="latin-1") as f:
                lines = f.readlines()
            # Repeat parsing logic...
            return parse_fluorometer_file_lines(lines, file_path)
        except Exception as e:
            raise ValueError(f"Could not decode file {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing file {file_path}: {e}")


def parse_fluorometer_file_lines(lines: list[str], file_path: Path) -> pd.DataFrame:
    """
    Helper function to parse fluorometer data from lines (for encoding fallback).

    Args:
        lines: List of file lines
        file_path: Path object for logging

    Returns:
        pandas DataFrame with parsed data
    """
    if len(lines) < 4:
        raise ValueError("File too short - expected at least 4 lines")

    # Parse header information
    serial_number = lines[0].strip()
    os_rev_line = lines[1].strip()
    header_line = lines[2].strip()

    # Validate header format
    if not header_line.startswith("Time (sec)"):
        raise ValueError("Invalid header format - expected 'Time (sec)' header")

    # Parse data lines (starting from line 3)
    data_rows = []

    for line_num, line in enumerate(lines[3:], start=4):
        line = line.strip()
        if not line:
            continue

        try:
            # Split by comma and clean up spaces
            parts = [part.strip() for part in line.split(",")]

            if len(parts) != 5:
                logger.warning(
                    f"Line {line_num} in {file_path.name}: expected 5 columns, got {len(parts)}"
                )
                continue

            # Parse each field
            time_sec = int(parts[0])
            battery_volts = float(parts[1])
            temperature_c = float(parts[2])
            sensor_ppb_rwt = float(parts[3])
            gain = int(parts[4])

            data_rows.append(
                {
                    "serial_number": serial_number,
                    "time_sec": time_sec,
                    "battery_volts": battery_volts,
                    "temperature_c": temperature_c,
                    "sensor_ppb_rwt": sensor_ppb_rwt,
                    "gain": gain,
                    "filename": file_path.name,
                    "os_rev": os_rev_line,
                }
            )

        except (ValueError, IndexError) as e:
            logger.warning(f"Line {line_num} in {file_path.name}: parsing error - {e}")
            continue

    if not data_rows:
        raise ValueError("No valid data rows found")

    df = pd.DataFrame(data_rows)

    # Convert timestamp to datetime in UTC
    if "time_sec" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["time_sec"], unit="s", utc=True)
        except Exception as e:
            logger.debug(f"Could not convert timestamps in {file_path.name}: {e}")

    return df


def get_fluorometer_summary(folder_path: str | Path) -> dict[str, Any]:
    """
    Get summary information about fluorometer data files in a folder.

    Args:
        folder_path: Path to the folder containing fluorometer data files

    Returns:
        Dictionary containing summary information
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    try:
        df = parse_fluorometer_folder(folder_path)

        summary = {
            "folder_path": str(folder_path),
            "total_records": len(df),
            "unique_serial_numbers": df["serial_number"].nunique()
            if "serial_number" in df.columns
            else 0,
            "files_processed": df["filename"].nunique()
            if "filename" in df.columns
            else 0,
            "time_range": {},
            "serial_numbers": [],
        }

        # Time range information
        if "timestamp" in df.columns:
            summary["time_range"] = {
                "start_timestamp": df["timestamp"].min(),
                "end_timestamp": df["timestamp"].max(),
                "duration_seconds": df["timestamp"].max() - df["timestamp"].min(),
            }

            if "timestamp" in df.columns:
                summary["time_range"]["start_datetime"] = df["timestamp"].min()
                summary["time_range"]["end_datetime"] = df["timestamp"].max()

        # Serial number breakdown
        if "serial_number" in df.columns:
            for serial in df["serial_number"].unique():
                serial_data = df[df["serial_number"] == serial]
                serial_info = {
                    "serial_number": serial,
                    "record_count": len(serial_data),
                    "files": serial_data["filename"].unique().tolist()
                    if "filename" in df.columns
                    else [],
                }

                if "time_sec" in serial_data.columns:
                    serial_info["time_range"] = {
                        "start": serial_data["time_sec"].min(),
                        "end": serial_data["time_sec"].max(),
                    }

                summary["serial_numbers"].append(serial_info)

        return summary

    except Exception as e:
        logger.error(f"Error getting fluorometer summary for {folder_path}: {e}")
        raise

def add_drifter_number(df: pd.DataFrame, metadata_file: str | Path) -> pd.DataFrame:
    """
    Add a 'drifter_number' column to the DataFrame by joining on a metadata table.

    Args:
        df: pandas DataFrame with an 'serial_number' column
        metadata_file: Path to CSV file containing pme sn to 'drifter' mappings

    Returns:
        DataFrame with an additional 'drifter' column
    """
    if "serial_number" not in df.columns:
        raise ValueError("DataFrame must contain a 'serial_number' column")

    # Load metadata mapping
    metadata = pd.read_csv(metadata_file, dtype=str)
    
    # filter for pme devices only
    metadata = metadata[metadata["device_type"].str.lower() == "pme"]

    # select relevant columns
    metadata = metadata[["device_sn", "drifter", "sensor_position"]].drop_duplicates()
    metadata["drifter"] = metadata["drifter"].astype("Int8")

    # Merge with metadata to get drifter numbers
    df = df.merge(metadata, left_on="serial_number", right_on="device_sn", how="left")

    return df


def add_rho_flag(df: pd.DataFrame, flag_periods_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a rho_flag column to the DataFrame based on flagging periods and data conditions.
    
    Args:
        df: DataFrame with fluorometer data including 'drifter', 'timestamp', and 'sensor_ppb_rwt' columns
        flag_periods_df: DataFrame with columns 'drifter', 'start', 'end' representing time periods
                        where rho_flag should be set to 4
    
    Returns:
        DataFrame with added 'rho_flag' column
        
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['drifter', 'timestamp', 'sensor_ppb_rwt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    required_flag_cols = ['drifter', 'start', 'end']
    missing_flag_cols = [col for col in required_flag_cols if col not in flag_periods_df.columns]
    if missing_flag_cols:
        raise ValueError(f"Flag periods DataFrame missing required columns: {missing_flag_cols}")
    
    # Initialize rho_flag with default value of 2
    df = df.copy()
    df['rho_flag'] = 2
    
    # Set flag to NaN where sensor_ppb_rwt is NaN
    df.loc[df['sensor_ppb_rwt'].isna(), 'rho_flag'] = pd.NA
    
    # Convert start/end times to datetime if they aren't already
    flag_periods_df = flag_periods_df.copy()
    flag_periods_df['start'] = pd.to_datetime(flag_periods_df['start'])
    flag_periods_df['end'] = pd.to_datetime(flag_periods_df['end'])
    
    # Set flag to 4 for specified time periods by drifter
    for _, flag_row in flag_periods_df.iterrows():
        mask = (
            (df['drifter'] == flag_row['drifter']) &
            (df['timestamp'] >= flag_row['start']) &
            (df['timestamp'] <= flag_row['end']) 
        )
        df.loc[mask, 'rho_flag'] = 4
    
    # Set flag to NaN where sensor_ppb_rwt is NaN (override any previous settings)
    df.loc[df['sensor_ppb_rwt'].isna(), 'rho_flag'] = pd.NA
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys

    # Create a DataFrame with flagging periods
    rho_flag_periods = pd.DataFrame({
        'drifter': [4],
        'start': [pd.to_datetime('2025-08-15 11:30', utc=True)],
        'end': [pd.to_datetime('2025-08-18 00:00', utc=True)]
    })

    if len(sys.argv) != 2:
        print("Usage: python fluorometer_parser.py <folder_path>")
        sys.exit(1)

    try:
        df = parse_fluorometer_folder(sys.argv[1])
        df = add_drifter_number(df, "data/drifter_metadata.csv")
        print("\nColumn types:")
        print(df.dtypes)
        df = add_rho_flag(df, rho_flag_periods)
        print(f"Parsed {len(df)} records from fluorometer data files:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())

        # Show summary
        summary = get_fluorometer_summary(sys.argv[1])
        print("\nFluorometer Data Summary:")
        print(f"Total records: {summary['total_records']}")
        print(f"Files processed: {summary['files_processed']}")
        print(f"Unique serial numbers: {summary['unique_serial_numbers']}")

        if summary["time_range"]:
            print(
                f"Time range: {summary['time_range'].get('start_timestamp')} to {summary['time_range'].get('end_timestamp')}"
            )

        df.to_csv("output/loc02_drifter_pme.csv", index=False)
        print("Combined data written to output/loc02_drifter_pme.csv")
        df.to_parquet("output/loc02_drifter_pme.parquet", index=False)
        print("Combined data written to output/loc02_drifter_pme.parquet")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
