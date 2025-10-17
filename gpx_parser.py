"""
Simple GPX file parser module for drifter position data.

This module provides functionality to parse GPX format files using the gpxpy library
and return the data as a pandas DataFrame.
"""

import logging
from pathlib import Path
from typing import Any

import gpxpy
import pandas as pd
from logging_config import setup_logging


# Setup logging
setup_logging()

logger = logging.getLogger(__name__)


def parse_gpx_to_dataframe(filename: str | Path) -> pd.DataFrame:
    """
    Parse a GPX file and return position data as a pandas DataFrame.

    Args:
        filename: Path to the GPX file to parse

    Returns:
        pandas DataFrame with columns: asset_id, timestamp, latitude, longitude, elevation

    Raises:
        FileNotFoundError: If the file doesn't exist
        gpxpy.gpx.GPXXMLSyntaxException: If the GPX file is malformed
        Exception: For other parsing errors
    """
    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"GPX file not found: {file_path}")

    logger.info(f"Parsing GPX file: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        # Extract data from all tracks
        data = []

        for track in gpx.tracks:
            # Use track name as asset_id, default to filename if no name
            asset_id = track.name if track.name else file_path.stem

            logger.debug(f"Processing track: {asset_id}")

            for segment in track.segments:
                for point in segment.points:
                    data.append(
                        {
                            "asset_id": asset_id,
                            "timestamp": point.time,
                            "latitude": point.latitude,
                            "longitude": point.longitude,
                            "elevation": point.elevation,
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure timestamp is a timezone-aware datetime in UTC
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Sort by timestamp if we have timestamp data
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Successfully parsed {len(df)} points from {file_path}")
        # Extract drifter number from asset_id (final integer in the string)
        df["drifter"] = (
            df["asset_id"]
            .str.extract(r"(\d+)$", expand=False)
            .astype(float)
            .astype("Int8")
        )

        return df

    except gpxpy.gpx.GPXXMLSyntaxException as e:
        logger.error(f"GPX XML syntax error in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing GPX file {file_path}: {e}")
        raise


def get_gpx_summary(filename: str | Path) -> dict[str, Any]:
    """
    Get summary information about a GPX file without full parsing.

    Args:
        filename: Path to the GPX file

    Returns:
        Dictionary containing summary information
    """
    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"GPX file not found: {file_path}")

    try:
        with open(file_path, encoding="utf-8") as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        summary = {
            "file_path": str(file_path),
            "creator": gpx.creator,
            "total_tracks": len(gpx.tracks),
            "total_points": 0,
            "tracks": [],
        }

        for track in gpx.tracks:
            track_points = 0
            times = []

            for segment in track.segments:
                track_points += len(segment.points)
                for point in segment.points:
                    if point.time:
                        times.append(point.time)

            track_info = {
                "name": track.name or "unnamed",
                "point_count": track_points,
                "start_time": min(times) if times else None,
                "end_time": max(times) if times else None,
            }

            summary["tracks"].append(track_info)
            summary["total_points"] += track_points

        return summary

    except Exception as e:
        logger.error(f"Error getting GPX summary for {file_path}: {e}")
        raise

def add_position_flag(
    df: pd.DataFrame, min_elevation: float = -1000, max_elevation: float = 1000
) -> pd.DataFrame:
    """
    Adds a 'position_flag' column to the DataFrame based on elevation.

    The flag is set to 2 if the 'elevation' is within the specified range (inclusive),
    or if 'elevation' is NaN. Otherwise, the flag is set to 4.

    Args:
        df: The input pandas DataFrame, must contain an 'elevation' column.
        min_elevation: The minimum valid elevation. Defaults to -1000.
        max_elevation: The maximum valid elevation. Defaults to 1000.

    Returns:
        The DataFrame with the 'position_flag' column added.
    """
    # Condition for valid elevation: between min_elevation and max_elevation, or NaN
    condition = (
        df["elevation"].between(min_elevation, max_elevation) | df["elevation"].isna()
    )

    # Set flag to 2 where condition is True, 4 otherwise
    df["position_flag"] = 4
    df.loc[condition, "position_flag"] = 2
    df["position_flag"] = df["position_flag"].astype("Int8")

    return df


def apply_flags_from_file(df: pd.DataFrame, flag_file: str | Path) -> pd.DataFrame:
    """
    Apply position flags to the DataFrame based on drifter number and time range from a flag file.
    
    The flag file should be a CSV with columns: drifter, start_time, end_time, position_flag
    
    Args:
        df: The input pandas DataFrame with drifter position data
        flag_file: Path to the CSV file containing flag definitions
        
    Returns:
        The DataFrame with position_flag column updated based on the flag file
    """
    flag_path = Path(flag_file)
    
    if not flag_path.exists():
        raise FileNotFoundError(f"Flag file not found: {flag_path}")
    
    logger.info(f"Applying flags from: {flag_path}")
    
    # Read the flag file
    flags_df = pd.read_csv(flag_path)
    
    # Convert time columns to datetime
    flags_df['start_time'] = pd.to_datetime(flags_df['start_time'], utc=True)
    flags_df['end_time'] = pd.to_datetime(flags_df['end_time'], utc=True)
    
    # Apply flags for each row in the flag file
    for _, flag_row in flags_df.iterrows():
        mask = (
            (df['drifter'] == flag_row['drifter']) &
            (df['timestamp'] >= flag_row['start_time']) &
            (df['timestamp'] <= flag_row['end_time'])
        )
        df.loc[mask, 'position_flag'] = flag_row['position_flag']
    
    logger.info(f"Successfully applied flags from {flag_path}")
    return df


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python gpx_parser.py <gpx_file>")
        sys.exit(1)

    try:
        df = parse_gpx_to_dataframe(sys.argv[1])
        df = add_position_flag(df)
        df = apply_flags_from_file(df, "data/loc02_drifter_pos_flags.csv")

        print(f"Parsed {len(df)} points:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())

        # Show summary
        summary = get_gpx_summary(sys.argv[1])
        print("\nGPX Summary:")
        print(f"Total tracks: {summary['total_tracks']}")
        print(f"Total points: {summary['total_points']}")
        
        df.to_csv("output/loc02_drifter_positions.csv", index=False)
        print("Parsed data written to output/loc02_drifter_positions.csv")
        df.to_parquet("output/loc02_drifter_positions.parquet", index=False)
        print("Parsed data written to output/loc02_drifter_positions.parquet")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
