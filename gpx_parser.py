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


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python gpx_parser.py <gpx_file>")
        sys.exit(1)

    try:
        df = parse_gpx_to_dataframe(sys.argv[1])
        print(f"Parsed {len(df)} points:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())

        # Show summary
        summary = get_gpx_summary(sys.argv[1])
        print("\nGPX Summary:")
        print(f"Total tracks: {summary['total_tracks']}")
        print(f"Total points: {summary['total_points']}")
        
        df.to_csv("parsed_gpx_data.csv", index=False)
        print("Parsed data written to parsed_gpx_data.csv")
        df.to_parquet("parsed_gpx_data.parquet", index=False)
        print("Parsed data written to parsed_gpx_data.parquet")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
