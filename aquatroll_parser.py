"""
AquaTROLL Data Parser

This module provides functionality to parse AquaTROLL drifter data files in HTML format.
The files contain metadata and sensor readings in an HTML table structure.
"""

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_aquatroll_file(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a single AquaTROLL HTML file and return metadata and sensor data.

    Args:
        file_path: Path to the AquaTROLL HTML file

    Returns:
        Dictionary containing:
        - 'metadata': Dict with parsed metadata sections
        - 'data': pandas DataFrame with sensor readings
        - 'columns': Dict mapping column names to their units and sensor info

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Parsing AquaTROLL file: {file_path.name}")

    try:
        # Read and parse HTML
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "html.parser")

        # Extract metadata from HTML meta tags
        meta_data = _extract_meta_tags(soup)

        # Find the main data table
        table = soup.find("table", id="isi-report")
        if not table:
            raise ValueError("Could not find data table with id 'isi-report'")

        # Parse metadata sections from table
        metadata = _parse_metadata_sections(table)
        metadata.update(meta_data)

        # Parse sensor data
        data_df, column_info = _parse_sensor_data(table)

        logger.info(f"Successfully parsed {len(data_df)} data records")

        return {"metadata": metadata, "data": data_df, "columns": column_info}

    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {e}")
        raise ValueError(f"Failed to parse AquaTROLL file: {e}")


def parse_aquatroll_folder(folder_path: str | Path) -> pd.DataFrame:
    """
    Parse all AquaTROLL HTML files in a folder and return combined data.

    Args:
        folder_path: Path to folder containing AquaTROLL HTML files

    Returns:
        pandas DataFrame with combined sensor data from all files

    Raises:
        FileNotFoundError: If the folder doesn't exist
        ValueError: If no valid files are found
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find all HTML files
    html_files = list(folder_path.glob("*.html")) + list(folder_path.glob("*.htm"))

    if not html_files:
        raise ValueError(f"No HTML files found in {folder_path}")

    logger.info(f"Found {len(html_files)} HTML files in {folder_path}")

    all_data = []

    for file_path in html_files:
        try:
            result = parse_aquatroll_file(file_path)
            data_df = result["data"].copy()

            # Add source file information
            data_df["source_file"] = file_path.name

            # Add metadata as columns
            metadata = result["metadata"]
            if "location_name" in metadata:
                data_df["location"] = metadata["location_name"]
            if "device_serial_number" in metadata:
                data_df["device_sn"] = metadata["device_serial_number"]
            if "log_name" in metadata:
                data_df["log_name"] = metadata["log_name"]

            all_data.append(data_df)

        except Exception as e:
            logger.warning(f"Skipping file {file_path.name}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid AquaTROLL files could be parsed")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert datetime column to UTC if it exists
    if "datetime" in combined_df.columns:
        combined_df["datetime"] = pd.to_datetime(combined_df["datetime"], utc=True)

    logger.info(
        f"Combined data from {len(all_data)} files: {len(combined_df)} total records"
    )

    return combined_df


def _extract_meta_tags(soup: BeautifulSoup) -> dict[str, str]:
    """Extract metadata from HTML meta tags."""
    meta_data = {}

    # Extract specific meta tags
    meta_mappings = {
        "isi-csv-file-name": "csv_filename",
        "isi-report-id": "report_id",
        "isi-report-version": "report_version",
        "isi-report-type": "report_type",
        "isi-report-created": "report_created",
    }

    for name, key in meta_mappings.items():
        meta_tag = soup.find("meta", attrs={"name": name})
        if meta_tag and meta_tag.get("content"):
            meta_data[key] = meta_tag["content"]

    return meta_data


def _parse_metadata_sections(table) -> dict[str, Any]:
    """Parse metadata sections from the HTML table."""
    metadata = {}

    # Find all section headers and their members
    section_headers = table.find_all("tr", class_="sectionHeader")

    for header in section_headers:
        section_name = header.find("td").get_text(strip=True)
        section_data = {}

        # Get the next sibling rows until we hit another section or data
        current = header.next_sibling
        while current:
            if current.name == "tr":
                if current.get("class"):
                    if "sectionMember" in current.get("class"):
                        # Parse section member
                        td = current.find("td")
                        if td:
                            # Extract label and value using spans
                            label_span = td.find("span", attrs={"isi-label": ""})
                            value_span = td.find("span", attrs={"isi-value": ""})

                            if label_span and value_span:
                                label = label_span.get_text(strip=True)
                                value = value_span.get_text(strip=True)
                                # Clean up the label (remove units etc.)
                                clean_label = label.lower().replace(" ", "_")
                                section_data[clean_label] = value
                    else:
                        # Hit another section or data, stop
                        break
                elif not current.get_text(strip=True):
                    # Empty row, continue
                    pass
                else:
                    # Hit data or other content, stop
                    break
            current = current.next_sibling

        if section_data:
            section_key = section_name.lower().replace(" ", "_")
            metadata[section_key] = section_data

            # Also add flattened keys for common metadata
            if section_key == "location_properties":
                if "location_name" in section_data:
                    metadata["location_name"] = section_data["location_name"]
                if "latitude" in section_data:
                    # Extract numeric value from "41.7738064 °"
                    lat_match = re.search(r"([+-]?\d+\.?\d*)", section_data["latitude"])
                    if lat_match:
                        metadata["latitude"] = float(lat_match.group(1))
                if "longitude" in section_data:
                    lon_match = re.search(
                        r"([+-]?\d+\.?\d*)", section_data["longitude"]
                    )
                    if lon_match:
                        metadata["longitude"] = float(lon_match.group(1))

            elif section_key == "instrument_properties":
                if "device_sn" in section_data:
                    metadata["device_serial_number"] = section_data["device_sn"]
                if "device_model" in section_data:
                    metadata["device_model"] = section_data["device_model"]

            elif section_key == "log_properties":
                if "log_name" in section_data:
                    metadata["log_name"] = section_data["log_name"]
                if "interval" in section_data:
                    metadata["logging_interval"] = section_data["interval"]

    return metadata


def _parse_sensor_data(table) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Parse sensor data from the HTML table."""
    # Find the data header row
    header_row = table.find("tr", class_="dataHeader")
    if not header_row:
        raise ValueError("Could not find data header row")

    # Extract column information
    header_cells = header_row.find_all("td")
    columns = []
    column_info = {}

    for cell in header_cells:
        col_name = cell.get_text(strip=True)
        columns.append(col_name)

        # Extract sensor information from attributes
        info = {}
        if cell.get("isi-device-serial-number"):
            info["device_sn"] = cell["isi-device-serial-number"]
        if cell.get("isi-sensor-serial-number"):
            info["sensor_sn"] = cell["isi-sensor-serial-number"]
        if cell.get("isi-sensor-type"):
            info["sensor_type"] = cell["isi-sensor-type"]
        if cell.get("isi-parameter-type"):
            info["parameter_type"] = cell["isi-parameter-type"]
        if cell.get("isi-unit-type"):
            info["unit_type"] = cell["isi-unit-type"]

        column_info[col_name] = info

    # Find all data rows
    data_rows = table.find_all("tr", class_="data")

    if not data_rows:
        raise ValueError("No data rows found")

    # Extract data
    data = []
    for row in data_rows:
        cells = row.find_all("td")
        if len(cells) == len(columns):
            row_data = [cell.get_text(strip=True) for cell in cells]
            data.append(row_data)

    if not data:
        raise ValueError("No valid data rows found")

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Clean up column names
    clean_columns = {}
    for col in df.columns:
        if col == "Date Time":
            clean_columns[col] = "datetime"
        else:
            # Extract parameter name and units
            # e.g., "Actual Conductivity (µS/cm) (577714)" -> "actual_conductivity"
            clean_name = col.split("(")[0].strip()
            clean_name = clean_name.lower().replace(" ", "_").replace("-", "_")
            clean_columns[col] = clean_name

    df = df.rename(columns=clean_columns)

    # Convert data types
    for col in df.columns:
        if col == "datetime":
            # Convert datetime column
            df[col] = pd.to_datetime(df[col])
        else:
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Update column_info with clean names
    clean_column_info = {}
    for orig_name, clean_name in clean_columns.items():
        clean_column_info[clean_name] = column_info[orig_name]

    return df, clean_column_info


def get_aquatroll_summary(file_path: str | Path) -> dict[str, Any]:
    """
    Get a summary of an AquaTROLL file without loading all data.

    Args:
        file_path: Path to the AquaTROLL HTML file

    Returns:
        Dictionary with file summary information
    """
    try:
        result = parse_aquatroll_file(file_path)
        metadata = result["metadata"]
        data_df = result["data"]

        summary = {
            "filename": Path(file_path).name,
            "location": metadata.get("location_name", "Unknown"),
            "device_model": metadata.get("device_model", "Unknown"),
            "device_sn": metadata.get("device_serial_number", "Unknown"),
            "log_name": metadata.get("log_name", "Unknown"),
            "start_time": data_df["datetime"].min()
            if "datetime" in data_df.columns
            else None,
            "end_time": data_df["datetime"].max()
            if "datetime" in data_df.columns
            else None,
            "num_records": len(data_df),
            "parameters": list(data_df.columns),
            "latitude": metadata.get("latitude"),
            "longitude": metadata.get("longitude"),
        }

        return summary

    except Exception as e:
        logger.error(f"Error getting summary for {file_path}: {e}")
        return {"filename": Path(file_path).name, "error": str(e)}

def add_drifter_number(df: pd.DataFrame, metadata_file: str | Path) -> pd.DataFrame:
    """
    Add a 'drifter_number' column to the DataFrame by joining on a metadata table.

    Args:
        df: pandas DataFrame with an 'asset_id' column
        metadata_file: Path to CSV file containing aquatroll sn to 'drifter_number' mappings

    Returns:
        DataFrame with an additional 'drifter' column
    """
    if "device_sn" not in df.columns:
        raise ValueError("DataFrame must contain a 'device_sn' column")

    # Load metadata mapping
    metadata = pd.read_csv(metadata_file, dtype=str)
    
    # filter for aquatroll devices only
    metadata = metadata[metadata["device_type"].str.lower() == "aquatroll"]

    # select relevant columns
    metadata = metadata[["device_sn", "drifter"]].drop_duplicates()

    # Merge with metadata to get drifter numbers
    df = df.merge(metadata[["device_sn", "drifter"]], on="device_sn", how="left")

    return df

if __name__ == "__main__":
    # Setup logging for standalone execution
    setup_logging()

    # Example usage
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            if Path(file_path).is_dir():
                # Parse folder
                df = parse_aquatroll_folder(file_path)
                df = add_drifter_number(df, "data/drifter_metadata.csv")
                print(f"Parsed {len(df)} total records from folder")
                print(f"Columns: {list(df.columns)}")
                print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                df.to_csv("combined_aquatroll_data.csv", index=False)
                print("Combined data written to combined_aquatroll_data.csv")
                df.to_parquet("combined_aquatroll_data.parquet", index=False)
                print("Combined data written to combined_aquatroll_data.parquet")
            else:
                # Parse single file
                result = parse_aquatroll_file(file_path)
                print(f"Metadata: {result['metadata']}")
                print(f"Data shape: {result['data'].shape}")
                print(f"Columns: {list(result['data'].columns)}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python aquatroll_parser.py <file_or_folder_path>")
