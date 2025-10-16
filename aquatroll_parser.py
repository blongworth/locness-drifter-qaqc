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


def parse_aquatroll_csv_file(file_path: str | Path) -> dict[str, Any]:
    """
    Parse a single AquaTROLL CSV file and return metadata and sensor data.

    Args:
        file_path: Path to the AquaTROLL CSV file

    Returns:
        Dictionary containing:
        - 'metadata': Dict with basic file metadata
        - 'data': pandas DataFrame with sensor readings
        - 'columns': Dict mapping column names to their units (extracted from headers)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Parsing AquaTROLL CSV file: {file_path.name}")

    try:
        # Read the CSV file, trying different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'cp1252']
        df = None
        
        for encoding in encodings_to_try:
            try:
                # First, try to read the file to detect the structure
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                
                # Find the header line (contains column names)
                header_line_idx = None
                for i, line in enumerate(lines):
                    if 'Date Time' in line and ',' in line:
                        header_line_idx = i
                        break
                
                if header_line_idx is None:
                    # Try standard CSV reading
                    df = pd.read_csv(file_path, encoding=encoding)
                else:
                    # Skip the metadata lines and read from the header
                    df = pd.read_csv(file_path, encoding=encoding, skiprows=header_line_idx)
                
                logger.info(f"Successfully read CSV file with {encoding} encoding")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        if df is None:
            raise ValueError("Could not read CSV file with any supported encoding")
        
        if df.empty:
            raise ValueError("CSV file is empty")

        # Extract metadata from CSV header section (if available)
        metadata = {
            "filename": file_path.name,
            "file_type": "csv",
            "num_records": len(df)
        }
        
        # Try to extract device info from the header lines we skipped
        # Use the same encoding that worked for reading the CSV
        working_encoding = None
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.readline()  # Test if we can read
                    working_encoding = encoding
                    break
            except UnicodeDecodeError:
                continue
        
        if working_encoding:
            try:
                with open(file_path, 'r', encoding=working_encoding) as f:
                    header_lines = []
                    for line in f:
                        if 'Date Time' in line:
                            break
                        header_lines.append(line.strip())
                    
                    # Parse metadata from header lines
                    for line in header_lines:
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip().lower().replace(' ', '_').replace('/', '_')
                            value = value.strip()
                            
                            if key == 'device_s_n':
                                metadata['device_serial_number'] = value
                            elif key == 'device_model':
                                metadata['device_model'] = value
                            elif key == 'device_firmware':
                                metadata['device_firmware'] = value
            except Exception:
                # If metadata extraction fails, continue without it
                pass

        # Parse column information and clean up column names
        column_info = {}
        clean_columns = {}
        
        # Define the exact column mappings for ISI CSV format
        isi_column_mappings = {
            'Date Time': 'timestamp',
            '1: Actual Conductivity': 'actual_conductivity',
            '2: Specific Conductivity': 'specific_conductivity', 
            '3: Salinity': 'salinity',
            '4: Resistivity': 'resistivity',
            '5: Water Density': 'density',
            '6: Total Dissolved Solids': 'total_dissolved_solids',
            '7: Dissolved Oxygen': 'rdo_concentration',  # mg/L version
            '8: Dissolved Oxygen': 'rdo_saturation',     # %sat version
            '9: Partial Pressure Oxygen': 'oxygen_partial_pressure',
            '10: Turbidity': 'turbidity', 
            '11: pH': 'ph',
            '12: pH mV': 'ph_mv',
            '13: ORP mV': 'orp',
            '14: Temperature': 'temperature',
            '15: External': 'external_voltage',
            '16: Battery': 'battery_capacity',
            '17: Barometer': 'barometric_pressure',
            '18: Pressure': 'pressure',
            '19: Depth': 'depth'
        }

        for col in df.columns:
            # Extract units from column names
            unit_match = re.search(r'\(([^)]+)\)', col)
            unit = unit_match.group(1) if unit_match else ""
            
            # Skip sensor serial number and data quality columns
            if 'sensor s/n' in col.lower() or 'data quality' in col.lower():
                continue
            
            # Clean column name by removing units and extra info
            clean_col = col
            # Remove units in parentheses and extra identifiers
            clean_col = re.sub(r'\s*\([^)]*\)', '', clean_col)
            # Remove special characters like µ, °, etc.
            clean_col = re.sub(r'[µ°³]', '', clean_col)
            clean_col = clean_col.strip()
            
            # Find matching standard column name
            clean_name = None
            if clean_col == 'Date Time':
                clean_name = 'timestamp'
            else:
                # Look for matches in our mapping
                for pattern, standard_name in isi_column_mappings.items():
                    if pattern in clean_col:
                        # Special handling for dissolved oxygen (distinguish mg/L vs %sat)
                        if 'Dissolved Oxygen' in clean_col:
                            if '%sat' in col or 'sat' in col.lower():
                                clean_name = 'rdo_saturation'
                            else:
                                clean_name = 'rdo_concentration'
                        else:
                            clean_name = standard_name
                        break
            
            # If we found a matching standard column name, add it
            if clean_name:
                clean_columns[col] = clean_name
                
                # Store column info
                column_info[clean_name] = {
                    "original_name": col,
                    "unit_type": unit,
                    "parameter_type": clean_name
                }

        # Rename columns and keep only the ones we want
        df = df.rename(columns=clean_columns)
        
        # Keep only the columns that were successfully mapped to standard names
        columns_to_keep = list(clean_columns.values())
        df = df[columns_to_keep]

        # Convert data types
        for col in df.columns:
            if col == "timestamp":
                # Convert timestamp column
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception as e:
                    logger.warning(f"Could not convert timestamp column: {e}")
            else:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add device serial number to DataFrame if we extracted it from metadata
        if "device_serial_number" in metadata:
            df["device_sn"] = metadata["device_serial_number"]

        # Extract additional metadata if timestamp is available
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            metadata.update({
                "start_time": df["timestamp"].min(),
                "end_time": df["timestamp"].max(),
                "duration_hours": (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
            })

        logger.info(f"Successfully parsed {len(df)} data records from CSV")

        return {"metadata": metadata, "data": df, "columns": column_info}

    except Exception as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise ValueError(f"Failed to parse AquaTROLL CSV file: {e}")


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

    # Convert timestamp column to UTC if it exists
    if "timestamp" in combined_df.columns:
        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], utc=True)

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
            clean_columns[col] = "timestamp"
        else:
            # Extract parameter name and units
            # e.g., "Actual Conductivity (µS/cm) (577714)" -> "actual_conductivity"
            clean_name = col.split("(")[0].strip()
            clean_name = clean_name.lower().replace(" ", "_").replace("-", "_")
            clean_columns[col] = clean_name

    df = df.rename(columns=clean_columns)

    # Convert data types
    for col in df.columns:
        if col == "timestamp":
            # Convert timestamp column
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
            "start_time": data_df["timestamp"].min()
            if "timestamp" in data_df.columns
            else None,
            "end_time": data_df["timestamp"].max()
            if "timestamp" in data_df.columns
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
    metadata = pd.read_csv(metadata_file)
    
    # filter for aquatroll devices only
    metadata = metadata[metadata["device_type"].str.lower() == "aquatroll"]

    # select relevant columns
    metadata = metadata[["device_sn", "drifter"]].drop_duplicates()
    metadata["drifter"] = metadata["drifter"].astype("Int8")
    # Merge with metadata to get drifter numbers
    df = df.merge(metadata[["device_sn", "drifter"]], on="device_sn", how="left")

    return df

def add_flags(df: pd.DataFrame, flag_ranges: dict[str, list[float]]) -> pd.DataFrame:
    """
    Add flag columns to the DataFrame based on specified value ranges.

    For each field in the flag_ranges dictionary, a new column named
    '<field>_flag' is added. The flag is set to 2 if the value is
    within the [min, max] range (inclusive), and 4 otherwise.

    Args:
        df: pandas DataFrame to add flags to.
        flag_ranges: Dictionary where keys are column names and values are
                     lists containing [min_value, max_value].

    Returns:
        DataFrame with added flag columns.
    """
    df_copy = df.copy()
    for field, value_range in flag_ranges.items():
        if field not in df_copy.columns:
            logger.warning(
                f"Field '{field}' not found in DataFrame. Skipping flagging."
            )
            continue

        min_val, max_val = value_range
        flag_col = f"{field}_flag"

        # Default flag is 4 (bad/outside range)
        df_copy[flag_col] = 4

        # Set flag to 2 for values within the specified range (inclusive)
        in_range_condition = (df_copy[field] >= min_val) & (df_copy[field] <= max_val)
        df_copy.loc[in_range_condition, flag_col] = 2
        
        # if the original data had NaNs, keep the flag as NaN
        df_copy.loc[df_copy[field].isna(), flag_col] = pd.NA

        # Ensure flag is integer type, allowing for NaNs if original data had them
        df_copy[flag_col] = df_copy[flag_col].astype("Int8")

    return df_copy

def main():
    # Setup logging for standalone execution
    setup_logging()

    # Example usage
    import sys
    
    # flag dictionary for sensor fields
    flag_dict = {
        "salinity": [30, 35],  # PSU
        "temperature": [19, 23],  # degrees Celsius
        "rdo_concentration": [7, 9],  # mg/L
        "ph": [8, 9],  # pH units
    }

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            # Parse folder
            df = parse_aquatroll_folder(file_path)
            # Also parse the CSV file and append to the combined data
            csv_file = "data/AT600_Drifter04_LOC02.CSV"
            try:
                csv_result = parse_aquatroll_csv_file(csv_file)
                csv_df = csv_result["data"].copy()
                
                # Add source file information
                csv_df["source_file"] = Path(csv_file).name

                # Add metadata as columns if available
                csv_metadata = csv_result["metadata"]
                if "location_name" in csv_metadata:
                    csv_df["location"] = csv_metadata["location_name"]
                if "device_serial_number" in csv_metadata:
                    csv_df["device_sn"] = csv_metadata["device_serial_number"]
                if "log_name" in csv_metadata:
                    csv_df["log_name"] = csv_metadata["log_name"]
                
                # Convert timestamp to UTC if it exists
                if "timestamp" in csv_df.columns:
                    csv_df["timestamp"] = pd.to_datetime(csv_df["timestamp"], utc=True)
                
                # Append to the combined dataframe
                df = pd.concat([df, csv_df], ignore_index=True)
                print(f"Appended {len(csv_df)} records from CSV file")
                
            except Exception as e:
                logger.warning(f"Could not parse CSV file {csv_file}: {e}")
            df = add_drifter_number(df, "data/drifter_metadata.csv")
            df = add_flags(df, flag_dict)
            print(f"Parsed {len(df)} total records from folder")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            df.to_csv("output/loc02_drifter_aquatroll.csv", index=False)
            print("Combined data written to output/loc02_drifter_aquatroll.csv")
            df.to_parquet("output/loc02_drifter_aquatroll.parquet", index=False)
            print("Combined data written to output/loc02_drifter_aquatroll.parquet")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python aquatroll_parser.py <file_or_folder_path>")

if __name__ == "__main__":
    main()