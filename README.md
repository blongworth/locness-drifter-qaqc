# locness-drifter-qaqc

## Overview

This project processes and integrates drifter data from multiple sensors including GPS position data, Aquatroll water quality sensors, and fluorometer sensors.

## Data Processing Workflow

### 1. Data Parsing and Combination

Three parser modules are used to parse and combine raw data from different sensors:

- **`gpx_parser.py`** - Parses GPX files containing GPS position data from drifter deployments
- **`aquatroll_parser.py`** - Parses Aquatroll sensor data files containing water quality measurements
- **`fluorometer_parser.py`** - Parses fluorometer sensor data files containing fluorescence measurements

Each parser reads the raw data files, processes them, and combines them into a unified dataset for its respective sensor type.

### 2. Data Integration

- **`quick_join.py`** - Joins the parsed sensor datasets together based on timestamps

The integration uses a **left join** on the position data, meaning that sensor data (Aquatroll and fluorometer) is only included in the final dataset where corresponding position data is available. This ensures all output rows have valid GPS coordinates. Commented code for a reverse join that will include all fluorometer records is also provided for reference.

### 3. Metadata Files

- **`drifter_metadata.csv`** - Links sensors to their respective drifters by serial number
- **`loc02_drifter_deployments.csv`** - Used to create a flag indicating when drifters are deployed
