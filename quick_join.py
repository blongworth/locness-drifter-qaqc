import polars as pl
from pathlib import Path

# Define file paths
gpx_file = "output/loc02_drifter_positions.parquet"
fluoro_file = "output/loc02_drifter_pme.parquet"
troll_file = "output/loc02_drifter_aquatroll.parquet"

# Read and sort the dataframes
# Sorting is required for asof joins
gpx_df = pl.read_parquet(gpx_file).sort("drifter", "timestamp")
fluoro_df = pl.read_parquet(fluoro_file).sort("drifter", "sensor_position", "timestamp")
troll_df = pl.read_parquet(troll_file).sort("drifter", "timestamp")

# Get unique combinations of drifter and sensor_position
unique_sensors = fluoro_df.select("drifter", "sensor_position").unique()

# Cross join GPS data with unique sensor combinations to create all combinations
gpx_expanded = gpx_df.join(unique_sensors, on="drifter", how="inner")

print(fluoro_df.columns)

# Now join with fluorometer data using asof join
combined_df = gpx_expanded.sort("drifter", "sensor_position", "timestamp").join_asof(
    fluoro_df, 
    on="timestamp", 
    by=["drifter", "sensor_position"], 
    tolerance="60s"
)

# Join the result with aquatroll data
final_df = combined_df.join_asof(
    troll_df, on="timestamp", by="drifter", tolerance="60s"
)

def add_drifter_deployed_flag(df: pl.DataFrame, deployments_file: str) -> pl.DataFrame:
    """Add deployed flags to the dataset based on logs.
    
    Args:
        df: DataFrame with timestamp column
        deployments_file: Path to CSV file with deployment logs
        
    Returns:
        pl.DataFrame: DataFrame with added 'deployed' column
    """
    if not Path(deployments_file).exists():
        raise FileNotFoundError(f"Underway flow flag file not found: {deployments_file}")
    
    deployments = pl.read_csv(deployments_file).select(
        "drifter", "deployed_utc", "recovered_utc"
    ).with_columns([
        pl.col("deployed_utc").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M").dt.replace_time_zone("UTC"),
        pl.col("recovered_utc").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M").dt.replace_time_zone("UTC"),
    ])
    print(deployments)

    # initialize 'deployed' column to false (not deployed)
    df = df.with_columns(pl.lit(False).alias("deployed"))

    # Iterate over each shutdown period and update flags
    for row in deployments.iter_rows(named=True):
        drifter = row["drifter"]
        start_time = row["deployed_utc"]
        end_time = row["recovered_utc"]
        
        # Create a mask for the time range of the shutdown
        mask = (
            (pl.col("drifter") == drifter) 
            & (pl.col("timestamp") > start_time) 
            & (pl.col("timestamp") < end_time)
        )
        
        # Set flags to 4 (bad data) for the shutdown period
        df = df.with_columns(
            pl.when(mask).then(True).otherwise(pl.col("deployed")).alias("deployed")
        )
    
    return df

flagged_df = add_drifter_deployed_flag(final_df, "data/loc02_drifter_deployments.csv")

# Display the final combined dataframe
print(flagged_df)

# Optionally, save the result to a new parquet file
flagged_df.write_parquet("output/loc02_drifter_combined.parquet")
flagged_df.write_csv("output/loc02_drifter_combined.csv")