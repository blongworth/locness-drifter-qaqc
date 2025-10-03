import polars as pl

# Define file paths
gpx_file = "parsed_gpx_data.parquet"
fluoro_file = "combined_fluorometer_data.parquet"
troll_file = "combined_aquatroll_data.parquet"

# Read and sort the dataframes
# Sorting is required for asof joins
gpx_df = pl.read_parquet(gpx_file).sort("drifter", "timestamp")
fluoro_df = pl.read_parquet(fluoro_file).sort("drifter", "sensor_position", "timestamp")
troll_df = pl.read_parquet(troll_file).sort("drifter", "timestamp")

# Join gpx and fluorometer data
combined_df = fluoro_df.join_asof(
    gpx_df, on="timestamp", by="drifter", tolerance="60s"
)

# Join the result with aquatroll data
final_df = combined_df.join_asof(
    troll_df, on="timestamp", by="drifter", tolerance="60s"
)

# Display the final combined dataframe
print(final_df)

# Optionally, save the result to a new parquet file
final_df.write_parquet("final_combined_data.parquet")
final_df.write_csv("final_combined_data.csv")