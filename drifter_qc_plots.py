import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import numpy as np
    import pandas as pd
    return mo, pl, px


@app.cell
def _(pl):
    # Load the data using polars
    df = pl.read_parquet("loc02_drifter_combined.parquet")
    df
    return (df,)


@app.cell
def _(df, px):
    # Create a scatter mapbox plot
    # Note: plotly.express works seamlessly with pandas, so we convert the polars DataFrame


    _fig = px.scatter_mapbox(
        df.to_pandas(),
        lat="latitude",
        lon="longitude",
        color="drifter",
        title="Drifter Positions",
        mapbox_style="open-street-map",
        zoom=10,
        height=600,
        labels={"sensor_rwt_ppb": "RWT (ppb)"},
    )

    _fig
    return


@app.cell
def _(df, pl):
    # Filter for the specific drifter
    drifter_5_df = df.filter(pl.col("drifter") == 5)
    return (drifter_5_df,)


@app.cell
def _(drifter_5_df, px):
    # Create a scatter mapbox plot
    # Note: plotly.express works seamlessly with pandas, so we convert the polars DataFrame
    _fig = px.scatter_mapbox(
        drifter_5_df.to_pandas(),
        lat="latitude",
        lon="longitude",
        color="sensor_ppb_rwt",
        title="Drifter 5 Position Colored by Sensor RWT (ppb)",
        mapbox_style="open-street-map",
        zoom=10,
        height=600,
        labels={"sensor_rwt_ppb": "RWT (ppb)"},
    )

    _fig
    return


@app.cell
def _(deployed, df, drifter_select, pl):

    selected_drifter = int(drifter_select.value)
    df_filtered = df.filter(
        (pl.col("drifter") == selected_drifter) &
        (pl.col("deployed") if deployed.value else pl.lit(True))
    )

    df_filtered
    return df_filtered, selected_drifter


@app.cell(hide_code=True)
def _(df, mo):
    # Extract unique drifter IDs for the selector
    drifter_ids = sorted(df["drifter"].unique().to_list())

    # Selector widget for drifter number
    drifter_select = mo.ui.dropdown(
        options=[str(d) for d in drifter_ids],  # marimo expects strings; keep labels simple
        value=str(drifter_ids[0]),
        label="Drifter"
    )

    # Boolean selector for deployment status
    deployed = mo.ui.checkbox(
        label="Deployed",
        value=False  # default unchecked; set True if you want default deployed
    )

    drifter_select, deployed
    return deployed, drifter_select


@app.cell
def _(df_filtered, px, selected_drifter):
    _fig = px.scatter_mapbox(
        df_filtered.to_pandas(),
        lat="latitude",
        lon="longitude",
        color="sensor_ppb_rwt",
        title=f"Drifter {selected_drifter}",
        mapbox_style="open-street-map",
        zoom=10,
        height=800,
        labels={"sensor_ppb_rwt": "RWT (ppb)"},
        hover_data={
                    "timestamp": True,},
    )

    _fig
    return


@app.cell
def _(df_filtered, px, selected_drifter):
    # Ensure points are in path order; skip if already sorted correctly
    df_path = df_filtered.to_pandas()

    # Points scatter
    scatter_fig = px.scatter_mapbox(
        df_path,
        lat="latitude",
        lon="longitude",
        color="sensor_ppb_rwt",
        title=f"Drifter {selected_drifter}",
        mapbox_style="open-street-map",
        zoom=10,
        height=800,
        labels={"sensor_ppb_rwt": "RWT (ppb)"},
        hover_data={"timestamp": True},
    )

    # Line trace connecting points in order
    line_fig = px.line_mapbox(
        df_path,
        lat="latitude",
        lon="longitude",
        hover_name="timestamp",
    )

    # Combine traces: add line traces into the scatter figure
    for tr in line_fig.data:
        # Style line: thin, semi-opaque to sit under points
        tr.line.width = 2
        tr.line.color = "black"
        tr.opacity = 0.5
        scatter_fig.add_trace(tr)

    # Optional: ensure line appears beneath markers
    # Plotly draws in order added; we added line after scatter, so swap order if needed
    # Move line to first position
    scatter_fig.data = tuple(list(scatter_fig.data)[-1:] + list(scatter_fig.data)[:-1])

    scatter_fig
    return


@app.cell
def _(df_filtered, px, selected_drifter):
    _fig_time_series = px.line(
        df_filtered.to_pandas(),
        x="timestamp",
        y="sensor_ppb_rwt",
        color="sensor_position",
        title=f"Sensor RWT vs. Time for Drifter {selected_drifter}",
        labels={
            "timestamp": "Time",
            "sensor_ppb_rwt": "Sensor RWT (ppb)",
            "sensor_position": "Sensor Position",
        },
        hover_data={"sensor_position": True},
    )

    _fig_time_series
    return


if __name__ == "__main__":
    app.run()
