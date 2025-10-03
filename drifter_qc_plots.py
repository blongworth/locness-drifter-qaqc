import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    return mo, pl, px


@app.cell
def _(mo):
    mo.md("""# Drifter 5 Position Map""")
    return


@app.cell
def _(pl):
    # Load the data using polars
    df = pl.read_parquet("final_combined_data.parquet")
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


if __name__ == "__main__":
    app.run()
