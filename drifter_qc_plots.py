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
    return mo, np, pl, px


@app.cell
def _(mo):
    mo.md("""# Drifter 5 Position Map""")
    return


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
def _(List, Tuple, math, np, pl, px):
    def scatter_mapbox_quantile_color(
        df_filtered: pl.DataFrame,
        selected_drifter: str,
        value_col: str = "sensor_ppb_rwt",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        n_quantiles: int = 10,
        height: int = 800,
    ):
        """
        Build a scatter_mapbox where color is based on quantile mapping of a skewed variable,
        implemented with Polars and Plotly, using the Viridis colorscale.

        - Computes quantile edges and uses their midpoints to create a continuous color scale
          that ensures equal data mass per color segment.
        - Adds a colorbar with ticks at the quantile midpoints and labels indicating quantile ranges.
        - Uses Viridis colorscale fixed.

        Parameters
        ----------
        df_filtered : pl.DataFrame
            Polars DataFrame containing lat/lon and value columns.
        selected_drifter : str
            Title context.
        value_col : str
            Column to color by (skewed).
        lat_col, lon_col : str
            Latitude/Longitude column names.
        n_quantiles : int
            Number of quantile buckets to spread colors across.
        height : int
            Figure height.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """

        # Validate required columns
        for col in [value_col, lat_col, lon_col]:
            if col not in df_filtered.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Drop rows with missing required fields
        #df = df_filtered.drop_nulls([lat_col, lon_col, value_col])
        df = df_filtered
        # If DataFrame is empty after cleaning, return an empty figure gracefully
        if df.height == 0:
            fig = px.scatter_mapbox(
                pl.DataFrame({lat_col: [], lon_col: []}).to_pandas(),
                lat=lat_col,
                lon=lon_col,
                title=f"Drifter {selected_drifter} (no data)",
                mapbox_style="open-street-map",
                zoom=10,
                height=height,
            )
            return fig

        # Extract values as numpy for quantile calculation
        v = df.select(pl.col(value_col)).to_numpy().ravel()
        v_min = float(v.min())
        v_max = float(v.max())

        # Handle single-value edge case
        if math.isclose(v_min, v_max):
            # Convert df to pandas for Plotly Express (px works with pandas)
            pdf = df.to_pandas()
            fig = px.scatter_mapbox(
                pdf,
                lat=lat_col,
                lon=lon_col,
                color_discrete_sequence=["#440154"],  # Viridis deep purple for consistency
                hover_data={"timestamp": True, value_col: True} if "timestamp" in df.columns else None,
                title=f"Drifter {selected_drifter}",
                mapbox_style="open-street-map",
                zoom=10,
                height=height,
                labels={value_col: "RWT (ppb)"},
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_traces(marker=dict(size=8, opacity=0.8))
            return fig

        # Quantile edges and midpoints (use approx percentiles for speed and robustness)
        q_positions = np.linspace(0.0, 1.0, n_quantiles + 1)
        # Polars' quantile supports 'nearest', 'linear' interpolation; linear gives smoother edges
        q_edges = np.array([
            float(df.select(pl.col(value_col).quantile(q, interpolation="linear")).item())
            for q in q_positions
        ])
        q_mids = (q_edges[:-1] + q_edges[1:]) / 2.0

        # Digitize: map each value to its quantile bin index (0..n_quantiles-1)
        # numpy.digitize uses right=False to put values equal to edge to the right bin; match that behavior
        inner_edges = q_edges[1:-1]
        bin_idx = np.digitize(v, inner_edges, right=False)
        bin_idx = np.clip(bin_idx, 0, n_quantiles - 1)

        # Map bin index to corresponding quantile midpoint as the color value
        v_quant_mids = q_mids[bin_idx]

        # Attach quantile-mapped values back to Polars DataFrame
        df = df.with_columns(pl.Series(name="_quantile_mid_value", values=v_quant_mids))

        # Normalize midpoints to [0,1] for colorscale positions
        qm_min, qm_max = float(q_mids.min()), float(q_mids.max())
        denom = (qm_max - qm_min) if (qm_max - qm_min) != 0 else 1e-12
        norm_positions = (q_mids - qm_min) / denom

        # Fixed Viridis colorscale sampled at our quantile midpoint positions
        base_colors = px.colors.sequential.Viridis
        def sample_colorscale(colors: List[str], t: float) -> str:
            n = len(colors)
            idx = int(round(t * (n - 1)))
            idx = max(0, min(n - 1, idx))
            return colors[idx]

        quant_colors = [sample_colorscale(base_colors, float(t)) for t in norm_positions]
        plotly_colorscale: List[Tuple[float, str]] = [(float(t), c) for t, c in zip(norm_positions, quant_colors)]

        # Tick labels show original value ranges per quantile
        def fmt(x: float) -> str:
            ax = abs(x)
            if ax >= 1000:
                return f"{x:,.0f}"
            elif ax >= 1:
                return f"{x:,.2f}"
            else:
                return f"{x:.4f}"

        tick_vals = q_mids
        tick_text = [f"{fmt(lo)}–{fmt(hi)}" for lo, hi in zip(q_edges[:-1], q_edges[1:])]

        # Plotly Express works with pandas; convert at the end to keep heavy lifting in Polars
        pdf = df.to_pandas()

        fig = px.scatter_mapbox(
            pdf,
            lat=lat_col,
            lon=lon_col,
            color="_quantile_mid_value",
            title=f"Drifter {selected_drifter}",
            mapbox_style="open-street-map",
            zoom=10,
            height=height,
            labels={"_quantile_mid_value": "RWT (ppb, quantile-mapped)"},
            hover_data={"timestamp": True, value_col: True} if "timestamp" in df.columns else None,
        )

        fig.update_coloraxes(
            colorscale=plotly_colorscale,
            cmin=qm_min,
            cmax=qm_max,
            colorbar=dict(
                title="RWT (ppb) quantile ranges",
                tickvals=list(tick_vals),
                ticktext=tick_text,
                lenmode="pixels",
                len=300,
            ),
        )

        fig.update_traces(marker=dict(size=8, opacity=0.8))

        return fig

    return (scatter_mapbox_quantile_color,)


@app.cell
def _(df_filtered, scatter_mapbox_quantile_color, selected_drifter):
    # Example usage with your existing variables (df_filtered, selected_drifter):
    _fig = scatter_mapbox_quantile_color(df_filtered.to_pandas(), selected_drifter)
    _fig

    return


if __name__ == "__main__":
    app.run()
