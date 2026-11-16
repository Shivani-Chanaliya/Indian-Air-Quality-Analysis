import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_ANALYSIS_ROOT = Path(__file__).resolve().parents[4]
AIR_QUALITY_DIR = DATA_ANALYSIS_ROOT / "Python" / "Indian_Air_Quality_Analysis"


@st.cache_data
def load_air_quality_data():
    air_quality = pd.read_csv(AIR_QUALITY_DIR / "Indian_Air_Quality_Analysis.csv")
    state_data = pd.read_csv(AIR_QUALITY_DIR / "Indian_States_Data.csv")
    geojson = json.loads(
        (AIR_QUALITY_DIR / "states_india.geojson").read_text(encoding="utf-8")
    )
    return air_quality, state_data, geojson


st.title("Indian Air Quality Analysis")
st.caption(
    "Explore pollutant measurements by state, compare population and area, "
    "and view the geographic distribution."
)

air_quality, state_data, geojson = load_air_quality_data()
air_quality["state"] = air_quality["state"].str.replace("_", " ", regex=False)
air_quality = air_quality.dropna(
    subset=["state", "pollutant_id", "pollutant_avg"]
)

pollutant = st.sidebar.selectbox(
    "Pollutant",
    sorted(air_quality["pollutant_id"].unique()),
)
pollutant_data = air_quality[air_quality["pollutant_id"] == pollutant]
state_averages = (
    pollutant_data.groupby("state", as_index=False)["pollutant_avg"]
    .mean()
    .merge(state_data, on="state", how="left")
    .dropna(subset=["Area (sq. km)", "Population", "id"])
)
state_averages["population_area_score"] = (
    state_averages["pollutant_avg"]
    * state_averages["Population"]
    / state_averages["Area (sq. km)"]
)

metric_columns = st.columns(3)
metric_columns[0].metric("Measurement records", f"{len(pollutant_data):,}")
metric_columns[1].metric(
    "States represented", f"{state_averages['state'].nunique():,}"
)
metric_columns[2].metric(
    f"Mean {pollutant} level",
    f"{pollutant_data['pollutant_avg'].mean():.2f}",
)

top_states = state_averages.nlargest(10, "pollutant_avg")
st.subheader(f"Top states by average {pollutant}")
bar_chart = px.bar(
    top_states.sort_values("pollutant_avg"),
    x="pollutant_avg",
    y="state",
    orientation="h",
    labels={"pollutant_avg": "Average level", "state": "State"},
    color="pollutant_avg",
    color_continuous_scale="Reds",
)
st.plotly_chart(bar_chart, use_container_width=True)

for feature in geojson["features"]:
    feature["id"] = feature["properties"]["state_code"]

map_chart = px.choropleth(
    state_averages,
    geojson=geojson,
    locations="id",
    featureidkey="id",
    color="pollutant_avg",
    hover_name="state",
    hover_data=["pollutant_avg"],
    labels={"pollutant_avg": "Average level"},
    title=f"Average {pollutant} by state",
    color_continuous_scale="Reds",
)
map_chart.update_geos(fitbounds="locations", visible=False)
st.subheader("Geographic distribution")
st.plotly_chart(map_chart, use_container_width=True)

scatter_chart = px.scatter(
    state_averages,
    x="Area (sq. km)",
    y="Population",
    color="pollutant_avg",
    hover_name="state",
    labels={"pollutant_avg": "Average level"},
    title="State population and area",
    color_continuous_scale="Reds",
)
st.plotly_chart(scatter_chart, use_container_width=True)

st.subheader("State-level data")
st.dataframe(
    state_averages.sort_values("pollutant_avg", ascending=False),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Population-area score is an exploratory ratio: average pollutant level "
    "multiplied by population and divided by area."
)
