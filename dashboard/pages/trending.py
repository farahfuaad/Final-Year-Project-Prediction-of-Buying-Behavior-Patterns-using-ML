import streamlit as st
import pandas as pd
import folium
import altair as alt
import plotly.express as px
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from pathlib import Path
import json
import math
import glob

# load data
df = pd.read_csv("data/cleaned_shopping_trends.csv")

maps_dir = Path(__file__).parent.parent / "data" / "maps"

def _extract_coords(obj):
    """Recursively extract (lon, lat) pairs from GeoJSON coordinate arrays."""
    coords = []
    if isinstance(obj, list):
        # leaf list of coordinates or nested lists
        if len(obj) >= 2 and isinstance(obj[0], (int, float)):
            # single coordinate pair [lon, lat]
            coords.append((obj[0], obj[1]))
        else:
            for item in obj:
                coords.extend(_extract_coords(item))
    return coords

def geojson_bounds_centroid(geojson_data):
    """Return (min_lon, min_lat, max_lon, max_lat) and centroid (lat, lon)."""
    coords = []
    geom = None
    if "features" in geojson_data:
        for f in geojson_data["features"]:
            geom = f.get("geometry", {})
            coords.extend(_extract_coords(geom.get("coordinates", [])))
    elif "geometry" in geojson_data:
        geom = geojson_data["geometry"]
        coords.extend(_extract_coords(geom.get("coordinates", [])))
    else:
        # assume it's a plain geometry object
        coords.extend(_extract_coords(geojson_data.get("coordinates", [])))

    if not coords:
        return None, None
    lons, lats = zip(*coords)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    centroid_lat = (min_lat + max_lat) / 2
    centroid_lon = (min_lon + max_lon) / 2
    return (min_lon, min_lat, max_lon, max_lat), (centroid_lat, centroid_lon)

def estimate_zoom_from_bbox(bbox):
    """Rudimentary zoom estimator for pydeck based on degrees span."""
    if bbox is None:
        return 1.5
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    span = max(lon_span, lat_span)
    # smaller span -> higher zoom. Values tuned heuristically.
    if span <= 0.02:
        return 12
    if span <= 0.2:
        return 9
    if span <= 1:
        return 6
    if span <= 10:
        return 4
    if span <= 40:
        return 2.5
    return 1.5

def find_geojson_for_location(location):
    """Try to find a geojson file matching the location string."""
    if not location:
        return None
    target = location.strip().lower()
    # exact filename match (without extension)
    for p in maps_dir.glob("*.geojson"):
        if p.stem.lower() == target.replace(" ", "_"):
            return p
    # substring match
    for p in maps_dir.glob("*.geojson"):
        if p.stem.lower() in target or target in p.stem.lower():
            return p
    # fallback: special common cases
    special = {
        "united states": "usa",
        "united kingdom": "uk",
        "england": "uk",
        "america": "usa",
        "china": "china",
        "singapore": "sg",
        "australia": "aus",
        "philippines": "phil",
        "malaysia": "my",
    }
    for k, v in special.items():
        if k in target:
            candidate = maps_dir / f"{v}.geojson"
            if candidate.exists():
                return candidate
    return None

# --- 3-column layout ---
col1, col2, col3 = st.columns([0.5, 2, 1])

with col1:
    st.header("Filters Stats")

    # safe column names (handle case differences)
    loc_col = "Location" if "Location" in df.columns else ("location" if "location" in df.columns else None)
    season_col = "Season" if "Season" in df.columns else ("season" if "season" in df.columns else None)
    cluster_col = "Cluster" if "Cluster" in df.columns else ("cluster" if "cluster" in df.columns else None)
    cat_col = "Category" if "Category" in df.columns else ("category" if "category" in df.columns else None)
    item_col = "Item Purchased" if "Item Purchased" in df.columns else ("item purchased" if "item purchased" in df.columns else None)
    purchase_intent_col = ("Purchase Intent Category" if "Purchase Intent Category" in df.columns
                           else ("purchase_intent_category" if "purchase_intent_category" in df.columns else None))

    # Location filter
    if loc_col:
        locations = ["All"] + sorted(df[loc_col].dropna().astype(str).unique().tolist())
        selected_location = st.selectbox("Location", locations, index=0)
    else:
        selected_location = None

    # Season filter
    if season_col:
        seasons = ["All"] + sorted(df[season_col].dropna().astype(str).unique().tolist())
        selected_season = st.selectbox("Season", seasons, index=0)
    else:
        selected_season = None

    # Cluster filter
    if cluster_col:
        clusters = ["All"] + sorted(df[cluster_col].dropna().astype(str).unique().tolist())
        selected_cluster = st.selectbox("Cluster", clusters, index=0)
    else:
        selected_cluster = None

    st.markdown("---")

    # Build filtered dataframe for stats & other widgets
    filtered_df = df.copy()
    if loc_col and selected_location and selected_location != "All":
        filtered_df = filtered_df[filtered_df[loc_col].astype(str) == selected_location]
    if season_col and selected_season and selected_season != "All":
        filtered_df = filtered_df[filtered_df[season_col].astype(str) == selected_season]
    if cluster_col and selected_cluster and selected_cluster != "All":
        filtered_df = filtered_df[filtered_df[cluster_col].astype(str) == selected_cluster]

    # Quick Stats
    total_customers = int(filtered_df.shape[0])
    dominant_category = (filtered_df[cat_col].mode().iloc[0]
                         if (cat_col and not filtered_df[cat_col].dropna().empty) else "N/A")
    top_item = (filtered_df[item_col].mode().iloc[0]
                if (item_col and not filtered_df[item_col].dropna().empty) else "N/A")
    purchase_intent = (filtered_df[purchase_intent_col].mode().iloc[0]
                       if (purchase_intent_col and not filtered_df[purchase_intent_col].dropna().empty) else "N/A")

    with st.container():
        st.markdown("#### Total Customers")
        st.markdown(f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.5em;text-align:center;font-weight:bold'>{total_customers:,}</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("#### Dominant Category")
        st.markdown(f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.2em;text-align:center;font-weight:bold'>{dominant_category}</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("#### Top Item Purchased")
        st.markdown(f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.2em;text-align:center;font-weight:bold'>{top_item}</div>", unsafe_allow_html=True)
    # Optionally add purchase intent as a card
    if purchase_intent_col:
        with st.container():
            st.markdown("#### Dominant Purchase Intent")
            st.markdown(f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.2em;text-align:center;font-weight:bold'>{purchase_intent}</div>", unsafe_allow_html=True)

with col2:
    st.header("Locate")
    # default world view
    initial_view = pdk.ViewState(latitude=0, longitude=0, zoom=1.5)

    if selected_location and selected_location != "All":
        geojson_path = find_geojson_for_location(selected_location)
        if geojson_path and geojson_path.exists():
            with st.spinner(f"Loading map for {selected_location}..."):
                geojson_data = json.loads(geojson_path.read_text(encoding="utf-8"))
                bbox, centroid = geojson_bounds_centroid(geojson_data)
                zoom = estimate_zoom_from_bbox(bbox)
                if centroid is not None:
                    view_state = pdk.ViewState(latitude=centroid[0], longitude=centroid[1], zoom=zoom, pitch=0)
                else:
                    # fallback to default world view if centroid is None
                    view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1.5, pitch=0)
                gj_layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=geojson_data,
                    stroked=True,
                    filled=True,
                    get_fill_color=[70, 130, 180, 120],
                    get_line_color=[10, 10, 10, 80],
                    pickable=True,
                )
                deck = pdk.Deck(layers=[gj_layer], initial_view_state=view_state, map_style="light")
                st.pydeck_chart(deck, use_container_width=True)
        else:
            st.warning(f"No map boundary found for '{selected_location}'. Falling back to coordinates if available.")
            # try to center on average lat/lon from dataset for that location
            lat_col = "Latitude" if "Latitude" in df.columns else ("latitude" if "latitude" in df.columns else None)
            lon_col = "Longitude" if "Longitude" in df.columns else ("longitude" if "longitude" in df.columns else None)
            subset = df[df[loc_col].astype(str) == selected_location]
            if lat_col and lon_col and not subset[[lat_col, lon_col]].dropna().empty:
                mean_lat = subset[lat_col].dropna().astype(float).mean()
                mean_lon = subset[lon_col].dropna().astype(float).mean()
                view_state = pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=4)
                deck = pdk.Deck(initial_view_state=view_state, map_style="light")
                st.pydeck_chart(deck, use_container_width=True)
            else:
                st.info("No coordinates available for this location in the dataset.")

    else:
        # show light world map
        deck = pdk.Deck(initial_view_state=initial_view, map_style="light")
        st.pydeck_chart(deck, use_container_width=True)

with col3:
    st.header("Summary / Actions")
    # example: show top items for selected location
    if selected_location and selected_location != "All":
        sub = df[df[loc_col].astype(str) == selected_location]
    else:
        sub = df.copy()
    if not sub.empty:
        st.subheader("Top items")
        if "Item Purchased" in sub.columns:
            top = sub["Item Purchased"].value_counts().head(5)
            st.write(top)
        st.subheader("Counts")
        st.write(sub.shape[0])
    else:
        st.write("No data for selected filters.")

# Continue with the rest of your page code...

