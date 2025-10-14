import streamlit as st
import pandas as pd
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

# load data for the map and stats
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

# ---- Main Content ----
topcol1, topcol2 = st.columns([1, 2])

with topcol1:
    st.header("What's Trending?")
    
    # variable column names handling
    loc_col = (
        "Location"
        if "Location" in df.columns else ("location" if "location" in df.columns else None)
        )
    season_col = (
        "Season" if "Season" in df.columns else ("season" if "season" in df.columns else None)
        )
    cluster_col = (
        "Cluster" if "Cluster" in df.columns else ("cluster" if "cluster" in df.columns else None)
        )
    cat_col = (
    "Category" if "Category" in df.columns else ("category" if "category" in df.columns else None)
    )
    item_col = (
    "Item Purchased" if "Item Purchased" in df.columns else ("item purchased" if "item purchased" in df.columns else None)
    )
    purchase_freq_col = (
    "Frequency of Purchases" if "Frequency of Purchases" in df.columns
    else ("purchase_frequency" if "purchase_frequency" in df.columns else None)
    )

    # --- filters ---
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

    # filtered dataframe for stats and charts
    filtered_df = df.copy()
    if loc_col and selected_location and selected_location != "All":
        filtered_df = filtered_df[filtered_df[loc_col].astype(str) == selected_location]
    if season_col and selected_season and selected_season != "All":
        filtered_df = filtered_df[filtered_df[season_col].astype(str) == selected_season]

    st.markdown("---")

with topcol2:
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

st.markdown("---")

# --- Stats cards ---
card1, card2, card3, card4 = st.columns(4)

# Stats cards
total_customers = int(filtered_df.shape[0])

dominant_category = (
    filtered_df[cat_col].mode().iloc[0]
    if (cat_col and not filtered_df[cat_col].dropna().empty)
    else "N/A"
)

top_item_purchased = (
    filtered_df[item_col].mode().iloc[0]
    if (item_col and not filtered_df[item_col].dropna().empty)
    else "N/A"
)

most_common_purchase_freq = (
    filtered_df[purchase_freq_col].mode().iloc[0]
    if (purchase_freq_col and not filtered_df[purchase_freq_col].dropna().empty)
    else "N/A"
)

with card1:
    with st.container():
        st.markdown("###### Total Customers")
        st.markdown(
            f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.2em;text-align:center;font-weight:bold'>{total_customers:,}</div>",
            unsafe_allow_html=True,
        )

with card2:
    with st.container():
        st.markdown("###### Dominant Category")
        st.markdown(
            f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.2em;text-align:center;font-weight:bold'>{dominant_category}</div>",
            unsafe_allow_html=True,
        )

with card3:
    with st.container():
        st.markdown("###### Top Item Purchased")
        st.markdown(
            f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.2em;text-align:center;font-weight:bold'>{top_item_purchased}</div>",
            unsafe_allow_html=True,
        )

with card4:
    with st.container():
        st.markdown("###### Most Common Purchase Frequency")
        st.markdown(
            f"<div style='background-color:rgba(240,242,246,0.7);padding:1.2em 1em;border-radius:10px;font-size:1.2em;text-align:center;font-weight:bold'>{most_common_purchase_freq}</div>",
            unsafe_allow_html=True,
        )
st.markdown("---")

#
st.header("Insights & Trends")
col1, col2 = st.columns(2)

# --- Trending Items by Cluster (top 3 per cluster) ---
with col1:
    st.subheader("Trending Items by Cluster")
    try:
        if cluster_col:
            top_items = (
                filtered_df.groupby([cluster_col, item_col])
                .size()
                .reset_index(name="count")
                .sort_values([cluster_col, "count"], ascending=[True, False])
                )
            top3 = top_items.groupby(cluster_col).head(3)
            if top3.empty:
                st.info("No item data available for clusters.")
            else:
                # Use facet columns if many clusters; fallback to single chart if few
                import plotly.express as px

            # Convert cluster to string for reliable facetting
            top3["_cluster_str"] = top3[cluster_col].astype(str)
            fig = px.bar(
                top3,
                x="count",
                y=item_col,
                color="_cluster_str",
                orientation="h",
                facet_col="_cluster_str",
                facet_col_wrap=1 if len(top3["_cluster_str"].unique()) > 3 else len(top3["_cluster_str"].unique()),
                height=300 + 80 * len(top3["_cluster_str"].unique()),
                labels={"count": "Count", item_col: "Item", "_cluster_str": "Cluster"},
                )
            fig.update_layout(showlegend=False, margin=dict(t=30, b=10, l=80, r=10))
            fig.update_yaxes(autorange="reversed")  # keep largest on top
            st.plotly_chart(fig, use_container_width=True)
        else:
            # No cluster column: show overall top 10 items
            top_overall = filtered_df[item_col].value_counts().head(10)
            st.bar_chart(top_overall)
    except Exception as e:
        st.error(f"Error building trending items chart: {e}")
        st.markdown("---")

with col2:
    # --- Purchase Frequency Distribution ---
    st.subheader("Purchase Frequency Distribution")
    # Define freq_col at the top of the scope so it is always available
    freq_col = "Frequency of Purchases" if "Frequency of Purchases" in filtered_df.columns else (
        "frequency of purchases" if "frequency of purchases" in filtered_df.columns else None
        )
    try:
        if freq_col:
            freq_counts = filtered_df[freq_col].value_counts().reset_index()
            freq_counts.columns = [freq_col, "count"]
            fig2 = px.pie(freq_counts, names=freq_col, values="count", title="Purchase Frequency", hole=0.35)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No 'Frequency of Purchases' column found.")
    except Exception as e:
        st.error(f"Error building frequency distribution: {e}")
        
    st.markdown("---")

# --- Cluster Characteristics from Apriori Rules ---
st.subheader("Cluster Characteristics (Trending Items & Dominant Features)")

# For deployment, uncomment the line below and comment the line after
#analysis_path = Path(__file__).parent.parent / "data" / "trending_item_analysis.txt"

# For local testing, uncomment the line below and comment the line above
analysis_path = Path("/Users/farahfuaad/Desktop/fyp/Final-Year-Project-Prediction-of-Consumer-Behaviour-using-ML/data/trending_item_analysis.txt")

if analysis_path.exists():
    with open(analysis_path, "r") as f:
        lines = f.readlines()

    clusters = []
    cluster = {}
    section = None  # Initialize section to avoid unbound error
    for line in lines:
        line = line.strip()
        if line.startswith("CLUSTER"):
            if cluster:
                clusters.append(cluster)
                cluster = {}
            cluster["header"] = line
            cluster["Top Items"] = []
            cluster["Dominant Characteristics"] = []
        elif line.startswith("Top Items:"):
            section = "Top Items"
        elif line.startswith("Dominant Characteristics:"):
            section = "Dominant Characteristics"
        elif line.startswith("-") or not line:
            continue
        elif line.startswith("•"):
            if section == "Top Items":
                cluster["Top Items"].append(line)
            elif section == "Dominant Characteristics":
                cluster["Dominant Characteristics"].append(line)
    if cluster:
        clusters.append(cluster)

    # Display as 3 columns of square cards with expanders
    n = len(clusters)
    for i in range(0, n, 3):
        col1, col2, col3 = st.columns(3)
        for idx, col in enumerate([col1, col2, col3]):
            if i + idx < n:
                c = clusters[i + idx]
                with col:
                    with st.container():
                        with st.expander(c.get("header", "Cluster")):
                            st.markdown("**Top Items:**")
                            if c.get("Top Items"):
                                for item in c.get("Top Items", []):
                                    st.markdown(f"- {item}")
                            else:
                                st.markdown("_No top items found._")
                            st.markdown("**Dominant Characteristics:**")
                            if c.get("Dominant Characteristics"):
                                for char in c.get("Dominant Characteristics", []):
                                    st.markdown(f"- {char}")
                            else:
                                st.markdown("_No dominant characteristics found._")
                            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No trending item analysis file found.")