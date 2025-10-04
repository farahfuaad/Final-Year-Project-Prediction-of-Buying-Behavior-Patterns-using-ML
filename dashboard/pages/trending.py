import streamlit as st
import pandas as pd
import folium
import altair as alt
import plotly.express as px
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from pathlib import Path

# load data
df = pd.read_csv("data/cleaned_shopping_trends.csv")

# Initialize geolocator with a user_agent to avoid errors
geolocator = Nominatim(user_agent="dynamic_map_app")

# Wrap geocode with a rate limiter to respect API usage policies
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# List of countries for the selectbox
countries = ["World"] + sorted(df['location'].dropna().unique().tolist())

# Create a title and a selectbox for the user to choose a country
st.title("🗺️ Streamlit Dynamic Map")
selected_country = st.selectbox("Select a country:", countries)

# Default view state for the world map
view_state = pdk.ViewState(
    latitude=0,
    longitude=0,
    zoom=1.5,
    pitch=0
)

# If a country is selected, find its coordinates and update the view
if selected_country != "World":
    with st.spinner("Fetching coordinates..."):
        try:
            location = geocode(selected_country)
            if location:
                view_state = pdk.ViewState(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    zoom=4, # Adjust zoom for better country view
                    pitch=45
                )
            else:
                st.warning(f"Could not find location for {selected_country}.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Create a deck.gl map
deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=view_state,
    layers=[],
)

# Render the map
st.pydeck_chart(deck)