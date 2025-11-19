import streamlit as st
from pathlib import Path
import os
import streamlit.components.v1 as components

# Set page config
st.set_page_config(page_title="Think Over", layout="wide", initial_sidebar_state="expanded")

# Inject CSS
style_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(style_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# sidebar logo configuration
main_body_logo = Path(__file__).parent / "img" / "app_logo.svg"
sidebar_logo = Path(__file__).parent / "img" / "app_logo2.svg"

st.logo(sidebar_logo, icon_image=main_body_logo, size="large")

# --- Insight toggle in sidebar (use checkbox, st.toggle doesn't exist) ---
with st.sidebar:
    params = st.query_params
    initial = params.get("insights", ["0"])[0] == "1"

    toggled = st.toggle("Show insights", value=True) # or value=false

    if toggled:
        st.query_params = {"insights": ["1"]}
    else:
        # remove the param when unchecked
        new_params = dict(params)  # params is mapping of lists
        if "insights" in new_params:
            new_params.pop("insights")
        st.query_params = new_params

# Define pages with correct paths
pages = [
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/trending.py", title="What's Trending"),
    st.Page("pages/deck.py", title="Presentation Deck")
    ]

# Navigation
pg = st.navigation(pages)
pg.run()
