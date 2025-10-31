import streamlit as st
from pathlib import Path
import os
import streamlit.components.v1 as components
from st_flexible_callout_elements import flexible_callout

# Set page config
st.set_page_config(page_title="Think Over", layout="wide", initial_sidebar_state="collapsed")

# Inject CSS
style_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(style_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# sidebar logo configuration
main_body_logo = Path(__file__).parent / "img" / "app_logo.svg"
sidebar_logo = Path(__file__).parent / "img" / "app_logo2.svg"

st.logo(sidebar_logo, icon_image=main_body_logo, size="large")

# --- Insight toggle in sidebar (use checkbox, st.toggle doesn't exist) ---
if "insight_toggle" not in st.session_state:
    st.session_state["insight_toggle"] = False

with st.sidebar:
    insight_toggle = st.toggle("Turn on for insights", key="insight_toggle", 
                               value=st.session_state["insight_toggle"])

# Define pages with correct paths
pages = [
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/trending.py", title="What's Trending"),
    st.Page("pages/deck.py", title="Presentation Deck")
    ]

# Navigation
pg = st.navigation(pages)
pg.run()
