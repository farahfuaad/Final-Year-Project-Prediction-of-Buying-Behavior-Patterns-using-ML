import streamlit as st
from pathlib import Path

# Set page config
st.set_page_config(page_title="Think Over", layout="wide", initial_sidebar_state="expanded")

# Path to logo
logo_path = Path(__file__).parent / "img" / "app_logo.svg"

# Display logo at the top of the main page
st.image(str(logo_path), width=150)

# Define pages with correct paths
pages = [
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/trending.py", title="What's Trending"),
    st.Page("pages/quiz.py", title="Quiz"),
]

# Navigation
pg = st.navigation(pages)
pg.run()