import streamlit as st
from pathlib import Path
import base64

ASSETS = Path(__file__).parents[1] / "img"

def read_image(path: Path):
    if not path.exists():
        return None
    return path

st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1>Prediction of Buying Behavior Patterns<br>using Machine Learning</h1>
        <p>
        <br>
        Farah Binti Fuaad (22009174)<br>
        Bachelor of Computer Science (Hons)<br><br>
        Supervised by: Assoc. Prof. Ts Dr Norshakirah Bt Abdul Aziz
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# --- 01 Introduction ---

st.markdown(
    """
    <div class="section-header">
      <div class="section-row">
        <div class="section-number">01</div>
        <div class="section-title">Introduction</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br><br>", unsafe_allow_html=True)

# Background
bg_html = """
<div class="hero" style="background-color: #1c3a13;">
    <div class="hero-left">
        <td1 style="color: #fcfcf7;">Background</td1>
        <pd1 style="color: #fcfcf7;">
            <br>In the digital marketplace, purchase behavior is increasingly influenced by viral trends and social media. 
            This shift has led to a surge in impulsive purchases, often driven by popularity rather than necessity. 
            As a result, consumers frequently buying items they rarely use, contributing to financial waste, environmental 
            concerns, and cluttered living spaces. Predictive analytics using machine learning will help to understand and 
            guide purchasing decisions more mindfully.
        </pd1>
    </div>
</div>
"""
st.markdown(bg_html.format(), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Problem Statement
colleft, colright = st.columns([1, 2])

with colleft:
     st.markdown("""
        <h3>Problem Statement</h3>
    """, unsafe_allow_html=True
    )

with colright:
        st.markdown("""
        <h3>Problem Statement</h3>
        <p><br>
        In an ideal consumer environment, individuals make purchasing decisions based on genuine needs, preferences, and long-term value. Buying behavior would be intentional. However, the reality is that consumer behavior is increasingly driven by viral trends and social media influence. 
        Many individuals make impulsive purchases based on popularity rather than necessity which causes: <br>
        - Wasted money<br>
        - Environmental harm<br>
        - Cluttered living spaces
        </p>
        """, unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)
