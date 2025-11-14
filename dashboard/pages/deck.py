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
st.markdown("<br><br>", unsafe_allow_html=True)

# Problem Statement
colleft, colgap, colright = st.columns([1, 0.2, 2])

with colleft:
    # embed the local image as a base64 data URI so the HTML can load it in the browser
    img_path = ASSETS / "probstate.png"
    if img_path.exists():
        img_bytes = img_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        img_html = f"""
        <figure style="text-align:center; margin:0;">
          <img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto;">
          <figcaption style="font-size:10px; color:#666; margin-top:8px;">
            Source: Cernansky, R. (2024, January 25). Fighting overconsumption: TikTok’s deinfluencing movement and no-spend challenges are a wake-up call for brands. Vogue Business. https://www.voguebusiness.com/story/sustainability/tiktoks-anti-overconsumption-movement-rule-of-5-wake-up-call-for-brands
          </figcaption>
        </figure>
        """
    else:
        img_html = """
        <figure style="text-align:center; margin:0;">
          <div style="color:#666; font-size:14px;">Image not found: img/probstate.png</div>
        </figure>
        """
    st.markdown(img_html, unsafe_allow_html=True)

with colright:
        st.markdown("""
        <h3 style="text-align: center;"><br><br>Problem Statement</h3>
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

# Research Questions and Objectives
colleft, colgap, colright = st.columns([1, 0.05, 1])

with colleft:
    st.markdown("""
        <h3 style="text-align: center;"><br><br>Research Questions</h3>
        <div class="deck-card" style="line-height: 1; font-size: 16px">
        How can machine learning be applied to predict customer purchase intentions based on shopping trends data?
        </div>
                
        <br>
        <div class="deck-card" style="line-height: 1; font-size: 16px">
        What techniques can be used to identify trending items across categories and frequency?
        </div>
                
        <br>
        <div class="deck-card" style="line-height: 1; font-size: 16px">
        How can a web-based dashboard enhance user awareness and encourage more mindful purchases ?

        <br>
        </div>
        """,unsafe_allow_html=True)
    
with colright:
    st.markdown("""
        <h3 style="text-align: center;"><br><br>Objectives</h3>
        <div class="deck-card" style="line-height: 1; font-size: 16px">
        To build a machine learning models to predict customer purchase intention.
        </div>
                
        <br>
        <div class="deck-card" style="line-height: 1; font-size: 16px">
        To detect cluster-specific trending items and purchase patterns through customer segmentation.
        </div>
                
        <br>
        <div class="deck-card" style="line-height: 1; font-size: 16px">
        To build a web dashboard that shows the user's intention prediction, trending items and data-driven insights.
        <br>
        </div>
        """,unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 02 Literature Review ---

st.markdown(
    """
    <div class="section-header">
      <div class="section-row">
        <div class="section-number">02</div>
        <div class="section-title">Literature Review</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br><br>", unsafe_allow_html=True)

# --- 03 Research Methodology ---

st.markdown(
    """
    <div class="section-header">
      <div class="section-row">
        <div class="section-number">03</div>
        <div class="section-title">Research Methodology</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# Approach
st.markdown(
    """
    <div style="display:flex; align-items:center;">
      <h3 style="margin:0; padding:0;">Approach</h3>
      <div style="flex:1; height:1px; background-color:#e0e0e0; margin-left:12px;"></div>
    </div>
    """,unsafe_allow_html=True)

colleft, colgap, colright = st.columns([1, 0.05, 1])

with colleft: 
    st.markdown(
    """
    <br>
    <h3><br><Methodological Approach: Agile Framework</h3>
        <p>
        - The project follows an Agile methodology, enabling iterative development, continuous feedback, and flexibility in refining the machine learning model and dashboard features.
        <br>- Each sprint focuses on a specific CRISP-DM phase, ensuring structured progress and adaptability.
        </p>
    """,unsafe_allow_html=True)

with colright:
    img_path = ASSETS / "agile.png"
    if img_path.exists():
        img_bytes = img_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        img_html = f"""
        <figure style="text-align:center; margin:0;">
          <img src="data:image/png;base64,{img_b64}" style="max-width:300px; width:100%; height:auto;">
        </figure>
        """
    else:
        img_html = """
        <figure style="text-align:center; margin:0;">
          <div style="color:#666; font-size:14px;">Image not found: img/agile.png</div>
        </figure>
        """
    st.markdown(img_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 04 Results and Discussion ---

st.markdown(
    """
    <div class="section-header">
      <div class="section-row">
        <div class="section-number">04</div>
        <div class="section-title">Results and Discussion</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br><br>", unsafe_allow_html=True)

# --- 05 Conclusion and Recommendations ---

st.markdown(
    """
    <div class="section-header">
      <div class="section-row">
        <div class="section-number">05</div>
        <div class="section-title">Conclusion and Recommendations</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br><br>", unsafe_allow_html=True)

# --- 06 References ---

st.markdown(
    """
    <div class="section-header">
      <div class="section-row">
        <div class="section-number">06</div>
        <div class="section-title">References</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br><br>", unsafe_allow_html=True)


