import streamlit as st
from pathlib import Path
import base64
import pandas as pd

ROOT = Path(__file__).parents[1]
ASSETS = ROOT / "img"
CSS_PATH = ROOT / "style.css"

def inject_css(path: Path):
    if path.exists():
        st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)

def b64_img(path: Path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()

# Inject shared deck styles
inject_css(CSS_PATH)

# --- HERO HEADER ---
st.markdown(
  """
  <section class="hero hero-center">
    <div class="hero-left">
      <td1>Prediction of Buying Behavior Patterns using Machine Learning</td1>
      <pd1>
        Farah Binti Fuaad (22009174) • Bachelor of Computer Science (Hons)<br>
        Supervisor: Assoc. Prof. Ts Dr Norshakirah Bt Abdul Aziz
      </pd1>
      <a href="#section-introduction" class="cta" style="color:#ffffff; text-decoration:none;">Explore →</a>
    </div>
  </section>
  """,
  unsafe_allow_html=True,
)

# ========================
# 01 Introduction
# ========================
st.markdown(
    """
    <div class="section-header" id="section-introduction">
      <div class="section-row">
        <div class="section-number">01</div>
        <div class="section-title">Introduction</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Background
intro_img_b64 = b64_img(ASSETS / "intro_hero.png") or b64_img(ASSETS / "probstate.png")
img_html = f'<img class="hero-image" src="data:image/png;base64,{intro_img_b64}" />' if intro_img_b64 else '<div class="hero-image">No image</div>'

st.markdown(
    f"""
    <section class="hero">
      <div class="hero-left">
        <h3>Background</h3>
        <pd1>
          In the digital marketplace, purchase behavior is increasingly influenced by viral trends and social media. This shift has led to a surge in impulsive purchases, often driven by popularity rather than necessity. As a result, consumers frequently buying items they rarely use, contributing to financial waste, environmental concerns, and cluttered living spaces. Predictive analytics using machine learning will help to understand and guide purchasing decisions more mindfully.
        </pd1>
        <div class="card-row" style="margin-top:1rem;">
          <div class="deck-card"><strong>Context</strong><br>Rise of impulsive buying due to social media trends.</div>
          <div class="deck-card"><strong>Problem</strong><br>Financial waste, environmental concerns, cluttered living spaces.</div>
          <div class="deck-card"><strong>Goal</strong><br>Promote mindful consumption using predictive analytics.</div>
        </div>
      </div>
      <div class="hero-right">{img_html}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# Problem Statement (inside Introduction)
colleft, colright = st.columns([1, 2], gap="large")
with colleft:
    img_b64 = b64_img(ASSETS / "probstate.png")
    if img_b64:
        st.markdown(
            f"""
            <figure style="text-align:center; margin:0;">
              <img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto; border-radius:12px;">
              <figcaption style="font-size:12px; color:#666; margin-top:8px;">
                Source: <a href="https://www.voguebusiness.com/story/sustainability/tiktoks-anti-overconsumption-movement-rule-of-5-wake-up-call-for-brands" target="_blank">Cernansky, R. (2024, Jan 25). Vogue Business.</a>
              </figcaption>
            </figure>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Image not found: img/probstate.png")

with colright:
    st.markdown(
        """
        <div class="deck-card" style="align-items:flex-start; text-align:left;">
          <h3 style="margin:0 0 .4rem 0;">Problem Statement</h3>
          <p style="margin:.2rem 0 0;">In an ideal consumer environment, individuals make purchasing decisions based on genuine needs, preferences, and long-term value. Buying behavior would be intentional.</p>
          <p style="margin:.6rem 0 0;">However, the reality is that consumer behavior is increasingly driven by viral trends and social media influence. Many individuals make impulsive purchases based on popularity rather than necessity which causes:</p>
          <ul style="margin:.4rem 0 0 1.2rem; color:#fff;">
            <li>Wasted money</li>
            <li>Environmental harm</li>
            <li>Cluttered living spaces</li>
          </li>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Objectives (inside Introduction)
st.markdown(
    """
    <div class="card-row">
      <div class="deck-card"><strong>Objective 1</strong><br>To build a machine learning models to predict customer purchase intention.</div>
      <div class="deck-card"><strong>Objective 2</strong><br>To detect cluster-specific trending items and purchase patterns through customer segmentation.</div>
      <div class="deck-card"><strong>Objective 3</strong><br>To build a web dashboard that shows the user's intention prediction, trending items and data-driven insights.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================
# 02 Literature Review
# ========================
st.markdown(
    """
    <div class="section-header" id="section-literature">
      <div class="section-row">
        <div class="section-number">02</div>
        <div class="section-title">Literature Review</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card-row">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Applications of Traditional Machine Learning in Consumer Analytics</strong>
        <p style="margin:0.4rem 0 0;">
        GhorbanTanhaei et al. (2024) compare RF, LR, and Gradient Boosting for forecasting customer behavior, emphasizing precision, recall, and ROC‑AUC for model selection and validation to support segmentation and resource optimization.
        </p>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Modern Machine Learning Models in Marketing Contexts</strong>
        <p style="margin:0.4rem 0 0;">
        Lin (2025) shows XGBoost and CatBoost capture non‑linearities and prevent overfitting with regularization, enabling accurate trend detection and personalization at e‑commerce scale.
        </p>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Behavioral and Psychological Insights</strong>
        <p style="margin:0.4rem 0 0;">
        Zhou et al. (2022) apply the Theory of Planned Behavior: attitudes, social norms, and perceived control shape purchase intention—improving interpretability and actionability when integrated into models.
        </p>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Clustering and Association Rule Mining in Consumer Analytics</strong>
        <p style="margin:0.4rem 0 0;">
        Wu & Wang (2020) highlight KMeans for segmentation; Apriori uncovers frequent itemsets. Alfred et al. (2023) show combined use yields deeper purchase pattern insights for targeted strategies.
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================
# 03 Methodology Overview
# ========================
st.markdown(
    """
    <div class="section-header" id="section-methodology">
      <div class="section-row">
        <div class="section-number">03</div>
        <div class="section-title">Methodology Overview</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Agile + CRISP-DM
st.markdown(
    """
    <div class="deck-card" style="align-items:flex-start; text-align:left;">
      <strong>Agile + CRISP‑DM integration</strong>
      <ul style="margin:.4rem 0 0 1.2rem;">
        <li>Iterative development with continuous feedback and flexible refinement of models and dashboard.</li>
        <li>Each sprint aligns to CRISP‑DM: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Data Understanding, Data Preparation, Modeling (inside Methodology)
st.markdown(
    """
    <div class="card-row">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Data Understanding</strong>
        <ul style="margin:.4rem 0 0 1.2rem;">
          <li>Field: Business analytics and customer behavior prediction.</li>
          <li>Source: "shopping_trends.csv" (add sources).</li>
        </ul>
        <p style="margin:.4rem 0 0;"><em>Prediction:</em></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Size: 12,944 records; 10,944 cleaned; 12 features.</li>
          <li>Target: purchase_intent_category (Wants‑based, Planned, Need‑based, Impulsive).</li>
        </ul>
        <p style="margin:.4rem 0 0;"><em>Clustering:</em></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Size: 12,944; 10 features.</li>
        </ul>
        <p style="margin:.4rem 0 0;"><em>Data Issues:</em></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Missing: Frequency of Purchases (1,131).</li>
          <li>Categorical: Gender, Item Purchased, Category, Location, Season, Discount Applied, Promo Code Used, Purchase Intent Category.</li>
        </ul>
      </div>

      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Data Preparation</strong>
        <p style="margin:.4rem 0 0;">Handling Missing Values</p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Impute Frequency of Purchases with mode to retain categorical frequency.</li>
        </ul>
        <p style="margin:.4rem 0 0;">Encoding Categorical Variables</p>
        <ul style="margin:.2rem 0 0 1.2rem">
          <li>Label Encoding to numeric; store encoders in le_dict for reuse and reversibility.</li>
        </ul>
        <p style="margin:.4rem 0 0;">One‑Hot for Apriori — why?</p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>KMeans/prediction: needs numeric vectors; LabelEncoder + MinMaxScaler enables distance computations. Note: label encoding imposes order; alternatives include one‑hot or k‑prototypes/k‑modes.</li>
          <li>Apriori: requires a 0/1 item presence matrix; one‑hot maps each category value to an item for support/confidence/lift.</li>
        </ul>
      </div>

      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Modeling</strong>
        <p style="margin:.4rem 0 0;"><em>Classification</em></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Linear SVM, XGBoost, Decision Tree, Logistic Regression, Random Forest.</li>
          <li>Preprocessing: SMOTE, StandardScaler, hyperparameter tuning.</li>
          <li>Split 80:20.</li>
        </ul>
        <p style="margin:.4rem 0 0;"><em>Unsupervised</em></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>KMeans clustering + Apriori rules.</li>
          <li>MinMaxScaler for clustering.</li>
        </ul>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================
# 04 Evaluation
# ========================
st.markdown(
    """
    <div class="section-header" id="section-evaluation">
      <div class="section-row">
        <div class="section-number">04</div>
        <div class="section-title">Evaluation</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Prediction metrics
st.markdown("Metrics: Accuracy, Precision, Recall, F1-score.")
metrics_df = pd.DataFrame([
    ("Linear SVM", 0.8177, 0.8281, 0.8177, 0.8207),
    ("XGBoost", 0.9821, 0.9827, 0.9821, 0.9819),
    ("Decision Tree", 0.8613, 0.8653, 0.8613, 0.8589),
    ("Logistic Regression", 0.8264, 0.8354, 0.8264, 0.8279),
    ("Random Forest", 0.9012, 0.9059, 0.9013, 0.9001),
    ("RBF SVM with Tuning", 0.9307, 0.9333, 0.9307, 0.9312),
    ("Logistic Regression with Tuning", 0.8246, 0.8334, 0.8246, 0.8264),
    ("XGBoost with Tuning", 0.9826, 0.9831, 0.9826, 0.9825),
], columns=["Model","Accuracy","Precision","Recall","F1"])
st.dataframe(metrics_df, use_container_width=True)
st.success("Result: XGBoost highest accuracy (98.26%).")

# Clustering & Association Rules
st.markdown(
    """
    <div class="deck-card" style="align-items:flex-start; text-align:left; margin-top:0.6rem;">
      <strong>Clustering & Association Rules</strong>
      <p style="margin:.4rem 0 0;">The Elbow Method (SSE/Inertia) measures how tightly data points are grouped in clusters.</p>
      <p style="margin:.2rem 0 0;">Observation: k = 2 had the highest cohesion score (0.3993) but was too broad.</p>
      <p style="margin:.2rem 0 0;">Selected k = 5 to extract richer behavioral insights without excessive splitting.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cluster_sizes_df = pd.DataFrame([
    (0, 4871, 37.63),
    (1, 1762, 13.61),
    (2, 1270, 9.81),
    (3, 3693, 28.53),
    (4, 1348, 10.41),
], columns=["Cluster","Count","Percent"])
st.dataframe(cluster_sizes_df, use_container_width=True)
st.caption("Uneven cluster sizes suggest some dominant customer profiles—common in retail where planned purchases or popular categories are widespread.")

rules_df = pd.DataFrame([
    (1, "If Location = Canada, Season = Spring ⇒ Gender = Male, Frequency = Every 3 Months, Item Purchased = Shoes", 0.0335, 0.9365, 7.5348),
    (1, "If Location = Canada, Frequency = Every 3 Months, Season = Spring ⇒ Gender = Male, Item Purchased = Shoes", 0.0335, 0.9833, 6.1441),
    (1, "If Gender = Male, Location = Canada, Season = Spring ⇒ Item Purchased = Shoes, Frequency = Every 3 Months", 0.0335, 0.9516, 6.1195),
    (4, "If Promo Code Used = Yes, Location = Kenya ⇒ Gender = Male, Item Purchased = Jacket, Frequency = Every 3 Months, Discount Applied = Yes, Purchase Intent = Impulsive, Season = Winter", 0.0304, 0.9111, 16.1602),
    (4, "If Promo Code Used = Yes, Location = Kenya ⇒ Gender = Male, Item Purchased = Jacket, Frequency = Every 3 Months, Purchase Intent = Impulsive, Season = Winter", 0.0304, 0.9111, 16.1602),
    (4, "If Location = Kenya, Discount Applied = Yes ⇒ Item Purchased = Jacket, Promo Code Used = Yes, Frequency = Every 3 Months, Purchase Intent = Impulsive", 0.0304, 0.9111, 16.1602),
], columns=["Cluster","Rule","Support","Confidence","Lift"])
st.dataframe(rules_df, use_container_width=True)

# Insights (still under Evaluation)
st.markdown(
    """
    <div class="card-row">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Cluster 1</strong>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Shoes are strongly associated with purchases in Canada during spring.</li>
          <li>High confidence (93.6%) indicates consistent occurrence under these conditions.</li>
          <li>Represents planned seasonal buying behavior.</li>
        </ul>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Cluster 4</strong>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Promo codes and discounts link to impulsive winter jacket purchases.</li>
          <li>Lift > 16 shows powerful promotional influence.</li>
          <li>Reflects promotion- and season-driven impulsivity.</li>
        </ul>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# What Consumers Can Learn (under Evaluation)
st.markdown(
    """
    <div class="card-row" style="margin-top:.4rem;">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>What Consumers Can Learn</strong>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Seasonality matters: certain products spike in specific seasons.</li>
          <li>Promotions trigger impulses: discounts and codes increase purchase likelihood.</li>
          <li>Actionable: plan essentials; recognize marketing nudges; use intent predictions to avoid unnecessary buys.</li>
        </ul>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Dashboard Visualization</strong>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Prediction Analysis page.</li>
          <li>Clustering Analysis page.</li>
        </ul>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================
# 05 Conclusion and recommendation
# ========================
st.markdown(
    """
    <div class="section-header" id="section-conclusion">
      <div class="section-row">
        <div class="section-number">05</div>
        <div class="section-title">Conclusion and Recommendation</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card-row">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Conclusion</strong>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>High-Accuracy Prediction: XGBoost achieved 98.26% accuracy.</li>
          <li>Behavior Insights: Seasonality and promotions drive impulsive buying (KMeans + Apriori).</li>
          <li>Interactive Dashboard: Streamlit with enhanced visuals for predictions and trends.</li>
          <li>End-to-End Development: cleaning → features → modeling → clustering → rules → dashboard.</li>
          <li>Supports mindful consumption and reduces impulsive spend via predictive analytics.</li>
        </ul>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <strong>Recommendations</strong>
        <ol style="margin:.3rem 0 0 1.2rem;">
          <li>Re-evaluate Cluster Numbers: explore k = 3 or k = 4 for clearer separation.</li>
          <li>Improve Apriori Rules: deduplicate and threshold by lift/confidence for clarity.</li>
          <li>Enhance Dashboard & Deployment: add business-focused analytics; deploy publicly.</li>
        </ol>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


