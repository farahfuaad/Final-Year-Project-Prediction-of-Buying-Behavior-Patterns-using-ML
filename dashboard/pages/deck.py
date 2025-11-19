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

# --- Title Header ---
st.markdown(
    """
    <div class="section-header" id="title-header">
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1>Prediction of Buying Behavior Patterns<br>using Machine Learning</h1>
        <p style="font-size:16px">
        <br>
        Farah Binti Fuaad (22009174)<br>
        Bachelor of Computer Science (Hons)<br><br>
        Supervisor: Assoc. Prof. Ts Dr Norshakirah Bt Abdul Aziz
        </p>
    </div>
    """, unsafe_allow_html=True)  

st.markdown("""
  <div style="display:flex; justify-content:center; align-items:center; margin:20px 0;">
    <a href="#section-introduction" class="cta" style="color:#ffffff; text-decoration:none;">Explore →</a>
  </div>
  """, unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)

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

st.markdown("<br><br>", unsafe_allow_html=True)

# Background
st.markdown(
  f"""
  <div class="intro-wrapper">
    <div class="quote-box">
    <div style="font-size:18px; margin:80px 120px; text-align:center;">
    For years, the way people shop has significantly changed. With the popularity gain on online platforms and social media, many <span class="highlight">consumers are influenced</span> by 
    <span class="highlight">viral trends</span>, <span class="highlight">influencer recommendations</span>, and <span class="highlight">digital advertisements</span>. 
    These influences often <span class="highlight">lead to impulsive purchases</span>
    on buying things not because they are needed, but because they are popular or trending. 
    This project aims to develop a <span class="highlight">predictive analytics</span> 
    using machine learning that will help users <span class="highlight">to understand and guide purchasing decisions</span> 
    more mindfully. By predicting customer purchase intentions, identifying trending product categories with combination of interactive dashboard, 
    the system can help users to <span class="highlight">“think over”</span> their decisions before buying.
    </div>
  """
  ,
  unsafe_allow_html=True,
)

st.markdown("<br><br>", unsafe_allow_html=True)

# Problem Statement (inside Introduction)
img_b64 = b64_img(ASSETS / "probstate.png")

# create an img tag only if the image was found, otherwise show a small fallback message
img_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto; border-radius:12px;">' if img_b64 else '<div style="font-size:12px;color:#666;padding:16px;border:1px dashed #ccc;border-radius:8px;">Image not found: probstate.png</div>'

st.markdown(
    f"""
    <div class="intro-wrapper">
      <div class="quote-box"">
        {img_tag}
        <div style="font-size:10px;color:#666;margin-top:10px;text-align:center;">
          Source: Cernansky, R. (2024, January 25). Fighting overconsumption: TikTok's deinfluencing movement and no-spend challenges are a wake-up call for brands. Vogue Business. <a href="https://www.voguebusiness.com/story/sustainability/tiktoks-anti-overconsumption-movement-rule-of-5-wake-up-call-for-brands" target="_blank">Link</a>
        </div>
      </div>
      <div class="divider-vert"></div>
      <div class="intro-content"><br>
        <h3 style="margin-top:0;">Problem Statement</h3>
        <pd1>
        In an ideal consumer environment, individuals make purchasing decisions based on genuine needs, preferences, and long-term value. Buying behavior would be intentional. However, the reality is that consumer behavior is increasingly driven by viral trends and social media influence. Many individuals make impulsive purchases based on popularity rather than necessity which causes:<br>
        </pd1>
        <div class="bg-stat-cards">
          <div class="deck-card">Wasted money</div>
          <div class="deck-card">Environmental harm</div>
          <div class="deck-card">Cluttered living spaces</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

# Objectives
st.markdown(
    """
    <div class="card-row">
      <div class="deck-card"><strong><br>Objective 1</strong>To build a machine learning models to predict customer purchase intention.<br><br><br></div>
      <div class="deck-card"><strong><br>Objective 2</strong>To detect cluster-specific trending items and purchase patterns through customer segmentation.<br><br></div>
      <div class="deck-card"><strong><br>Objective 3</strong>To build a web dashboard that shows the user's intention prediction, trending items and data-driven insights.<br><br></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)


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
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-row">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
      <div class="badge">01</div>
        <strong>Applications of Traditional Machine Learning in Consumer Analytics</strong>
        <p style="margin:0.4rem 0 0; color: rgb(246, 247, 239);">
        GhorbanTanhaei et al. (2024) compare RF, LR, and Gradient Boosting for forecasting customer behavior, emphasizing precision, recall, and ROC‑AUC for model selection and validation to support segmentation and resource optimization.
        <br></p>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
      <div class="badge">02</div>
        <strong>Modern Machine Learning Models in Marketing Contexts</strong>
        <p style="margin:0.4rem 0 0; color: rgb(246, 247, 239);">
        Lin (2025) shows XGBoost and CatBoost capture non-linearities and prevent overfitting with regularization, enabling accurate trend detection and personalization at e‑commerce scale.
        <br><br><br></p>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
      <div class="badge">03</div>
        <strong>Behavioral and Psychological Insights</strong>
        <p style="margin:0.4rem 0 0; color: rgb(246, 247, 239);">
        Zhou et al. (2022) apply the Theory of Planned Behavior: attitudes, social norms, and perceived control shape purchase intention—improving interpretability and actionability when integrated into models.
        <br><br></p>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
      <div class="badge">04</div>
        <strong>Clustering and Association Rule Mining in Consumer Analytics</strong>
        <p style="margin:0.4rem 0 0; color: rgb(246, 247, 239);">
        Wu & Wang (2020) highlight KMeans for segmentation; Apriori uncovers frequent itemsets. Alfred et al. (2023) show combined use yields deeper purchase pattern insights for targeted strategies.
        <br><br><br></p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

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

st.markdown("<br>", unsafe_allow_html=True)

# Agile + CRISP-DM integration
img_agile = b64_img(ASSETS / "agile.png")
img_crisp = b64_img(ASSETS / "crispdm.svg")

# create an img tag only if the image was found, otherwise show a small fallback message
img_ag = f'<img src="data:image/png;base64,{img_agile}" style="max-width:80%; height:auto; border-radius:12px;">' if img_agile else '<div style="font-size:12px;color:#666;padding:16px;border:1px dashed #ccc;border-radius:8px;">Image not found: agile.png</div>'
img_cdm = f'<img src="data:image/svg+xml;base64,{img_crisp}" style="max-width:50%; height:auto; border-radius:12px;">' if img_crisp else '<div style="font-size:12px;color:#666;padding:16px;border:1px dashed #ccc;border-radius:8px;">Image not found: crispdm.svg</div>'

st.markdown(
  f"""
  <div class="intro-wrapper">
    <div class="quote-box">
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:10vh; margin:0 auto; padding:24px;">
        <div style="display:flex; gap:24px; align-items:center; justify-content:center; width:100%; max-width:1100px;">
          <div style="flex:1; display:flex; align-items:center; justify-content:center;">
            {img_ag}
          </div>
          <div class="divider-vert"></div>
          <div style="flex:1; display:flex; align-items:center; justify-content:center;">
            {img_cdm}
          </div>
        </div>
      </div>
    </div>
  </div><br>
  <div style="width:100%; margin-left: 60px; display:flex; align-items:center; justify-content:center;">
    <h3 style="margin-top:0;">Agile + CRISP-DM integration</h3>
    <ul style="display:inline-block; text-align:left; margin-right:65px;"><br><br>
      <li>The project follows an Agile methodology, enabling iterative development, continuous feedback, and flexibility in refining the machine learning model and dashboard features.</li>
      <li>Each sprint aligns to CRISP-DM: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation.</li>
    </ul>
  </div>
  </div>
  """,
  unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

# Data Understanding, Data Preparation, Modeling 
st.markdown(
    """
    <div class="card-row">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
      <div class="badge">Data Understanding</div>
        <ul style="margin:.4rem 0 0 1.2rem;">
          <li>Field: Business analytics and customer behavior prediction.</li>
          <li>Source: "shopping_trends.csv" <a href="https://www.sciencedirect.com/science/article/pii/S2666720724000924?via%3Dihub" target="_blank" rel="noopener noreferrer" style="color:#ffffff;">GhorbanTanhaei et al. (2024)</a>.</li>
        </ul>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Prediction:</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Size: 12,944 records; 10,944 cleaned; 12 features.</li>
          <li>Target: purchase_intent_category (Wants-based, Planned, Need-based, Impulsive).</li>
        </ul>
        <p style="margin:.4rem 0 0 0; color: rgb(246, 247, 239);"><strong>| Clustering:</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Size: 12,944; 10 features.</li>
        </ul>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Data Issues:</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Missing: Frequency of Purchases (1,131).</li>
          <li>Categorical: Gender, Item Purchased, Category, Location, Season, Discount Applied, Promo Code Used, Purchase Intent Category.</li>
        </ul>
      </div>

      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <div class="badge">Data Preparation</div>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Handling Missing Values</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Impute missing values using mode, as it best represents categorical frequency.</li>
        </ul>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Encoding Categorical Variables</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem">
          <li>Label Encoding to numeric; store encoders in le_dict for reuse and reversibility.</li>
        </ul>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| One-Hot Encoding for Apriori</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Requires a 0/1 item presence matrix; one-hot maps each category value to an item for support/confidence/lift.</li>
        </ul><br><br><br><br><br><br><br><br>
      </div>

      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <div class="badge">Modeling</div>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Classification</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Linear SVM, XGBoost, Decision Tree, Logistic Regression, Random Forest.</li>
          <li>Preprocessing: SMOTE, StandardScaler, hyperparameter tuning.</li>
          <li>Split using 80:20 ratio</li>
        </ul>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Unsupervised</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>KMeans clustering + Apriori rules.</li>
          <li>Preprocessing: MinMaxScaler</li>
        </ul><br><br><br><br><br><br><br><br><br><br>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

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

st.markdown("<br>", unsafe_allow_html=True)

# Prediction metrics
st.markdown(
    f"""
    <div class="intro-content"><br>
        <h3 style="margin-top:0;">|  Results based on multiple of machine learning Models</h3>
    </div>
    """,
    unsafe_allow_html=True,
)
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

img_xbg = b64_img(ASSETS / "xbg.jpg")
img_svmtune = b64_img(ASSETS / "svmtuning.jpg")

# create an img tag only if the image was found, otherwise show a small fallback message
img_xgb = f'<img src="data:image/png;base64,{img_xbg}" style="max-width:65%; height:auto; border-radius:12px;">' if img_xbg else '<div style="font-size:12px;color:#666;padding:16px;border:1px dashed #ccc;border-radius:8px;">Image not found: xbg.jpg</div>'
img_svm = f'<img src="data:image/png;base64,{img_svmtune}" style="max-width:65%; height:auto; border-radius:12px;">' if img_svmtune else '<div style="font-size:12px;color:#666;padding:16px;border:1px dashed #ccc;border-radius:8px;">Image not found: svmtuning.jpg</div>'
st.markdown(
    f"""
    <div class="intro-content"><br>
        <h3 style="margin-top:0;">|  Evaluation on Prediction</h3>
    </div>
    <div class="intro-wrapper">
      <div class="quote-box" style="text-align:center;">
        {img_xgb}
      </div>
      <div class="divider-vert"></div>
      <div class="intro-content"><br>
        <pd1>
        XGBoost with Tuning not only achieved the highest scores across all metrics but also demonstrated a well-balanced confusion matrix, with minimal false positives and false negatives. 
        </pd1>
      </div>
    </div>
    <br><br>
    <div class="intro-wrapper">
      <div class="intro-content"><br>
        <pd1>
        Worse, after hyperparameter tuning on RBF kernel, the model's predictions became even more skewed, classifying every instance to one category that is dominant. This resulted in a complete failure to recognize other purchase intent categories, despite the tuned model reporting a deceptively high accuracy. 
        </pd1>
      </div>
      <div class="divider-vert"></div>
      <div class="quote-box" style="text-align:center;">
        {img_svm}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br>", unsafe_allow_html=True)

# Clustering & Association Rules
img_ss = b64_img(ASSETS / "sse.jpg")
img_elb = b64_img(ASSETS / "elbow-silhouette.jpg")

# create an img tag only if the image was found, otherwise show a small fallback message
img_sse = f'<img src="data:image/png;base64,{img_ss}" style="max-width:55%; height:auto; border-radius:12px;">' if img_ss else '<div style="font-size:12px;color:#666;padding:16px;border:1px dashed #ccc;border-radius:8px;">Image not found: sse.jpg</div>'
img_elbow = f'<img src="data:image/png;base64,{img_elb}" style="max-width:55%; height:auto; border-radius:12px;">' if img_elb else '<div style="font-size:12px;color:#666;padding:16px;border:1px dashed #ccc;border-radius:8px;">Image not found: elbow-silhouette.jpg</div>'

st.markdown(
  f"""
  <div class="intro-content"><br>
        <h3 style="margin-top:0;">|  Evaluation on Clustering & Association Rules</h3>
  </div>
  <div class="deck-card" style="align-items:center; text-align:center; margin-top:0.6rem;">
    <br>
    <p style="margin:.4rem 0 0; color: rgb(246, 247, 239); font-size:16px; text-align:center;">| The Elbow Method uses inertia, also known as Sum of Squared Errors (SSE), which measures how tightly data points are grouped within each cluster. |</p>
    <div style="display:flex; justify-content:center; margin:16px 0;">
    {img_sse}
    </div>
    <p style="margin:.2rem 0 0; color: rgb(246, 247, 239); font-size:16px;text-align:center;">| Observation: k = 2 had the highest cohesion score (0.3993) but was too broad to capture complex customer behaviors. |</p>
    <div style="display:flex; justify-content:center; margin:16px 0;">
    {img_elbow}
    </div>
    <p style="margin:.2rem 0 0; color: rgb(246, 247, 239); font-size:16px; text-align:center;">| Selected k = 5 to extract richer behavioral insights without excessive splitting. |</p>
    <br>
  </div>
  """,
  unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# Cluster sizes
cluster_sizes_df = pd.DataFrame([
  (0, 4871, 37.63),
  (1, 1762, 13.61),
  (2, 1270, 9.81),
  (3, 3693, 28.53),
  (4, 1348, 10.41),
], columns=["Cluster","Count","Percent"])

st.markdown(
  """
  <br><br>
  <div class="intro-content"><br>
    <h3 style="margin-top:0;">|  Evaluation on Clustering</h3>
  </div>
  """,
  unsafe_allow_html=True,
)

st.dataframe(cluster_sizes_df, use_container_width=True)

st.markdown("""
  <div class="intro-wrapper">
      <div class="intro-content">
      <pd1>
      Cluster sizes in an uneven distribution, suggests that certain customer profiles have a significant influence on the dataset, an occurrence that is commonly observed in retail data where planned purchases or popular categories are more widespread.
      </pd1><br>
      </div>
    </div>
  """, unsafe_allow_html=True)

# Association Rules for Cluster 1 and 4
rules_df = pd.DataFrame([
    (1, "If Location = Canada, Season = Spring ⇒ Gender = Male, Frequency = Every 3 Months, Item Purchased = Shoes", 0.0335, 0.9365, 7.5348),
    (1, "If Location = Canada, Frequency = Every 3 Months, Season = Spring ⇒ Gender = Male, Item Purchased = Shoes", 0.0335, 0.9833, 6.1441),
    (1, "If Gender = Male, Location = Canada, Season = Spring ⇒ Item Purchased = Shoes, Frequency = Every 3 Months", 0.0335, 0.9516, 6.1195),
    (4, "If Promo Code Used = Yes, Location = Kenya ⇒ Gender = Male, Item Purchased = Jacket, Frequency = Every 3 Months, Discount Applied = Yes, Purchase Intent = Impulsive, Season = Winter", 0.0304, 0.9111, 16.1602),
    (4, "If Promo Code Used = Yes, Location = Kenya ⇒ Gender = Male, Item Purchased = Jacket, Frequency = Every 3 Months, Purchase Intent = Impulsive, Season = Winter", 0.0304, 0.9111, 16.1602),
    (4, "If Location = Kenya, Discount Applied = Yes ⇒ Item Purchased = Jacket, Promo Code Used = Yes, Frequency = Every 3 Months, Purchase Intent = Impulsive", 0.0304, 0.9111, 16.1602),
], columns=["Cluster","Rule","Support","Confidence","Lift"])

st.markdown(
  """
  <br><br>
  <div class="intro-content"><br>
    <h3 style="margin-top:0;">|  Evaluation on Apriori Rules<br></h3>
  </div>
  """,
  unsafe_allow_html=True,
)

st.dataframe(rules_df, use_container_width=True)

# Insights (still under Evaluation)
st.markdown(
    f"""
    <div class="intro-wrapper">
      <div class="intro-content"><br>
        <h3 style="margin-top:0;">Cluster 1</h3>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Shoes are strongly associated with purchases in Canada during spring.</li>
          <li>High confidence (93.6%) indicates consistent occurrence under these conditions.</li>
          <li>Represents planned seasonal buying behavior.</li>
        </ul><br>
      </div>
      <div class="divider-vert"></div>
      <div class="intro-content"><br>
        <h3 style="margin-top:0;">Cluster 4</h3>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Promo codes and discounts link to impulsive winter jacket purchases.</li>
          <li>Lift > 16 shows powerful promotional influence.</li>
          <li>Reflects promotion- and season-driven impulsivity.</li>
        </ul><br>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

# What Consumers Can Learn (under Evaluation)
st.markdown(
    """
    <div class="card-row" style="margin-top:.4rem;">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <div class="badge">What Consumers Can Learn</div>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Seasonality matters: certain products spike in specific seasons.</li>
          <li>Promotions trigger impulses: discounts and codes increase purchase likelihood.</li>
          <li>Actionable: plan essentials; recognize marketing nudges; use intent predictions to avoid unnecessary buys.</li>
        </ul>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <div class="badge">Dashboard Visualization</div>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Prediction Analysis page.</li>
          <li>Clustering Analysis page.</li>
        </ul><br><br>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

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

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="card-row">
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <div class="badge">Conclusion</div>
        <ul style="margin:.3rem 0 0 1.2rem;">
          <li>Developed and tuned five ML models; XGBoost achieved 98.26% accuracy, outperforming others.</li>
          <li>Identified seasonal and promotional factors driving impulsive buying through clustering and association rules. Applied KMeans clustering and Apriori rules to uncover purchase patterns and behavioral segments.</li>
          <li>Built a Streamlit dashboard with HTML/CSS enhancements visualization of predictions and trends.</li>
          <li>Personally handled data cleaning, feature engineering, model training, evaluation, clustering, rule mining, and dashboard design.</li>
          <li>Project meets objectives and demonstrates the value of predictive analytics in reducing impulsive buying and promoting mindful consumption.</li>
        </ul><br>
      </div>
      <div class="deck-card" style="align-items:flex-start; text-align:left;">
        <div class="badge">Recommendations</div>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Re-evaluate Cluster Numbers</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Explore k = 3 or k = 4 for better separation and interpretability.</li>
        </ul>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Improve Apriori Rules</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Validate and simplify rules to remove duplicates and noise by lift/confidence for clearer insights.</li>
        </ul>
        <p style="margin:.4rem 0 0; color: rgb(246, 247, 239);"><strong>| Enhance Dashboard & Deployment</strong></p>
        <ul style="margin:.2rem 0 0 1.2rem;">
          <li>Add business-focused features for deeper analytics.</li>
          <li>Deploy publicly to empower consumers and support strategic business decisions.</li>
        </ul><br>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

# ========================
# References
# ========================
st.markdown(
    """
    <div class="section-header" id="section-references">
      <div class="section-row">
        <div class="section-title">References</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
  """
  <br>
  <div style="text-align:left;font-size:14px;color:rgba(28,58,19,.7);">
    <ol style="margin:.3rem 0 0 1.2rem;">
    <li>Alfred, R., Loh, B. J., Obit, J. H., Lim, Y., &amp; Haviluddin, H. (2023). Concept Trending of Social Media Data Using Apriori Algorithm. IAENG International Journal of Computer Science, 50(1). <a href="https://www.iaeng.org/IJCS/issues_v50/issue_1/IJCS_50_1_32.pdf" target="_blank">Link</a></li>
    <li>Andriuška, G. (2025a, May 3). The Burden of Excess: How overconsumption haunts life and legacy. overconsumption.org. <a href="https://overconsumption.org/blogs/news/the-burden-of-excess-how-overconsumption-haunts-life-and-legacy" target="_blank">Link</a></li>
    <li>Andriuška, G. (2025b, May 3). The hidden cost of owning too much: How overconsumption steals our time, peace, and planet. overconsumption.org. <a href="https://overconsumption.org/blogs/news/overconsumption-environment-mental-health-solutions" target="_blank">Link</a></li>
    <li>Cernansky, R. (2024b, January 25). Fighting overconsumption: TikTok's deinfluencing movement and no-spend challenges are a wake-up call for brands. Vogue Business. <a href="https://www.voguebusiness.com/story/sustainability/tiktoks-anti-overconsumption-movement-rule-of-5-wake-up-call-for-brands" target="_blank">Link</a></li>
    <li>Ecommerce Consumer Behavior Analysis data. (2025, March 3). Kaggle. <a href="https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data" target="_blank">Link</a></li>
    <li>GeeksforGeeks. (2025a, August 22). K means Clustering - Introduction. <a href="https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/" target="_blank">Link</a></li>
    <li>GeeksforGeeks. (2025b, September 18). Apriori algorithm. <a href="https://www.geeksforgeeks.org/machine-learning/apriori-algorithm/" target="_blank">Link</a></li>
    <li>GeeksforGeeks. (2025c, October 3). Data PreProcessing with Sklearn using Standard and Minmax scaler. <a href="https://www.geeksforgeeks.org/machine-learning/data-pre-processing-wit-sklearn-using-standard-and-minmax-scaler/" target="_blank">Link</a></li>
    <li>GhorbanTanhaei, H., Boozary, P., Sheykhan, S., Rabiee, M., Rahmani, F., &amp; Hosseini, I. (2024). Predictive analytics in Customer behavior: Anticipating trends and preferences. Results in Control and Optimization, 100462. <a href="https://doi.org/10.1016/j.rico.2024.100462" target="_blank">Link</a></li>
    <li>Kasemrat, R., Kraiwanit, T., &amp; Yuenyong, N. (2025). Predictive analytics in customer behavior: Unveiling economic and governance insights through machine learning. Journal of Governance and Regulation, 14(1, SI), 318–331. <a href="https://doi.org/10.22495/jgrv14i1siart8" target="_blank">Link</a></li>
    <li>Lawton, M. (2025, January 8). “Maybe you'll realise what you have is good enough”: Why influencers are facing a pushback. BBC. <a href="https://www.bbc.com/culture/article/20250107-why-the-pushback-against-influencers-is-growing" target="_blank">Link</a></li>
    <li>Lin, J. (2025). Application of machine learning in predicting consumer behavior and precision marketing. PLoS ONE, 20(5), e0321854. <a href="https://doi.org/10.1371/journal.pone.0321854" target="_blank">Link</a></li>
    <li>Patil, V. (2021, August 23). Clustering and profiling customers using K-Means. Medium. <a href="https://medium.com/analytics-vidhya/clustering-and-profiling-customers-using-k-means-9afa4277427" target="_blank">Link</a></li>
    <li>Understanding 'underconsumption core': How a new trend is challenging consumer culture. (2024, July 31). The Straits Times. <a href="https://www.straitstimes.com/opinion/understanding-underconsumption-core-how-a-new-trend-is-challenging-consumer-culture" target="_blank">Link</a></li>
    <li>Wu, L., &amp; Wang, Z. (2020). Research on Top-K Association rules mining algorithm based on clustering. Journal of Physics: Conference Series, 1682(1), 012064. <a href="https://doi.org/10.1088/1742-6596/1682/1/012064" target="_blank">Link</a></li>
    <li>Zhou, Y., Loi, A. M., Tan, G. W., Lo, P., &amp; Lim, W. (2022). The survey dataset of the influence of theory of planned behaviour on purchase behaviour on social media. Data in Brief, 42, 108239. <a href="https://doi.org/10.1016/j.dib.2022.108239" target="_blank">Link</a></li>
    </ol>
  </div>
  """,
  unsafe_allow_html=True,
)

