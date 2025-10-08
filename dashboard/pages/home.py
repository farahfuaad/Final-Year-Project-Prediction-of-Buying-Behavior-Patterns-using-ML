import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# st.set_page_config(page_title="Behavior Prediction", layout="wide")  # removed: call once in main app

DATA_PATH = Path(__file__).parent.parent / "data" / "cleaned_prediction.csv"
MODEL_PATH = Path(__file__).parent.parent / "ml" / "purchase_intent_model.pkl"

# Load dataset (used to build training model and populate choice lists)
if DATA_PATH.exists():
    df_all = pd.read_csv(DATA_PATH)
else:
    df_all = pd.DataFrame()

st.title("Purchase Intent Prediction")

# --- Section 1: User Input Form ---
with st.form("user_input_form"):
    st.header("Simulate / Input Behavior Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        gender_options = df_all["Gender"].dropna().unique().tolist() if "Gender" in df_all.columns else ["Male", "Female", "Other"]
        gender = st.selectbox("Gender", gender_options)
    with c2:
        category_options = df_all["Category"].dropna().unique().tolist() if "Category" in df_all.columns else ["Clothing", "Accessories", "Footwear", "Outerwear", "Other"]
        product_category = st.selectbox("Product Category Browsed", category_options)
        time_spent = st.number_input("Time Spent on Page (seconds)", min_value=0, max_value=3600, value=60)
    with c3:
        social_score = st.slider("Social Media Influence Score (1-10)", 1, 10, 5)
        freq_options = df_all["Frequency of Purchases"].dropna().unique().tolist() if "Frequency of Purchases" in df_all.columns else ["Monthly","Every 3 Months","Quarterly","Annually","Weekly"]
        past_freq = st.selectbox("Past Purchase Frequency", freq_options)
    submitted = st.form_submit_button("Predict Purchase Intent")

# --- Utility: build / load model ---
@st.cache_data(show_spinner=False)
def build_or_load_model(df: pd.DataFrame):
    # features we will use for prediction (match user inputs)
    features = ["Age", "Gender", "Category", "Frequency of Purchases"]
    target = "Purchase Intent Category"
    # require columns
    if not set(features + [target]).issubset(df.columns):
        return None, None

    # Prepare training data (drop rows with missing target or features)
    train_df = df[features + [target]].dropna()
    X = train_df[features]
    y = train_df[target].astype(str)

    # Column transformer
    cat_cols = ["Gender", "Category", "Frequency of Purchases"]
    num_cols = ["Age"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ], remainder="drop")

    pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipe.fit(X, y)
    # save a copy for reuse (best-effort)
    try:
        joblib.dump(pipe, MODEL_PATH)
    except Exception:
        pass
    return pipe, features

# Try load model file if present, else build from CSV
model = None
model_features = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        # attempt to infer feature names from training step; fallback to our features
        model_features = ["Age", "Gender", "Category", "Frequency of Purchases"]
    except Exception:
        model = None

if model is None and not df_all.empty:
    model, model_features = build_or_load_model(df_all)

# --- Section 2: Prediction Output ---
if submitted:
    st.header("Prediction Output")
    if model is None:
        st.error("No model available — ensure data file [data/cleaned_prediction.csv] exists and includes 'Purchase Intent Category'. See ml/intention_prediction.ipynb for training details.")
    else:
        # prepare input for model
        input_dict = {
            "Age": [int(age)],
            "Gender": [gender],
            "Category": [product_category],
            "Frequency of Purchases": [past_freq],
        }
        X_input = pd.DataFrame(input_dict)
        try:
            pred = model.predict(X_input)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[0]
                classes = model.classes_
                proba = float(np.max(probs))
                probs_dict = dict(zip(classes, [float(p) for p in probs]))
            else:
                proba = 0.0
                probs_dict = {str(pred): 1.0}
            # Display results
            col1, col2 = st.columns([2,1])
            with col1:
                st.metric("Predicted Purchase Intent", str(pred))
                st.write(f"Confidence: {proba*100:.0f}%")
                # show brief suggested action mapping (optional)
                action_map = {
                    "Impulsive": "Offer discount",
                    "Need-based": "Promote sustainable alternative",
                    "Trend-driven": "Highlight new arrivals",
                    "Wants-based": "General promotion / browse assistance"
                }
                suggested = action_map.get(str(pred), "Nudge / A/B test offers")
                st.info(f"Suggested Action: {suggested}")
            with col2:
                st.write("Prediction probabilities")
                st.json(probs_dict)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Show small help + data links
st.markdown("Model target: `Purchase Intent Category` (sourced from dataset column).")
st.markdown("- Dataset: " + ("[data/cleaned_prediction.csv](../data/cleaned_prediction.csv)" if DATA_PATH.exists() else "`data/cleaned_prediction.csv` not found"))
st.markdown("- Training notebook (reference): [ml/intention_prediction.ipynb](../../ml/intention_prediction.ipynb)")

# Optional: display a few rows from dataset for context
if not df_all.empty:
    with st.expander("Sample data (first 5 rows)"):
        st.dataframe(df_all.head()[["Age","Gender","Category","Frequency of Purchases","Purchase Intent Category"]])

