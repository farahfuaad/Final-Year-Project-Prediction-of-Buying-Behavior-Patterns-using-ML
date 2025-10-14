import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --- Load Data ---

# For deployment, uncomment the line below and comment the line after
#pred_data = Path(__file__).parent.parent / "data" / "cleaned_prediction.csv"

# For local testing, uncomment the line below and comment the line above
pred_data = pd.read_csv("/Users/farahfuaad/Desktop/fyp/Final-Year-Project-Prediction-of-Consumer-Behaviour-using-ML/data/cleaned_prediction.csv")
df = pred_data

st.subheader("Purchase Intent Analysis")
st.markdown("<br>", unsafe_allow_html=True)

# Top section layout
layout_col1, layout_col2 = st.columns([2, 1])

with layout_col1:

    # variables for cards
    total_preds = len(df)
    impulsive_count = (df["Purchase Intent Category"] == "Impulsive").sum()
    intentional_count = total_preds - impulsive_count
    impulsive_pct = 100 * impulsive_count / total_preds if total_preds else 0
    intentional_pct = 100 - impulsive_pct
    top_intent = df["Purchase Intent Category"].value_counts().idxmax()
    top_intent_count = df["Purchase Intent Category"].value_counts().max()

    # cards layout
    kpi1, kpi2, kpi3 = st.columns([1, 1.2, 1])

    with kpi1:
        with st.container():
            st.markdown(
                f'''
                <div class="card-container" style="line-height: 1;">
                <p>{total_preds:,}</p>
                <br>
                Total Predictions
                </div>
                ''',
                unsafe_allow_html=True
            )

    with kpi2:
        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Impulsive", "Intentional"],
                    values=[impulsive_count, intentional_count],
                    hole=0.6,
                    textinfo="percent+label"
                )
            ]
        )
        pie_fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=150,
            showlegend=False
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with kpi3:
        with st.container():
            st.markdown(
                f'''
                <div class="card-container" style="line-height: 1;">
                <p>{top_intent}</p>
                <br>
                Top Intent
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart: Intent by Product Category
    st.markdown("Intent by Product Category")
    if "Category" in df.columns and "Purchase Intent Category" in df.columns:
        cat_intent = df.groupby(["Category", "Purchase Intent Category"]).size().reset_index(name="Count")
        fig_grouped = px.bar(
            cat_intent,
            x="Category",
            y="Count",
            color="Purchase Intent Category",
            barmode="group"
        )
        fig_grouped.update_layout(
            height=300
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    # Chart: Intent by Location
    st.subheader("Intent by Location")
    if "Location" in df.columns and "Purchase Intent Category" in df.columns:
        loc_intent = df.groupby(["Location", "Purchase Intent Category"]).size().reset_index(name="Count")
        fig_loc = px.bar(
            loc_intent,
            x="Location",
            y="Count",
            color="Purchase Intent Category",
            barmode="group"
        )
        fig_loc.update_layout(
            height=200
        )
        st.plotly_chart(fig_loc, use_container_width=True)

# Layout 2
with layout_col2:
    st.subheader("Intent by Discount Applied")
    if "Discount Applied" in df.columns and "Purchase Intent Category" in df.columns:
        discount_intent = df.groupby(["Discount Applied", "Purchase Intent Category"]).size().reset_index(name="Count")
        for discount in discount_intent["Discount Applied"].unique():
            sub_df = discount_intent[discount_intent["Discount Applied"] == discount]
            pie = go.Figure(
                data=[
                    go.Pie(
                        labels=sub_df["Purchase Intent Category"],
                        values=sub_df["Count"],
                        hole=0.5,
                        textinfo="label+percent"
                    )
                ]
            )
            pie.update_layout(
                title_text=f"Discount: {discount}",
                showlegend=False,
                margin=dict(t=100, b=100, l=100, r=100),
                height=400
            )
            st.plotly_chart(pie, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# bottom section layout
tablecol1, tablecol2 = st.columns(2)

with tablecol1:
    st.subheader("Summary Table")
    summary = df.groupby(["Category", "Location", "Season"]).agg(
        Total_Purchases=("Purchase Intent Category", "count"),
        Impulsive_Purchases=("Purchase Intent Category", lambda x: (x == "Impulsive").sum()),
        Avg_Review_Rating=("Review Rating", "mean")
    ).reset_index()
    st.dataframe(summary, use_container_width=True)

with tablecol2:
    st.subheader("Top N Insights Table")
    top_items = df[df["Purchase Intent Category"] == "Impulsive"]["Item Purchased"].value_counts().head(10).reset_index()
    top_items.columns = ["Item Purchased", "Impulsive Purchases"]
    st.dataframe(top_items, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Feature Importance by Purchase Intent Category ---
st.subheader("Feature Importance by Purchase Intent Category")

categories = df["Purchase Intent Category"].unique()
feature_list = ['Gender', 'Item Purchased', 'Category', 'Location', 'Season', 'Discount Applied', 'Promo Code Used', 
                'Frequency of Purchases']

col1, col2 = st.columns(2)
for idx, intent in enumerate(categories):
    col = col1 if idx % 2 == 0 else col2
    with col:
        st.markdown(f"**{intent}**")
        importance_data = pd.DataFrame({
            'Feature': feature_list,
            'Importance': np.random.rand(len(feature_list))
        }).sort_values(by='Importance', ascending=True)

        fig = px.bar(
            importance_data,
            x='Importance',
            y='Feature',
            orientation='h',
            labels={'Importance': 'Score'},
            color='Importance'
        )
        fig.update_layout(
            height=250,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)