import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --- Top Header ---
st.markdown(
    """
    <div style='display:flex;align-items:center;justify-content:space-between;'>
        <div>
            <h1>Purchase Intent Analysis</h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.5em;margin-bottom:1.5em;border-top:1px solid #23272f;'>", unsafe_allow_html=True)

# --- Load Data ---
pred_data = pd.read_csv("/Users/farahfuaad/Desktop/fyp/Final-Year-Project-Prediction-of-Consumer-Behaviour-using-ML/data/cleaned_prediction.csv")
df = pred_data

# --- KPI Cards Row ---
total_preds = len(df)
impulsive_count = (df["Purchase Intent Category"] == "Impulsive").sum()
intentional_count = total_preds - impulsive_count
impulsive_pct = 100 * impulsive_count / total_preds if total_preds else 0
intentional_pct = 100 - impulsive_pct
top_intent = df["Purchase Intent Category"].value_counts().idxmax()
top_intent_count = df["Purchase Intent Category"].value_counts().max()

kpi1, kpi2, kpi3 = st.columns([1, 1.2, 1])

with kpi1:
    st.metric("Total Predictions", f"{total_preds:,}")

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
        height=180,
        showlegend=False
    )
    st.plotly_chart(pie_fig, use_container_width=True)

with kpi3:
    st.metric("Top Intent", f"{top_intent} ({top_intent_count})")

st.markdown("<br>", unsafe_allow_html=True)

# --- Main Grid Section ---
maincol1, maincol2 = st.columns([2, 1])

with maincol1:
    st.subheader("Distribution of Intent Categories")
    if "Purchase Intent Category" in df.columns:
        intent_counts = df["Purchase Intent Category"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig_bar = px.bar(
            intent_counts,
            x="Count",
            y="Intent",
            color="Intent",
            orientation="h"
        )
        fig_bar.update_layout(
            showlegend=False,
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with maincol2:
    st.subheader("Intent by Product Category")
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
            height=140
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

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
            height=140
        )
        st.plotly_chart(fig_loc, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Additional Charts Row (3 cards) ---
addcol1, addcol2, addcol3 = st.columns(3)

with addcol1:
    st.subheader("Intent by Season")
    if "Season" in df.columns and "Purchase Intent Category" in df.columns:
        season_intent = df.groupby(["Season", "Purchase Intent Category"]).size().reset_index(name="Count")
        fig_season = px.bar(
            season_intent,
            x="Season",
            y="Count",
            color="Purchase Intent Category",
            barmode="stack"
        )
        fig_season.update_layout(
            height=180
        )
        st.plotly_chart(fig_season, use_container_width=True)

with addcol2:
    st.subheader("Intent vs. Review Rating")
    if "Purchase Intent Category" in df.columns and "Review Rating" in df.columns:
        fig_box = px.box(
            df,
            x="Purchase Intent Category",
            y="Review Rating",
            color="Purchase Intent Category"
        )
        fig_box.update_layout(
            height=180,
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)

with addcol3:
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
                margin=dict(t=10, b=10, l=10, r=10),
                height=120
            )
            st.plotly_chart(pie, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Bottom Section: Tables and Insights ---
tablecol1, tablecol2 = st.columns(2)

with tablecol1:
    st.subheader("Summary Table")
    summary = df.groupby(["Category", "Location", "Season"]).agg(
        Total_Purchases=("Purchase Intent Category", "count"),
        Impulsive_Purchases=("Purchase Intent Category", lambda x: (x == "Impulsive").sum()),
        Avg_Review_Rating=("Review Rating", "mean")
    ).reset_index()
    st.dataframe(summary, use_container_width=True)
    st.download_button("Download Summary Table", summary.to_csv(index=False), file_name="summary_table.csv")

with tablecol2:
    st.subheader("Top N Insights Table")
    top_items = df[df["Purchase Intent Category"] == "Impulsive"]["Item Purchased"].value_counts().head(10).reset_index()
    top_items.columns = ["Item Purchased", "Impulsive Purchases"]
    st.dataframe(top_items, use_container_width=True)
    st.download_button("Download Top Insights", top_items.to_csv(index=False), file_name="top_impulsive_items.csv")

st.markdown("<br>", unsafe_allow_html=True)

# --- Feature Importance & Recommendations ---
featcol1, featcol2 = st.columns([1.2, 1])

with featcol1:
    st.subheader("Feature Importance")
    importance_data = pd.DataFrame({
        'Feature': ['Category', 'Location', 'Season', 'Frequency of Purchases', 'Review Rating'],
        'Importance': np.random.rand(5)
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
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

with featcol2:
    st.subheader("Suggested Actions")
    with st.expander("See Recommendations"):
        if impulsive_pct > 50:
            st.info("Impulsive purchases dominate → Recommend sustainability campaigns.", icon="🌱")
        elif intentional_pct > 50:
            st.info("Intentional purchases dominate → Focus on loyalty and personalized offers.", icon="🎁")
        else:
            st.info("Balanced intent types. Consider targeted strategies for each segment.", icon="💡")