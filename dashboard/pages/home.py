import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --- Top bar ---
st.markdown(
    "<h1 style='color:#FF6F61; margin-bottom:0;'>Purchase Intent Analysis</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0;margin-bottom:1.5em;border-top:2px solid #222;'>", unsafe_allow_html=True)

# --- Load Data ---
DATA_PATH = Path(__file__).parent.parent / "data" / "cleaned_prediction.csv"

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    st.warning("Prediction dataset not found. Showing demo statistics.")
    df = pd.DataFrame({
        "Purchase Intent Category": np.random.choice(
            ["Impulsive", "Need-based", "Planned", "Wants-based"], 200
        ),
        "Category": np.random.choice(["Accessories", "Clothing", "Outerwear"], 200),
        "Location": np.random.choice(["Malaysia", "Spain", "UK", "Argentina"], 200),
        "Confidence Score": np.random.uniform(0.7, 0.99, 200),
        "Review Rating": np.random.uniform(1, 5, 200),
    })

# --- Section 1: Overview Cards (Top Row) ---
total_preds = len(df)
impulsive_count = (df["Purchase Intent Category"] == "Impulsive").sum()
intentional_count = total_preds - impulsive_count
impulsive_pct = 100 * impulsive_count / total_preds if total_preds else 0
intentional_pct = 100 - impulsive_pct
avg_conf = df["Confidence Score"].mean() if "Confidence Score" in df.columns else np.nan

kpi1, kpi2, kpi3 = st.columns([1, 1.2, 1])

with kpi1:
    st.metric("Total Predictions Made", f"{total_preds:,}")

with kpi2:
    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=["Impulsive", "Intentional"],
                values=[impulsive_count, intentional_count],
                hole=0.5,
                marker_colors=["#FF6F61", "#6EC6FF"],
                textinfo="label+percent"
            )
        ]
    )
    pie_fig.update_layout(
        title_text="Impulsive vs. Intentional",
        showlegend=False,
        margin=dict(t=40, b=0, l=0, r=0),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFF"
    )
    st.plotly_chart(pie_fig, use_container_width=True)

with kpi3:
    try:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_conf*100 if not np.isnan(avg_conf) else 0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Avg. Confidence (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF6F61"},
                'bgcolor': "#222",
                'borderwidth': 2,
                'bordercolor': "#444",
            }
        ))
        gauge_fig.update_layout(height=220, margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge_fig, use_container_width=True)
    except Exception:
        st.metric("Avg. Confidence Score", f"{avg_conf*100:.1f}%" if not np.isnan(avg_conf) else "N/A")

st.markdown("<hr style='margin-top:0.5em;margin-bottom:1.5em;border-top:1px solid #333;'>", unsafe_allow_html=True)

# --- Section 2: Charts (Middle Rows, 2x3 grid) ---
chartcol1, chartcol2, chartcol3 = st.columns(3)
chartcol4, chartcol5, _ = st.columns(3)  # Remove chartcol6

# Chart 1: Distribution of Intent Categories (Horizontal bar)
with chartcol1:
    st.subheader("Distribution of Intent Categories")
    if "Purchase Intent Category" in df.columns:
        intent_counts = df["Purchase Intent Category"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig_bar = px.bar(
            intent_counts,
            x="Count",
            y="Intent",
            color="Intent",
            orientation="h",
            title="",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_bar.update_layout(
            showlegend=False,
            height=300,
            plot_bgcolor="#222",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FFF"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Chart 2: Intent by Product Category (Grouped bar)
with chartcol2:
    st.subheader("Intent by Product Category")
    if "Category" in df.columns and "Purchase Intent Category" in df.columns:
        cat_intent = df.groupby(["Category", "Purchase Intent Category"]).size().reset_index(name="Count")
        fig_grouped = px.bar(
            cat_intent,
            x="Category",
            y="Count",
            color="Purchase Intent Category",
            barmode="group",
            title="",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_grouped.update_layout(
            height=300,
            plot_bgcolor="#222",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FFF"
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

# Chart 3: Intent by Location (Bar chart)
with chartcol3:
    st.subheader("Intent by Location")
    if "Location" in df.columns and "Purchase Intent Category" in df.columns:
        loc_intent = df.groupby(["Location", "Purchase Intent Category"]).size().reset_index(name="Count")
        fig_loc = px.bar(
            loc_intent,
            x="Location",
            y="Count",
            color="Purchase Intent Category",
            barmode="group",
            title="",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_loc.update_layout(
            height=300,
            plot_bgcolor="#222",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FFF"
        )
        st.plotly_chart(fig_loc, use_container_width=True)

# Chart 4: Confidence Score Distribution (Histogram)
with chartcol4:
    st.subheader("Confidence Score Distribution")
    if "Confidence Score" in df.columns:
        fig_hist = px.histogram(
            df,
            x="Confidence Score",
            nbins=20,
            title="",
            color_discrete_sequence=["#FF6F61"]
        )
        fig_hist.update_layout(
            height=300,
            plot_bgcolor="#222",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FFF"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# Chart 5: Intent vs. Review Rating (Heatmap)
with chartcol5:
    st.subheader("Intent vs. Review Rating")
    if "Purchase Intent Category" in df.columns and "Review Rating" in df.columns:
        heatmap_data = df.groupby(["Purchase Intent Category", "Review Rating"]).size().reset_index(name="Count")
        fig_heat = px.density_heatmap(
            heatmap_data,
            x="Review Rating",
            y="Purchase Intent Category",
            z="Count",
            color_continuous_scale="Peach",
            title=""
        )
        fig_heat.update_layout(
            height=300,
            plot_bgcolor="#222",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FFF"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("<hr style='margin-top:0.5em;margin-bottom:1.5em;border-top:1px solid #333;'>", unsafe_allow_html=True)

# --- Section 3: Feature Importance + Suggested Actions ---
featcol1, featcol2 = st.columns([1.2, 1])

with featcol1:
    st.subheader("Top Influencing Features")
    # Dummy feature importance (replace with model SHAP or feature_importances_)
    importance_data = pd.DataFrame({
        'Feature': ['Category', 'Location', 'Season', 'Frequency of Purchases', 'Review Rating'],
        'Importance': np.random.rand(5)
    }).sort_values(by='Importance', ascending=True)

    fig = px.bar(
        importance_data,
        x='Importance',
        y='Feature',
        orientation='h',
        title="",
        labels={'Importance': 'Score'},
        color='Importance',
        color_continuous_scale=px.colors.sequential.Peach
    )
    fig.update_layout(
        height=350,
        plot_bgcolor="#222",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FFF",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

with featcol2:
    st.subheader("Suggested Actions Summary")
    if impulsive_pct > 50:
        st.info("Impulsive purchases dominate → Recommend sustainability campaigns.", icon="🌱")
    elif intentional_pct > 50:
        st.info("Intentional purchases dominate → Focus on loyalty and personalized offers.", icon="🎁")
    elif avg_conf > 0.85:
        st.info("High confidence in predictions → Use insights for targeted marketing.", icon="✅")
    else:
        st.info("Balanced intent types. Consider targeted strategies for each segment.", icon="💡")
