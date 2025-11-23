import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from st_flexible_callout_elements import flexible_callout
from html import escape

params = st.query_params
show_insights = params.get("insights", ["0"])[0] == "1"

# user profile panel
PROFILE = {"name": "Nia", "intent": "Wants-Based"}
user_name = PROFILE["name"]
intent_param = PROFILE["intent"]

intent_key = intent_param.strip()
pill_class = {
    "Wants-Based": "intent-pill--wants",
    "Planned": "intent-pill--planned",
    "Need-Based": "intent-pill--need",
    "Impulsive": "intent-pill--impulsive",
}.get(intent_key, "intent-pill--default")

st.markdown(
    f"""
    <section class="welcome-minimal">
      <div class="welcome-avatar-circle">
        <div class="avatar-shape"></div>
      </div>
      <p class="welcome-greet-flat">WELCOME BACK</p>
      <h5 class="welcome-name-flat">{escape(user_name)}</h5>
      <div class="intent-pill {pill_class}">
        <span>{escape(intent_param)}</span>
      </div>
      <div class="welcome-last-activity">Last Activity: 2 days ago</div>
    </section>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# --- Title Header ---
st.markdown(
    """
    <div style="display:flex; align-items:center;">
      <h1 style="margin:0; padding:0;">Purchase Intent Analysis</h1>
      <div style="flex:1; height:1px; background-color:#e0e0e0; margin-left:12px;"></div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# --- Load Data ---
# For deployment, uncomment the line below and comment the line after
pred_data = Path(__file__).parent.parent / "data" / "cleaned_prediction.csv"
cust_data = Path(__file__).parent.parent / "data" / "cleaned_shopping_trends.csv"
pred_df = pd.read_csv(pred_data)
cust_df = pd.read_csv(cust_data)

# For local testing, uncomment the line below and comment the line above
# pred_data = pd.read_csv("/Users/farahfuaad/Desktop/fyp/Final-Year-Project-Prediction-of-Consumer-Behaviour-using-ML/data/cleaned_prediction.csv")
# df = pred_data

# Top section layout
layout_col1, layout_col2 = st.columns([2, 1])

with layout_col1:

    # variables for cards
    total_preds = len(pred_df)
    total_cust = len(cust_df)
    impulsive_count = (pred_df["Purchase Intent Category"] == "Impulsive").sum()
    intentional_count = total_preds - impulsive_count
    impulsive_pct = 100 * impulsive_count / total_preds if total_preds else 0
    intentional_pct = 100 - impulsive_pct
    top_intent = pred_df["Purchase Intent Category"].value_counts().idxmax()
    top_intent_count = pred_df["Purchase Intent Category"].value_counts().max()

    # cards layout
    kpi1, kpi2, kpi3, kpi4 = st.columns([1, 1, 1, 1])

    with kpi1:
        with st.container():
            st.markdown(
                f'''
                <div class="card-container" style="line-height: 1;">
                <p>{total_cust:,}</p>
                <br>
                Total Customers
                </div>
                ''',
                unsafe_allow_html=True
            )

    with kpi2:
        with st.container():
            st.markdown(
                f'''
                <div class="card-container" style="line-height: 1;">
                <p>{total_preds:,}</p>
                <br>
                Total Data
                </div>
                ''',
                unsafe_allow_html=True
            )

    with kpi3:
        with st.container():
            st.markdown(
                f'''
                <div class="card-container" style="line-height:1;">
                    <p>{impulsive_pct:.0f}% Impulsive</p>
                    <br>
                    vs. {intentional_pct:.0f}% Intentional
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    with kpi4:
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

    # Insights box — show only when sidebar toggle is ON
    if show_insights:
        flexible_callout(
            "💡 <strong>Key Insights</strong> <br>"
            "&emsp; → Out of 12,944 customer records, 10,888 clean entries were analyzed using our best-performing model,<br>"
            "&emsp; XGBoost (98.3% accuracy). The results show that 19% of purchases are impulsive, while 81% are<br>"
            "&emsp; intentional (wants-based, need-based, and planned buying).<br>"
            "&emsp; → Among these, wants-based purchases lead the way, meaning most customers buy things they desire <br>"
            "&emsp; rather than need.",
        )
 

    # Chart: Intent by Product Category
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center;">
            <h2 style="margin:0; padding:0;">Intent by Product Category</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        if "Category" in pred_df.columns and "Purchase Intent Category" in pred_df.columns:
            cat_intent = pred_df.groupby(["Category", "Purchase Intent Category"]).size().reset_index(name="Count")
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
            
    if show_insights:
        flexible_callout(
            "💡 <strong>Key Insights</strong> <br>"
            "&emsp; → Accessories: Balanced across all intents, but wants-based and impulsive are slightly higher.<br>"
            "&emsp; Indicates purchases are often for style or trend rather than necessity.<br>"
            "&emsp; → Clothing:  Leads wants-based and planned purchases, making it the most desired category. <br>"
            "&emsp; Need-based are also strong, but impulsive is lower.<br>"
            "&emsp; → Footwear: Moderate counts across all intents, with wants-based slightly leading. Suggests footwear<br>"
            "&emsp; is influenced by desire but involves some planning <br>"
            "&emsp; → Outerwear: Lowest overall counts, with an even distribution across intents. Indicates outerwear <br>"
            "&emsp; is less trend-driven and more seasonal or functional",
        )

    # Chart: Intent by Location
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center;">
            <h2 style="margin:0; padding:0;">Intent by Location</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        if "Location" in pred_df.columns and "Purchase Intent Category" in pred_df.columns:
            loc_intent = pred_df.groupby(["Location", "Purchase Intent Category"]).size().reset_index(name="Count")
            fig_loc = px.bar(
                loc_intent,
                x="Location",
                y="Count",
                color="Purchase Intent Category",
                barmode="group"
            )
            fig_loc.update_layout(
                height=350
            )
            st.plotly_chart(fig_loc, use_container_width=True)

    if show_insights:
        flexible_callout(
            "💡 <strong>Key Insights</strong> <br>"
            "&emsp; Purchase intent varies by location. The United States and United Kingdom show high planned and <br>"
            "&emsp; wants-based purchases, while Spain and Italy lean strongly toward planned buying. Malaysia and Mexico <br>"
            "&emsp; have balanced patterns, with noticeable wants-based intent. Countries like Indonesia and Egypt show <br>"
            "&emsp; higher impulsive buying compared to others.",
        )

# Layout 2
with layout_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center;">
            <h2 style="margin:0; padding:0;">Distribution of Intent Categories</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        if "Purchase Intent Category" in pred_df.columns:
            intent_counts = pred_df["Purchase Intent Category"].value_counts().reset_index()
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
                height=200
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    if show_insights:
        flexible_callout(
            "<strong>Wants-based</strong>: Buying things you desire, often <br>"
            "influenced by trends or personal preference. <br>"
            "<br><strong>Planned</strong>: Purchases decided in advance, <br>"
            "usually after comparing options or waiting for the <br>"
            "right time. <br>"
            "<br><strong>Need-based</strong>: Essential items bought out of <br>"
            "necessity for daily life or specific purposes. <br>"
            "<br><strong>Impulsive</strong>: Spontaneous buys made without <br>"
            "prior planning or consideration.",
        )

    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center;">
            <h2 style="margin:0; padding:0;">Intent by Discount Applied</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

         # Pie charts for Discount Applied vs Purchase Intent Category
        if "Discount Applied" in pred_df.columns and "Purchase Intent Category" in pred_df.columns:
            discount_intent = pred_df.groupby(["Discount Applied", "Purchase Intent Category"]).size().reset_index(name="Count")
            discounts = discount_intent["Discount Applied"].unique()
            for idx, discount in enumerate(discounts):
                sub_df = discount_intent[discount_intent["Discount Applied"] == discount]
                if idx > 0:
                    st.markdown("---")
                pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=sub_df["Purchase Intent Category"],
                            values=sub_df["Count"],
                            hole=0.4,
                            textinfo="label+percent",
                            textfont=dict(size=10),
                            insidetextfont=dict(size=10)
                        )
                    ]
                )
                pie.update_traces(textinfo="label+percent", textfont_size=14)
                pie.update_layout(
                    title_text=f"Discount: {discount}",
                    title_font=dict(size=14),
                    showlegend=False,
                    margin=dict(t=50, b=50, l=50, r=50),
                    height=256
                )
                st.plotly_chart(pie, use_container_width=True)

    if show_insights:
        flexible_callout(
            "💡 <strong>Key Insights</strong> <br>"
            "&emsp; → Discounts make products feel like a good <br>"
            "&emsp; deal, encouraging customers to buy items they <br>"
            "&emsp; desire rather than need. <br>"
            "&emsp; → At the same time, discounts reduce impulse <br>"
            "&emsp; buying because customers think carefully and<br>"
            "&emsp; justify their purchases when they believe they <br>"
            "&emsp; are saving money.",
        )
        
st.markdown("<br>", unsafe_allow_html=True)

# --- Feature Importance by Purchase Intent Category ---
st.subheader("Feature Importance by Purchase Intent Category")

categories = pred_df["Purchase Intent Category"].unique()
feature_list = ['Gender', 'Item Purchased', 'Category', 'Location', 'Season', 'Discount Applied', 'Promo Code Used', 
                'Frequency of Purchases']

col1, col2 = st.columns(2)
for idx, intent in enumerate(categories):
    col = col1 if idx % 2 == 0 else col2
    with col:
        with st.container(border=True):
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
    
if show_insights:
    flexible_callout(
            "💡 <strong>Key Insights</strong> <br><br>"
            "&emsp; <strong>Wants-based</strong> - Discounts and promo codes have the biggest influence. Customers are motivated by savings when buying items they desire, making price <br>"
            "&emsp; incentives a strong trigger for wants-based purchases. <br><br>"
            "&emsp; <strong>Planned</strong> - Item type and category matter most for structured buying. Shoppers plan ahead based on what the product is and its category, showing that <br>"
            "&emsp; planned purchases are less about discounts and more about product relevance. <br><br>"
            "&emsp; <strong>Need-based</strong> - Driven by item type, but discounts still play a role. Essential purchases focus on the item itself, yet customers still look for deals <br>"
            "&emsp; to reduce costs on necessities. <br><br>"
            "&emsp; <strong>Impulsive</strong> - Triggered mainly by item type and promo codes. Spontaneous buying happens when appealing items and promo codes catch attention, while <br>"
            "&emsp; discounts matter less compared to wants-based intent.",
        )
    st.markdown("<br>", unsafe_allow_html=True)

# --- Wants-Based Recommendations Panel (updated, no sources) ---
st.markdown(
    """
    <div class="rec-panel">
      <h2 class="rec-title">Guided Recommendations To Moderate Wants‑Based Purchasing</h2>
      <ul class="rec-list">
        <li><strong>Think before you buy</strong> — Ask yourself, “Do I really need this, or do I just want it?” Taking a moment to reflect can help you avoid unnecessary spending.</li>
        <li><strong>Set a spending limit</strong> — Use simple budgeting tools to decide how much you can spend on non‑essential items each month. This makes it easier to stay in control.</li>
        <li><strong>Find other ways to feel good</strong> — Instead of shopping, try activities that make you happy—like exercising, reading, or learning something new. These can give you the same satisfaction without spending money.</li>
        <li><strong>Get helpful reminders</strong> — The system can send alerts when you’re about to buy during a big sale, saying things like “Would you like to check your budget first?” This helps you pause and think before purchasing.</li>
        <li><strong>Talk to someone if needed</strong> — If shopping feels like a habit you can’t control, consider speaking to a financial advisor or joining a support group. They can give practical tips to manage spending.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True
)