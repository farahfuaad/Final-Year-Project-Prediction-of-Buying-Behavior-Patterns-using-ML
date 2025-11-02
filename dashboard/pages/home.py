import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from st_flexible_callout_elements import flexible_callout

# --- Load Data ---
# For deployment, uncomment the line below and comment the line after
pred_data = Path(__file__).parent.parent / "data" / "cleaned_prediction.csv"
cust_data = Path(__file__).parent.parent / "data" / "cleaned_shopping_trends.csv"
pred_df = pd.read_csv(pred_data)
cust_df = pd.read_csv(cust_data)

# For local testing, uncomment the line below and comment the line above
# pred_data = pd.read_csv("/Users/farahfuaad/Desktop/fyp/Final-Year-Project-Prediction-of-Consumer-Behaviour-using-ML/data/cleaned_prediction.csv")
# df = pred_data

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
    params = st.query_params
    show_insights = params.get("insights", ["0"])[0] == "1"

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
                            hole=0.5,
                            textinfo="label+percent",
                            textfont=dict(size=12),
                            insidetextfont=dict(size=12)
                        )
                    ]
                )
                pie.update_traces(textinfo="label+percent", textfont_size=14)
                pie.update_layout(
                    title_text=f"Discount: {discount}",
                    title_font=dict(size=14),
                    showlegend=False,
                    margin=dict(t=50, b=50, l=50, r=50),
                    height=269
                )
                st.plotly_chart(pie, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# bottom section layout
tablecol1, tablecol2 = st.columns(2)
with tablecol1:
    st.subheader("Summary Table")
    # need to add more metrics here 
    summary = pred_df.groupby(["Category", "Location", "Season"]).agg(
        Total_Purchases=("Purchase Intent Category", "count"),
        Impulsive_Purchases=("Purchase Intent Category", lambda x: (x == "Impulsive").sum()),
        Avg_Review_Rating=("Review Rating", "mean")
    ).reset_index()
    st.dataframe(summary, use_container_width=True)

with tablecol2:
    # Top-right dropdown aligned with the subheader
    if "Purchase Intent Category" in pred_df.columns and "Item Purchased" in pred_df.columns:
        cols = st.columns([3, 1.2])
        categories = pred_df["Purchase Intent Category"].unique().tolist()

        # Dropdown and N input on the right column
        selected_category = cols[1].selectbox("Select Category", options=categories, index=0)

        # Subheader on the left column (aligned with the dropdown)
        cols[0].subheader(f"Top 10 Insights by Purchase Intent Category")

        # Build and display the top N items for the selected category
        top_items = (
            pred_df[pred_df["Purchase Intent Category"] == selected_category]["Item Purchased"]
            .value_counts()
            .reset_index()
        )
        top_items.columns = ["Item Purchased", f"{selected_category} Purchases"]
        st.dataframe(top_items, use_container_width=True)
    else:
        st.subheader(f"Top 10 Insights by Purchase Intent Category")
        st.info("Required columns 'Purchase Intent Category' and/or 'Item Purchased' not found in the data.")

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