"""
Nigerian Retail & E-Commerce Customer Segmentation Dashboard
Author: Lawal
Built on top of the DS-02 K-Prototypes clustering project.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nigerian Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD ARTIFACTS (model, encoders, labeled dataset)
# ---------------------------------------------------------
# NOTE: You need to save these from your notebook first. See bottom of
# this file / the accompanying instructions for exactly what to export.

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR,"model","kprototypes_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR,"data","labeled_customers.csv"))
    return df

# Cluster label mapping — update these to match your actual cluster numbers
CLUSTER_LABELS = {
    0: "Budget/Occasional Shoppers",
    1: "Frequent Mid-Spenders",
    2: "Premium Customers",
    3: "Occasional Big-Ticket Buyers"
}

CLUSTER_DESCRIPTIONS = {
    "Budget/Occasional Shoppers": "Low spend, infrequent orders, lower lifetime value. "
        "Best reached with low-cost re-engagement offers.",
    "Frequent Mid-Spenders": "Moderate spend, high purchase frequency. "
        "Good candidates for upsell and bundle offers.",
    "Premium Customers": "High spend, high lifetime value, consistent purchasing. "
        "Prioritize for retention and loyalty programs.",
    "Occasional Big-Ticket Buyers": "Low frequency but high average order value. "
        "Best reached with timely, high-relevance promotions."
}

# Categorical options — update to match your actual dataset categories
CATEGORY_OPTIONS = ["Electronics", "Fashion", "Groceries", "Home & Living", "Beauty"]
FREQUENCY_OPTIONS = ["Low", "Medium", "High"]
SEASONAL_OPTIONS = ["Yes", "No"]

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("🛍️ Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Cluster Explorer", "Try It: Predict a Segment"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this project**\n\n"
    "K-Prototypes clustering on 150,000 Nigerian retail/e-commerce customer "
    "records, segmenting customers into 4 behavioral groups using mixed "
    "numeric and categorical features."
)
st.sidebar.markdown("[View on GitHub](https://github.com/lawalyinka-ds/nigerian-customer-segmentation)")

# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
if page == "Overview":
    st.title("Nigerian Retail & E-Commerce Customer Segmentation")
    st.markdown(
        "Customer segmentation using **K-Prototypes clustering** on a "
        "150,000-row Nigerian retail dataset, combining numeric spending "
        "behavior with categorical shopping patterns."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "150,000")
    col2.metric("Clusters Found", "4")
    col3.metric("ARI Score", "0.4601")
    col4.metric("Method", "K-Prototypes")

    st.markdown("---")
    st.subheader("The Four Segments")

    for label, desc in CLUSTER_DESCRIPTIONS.items():
        with st.expander(f"📌 {label}"):
            st.write(desc)

    st.markdown("---")
    st.subheader("Key Insight")
    st.info(
        "Clustering surfaced **purchase frequency** as a major differentiator "
        "between customers — a dimension the original five-label segment "
        "scheme did not explicitly capture."
    )

# ---------------------------------------------------------
# PAGE 2: CLUSTER EXPLORER
# ---------------------------------------------------------
elif page == "Cluster Explorer":
    st.title("Cluster Explorer")
    st.markdown("Explore how the four customer segments differ across key metrics.")

    try:
        df = load_data()

        selected_clusters = st.multiselect(
            "Filter by segment",
            options=list(CLUSTER_LABELS.values()),
            default=list(CLUSTER_LABELS.values())
        )

        filtered_df = df[df["segment_label"].isin(selected_clusters)]

        st.markdown(f"**Showing {len(filtered_df):,} customers**")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Average Spend by Segment")
            fig, ax = plt.subplots()
            sns.barplot(data=filtered_df, x="segment_label", y="total_spend_ngn", ax=ax)
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

        with col2:
            st.subheader("Segment Sizes")
            fig, ax = plt.subplots()
            filtered_df["segment_label"].value_counts().plot(kind="bar", ax=ax)
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

        st.subheader("Sample Records")
        st.dataframe(filtered_df.head(20))

    except FileNotFoundError:
        st.warning(
            "⚠️ Data file not found. Add `data/labeled_customers.csv` "
            "(your dataset with a `segment_label` column) to enable this page."
        )

# ---------------------------------------------------------
# PAGE 3: PREDICT A SEGMENT
# ---------------------------------------------------------
elif page == "Try It: Predict a Segment":
    st.title("Predict a Customer's Segment")
    st.markdown("Enter customer details below to see which segment they'd fall into.")

    col1, col2 = st.columns(2)

    with col1:
        total_spend = st.number_input("Total Spend (NGN)", min_value=0, value=50000, step=1000)
        avg_order_value = st.number_input("Average Order Value (NGN)", min_value=0, value=5000, step=500)
        total_orders = st.number_input("Total Orders", min_value=0, value=10, step=1)
        lifetime_value = st.number_input("Lifetime Value (NGN)", min_value = 0, value = 200000, step = 1000)

    with col2:
        last_purchase_days = st.number_input("Days Since Last Purchase", min_value=0, value=30, step=1)
        preferred_category = st.selectbox("Preferred Category", CATEGORY_OPTIONS)
        purchase_frequency = st.selectbox("Purchase Frequency", FREQUENCY_OPTIONS)
        seasonal_buyer = st.selectbox("Seasonal Buyer", SEASONAL_OPTIONS)

    if st.button("Predict Segment", type="primary"):
        try:
            model = load_model()

            # Build input row — order and preprocessing must match training pipeline
            input_data = np.array([[
                np.log1p(avg_order_value),
                total_orders,
                np.log1p(total_spend),
                last_purchase_days,
                lifetime_value,
                purchase_frequency,
                preferred_category,
                seasonal_buyer
            ]], dtype=object)

            categorical_indices = [5, 6, 7]  # positions of categorical columns
            prediction = model.predict(input_data, categorical=categorical_indices)
            cluster_num = int(prediction[0])
            label = CLUSTER_LABELS.get(cluster_num, "Unknown")

            st.success(f"### Predicted Segment: **{label}**")
            st.write(CLUSTER_DESCRIPTIONS.get(label, ""))

        except FileNotFoundError:
            st.warning(
                "⚠️ Model file not found. Add `model/kprototypes_model.pkl` "
                "(your saved trained model) to enable predictions."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built by Lawal · Data Science Portfolio · lawalyinka-ds")
