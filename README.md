#  Streamlit_app link 
https://nigerian-customer-segmentation-hfjkwh6bk4n5hwlnuldtk4.streamlit.app/

# nigerian-customer-segmentation
Customer segmentation of Nigerian retail and e-commerce customers using K-Prototypes clustering.
#  Nigerian Retail & E-Commerce Customer Segmentation

##  Project Overview

This project applies unsupervised machine learning to segment customers within a Nigerian retail and e-commerce dataset based on their purchasing behaviour and customer characteristics.

The main objective is to identify groups of customers with similar characteristics and purchasing patterns, allowing businesses to better understand their customer base and develop more targeted marketing and customer-retention strategies.

Because the dataset contains both numerical and categorical variables, the K-Prototypes clustering algorithm was used to perform customer segmentation.

## Project Objectives

- Identify distinct customer segments.
- Analyze customer purchasing behaviour.
- Understand differences in customer spending patterns.
- Profile the characteristics of each customer cluster.
- Develop business recommendations based on the identified segments.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- K-Prototypes
- Jupyter Notebook

## Methodology

1. Data cleaning
2. Exploratory data analysis
3. Outlier analysis
4. Feature transformation
5. Feature scaling
6. K-Prototypes clustering
7. Cluster profiling
8. Business recommendations

## Project Structure

Nigerian_Retail_And_Ecommerce_Customer_Segmentation_Data/
│
├── Data/
│   └── original_nigerian_retail_and_ecommerce_customer_segmentation_data.csv
│
├── Images/
│   └── Charts and visualizations
│
├── Notebooks/
│   └── nigeria_retail_and_ecommerce_customer_segmentation.ipynb
│
├── Outputs/
│   └── Final clustering output
│
└── README.md

## Visualizations

### Numerical Distributions


![Numerical Distributions](Images/01_numerical_distributions.png)


Distribution plots of the key numeric features (Total Spend, Avg Order Value, Lifetime Value, Total Orders, Last Purchase Days Ago) before transformation, revealing right-skew that motivated log transformation during preprocessing.

### Outlier Detection


![Outlier - Total Spend](Images/02_outlier_total_spend_ngn.png)




![Outlier - Avg Order Value](Images/02_outlier_avg_order_value_ngn.png)




![Outlier - Lifetime Value](Images/02_outlier_lifetime_value_ngn.png)




![Outlier - Total Orders](Images/02_outlier_total_orders.png)




![Outlier - Last Purchase Days Ago](Images/02_outlier_last_purchase_days_ago.png)


Boxplots used to identify outliers across the five core numeric features on the 150,000-row dataset. Extreme values were reviewed rather than blindly dropped, since high spenders/lifetime value customers are business-relevant, not noise.

### Categorical Breakdown


![Preferred Category](Images/03_categorical_preferred_category.png)




![Purchase Frequency](Images/03_categorical_purchase_frequency.png)




![Seasonal Buyer](Images/03_categorical_seasonal_buyer.png)


Distribution of customers across preferred product category, purchase frequency, and seasonal buying behavior — the categorical features fed into K-Prototypes alongside the scaled numeric ones.

### Correlation Heatmap


![Correlation Heatmap](Images/04_correlation_heatmap.png)


Correlation matrix of numeric features, used to check for multicollinearity before clustering and to understand which spending behaviors move together.

### Elbow Method


![Elbow Method](Images/05_elbow_method.png)


Cost curve across k values for K-Prototypes. While the elbow suggested a reasonable range, k=4 was ultimately selected based on Adjusted Rand Index (0.4601) and cluster interpretability rather than the elbow alone.

### Cluster Visualization


![Cluster Visualization Heatmap](Images/06_cluster_visualization_heatmap.png)


Heatmap of average feature values per cluster, used to profile and label the four segments: **Budget/Occasional Shoppers**, **Frequent Mid-Spenders**, **Premium Customers**, and **Occasional Big-Ticket Buyers**.

### Final Cluster Distribution


![Final Cluster Distribution](Images/07_final_cluster_distribution_chart.png)


Distribution of customers across the four final clusters, showing relative segment sizes across the customer base.

## Results

Using K-Prototypes clustering on the 150,000-row Nigerian retail and e-commerce dataset, four distinct customer segments were identified from a mix of numeric (spend, order value, lifetime value, order count, recency) and categorical (preferred category, purchase frequency, seasonal buyer status) features.

- **Optimal k = 4**, selected using Adjusted Rand Index (ARI = 0.4601) and cluster interpretability, rather than relying on the elbow method alone.
- The four segments identified:
  | Cluster | Label | Key Characteristics |
  |---|---|---|
  | 0 | Budget/Occasional Shoppers | Low spend, infrequent orders, lower lifetime value |
  | 1 | Frequent Mid-Spenders | Moderate spend, high purchase frequency |
  | 2 | Premium Customers | High spend, high lifetime value, consistent purchasing |
  | 3 | Occasional Big-Ticket Buyers | Low frequency but high average order value |
- Cluster sizes were reasonably balanced, with no single segment dominating the customer base (see final cluster distribution chart above).

## Key Insights

- **Purchase frequency emerged as a hidden dimension**: The original dataset came with five predefined customer segment labels, but those labels didn't explicitly capture purchase frequency as a differentiator. Clustering on the raw features surfaced frequency as a major axis separating customers — meaning the existing segmentation scheme was missing a behaviorally important signal.
- **Spend and lifetime value don't always move together**: Occasional Big-Ticket Buyers show that infrequent customers can still be high-value, which a simple "frequent = valuable" heuristic would miss.
- **Premium Customers combine both frequency and spend**, making them the clearest candidates for loyalty or retention-focused campaigns, while Frequent Mid-Spenders represent volume-driven revenue rather than high-margin revenue.
- Mixed-type clustering (K-Prototypes) was necessary here — a pure K-Means approach would have required discarding the categorical behavioral features (preferred category, seasonal buyer), losing meaningful segmentation signal.

## Business Recommendations

- **Premium Customers**: Prioritize retention — loyalty programs, early access to new products/categories, personalized outreach. Losing this segment has the highest revenue impact per customer.
- **Frequent Mid-Spenders**: Focus on upsell and cross-sell strategies (bundle offers, category expansion) to shift them toward higher average order value without losing their purchase frequency.
- **Occasional Big-Ticket Buyers**: Target with timely, high-relevance promotions around their likely purchase windows (e.g., seasonal or event-based marketing) rather than high-frequency messaging, which risks being ignored.
- **Budget/Occasional Shoppers**: Use low-cost re-engagement tactics (discount codes, reminder emails) to increase order frequency before investing further marketing spend, since this segment currently returns the least value per touchpoint.
- **Segmentation strategy**: Recommend supplementing the business's existing five-label segment scheme with the frequency-based clusters identified here, since it captures a dimension the current scheme misses — this could improve targeting precision for campaigns tied to purchase cadence.

## Author

**Lawal Yinka**

Aspiring Data Scientist
