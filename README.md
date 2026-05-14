---
title: SegmentIQ
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
python_version: "3.11"
---

# SegmentIQ — Customer Segmentation

An end-to-end unsupervised machine learning system for customer segmentation using RFM analysis, K-Means clustering, and an interactive business intelligence dashboard.

**Live Demo:** https://huggingface.co/spaces/Demerchanthart/segmentiq

---

## Problem Statement

Not all customers are equal. A retail business with 5,878 customers cannot treat them all the same way — sending identical marketing to Champions and Lapsed Customers wastes budget and misses revenue opportunities.

**SegmentIQ** solves this by:
- Transforming 1 million+ raw transactions into meaningful per-customer behaviour profiles
- Discovering natural customer groups using unsupervised machine learning
- Quantifying the revenue opportunity from each segment
- Providing actionable retention and reactivation strategies

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) |
| Raw transactions | 1,067,371 |
| Clean customers | 5,878 |
| Total revenue | £17,374,804 |
| Period | December 2009 — December 2011 |
| Country | UK-based online retailer |

### Data Cleaning Steps
- Removed missing `CustomerID` (~25% of rows)
- Removed cancelled transactions (Invoice starting with C)
- Removed negative quantities and zero prices
- Removed duplicate rows

---

## Results

### Cluster Performance

| Metric | Value |
|---|---|
| Algorithm | K-Means |
| Optimal K | 2 |
| Silhouette Score | 0.4386 |
| Davies-Bouldin Score | 0.8727 |

### Segment Profiles

| Segment | Customers | Recency | Frequency | Avg Monetary |
|---|---|---|---|---|
| **Champions** | 2,320 (39.5%) | 51 days | 12.7 orders | £6,547 |
| **Lapsed Customers** | 3,558 (60.5%) | 300 days | 2.1 orders | £614 |

### Key Business Insights

- Champions spend **10.7x more** than Lapsed Customers (£6,547 vs £614)
- Champions buy **6x more often** (12.7 vs 2.1 orders)
- Champions purchased **6x more recently** (51 vs 300 days)
- Converting 10% of Lapsed Customers to Champions adds **£552,000** in incremental revenue

---

## RFM Analysis

RFM (Recency, Frequency, Monetary) is the industry standard for customer segmentation in retail and e-commerce:

| Feature | Description | Calculation |
|---|---|---|
| **Recency** | How recently did the customer buy? | Days since last purchase |
| **Frequency** | How often do they buy? | Number of unique invoices |
| **Monetary** | How much do they spend? | Total revenue generated |

### Transformations Applied
- Log transform (`log1p`) to reduce right skewness on all three features
- StandardScaler to normalise for clustering
- Quartile scoring (1-4) per dimension for RFM score calculation

---

## Project Structure

```
segmentiq/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/                        # online_retail_II.csv
│   └── processed/
│       ├── rfm_features.csv        # RFM per customer
│       └── clustered_customers.csv # With segment labels
├── notebooks/
│   ├── 01_eda.ipynb                # Data quality, trends, Pareto analysis
│   ├── 02_feature_engineering.ipynb# RFM computation, log transform, scoring
│   ├── 03_modeling.ipynb           # Elbow curve, silhouette, K-Means, DBSCAN
│   └── 04_cluster_analysis.ipynb  # Snake plot, revenue analysis, recommendations
├── src/
│   ├── config.py                   # Paths, segment names, colours
│   ├── preprocess.py               # Data cleaning pipeline
│   ├── features.py                 # RFM feature engineering
│   └── cluster.py                  # Clustering logic
├── app/
│   └── streamlit_app.py            # 6-page dashboard
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── cluster_stats.json
│   └── feature_cols.json
├── Dockerfile                      # Hugging Face Spaces deployment
├── runtime.txt                     # Python 3.11
└── requirements.txt
```

---

## Notebooks

### 01 — EDA
- Dataset overview across 1 million+ transactions
- Data quality assessment — 5 types of issues identified and quantified
- Monthly revenue and customer trends
- Top products by revenue and quantity
- Pareto analysis — top 20% of customers generate majority of revenue
- Geographic revenue distribution

### 02 — Feature Engineering
- RFM computation per customer from raw transactions
- Distribution analysis — raw vs log-transformed
- Skewness comparison before and after log transform
- RFM quartile scoring (1-4 per dimension)
- Manual segment labelling from score combinations
- StandardScaler normalisation

### 03 — Modeling
- Elbow curve testing K from 2 to 10
- Silhouette analysis for K=3, 4, 5 with blade plots
- Final K-Means model training
- DBSCAN comparison — density-based alternative
- Cluster profiling on original RFM values
- PCA 2D visualisation
- Business segment mapping

### 04 — Cluster Analysis
- Segment overview with revenue concentration
- Snake plot — normalised RFM comparison across segments
- Revenue concentration analysis (customer share vs revenue share)
- Purchase behaviour and order value by segment
- Geographic distribution per segment
- Business recommendations with revenue opportunity calculation

---

## Dashboard Pages

| Page | Description |
|---|---|
| Overview | Key metrics, donut chart, revenue bar chart, segment profile table |
| Segment Analysis | Per-segment RFM distributions and scatter plots |
| Snake Plot | Normalised RFM comparison with interpretation guide |
| Revenue Intelligence | Revenue concentration charts and reactivation calculator |
| Customer Explorer | Look up individual customers by ID, filterable browse table |
| Recommendations | Actionable strategies per segment with budget allocation chart |

---

## What Makes This Different From Supervised Projects

| Aspect | Supervised (ChurnGuard, FraudShield) | This Project |
|---|---|---|
| Learning type | Supervised | **Unsupervised** |
| Labels | Yes | **No labels** |
| Evaluation | ROC-AUC, PR-AUC | **Silhouette, Davies-Bouldin** |
| New algorithms | XGBoost, Isolation Forest | **K-Means, DBSCAN, PCA** |
| Output | Probability score | **Cluster assignment + strategy** |
| Data challenge | Class imbalance | **Feature engineering from transactions** |

---

## Quickstart

```bash
git clone https://github.com/Hartyplaza/segmentiq.git
cd segmentiq
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/raw/online_retail_II.csv
# Then run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy |
| ML | scikit-learn (K-Means, DBSCAN, PCA, StandardScaler) |
| Visualisation | matplotlib, seaborn, Plotly |
| Dashboard | Streamlit |
| Deployment | Hugging Face Spaces (Docker) |

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer

- LinkedIn: [linkedin.com/in/hart-ofigwe](https://www.linkedin.com/in/hart-ofigwe)
- GitHub: [github.com/Hartyplaza](https://github.com/Hartyplaza)
- Live Demo: [huggingface.co/spaces/Demerchanthart/segmentiq](https://huggingface.co/spaces/Demerchanthart/segmentiq)
