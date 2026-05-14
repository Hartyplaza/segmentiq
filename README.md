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

An end-to-end unsupervised machine learning system for customer segmentation using RFM analysis, K-Means clustering, and interactive business intelligence dashboard.

**Live Demo:** TBD

---

## Project Overview

Customer segmentation identifies distinct groups of customers based on purchasing behaviour. This enables targeted marketing, personalised retention strategies, and data-driven business decisions.

This project uses RFM (Recency, Frequency, Monetary) analysis combined with K-Means and DBSCAN clustering to discover natural customer segments from 1 million+ retail transactions.

---

## Dataset

- **Source:** [Kaggle — Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- **Size:** 1,067,371 transactions
- **Period:** December 2009 — December 2011
- **Customers:** UK-based online retailer

---

## Project Structure

```
segmentiq/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/                   # online_retail_II.xlsx
│   └── processed/             # rfm_features.csv, clustered_customers.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_cluster_analysis.ipynb
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── features.py
│   └── cluster.py
├── api/
│   ├── main.py
│   └── schemas.py
├── app/
│   └── streamlit_app.py
├── models/
├── requirements.txt
└── README.md
```

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer
- LinkedIn: [linkedin.com/in/hart-ofigwe](https://www.linkedin.com/in/hart-ofigwe)
- GitHub: [github.com/Hartyplaza](https://github.com/Hartyplaza)
