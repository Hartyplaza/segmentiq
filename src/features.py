"""
RFM feature engineering for customer segmentation.
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.config import SNAPSHOT_DATE_OFFSET


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM features for each customer.

    Recency:  days since last purchase (lower = better)
    Frequency: number of unique invoices
    Monetary:  total revenue generated
    """
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=SNAPSHOT_DATE_OFFSET)
    logger.info(f"Snapshot date: {snapshot_date.date()}")

    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('Invoice',     'nunique'),
        Monetary  = ('Revenue',     'sum'),
    ).reset_index()

    logger.info(f"RFM shape: {rfm.shape}")
    logger.info(f"\n{rfm.describe().round(2)}")

    return rfm


def add_rfm_scores(rfm: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    """
    Add RFM quintile scores (1-4) for each dimension.
    Higher score = better customer on that dimension.
    Note: Recency is reversed (lower days = higher score).
    """
    rfm = rfm.copy()

    rfm['R_Score'] = pd.qcut(rfm['Recency'],   q=bins,
                              labels=range(bins, 0, -1),
                              duplicates='drop').astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=bins,
                              labels=range(1, bins+1),
                              duplicates='drop').astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'],  q=bins,
                              labels=range(1, bins+1),
                              duplicates='drop').astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + \
                          rfm['F_Score'].astype(str) + \
                          rfm['M_Score'].astype(str)

    return rfm


def transform_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to reduce skewness before clustering.
    Monetary and Frequency are heavily right-skewed.
    """
    rfm = rfm.copy()
    rfm['log_Recency']   = np.log1p(rfm['Recency'])
    rfm['log_Frequency'] = np.log1p(rfm['Frequency'])
    rfm['log_Monetary']  = np.log1p(rfm['Monetary'])
    return rfm
