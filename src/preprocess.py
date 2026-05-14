"""
Data cleaning and preprocessing for Online Retail II dataset.
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.config import RAW_DATA_FILE


def load_raw_data(filepath=RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw Excel data — supports both .xlsx sheets."""
    logger.info(f"Loading data from {filepath}")
    try:
        # Dataset has two sheets — Year 2009-2010 and 2010-2011
        df1 = pd.read_excel(filepath, sheet_name='Year 2009-2010', engine='openpyxl')
        df2 = pd.read_excel(filepath, sheet_name='Year 2010-2011', engine='openpyxl')
        df  = pd.concat([df1, df2], ignore_index=True)
    except Exception:
        df = pd.read_excel(filepath, engine='openpyxl')
    logger.info(f"Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the retail dataset:
    - Remove missing CustomerID
    - Remove cancelled transactions (InvoiceNo starting with C)
    - Remove negative quantities and prices
    - Remove duplicates
    - Parse dates
    """
    df = df.copy()

    logger.info(f"Starting shape: {df.shape}")

    # Standardise column names
    df.columns = df.columns.str.strip()
    if 'Customer ID' in df.columns:
        df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)

    # Remove missing CustomerID — cannot attribute to a customer
    before = len(df)
    df.dropna(subset=['CustomerID'], inplace=True)
    logger.info(f"Removed {before - len(df):,} rows with missing CustomerID")

    # Remove cancellations (InvoiceNo starts with C)
    before = len(df)
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    logger.info(f"Removed {before - len(df):,} cancelled transactions")

    # Remove negative quantities and prices
    before = len(df)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    logger.info(f"Removed {before - len(df):,} rows with negative qty/price")

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - len(df):,} duplicate rows")

    # Parse dates
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Add revenue column
    df['Revenue'] = df['Quantity'] * df['Price']

    # Convert CustomerID to integer
    df['CustomerID'] = df['CustomerID'].astype(int)

    logger.info(f"Final clean shape: {df.shape}")
    logger.info(f"Unique customers: {df['CustomerID'].nunique():,}")
    logger.info(f"Date range: {df['InvoiceDate'].min()} — {df['InvoiceDate'].max()}")

    return df
