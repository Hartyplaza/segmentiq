"""
Project-wide configuration and paths.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
DATA_DIR           = BASE_DIR / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR         = BASE_DIR / "models"

# ── Data ───────────────────────────────────────────────────────────────────────
RAW_DATA_FILE      = RAW_DATA_DIR / "online_retail_II.xlsx"
RFM_FEATURES_FILE  = PROCESSED_DATA_DIR / "rfm_features.csv"
CLUSTERED_FILE     = PROCESSED_DATA_DIR / "clustered_customers.csv"

# ── Model ──────────────────────────────────────────────────────────────────────
KMEANS_MODEL_FILE  = MODELS_DIR / "kmeans_model.pkl"
SCALER_FILE        = MODELS_DIR / "scaler.pkl"
CLUSTER_STATS_FILE = MODELS_DIR / "cluster_stats.json"

# ── Clustering ─────────────────────────────────────────────────────────────────
RANDOM_STATE       = 42
N_CLUSTERS         = 4        # optimal found from elbow + silhouette
MAX_CLUSTERS       = 10       # range to test

# ── RFM ────────────────────────────────────────────────────────────────────────
SNAPSHOT_DATE_OFFSET = 1      # days after last transaction for recency calc

# ── Segment names ──────────────────────────────────────────────────────────────
SEGMENT_NAMES = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At-Risk Customers",
    3: "Lost Customers",
}

SEGMENT_COLORS = {
    "Champions":        "#059669",
    "Loyal Customers":  "#2563eb",
    "At-Risk Customers":"#d97706",
    "Lost Customers":   "#dc2626",
}
