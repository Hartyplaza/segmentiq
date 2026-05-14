"""
Clustering logic for customer segmentation.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from loguru import logger

from src.config import (
    KMEANS_MODEL_FILE, SCALER_FILE, CLUSTER_STATS_FILE,
    N_CLUSTERS, MAX_CLUSTERS, RANDOM_STATE
)


def find_optimal_clusters(X_scaled: np.ndarray,
                           max_k: int = MAX_CLUSTERS) -> dict:
    """
    Test K from 2 to max_k and return inertia and silhouette scores.
    Used to build the elbow curve and silhouette analysis.
    """
    results = {'k': [], 'inertia': [], 'silhouette': []}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        results['k'].append(k)
        results['inertia'].append(km.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))
        logger.info(f"K={k} | Inertia={km.inertia_:.0f} | Silhouette={silhouette_score(X_scaled, labels):.4f}")
    return results


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = N_CLUSTERS):
    """Train final K-Means model."""
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    db     = davies_bouldin_score(X_scaled, labels)
    logger.info(f"K-Means trained | Clusters={n_clusters} | Silhouette={sil:.4f} | DB={db:.4f}")
    return km, labels, sil, db


def scale_features(rfm_transformed: pd.DataFrame,
                   feature_cols: list) -> tuple:
    """Scale RFM features for clustering."""
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_transformed[feature_cols])
    return scaler, X_scaled


def save_model(km, scaler, cluster_stats: dict):
    """Save trained model artifacts."""
    KMEANS_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(km,     KMEANS_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    with open(CLUSTER_STATS_FILE, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    logger.info("Model artifacts saved")


def load_model():
    """Load trained model artifacts."""
    km     = joblib.load(KMEANS_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    with open(CLUSTER_STATS_FILE) as f:
        stats = json.load(f)
    return km, scaler, stats


def assign_segment_labels(rfm: pd.DataFrame,
                           segment_map: dict) -> pd.DataFrame:
    """
    Assign human-readable segment names to cluster numbers
    based on RFM profile of each cluster.
    """
    rfm = rfm.copy()
    rfm['Segment'] = rfm['Cluster'].map(segment_map)
    return rfm
