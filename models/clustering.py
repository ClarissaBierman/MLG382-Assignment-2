"""
clustering.py  — CRISP-DM Phase 4: Unsupervised Segmentation
═════════════════════════════════════════════════════════════
OWNER:  Role 4 — Unsupervised Learning & Interpretability Specialist
STATUS: NEW FILE — implement everything below.

Your task is to segment market conditions into distinct "regimes" using
K-Means clustering. The clusters will be shown in the Dash dashboard's
"Market Regimes" tab (see app.py — build_regimes_tab is a stub waiting
for your output).

Cluster meaning (target interpretation, 3 clusters):
  Cluster 0 — Low-volatility / bullish trend
  Cluster 1 — High-volatility / uncertain
  Cluster 2 — Bearish / drawdown regime

These labels give traders context: "the model predicts a rise, but we
are currently in a high-volatility regime — treat with caution."
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — do not change (app.py references these)
# ─────────────────────────────────────────────────────────────────────────────

N_CLUSTERS   = 3
RANDOM_STATE = 42

# Human-readable labels and colours for the dashboard
CLUSTER_LABELS = {
    0: "Bullish / Low-Vol",
    1: "High-Vol / Uncertain",
    2: "Bearish / Drawdown",
}
CLUSTER_COLORS = {
    0: "#00e5a0",   # green
    1: "#ffc440",   # amber
    2: "#ff4d6d",   # red
}


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): implement build_cluster_features
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and engineer features for K-Means clustering.
    The goal is to characterise each trading day by its market *regime*.

    Suggested features (you may add more):
      - Daily_Return  (Close.pct_change())
      - HV_5          (5-day rolling std of log returns * sqrt(252))
      - HV_21         (21-day rolling std)
      - Volume_Ratio  (Volume / 20-day SMA of Volume)
      - RSI_14        (if present in df)
      - BB_Pct        (position within Bollinger Bands, 0–1)
      - SP500_Return  (broad market context, if present in df)
      - VIX           (fear index, if present in df)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicator DataFrame produced by add_technical_indicators().
        May already contain RSI_14, BB_Pct, Volume_Ratio etc.

    Returns
    -------
    pd.DataFrame
        Rows aligned with df.index (NaN rows dropped), columns = cluster features.
    """
    # TODO: implement
    # Hint: compute any missing features from df["Close"], df["Volume"] etc.
    #       Drop rows with NaN after computing rolling windows.
    raise NotImplementedError(
        "build_cluster_features not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): implement fit_kmeans
# ─────────────────────────────────────────────────────────────────────────────

def fit_kmeans(features: pd.DataFrame) -> tuple:
    """
    Fit K-Means with N_CLUSTERS on the provided feature matrix.

    Steps:
      1. Scale features with StandardScaler (fit on features).
      2. Fit KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10).
      3. Compute silhouette_score on the scaled data.
      4. Compute cluster-centre summary (inverse-transform centres for interpretability).

    Returns
    -------
    labels      : np.ndarray  — cluster label per row (0, 1, or 2)
    scaler      : fitted StandardScaler
    kmeans      : fitted KMeans object
    silhouette  : float  — silhouette score (printed in app as model quality)
    centres_df  : pd.DataFrame  — cluster centres in original feature scale,
                                   rows = cluster ids, columns = feature names
    """
    # TODO: implement
    raise NotImplementedError(
        "fit_kmeans not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): implement get_cluster_statistics
# ─────────────────────────────────────────────────────────────────────────────

def get_cluster_statistics(df_features: pd.DataFrame,
                            labels: np.ndarray) -> pd.DataFrame:
    """
    Summarise each cluster with descriptive statistics.

    Returns a DataFrame with one row per cluster containing:
      - Count         : number of trading days in cluster
      - % of Days     : proportion of all days
      - Mean Return   : average daily return in that regime
      - Mean HV_21    : average historical volatility
      - Mean RSI      : average RSI_14 (if present)
      - Label         : human-readable name from CLUSTER_LABELS
    """
    # TODO: implement
    raise NotImplementedError(
        "get_cluster_statistics not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): implement elbow_analysis
# ─────────────────────────────────────────────────────────────────────────────

def elbow_analysis(features: pd.DataFrame,
                   k_range: range = range(2, 9)) -> pd.DataFrame:
    """
    Compute inertia and silhouette score for each k in k_range.
    Used to justify the choice of k=3 in the Technical Report.

    Returns a DataFrame with columns:
        k, inertia, silhouette_score
    """
    # TODO: implement
    raise NotImplementedError(
        "elbow_analysis not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper used by app.py  (do not rename or change signature)
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(df_with_indicators: pd.DataFrame) -> dict:
    """
    Full pipeline: features → fit → stats.
    Called by the Dash callback in app.py.

    Returns a dict with keys:
      "labels"       : np.ndarray aligned with df_with_indicators.dropna() index
      "stats"        : pd.DataFrame from get_cluster_statistics()
      "silhouette"   : float
      "centres_df"   : pd.DataFrame
      "feature_df"   : pd.DataFrame (features used)
      "elbow_df"     : pd.DataFrame from elbow_analysis()
      "index"        : DatetimeIndex — dates corresponding to labels
    """
    feat_df    = build_cluster_features(df_with_indicators)
    labels, scaler, kmeans, silhouette, centres_df = fit_kmeans(feat_df)
    stats      = get_cluster_statistics(feat_df, labels)
    elbow_df   = elbow_analysis(feat_df)

    return {
        "labels":      labels,
        "stats":       stats,
        "silhouette":  silhouette,
        "centres_df":  centres_df,
        "feature_df":  feat_df,
        "elbow_df":    elbow_df,
        "index":       feat_df.index,
    }
