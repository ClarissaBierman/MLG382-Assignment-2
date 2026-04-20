"""
clustering.py  — CRISP-DM Phase 4: Unsupervised Segmentation
═════════════════════════════════════════════════════════════
Segments market conditions into 3 "regimes" using K-Means clustering.
Results feed the "Market Regimes" tab in the Dash dashboard.

  Cluster 0 — Bullish / Low-Volatility
  Cluster 1 — High-Volatility / Uncertain
  Cluster 2 — Bearish / Drawdown
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (app.py references these — do not rename)
# ─────────────────────────────────────────────────────────────────────────────

N_CLUSTERS   = 3
RANDOM_STATE = 42

CLUSTER_LABELS = {
    0: "Bullish / Low-Vol",
    1: "High-Vol / Uncertain",
    2: "Bearish / Drawdown",
}
CLUSTER_COLORS = {
    0: "#00e5a0",
    1: "#ffc440",
    2: "#ff4d6d",
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Feature engineering for clustering
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix that characterises each trading day's market regime.

    Parameters
    ----------
    df : OHLCV + indicator DataFrame from add_technical_indicators().

    Returns
    -------
    pd.DataFrame with NaN rows dropped, ready for StandardScaler + KMeans.
    """
    feat = pd.DataFrame(index=df.index)

    # Daily return
    feat["Daily_Return"] = df["Close"].pct_change() * 100

    # Short-term & medium-term historical volatility
    log_r = np.log(df["Close"] / df["Close"].shift(1))
    feat["HV_5"]  = log_r.rolling(5,  min_periods=5).std()  * np.sqrt(252) * 100
    feat["HV_21"] = log_r.rolling(21, min_periods=21).std() * np.sqrt(252) * 100

    # Volume ratio (use pre-computed column if available)
    if "Volume_Ratio" in df.columns:
        feat["Volume_Ratio"] = df["Volume_Ratio"]
    else:
        v_sma = df["Volume"].rolling(20, min_periods=1).mean()
        feat["Volume_Ratio"] = df["Volume"] / (v_sma + 1e-9)

    # Momentum — RSI
    if "RSI_14" in df.columns:
        feat["RSI_14"] = df["RSI_14"]
    else:
        # Compute inline if not present
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs    = gain / (loss + 1e-9)
        feat["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger %B position
    if "BB_Pct" in df.columns:
        feat["BB_Pct"] = df["BB_Pct"]
    else:
        sma20 = df["Close"].rolling(20, min_periods=1).mean()
        std20 = df["Close"].rolling(20, min_periods=1).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        feat["BB_Pct"] = (df["Close"] - lower) / (upper - lower + 1e-9)

    # Market sentiment — VIX (if available)
    if "VIX" in df.columns:
        feat["VIX"] = df["VIX"]

    # Broad market context — S&P 500 return (if available)
    if "SP500_Return" in df.columns:
        feat["SP500_Return"] = df["SP500_Return"] * 100

    feat.dropna(inplace=True)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — K-Means fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_kmeans(features: pd.DataFrame) -> tuple:
    """
    Scale → KMeans(k=3) → silhouette → cluster centres.

    Returns
    -------
    labels      : np.ndarray — cluster label per row (after re-ordering by mean return)
    scaler      : fitted StandardScaler
    kmeans      : fitted KMeans object
    silhouette  : float
    centres_df  : pd.DataFrame — centres in original feature scale
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(features.values)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    raw_labels = km.fit_predict(X_sc)

    # Re-order so cluster 0 = best return, cluster 2 = worst return
    ret_col = features.columns.get_loc("Daily_Return") if "Daily_Return" in features.columns else 0
    mean_ret = {c: features.values[raw_labels == c, ret_col].mean() for c in range(N_CLUSTERS)}
    order = sorted(mean_ret, key=mean_ret.get, reverse=True)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in raw_labels])

    # Silhouette score
    try:
        sil = float(silhouette_score(X_sc, labels))
    except Exception:
        sil = float("nan")

    # Cluster centres in original scale
    centres_orig = scaler.inverse_transform(km.cluster_centers_)
    # Re-order rows to match new label mapping
    reorder_idx  = [order.index(c) for c in range(N_CLUSTERS)]
    centres_df   = pd.DataFrame(
        centres_orig[reorder_idx],
        columns=features.columns,
        index=[CLUSTER_LABELS[i] for i in range(N_CLUSTERS)],
    )

    return labels, scaler, km, sil, centres_df


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Cluster summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def get_cluster_statistics(df_features: pd.DataFrame,
                            labels: np.ndarray) -> pd.DataFrame:
    """
    Returns one summary row per cluster.
    """
    rows = []
    n_total = len(labels)

    for cid in range(N_CLUSTERS):
        mask = labels == cid
        subset = df_features[mask]

        row = {
            "Cluster":       cid,
            "Label":         CLUSTER_LABELS[cid],
            "Count":         int(mask.sum()),
            "% of Days":     round(mask.sum() / n_total * 100, 1),
            "Mean Return":   round(float(subset["Daily_Return"].mean()), 4)
                             if "Daily_Return" in subset.columns else None,
            "Mean HV_21":    round(float(subset["HV_21"].mean()), 4)
                             if "HV_21" in subset.columns else None,
            "Mean RSI":      round(float(subset["RSI_14"].mean()), 2)
                             if "RSI_14" in subset.columns else None,
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("Cluster")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Elbow analysis (justify k=3)
# ─────────────────────────────────────────────────────────────────────────────

def elbow_analysis(features: pd.DataFrame,
                   k_range: range = range(2, 9)) -> pd.DataFrame:
    """
    Compute inertia and silhouette score for k = 2 … 8.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(features.values)

    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_sc)
        try:
            sil = float(silhouette_score(X_sc, km.labels_))
        except Exception:
            sil = float("nan")
        records.append({
            "k":               k,
            "inertia":         float(km.inertia_),
            "silhouette_score": sil,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper  (called by app.py — do not rename)
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(df_with_indicators: pd.DataFrame) -> dict:
    """
    Full pipeline: build features → fit KMeans → stats → elbow.

    Returns
    -------
    dict with keys:
        labels, stats, silhouette, centres_df, feature_df, elbow_df, index
    """
    feat_df = build_cluster_features(df_with_indicators)
    labels, scaler, kmeans, sil, centres_df = fit_kmeans(feat_df)
    stats    = get_cluster_statistics(feat_df, labels)
    elbow_df = elbow_analysis(feat_df)

    return {
        "labels":      labels,
        "stats":       stats,
        "silhouette":  sil,
        "centres_df":  centres_df,
        "feature_df":  feat_df,
        "elbow_df":    elbow_df,
        "index":       feat_df.index,
    }
