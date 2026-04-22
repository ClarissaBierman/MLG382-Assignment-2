"""
shap_analysis.py  — Model Interpretability with SHAP Values
═════════════════════════════════════════════════════════════
OWNER:  Role 4 — Unsupervised Learning & Interpretability Specialist
STATUS: COMPLETE

Computes SHAP values to explain why the model makes each prediction.

Install:  pip install shap
Then add  shap  to requirements.txt
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed. Run:  pip install shap")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Compute SHAP values
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        model_name: str = "") -> dict:
    X_test = X_test.iloc[:100]
    """
    Compute SHAP values for a fitted model on the test set.

    Explainer selection:
      - RandomForest, XGBoost, GradientBoosting → TreeExplainer  (fast, exact)
      - LinearRegression, Ridge (Pipeline)       → LinearExplainer
      - Everything else                          → KernelExplainer (slow, 100 rows)

    Parameters
    ----------
    model      : fitted sklearn estimator or Pipeline
    X_train    : training features (background data for explainer)
    X_test     : test features to explain
    model_name : string label (for display)

    Returns
    -------
    dict with keys:
        shap_values, expected_value, X_test, model_name, feature_names
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap not installed. Run:  pip install shap")

    feature_names = list(X_test.columns)

    # Unwrap Pipeline to get the inner estimator
    inner = model
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model", model)

    model_type = type(inner).__name__

    # Choose the right explainer
    if model_type in ("RandomForestRegressor",
                      "XGBRegressor",
                      "GradientBoostingRegressor",
                      "DecisionTreeRegressor"):
        explainer   = shap.TreeExplainer(inner)
        shap_values = explainer.shap_values(X_test)
        expected    = float(explainer.expected_value
                            if np.isscalar(explainer.expected_value)
                            else explainer.expected_value[0])

    elif model_type in ("LinearRegression", "Ridge", "Lasso", "ElasticNet"):
        # For Pipelines, transform X_train/X_test through the scaler first
        if hasattr(model, "named_steps") and "scaler" in model.named_steps:
            sc = model.named_steps["scaler"]
            X_tr_sc = sc.transform(X_train)
            X_te_sc = sc.transform(X_test)
        else:
            X_tr_sc = X_train.values
            X_te_sc = X_test.values

        explainer   = shap.LinearExplainer(inner, X_tr_sc)
        shap_values = explainer.shap_values(X_te_sc)
        expected    = float(explainer.expected_value)

    else:
        # Kernel explainer — slow, use a 100-row sample as background
        background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
        explainer  = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test.iloc[:200])   # limit to 200 rows
        expected    = float(explainer.expected_value)

    # Ensure 2-D (some explainers return 1-D for single output)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    return {
        "shap_values":    shap_values,
        "expected_value": expected,
        "X_test":         X_test,
        "model_name":     model_name,
        "feature_names":  feature_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Global feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_global_importance(shap_values: np.ndarray,
                           feature_names: list,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Global importance = mean(|SHAP|) per feature, sorted descending.

    Returns
    -------
    pd.DataFrame with columns: feature, mean_abs_shap
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Local (single-prediction) explanation
# ─────────────────────────────────────────────────────────────────────────────

def get_local_explanation(shap_values: np.ndarray,
                           X_test: pd.DataFrame,
                           sample_idx: int,
                           top_n: int = 10) -> pd.DataFrame:
    """
    Waterfall chart data for one prediction.

    Returns
    -------
    pd.DataFrame with columns:
        feature, shap_value, feature_value, direction
    Sorted by |shap_value| descending.
    """
    row_shap  = shap_values[sample_idx]
    row_feats = X_test.iloc[sample_idx]
    feature_names = list(X_test.columns)

    df = pd.DataFrame({
        "feature":       feature_names,
        "shap_value":    row_shap,
        "feature_value": row_feats.values,
    })
    df["abs_shap"]  = df["shap_value"].abs()
    df["direction"] = df["shap_value"].apply(lambda v: "positive" if v >= 0 else "negative")
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df[["feature", "shap_value", "feature_value", "direction"]]


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Cluster-level SHAP summary
# ─────────────────────────────────────────────────────────────────────────────

def get_cluster_shap_summary(model, X_by_cluster: dict,
                              X_train: pd.DataFrame) -> dict:
    """
    Mean SHAP values per market-regime cluster.

    Parameters
    ----------
    model        : fitted estimator (best model from StockModelTrainer)
    X_by_cluster : dict  {cluster_id: pd.DataFrame of test rows in that cluster}
    X_train      : full training DataFrame (background for TreeExplainer)

    Returns
    -------
    dict mapping cluster_id → pd.DataFrame with columns:
        feature, mean_shap, mean_abs_shap
    Sorted by mean_abs_shap descending.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap not installed. Run:  pip install shap")

    inner = model
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model", model)

    # Build one explainer for all clusters
    model_type = type(inner).__name__
    if model_type in ("RandomForestRegressor", "XGBRegressor",
                      "GradientBoostingRegressor"):
        explainer = shap.TreeExplainer(inner)
    else:
        background = shap.sample(X_train, min(100, len(X_train)), random_state=42)
        explainer  = shap.KernelExplainer(model.predict, background)

    summaries = {}
    for cid, X_c in X_by_cluster.items():
        if len(X_c) == 0:
            continue
        sv = explainer.shap_values(X_c)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        df = pd.DataFrame({
            "feature":       list(X_c.columns),
            "mean_shap":     sv.mean(axis=0),
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        summaries[cid] = df

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper  (called by app.py — do not rename)
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(trainer, model_name: str = None) -> dict:
    """
    Full SHAP pipeline for the best (or named) trained model.

    Parameters
    ----------
    trainer    : fitted StockModelTrainer instance
    model_name : override which model to explain (default = trainer.best_model_name)

    Returns
    -------
    dict with keys:
        shap_values, expected_value, X_test, feature_names,
        global_importance, model_name
    """
    if model_name is None:
        model_name = trainer.best_model_name

    model   = trainer.trained_models[model_name]
    X_train = trainer.X_train
    X_test  = trainer.X_test

    result = compute_shap_values(model, X_train, X_test, model_name)
    result["global_importance"] = get_global_importance(
        result["shap_values"], result["feature_names"]
    )
    return result
