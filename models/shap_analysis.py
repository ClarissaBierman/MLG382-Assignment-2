"""
shap_analysis.py  — Model Interpretability with SHAP Values
═════════════════════════════════════════════════════════════
OWNER:  Role 4 — Unsupervised Learning & Interpretability Specialist
STATUS: NEW FILE — implement everything below.

SHAP (SHapley Additive exPlanations) tells us *why* the model made a
specific prediction. For a stock price prediction tool, this answers:
  "Why does the model predict AAPL will rise tomorrow?"
  → "Because RSI is oversold (Lag1 pushed it up by +$2.10), and VIX
     dropped (reducing fear, adding +$1.30)."

This output is shown in the "SHAP Insights" tab of the Dash dashboard.
The dashboard stub is in app.py — build_shap_tab() — waiting for you.

Install shap:  pip install shap
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
    print("WARNING: shap not installed. Run: pip install shap")


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): implement compute_shap_values
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        model_name: str = "") -> dict:
    """
    Compute SHAP values for a fitted model on the test set.

    Use the appropriate explainer:
      - Tree-based (RF, XGBoost, GBM) → shap.TreeExplainer  (fast, exact)
      - Linear models                 → shap.LinearExplainer
      - Fallback                      → shap.KernelExplainer (slow, sample 100 rows)

    For Pipeline models, extract the inner estimator via
    model.named_steps["model"] before passing to the explainer.

    Returns a dict with:
      "shap_values"    : np.ndarray shape (n_test_samples, n_features)
      "expected_value" : float — base value (mean prediction)
      "X_test"         : pd.DataFrame — test features (for waterfall/beeswarm)
      "model_name"     : str
      "feature_names"  : list[str]
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap package not installed. Run: pip install shap")

    # TODO: implement
    # Hint: For Pipeline, do:
    #   inner = model.named_steps.get("model", model)
    #   Then branch on type(inner).__name__ to choose explainer.
    raise NotImplementedError(
        "compute_shap_values not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): implement get_global_importance
# ─────────────────────────────────────────────────────────────────────────────

def get_global_importance(shap_values: np.ndarray,
                           feature_names: list,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Compute global SHAP feature importance = mean(|SHAP|) per feature.

    Returns a DataFrame sorted descending:
        columns: ["feature", "mean_abs_shap"]
        rows:    top_n features
    """
    # TODO: implement
    raise NotImplementedError(
        "get_global_importance not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): implement get_local_explanation
# ─────────────────────────────────────────────────────────────────────────────

def get_local_explanation(shap_values: np.ndarray,
                           X_test: pd.DataFrame,
                           sample_idx: int,
                           top_n: int = 10) -> pd.DataFrame:
    """
    Explain a single prediction (waterfall chart data).

    For sample at `sample_idx` in X_test, return the top_n features
    with their SHAP contribution and actual feature value.

    Returns a DataFrame:
        columns: ["feature", "shap_value", "feature_value", "direction"]
        direction: "positive" or "negative" (sign of shap_value)
        sorted by |shap_value| descending
    """
    # TODO: implement
    raise NotImplementedError(
        "get_local_explanation not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TODO (Interpretability Specialist): interpret cluster drivers
# ─────────────────────────────────────────────────────────────────────────────

def get_cluster_shap_summary(model, X_by_cluster: dict,
                              X_train: pd.DataFrame) -> dict:
    """
    Compute mean SHAP values separately for each market regime cluster.

    This reveals: "In bearish regimes, which features drive down predictions?"
    vs "In bullish regimes, which features push predictions up?"

    Parameters
    ----------
    model       : fitted estimator (best model from StockModelTrainer)
    X_by_cluster: dict mapping cluster_id → pd.DataFrame of test rows
                  in that cluster.  Build this from clustering labels.
    X_train     : full training set (for TreeExplainer background data)

    Returns
    -------
    dict mapping cluster_id → pd.DataFrame with columns
        ["feature", "mean_shap", "mean_abs_shap"]
    sorted by mean_abs_shap descending.
    """
    # TODO: implement
    raise NotImplementedError(
        "get_cluster_shap_summary not implemented — Interpretability Specialist task"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper used by app.py  (do not rename or change signature)
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(trainer,          # StockModelTrainer instance
                       model_name: str = None) -> dict:
    """
    Full SHAP pipeline for the best (or specified) trained model.
    Called by the Dash callback in app.py.

    Returns a dict with keys:
      "shap_values"      : np.ndarray
      "expected_value"   : float
      "X_test"           : pd.DataFrame
      "feature_names"    : list[str]
      "global_importance": pd.DataFrame  — from get_global_importance()
      "model_name"       : str
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
