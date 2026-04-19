"""
ml_models.py  — CRISP-DM Phase 4 (Modelling) & Phase 5 (Evaluation)
═════════════════════════════════════════════════════════════════════
OWNER:  Role 3 — Lead ML Engineer
STATUS: SKELETON — implement all TODO sections below.

Your responsibilities:
  1. Define at least 5 regression models in MODEL_REGISTRY.
  2. Implement StockModelTrainer.train_all() — fit every model, store
     train/test predictions, compute evaluation metrics.
  3. Implement walk-forward cross-validation in train_all().
  4. Implement forecast_future() for iterative n-day forecasting.
  5. Extract and store feature importances for tree-based models.
  6. Return a clean metrics DataFrame via get_metrics_df().

The Dash app calls:
    trainer = StockModelTrainer()
    trainer.train_all(X, y)
    # then reads trainer.metrics, trainer.test_predictions, etc.
So your attribute names must match exactly.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# TODO (ML Engineer): Populate MODEL_REGISTRY with at least 5 models.
# Keys   = model name (str) — must be unique, used throughout the dashboard.
# Values = unfitted estimator or sklearn Pipeline.
#
# Required models (minimum):
#   "Linear Regression"  — baseline, wrap in Pipeline with StandardScaler
#   "Ridge Regression"   — regularised linear, wrap in Pipeline
#   "Random Forest"      — RandomForestRegressor
#   "XGBoost"            — XGBRegressor
#   "Gradient Boosting"  — GradientBoostingRegressor
#
# Tune hyperparameters as you see fit.  Document your choices in the report.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict = {
    # TODO: add your models here
    # Example:
    # "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
}

# Colour map used by the Dash charts — one hex colour per model name.
# Keep keys in sync with MODEL_REGISTRY.
MODEL_COLORS: dict = {
    "Linear Regression":  "#60a5fa",
    "Ridge Regression":   "#a78bfa",
    "Random Forest":      "#34d399",
    "XGBoost":            "#fbbf24",
    "Gradient Boosting":  "#f87171",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute evaluation metrics for one model
# TODO (ML Engineer): implement this function.
# Required keys in returned dict:
#   "RMSE", "MAE", "R²", "MAPE (%)", "Direction Acc (%)"
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression evaluation metrics.

    Metrics to include:
      RMSE             — root mean squared error
      MAE              — mean absolute error
      R²               — coefficient of determination
      MAPE (%)         — mean absolute percentage error × 100
      Direction Acc (%)— % of days where sign(Δpred) == sign(Δactual)
                         i.e. did the model predict the direction correctly?
                         Use np.diff on both arrays.

    Returns a dict with exactly those 5 keys.
    """
    # TODO: implement using sklearn.metrics or manual numpy calculations
    raise NotImplementedError("_compute_metrics not implemented — ML Engineer task")


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer class
# ─────────────────────────────────────────────────────────────────────────────

class StockModelTrainer:
    """
    Train, evaluate and expose predictions for all models in MODEL_REGISTRY.

    Attributes populated by train_all() — the Dash app reads these directly:
      trained_models    : dict[str, fitted_estimator]
      metrics           : dict[str, dict]  — test-set metrics per model
      cv_scores         : dict[str, np.ndarray]  — R² per CV fold
      feature_importances: dict[str, pd.Series]  — sorted, top features
      train_predictions : dict[str, np.ndarray]
      test_predictions  : dict[str, np.ndarray]
      y_train, y_test   : pd.Series
      feature_names     : list[str]
      best_model_name   : str  — model with lowest test RMSE
    """

    def __init__(self, test_size: float = 0.2, n_cv_splits: int = 5):
        self.test_size   = test_size
        self.n_cv_splits = n_cv_splits

        # ── Attributes populated by train_all ────────────────────────────────
        self.trained_models:      dict = {}
        self.metrics:             dict = {}
        self.cv_scores:           dict = {}
        self.feature_importances: dict = {}
        self.train_predictions:   dict = {}
        self.test_predictions:    dict = {}
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names:  list = []
        self.best_model_name: str = ""

    # ── Private: chronological 80/20 split ───────────────────────────────────

    def _time_split(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split X, y chronologically. Do NOT shuffle."""
        split = int(len(X) * (1 - self.test_size))
        self.X_train = X.iloc[:split]
        self.X_test  = X.iloc[split:]
        self.y_train = y.iloc[:split]
        self.y_test  = y.iloc[split:]
        self.feature_names = list(X.columns)

    # ── TODO (ML Engineer): implement train_all ───────────────────────────────

    def train_all(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train every model in MODEL_REGISTRY and populate all attributes.

        Steps:
          1. Call self._time_split(X, y).
          2. For each model name + prototype in MODEL_REGISTRY:
               a. Deep-copy the prototype (use copy.deepcopy).
               b. Fit on X_train / y_train.
               c. Predict on X_train → store in self.train_predictions[name].
               d. Predict on X_test  → store in self.test_predictions[name].
               e. Compute metrics via _compute_metrics → self.metrics[name].
               f. Store the fitted model in self.trained_models[name].
          3. Walk-forward cross-validation (sklearn.model_selection.TimeSeriesSplit)
               — n_splits = self.n_cv_splits
               — Metric: R² on each validation fold
               — Store array of fold scores in self.cv_scores[name].
          4. Extract feature importances via self._extract_feature_importances().
          5. Set self.best_model_name = model with lowest RMSE on the test set.
        """
        # TODO: implement
        raise NotImplementedError("train_all not implemented — ML Engineer task")

    # ── TODO (ML Engineer): implement feature importance extraction ───────────

    def _extract_feature_importances(self) -> None:
        """
        For each fitted model in self.trained_models:
          - If it has .feature_importances_ → use directly.
          - If it has .coef_ (linear models via Pipeline) → use |coef_|.
          - Store as a pd.Series sorted descending in self.feature_importances[name].

        For Pipeline objects, access the inner estimator via
        model.named_steps["model"].
        """
        # TODO: implement
        pass

    # ── TODO (ML Engineer): implement future forecast ─────────────────────────

    def forecast_future(self, X_last_row: np.ndarray,
                        n_days: int = 30,
                        model_name: str = None) -> np.ndarray:
        """
        Iterative n-day forecast using the best (or specified) model.

        Algorithm:
          1. Start with X_last_row (the most recent feature vector).
          2. Predict one step → append to results.
          3. Shift the Close lag features forward by one position
             (Lag1 ← prediction, Lag2 ← old Lag1, etc.).
          4. Repeat n_days times.
          5. Return np.ndarray of predicted prices.

        Hint: self.feature_names contains the ordered feature name list.
              Look for "Close_Lag" in the names to find the lag columns.
        """
        # TODO: implement
        raise NotImplementedError("forecast_future not implemented — ML Engineer task")

    # ── Accessor: metrics DataFrame ───────────────────────────────────────────

    def get_metrics_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per model containing all metrics
        plus CV R² mean and std.  Called by the Dash dashboard.
        Do not modify the column names — the app reads them directly.
        """
        rows = []
        for name, m in self.metrics.items():
            row = {
                "Model":      name,
                **m,
                "CV R² Mean": float(np.mean(self.cv_scores.get(name, [np.nan]))),
                "CV R² Std":  float(np.std(self.cv_scores.get(name, [np.nan]))),
            }
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Model")
        num_cols = df.select_dtypes("number").columns
        df[num_cols] = df[num_cols].round(4)
        return df

    def get_best_predictions(self):
        """Return (predictions, y_test) for the best model."""
        return self.test_predictions[self.best_model_name], self.y_test

    def get_top_features(self, model_name: str = None, top_n: int = 20) -> pd.Series:
        """Return top_n features by importance for a given model."""
        if model_name is None:
            model_name = self.best_model_name
        fi = self.feature_importances.get(model_name)
        if fi is None:
            return pd.Series(dtype=float)
        return fi.head(top_n)
