import numpy as np
import pandas as pd
import copy
 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
 
import warnings
warnings.filterwarnings("ignore")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# OPTIMIZED: reduced estimators for Render free tier speed
# RF/XGB/GB were the bottleneck — 3-5x faster with these values
# ─────────────────────────────────────────────────────────────────────────────
 
MODEL_REGISTRY: dict = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
 
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
 
    "Random Forest": RandomForestRegressor(
        n_estimators=20,       # was 50 — 2.5x faster, minimal accuracy loss
        max_depth=6,           # was 8  — prevents overfitting on small free-tier runs
        n_jobs=-1,             # NEW: use all available cores
        random_state=42
    ),
 
    "XGBoost": XGBRegressor(
        n_estimators=40,       # was 100 — 2.5x faster
        max_depth=4,
        learning_rate=0.1,     # was 0.05 — higher lr compensates for fewer trees
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",    # NEW: fastest XGBoost mode
        random_state=42,
        verbosity=0,
    ),
 
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=40,       # was 100 — 2.5x faster
        learning_rate=0.1,     # was 0.05 — compensates for fewer trees
        max_depth=3,
        subsample=0.8,         # NEW: adds regularisation + speed via row sampling
    ),
}
 
 
MODEL_COLORS: dict = {
    "Linear Regression":  "#60a5fa",
    "Ridge Regression":   "#a78bfa",
    "Random Forest":      "#34d399",
    "XGBoost":            "#fbbf24",
    "Gradient Boosting":  "#f87171",
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
 
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
 
    actual_diff    = np.diff(y_true)
    pred_diff      = np.diff(y_pred)
    direction_acc  = float(np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100)
 
    return {
        "RMSE":               rmse,
        "MAE":                mae,
        "R²":                 r2,
        "MAPE (%)":           mape,
        "Direction Acc (%)":  direction_acc,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Parallel helpers — joblib calls these at the module level
# ─────────────────────────────────────────────────────────────────────────────
 
def _train_one_model(name, prototype, X_train, X_test, y_train, y_test,
                     n_cv_splits, feature_names):
    """Train a single model + run CV. Called in parallel via joblib."""
    model = copy.deepcopy(prototype)
    model.fit(X_train, y_train)
 
    train_pred = np.array(model.predict(X_train), dtype=float)
    test_pred  = np.array(model.predict(X_test),  dtype=float)
    metrics    = _compute_metrics(y_test.to_numpy(), test_pred)
 
    # Cross-validation
    tscv   = TimeSeriesSplit(n_splits=n_cv_splits)
    scores = []
    for tr_idx, val_idx in tscv.split(X_train):
        cv_m = copy.deepcopy(prototype)
        cv_m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = cv_m.predict(X_train.iloc[val_idx])
        scores.append(r2_score(y_train.iloc[val_idx], preds))
 
    # Feature importance
    inner = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(inner, "feature_importances_"):
        fi_values = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        fi_values = np.abs(inner.coef_)
    else:
        fi_values = None
 
    fi_series = None
    if fi_values is not None:
        fi_series = pd.Series(fi_values, index=feature_names).sort_values(ascending=False).astype(float)
 
    return name, model, train_pred, test_pred, metrics, np.array(scores, dtype=float), fi_series
 
 
# ─────────────────────────────────────────────────────────────────────────────
# StockModelTrainer
# ─────────────────────────────────────────────────────────────────────────────
 
class StockModelTrainer:
 
    def __init__(self, test_size: float = 0.2, n_cv_splits: int = 3):
        # OPTIMIZED: default CV splits 5 → 3
        # Saves 2 full training runs per model = 10 fewer model fits total
        self.test_size   = test_size
        self.n_cv_splits = n_cv_splits
 
        self.trained_models      = {}
        self.metrics             = {}
        self.cv_scores           = {}
        self.feature_importances = {}
        self.train_predictions   = {}
        self.test_predictions    = {}
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = []
        self.best_model_name = ""
 
    def _time_split(self, X: pd.DataFrame, y: pd.Series):
        split        = int(len(X) * (1 - self.test_size))
        self.X_train = X.iloc[:split].copy()
        self.X_test  = X.iloc[split:].copy()
        self.y_train = y.iloc[:split].copy()
        self.y_test  = y.iloc[split:].copy()
        self.feature_names = list(X.columns)
 
    def train_all(self, X: pd.DataFrame, y: pd.Series):
        self._time_split(X, y)
 
        # OPTIMIZED: train all models in parallel using joblib
        # On Render free tier (limited CPU) this still helps because
        # sklearn releases the GIL during C-extension computation
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_train_one_model)(
                name, proto,
                self.X_train, self.X_test,
                self.y_train, self.y_test,
                self.n_cv_splits, self.feature_names,
            )
            for name, proto in MODEL_REGISTRY.items()
        )
 
        for name, model, tr_pred, te_pred, metrics, cv_sc, fi in results:
            self.trained_models[name]      = model
            self.train_predictions[name]   = tr_pred
            self.test_predictions[name]    = te_pred
            self.metrics[name]             = metrics
            self.cv_scores[name]           = cv_sc
            if fi is not None:
                self.feature_importances[name] = fi
 
        self.best_model_name = min(
            self.metrics, key=lambda m: self.metrics[m]["RMSE"]
        )
 
    def forecast_future(self, X_last_row: np.ndarray,
                        n_days: int = 30,
                        model_name: str = None) -> np.ndarray:
        if model_name is None:
            model_name = self.best_model_name
 
        model   = self.trained_models[model_name]
        current = np.array(X_last_row, dtype=float).copy()
        preds   = []
 
        lag_indices = sorted([
            i for i, col in enumerate(self.feature_names)
            if "lag" in col.lower()
        ])
 
        for _ in range(n_days):
            pred = float(model.predict(current.reshape(1, -1))[0])
            preds.append(pred)
            if lag_indices:
                for i in range(len(lag_indices) - 1, 0, -1):
                    current[lag_indices[i]] = current[lag_indices[i - 1]]
                current[lag_indices[0]] = pred
 
        return np.array(preds, dtype=float)
 
    def get_metrics_df(self) -> pd.DataFrame:
        rows = []
        for name, m in self.metrics.items():
            row = {
                "Model":      name,
                **m,
                "CV R² Mean": float(np.mean(self.cv_scores.get(name, [np.nan]))),
                "CV R² Std":  float(np.std(self.cv_scores.get(name, [np.nan]))),
            }
            rows.append(row)
        df      = pd.DataFrame(rows).set_index("Model")
        num_col = df.select_dtypes("number").columns
        df[num_col] = df[num_col].round(4)
        return df
 
    def get_best_predictions(self):
        return self.test_predictions[self.best_model_name], self.y_test
 
    def get_top_features(self, model_name: str = None, top_n: int = 20):
        if model_name is None:
            model_name = self.best_model_name
        fi = self.feature_importances.get(model_name)
        if fi is None:
            return pd.Series(dtype=float)
        return fi.head(top_n)