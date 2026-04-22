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

import warnings
warnings.filterwarnings("ignore")


#Registry for each model

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
        n_estimators=50,
        max_depth=8,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3
    ),
}


MODEL_COLORS: dict = {
    "Linear Regression":  "#60a5fa",
    "Ridge Regression":   "#a78bfa",
    "Random Forest":      "#34d399",
    "XGBoost":            "#fbbf24",
    "Gradient Boosting":  "#f87171",
}


#Definition for computing metrics regarding each model

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    direction_acc = float(
        np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE (%)": mape,
        "Direction Acc (%)": direction_acc
    }


#Training each model according to structure

class StockModelTrainer:

    def __init__(self, test_size: float = 0.2, n_cv_splits: int = 5):
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
        split = int(len(X) * (1 - self.test_size))

        self.X_train = X.iloc[:split].copy()
        self.X_test  = X.iloc[split:].copy()

        self.y_train = y.iloc[:split].copy()
        self.y_test  = y.iloc[split:].copy()

        self.feature_names = list(X.columns)

    def train_all(self, X: pd.DataFrame, y: pd.Series):

        self._time_split(X, y)
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        for name, prototype in MODEL_REGISTRY.items():

            model = copy.deepcopy(prototype)

            # Training each model
            model.fit(self.X_train, self.y_train)

            # Making predictions for each model
            train_pred = model.predict(self.X_train)
            test_pred  = model.predict(self.X_test)

            # Storing predictions ensuring they are json safe
            self.train_predictions[name] = np.array(train_pred, dtype=float)
            self.test_predictions[name]  = np.array(test_pred, dtype=float)

            # Metrics
            self.metrics[name] = _compute_metrics(
                self.y_test.to_numpy(),
                test_pred
            )

            # Store model
            self.trained_models[name] = model
          
            scores = []

            for train_idx, val_idx in tscv.split(self.X_train):

                X_tr = self.X_train.iloc[train_idx]
                X_val = self.X_train.iloc[val_idx]

                y_tr = self.y_train.iloc[train_idx]
                y_val = self.y_train.iloc[val_idx]

                cv_model = copy.deepcopy(prototype)
                cv_model.fit(X_tr, y_tr)

                preds = cv_model.predict(X_val)
                scores.append(r2_score(y_val, preds))

            self.cv_scores[name] = np.array(scores, dtype=float)

        self._extract_feature_importances()

        #Choosing best model for predictions
        self.best_model_name = min(
            self.metrics,
            key=lambda m: self.metrics[m]["RMSE"]
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _extract_feature_importances(self):

        for name, model in self.trained_models.items():
            
            if isinstance(model, Pipeline):
                estimator = model.named_steps["model"]
            else:
                estimator = model

            if hasattr(estimator, "feature_importances_"):
                values = estimator.feature_importances_

            elif hasattr(estimator, "coef_"):
                values = np.abs(estimator.coef_)

            else:
                continue

            fi = pd.Series(values, index=self.feature_names)
            fi = fi.sort_values(ascending=False)

            # Ensure JSON-safe
            self.feature_importances[name] = fi.astype(float)

#Using models in order to make robust predictions

    def forecast_future(self, X_last_row: np.ndarray,
                        n_days: int = 30,
                        model_name: str = None) -> np.ndarray:

        if model_name is None:
            model_name = self.best_model_name

        model = self.trained_models[model_name]

        current = np.array(X_last_row, dtype=float).copy()
        preds = []

        # Detect lag features robustly
        lag_indices = [
            i for i, col in enumerate(self.feature_names)
            if "lag" in col.lower()
        ]

        # Sort by lag number if present
        lag_indices = sorted(lag_indices)

        for _ in range(n_days):

            pred = float(model.predict(current.reshape(1, -1))[0])
            preds.append(pred)

            # Shift lag values
            if lag_indices:
                for i in range(len(lag_indices)-1, 0, -1):
                    current[lag_indices[i]] = current[lag_indices[i-1]]

                # Insert new prediction at Lag1
                current[lag_indices[0]] = pred

        return np.array(preds, dtype=float)

    # ─────────────────────────────────────────────────────────────────────────

    def get_metrics_df(self) -> pd.DataFrame:

        rows = []
        for name, m in self.metrics.items():
            row = {
                "Model": name,
                **m,
                "CV R² Mean": float(np.mean(self.cv_scores.get(name, [np.nan]))),
                "CV R² Std":  float(np.std(self.cv_scores.get(name, [np.nan])))
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")

        num_cols = df.select_dtypes("number").columns
        df[num_cols] = df[num_cols].round(4)

        return df

    # ─────────────────────────────────────────────────────────────────────────

    def get_best_predictions(self):
        return self.test_predictions[self.best_model_name], self.y_test

    def get_top_features(self, model_name: str = None, top_n: int = 20):

        if model_name is None:
            model_name = self.best_model_name

        fi = self.feature_importances.get(model_name)

        if fi is None:
            return pd.Series(dtype=float)

        return fi.head(top_n)