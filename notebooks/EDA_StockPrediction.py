"""
EDA_StockPrediction.py  — CRISP-DM Phase 2: Data Understanding
════════════════════════════════════════════════════════════════
OWNER:  Role 2 — Data Engineer & Preprocessing Specialist
STATUS: SKELETON — implement all TODO sections.

Convert to Jupyter Notebook:
    pip install jupytext
    jupytext --to notebook EDA_StockPrediction.py

Or run directly:
    python EDA_StockPrediction.py

All output figures should be saved to notebooks/figures/.
These figures are referenced in the Technical Report (Role 1).
"""

# %% [markdown]
# # 📊 Stock Price Prediction — Exploratory Data Analysis
# **Project:** StockOracle — AI-Powered Stock Price Forecasting
# **Owner:** Role 2 — Data Engineer & Preprocessing Specialist
#
# ## Objectives
# 1. Understand the raw OHLCV data structure and quality.
# 2. Identify missing values, outliers, and data quality issues.
# 3. Explore price distributions, returns, and volatility patterns.
# 4. Visualise technical indicators and their relationships.
# 5. Analyse correlations between features and the prediction target.
# 6. Provide insights for the Technical Report and final video.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")    # remove this line if running interactively in Jupyter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
# Adjust this if running from a different working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_fetcher import load_full_dataset, get_company_info
from utils.feature_engineering import (
    add_technical_indicators, add_lag_features,
    add_time_features, prepare_features
)
from models.ml_models import StockModelTrainer, MODEL_COLORS

# ── Output directory for figures ────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# ── Plot theme ───────────────────────────────────────────────────────────────
# TODO (Data Engineer): customise this to match the dashboard's dark theme.
plt.style.use("dark_background")
rcParams["font.family"]    = "monospace"
rcParams["axes.facecolor"] = "#0f1a2e"
rcParams["figure.facecolor"] = "#060b14"
rcParams["text.color"]     = "#dce8f5"
rcParams["axes.labelcolor"] = "#6b8aad"
rcParams["xtick.color"]    = "#6b8aad"
rcParams["ytick.color"]    = "#6b8aad"
rcParams["grid.color"]     = "#1a2d4a"
rcParams["grid.linewidth"] = 0.5
rcParams["axes.spines.top"]   = False
rcParams["axes.spines.right"] = False

NEON = {
    "blue":   "#00c8ff",
    "purple": "#7b2fff",
    "green":  "#00e5a0",
    "yellow": "#ffc440",
    "red":    "#ff4d6d",
}

# ── Data parameters — change as needed ──────────────────────────────────────
TICKER = "AAPL"
END    = datetime.today().strftime("%Y-%m-%d")
START  = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")


# %% [markdown]
# ## Section 1 — Data Loading & Overview
# **TODO:** Fetch the data, inspect its shape, dtypes, and missing values.

# %%
# TODO (Data Engineer): fetch data and print a summary.
# Expected output:
#   - Shape (rows × columns)
#   - First 5 rows
#   - Data types of each column
#   - Count of missing values per column
#   - Descriptive statistics (df.describe())
#
# Use load_full_dataset() — it fetches OHLCV + VIX + S&P500 return.
#
# Example:
#   df_raw  = load_full_dataset(TICKER, START, END)
#   company = get_company_info(TICKER)
#   print(df_raw.head())
#   ...

raise NotImplementedError(
    "Section 1 not implemented — Data Engineer task.\n"
    "Fetch the data and print a structured summary."
)


# %% [markdown]
# ## Section 2 — Price History Chart
# **TODO:** Plot adjusted close price + volume over the full date range.

# %%
# TODO (Data Engineer): create a 2-panel figure:
#   Panel 1 (large): line chart of Close price with fill
#   Panel 2 (small): volume bar chart, coloured green/red by direction
#
# Save to:  figures/fig01_price_history.png
#
# Hint: use plt.subplots(2, 1, gridspec_kw={"height_ratios":[3,1]})

raise NotImplementedError("Section 2 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 3 — Returns Analysis
# **TODO:** Analyse daily returns distribution and rolling volatility.

# %%
# TODO (Data Engineer): create a 3-panel figure:
#   Panel 1: histogram of daily % returns with mean and ±1σ lines
#   Panel 2: Q-Q plot against a normal distribution (use scipy.stats.probplot)
#   Panel 3: 30-day rolling annualised volatility = rolling(30).std() * sqrt(252)
#
# Report skewness and kurtosis in a print statement.
# Save to: figures/fig02_returns.png

raise NotImplementedError("Section 3 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 4 — Technical Indicators Visualisation
# **TODO:** Visualise the key indicators your feature_engineering module produces.

# %%
# TODO (Data Engineer): call add_technical_indicators(df_raw.copy()) and
# create a 4-panel stacked chart (shared x-axis):
#   Panel 1: Close + SMA_50 + SMA_200 + Bollinger Bands
#   Panel 2: RSI_14 with overbought (70) / oversold (30) lines
#   Panel 3: MACD line, Signal line, Histogram bars
#   Panel 4: Volume bars + Volume_SMA20 line
#
# Print the number of feature columns produced.
# Save to: figures/fig03_indicators.png

raise NotImplementedError("Section 4 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 5 — Correlation Analysis
# **TODO:** Find which features correlate most with the prediction target.

# %%
# TODO (Data Engineer): call prepare_features(df_raw.copy()) to get the
# full feature matrix, then:
#   1. Compute |Pearson correlation| of each feature with Target.
#   2. Plot a horizontal bar chart of the top 30 features.
#   3. Print the top 5 features with their correlation values.
#
# Save to: figures/fig04_correlations.png
#
# Discussion question for report:
#   Which category of features (MA, momentum, volume) has the highest correlation?

raise NotImplementedError("Section 5 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 6 — Model Training & Evaluation
# **TODO:** Train all models and print the metrics table.

# %%
# TODO (Data Engineer / ML Engineer collaboration):
#   1. Call prepare_features() to get X, y.
#   2. Instantiate StockModelTrainer and call train_all(X, y).
#   3. Print the metrics DataFrame.
#   4. Print the best model name.
#
# Note: This section depends on Role 3 (ML Engineer) completing ml_models.py.

raise NotImplementedError("Section 6 not implemented — requires ml_models.py")


# %% [markdown]
# ## Section 7 — Actual vs Predicted (All Models)
# **TODO:** Plot one panel per model showing test-set predictions vs actual.

# %%
# TODO (Data Engineer): after train_all():
#   Create a figure with len(MODEL_REGISTRY) subplots (shared x-axis).
#   Each panel: actual (solid) vs predicted (dashed) on the test dates.
#   Title each panel with model name + RMSE + R² + MAPE.
#
# Save to: figures/fig05_all_predictions.png

raise NotImplementedError("Section 7 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 8 — Feature Importance
# **TODO:** Visualise the top-20 feature importances for tree-based models.

# %%
# TODO (Data Engineer): for each tree-based model (Random Forest, XGBoost,
# Gradient Boosting):
#   Plot a horizontal bar chart of the top 20 features by importance.
#   Use MODEL_COLORS for bar colours.
#
# Save to: figures/fig06_feature_importance.png

raise NotImplementedError("Section 8 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 9 — Cross-Validation Results
# **TODO:** Bar chart of walk-forward CV R² scores.

# %%
# TODO (Data Engineer): plot a grouped bar chart:
#   x-axis: model names
#   y-axis: mean CV R²
#   error bars: std of CV R²
#   Add the mean value as text above each bar.
#
# Save to: figures/fig07_cv_scores.png

raise NotImplementedError("Section 9 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 10 — Seasonality Analysis
# **TODO:** Identify month-of-year and day-of-week return patterns.

# %%
# TODO (Data Engineer): create a 2-panel figure:
#   Panel 1: seaborn heatmap of average monthly return per year
#            (pivot: rows=year, columns=month, values=mean_monthly_return)
#   Panel 2: bar chart of average daily return by day-of-week
#
# Discuss: is there a January effect? Day-of-week anomaly?
# Save to: figures/fig08_seasonality.png

raise NotImplementedError("Section 10 not implemented — Data Engineer task")


# %% [markdown]
# ## Section 11 — Summary & Insights
# **TODO:** Write a brief text summary of your EDA findings.

# %%
# TODO (Data Engineer): print a structured summary covering:
#   - Dataset: ticker, date range, number of trading days
#   - Data quality: any missing values found? How handled?
#   - Return distribution: approximately normal? Fat tails? Skew direction?
#   - Most correlated features with target
#   - Any visible seasonality patterns?
#   - Recommended feature set for modeling (based on correlation analysis)
#
# This summary will be used directly in the Technical Report by Role 1.

print("TODO: write EDA summary — Data Engineer task")
