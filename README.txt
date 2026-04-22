# StockOracle — AI-Powered Stock Prediction Studio

> Multi-model ML forecasting with technical indicators, market sentiment signals, and company fundamentals — built with Plotly Dash and deployable to Render.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dashboard Tabs](#dashboard-tabs)
4. [Project Structure](#project-structure)
5. [Architecture & Data Flow](#architecture--data-flow)
6. [Technical Indicators](#technical-indicators)
7. [ML Models](#ml-models)
8. [Getting Started](#getting-started)
9. [Configuration](#configuration)
10. [Deployment](#deployment)
11. [Module Reference](#module-reference)

---

## Overview

StockOracle is an interactive stock analysis and prediction dashboard built with Plotly Dash. It fetches live OHLCV data from Yahoo Finance, engineers ~60 technical and time-series features, trains five machine learning models simultaneously, and visualises predictions, market regime clusters, and model interpretability — all in a single browser-based interface.

---

## Features

- **Live & Static Data** — Pull live OHLCV data from Yahoo Finance or fall back to a pre-saved CSV snapshot
- **Multi-Model Training** — Five regression models trained on every run: Linear Regression, Ridge Regression, Random Forest, XGBoost, and Gradient Boosting
- **Time-Series Cross-Validation** — 5-fold `TimeSeriesSplit` to prevent data leakage and produce honest performance estimates
- **Future Forecasting** — Iterative next-price prediction out to 7, 14, 30, or 60 trading days using the best-performing model
- **Market Regime Clustering** — K-Means (k=3) segments days into Bullish, High-Volatility, and Bearish regimes
- **SHAP Interpretability** — Global and per-prediction feature attribution using TreeExplainer, LinearExplainer, or KernelExplainer
- **Company Fundamentals** — P/E, forward P/E, P/B, EPS, beta, dividend yield, analyst ratings, and more pulled from Yahoo Finance
- **Seasonality Analysis** — Monthly average return heatmap and annual performance bar chart
- **Live Clock & KPI Bar** — Real-time clock, current price, 52-week high/low, volume, market cap, P/E ratio, and best model RMSE displayed at the top of every analysis

---

## Dashboard Tabs

### Market Overview
Candlestick chart with SMA-5 and SMA-20 overlays, volume bars, and a company summary card showing sector, industry, and recent performance. Presents a quick price-history snapshot for the selected date range.

### ML Predictions
Shows actual vs. predicted prices for the test set across all five models, plus the future forecast curve extending beyond the last known date. Includes a full metrics table (RMSE, MAE, R², MAPE, Direction Accuracy, CV R² mean and std) so models can be compared at a glance.

### Technical Analysis
Deep-dive charts across four indicator groups — Momentum (RSI, MACD, Stochastic, Williams %R), Volatility (Bollinger Bands, ATR, Historical Volatility), Volume (OBV, Volume Ratio), and a broad indicator panel. Each chart is rendered with the dark theme colour palette.

### Model Insights
Feature importance visualised four ways: a horizontal bar chart of the top-20 global importances for the best model, a pie chart breaking importance down by feature category, a cumulative importance curve (highlighting how many features explain 80% of predictive power), and a side-by-side comparison across all five models.

### Fundamentals
Company fact sheet (market cap, P/E ratios, EPS, beta, 52-week range, analyst target and rating), a monthly seasonality heatmap showing average return by month and year, and an annual performance bar chart.

---

## Project Structure

```
.
├── app.py                    # Dash application entry point, all callbacks and tab builders
├── clean_stock_data.py       # Data cleaning helper called by the Run Analysis button
├── data_fetcher.py           # Yahoo Finance OHLCV, VIX, S&P 500 fetchers and company info
├── feature_engineering.py    # ~60 technical indicator and lag/time feature pipeline
├── ml_models.py              # Model registry, StockModelTrainer class, metrics
├── clustering.py             # K-Means market regime segmentation pipeline
├── shap_analysis.py          # SHAP value computation and global/local summaries
├── custom.css                # Dark-theme custom stylesheet for the Dash app
├── render.yaml               # Render.com deployment config (gunicorn)
├── requirements.txt          # Python dependencies
└── data/
    └── ml_ready/             # Auto-created on first run — stores ticker training CSVs
```

> **Note:** `app.py` imports from `utils/` and `models/` sub-packages in the deployed layout. If running locally with the flat file structure, adjust the import paths in `app.py` accordingly.

---

## Architecture & Data Flow

```
User clicks "Run Analysis"
        │
        ▼
clean_stock_data.py          ← yfinance raw OHLCV download
        │  Flatten MultiIndex, add MA5/MA20/Target, ffill, dropna
        │  Save to data/ml_ready/<TICKER>_training_data.csv
        ▼
data_fetcher.py              ← Merge VIX sentiment + S&P 500 return
        ▼
feature_engineering.py       ← add_technical_indicators()
        │                       add_lag_features()
        │                       add_time_features()
        │                       prepare_features() → (X, y, feature_names)
        ▼
ml_models.py                 ← StockModelTrainer.train_all(X, y)
        │                       TimeSeriesSplit CV, metrics, forecasting
        ▼
clustering.py                ← run_clustering() → K-Means regime labels
        │
        ▼
shap_analysis.py             ← run_shap_analysis() → global importance
        │
        ▼
app.py                       ← Serialise to dcc.Store → render 5 tabs
```

---

## Technical Indicators

`feature_engineering.py` produces the following columns, all of which become model input features:

| Category | Indicators |
|---|---|
| Moving Averages | SMA 5/10/20/50/100/200, EMA 5/10/20/50/100/200 |
| MA Crossovers | Price vs SMA5/20/50/200, Golden Cross flag |
| Momentum | RSI-14, RSI-7, MACD + Signal + Histogram, Stochastic %K/%D, Williams %R, CCI, ROC 1/5/10/20 |
| Volatility | Bollinger Bands (upper/mid/lower/width/%B), ATR-14, Normalised ATR, Historical Volatility-21 |
| Volume | OBV, Volume SMA-20, Volume Ratio |
| Price Action | HL Range, OC Range, Upper Shadow, Lower Shadow, Daily Return |
| Lag Features | Close Lag 1/2/3/5/10/20 |
| Time Features | Day of Week, Month, Quarter, Day of Year (sin/cos encoded), Week of Year |
| Market Context | VIX, S&P 500 Daily Return |

---

## ML Models

All models are defined in `MODEL_REGISTRY` inside `ml_models.py`:

| Model | Notes |
|---|---|
| Linear Regression | StandardScaler pipeline; coefficients used as feature importance |
| Ridge Regression | L2-regularised linear model (α=1.0); StandardScaler pipeline |
| Random Forest | 50 estimators, max depth 8 |
| XGBoost | 100 estimators, lr=0.05, max depth 4, subsampling |
| Gradient Boosting | 100 estimators, lr=0.05, max depth 3 |

The best model is selected by lowest RMSE on the held-out test set (the final 20% of the time series). Only this model is used for future forecasting and SHAP analysis.

### Evaluation Metrics

Each model is scored on: RMSE, MAE, R², MAPE (%), Direction Accuracy (%), CV R² mean, and CV R² standard deviation.

### Forecasting

`StockModelTrainer.forecast_future()` iteratively predicts one step ahead, shifting lag features at each step, for up to 60 trading days.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Internet access for live Yahoo Finance data

### Installation

# Verify Python version
py --version
# Must output: Python 3.12.10
# If not installed:
winget install -e --id Python.Python.3.12

# Clone the repository
git clone https://github.com/your-org/stockoracle.git
cd stockoracle

# Create and activate a virtual environment
py -3.12 -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip/setuptools/wheel inside venv
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install extras for visualization
pip install matplotlib seaborn

# Ensure pandas version is correct
pip install pandas==2.2.2

# Verify installed packages
pip list
### Running Locally

```bash
python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

### First Analysis

1. Select a ticker from the dropdown (default: AAPL) or type any valid Yahoo Finance symbol
2. Choose **Live (Yahoo Finance)** or **Static CSV Snapshot** as the data source
3. Set your date range (default: last 5 years)
4. Pick a forecast horizon (1d, 5d, or 21d target) and future forecast window (7–60 days)
5. Click **Run Analysis** — the progress indicator will animate while data is fetched and models are trained
6. Navigate the five tabs to explore results

---

## Configuration

### Supported Tickers (Pre-loaded)

AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, META, BRK-B, JPM, SPY, IBIT

Any valid Yahoo Finance ticker symbol can also be typed directly into the search box.

### Forecast Horizons

| Option | Target variable |
|---|---|
| Next Day (1d) | Tomorrow's closing price |
| Next Week (5d) | Closing price 5 trading days out |
| Next Month (21d) | Closing price 21 trading days out |

### Data Source

| Option | Behaviour |
|---|---|
| Live (Yahoo Finance) | Downloads fresh OHLCV via `yfinance`, merges VIX + S&P 500 context, saves a CSV to `data/ml_ready/` |
| Static CSV Snapshot | Loads `data/ml_ready/AAPL_training_data.csv` — useful for demos without internet access |

---

## Deployment

StockOracle is configured for one-click deployment to [Render](https://render.com) via `render.yaml`.

```yaml
services:
  - type: web
    name: stockoracle
    env: python
    region: oregon
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:server --workers 1 --timeout 120 --bind 0.0.0.0:$PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

**Deployment steps:**

1. Push the repository to GitHub
2. Create a new Web Service on Render and connect the repo
3. Render will detect `render.yaml` automatically and configure the build
4. The app is served via Gunicorn with a 120-second timeout to accommodate model training on cold starts

> The free Render tier has limited RAM. `app.py` runs `gc.collect()` after training to free memory before serialisation. If you hit memory limits, upgrade the plan or reduce `n_estimators` in `MODEL_REGISTRY`.

---

## Module Reference

### `app.py`
Entry point. Defines the Dash layout, all `@app.callback` functions, and the five tab-builder functions (`build_overview_tab`, `build_prediction_tab`, `build_technical_tab`, `build_models_tab`, `build_fundamentals_tab`). State is passed between callbacks using in-memory `dcc.Store` components.

### `clean_stock_data.py`
`clean_stock_data_for_dashboard(ticker, start_date, end_date)` — Downloads raw OHLCV, flattens MultiIndex columns, computes MA5/MA20/Target, forward-fills and drops NaN rows, and persists the result to `data/ml_ready/<TICKER>_training_data.csv`.

### `data_fetcher.py`
`load_full_dataset()` — Fetches OHLCV, merges VIX and S&P 500 returns into a single DataFrame.  
`get_company_info()` — Returns fundamental metrics dict from `yf.Ticker().info`.  
`format_market_cap()` / `format_percentage()` — Display formatting helpers.

### `feature_engineering.py`
`add_technical_indicators(df)` — Adds all ~40 indicator columns to an OHLCV DataFrame.  
`add_lag_features(df)` — Appends lagged close price columns (lags 1, 2, 3, 5, 10, 20).  
`add_time_features(df)` — Cyclically encodes calendar features.  
`prepare_features(df)` — Master pipeline returning `(X, y, feature_names)`.

### `ml_models.py`
`MODEL_REGISTRY` — Dict of model name → sklearn estimator or Pipeline.  
`StockModelTrainer` — Handles time-split, training, CV scoring, metrics, feature importance extraction, and iterative future forecasting.  
`StockModelTrainer.train_all(X, y)` — Trains and evaluates all registry models in one call.  
`StockModelTrainer.forecast_future(X_last_row, n_days)` — Returns an array of future price predictions.

### `clustering.py`
`run_clustering(df_with_indicators)` — Full pipeline returning labels, cluster statistics, silhouette score, cluster centres, and elbow-analysis data.  
Cluster labels: `0 = Bullish / Low-Vol`, `1 = High-Vol / Uncertain`, `2 = Bearish / Drawdown`.

### `shap_analysis.py`
`run_shap_analysis(trainer)` — Selects the appropriate SHAP explainer for the best model and returns SHAP values, expected value, and a global importance DataFrame.  
`get_global_importance()` — Mean absolute SHAP per feature, sorted descending.  
`get_local_explanation()` — Waterfall chart data for a single prediction row.  
`get_cluster_shap_summary()` — Per-regime mean SHAP values for interpretability across market conditions.

---

## Dependencies

| Package | Role |
|---|---|
| dash + dash-bootstrap-components | Web framework and UI components |
| plotly | Interactive charts |
| pandas / numpy | Data manipulation |
| yfinance | Yahoo Finance data |
| scikit-learn | ML models, preprocessing, metrics |
| xgboost | Gradient boosted trees |
| shap | Model interpretability |
| scipy / joblib | Supporting ML utilities |
| gunicorn | Production WSGI server |
| ta | Additional technical analysis helpers |
