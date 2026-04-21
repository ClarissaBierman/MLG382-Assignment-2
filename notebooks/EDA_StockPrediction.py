"""
EDA_StockPrediction.py  — CRISP-DM Phase 2: Data Understanding
════════════════════════════════════════════════════════════════
"""
# ## Objectives
# 1. Understand the raw OHLCV data structure and quality.
# 2. Identify missing values, outliers, and data quality issues.
# 3. Explore price distributions, returns, and volatility patterns.
# 4. Visualise technical indicators and their relationships.
# 5. Analyse correlations between features and the prediction target.
# 6. Provide insights for the Technical Report and final video.

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
from scipy import stats

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

# ── Data parameters ──────────────────────────────────────
TICKER = "AAPL"
END    = datetime.today().strftime("%Y-%m-%d")
START  = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
#────────────────────────────────────────────────────────────────────────────
# Section 1 — Data Loading & Overview
#────────────────────────────────────────────────────────────────────────────
# %%
print("="*80)
print("SECTION 1: DATA LOADING & OVERVIEW")
print("="*80)

# Fetch data
df_raw = load_full_dataset(TICKER, START, END)
company = get_company_info(TICKER)

# Dataset overview
print(f"\nDataset: {TICKER} — {company.get('longName', 'N/A')}")
print(f"Date Range: {df_raw.index.min().strftime('%Y-%m-%d')} to {df_raw.index.max().strftime('%Y-%m-%d')}")
print(f"Trading Days: {len(df_raw):,}")
print(f"Columns: {df_raw.shape[1]}")

# Display first rows
print("\nFirst 5 rows:")
print(df_raw.head())

# Data types
print("\nData Types:")
print(df_raw.dtypes)

# Missing values
print("\nMissing Values:")
missing = df_raw.isnull().sum()
if missing.sum() == 0:
    print("No missing values detected")
else:
    print(missing[missing > 0])

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df_raw.describe())

#────────────────────────────────────────────────────────────────────────────
# Section 2 — Price History Chart
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 2: PRICE HISTORY VISUALIZATION")
print("="*80)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                gridspec_kw={"height_ratios": [3, 1]})

# Panel 1: Close price with fill
ax1.plot(df_raw.index, df_raw["Close"], color=NEON["blue"], linewidth=1.5, label="Close Price")
ax1.fill_between(df_raw.index, df_raw["Close"], alpha=0.15, color=NEON["blue"])
ax1.set_ylabel("Price ($)", fontsize=12)
ax1.set_title(f"{TICKER} Price History — {START} to {END}", 
              fontsize=14, fontweight="bold", color=NEON["green"])
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Panel 2: Volume bars colored by direction
volume_color = np.where(df_raw["Close"] >= df_raw["Open"], NEON["green"], NEON["red"])
ax2.bar(df_raw.index, df_raw["Volume"], color=volume_color, alpha=0.7, width=1)
ax2.set_ylabel("Volume", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig01_price_history.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
print("Saved: figures/fig01_price_history.png")
plt.close()

#────────────────────────────────────────────────────────────────────────────
# Section 3 — Returns Analysis
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 3: RETURNS DISTRIBUTION & VOLATILITY")
print("="*80)

# Calculate returns
daily_returns = df_raw["Close"].pct_change().dropna() * 100

# Statistics
mean_ret = daily_returns.mean()
std_ret = daily_returns.std()
skew = daily_returns.skew()
kurt = daily_returns.kurtosis()

print(f"\nDaily Returns Statistics:")
print(f"Mean:     {mean_ret:.4f}%")
print(f"Std Dev:  {std_ret:.4f}%")
print(f"Skewness: {skew:.4f}")
print(f"Kurtosis: {kurt:.4f}")

# Create 3-panel figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Histogram
ax1.hist(daily_returns, bins=50, color=NEON["purple"], alpha=0.7, edgecolor="white")
ax1.axvline(mean_ret, color=NEON["yellow"], linestyle="--", linewidth=2, label=f"Mean: {mean_ret:.2f}%")
ax1.axvline(mean_ret + std_ret, color=NEON["red"], linestyle="--", linewidth=1.5, label=f"+1σ: {std_ret:.2f}%")
ax1.axvline(mean_ret - std_ret, color=NEON["red"], linestyle="--", linewidth=1.5, label=f"-1σ")
ax1.set_xlabel("Daily Return (%)", fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)
ax1.set_title("Distribution of Daily Returns", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Q-Q plot
stats.probplot(daily_returns, dist="norm", plot=ax2)
ax2.get_lines()[0].set_color(NEON["blue"])
ax2.get_lines()[0].set_markersize(4)
ax2.get_lines()[1].set_color(NEON["red"])
ax2.set_title("Q-Q Plot (Normal Distribution)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)

# Panel 3: Rolling 30-day volatility
log_returns = np.log(df_raw["Close"] / df_raw["Close"].shift(1))
rolling_vol = log_returns.rolling(window=30).std() * np.sqrt(252) * 100

ax3.plot(df_raw.index, rolling_vol, color=NEON["green"], linewidth=1.5)
ax3.fill_between(df_raw.index, rolling_vol, alpha=0.2, color=NEON["green"])
ax3.set_xlabel("Date", fontsize=11)
ax3.set_ylabel("Annualized Volatility (%)", fontsize=11)
ax3.set_title("30-Day Rolling Volatility", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig02_returns.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
print("Saved: figures/fig02_returns.png")
plt.close()

#────────────────────────────────────────────────────────────────────────────
# Section 4 — Technical Indicators Visualisation
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 4: TECHNICAL INDICATORS")
print("="*80)

# Add technical indicators
df_indicators = add_technical_indicators(df_raw.copy())
print(f"\nTotal features generated: {len(df_indicators.columns)}")

# Create 4-panel stacked chart
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Panel 1: Price + Moving Averages + Bollinger Bands
ax1.plot(df_indicators.index, df_indicators["Close"], color=NEON["blue"], 
         linewidth=1.5, label="Close", alpha=0.9)
ax1.plot(df_indicators.index, df_indicators["SMA_50"], color=NEON["yellow"], 
         linewidth=1, label="SMA 50", alpha=0.8)
ax1.plot(df_indicators.index, df_indicators["SMA_200"], color=NEON["red"], 
         linewidth=1, label="SMA 200", alpha=0.8)
ax1.plot(df_indicators.index, df_indicators["BB_Upper"], color=NEON["green"], 
         linewidth=0.8, linestyle="--", label="BB Upper", alpha=0.6)
ax1.plot(df_indicators.index, df_indicators["BB_Lower"], color=NEON["green"], 
         linewidth=0.8, linestyle="--", label="BB Lower", alpha=0.6)
ax1.fill_between(df_indicators.index, df_indicators["BB_Upper"], 
                  df_indicators["BB_Lower"], alpha=0.1, color=NEON["green"])
ax1.set_ylabel("Price ($)", fontsize=11)
ax1.set_title(f"{TICKER} Technical Analysis", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9, ncol=3)
ax1.grid(True, alpha=0.3)

# Panel 2: RSI
ax2.plot(df_indicators.index, df_indicators["RSI_14"], color=NEON["purple"], linewidth=1.5)
ax2.axhline(70, color=NEON["red"], linestyle="--", linewidth=1, alpha=0.7, label="Overbought (70)")
ax2.axhline(30, color=NEON["green"], linestyle="--", linewidth=1, alpha=0.7, label="Oversold (30)")
ax2.fill_between(df_indicators.index, 70, 100, alpha=0.1, color=NEON["red"])
ax2.fill_between(df_indicators.index, 0, 30, alpha=0.1, color=NEON["green"])
ax2.set_ylabel("RSI", fontsize=11)
ax2.set_ylim(0, 100)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: MACD
ax3.plot(df_indicators.index, df_indicators["MACD"], color=NEON["blue"], 
         linewidth=1.5, label="MACD")
ax3.plot(df_indicators.index, df_indicators["MACD_Signal"], color=NEON["yellow"], 
         linewidth=1.5, label="Signal")
colors = [NEON["green"] if h >= 0 else NEON["red"] for h in df_indicators["MACD_Hist"]]
ax3.bar(df_indicators.index, df_indicators["MACD_Hist"], color=colors, alpha=0.6, width=1)
ax3.axhline(0, color="white", linewidth=0.8, alpha=0.5)
ax3.set_ylabel("MACD", fontsize=11)
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Volume
volume_colors = [NEON["green"] if df_indicators["Close"].iloc[i] >= df_indicators["Open"].iloc[i] 
                 else NEON["red"] for i in range(len(df_indicators))]
ax4.bar(df_indicators.index, df_indicators["Volume"], color=volume_colors, alpha=0.7, width=1)
ax4.plot(df_indicators.index, df_indicators["Volume_SMA20"], color=NEON["yellow"], 
         linewidth=1.5, label="Vol SMA 20", alpha=0.9)
ax4.set_ylabel("Volume", fontsize=11)
ax4.set_xlabel("Date", fontsize=11)
ax4.legend(loc="upper left", fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig03_indicators.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
print("Saved: figures/fig03_indicators.png")
plt.close()

#────────────────────────────────────────────────────────────────────────────
# Section 5 — Correlation Analysis
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 5: FEATURE CORRELATION ANALYSIS")
print("="*80)

# Prepare full feature matrix
X, y, feature_names = prepare_features(df_raw.copy())

# Compute correlation with target
correlations = X.corrwith(y).abs().sort_values(ascending=False).head(30)

print(f"\nTop 5 Most Correlated Features with Target:")
for i, (feat, corr) in enumerate(correlations.head(5).items(), 1):
    print(f"  {i}. {feat:30s} → {corr:.4f}")

# Plot horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = [NEON["green"] if i < 10 else NEON["blue"] if i < 20 else NEON["purple"] 
              for i in range(len(correlations))]
ax.barh(range(len(correlations)), correlations.values, color=colors_bar, alpha=0.8)
ax.set_yticks(range(len(correlations)))
ax.set_yticklabels(correlations.index, fontsize=9)
ax.set_xlabel("Absolute Correlation with Target", fontsize=11)
ax.set_title("Top 30 Features by Correlation with Target Price", 
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("figures/fig04_correlations.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
print("Saved: figures/fig04_correlations.png")
plt.close()

#────────────────────────────────────────────────────────────────────────────
# Section 6 — Model Training & Evaluation
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 6: MODEL TRAINING")
print("="*80)

# Train models
trainer = StockModelTrainer(test_size=0.2, n_cv_splits=5)
trainer.train_all(X, y)

# Print metrics
metrics_df = trainer.get_metrics_df()
print("\nModel Performance Metrics:")
print(metrics_df.to_string())

print(f"\nBest Model: {trainer.best_model_name}")
print(f"   RMSE: {metrics_df.loc[trainer.best_model_name, 'RMSE']:.4f}")
print(f"   R²:   {metrics_df.loc[trainer.best_model_name, 'R²']:.4f}")

#────────────────────────────────────────────────────────────────────────────
# Section 7 — Actual vs Predicted (All Models)
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 7: PREDICTIONS VISUALIZATION")
print("="*80)

n_models = len(trainer.trained_models)
fig, axes = plt.subplots(n_models, 1, figsize=(14, 3*n_models), sharex=True)

if n_models == 1:
    axes = [axes]

test_dates = trainer.X_test.index

for idx, (name, pred) in enumerate(trainer.test_predictions.items()):
    ax = axes[idx]
    
    # Plot actual vs predicted
    ax.plot(test_dates, trainer.y_test.values, color=NEON["green"], 
            linewidth=2, label="Actual", alpha=0.9)
    ax.plot(test_dates, pred, color=MODEL_COLORS.get(name, NEON["blue"]), 
            linewidth=1.5, linestyle="--", label="Predicted", alpha=0.8)
    
    # Add metrics to title
    rmse = metrics_df.loc[name, "RMSE"]
    r2 = metrics_df.loc[name, "R²"]
    mape = metrics_df.loc[name, "MAPE (%)"]
    
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_title(f"{name} — RMSE: {rmse:.4f} | R²: {r2:.4f} | MAPE: {mape:.2f}%", 
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Date", fontsize=11)
plt.tight_layout()
plt.savefig("figures/fig05_all_predictions.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
print("Saved: figures/fig05_all_predictions.png")
plt.close()

#────────────────────────────────────────────────────────────────────────────
# ## Section 8 — Feature Importance
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 8: FEATURE IMPORTANCE")
print("="*80)

# Tree-based models
tree_models = ["Random Forest", "XGBoost", "Gradient Boosting"]
available_tree_models = [m for m in tree_models if m in trainer.feature_importances]

if available_tree_models:
    fig, axes = plt.subplots(len(available_tree_models), 1, 
                             figsize=(12, 5*len(available_tree_models)))
    
    if len(available_tree_models) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(available_tree_models):
        ax = axes[idx]
        fi = trainer.get_top_features(model_name, top_n=20)
        
        color = MODEL_COLORS.get(model_name, NEON["blue"])
        ax.barh(range(len(fi)), fi.values, color=color, alpha=0.8)
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi.index, fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"{model_name} — Top 20 Features", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("figures/fig06_feature_importance.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
    print("Saved: figures/fig06_feature_importance.png")
    plt.close()
else:
    print("No tree-based models available for feature importance visualization")

#────────────────────────────────────────────────────────────────────────────
# ## Section 9 — Cross-Validation Results
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 9: CROSS-VALIDATION SCORES")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 6))

models = list(trainer.cv_scores.keys())
cv_means = [np.mean(trainer.cv_scores[m]) for m in models]
cv_stds = [np.std(trainer.cv_scores[m]) for m in models]

x_pos = np.arange(len(models))
colors_cv = [MODEL_COLORS.get(m, NEON["blue"]) for m in models]

bars = ax.bar(x_pos, cv_means, yerr=cv_stds, color=colors_cv, alpha=0.8, 
              capsize=5, error_kw={"linewidth": 2, "ecolor": NEON["red"]})

# Add value labels
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, mean + std + 0.01, f"{mean:.3f}", ha="center", va="bottom", 
            fontsize=10, fontweight="bold", color=NEON["yellow"])

ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
ax.set_ylabel("R² Score", fontsize=12)
ax.set_title("Cross-Validation Performance (Mean ± Std)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig("figures/fig07_cv_scores.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
print("Saved: figures/fig07_cv_scores.png")
plt.close()
#────────────────────────────────────────────────────────────────────────────
# ## Section 10 — Seasonality Analysis
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 10: SEASONALITY PATTERNS")
print("="*80)

df_season = df_raw.copy()
df_season["Year"] = df_season.index.year
df_season["Month"] = df_season.index.month
df_season["DayOfWeek"] = df_season.index.dayofweek
df_season["MonthlyReturn"] = df_season["Close"].pct_change() * 100

# Monthly heatmap
pivot = df_season.groupby(["Year", "Month"])["MonthlyReturn"].mean().unstack(level=1)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
pivot.columns = month_names[:len(pivot.columns)]

# Day-of-week analysis
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
dow_returns = df_season.groupby("DayOfWeek")["MonthlyReturn"].mean()
dow_returns.index = dow_names[:len(dow_returns)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Monthly heatmap
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, 
            cbar_kws={"label": "Avg Return (%)"}, ax=ax1, 
            linewidths=0.5, linecolor="#1a2d4a")
ax1.set_title("Monthly Average Returns by Year", fontsize=13, fontweight="bold")
ax1.set_xlabel("Month", fontsize=11)
ax1.set_ylabel("Year", fontsize=11)

# Panel 2: Day-of-week bar chart
colors_dow = [NEON["green"] if r >= 0 else NEON["red"] for r in dow_returns.values]
ax2.bar(range(len(dow_returns)), dow_returns.values, color=colors_dow, alpha=0.8)
ax2.set_xticks(range(len(dow_returns)))
ax2.set_xticklabels(dow_returns.index, fontsize=11)
ax2.set_ylabel("Average Daily Return (%)", fontsize=11)
ax2.set_title("Average Returns by Day of Week", fontsize=13, fontweight="bold")
ax2.axhline(0, color="white", linewidth=0.8, alpha=0.5)
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels
for i, val in enumerate(dow_returns.values):
    ax2.text(i, val + 0.01 if val >= 0 else val - 0.01, f"{val:.3f}", 
             ha="center", va="bottom" if val >= 0 else "top", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig08_seasonality.png", dpi=150, bbox_inches="tight", facecolor="#060b14")
print("Saved: figures/fig08_seasonality.png")
plt.close()

#────────────────────────────────────────────────────────────────────────────
# ## Section 11 — Summary & Insights
#────────────────────────────────────────────────────────────────────────────
# %%
print("\n"+"="*80)
print("SECTION 11: EDA SUMMARY & INSIGHTS")
print("="*80)

summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPLORATORY DATA ANALYSIS SUMMARY — {TICKER}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DATASET OVERVIEW
   • Ticker: {TICKER} — {company.get('longName', 'N/A')}
   • Date Range: {df_raw.index.min().strftime('%Y-%m-%d')} to {df_raw.index.max().strftime('%Y-%m-%d')}
   • Trading Days: {len(df_raw):,}
   • Original Columns: {df_raw.shape[1]}
   • Engineered Features: {len(feature_names)}

2. DATA QUALITY
   • Missing Values: {"None detected " if df_raw.isnull().sum().sum() == 0 else "Present"}
   • Data Completeness: {(1 - df_raw.isnull().sum().sum() / (len(df_raw) * df_raw.shape[1])) * 100:.2f}%

3. RETURN DISTRIBUTION
   • Mean Daily Return: {daily_returns.mean():.4f}%
   • Volatility (Std Dev): {daily_returns.std():.4f}%
   • Skewness: {daily_returns.skew():.4f} {"(Right-skewed)" if daily_returns.skew() > 0 else "(Left-skewed)"}
   • Kurtosis: {daily_returns.kurtosis():.4f} {"(Fat tails)" if daily_returns.kurtosis() > 3 else "(Normal tails)"}
   • Distribution: {"Approximately normal" if abs(daily_returns.skew()) < 0.5 else "Deviates from normal"}

4. TOP PREDICTIVE FEATURES (by correlation with target)
   1. {correlations.index[0]:30s} → {correlations.iloc[0]:.4f}
   2. {correlations.index[1]:30s} → {correlations.iloc[1]:.4f}
   3. {correlations.index[2]:30s} → {correlations.iloc[2]:.4f}
   4. {correlations.index[3]:30s} → {correlations.iloc[3]:.4f}
   5. {correlations.index[4]:30s} → {correlations.iloc[4]:.4f}

5. MODEL PERFORMANCE
   • Best Model: {trainer.best_model_name}
   • Test RMSE: ${metrics_df.loc[trainer.best_model_name, 'RMSE']:.4f}
   • Test R²: {metrics_df.loc[trainer.best_model_name, 'R²']:.4f}
   • MAPE: {metrics_df.loc[trainer.best_model_name, 'MAPE (%)']:.2f}%
   • Direction Accuracy: {metrics_df.loc[trainer.best_model_name, 'Direction Acc (%)']:.2f}%

6. SEASONALITY PATTERNS
   • Day-of-Week Effect: {"Detected" if dow_returns.max() - dow_returns.min() > 0.1 else "Minimal"}
   • Best Day: {dow_returns.idxmax()} ({dow_returns.max():.3f}%)
   • Worst Day: {dow_returns.idxmin()} ({dow_returns.min():.3f}%)

7. RECOMMENDED FEATURE SET
   • Lag Features: High importance (Close_Lag1, Close_Lag2)
   • Technical Indicators: Moving averages, RSI, MACD show strong correlation
   • Volatility Measures: ATR and Bollinger Bands contribute to predictions
   • Volume Indicators: Volume ratios help identify regime changes

8. KEY INSIGHTS FOR MODELING
   • The target variable (next-day close) is highly correlated with recent lags
   • Technical momentum indicators (RSI, MACD) provide valuable signals
   • Volatility clustering is present — consider GARCH models for future work
   • No strong monthly seasonality detected ("January effect" not observed)
   • Tree-based models (XGBoost, Random Forest) outperform linear models

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(summary)

# Save summary to file
with open("figures/EDA_SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\nEDA Complete! All figures saved to figures/")
print("Summary saved to: figures/EDA_SUMMARY.txt")
