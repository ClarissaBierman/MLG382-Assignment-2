"""
feature_engineering.py  — CRISP-DM Phase 3: Data Preparation
══════════════════════════════════════════════════════════════
OWNER:  Role 2 — Data Engineer & Preprocessing Specialist
STATUS: SKELETON — implement all TODO sections below.

Your job is to implement each technical indicator from scratch using
pandas / numpy only (no ta-lib dependency). Each function already has
its signature, docstring, and return type defined. Fill in the bodies.

When done, your functions will be called automatically by the Dash app
(via prepare_features) to build the feature matrix for the ML models.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Moving Averages
# TODO (Data Engineer): implement _sma and _ema.
# Hint: pd.Series.rolling().mean() and pd.Series.ewm().mean()
# ─────────────────────────────────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average over `window` periods."""
    # TODO: implement
    raise NotImplementedError("_sma not implemented — Data Engineer task")


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average with given span (adjust=False)."""
    # TODO: implement
    raise NotImplementedError("_ema not implemented — Data Engineer task")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Momentum Indicators
# TODO (Data Engineer): implement each indicator below.
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss.
    Values > 70 = overbought, < 30 = oversold.
    """
    # TODO: implement using price diff, clip gains/losses, rolling mean
    raise NotImplementedError("_rsi not implemented — Data Engineer task")


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Moving Average Convergence Divergence.
    Returns: (macd_line, signal_line, histogram)
      macd_line   = EMA(fast) - EMA(slow)
      signal_line = EMA(macd_line, span=signal)
      histogram   = macd_line - signal_line
    """
    # TODO: implement using _ema
    raise NotImplementedError("_macd not implemented — Data Engineer task")


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands.
    Returns: (upper, mid, lower)
      mid   = SMA(window)
      upper = mid + num_std * rolling_std(window)
      lower = mid - num_std * rolling_std(window)
    """
    # TODO: implement
    raise NotImplementedError("_bollinger_bands not implemented — Data Engineer task")


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         window: int = 14) -> pd.Series:
    """
    Average True Range.
    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = rolling mean of TR over `window`.
    """
    # TODO: implement — shift close by 1 for previous close
    raise NotImplementedError("_atr not implemented — Data Engineer task")


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.
    Add volume when price closes up, subtract when closes down. Cumsum.
    """
    # TODO: implement using np.sign(close.diff()) * volume
    raise NotImplementedError("_obv not implemented — Data Engineer task")


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_window: int = 14, d_window: int = 3):
    """
    Stochastic Oscillator.
    %K = 100 * (close - lowest_low(k_window)) / (highest_high - lowest_low)
    %D = SMA(%K, d_window)
    Returns: (%K, %D)
    """
    # TODO: implement
    raise NotImplementedError("_stochastic not implemented — Data Engineer task")


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                window: int = 14) -> pd.Series:
    """
    Williams %R.
    = -100 * (highest_high - close) / (highest_high - lowest_low)
    Range: -100 (oversold) to 0 (overbought).
    """
    # TODO: implement
    raise NotImplementedError("_williams_r not implemented — Data Engineer task")


def _cci(high: pd.Series, low: pd.Series, close: pd.Series,
         window: int = 20) -> pd.Series:
    """
    Commodity Channel Index.
    Typical Price = (H + L + C) / 3
    CCI = (TP - SMA(TP, window)) / (0.015 * mean_abs_deviation)
    """
    # TODO: implement
    raise NotImplementedError("_cci not implemented — Data Engineer task")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Full feature builder
# TODO (Data Engineer): call all the indicator functions above and assign
# the results as new columns on the DataFrame copy.
# Use the exact column naming convention shown in the docstring.
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ~40 technical indicator columns to an OHLCV DataFrame.

    Required input columns:  Open, High, Low, Close, Volume
    Returns a new DataFrame (does not modify the original).

    Expected output columns (must match exactly for app.py to work):
        SMA_5, SMA_10, SMA_20, SMA_50, SMA_100, SMA_200
        EMA_5, EMA_10, EMA_20, EMA_50, EMA_100, EMA_200
        Price_vs_SMA5, Price_vs_SMA20, Price_vs_SMA50, Price_vs_SMA200
            (formula: close / SMA - 1)
        GoldenCross  (int: 1 if SMA_50 > SMA_200 else 0)
        RSI_14, RSI_7
        MACD, MACD_Signal, MACD_Hist
        Stoch_K, Stoch_D
        Williams_R
        CCI
        ROC_1, ROC_5, ROC_10, ROC_20   (close.pct_change(n) * 100)
        BB_Upper, BB_Mid, BB_Lower
        BB_Width  = (upper - lower) / mid
        BB_Pct    = (close - lower) / (upper - lower)
        ATR_14
        ATR_Norm  = ATR_14 / close
        HV_21     = 21-day rolling std of log returns * sqrt(252)
        OBV
        Volume_SMA20
        Volume_Ratio = Volume / Volume_SMA20
        HL_Range     = (High - Low) / Close
        OC_Range     = (Close - Open) / Open
        Upper_Shadow = (High - max(Close,Open)) / Close
        Lower_Shadow = (min(Close,Open) - Low) / Close
        Daily_Return = Close.pct_change()
    """
    df = df.copy()
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    # TODO: implement all columns listed above
    # Example of first few to get you started:
    # df["SMA_5"]  = _sma(c, 5)
    # df["SMA_20"] = _sma(c, 20)
    # ...

    raise NotImplementedError(
        "add_technical_indicators not implemented — Data Engineer task.\n"
        "Implement all indicator helper functions in SECTION 2 first, "
        "then fill in this function to build the full feature set."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Lag & time features  (already implemented — do not modify)
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     target_col: str = "Close",
                     lags: list = None) -> pd.DataFrame:
    """Add lagged values of target_col."""
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20]
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_Lag{lag}"] = df[target_col].shift(lag)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclically encode day-of-week, month, quarter."""
    df = df.copy()
    idx = df.index
    df["DayOfWeek"]  = idx.dayofweek
    df["Month"]      = idx.month
    df["Quarter"]    = idx.quarter
    df["DayOfYear"]  = idx.dayofyear
    df["DOW_sin"]    = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
    df["DOW_cos"]    = np.cos(2 * np.pi * df["DayOfWeek"] / 5)
    df["Month_sin"]  = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]  = np.cos(2 * np.pi * df["Month"] / 12)
    df["WeekOfYear"] = idx.isocalendar().week.values.astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Master pipeline  (do not modify — called by app.py and EDA)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame,
                     target_col: str = "Close",
                     forecast_horizon: int = 1) -> tuple:
    """
    Full pipeline: indicators → lags → time features → target → dropna.
    Returns (X: DataFrame, y: Series, feature_names: list).
    """
    df = add_technical_indicators(df)
    df = add_lag_features(df, target_col)
    df = add_time_features(df)
    df["Target"] = df[target_col].shift(-forecast_horizon)
    df.dropna(inplace=True)

    exclude = {"Open", "High", "Low", "Close", "Volume", "Target",
               "Adj Close", "Dividends", "Stock Splits"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols], df["Target"], feature_cols


def get_feature_groups() -> dict:
    return {
        "Moving Averages": ["SMA_", "EMA_"],
        "Momentum":        ["RSI", "MACD", "Stoch", "Williams", "CCI", "ROC"],
        "Volatility":      ["BB_", "ATR", "HV"],
        "Volume":          ["OBV", "Volume"],
        "Price Action":    ["HL_Range", "OC_Range", "Shadow", "Return"],
        "Lag Features":    ["_Lag"],
        "Time Features":   ["DOW", "Month", "Quarter", "Week", "Day"],
        "Market Context":  ["VIX", "SP500"],
    }
