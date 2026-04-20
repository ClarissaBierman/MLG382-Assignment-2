"""
app.py  ─  StockOracle: AI-Powered Stock Price Prediction Dashboard
─────────────────────────────────────────────────────────────────────
Run:  python app.py
Then open:  http://127.0.0.1:8050
"""

import warnings
warnings.filterwarnings("ignore")

import json
import traceback
from datetime import datetime, timedelta
from clean_stock_data import clean_stock_data_for_dashboard

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

# Local modules
from utils.data_fetcher import (
    load_full_dataset, get_company_info,
    format_market_cap, format_percentage, get_default_date_range,
    load_static_dataset
)
from utils.feature_engineering import prepare_features, add_technical_indicators
from models.ml_models import StockModelTrainer, MODEL_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# Dash App Init
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="StockOracle — Prediction Studio",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server   # For Gunicorn / Render deployment

# ─────────────────────────────────────────────────────────────────────────────
# Chart / Color Palette
# ─────────────────────────────────────────────────────────────────────────────

THEME = dict(
    bg          = "#060b14",
    paper       = "#0f1a2e",
    grid        = "#1a2d4a",
    text        = "#dce8f5",
    text_dim    = "#6b8aad",
    accent1     = "#00c8ff",
    accent2     = "#7b2fff",
    accent3     = "#00e5a0",
    accent4     = "#ffc440",
    negative    = "#ff4d6d",
    candle_up   = "#00e5a0",
    candle_down = "#ff4d6d",
    font        = "JetBrains Mono, Space Grotesk, monospace",
)

POPULAR_TICKERS = [
    {"label": "Apple (AAPL)",          "value": "AAPL"},
    {"label": "Microsoft (MSFT)",      "value": "MSFT"},
    {"label": "NVIDIA (NVDA)",         "value": "NVDA"},
    {"label": "Alphabet (GOOGL)",      "value": "GOOGL"},
    {"label": "Amazon (AMZN)",         "value": "AMZN"},
    {"label": "Tesla (TSLA)",          "value": "TSLA"},
    {"label": "Meta (META)",           "value": "META"},
    {"label": "Berkshire (BRK-B)",     "value": "BRK-B"},
    {"label": "JPMorgan (JPM)",        "value": "JPM"},
    {"label": "S&P 500 ETF (SPY)",     "value": "SPY"},
    {"label": "Bitcoin ETF (IBIT)",    "value": "IBIT"},
]

DEFAULT_START, DEFAULT_END = get_default_date_range()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: base plotly layout
# ─────────────────────────────────────────────────────────────────────────────

def base_layout(height=420):
    return dict(
        height=height,
        paper_bgcolor=THEME["paper"],
        plot_bgcolor=THEME["bg"],
        font=dict(family=THEME["font"], color=THEME["text"], size=12),
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=THEME["grid"],
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor=THEME["grid"], gridwidth=0.5,
            showgrid=True, zeroline=False,
            color=THEME["text_dim"], linecolor=THEME["grid"],
        ),
        yaxis=dict(
            gridcolor=THEME["grid"], gridwidth=0.5,
            showgrid=True, zeroline=False,
            color=THEME["text_dim"], linecolor=THEME["grid"],
        ),
        hoverlabel=dict(
            bgcolor=THEME["paper"], bordercolor=THEME["grid"],
            font=dict(family=THEME["font"], size=12, color=THEME["text"]),
        ),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div([

    # ── Hidden stores ─────────────────────────────────────────────────────────
    dcc.Store(id="store-ohlcv",    storage_type="memory"),
    dcc.Store(id="store-features", storage_type="memory"),
    dcc.Store(id="store-models",   storage_type="memory"),
    dcc.Store(id="store-company",  storage_type="memory"),
    dcc.Store(id="analysis-done", storage_type="memory"),

    html.Div(className="page-wrapper", children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(className="app-header", children=[
            html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"flex-start"}, children=[
                html.Div([
                    html.Div("BC ANALYTICS", className="header-badge"),
                    html.H1("StockOracle — Prediction Studio", className="header-title"),
                    html.P(
                        "Multi-model ML forecasting with technical indicators, sentiment signals, "
                        "and market fundamentals. Select a ticker and click Run Analysis.",
                        className="header-sub"
                    ),
                ]),
                html.Div([
                    html.Span(className="status-badge live", children="LIVE DATA"),
                    html.Div(
                        id="live-clock",
                        style={"fontFamily":"JetBrains Mono","fontSize":"12px",
                               "color":"#6b8aad","marginTop":"8px","textAlign":"right"}
                    ),
                ], style={"textAlign":"right"}),
            ]),
        ]),

        # ── Control Bar ────────────────────────────────────────────────────────
        html.Div(className="control-bar", children=[

            html.Div(className="control-group", style={"flex":"1","minWidth":"200px"}, children=[
                html.Label("Ticker Symbol", className="control-label"),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=POPULAR_TICKERS,
                    value="AAPL",
                    clearable=False,
                    searchable=True,
                    style={"minWidth":"220px"},
                ),
            ]),

            html.Div(className="control-group", children=[
                html.Label("Data Source", className="control-label"),
                dcc.Dropdown(
                    id="data-source",
                    options=[
                        {"label": "Live (Yahoo Finance)", "value": "live"},
                        {"label": "Static CSV Snapshot", "value": "static"},
                    ],
                    value="live",  # default
                    clearable=False,
                    style={"minWidth":"180px"},
                ),
            ]),

            html.Div(className="control-group", children=[
                html.Label("Date Range", className="control-label"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=DEFAULT_START,
                    end_date=DEFAULT_END,
                    display_format="YYYY-MM-DD",
                    style={"fontSize":"13px"},
                ),
            ]),

            html.Div(className="control-group", children=[
                html.Label("Forecast Horizon", className="control-label"),
                dcc.Dropdown(
                    id="forecast-horizon",
                    options=[
                        {"label": "Next Day (1d)",    "value": 1},
                        {"label": "Next Week (5d)",   "value": 5},
                        {"label": "Next Month (21d)", "value": 21},
                    ],
                    value=1,
                    clearable=False,
                    style={"minWidth":"180px"},
                ),
            ]),

            html.Div(className="control-group", children=[
                html.Label("Future Forecast", className="control-label"),
                dcc.Dropdown(
                    id="future-days",
                    options=[
                        {"label": "7 days",  "value": 7},
                        {"label": "14 days", "value": 14},
                        {"label": "30 days", "value": 30},
                        {"label": "60 days", "value": 60},
                    ],
                    value=30,
                    clearable=False,
                    style={"minWidth":"130px"},
                ),
            ]),

            html.Div(style={"marginLeft":"auto","paddingTop":"18px"}, children=[
                html.Button(
                    "Run Analysis",
                    id="run-btn",
                    n_clicks=0,
                    className="predict-btn",
                ),
            ]),
        ]),

        html.Div(id="custom-loader", className="custom-loader"),
        html.Div(id="kpi-row", className="kpi-row"),
        html.Div(id="error-alert"),

        # ── Tabs ───────────────────────────────────────────────────────────────
        dcc.Tabs(
            id="main-tabs",
            value="tab-overview",
            className="custom-tabs",
            children=[
                dcc.Tab(label="Market Overview",    value="tab-overview"),
                dcc.Tab(label="ML Predictions",     value="tab-predict"),
                dcc.Tab(label="Technical Analysis", value="tab-technical"),
                dcc.Tab(label="Model Insights",     value="tab-models"),
                dcc.Tab(label="Fundamentals",       value="tab-fundamentals"),
            ],
        ),

        # Tab content area
        html.Div(id="tab-content"),

    ]),

    # Live clock interval
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),

    # Loader interval (for progress circle)
    dcc.Interval(id="loader-interval", interval=300, n_intervals=0, disabled=True),

], id="root-div", style={"background": THEME["bg"], "minHeight": "100vh"})


# ─────────────────────────────────────────────────────────────────────────────
# Callback: live clock
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(Output("live-clock","children"),
              Input("clock-interval","n_intervals"))
def update_clock(_):
    now = datetime.now()
    return now.strftime("%A, %d %b %Y  |  %H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Callback: fetch data & train models on button click
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-ohlcv", "data"),
    Output("store-features", "data"),
    Output("store-models", "data"),
    Output("store-company", "data"),
    Output("kpi-row", "children"),
    Output("error-alert", "children"),
    Output("analysis-done", "data"),
    Input("run-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("forecast-horizon", "value"),
    State("future-days", "value"),
    State("data-source", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, ticker, start_date, end_date, horizon, future_days, data_source):
    if not n_clicks:
        return [None]*4 + [[], None, False]
    try:
        # 1. Fetch data (choose source)
        if data_source == "static":
            df = load_static_dataset()
            company = get_company_info(ticker)
        else:
            # Use teammate’s cleaner instead of raw loader
            df = clean_stock_data_for_dashboard(ticker, start_date, end_date)
            company = get_company_info(ticker)

        # 2. Feature engineering
        X, y, feat_names = prepare_features(df, forecast_horizon=int(horizon))

        # 3. Train models
        trainer = StockModelTrainer(test_size=0.2, n_cv_splits=5)
        trainer.train_all(X, y)

        # 4. Build forecast
        last_features = X.iloc[-1].values
        forecast = trainer.forecast_future(last_features, n_days=int(future_days))

        # 5. Serialise everything for dcc.Store
        df_ind = add_technical_indicators(df.copy())

        ohlcv_store = df.reset_index().to_json(date_format="iso", orient="split")
        df_ind_store = df_ind.reset_index().to_json(date_format="iso", orient="split")

        model_data = {
            "best":   trainer.best_model_name,
            "metrics": trainer.get_metrics_df().reset_index().to_json(orient="split"),
            "preds_test":  {n: p.tolist() for n, p in trainer.test_predictions.items()},
            "preds_train": {n: p.tolist() for n, p in trainer.train_predictions.items()},
            "y_test":   trainer.y_test.tolist(),
            "y_train":  trainer.y_train.tolist(),
            "test_dates":  [str(d) for d in trainer.y_test.index],
            "train_dates": [str(d) for d in trainer.y_train.index],
            "feature_names": feat_names,
            "feature_importances": {
                n: fi.to_dict() for n, fi in trainer.feature_importances.items()
            },
            "cv_scores": {n: v.tolist() for n, v in trainer.cv_scores.items()},
            "forecast": forecast.tolist(),
            "future_days": int(future_days),
            "horizon": int(horizon),
        }

        # 6. KPI cards
        last_row  = df.iloc[-1]
        prev_row  = df.iloc[-2] if len(df) > 1 else last_row
        curr_close = float(last_row["Close"])
        prev_close = float(prev_row["Close"])
        daily_chg  = (curr_close - prev_close) / prev_close * 100
        chg_class  = "positive" if daily_chg >= 0 else "negative"
        chg_sign   = "▲" if daily_chg >= 0 else "▼"

        year_data = df.tail(252)
        hi52 = float(year_data["High"].max())
        lo52 = float(year_data["Low"].min())

        kpi_defs = [
            ("CURRENT PRICE",   f"${curr_close:,.2f}",  f"{chg_sign} {abs(daily_chg):.2f}%", chg_class),
            ("1-DAY CHANGE",    f"{daily_chg:+.2f}%",   f"vs prev close ${prev_close:,.2f}", chg_class),
            ("52W HIGH",        f"${hi52:,.2f}",         "", "positive"),
            ("52W LOW",         f"${lo52:,.2f}",         "", "negative"),
            ("VOLUME",          f"{int(last_row['Volume']):,}", "", ""),
            ("MARKET CAP",      format_market_cap(company.get("marketCap")), "", ""),
            ("P/E RATIO",       f"{company.get('trailingPE') or 'N/A'}", "", ""),
            ("BEST MODEL",      trainer.best_model_name.split()[0],
             f"RMSE ${trainer.metrics[trainer.best_model_name]['RMSE']:.2f}", "positive"),
        ]

        kpi_cards = [
            html.Div(className="kpi-card", children=[
                html.Div(label, className="kpi-label"),
                html.Div(value, className="kpi-value"),
                html.Div(sub,   className=f"kpi-change {sub_cls}") if sub else html.Span(),
            ]) for label, value, sub, sub_cls in kpi_defs
        ]

        return (ohlcv_store, df_ind_store, json.dumps(model_data),
                json.dumps(company), kpi_cards, None, True)

    except Exception as e:
        err = html.Div(className="info-alert", children=[
            html.Span("⚠", style={"fontSize":"18px","color":"#ffc440"}),
            html.Div([
                html.Strong("Error loading data: ", style={"color":"#ffc440"}),
                html.Span(str(e), style={"color":"#dce8f5"}),
                html.Br(),
                html.Small(traceback.format_exc()[-300:], style={"color":"#6b8aad","fontSize":"11px"}),
            ]),
        ])
        return [None]*4 + [[], err, False]


# ─────────────────────────────────────────────────────────────────────────────
# Callback: render tab content
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("store-ohlcv", "data"),     # NEW
    Input("store-features", "data"),  # NEW
    Input("store-models", "data"),    # NEW
    Input("store-company", "data"),   # NEW
)
def render_tab(tab, ohlcv_json, feat_json, model_json, company_json):
    if not ohlcv_json:
        return html.Div(className="loading-text", children=[
            html.Div("No data loaded yet.", style={"fontSize":"16px","marginBottom":"8px"}),
            html.Div("Select a ticker and click Run Analysis to begin.", style={"color":"#3a5270"}),
        ])

    df   = pd.read_json(ohlcv_json, orient="split").set_index("Date")
    df_i = pd.read_json(feat_json, orient="split").set_index("Date")
    md   = json.loads(model_json)
    co   = json.loads(company_json)

    if tab == "tab-overview":
        return build_overview_tab(df, df_i, co)
    elif tab == "tab-predict":
        return build_prediction_tab(df, md)
    elif tab == "tab-technical":
        return build_technical_tab(df_i)
    elif tab == "tab-models":
        return build_models_tab(md)
    elif tab == "tab-fundamentals":
        return build_fundamentals_tab(co, df)
    return html.Div("Select a tab.")

# ─────────────────────────────────────────────────────────────────────────────
# Callback: reset loader interval AND analysis-done flag on button click
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("analysis-done", "data", allow_duplicate=True),
    Output("loader-interval", "disabled"),
    Output("loader-interval", "n_intervals"),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_loader(n_clicks):
    return False, False, 0

# ─────────────────────────────────────────────────────────────────────────────
# Callback: render loading circle loader
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("custom-loader", "children"),
    Output("loader-interval", "disabled", allow_duplicate=True),
    Input("loader-interval", "n_intervals"),
    State("analysis-done", "data"),
    prevent_initial_call=True,
)
def show_loader(n_intervals, analysis_done):
    if analysis_done:
        return None, True

    # Animate up to 97% so it never falsely hits 100% before done
    progress = min(n_intervals * 3, 97)
    loader = html.Div([
        html.Div(className="loader-ring-wrap", children=[
            html.Div(className="orbit-loader",
                     children=[html.Div() for _ in range(24)]),
            html.Div(className="loader-center-text", children=[
                html.Div(f"{progress}%", className="loader-progress-num"),
                html.Div("Analysing", className="loader-progress-label"),
            ]),
        ]),
        # Status line below the ring
        html.Div("Processing data  ·  training models", className="loader-subtext"),
    ], className="loader-wrapper")

    return loader, False
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Market Overview
# ─────────────────────────────────────────────────────────────────────────────

def build_overview_tab(df: pd.DataFrame, df_i: pd.DataFrame, co: dict):
    # Candlestick + volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=THEME["candle_up"],
        decreasing_line_color=THEME["candle_down"],
        increasing_fillcolor=THEME["candle_up"],
        decreasing_fillcolor=THEME["candle_down"],
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    # SMA 50 & 200
    if "SMA_50" in df_i.columns:
        fig.add_trace(go.Scatter(
            x=df_i.index, y=df_i["SMA_50"],
            name="SMA 50", line=dict(color=THEME["accent4"], width=1.2, dash="dot"),
        ), row=1, col=1)
    if "SMA_200" in df_i.columns:
        fig.add_trace(go.Scatter(
            x=df_i.index, y=df_i["SMA_200"],
            name="SMA 200", line=dict(color=THEME["accent2"], width=1.2, dash="dot"),
        ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df_i.columns:
        fig.add_trace(go.Scatter(
            x=df_i.index, y=df_i["BB_Upper"],
            name="BB Upper", line=dict(color=THEME["accent1"], width=0.8, dash="dash"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_i.index, y=df_i["BB_Lower"],
            name="BB Lower", line=dict(color=THEME["accent1"], width=0.8, dash="dash"),
            fill="tonexty", fillcolor="rgba(0,200,255,0.04)", showlegend=True,
        ), row=1, col=1)

    # Volume bars
    colors_vol = [THEME["candle_up"] if r >= 0 else THEME["candle_down"]
                  for r in df["Close"].pct_change().fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors_vol, showlegend=False,
    ), row=2, col=1)

    lo = base_layout(550)
    lo.update({"xaxis2": lo["xaxis"], "yaxis2": lo["yaxis"],
                "xaxis_rangeslider_visible": False})
    fig.update_layout(**lo)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, title_font=dict(size=11))
    fig.update_yaxes(title_text="Volume",    row=2, col=1, title_font=dict(size=11))

    # Daily returns distribution
    ret = df["Close"].pct_change().dropna() * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=ret, nbinsx=80, name="Daily Return",
        marker_color=THEME["accent1"], opacity=0.7,
    ))
    fig2.add_vline(x=ret.mean(), line_dash="dash", line_color=THEME["accent4"],
                   annotation_text=f"Mean: {ret.mean():.2f}%")
    fig2.update_layout(**base_layout(300),
                       xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                       bargap=0.05)

    # Rolling 30-day volatility
    vol_30 = ret.rolling(30).std() * np.sqrt(252)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=vol_30.index, y=vol_30,
        name="Annualised Vol", fill="tozeroy",
        line=dict(color=THEME["accent2"], width=1.5),
        fillcolor="rgba(123,47,255,0.12)",
    ))
    fig3.update_layout(**base_layout(280),
                       yaxis_title="Annualised Volatility", xaxis_title="")

    return html.Div([
        html.Div(className="chart-card", children=[
            html.Div("OHLCV Chart + Bollinger Bands", className="chart-card-title"),
            dcc.Graph(figure=fig, config={"displayModeBar": True}),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("Daily Returns Distribution", className="chart-card-title"),
                dcc.Graph(figure=fig2, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("30-Day Rolling Annualised Volatility", className="chart-card-title"),
                dcc.Graph(figure=fig3, config={"displayModeBar": False}),
            ]),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML Predictions
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_tab(df: pd.DataFrame, md: dict):
    test_dates  = pd.to_datetime(md["test_dates"])
    train_dates = pd.to_datetime(md["train_dates"])
    y_test  = np.array(md["y_test"])
    y_train = np.array(md["y_train"])
    best    = md["best"]
    horizon = md["horizon"]
    fut_n   = md["future_days"]

    # ── Fig 1: All models vs actual on test set ───────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates, y=y_test,
        name="Actual", line=dict(color=THEME["text"], width=2),
    ))
    for name, preds in md["preds_test"].items():
        fig.add_trace(go.Scatter(
            x=test_dates, y=preds,
            name=name,
            line=dict(color=MODEL_COLORS.get(name, "#aaa"), width=1.5, dash="dot"),
            opacity=0.85,
        ))
    fig.update_layout(**base_layout(420),
                      title=dict(text=f"Model Predictions vs Actual  |  Test Set  |  Horizon: {horizon}d",
                                 font=dict(size=13, color=THEME["text_dim"])),
                      yaxis_title="Close Price ($)")

    # ── Fig 2: Best model + confidence band ───────────────────────────────────
    best_pred = np.array(md["preds_test"][best])
    residuals = y_test - best_pred
    std_err   = residuals.std()

    fig2 = go.Figure()
    # Train actual
    fig2.add_trace(go.Scatter(
        x=train_dates, y=y_train,
        name="Train Actual", line=dict(color=THEME["text_dim"], width=1),
        opacity=0.5,
    ))
    # Train shade
    fig2.add_vrect(
        x0=str(train_dates[0]), x1=str(train_dates[-1]),
        fillcolor="rgba(0,200,255,0.03)", line_width=0, layer="below",
        annotation_text="TRAINING", annotation_position="top left",
        annotation_font_size=10, annotation_font_color=THEME["text_dim"],
    )
    # Test actual
    fig2.add_trace(go.Scatter(
        x=test_dates, y=y_test,
        name="Test Actual", line=dict(color=THEME["text"], width=2),
    ))
    # Best model prediction
    fig2.add_trace(go.Scatter(
        x=test_dates, y=best_pred,
        name=f"{best} (Best)", line=dict(color=THEME["accent1"], width=2),
    ))
    # Confidence band ±1 std
    fig2.add_trace(go.Scatter(
        x=list(test_dates) + list(test_dates[::-1]),
        y=list(best_pred + std_err) + list((best_pred - std_err)[::-1]),
        fill="toself", fillcolor="rgba(0,200,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ Band", showlegend=True,
    ))

    # Future forecast
    last_date = pd.Timestamp(md["test_dates"][-1])
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=fut_n)
    forecast_vals = np.array(md["forecast"])

    fig2.add_trace(go.Scatter(
        x=future_dates, y=forecast_vals,
        name=f"Forecast ({fut_n}d)",
        line=dict(color=THEME["accent3"], width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=4),
    ))
    # Forecast confidence cone
    fig2.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(forecast_vals + std_err * 1.5) + list((forecast_vals - std_err * 1.5)[::-1]),
        fill="toself", fillcolor="rgba(0,229,160,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Forecast Band",
    ))

    fig2.update_layout(**base_layout(480),
                       title=dict(text=f"{best} — Full Timeline + {fut_n}-Day Forecast",
                                  font=dict(size=13, color=THEME["text_dim"])),
                       yaxis_title="Close Price ($)")

    # ── Fig 3: Prediction error over time ─────────────────────────────────────
    errors = y_test - best_pred
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=test_dates, y=errors,
        marker_color=[THEME["candle_up"] if e >= 0 else THEME["candle_down"] for e in errors],
        name="Prediction Error",
    ))
    fig3.add_hline(y=0, line_color=THEME["text_dim"], line_width=1)
    fig3.update_layout(**base_layout(280), yaxis_title="Error ($)", xaxis_title="")

    # ── Fig 4: Scatter actual vs predicted ────────────────────────────────────
    fig4 = go.Figure()
    mn = min(y_test.min(), best_pred.min())
    mx = max(y_test.max(), best_pred.max())
    fig4.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", name="Perfect Fit",
        line=dict(color=THEME["text_dim"], dash="dash", width=1.5),
    ))
    fig4.add_trace(go.Scatter(
        x=y_test, y=best_pred,
        mode="markers", name="Predictions",
        marker=dict(color=THEME["accent1"], size=4, opacity=0.6,
                    line=dict(color=THEME["accent2"], width=0.5)),
    ))
    fig4.update_layout(**base_layout(320),
                       xaxis_title="Actual Price ($)",
                       yaxis_title="Predicted Price ($)")

    return html.Div([
        html.Div(className="chart-card", children=[
            html.Div("All Models — Test Set Comparison", className="chart-card-title"),
            dcc.Graph(figure=fig, config={"displayModeBar": True}),
        ]),
        html.Div(className="chart-card", children=[
            html.Div(f"Best Model ({best}) — Historical + Future Forecast", className="chart-card-title"),
            dcc.Graph(figure=fig2, config={"displayModeBar": True}),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("Prediction Error Over Time", className="chart-card-title"),
                dcc.Graph(figure=fig3, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("Actual vs Predicted Scatter", className="chart-card-title"),
                dcc.Graph(figure=fig4, config={"displayModeBar": False}),
            ]),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Technical Analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_technical_tab(df: pd.DataFrame):
    # ── Sub-plot grid: Price + RSI + MACD + Stoch ─────────────────────────────
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        vertical_spacing=0.03,
        subplot_titles=("Price + EMA", "RSI (14)", "MACD", "Stochastic Oscillator"),
    )

    # Price + EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"],
                              name="Close", line=dict(color=THEME["text"], width=1.5)),  row=1, col=1)
    for span, col in [(12, THEME["accent1"]), (26, THEME["accent4"]), (50, THEME["accent2"])]:
        col_name = f"EMA_{span}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name],
                                     name=f"EMA {span}",
                                     line=dict(color=col, width=1, dash="dot")), row=1, col=1)

    # Bollinger Bands on price
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
                                  name="BB Upper", line=dict(color="rgba(0,200,255,0.4)", width=0.8),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
                                  name="BB", line=dict(color="rgba(0,200,255,0.4)", width=0.8),
                                  fill="tonexty", fillcolor="rgba(0,200,255,0.04)"), row=1, col=1)

    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"],
                                  name="RSI 14", line=dict(color=THEME["accent3"], width=1.5)),
                       row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=THEME["negative"],
                       line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=THEME["accent3"],
                       line_width=0.8, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,77,109,0.05)",
                       line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,229,160,0.05)",
                       line_width=0, row=2, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                                  name="MACD", line=dict(color=THEME["accent1"], width=1.5)),
                       row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
                                  name="Signal", line=dict(color=THEME["accent4"], width=1.2, dash="dot")),
                       row=3, col=1)
        hist_colors = [THEME["candle_up"] if v >= 0 else THEME["candle_down"]
                       for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
                              name="Histogram", marker_color=hist_colors,
                              showlegend=False), row=3, col=1)

    # Stochastic
    if "Stoch_K" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"],
                                  name="%K", line=dict(color=THEME["accent2"], width=1.5)),
                       row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"],
                                  name="%D", line=dict(color=THEME["accent4"], width=1.2, dash="dot")),
                       row=4, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color=THEME["negative"],   line_width=0.8, row=4, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color=THEME["accent3"],    line_width=0.8, row=4, col=1)

    lo = base_layout(820)
    lo["margin"] = dict(l=60, r=20, t=50, b=40)
    lo["xaxis_rangeslider_visible"] = False
    fig.update_layout(**lo)
    fig.update_annotations(font=dict(size=11, color=THEME["text_dim"]))

    # ── ATR + OBV side by side ─────────────────────────────────────────────────
    fig_atr = go.Figure()
    if "ATR_14" in df.columns:
        fig_atr.add_trace(go.Scatter(
            x=df.index, y=df["ATR_14"],
            fill="tozeroy", name="ATR(14)",
            line=dict(color=THEME["accent4"], width=1.5),
            fillcolor="rgba(255,196,64,0.1)",
        ))
    fig_atr.update_layout(**base_layout(280), yaxis_title="ATR ($)")

    fig_obv = go.Figure()
    if "OBV" in df.columns:
        fig_obv.add_trace(go.Scatter(
            x=df.index, y=df["OBV"],
            fill="tozeroy", name="OBV",
            line=dict(color=THEME["accent2"], width=1.5),
            fillcolor="rgba(123,47,255,0.1)",
        ))
    fig_obv.update_layout(**base_layout(280), yaxis_title="OBV")

    # ── Williams %R + CCI ─────────────────────────────────────────────────────
    fig_wir = go.Figure()
    if "Williams_R" in df.columns:
        fig_wir.add_trace(go.Scatter(
            x=df.index, y=df["Williams_R"],
            name="Williams %R", line=dict(color=THEME["accent1"], width=1.5),
        ))
        fig_wir.add_hline(y=-20, line_dash="dash", line_color=THEME["negative"],  line_width=0.8)
        fig_wir.add_hline(y=-80, line_dash="dash", line_color=THEME["accent3"],   line_width=0.8)
    fig_wir.update_layout(**base_layout(280), yaxis_title="Williams %R")

    fig_cci = go.Figure()
    if "CCI" in df.columns:
        fig_cci.add_trace(go.Scatter(
            x=df.index, y=df["CCI"],
            name="CCI(20)", line=dict(color=THEME["accent3"], width=1.5),
            fill="tozeroy", fillcolor="rgba(0,229,160,0.07)",
        ))
        fig_cci.add_hline(y=100,  line_dash="dash", line_color=THEME["negative"], line_width=0.8)
        fig_cci.add_hline(y=-100, line_dash="dash", line_color=THEME["accent3"],  line_width=0.8)
    fig_cci.update_layout(**base_layout(280), yaxis_title="CCI")

    return html.Div([
        html.Div(className="chart-card", children=[
            html.Div("Multi-Indicator Technical Dashboard", className="chart-card-title"),
            dcc.Graph(figure=fig, config={"displayModeBar": True}),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("Average True Range (ATR-14)", className="chart-card-title"),
                dcc.Graph(figure=fig_atr, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("On-Balance Volume (OBV)", className="chart-card-title"),
                dcc.Graph(figure=fig_obv, config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("Williams %R  (Overbought > -20  |  Oversold < -80)", className="chart-card-title"),
                dcc.Graph(figure=fig_wir, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("Commodity Channel Index (CCI-20)", className="chart-card-title"),
                dcc.Graph(figure=fig_cci, config={"displayModeBar": False}),
            ]),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Model Insights
# ─────────────────────────────────────────────────────────────────────────────

def build_models_tab(md: dict):
    best = md["best"]
    metrics_df = pd.read_json(md["metrics"], orient="split")
    fi_data    = md["feature_importances"]
    cv_data    = md["cv_scores"]

    # ── Best model highlight ───────────────────────────────────────────────────
    best_metrics = metrics_df[metrics_df["Model"] == best].iloc[0]
    best_card = html.Div(className="best-model-card", children=[
        html.Div(best, style={"fontFamily":"Syne","fontSize":"18px","fontWeight":"800","color":"#dce8f5"}),
        html.Div(style={"display":"flex","gap":"32px","marginTop":"10px"}, children=[
            html.Div([html.Div("RMSE",  style={"fontSize":"10px","color":"#6b8aad","letterSpacing":"1px"}),
                      html.Div(f"${best_metrics['RMSE']:.4f}", style={"fontSize":"18px","fontFamily":"JetBrains Mono","color":"#00c8ff"})]),
            html.Div([html.Div("MAE",   style={"fontSize":"10px","color":"#6b8aad","letterSpacing":"1px"}),
                      html.Div(f"${best_metrics['MAE']:.4f}",  style={"fontSize":"18px","fontFamily":"JetBrains Mono","color":"#00e5a0"})]),
            html.Div([html.Div("R²",    style={"fontSize":"10px","color":"#6b8aad","letterSpacing":"1px"}),
                      html.Div(f"{best_metrics['R²']:.4f}",    style={"fontSize":"18px","fontFamily":"JetBrains Mono","color":"#7b2fff"})]),
            html.Div([html.Div("MAPE",  style={"fontSize":"10px","color":"#6b8aad","letterSpacing":"1px"}),
                      html.Div(f"{best_metrics['MAPE (%)']:.2f}%", style={"fontSize":"18px","fontFamily":"JetBrains Mono","color":"#ffc440"})]),
            html.Div([html.Div("DIR ACC", style={"fontSize":"10px","color":"#6b8aad","letterSpacing":"1px"}),
                      html.Div(f"{best_metrics['Direction Acc (%)']:.1f}%", style={"fontSize":"18px","fontFamily":"JetBrains Mono","color":"#00e5a0"})]),
        ]),
    ])

    # ── Model comparison table ─────────────────────────────────────────────────
    cols = [{"name": c, "id": c,
             "type": "numeric", "format": {"specifier": ".4f"}}
            for c in metrics_df.columns]
    cols[0]["type"] = "text"

    comparison_table = dash_table.DataTable(
        data=metrics_df.to_dict("records"),
        columns=cols,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor":"#132038","color":"#00c8ff","fontFamily":"JetBrains Mono",
                       "fontSize":"12px","fontWeight":"600","border":"1px solid #1a2d4a"},
        style_cell={"backgroundColor":"#0f1a2e","color":"#dce8f5","fontFamily":"JetBrains Mono",
                    "fontSize":"13px","border":"1px solid #1a2d4a","padding":"10px 14px"},
        style_data_conditional=[
            {"if": {"filter_query": f"{{Model}} = '{best}'"},
             "backgroundColor": "rgba(0,200,255,0.08)", "color": "#00c8ff"},
        ],
    )

    # ── Grouped bar: RMSE & R² ─────────────────────────────────────────────────
    fig_bar = go.Figure()
    model_names = metrics_df["Model"].tolist()
    rmse_vals   = metrics_df["RMSE"].tolist()
    r2_vals     = metrics_df["R²"].tolist()

    fig_bar.add_trace(go.Bar(
        x=model_names, y=rmse_vals, name="RMSE",
        marker_color=[MODEL_COLORS.get(n, "#aaa") for n in model_names],
        yaxis="y", offsetgroup=1,
    ))
    fig_bar.add_trace(go.Bar(
        x=model_names, y=r2_vals, name="R²",
        marker_color=[MODEL_COLORS.get(n, "#aaa") for n in model_names],
        opacity=0.5, yaxis="y2", offsetgroup=2,
    ))
    lo = base_layout(340)
    lo["yaxis2"] = dict(title="R²", overlaying="y", side="right",
                        color=THEME["text_dim"], showgrid=False, gridcolor=THEME["grid"])
    lo["yaxis"]["title"] = "RMSE ($)"
    lo["barmode"] = "group"
    fig_bar.update_layout(**lo)

    # ── CV scores box plot ─────────────────────────────────────────────────────
    fig_cv = go.Figure()
    for name, scores in cv_data.items():
        fig_cv.add_trace(go.Box(
            y=scores, name=name,
            marker_color=MODEL_COLORS.get(name, "#aaa"),
            line_color=MODEL_COLORS.get(name, "#aaa"),
            boxpoints="all", jitter=0.3, pointpos=-1.8,
        ))
    fig_cv.update_layout(**base_layout(340), yaxis_title="Cross-Val R²")

    # ── Feature importance for best model ─────────────────────────────────────
    fi_best = fi_data.get(best, {})
    if fi_best:
        fi_series = pd.Series(fi_best).sort_values(ascending=False).head(20)
        max_val = fi_series.max() or 1
        fi_bars = html.Div([
            html.Div(className="fi-bar-row", children=[
                html.Div(name, className="fi-bar-name"),
                html.Div(className="fi-bar-track", children=[
                    html.Div(className="fi-bar-fill",
                             style={"width": f"{val/max_val*100:.1f}%"}),
                ]),
                html.Div(f"{val:.4f}", className="fi-bar-val"),
            ]) for name, val in fi_series.items()
        ])
    else:
        fi_bars = html.Div("Feature importances not available for this model.",
                           style={"color": THEME["text_dim"]})

    # ── Feature importance bar chart ───────────────────────────────────────────
    if fi_best:
        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            x=fi_series.values, y=fi_series.index,
            orientation="h", name="Importance",
            marker=dict(
                color=fi_series.values,
                colorscale=[[0, THEME["accent2"]], [0.5, THEME["accent1"]], [1, THEME["accent3"]]],
                showscale=False,
            ),
        ))
        lo_fi = base_layout(500)
        lo_fi["margin"] = dict(l=200, r=20, t=30, b=40)
        lo_fi["yaxis"]["autorange"] = "reversed"
        fig_fi.update_layout(**lo_fi, xaxis_title="Feature Importance")
    else:
        fig_fi = go.Figure()

    return html.Div([
        best_card,
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("Model Performance Comparison (RMSE & R²)", className="chart-card-title"),
                dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("Walk-Forward Cross-Validation R² (5-Fold)", className="chart-card-title"),
                dcc.Graph(figure=fig_cv, config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("All Models — Metrics Table", className="chart-card-title"),
            comparison_table,
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div(f"Top 20 Feature Importances — {best}", className="chart-card-title"),
                fi_bars,
            ]),
            html.Div(className="chart-card", children=[
                html.Div(f"Feature Importance Chart — {best}", className="chart-card-title"),
                dcc.Graph(figure=fig_fi, config={"displayModeBar": False}),
            ]),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Market Regimes
# ─────────────────────────────────────────────────────────────────────────────
def build_regimes_tab(df_i: pd.DataFrame, md: dict):
    """
    K-Means (k=3) market regime segmentation built directly from the
    indicators DataFrame.  No external clustering.py required.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    CLUSTER_COLORS = {0: "#00e5a0", 1: "#ffc440", 2: "#ff4d6d"}
    CLUSTER_LABELS = {0: "Bullish / Low-Vol", 1: "High-Vol / Uncertain", 2: "Bearish / Drawdown"}

    # ── Build feature matrix for clustering ───────────────────────────────────
    feat_cols = []
    df_c = df_i.copy()

    # Daily return
    df_c["_ret"]   = df_c["Close"].pct_change() * 100
    feat_cols.append("_ret")

    # Short & long historical volatility
    log_r = np.log(df_c["Close"] / df_c["Close"].shift(1))
    df_c["_hv5"]  = log_r.rolling(5).std()  * np.sqrt(252) * 100
    df_c["_hv21"] = log_r.rolling(21).std() * np.sqrt(252) * 100
    feat_cols += ["_hv5", "_hv21"]

    # Volume ratio (use pre-computed or compute fallback)
    if "Volume_Ratio" in df_c.columns:
        df_c["_volr"] = df_c["Volume_Ratio"]
    else:
        vsma = df_c["Volume"].rolling(20).mean()
        df_c["_volr"] = df_c["Volume"] / (vsma + 1e-9)
    feat_cols.append("_volr")

    # RSI if available
    if "RSI_14" in df_c.columns:
        df_c["_rsi"] = df_c["RSI_14"]
        feat_cols.append("_rsi")

    # Bollinger %B if available
    if "BB_Pct" in df_c.columns:
        df_c["_bbp"] = df_c["BB_Pct"]
        feat_cols.append("_bbp")

    # VIX if available
    if "VIX" in df_c.columns:
        df_c["_vix"] = df_c["VIX"]
        feat_cols.append("_vix")

    feat_df = df_c[feat_cols].dropna()
    if len(feat_df) < 30:
        return html.Div(className="chart-card",
                        style={"textAlign":"center","padding":"60px 20px"},
                        children=[html.Div("Not enough data for clustering (need 30+ rows).",
                                           style={"color": THEME["text_dim"]})])

    # ── Fit K-Means ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(feat_df.values)
    km     = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw_labels = km.fit_predict(X_sc)

    # ── Re-order clusters by mean return (0=best, 2=worst) ───────────────────
    mean_ret_per_cluster = {c: feat_df["_ret"].values[raw_labels == c].mean()
                            for c in range(3)}
    sorted_clusters = sorted(mean_ret_per_cluster, key=mean_ret_per_cluster.get, reverse=True)
    remap = {old: new for new, old in enumerate(sorted_clusters)}
    labels = np.array([remap[l] for l in raw_labels])

    # Silhouette
    try:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X_sc, labels)
        sil_str = f"{sil:.3f}"
    except Exception:
        sil_str = "N/A"

    # ── Elbow analysis ────────────────────────────────────────────────────────
    inertias, sil_scores, ks = [], [], list(range(2, 9))
    for k in ks:
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc)
        inertias.append(km_k.inertia_)
        try:
            from sklearn.metrics import silhouette_score as ss
            sil_scores.append(ss(X_sc, km_k.labels_))
        except Exception:
            sil_scores.append(0)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                                    name="Inertia", line=dict(color=THEME["accent1"], width=2),
                                    marker=dict(size=7)))
    fig_elbow.add_trace(go.Scatter(x=ks, y=sil_scores, mode="lines+markers",
                                    name="Silhouette", yaxis="y2",
                                    line=dict(color=THEME["accent3"], width=2, dash="dot"),
                                    marker=dict(size=7)))
    lo_e = base_layout(300)
    lo_e["yaxis2"] = dict(title="Silhouette", overlaying="y", side="right",
                          color=THEME["text_dim"], showgrid=False)
    lo_e["yaxis"]["title"] = "Inertia"
    lo_e["xaxis"]["title"] = "k (number of clusters)"
    fig_elbow.add_vline(x=3, line_dash="dash", line_color=THEME["accent4"],
                         annotation_text="k=3 chosen", annotation_font_size=11)
    fig_elbow.update_layout(**lo_e,
        title=dict(text="Elbow Analysis — Justifying k=3",
                   font=dict(size=13, color=THEME["text_dim"])))

    # ── Colour-coded price chart ──────────────────────────────────────────────
    close_aligned = df_c["Close"].loc[feat_df.index]
    fig_price = go.Figure()
    for cid in [0, 1, 2]:
        mask = labels == cid
        dates_c  = feat_df.index[mask]
        prices_c = close_aligned.loc[dates_c]
        fig_price.add_trace(go.Scatter(
            x=dates_c, y=prices_c,
            mode="markers", name=CLUSTER_LABELS[cid],
            marker=dict(color=CLUSTER_COLORS[cid], size=4, opacity=0.75),
        ))
    # Underlying price line
    fig_price.add_trace(go.Scatter(
        x=close_aligned.index, y=close_aligned,
        mode="lines", name="Close",
        line=dict(color="rgba(220,232,245,0.2)", width=1),
        showlegend=False,
    ))
    fig_price.update_layout(**base_layout(380),
        title=dict(text="Price Coloured by Market Regime",
                   font=dict(size=13, color=THEME["text_dim"])),
        yaxis_title="Close Price ($)")

    # ── Scatter: volatility vs return coloured by cluster ────────────────────
    fig_scat = go.Figure()
    for cid in [0, 1, 2]:
        mask = labels == cid
        fig_scat.add_trace(go.Scatter(
            x=feat_df["_hv21"].values[mask],
            y=feat_df["_ret"].values[mask],
            mode="markers", name=CLUSTER_LABELS[cid],
            marker=dict(color=CLUSTER_COLORS[cid], size=5, opacity=0.6,
                        line=dict(width=0.3, color="rgba(0,0,0,0.3)")),
        ))
    fig_scat.update_layout(**base_layout(360),
        xaxis_title="21-Day Historical Volatility (%)",
        yaxis_title="Daily Return (%)",
        title=dict(text="Volatility vs Return by Regime",
                   font=dict(size=13, color=THEME["text_dim"])))

    # ── Cluster stats table ───────────────────────────────────────────────────
    rows = []
    for cid in [0, 1, 2]:
        mask = labels == cid
        rets  = feat_df["_ret"].values[mask]
        vols  = feat_df["_hv21"].values[mask]
        rows.append({
            "Cluster":      f"{cid}  —  {CLUSTER_LABELS[cid]}",
            "Days":         int(mask.sum()),
            "% of Data":    f"{mask.mean()*100:.1f}%",
            "Mean Return":  f"{rets.mean():+.3f}%",
            "Mean HV-21":   f"{vols.mean():.1f}%",
            "Max Return":   f"{rets.max():+.2f}%",
            "Min Return":   f"{rets.min():+.2f}%",
        })
    stats_table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0].keys()],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor":"#132038","color":"#00c8ff","fontFamily":"JetBrains Mono",
                       "fontSize":"12px","fontWeight":"600","border":"1px solid #1a2d4a"},
        style_cell={"backgroundColor":"#0f1a2e","color":"#dce8f5","fontFamily":"JetBrains Mono",
                    "fontSize":"12px","border":"1px solid #1a2d4a","padding":"10px 14px"},
        style_data_conditional=[
            {"if": {"filter_query": '{Cluster} contains "0"'}, "color": "#00e5a0"},
            {"if": {"filter_query": '{Cluster} contains "1"'}, "color": "#ffc440"},
            {"if": {"filter_query": '{Cluster} contains "2"'}, "color": "#ff4d6d"},
        ],
    )

    # ── Regime distribution donut ─────────────────────────────────────────────
    counts = [int((labels == c).sum()) for c in range(3)]
    fig_donut = go.Figure(go.Pie(
        labels=[CLUSTER_LABELS[c] for c in range(3)],
        values=counts,
        hole=0.55,
        marker=dict(colors=[CLUSTER_COLORS[c] for c in range(3)],
                    line=dict(color=THEME["bg"], width=2)),
        textfont=dict(family=THEME["font"], size=12, color=THEME["text"]),
    ))
    fig_donut.update_layout(**base_layout(320),
        title=dict(text=f"Regime Distribution  |  Silhouette: {sil_str}",
                   font=dict(size=13, color=THEME["text_dim"])),
        showlegend=True)

    return html.Div([
        html.Div(className="chart-card", children=[
            html.Div("Price History Coloured by Market Regime", className="chart-card-title"),
            dcc.Graph(figure=fig_price, config={"displayModeBar": True}),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("Volatility vs Return by Regime (Scatter)", className="chart-card-title"),
                dcc.Graph(figure=fig_scat, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("Regime Distribution", className="chart-card-title"),
                dcc.Graph(figure=fig_donut, config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Elbow Analysis — Choosing k=3", className="chart-card-title"),
            dcc.Graph(figure=fig_elbow, config={"displayModeBar": False}),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Cluster Summary Statistics", className="chart-card-title"),
            stats_table,
        ]),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — SHAP Insights
# ─────────────────────────────────────────────────────────────────────────────

def build_shap_tab(md: dict):
    """
    Model interpretability tab using feature importances from the trained
    models. Displays global importance rankings, per-model comparisons, and
    feature group breakdowns. No shap package required.
    """
    best = md.get("best", "N/A")
    fi_data  = md.get("feature_importances", {})
    feat_names = md.get("feature_names", [])

    if not fi_data:
        return html.Div(className="chart-card",
                        style={"textAlign":"center","padding":"60px 20px"},
                        children=[html.Div("Feature importance data not available. Run analysis first.",
                                           style={"color": THEME["text_dim"]})])

    # ── Fig 1: Top-20 global importance bar chart for best model ──────────────
    fi_best = fi_data.get(best, {})
    if fi_best:
        fi_series = pd.Series(fi_best).sort_values(ascending=False).head(20)
        fig_global = go.Figure()
        fig_global.add_trace(go.Bar(
            x=fi_series.values,
            y=fi_series.index,
            orientation="h",
            marker=dict(
                color=fi_series.values,
                colorscale=[[0, THEME["accent2"]], [0.4, THEME["accent1"]], [1, THEME["accent3"]]],
                showscale=False,
            ),
            name="Importance",
        ))
        lo_g = base_layout(520)
        lo_g["margin"] = dict(l=220, r=20, t=30, b=40)
        lo_g["yaxis"]["autorange"] = "reversed"
        fig_global.update_layout(**lo_g,
            xaxis_title="Feature Importance",
            title=dict(text=f"Top 20 Feature Importances — {best}",
                       font=dict(size=13, color=THEME["text_dim"])))
    else:
        fig_global = go.Figure()

    # ── Fig 2: Feature group importance breakdown (pie) ───────────────────────
    group_map = {
        "Moving Averages": ["SMA_", "EMA_", "Price_vs", "GoldenCross"],
        "Momentum":        ["RSI", "MACD", "Stoch", "Williams", "CCI", "ROC"],
        "Volatility":      ["BB_", "ATR", "HV"],
        "Volume":          ["OBV", "Volume"],
        "Price Action":    ["HL_Range", "OC_Range", "Shadow", "Daily_Return"],
        "Lag Features":    ["_Lag"],
        "Time Features":   ["DOW", "Month", "Quarter", "Week", "DayOf"],
        "Market Context":  ["VIX", "SP500"],
    }
    group_colors = ["#00c8ff","#7b2fff","#00e5a0","#ffc440","#ff4d6d","#a78bfa","#34d399","#f87171"]

    if fi_best:
        fi_all = pd.Series(fi_best)
        group_totals = {}
        for grp, prefixes in group_map.items():
            mask = fi_all.index.to_series().apply(
                lambda n: any(p in n for p in prefixes))
            group_totals[grp] = float(fi_all[mask].sum())

        fig_pie = go.Figure(go.Pie(
            labels=list(group_totals.keys()),
            values=list(group_totals.values()),
            hole=0.5,
            marker=dict(colors=group_colors, line=dict(color=THEME["bg"], width=2)),
            textfont=dict(family=THEME["font"], size=11, color=THEME["text"]),
        ))
        fig_pie.update_layout(**base_layout(340),
            title=dict(text="Importance by Feature Group",
                       font=dict(size=13, color=THEME["text_dim"])))
    else:
        fig_pie = go.Figure()

    # ── Fig 3: All models side-by-side — top-10 features ─────────────────────
    tree_models = [n for n in fi_data if fi_data[n]]
    if tree_models:
        # Collect union of top-10 features across all models
        all_top = set()
        for name in tree_models:
            s = pd.Series(fi_data[name]).sort_values(ascending=False).head(10)
            all_top.update(s.index.tolist())
        all_top = list(all_top)[:15]

        fig_compare = go.Figure()
        for name in tree_models:
            fi_s = pd.Series(fi_data[name])
            vals = [fi_s.get(f, 0) for f in all_top]
            fig_compare.add_trace(go.Bar(
                name=name, x=all_top, y=vals,
                marker_color=MODEL_COLORS.get(name, "#aaa"),
                opacity=0.85,
            ))
        lo_c = base_layout(380)
        lo_c["barmode"] = "group"
        lo_c["xaxis"]["tickangle"] = -35
        fig_compare.update_layout(**lo_c,
            yaxis_title="Feature Importance",
            title=dict(text="Model Comparison — Top Features Side by Side",
                       font=dict(size=13, color=THEME["text_dim"])))
    else:
        fig_compare = go.Figure()

    # ── Fig 4: Cumulative importance curve (how many features capture 80%) ────
    if fi_best:
        fi_sorted = pd.Series(fi_best).sort_values(ascending=False)
        cumsum    = fi_sorted.cumsum() / fi_sorted.sum() * 100
        idx_80    = int((cumsum < 80).sum()) + 1
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=list(range(1, len(cumsum)+1)), y=cumsum.values,
            mode="lines", fill="tozeroy",
            line=dict(color=THEME["accent1"], width=2),
            fillcolor="rgba(0,200,255,0.08)",
            name="Cumulative Importance",
        ))
        fig_cum.add_hline(y=80, line_dash="dash", line_color=THEME["accent4"],
                           annotation_text="80% threshold")
        fig_cum.add_vline(x=idx_80, line_dash="dash", line_color=THEME["accent4"],
                           annotation_text=f"{idx_80} features",
                           annotation_position="top right")
        fig_cum.update_layout(**base_layout(300),
            xaxis_title="Number of Features (ranked)",
            yaxis_title="Cumulative Importance (%)",
            title=dict(text=f"Cumulative Importance — {idx_80} features explain 80%",
                       font=dict(size=13, color=THEME["text_dim"])))
    else:
        fig_cum = go.Figure()

    # ── Insight summary card ──────────────────────────────────────────────────
    if fi_best:
        top3 = pd.Series(fi_best).sort_values(ascending=False).head(3)
        top3_items = [
            html.Div(style={"display":"flex","justifyContent":"space-between",
                            "padding":"6px 0","borderBottom":f"1px solid {THEME['grid']}"}, children=[
                html.Span(f"#{i+1}  {feat}", style={"fontFamily":"JetBrains Mono","fontSize":"12px",
                                                      "color":THEME["accent1"]}),
                html.Span(f"{val:.4f}", style={"fontFamily":"JetBrains Mono","fontSize":"12px",
                                                "color":THEME["text_dim"]}),
            ]) for i, (feat, val) in enumerate(top3.items())
        ]
        insight_card = html.Div(className="best-model-card", style={"marginBottom":"20px"}, children=[
            html.Div(f"Top Predictors for {best}",
                     style={"fontFamily":"Syne","fontSize":"16px","fontWeight":"800",
                            "color":THEME["text"],"marginBottom":"12px"}),
            html.Div("The following features have the highest predictive impact:", 
                     style={"color":THEME["text_dim"],"fontSize":"12px","marginBottom":"10px"}),
            *top3_items,
            html.Div(f"Total features used: {len(fi_best)}",
                     style={"marginTop":"10px","fontSize":"11px","color":THEME["text_muted"],
                            "fontFamily":"JetBrains Mono"}),
        ])
    else:
        insight_card = html.Span()

    return html.Div([
        insight_card,
        html.Div(className="chart-card", children=[
            html.Div(f"Global Feature Importance — {best} (Top 20)", className="chart-card-title"),
            dcc.Graph(figure=fig_global, config={"displayModeBar": False}),
        ]),
        html.Div(className="two-col", children=[
            html.Div(className="chart-card", children=[
                html.Div("Importance by Feature Category", className="chart-card-title"),
                dcc.Graph(figure=fig_pie, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                html.Div("Cumulative Importance Curve", className="chart-card-title"),
                dcc.Graph(figure=fig_cum, config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("All Models — Feature Importance Comparison", className="chart-card-title"),
            dcc.Graph(figure=fig_compare, config={"displayModeBar": True}),
        ]),
    ])
# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Fundamentals
# ─────────────────────────────────────────────────────────────────────────────

def build_fundamentals_tab(co: dict, df: pd.DataFrame):
    def _stat(label, value, color=None):
        return html.Div(
            style={"display":"flex","justifyContent":"space-between","alignItems":"center",
                   "padding":"10px 0","borderBottom":f"1px solid {THEME['grid']}"},
            children=[
                html.Span(label, style={"color": THEME["text_dim"], "fontSize": "12px",
                                         "fontFamily": "JetBrains Mono"}),
                html.Span(str(value) if value is not None else "N/A",
                          style={"color": color or THEME["text"],
                                 "fontFamily": "JetBrains Mono", "fontWeight": "600",
                                 "fontSize": "13px"}),
            ],
        )

    pe  = co.get("trailingPE")
    fpe = co.get("forwardPE")
    pb  = co.get("priceToBook")
    dy  = co.get("dividendYield")
    eps = co.get("trailingEps")
    bet = co.get("beta")
    rec = co.get("recommendationKey", "N/A")
    tgt = co.get("targetMeanPrice")

    fundamentals_card = html.Div(className="chart-card", style={"gridColumn":"span 1"}, children=[
        html.Div(f"{co.get('longName', 'N/A')}", className="chart-card-title"),
        html.Div([
            html.Span(co.get("sector",""), style={"color": THEME["accent1"], "fontSize":"12px",
                                                   "fontFamily":"JetBrains Mono","marginRight":"8px"}),
            html.Span(co.get("industry",""), style={"color": THEME["text_dim"], "fontSize":"12px"}),
        ], style={"marginBottom":"12px"}),
        _stat("Market Cap",        format_market_cap(co.get("marketCap"))),
        _stat("Trailing P/E",      f"{pe:.2f}" if pe else "N/A",      THEME["accent4"] if pe else None),
        _stat("Forward P/E",       f"{fpe:.2f}" if fpe else "N/A"),
        _stat("Price / Book",      f"{pb:.2f}" if pb else "N/A"),
        _stat("EPS (Trailing)",    f"${eps:.2f}" if eps else "N/A"),
        _stat("Dividend Yield",    format_percentage(dy)),
        _stat("Beta",              f"{bet:.2f}" if bet else "N/A",     THEME["negative"] if bet and float(bet) > 1.5 else None),
        _stat("52W High",          f"${co.get('fiftyTwoWeekHigh', 'N/A')}"),
        _stat("52W Low",           f"${co.get('fiftyTwoWeekLow',  'N/A')}"),
        _stat("Target Price",      f"${tgt:.2f}" if tgt else "N/A",    THEME["accent3"]),
        _stat("Analyst Rating",    rec.upper() if rec else "N/A",       THEME["accent3"]),
    ])

    # Monthly seasonality heatmap
    df2 = df.copy()
    df2["Year"]  = df2.index.year
    df2["Month"] = df2.index.month
    df2["MRet"]  = df2["Close"].pct_change() * 100
    pivot = df2.groupby(["Year","Month"])["MRet"].mean().unstack(level=1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, THEME["negative"]], [0.5, THEME["paper"]], [1, THEME["accent3"]]],
        zmid=0, colorbar=dict(title="Avg %", tickfont=dict(size=10, color=THEME["text_dim"])),
        hoverongaps=False,
        hovertemplate="<b>%{x} %{y}</b><br>Avg Return: %{z:.2f}%<extra></extra>",
    ))
    fig_heat.update_layout(**base_layout(380), title=dict(
        text="Monthly Average Return Heatmap",
        font=dict(size=13, color=THEME["text_dim"])))

    # Year performance bar
    annual = df2.groupby("Year")["Close"].apply(lambda g: (g.iloc[-1]-g.iloc[0])/g.iloc[0]*100)
    fig_ann = go.Figure(go.Bar(
        x=annual.index, y=annual.values,
        marker_color=[THEME["candle_up"] if v >= 0 else THEME["candle_down"] for v in annual.values],
        name="Annual Return",
    ))
    fig_ann.add_hline(y=0, line_color=THEME["text_dim"], line_width=0.8)
    fig_ann.update_layout(**base_layout(300), yaxis_title="Return (%)")

    return html.Div([
        html.Div(className="two-col", children=[
            fundamentals_card,
            html.Div(className="chart-card", children=[
                html.Div("Annual Performance", className="chart-card-title"),
                dcc.Graph(figure=fig_ann, config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-card", children=[
            html.Div("Monthly Seasonality Heatmap", className="chart-card-title"),
            dcc.Graph(figure=fig_heat, config={"displayModeBar": False}),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
