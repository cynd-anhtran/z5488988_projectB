"""
Quantise — Systematic Multi-Asset Investment Dashboard
=======================================================
FINS3645 FinTech Project Part B  |  Station 4

A deployed investor-facing app offering three systematically managed funds
(Equity, Crypto, Combined) built from out-of-sample backtested optimal
portfolios, plus a news-sentiment analytics layer across 10 equity sectors.

Loads ONLY precomputed CSVs from results/ — no heavy scoring or backtest recomputation.
"""
from __future__ import annotations

import html
import pathlib
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

# ---------------------------------------------------------------------------
# Design system — teal / cyan fintech palette
# ---------------------------------------------------------------------------
DARK_BG      = "#1E2B1E"
DARK_GREEN   = "#0D1F0D"
TEAL         = "#4ABEB2"
TEAL_DARK    = "#2A9D8F"
TEAL_LIGHT   = "#5BC5B6"
CYAN         = "#4EE4C0"
CYAN_PALE    = "#F1FAF8"
WHITE        = "#FFFFFF"
GREY_50      = "#F9FAFB"
GREY_100     = "#F3F4F6"
GREY_200     = "#E5E7EB"
GREY_300     = "#D1D5DB"
GREY_500     = "#6B7280"
GREY_700     = "#374151"
GREY_900     = "#111827"
SUCCESS      = "#10B981"
DANGER       = "#EF4444"
WARN         = "#F59E0B"
INFO_BLUE    = "#3B82F6"

UNIVERSE_COLORS = {
    "Equity":   {"bg": "#DBEAFE", "text": "#1E40AF", "dot": "#3B82F6"},
    "Crypto":   {"bg": "#FDE68A", "text": "#92400E", "dot": "#F59E0B"},
    "Combined": {"bg": "#D1FAE5", "text": "#065F46", "dot": "#10B981"},
}
SECTOR_COLORS = {
    "Comm":        "#6366F1",
    "Consumer":    "#EC4899",
    "Energy":      "#F97316",
    "Financials":  "#14B8A6",
    "Healthcare":  "#8B5CF6",
    "Industrials": "#64748B",
    "Materials":   "#D97706",
    "RealEstate":  "#0EA5E9",
    "Tech":        "#22C55E",
    "Utilities":   "#A855F7",
}

STRATEGY_COLORS = {
    "Equal-weight (1/N)":    GREY_700,
    "Minimum-variance":      TEAL_DARK,
    "Max-Sharpe (tangency)": "#E63946",
    "Risk parity":           INFO_BLUE,
    "MinVar + sentiment":    WARN,
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_fund_returns():
    return pd.read_csv(DATA / "fund_returns.csv", parse_dates=["date"])

@st.cache_data
def load_fund_weights():
    return pd.read_csv(DATA / "fund_weights.csv", parse_dates=["date"])

@st.cache_data
def load_performance_metrics():
    return pd.read_csv(TABLES / "performance_metrics.csv")

@st.cache_data
def load_sector_sentiment():
    return pd.read_csv(DATA / "sector_sentiment_index.csv", parse_dates=["date"])

@st.cache_data
def load_headline_panel():
    pq_path = DATA / "headline_panel.parquet"
    csv_path = DATA / "headline_panel.csv"
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["trading_date"],
                           dtype={"publisher": str}, low_memory=False)
    return None

@st.cache_data
def load_tx_cost_comparison():
    path = TABLES / "tx_cost_comparison.csv"
    if path.exists():
        return pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    return None


@st.cache_data
def load_app_stats():
    """Derive landing-page claims from committed app artifacts."""
    weights = load_fund_weights()
    sentiment = load_sector_sentiment()
    headline_path = DATA / "headline_panel.parquet"
    headline_count = None
    if headline_path.exists():
        try:
            import pyarrow.parquet as pq
            headline_count = pq.ParquetFile(headline_path).metadata.num_rows
        except Exception:
            pass
    return {
        "assets": int(weights["ticker"].nunique()),
        "sectors": int(sentiment["sector"].nunique()),
        "headlines": headline_count,
    }


# ---------------------------------------------------------------------------
# Chart helper — polished with area fill option
# ---------------------------------------------------------------------------
def clean_fig(width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=GREY_500, labelsize=8, length=0)
    ax.grid(True, axis="y", color=GREY_200, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def universe_badge(uni: str) -> str:
    uni = html.escape(str(uni))
    c = UNIVERSE_COLORS.get(uni, {"bg": GREY_200, "text": GREY_700})
    return (f'<span style="background:{c["bg"]};color:{c["text"]};'
            f'padding:3px 12px;border-radius:12px;font-size:0.72rem;'
            f'font-weight:600;letter-spacing:0.02em;display:inline-block;">{uni}</span>')

def ticker_badge(ticker: str) -> str:
    """Hera.I-style letter-initial circle badge + teal ticker name."""
    ticker = html.escape(str(ticker))
    letter = ticker[0].upper() if ticker else "?"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:8px;">'
        f'<span style="width:26px;height:26px;border-radius:50%;'
        f'background:{DARK_BG};color:{CYAN};font-size:0.68rem;font-weight:700;'
        f'display:inline-flex;align-items:center;justify-content:center;'
        f'flex-shrink:0;">{letter}</span>'
        f'<span style="color:{TEAL_DARK};font-weight:600;font-size:0.82rem;">'
        f'{ticker}</span></span>'
    )

def sector_dot(sector: str) -> str:
    sector = html.escape(str(sector))
    c = SECTOR_COLORS.get(sector, GREY_500)
    return (f'<span style="display:inline-flex;align-items:center;gap:6px;">'
            f'<span style="width:8px;height:8px;border-radius:50%;'
            f'background:{c};flex-shrink:0;"></span>'
            f'<span>{sector}</span></span>')


# ---------------------------------------------------------------------------
# Styled HTML table — Hera.I reference style
# ---------------------------------------------------------------------------
def styled_html_table(df, highlight_col=None, highlight_max=True,
                      universe_col=None, sector_col=None, ticker_col=None,
                      fmt2_cols=None, max_height=None, compact=False):
    """Professional HTML table with colour-coded badges and clean borders."""
    fmt2_cols = fmt2_cols or []
    font_size = "0.78rem" if compact else "0.82rem"
    pad = "8px 12px" if compact else "11px 16px"

    best_idx = None
    if highlight_col and highlight_col in df.columns:
        try:
            vals = pd.to_numeric(df[highlight_col], errors="coerce")
            best_idx = vals.idxmax() if highlight_max else vals.idxmin()
        except Exception:
            pass

    # Header
    header = "".join(
        f'<th style="position:sticky;top:0;padding:{pad};text-align:left;'
        f'font-size:0.7rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:{GREY_500};background:{GREY_50};'
        f'border-bottom:2px solid {GREY_200};z-index:1;">{col}</th>'
        for col in (html.escape(str(c)) for c in df.columns)
    )

    # Rows
    rows = ""
    for i, (idx, row) in enumerate(df.iterrows()):
        is_best = (idx == best_idx) if best_idx is not None else False
        bg = CYAN_PALE if is_best else (WHITE if i % 2 == 0 else GREY_50)
        left_border = f"border-left:3px solid {TEAL};" if is_best else ""
        weight = "600" if is_best else "400"

        cells = ""
        for col in df.columns:
            val = row[col]
            style = (f'padding:{pad};font-size:{font_size};color:{GREY_900};'
                     f'font-weight:{weight};border-bottom:1px solid {GREY_100};'
                     f'white-space:nowrap;vertical-align:middle;')

            if universe_col and col == universe_col and val in UNIVERSE_COLORS:
                content = universe_badge(val)
            elif sector_col and col == sector_col and val in SECTOR_COLORS:
                content = sector_dot(val)
            elif ticker_col and col == ticker_col:
                content = ticker_badge(str(val))
            elif col in fmt2_cols:
                try:
                    content = f"{float(val):.2f}"
                except (ValueError, TypeError):
                    content = str(val)
            else:
                content = html.escape(str(val)) if pd.notna(val) else "—"

            cells += f'<td style="{style}">{content}</td>'

        rows += (
            f'<tr style="background:{bg};{left_border}'
            f'transition:background 0.12s;"'
            f' onmouseover="this.style.background=\'#F0FDFA\'"'
            f' onmouseout="this.style.background=\'{bg}\'"'
            f'>{cells}</tr>'
        )

    scroll = f"max-height:{max_height}px;overflow-y:auto;" if max_height else ""

    return (
        f'<div style="{scroll}border-radius:10px;'
        f'border:1px solid {GREY_200};margin-bottom:1rem;overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;'
        f'font-family:Inter,-apple-system,sans-serif;">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def apply_sort_controls(df, options, key, default_label):
    """Small, deployment-friendly sorting control for precomputed tables."""
    sort_c1, sort_c2 = st.columns([2, 1])
    labels = list(options)
    with sort_c1:
        label = st.selectbox("Sort by", labels, index=labels.index(default_label),
                             key=f"{key}_column")
    with sort_c2:
        descending = st.toggle("Highest / newest first", value=True,
                               key=f"{key}_descending")
    return df.sort_values(options[label], ascending=not descending,
                          na_position="last", kind="stable")


# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Quantise — Systematic Funds",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,600&family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #111827;
    }
    /* Force white background everywhere */
    .main, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"], [data-testid="stMain"], .block-container,
    [data-testid="stBottomBlockContainer"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stMain"] [data-testid="stVerticalBlock"] {
        background-color: transparent;
    }
    .main .block-container {
        padding-top: 1rem; padding-bottom: 2rem; max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E2B1E 0%, #0D1F0D 100%);
        border-right: 1px solid rgba(78,228,192,0.08);
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
    section[data-testid="stSidebar"] .stRadio label {
        border-radius: 8px; padding: 8px 12px; transition: all 0.2s ease;
        font-weight: 500; letter-spacing: 0.01em;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(78, 228, 192, 0.12);
        padding-left: 16px;
    }

    /* Typography */
    h1 { color: #111827 !important; font-weight: 800 !important;
         letter-spacing: -0.03em !important; font-size: 1.9rem !important; }
    h2, h3 { color: #111827 !important; font-weight: 700 !important; }

    .section-label {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.12em; color: #2A9D8F; margin-bottom: 4px;
        display: inline-block; position: relative;
    }
    .section-label::after {
        content: ""; display: block; width: 24px; height: 2px;
        background: #4ABEB2; margin-top: 4px; border-radius: 1px;
    }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 3px;
        letter-spacing: -0.01em;
    }
    .section-subtitle {
        font-size: 0.82rem; color: #6B7280; margin-bottom: 16px; line-height: 1.6;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #F9FAFB; border: 1px solid #E5E7EB;
        border-radius: 12px; padding: 16px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #4ABEB2;
        box-shadow: 0 3px 8px rgba(42,157,143,0.1);
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.7rem !important; font-weight: 600 !important;
        color: #6B7280 !important; text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important; font-weight: 800 !important;
        color: #111827 !important; letter-spacing: -0.02em;
    }

    /* Chart card wrapper */
    .chart-card {
        background: #FFFFFF; border: 1px solid #E5E7EB;
        border-radius: 12px; padding: 20px 20px 12px; margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .chart-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.03);
        border-color: #D1D5DB;
    }
    .chart-card-title {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.07em; color: #6B7280; margin-bottom: 12px;
    }

    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.9rem !important; }
    hr { border: none; border-top: 1px solid #E5E7EB; margin: 1.5rem 0; }

    .stDownloadButton > button {
        background: #1E2B1E; color: white; border: none;
        border-radius: 8px; font-weight: 600; font-size: 0.82rem; padding: 8px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: #2A9D8F; color: white;
        box-shadow: 0 4px 8px rgba(42,157,143,0.2);
        transform: translateY(-1px);
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #1E2B1E 0%, #0D3B35 50%, #2A9D8F 100%);
        border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(30,43,30,0.15), 0 2px 6px rgba(42,157,143,0.1);
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ""; position: absolute; top: -40%; right: -10%;
        width: 300px; height: 300px; border-radius: 50%;
        background: radial-gradient(circle, rgba(78,228,192,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero .hero-label {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.14em; color: #4EE4C0; margin-bottom: 6px;
    }
    .hero .hero-title {
        font-size: 1.3rem; font-weight: 700; margin-bottom: 18px;
        letter-spacing: -0.01em;
    }
    .hero .kpi-row { display: flex; gap: 24px; flex-wrap: wrap; }
    .hero .kpi-item {
        flex: 1; min-width: 140px; padding: 8px 0;
        border-left: 2px solid rgba(78,228,192,0.2); padding-left: 14px;
    }
    .hero .kpi-item:first-child { border-left: none; padding-left: 0; }
    .hero .kpi-item .kpi-val {
        font-size: 1.45rem; font-weight: 800; line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .hero .kpi-item .kpi-lbl {
        font-size: 0.65rem; text-transform: uppercase;
        letter-spacing: 0.08em; opacity: 0.65; margin-bottom: 2px; font-weight: 600;
    }
    .hero .kpi-item .kpi-sub { font-size: 0.7rem; opacity: 0.5; margin-top: 2px; }

    .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    div[data-testid="stAlert"] { border-radius: 10px; }

    /* Form submit buttons */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #1E2B1E 0%, #2A9D8F 100%);
        color: white; border: none; border-radius: 10px;
        font-weight: 700; font-size: 0.88rem; padding: 12px 24px;
        box-shadow: 0 3px 8px rgba(42,157,143,0.15);
        transition: all 0.25s ease; letter-spacing: 0.02em;
    }
    .stFormSubmitButton > button:hover {
        box-shadow: 0 6px 16px rgba(42,157,143,0.25);
        transform: translateY(-2px);
    }

    /* Slider track */
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        font-weight: 700; color: #2A9D8F;
    }

    /* Selectbox and multiselect polish */
    div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    /* Expander polish */
    .streamlit-expanderHeader {
        font-weight: 600 !important; font-size: 0.88rem !important;
        border-radius: 10px;
    }

    /* Smooth page dividers */
    hr { border: none; border-top: 1px solid #E5E7EB; margin: 1.8rem 0; }

    /* ===== LANDING PAGE ===== */
    .lp-hero-full {
        background-color:#06251F;
        background-image:
            linear-gradient(rgba(120,255,171,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(120,255,171,0.035) 1px, transparent 1px),
            linear-gradient(145deg, #05241E 0%, #06352B 66%, #0B4A3B 100%);
        background-size:40px 40px, 40px 40px, auto;
        border-radius: 20px; padding: 56px 48px 48px; color: white; margin-bottom: 2rem;
        position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(13,31,13,0.25), 0 2px 8px rgba(42,157,143,0.15);
    }
    .lp-hero-full::before {
        content: ""; position: absolute; top: -60%; right: -15%;
        width: 500px; height: 500px; border-radius: 50%;
        background: radial-gradient(circle, rgba(78,228,192,0.1) 0%, transparent 65%);
        pointer-events: none;
    }
    .lp-hero-full::after {
        content: ""; position: absolute; bottom: -48%; left: 6%;
        width: 88%; height: 280px; border-radius: 50%;
        background: radial-gradient(ellipse, rgba(78,228,192,0.30) 0%, transparent 68%);
        pointer-events: none;
    }
    .lp-trust { font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.14em; color: #4EE4C0; margin-bottom: 12px; }
    .lp-h1 { font-family: 'Space Grotesk', sans-serif;
        font-size: 2.65rem; font-weight: 700; line-height: 1.08;
        letter-spacing: -0.03em; margin-bottom: 14px; }
    .lp-serif-accent { font-family:'Cormorant Garamond', Georgia, serif;
        font-style:italic; font-weight:600; color:#4EE4C0; letter-spacing:-0.025em; }
    .lp-sub { font-size: 1.05rem; opacity: 0.75; line-height: 1.6; max-width: 560px;
        margin-bottom: 28px; }
    .lp-btn {
        display: inline-block; padding: 12px 28px; border-radius: 10px;
        font-weight: 700; font-size: 0.88rem; text-decoration: none;
        letter-spacing: 0.02em; cursor: pointer; transition: all 0.25s ease;
        border: none;
    }
    .lp-btn-primary { background: #4EE4C0; color: #0D1F0D; }
    .lp-btn-primary:hover { background: #3DD4B0; transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(78,228,192,0.3); }
    .lp-btn-ghost { background: transparent; color: white;
        border: 1.5px solid rgba(255,255,255,0.3); margin-left: 12px; }
    .lp-btn-ghost:hover { border-color: #4EE4C0; color: #4EE4C0; }

    .lp-stats-row { display: flex; gap: 32px; margin-top: 32px; flex-wrap: wrap; }
    .lp-stat { text-align: center; flex: 1; min-width: 100px; }
    .lp-stat-val { font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; }
    .lp-stat-lbl { font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.08em; opacity: 0.55; margin-top: 2px; font-weight: 600; }

    .lp-logos { display: flex; gap: 36px; align-items: center; justify-content: center;
        flex-wrap: wrap; padding: 24px 0; opacity: 0.55; }
    .lp-logos span { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.04em;
        color: #6B7280; }

    .lp-section { padding: 20px 0; }
    .lp-section-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.14em; color: #2A9D8F; text-align: center; margin-bottom: 6px; }
    .lp-section-title { font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem; font-weight: 800; color: #111827;
        text-align: center; letter-spacing: -0.02em; margin-bottom: 6px; }
    .lp-section-sub { font-size: 0.9rem; color: #6B7280; text-align: center;
        max-width: 520px; margin: 0 auto 32px; line-height: 1.6; }

    .lp-feature-card {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
        padding: 28px 24px; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.25s ease; height: 100%;
    }
    .lp-feature-card:hover {
        border-color: #2A9D8F; transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(42,157,143,0.12);
    }
    .feature-link { display:block; height:100%; color:inherit !important;
        text-decoration:none !important; cursor:pointer; }
    .feature-link:focus-visible { outline:3px solid rgba(74,190,178,.35);
        outline-offset:3px; border-radius:14px; }
    .lp-feature-icon { font-size: 2rem; margin-bottom: 14px; display: block; }
    .lp-feature-title { font-size: 1rem; font-weight: 700; color: #111827;
        margin-bottom: 8px; }
    .lp-feature-desc { font-size: 0.82rem; color: #6B7280; line-height: 1.6; }

    .lp-step { text-align: center; padding: 12px; }
    .lp-step-num {
        width: 40px; height: 40px; border-radius: 50%;
        background: linear-gradient(135deg, #1E2B1E 0%, #2A9D8F 100%);
        color: #4EE4C0; font-size: 1rem; font-weight: 800;
        display: inline-flex; align-items: center; justify-content: center;
        margin-bottom: 14px;
        box-shadow: 0 3px 8px rgba(42,157,143,0.2);
    }
    .lp-step-title { font-size: 0.95rem; font-weight: 700; color: #111827;
        margin-bottom: 6px; }
    .lp-step-desc { font-size: 0.82rem; color: #6B7280; line-height: 1.55; }

    .lp-plan-card {
        background: #FFFFFF; border: 1.5px solid #E5E7EB; border-radius: 16px;
        padding: 32px 28px; position: relative; transition: all 0.25s ease;
    }
    .lp-plan-card:hover {
        border-color: #2A9D8F;
        box-shadow: 0 8px 28px rgba(42,157,143,0.1);
    }
    .lp-plan-featured {
        border-color: #2A9D8F; background: #F0FDFA;
        box-shadow: 0 4px 16px rgba(42,157,143,0.1);
    }
    .lp-plan-badge {
        position: absolute; top: -12px; left: 50%; transform: translateX(-50%);
        background: linear-gradient(135deg, #1E2B1E, #2A9D8F);
        color: #4EE4C0; font-size: 0.62rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        padding: 4px 16px; border-radius: 20px;
    }
    .lp-plan-name { font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #6B7280; margin-bottom: 4px; }
    .lp-plan-price { font-size: 2rem; font-weight: 800; color: #111827;
        margin-bottom: 4px; letter-spacing: -0.03em; }
    .lp-plan-period { font-size: 0.78rem; color: #9CA3AF; margin-bottom: 20px; }
    .lp-plan-features { list-style: none; padding: 0; margin: 0; }
    .lp-plan-features li { font-size: 0.82rem; color: #374151; padding: 6px 0;
        border-bottom: 1px solid #F3F4F6; display: flex; align-items: center; gap: 8px; }
    .lp-plan-features li::before { content: "✓"; color: #2A9D8F; font-weight: 700; }
    .research-disclaimer {
        background:#F7F9F8; border:1px solid #DDE5E2; border-left:3px solid #2A9D8F;
        border-radius:10px; padding:13px 16px; color:#52605B; font-size:0.76rem;
        line-height:1.55; margin:16px 0;
    }

    .lp-faq-item {
        border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px 22px;
        margin-bottom: 10px; background: #FFFFFF;
        transition: border-color 0.2s ease;
    }
    .lp-faq-item:hover { border-color: #D1D5DB; }
    .lp-faq-q { font-size: 0.9rem; font-weight: 600; color: #111827; }
    .lp-faq-a { font-size: 0.82rem; color: #6B7280; line-height: 1.6; margin-top: 8px; }

    .lp-cta-bar {
        background: linear-gradient(135deg, #1E2B1E 0%, #0D3B35 50%, #2A9D8F 100%);
        border-radius: 16px; padding: 40px 36px; text-align: center; color: white;
        margin: 2rem 0;
        box-shadow: 0 6px 24px rgba(13,31,13,0.2);
    }
    .lp-cta-title { font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem; font-weight: 800; margin-bottom: 8px;
        letter-spacing: -0.02em; }
    .lp-cta-sub { font-size: 0.9rem; opacity: 0.7; margin-bottom: 24px; }

    .lp-footer {
        display: flex; gap: 48px; flex-wrap: wrap; padding: 32px 0 16px;
        border-top: 1px solid #E5E7EB; margin-top: 1rem;
    }
    .lp-footer-col { flex: 1; min-width: 140px; }
    .lp-footer-col h4 { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #111827; margin-bottom: 12px; }
    .lp-footer-col span { display: block; font-size: 0.82rem; color: #6B7280;
        padding: 3px 0; }
    .lp-footer-copy { font-size: 0.72rem; color: #9CA3AF; text-align: center;
        padding: 12px 0; }

    /* ===== NAVIGATION TOOLBAR ===== */
    .nav-toolbar {
        display: flex; align-items: center; gap: 0;
        background: #FFFFFF; border: 1px solid #E5E7EB;
        border-radius: 12px; padding: 4px 6px; margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .nav-toolbar .nav-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem; font-weight: 700; color: #2A9D8F;
        letter-spacing: -0.04em; padding: 6px 18px 6px 14px;
        border-right: 1px solid #E5E7EB; margin-right: 6px;
        line-height: 1;
    }
    .nav-toolbar .nav-link {
        display: inline-block; padding: 8px 18px; border-radius: 8px;
        font-size: 0.82rem; font-weight: 600; color: #6B7280;
        text-decoration: none; cursor: pointer; transition: all 0.2s ease;
        letter-spacing: 0.01em;
    }
    .nav-toolbar .nav-link:hover {
        background: #F0FDFA; color: #2A9D8F;
    }
    .nav-toolbar .nav-link.active {
        background: linear-gradient(135deg, #1E2B1E, #2A9D8F);
        color: #4EE4C0; font-weight: 700;
    }

    /* ===== QUANTISE BRAND ===== */
    .quantise-brand {
        font-family:'Cormorant Garamond', Georgia, serif;
        font-style:italic; font-weight:600; letter-spacing:-0.025em;
    }

    /* ===== GUIDE BUBBLE ===== */
    .guide-bubble {
        background: #F0FDFA; border: 1px solid #D1FAE5;
        border-radius: 12px; padding: 14px 18px; margin: 10px 0 16px;
        position: relative;
    }
    .guide-bubble::before {
        content: "💡"; position: absolute; top: -10px; left: 16px;
        background: #F0FDFA; padding: 0 4px; font-size: 0.9rem;
    }
    .guide-bubble .guide-label {
        font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: #2A9D8F; margin-bottom: 5px;
    }
    .guide-bubble .guide-text {
        font-size: 0.8rem; color: #374151; line-height: 1.6;
    }

    /* Welcome banner */
    .welcome-banner {
        background: linear-gradient(135deg, #F0FDFA 0%, #DBEAFE 100%);
        border: 1px solid #D1FAE5; border-radius: 14px;
        padding: 18px 24px; margin-bottom: 18px;
    }
    .welcome-banner .wb-title {
        font-size: 0.95rem; font-weight: 700; color: #111827; margin-bottom: 4px;
    }
    .welcome-banner .wb-text {
        font-size: 0.8rem; color: #6B7280; line-height: 1.6;
    }
    .journey-strip {
        display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
        margin:0 0 18px;
    }
    .journey-step {
        border:1px solid #E5E7EB; border-radius:12px; padding:12px 14px;
        background:#FFFFFF; min-height:72px; height:100%;
        transition:border-color .18s ease, transform .18s ease, box-shadow .18s ease;
    }
    .journey-step strong { display:block; color:#111827; font-size:0.82rem; }
    .journey-step span { color:#6B7280; font-size:0.7rem; line-height:1.4; }
    .journey-step.active { border-color:#4ABEB2; background:#F0FDFA; }
    .journey-link { display:block; color:inherit !important; text-decoration:none !important; }
    .journey-link:hover .journey-step { border-color:#4ABEB2; transform:translateY(-2px);
        box-shadow:0 5px 14px rgba(42,157,143,.10); }
    .journey-link:focus-visible { outline:3px solid rgba(74,190,178,.35);
        outline-offset:3px; border-radius:12px; }
    .section-anchor { display:block; position:relative; top:-18px; visibility:hidden; }
    .page-wordmark { font-family:'Cormorant Garamond', Georgia, serif;
        font-size:2rem; line-height:1; font-style:italic; font-weight:600;
        color:#2A9D8F; letter-spacing:-.025em; margin:2px 0 18px; }
    .page-wordmark span { font-family:'Inter',sans-serif; font-size:.64rem;
        font-style:normal; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
        color:#9CA3AF; margin-left:10px; }
    .hero-wordmark { font-family:'Cormorant Garamond', Georgia, serif;
        font-size:3.6rem; line-height:.9; font-style:italic; font-weight:600;
        color:#4EE4C0; letter-spacing:-.035em; margin-bottom:24px; }
    .journey-number {
        display:inline-flex; width:20px; height:20px; align-items:center;
        justify-content:center; border-radius:50%; background:#1E2B1E;
        color:#4EE4C0 !important; font-weight:700; margin-bottom:6px;
    }
    @media (max-width: 760px) {
        .main .block-container { padding-left:1rem; padding-right:1rem; }
        .lp-h1, .hero-title { font-size:1.8rem !important; line-height:1.15 !important; }
        .lp-stats-row, .kpi-row { display:grid !important; grid-template-columns:1fr 1fr; }
        .journey-strip { grid-template-columns:1fr 1fr; }
        .guide-bubble, .welcome-banner { padding:14px 15px; }
        .hero-wordmark { font-size:2.75rem; }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SIDEBAR + SESSION-STATE NAVIGATION
# =========================================================================
PAGES = ["Home", "Funds", "Sentiment", "Data Explorer"]

# Initialise page in session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Feature cards use a lightweight query parameter for cross-page deep links.
requested_page = st.query_params.get("page")
if requested_page in PAGES:
    st.session_state.page = requested_page

def _go(target):
    """Navigation callback; runs before widgets are rebuilt."""
    st.query_params.clear()
    st.session_state.page = target


def _go_fund(universe):
    """Open the Funds page with a fund family preselected."""
    st.query_params.clear()
    st.session_state.page = "Funds"
    st.session_state.fund_family = universe


def _sidebar_changed():
    """A manual sidebar choice takes precedence over a feature deep link."""
    st.query_params.clear()

st.sidebar.markdown(
    "<div style='padding:8px 0;'>"
    "<span class='quantise-brand' style='font-size:2rem;"
    "color:#4EE4C0 !important;'>Quantise</span><br>"
    "<span style='font-size:0.78rem;opacity:0.6;'>Systematic Multi-Asset Funds</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# Use one session-state key for sidebar and button navigation.
page = st.sidebar.radio(
    "Navigate", PAGES, label_visibility="collapsed", key="page",
    on_change=_sidebar_changed)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.7rem; opacity:0.45; line-height:1.7;'>"
    "FINS3645 FinTech<br>z5488988<br><br>"
    "Walk-forward OOS backtests<br>"
    "Long-only · Monthly rebalance<br>"
    "Equity &radic;252 &nbsp;|&nbsp; Crypto &radic;365<br><br>"
    "Academic prototype · Not financial advice"
    "</div>", unsafe_allow_html=True)

if page != "Home":
    st.markdown(
        '<div class="page-wordmark">Quantise'
        '<span>Systematic research dashboard</span></div>',
        unsafe_allow_html=True,
    )

# =========================================================================
# HOME / LANDING PAGE
# =========================================================================
if page == "Home":

    app_stats = load_app_stats()
    headline_label = (f"{app_stats['headlines'] / 1000:.0f}K+"
                      if app_stats["headlines"] else "Historical")

    # ---------- HERO ----------
    st.markdown(f"""
    <div class="lp-hero-full">
        <div class="hero-wordmark">Quantise</div>
        <div class="lp-trust">Academic research prototype · Walk-forward evidence</div>
        <div class="lp-h1" style="font-family:'Space Grotesk',sans-serif; font-size:2.8rem;">
            Systematic portfolios,<br><span class="lp-serif-accent">built on evidence</span></div>
        <div class="lp-sub">
            Explore four portfolio methods across equity, crypto, and combined universes—
            evaluated with walk-forward out-of-sample backtests and historical news sentiment.
        </div>
        <div class="lp-stats-row">
            <div class="lp-stat">
                <div class="lp-stat-val">{app_stats['assets']}</div>
                <div class="lp-stat-lbl">Assets Tracked</div>
            </div>
            <div class="lp-stat">
                <div class="lp-stat-val">5</div>
                <div class="lp-stat-lbl">Strategies</div>
            </div>
            <div class="lp-stat">
                <div class="lp-stat-val">{headline_label}</div>
                <div class="lp-stat-lbl">Headlines Scored</div>
            </div>
            <div class="lp-stat">
                <div class="lp-stat-val">{app_stats['sectors']}</div>
                <div class="lp-stat-lbl">Sectors Monitored</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero CTA buttons — these actually navigate
    hero_b1, hero_b2, hero_b3, hero_space = st.columns([1, 1, 1, 2])
    with hero_b1:
        st.button("Explore Funds →", key="hero_funds", width="stretch",
                  on_click=_go, args=("Funds",))
    with hero_b2:
        st.button("View Sentiment", key="hero_sentiment", width="stretch",
                  on_click=_go, args=("Sentiment",))
    with hero_b3:
        st.button("Browse Data", key="hero_data", width="stretch",
                  on_click=_go, args=("Data Explorer",))

    # ---------- TRUST LOGOS ----------
    st.markdown("""
    <div class="lp-logos">
        <span>Built with</span>
        <span style="opacity:0.8;">&#9679; Python</span>
        <span style="opacity:0.8;">&#9679; Streamlit</span>
        <span style="opacity:0.8;">&#9679; SciPy</span>
        <span style="opacity:0.8;">&#9679; finVADER</span>
        <span style="opacity:0.8;">&#9679; Matplotlib</span>
        <span style="opacity:0.8;">&#9679; Pandas</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- HOW IT WORKS — 3 STEPS ----------
    st.markdown("""
    <div class="lp-section">
        <div class="lp-section-label">How It Works</div>
        <div class="lp-section-title">From Data to Smart Portfolios<br>in 3 Steps</div>
        <div class="lp-section-sub">Our systematic pipeline transforms raw market data and
        news headlines into optimised, risk-managed fund allocations.</div>
    </div>
    """, unsafe_allow_html=True)

    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("""
        <div class="lp-step">
            <div class="lp-step-num">1</div>
            <div class="lp-step-title">Collect & Clean</div>
            <div class="lp-step-desc">Historical daily prices for 50 US large-cap equities and 10 crypto assets.
            146K+ news headlines scored for sentiment using finVADER with
            SentiBignomics and Henry financial lexicons.</div>
        </div>
        """, unsafe_allow_html=True)
    with step2:
        st.markdown("""
        <div class="lp-step">
            <div class="lp-step-num">2</div>
            <div class="lp-step-title">Optimise & Backtest</div>
            <div class="lp-step-desc">Four portfolio strategies — equal-weight, minimum-variance,
            max-Sharpe, and risk-parity — tested through a walk-forward
            expanding-window OOS backtest with no look-ahead bias.</div>
        </div>
        """, unsafe_allow_html=True)
    with step3:
        st.markdown("""
        <div class="lp-step">
            <div class="lp-step-num">3</div>
            <div class="lp-step-title">Monitor & Tilt</div>
            <div class="lp-step-desc">Sector sentiment scores are computed daily and used
            to tilt portfolio weights toward improving-sentiment sectors —
            blending quantitative optimisation with news signal intelligence.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- FEATURES GRID ----------
    st.markdown("""
    <div class="lp-section">
        <div class="lp-section-label">Product Features</div>
        <div class="lp-section-title">Everything You Need for<br>Systematic Investing</div>
        <div class="lp-section-sub">Interactive analytics and research tools,
        designed for transparency and evidence-based decision-making.</div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <a class="feature-link" href="?page=Funds#evaluate" target="_self">
        <div class="lp-feature-card">
            <span class="lp-feature-icon">&#x1F4C8;</span>
            <div class="lp-feature-title">Portfolio Analytics</div>
            <div class="lp-feature-desc">Growth of $1, drawdown analysis, Sharpe ratios,
            and full risk scorecards for every strategy across three fund families.
            Compare any two funds side-by-side with overlaid charts.</div>
        </div>
        </a>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <a class="feature-link" href="?page=Sentiment#sentiment-trends" target="_self">
        <div class="lp-feature-card">
            <span class="lp-feature-icon">&#x1F4F0;</span>
            <div class="lp-feature-title">Sentiment Intelligence</div>
            <div class="lp-feature-desc">Historical sector Fear & Greed index built from
            financial news headlines. Monthly heatmaps reveal sentiment regimes.
            Browse individual headlines filtered by sector and ticker.</div>
        </div>
        </a>
        """, unsafe_allow_html=True)

    f3, f4 = st.columns(2)
    with f3:
        st.markdown("""
        <a class="feature-link" href="?page=Sentiment#sentiment-fusion" target="_self">
        <div class="lp-feature-card">
            <span class="lp-feature-icon">&#x1F9EA;</span>
            <div class="lp-feature-title">What-If Simulator</div>
            <div class="lp-feature-desc">Drag the tilt slider to explore how different
            sentiment intensities reshape the portfolio. Set custom allocations
            across fund families and see backtested results instantly.</div>
        </div>
        </a>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <a class="feature-link" href="?page=Funds#risk-scenario" target="_self">
        <div class="lp-feature-card">
            <span class="lp-feature-icon">&#x1F9E0;</span>
            <div class="lp-feature-title">Risk Scenario Explorer</div>
            <div class="lp-feature-desc">A three-question quiz maps your risk appetite
            to an illustrative allocation across Equity, Crypto, and Combined funds—
            with backtested performance for the resulting scenario.</div>
        </div>
        </a>
        """, unsafe_allow_html=True)

    f5, f6 = st.columns(2)
    with f5:
        st.markdown("""
        <a class="feature-link" href="?page=Funds#correlation" target="_self">
        <div class="lp-feature-card">
            <span class="lp-feature-icon">&#x1F50D;</span>
            <div class="lp-feature-title">Correlation Explorer</div>
            <div class="lp-feature-desc">Interactive heatmap showing strategy-level
            return correlations. Slide the date range to examine how co-movement
            shifts across bull and bear market regimes.</div>
        </div>
        </a>
        """, unsafe_allow_html=True)
    with f6:
        st.markdown("""
        <a class="feature-link" href="?page=Data%20Explorer#data-browser" target="_self">
        <div class="lp-feature-card">
            <span class="lp-feature-icon">&#x1F4BE;</span>
            <div class="lp-feature-title">Data Explorer</div>
            <div class="lp-feature-desc">Full transparency — browse, filter, and download
            every dataset: daily fund returns, monthly weight snapshots,
            performance metrics, sentiment scores, and raw headlines.</div>
        </div>
        </a>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- PLANS / TIERS ----------
    st.markdown("""
    <div class="lp-section">
        <div class="lp-section-label">Fund Families</div>
        <div class="lp-section-title">Three Funds, One Platform</div>
        <div class="lp-section-sub">Choose the universe that matches your investment thesis.
        Each base fund is evaluated across four methods using a walk-forward out-of-sample design.</div>
    </div>
    """, unsafe_allow_html=True)

    # Load metrics for plan cards
    _lp_metrics = load_performance_metrics()
    _lp_base = _lp_metrics[
        (_lp_metrics["tx_cost_bps"] == 0) &
        (~_lp_metrics["method"].str.contains("sentiment|base", case=False, na=False))
    ]

    p1, p2, p3 = st.columns(3)

    plan_data = [
        ("Equity", 50, "252", [
            "50 US large-cap equities across 10 sectors",
            "4 optimisation strategies",
            "Sentiment fusion overlay",
            "Monthly portfolio rebalancing",
            "Full risk scorecard",
        ]),
        ("Combined", 60, "252", [
            "Multi-asset diversification",
            "Cross-asset correlation analysis",
            "4 optimisation strategies",
            "Transaction cost modelling",
            "Walk-forward OOS backtest",
        ]),
        ("Crypto", 10, "365", [
            "BTC, ETH, SOL, ADA & more",
            "365-day annualisation",
            "High-volatility risk metrics",
            "4 optimisation strategies",
            "Drawdown analysis",
        ]),
    ]

    for col, (uni, n_assets, ann, features) in zip([p1, p2, p3], plan_data):
        uni_metrics = _lp_base[_lp_base["universe"] == uni]
        if not uni_metrics.empty:
            best_row = uni_metrics.loc[uni_metrics["sharpe"].idxmax()]
            best_sharpe = float(best_row["sharpe"])
            best_method = str(best_row["method"])
        else:
            best_sharpe, best_method = 0.0, "Unavailable"
        featured = "lp-plan-featured" if uni == "Combined" else ""
        # Keep a non-empty first child so Markdown does not terminate the HTML
        # block on a blank line for the non-featured cards.
        badge = ('<div class="lp-plan-badge">Cross-asset</div>'
                 if uni == "Combined" else '<span style="display:none"></span>')

        features_html = "".join(f"<li>{f}</li>" for f in features)
        c = UNIVERSE_COLORS.get(uni, {"dot": GREY_500})

        with col:
            card_html = f"""
            <div class="lp-plan-card {featured}" style="border-top:3px solid {c['dot']}">
                {badge}
                <div class="lp-plan-name">{uni}</div>
                <div class="lp-plan-price">{n_assets} <span style="font-size:0.9rem;font-weight:400;color:#9CA3AF">assets</span></div>
                <div class="lp-plan-period">{ann}-day annualisation convention</div>
                <div style="font-size:0.78rem;font-weight:700;color:#2A9D8F;margin-bottom:14px">Highest OOS Sharpe: {best_sharpe:.2f}<br><span style="font-weight:500;color:#6B7280">{html.escape(best_method)}</span></div>
                <ul class="lp-plan-features">{features_html}</ul>
            </div>
            """
            st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)
            st.button(f"Explore {uni}", key=f"plan_{uni}", width="stretch",
                      on_click=_go_fund, args=(uni,))

    st.markdown("""
    <div class="research-disclaimer"><strong>Research prototype.</strong> These are simulated,
    non-investable fund concepts—not financial products or offers to invest. Results are historical
    walk-forward backtests before any management fee and may not reflect live trading outcomes.
    Past performance is not a reliable indicator of future performance.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- FAQ ----------
    st.markdown("""
    <div class="lp-section">
        <div class="lp-section-label">FAQ</div>
        <div class="lp-section-title">Got Questions?<br>We've Got Answers.</div>
        <div class="lp-section-sub">Everything you need to know before exploring the platform.</div>
    </div>
    """, unsafe_allow_html=True)

    faqs = [
        ("What is a walk-forward out-of-sample backtest?",
         "At each monthly rebalance, we estimate expected returns and covariance from ALL "
         "past data only (expanding window), solve for optimal weights, then apply them to "
         "the NEXT month's returns. This prevents look-ahead bias — the portfolio never "
         "sees future data, simulating real-world conditions."),
        ("How does the sentiment fusion work?",
         "We score 146K+ financial news headlines using finVADER (VADER + SentiBignomics + "
         "Henry lexicons), compute a daily sector-level Fear & Greed index, then z-score "
         "and lag the signal by one day. The lagged score tilts portfolio weights toward "
         "sectors with improving sentiment. The tilt strength is configurable via the "
         "What-If slider."),
        ("Why are the crypto fund's risk-adjusted results relatively weak?",
         "The crypto out-of-sample period runs from July 2021 to December 2023 and includes "
         "the severe 2022 crypto downturn. High volatility and large drawdowns reduced the "
         "Sharpe ratios even when some strategies produced positive returns. The app reports "
         "that result rather than implying that diversification guarantees protection."),
        ("What are the four optimisation strategies?",
         "Equal-weight (1/N) — naive benchmark. Minimum-variance — minimises portfolio "
         "volatility (SLSQP, long-only). Max-Sharpe (tangency) — targets the highest "
         "risk-adjusted return via convex transform. Risk parity — equalises each asset's "
         "risk contribution (Maillard et al., 2010)."),
        ("Can I download the raw data?",
         "Yes — the Data Explorer page lets you browse, filter, and download every dataset "
         "as CSV: fund returns, portfolio weights, performance metrics, sector sentiment "
         "scores, and raw headlines."),
    ]

    for question, answer in faqs:
        with st.expander(question):
            st.markdown(f'<div class="lp-faq-a">{answer}</div>', unsafe_allow_html=True)

    # ---------- FOOTER ----------
    st.markdown(f"""
    <div class="lp-footer">
        <div class="lp-footer-col">
            <h4 style="color:#2A9D8F; font-size:1.3rem; font-weight:800;
                letter-spacing:-0.03em; text-transform:none;
                font-family:'Space Grotesk',sans-serif;">Quantise</h4>
            <span>Systematic multi-asset research dashboard</span>
            <span style="margin-top:8px;">FINS3645 FinTech &middot; z5488988</span>
        </div>
        <div class="lp-footer-col">
            <h4>Platform</h4>
            <span>Fund Dashboard</span>
            <span>Sentiment Analytics</span>
            <span>Data Explorer</span>
            <span>Risk Scenario Explorer</span>
        </div>
        <div class="lp-footer-col">
            <h4>Methodology</h4>
            <span>Walk-Forward Backtest</span>
            <span>Portfolio Optimisation</span>
            <span>Sentiment Scoring</span>
            <span>Transaction Costs</span>
        </div>
        <div class="lp-footer-col">
            <h4>Technology</h4>
            <span>Python &middot; SciPy</span>
            <span>finVADER &middot; NLTK</span>
            <span>Streamlit</span>
            <span>Matplotlib &middot; Pandas</span>
        </div>
    </div>
    <div class="lp-footer-copy">
        &copy; 2026 Quantise &middot; FINS3645 academic research prototype &middot;
        Not a financial product, offer, or personal financial advice. Historical backtests only.
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# TAB 1: FUNDS
# =========================================================================
elif page == "Funds":

    fund_returns = load_fund_returns()
    fund_weights = load_fund_weights()
    metrics = load_performance_metrics()
    asset_count = int(fund_weights["ticker"].nunique())

    universes = ["Equity", "Crypto", "Combined"]
    base_methods = ["Equal-weight (1/N)", "Minimum-variance",
                    "Max-Sharpe (tangency)", "Risk parity"]
    fund_profiles = {
        "Equity": {
            "objective": "Long-term growth from diversified US large-cap equities.",
            "risk": "Growth · market-sensitive",
            "universe": "50 equities · 10 sectors",
        },
        "Crypto": {
            "objective": "High-growth exposure to a diversified digital-asset basket.",
            "risk": "Aggressive · high volatility",
            "universe": "10 major crypto assets",
        },
        "Combined": {
            "objective": "Blend equity growth with digital-asset diversification.",
            "risk": "Growth · multi-asset",
            "universe": "60 assets across two markets",
        },
    }

    # --- Welcome guide ---
    st.markdown("""
    <div class="welcome-banner">
        <div class="wb-title">Welcome to the Fund Dashboard</div>
        <div class="wb-text">
            This page lets you explore three systematically managed fund families — Equity, Crypto, and Combined.
            Start by selecting a fund family and strategies below the hero, then scroll to see growth charts,
            risk scorecards, and interactive tools like Fund Comparison and the Risk Scenario Explorer.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="journey-strip" aria-label="Investor journey">
        <a class="journey-link" href="#discover"><div class="journey-step active">
            <span class="journey-number">1</span><strong>Discover</strong>
            <span>Choose a fund family and understand its purpose.</span></div></a>
        <a class="journey-link" href="#evaluate"><div class="journey-step">
            <span class="journey-number">2</span><strong>Evaluate</strong>
            <span>Review risk, return, drawdown and holdings.</span></div></a>
        <a class="journey-link" href="#build"><div class="journey-step">
            <span class="journey-number">3</span><strong>Build</strong>
            <span>Create an allocation with explicit strategy choices.</span></div></a>
        <a class="journey-link" href="#compare"><div class="journey-step">
            <span class="journey-number">4</span><strong>Compare</strong>
            <span>Stress-test alternatives before deciding.</span></div></a>
    </div>
    """, unsafe_allow_html=True)

    # --- Hero ---
    base_m = metrics[
        (metrics["tx_cost_bps"] == 0) &
        (~metrics["method"].str.contains("sentiment|base", case=False, na=False))
    ]
    eq_best = base_m[base_m["universe"] == "Equity"].nlargest(1, "sharpe").iloc[0]
    oos_start = fund_returns["date"].min().strftime("%b %Y")
    oos_end = fund_returns["date"].max().strftime("%b %Y")

    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">Fund Dashboard</div>
        <div class="hero-title">Systematic Multi-Asset Portfolio Analytics</div>
        <div class="kpi-row">
            <div class="kpi-item">
                <div class="kpi-lbl">Total Assets</div>
                <div class="kpi-val">{asset_count}</div>
                <div class="kpi-sub">50 equities + 10 crypto</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">Strategies</div>
                <div class="kpi-val">4 + Fusion</div>
                <div class="kpi-sub">+ sentiment overlay</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">Highest Equity OOS Sharpe</div>
                <div class="kpi-val">{eq_best['sharpe']:.2f}</div>
                <div class="kpi-sub">{eq_best['method']}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">OOS Period</div>
                <div class="kpi-val">{oos_start} – {oos_end}</div>
                <div class="kpi-sub">Walk-forward backtest</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Selectors ---
    st.markdown('<span id="discover" class="section-anchor"></span>',
                unsafe_allow_html=True)
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_universe = st.selectbox("Fund Family", universes, index=0,
                                         key="fund_family")
    with col_sel2:
        available_methods = base_methods.copy()
        if selected_universe == "Equity":
            available_methods.append("MinVar + sentiment")
        selected_methods = st.multiselect("Strategies", available_methods,
                                          default=available_methods)
    if not selected_methods:
        st.warning("Please select at least one strategy.")
        st.stop()

    st.markdown(f'<div class="badge-row">{universe_badge(selected_universe)}</div>',
                unsafe_allow_html=True)

    profile = fund_profiles[selected_universe]
    p1, p2, p3, p4 = st.columns([1.5, 1, 1, 1])
    p1.markdown(f"**Investment objective**  \n{profile['objective']}")
    p2.markdown(f"**Risk profile**  \n{profile['risk']}")
    p3.markdown(f"**Investment universe**  \n{profile['universe']}")
    p4.markdown("**Portfolio process**  \nLong-only · monthly rebalance")
    st.caption("Research prototype · Walk-forward out-of-sample results · "
               "Backtested performance is not a guarantee of future results.")

    fr = fund_returns[
        (fund_returns["universe"] == selected_universe) &
        (fund_returns["method"].isin(selected_methods))
    ].copy()

    # --- Summary KPIs ---
    sel_metrics = metrics[
        (metrics["universe"] == selected_universe) &
        (metrics["tx_cost_bps"] == 0) &
        (metrics["method"].isin(selected_methods))
    ]
    if not sel_metrics.empty:
        best = sel_metrics.loc[sel_metrics["sharpe"].idxmax()]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Highest OOS Sharpe", f"{best['sharpe']:.2f}", delta=best["method"])
        k2.metric("Ann. Return", f"{best['ann_return']*100:.1f}%")
        k3.metric("Ann. Volatility", f"{best['ann_vol']*100:.1f}%")
        k4.metric("Max Drawdown", f"{best['max_drawdown']*100:.1f}%")

    st.markdown("---")

    # --- Charts in card containers ---
    st.markdown('<span id="evaluate" class="section-anchor"></span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Performance</div>'
        '<div class="section-title">Growth of $1 & Drawdowns</div>'
        '<div class="section-subtitle">Cumulative out-of-sample performance and '
        'peak-to-trough drawdowns for each strategy.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">How to read these charts</div>
        <div class="guide-text">
            <strong>Growth of $1</strong> shows what $1 invested at the start of the backtest period would be worth today.
            A line above 1.0 means the strategy made money; below 1.0 means a loss.
            <strong>Drawdowns</strong> show peak-to-trough declines — how much value the strategy lost from its highest point.
            Deeper drawdowns mean more pain for investors holding through downturns.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card"><div class="chart-card-title">'
                    'Cumulative Growth</div>', unsafe_allow_html=True)
        fig, ax = clean_fig(7, 4)
        for method in selected_methods:
            sub = fr[fr["method"] == method].sort_values("date")
            if sub.empty:
                continue
            color = STRATEGY_COLORS.get(method, GREY_500)
            ax.plot(sub["date"], sub["growth_of_1"], color=color, lw=2.2,
                    label=method, zorder=3)
            # Area fill for the first (best) strategy only
            if method == selected_methods[0]:
                ax.fill_between(sub["date"], 1, sub["growth_of_1"],
                                alpha=0.06, color=color, zorder=1)
        ax.axhline(1, color=GREY_300, lw=0.7, ls="--", alpha=0.5, zorder=2)
        ax.set_ylabel("Value of $1", fontsize=9, color=GREY_500)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left",
                  handlelength=1.5, labelspacing=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-card-title">'
                    'Drawdown Analysis</div>', unsafe_allow_html=True)
        fig, ax = clean_fig(7, 4)
        for method in selected_methods:
            sub = fr[fr["method"] == method].sort_values("date")
            if sub.empty:
                continue
            wealth = sub["growth_of_1"].values
            running_max = np.maximum.accumulate(wealth)
            dd = wealth / running_max - 1.0
            color = STRATEGY_COLORS.get(method, GREY_500)
            ax.fill_between(sub["date"], dd, 0, alpha=0.12, color=color, zorder=1)
            ax.plot(sub["date"], dd, color=color, lw=1.8, label=method, zorder=3)
        ax.axhline(0, color=GREY_300, lw=0.7, zorder=2)
        ax.set_ylabel("Drawdown", fontsize=9, color=GREY_500)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=7.5, frameon=False, loc="lower left",
                  handlelength=1.5, labelspacing=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Performance scorecard ---
    st.markdown(
        '<div class="section-label">Metrics</div>'
        '<div class="section-title">Performance Scorecard</div>'
        '<div class="section-subtitle">Risk-adjusted returns, drawdowns, and tail-risk '
        'metrics. Highest out-of-sample Sharpe row highlighted.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">Understanding the metrics</div>
        <div class="guide-text">
            <strong>Sharpe Ratio</strong> compares average return with historical volatility; higher values indicate
            more return per unit of measured risk, but interpretation depends on the sample and assumptions.
            <strong>Sortino</strong> is similar but only penalises downside volatility.
            <strong>Max Drawdown</strong> is the worst peak-to-trough loss.
            <strong>VaR 95%</strong> is the daily loss you'd expect to exceed only 5% of the time.
            <strong>ES 95%</strong> (Expected Shortfall) is the average loss on those worst 5% of days.
        </div>
    </div>
    """, unsafe_allow_html=True)

    sel_base = metrics[
        (metrics["universe"] == selected_universe) &
        (metrics["tx_cost_bps"] == 0) &
        (metrics["method"].isin(selected_methods))
    ].copy()

    if not sel_base.empty:
        show = sel_base[["method", "ann_return", "ann_vol", "sharpe", "sortino",
                         "max_drawdown", "var_95", "es_95", "total_return"]].copy()
        show.columns = ["Strategy", "Ann. Return", "Ann. Vol", "Sharpe", "Sortino",
                        "Max DD", "VaR 95%", "ES 95%", "Total Return"]
        show = show.reset_index(drop=True)
        for col in ["Ann. Return", "Ann. Vol", "Max DD", "Total Return"]:
            show[col] = show[col].apply(lambda x: f"{x*100:.1f}%")
        for col in ["VaR 95%", "ES 95%"]:
            show[col] = show[col].apply(lambda x: f"{x*100:.2f}%")

        st.markdown(
            styled_html_table(show, highlight_col="Sharpe", highlight_max=True,
                              fmt2_cols=["Sharpe", "Sortino"]),
            unsafe_allow_html=True)

    # TX cost
    tc_data = metrics[
        (metrics["universe"] == selected_universe) &
        (metrics["method"].isin(base_methods))
    ].copy()
    tc_0 = tc_data[tc_data["tx_cost_bps"] == 0][["method", "sharpe"]].rename(
        columns={"sharpe": "Sharpe (0 bps)"})
    tc_10 = tc_data[tc_data["tx_cost_bps"] == 10][["method", "sharpe"]].rename(
        columns={"sharpe": "Sharpe (10 bps)"})
    if not tc_0.empty and not tc_10.empty:
        tc_compare = tc_0.merge(tc_10, on="method")
        tc_compare["Impact"] = tc_compare.apply(
            lambda r: f"{r['Sharpe (10 bps)'] - r['Sharpe (0 bps)']:+.3f}", axis=1)
        tc_compare = tc_compare.rename(columns={"method": "Strategy"})
        with st.expander("Transaction Cost Impact (10 bps one-way)"):
            st.markdown(
                '<div class="section-subtitle">Sharpe ratio change after deducting '
                '10 bps per unit of turnover at each monthly rebalance.</div>',
                unsafe_allow_html=True)
            st.markdown(
                styled_html_table(tc_compare.reset_index(drop=True),
                                  fmt2_cols=["Sharpe (0 bps)", "Sharpe (10 bps)"]),
                unsafe_allow_html=True)

    st.markdown("---")

    # --- Holdings with ticker badges ---
    st.markdown(
        '<div class="section-label">Composition</div>'
        '<div class="section-title">Current Holdings</div>'
        '<div class="section-subtitle">Latest portfolio weights. Top 15 positions '
        'are shown in the bar chart; full list in the scrollable table.</div>',
        unsafe_allow_html=True)

    hold_c1, hold_c2 = st.columns([1, 2])
    with hold_c1:
        holdings_method = st.selectbox("Strategy", selected_methods,
                                       key="holdings_strategy")

    fw_sel = fund_weights[
        (fund_weights["universe"] == selected_universe) &
        (fund_weights["method"] == holdings_method) &
        (fund_weights["weight"] > 0.001)
    ]

    if not fw_sel.empty:
        latest_date = fw_sel["date"].max()
        latest = fw_sel[fw_sel["date"] == latest_date].sort_values("weight", ascending=False)
        n_pos = len(latest)
        st.caption(f"As of {latest_date.strftime('%Y-%m-%d')} · {n_pos} positions")

        col_chart, col_table = st.columns([1.3, 1])
        with col_chart:
            st.markdown('<div class="chart-card"><div class="chart-card-title">'
                        'Top 15 Weights</div>', unsafe_allow_html=True)
            top = latest.head(15)
            fig, ax = clean_fig(6, 5)
            n_bars = len(top)
            # Teal gradient bars
            from matplotlib.colors import LinearSegmentedColormap
            teal_cmap = LinearSegmentedColormap.from_list(
                "teal_grad", ["#B2DFDB", TEAL_DARK], N=n_bars)
            bar_colors = [teal_cmap(i / max(n_bars - 1, 1)) for i in range(n_bars)]
            ax.barh(top["ticker"].values[::-1], top["weight"].values[::-1],
                    color=bar_colors[::-1], edgecolor="white", height=0.65,
                    zorder=3)
            ax.set_xlabel("Weight", fontsize=8.5, color=GREY_500)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_table:
            disp = latest[["ticker", "weight"]].copy().reset_index(drop=True)
            disp["weight"] = disp["weight"].apply(lambda x: f"{x*100:.2f}%")
            disp.columns = ["Ticker", "Weight"]
            st.markdown(
                styled_html_table(disp, ticker_col="Ticker", max_height=420,
                                  compact=True),
                unsafe_allow_html=True)

    st.markdown("---")

    # --- Allocation simulator ---
    st.markdown('<span id="build" class="section-anchor"></span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Simulator</div>'
        '<div class="section-title">Set Your Allocation</div>'
        '<div class="section-subtitle">Drag sliders to allocate across fund families. '
        'Choose the strategy used by each sleeve, then review the historical blend.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">Try it yourself</div>
        <div class="guide-text">
            Drag the three sliders below to set your desired allocation across Equity, Crypto, and Combined funds.
            Choose both an allocation and a strategy for each fund family. The tool normalises your inputs to 100%,
            aligns all return series to common dates, and shows a historical illustration. This is not a forecast or
            an investment recommendation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    alloc_cols = st.columns(3)
    allocs = {}
    alloc_methods = {}
    for i, uni in enumerate(universes):
        with alloc_cols[i]:
            allocs[uni] = st.slider(f"{uni} (%)", 0, 100,
                                    33 if uni != "Combined" else 34, key=f"alloc_{uni}")
            method_options = base_methods + (["MinVar + sentiment"] if uni == "Equity" else [])
            alloc_methods[uni] = st.selectbox(
                f"{uni} strategy", method_options, index=1, key=f"alloc_method_{uni}")

    total_alloc = sum(allocs.values())
    if total_alloc == 0:
        st.info("Set at least one allocation above 0%.")
    else:
        norm = {k: v / total_alloc for k, v in allocs.items()}
        badges = " ".join(
            f"{universe_badge(uni)} {norm[uni] * 100:.0f}% · {html.escape(alloc_methods[uni])}"
                          for uni in universes if allocs[uni] > 0)
        st.markdown(f'<div style="margin-bottom:12px;">{badges}</div>',
                    unsafe_allow_html=True)

        active_universes = [uni for uni in universes if norm[uni] > 0]
        if "Combined" in active_universes and len(active_universes) > 1:
            st.caption("Note: the Combined fund already contains equity and crypto, so using it "
                       "with either single-asset fund creates overlapping exposure.")

        blended_parts = []
        for uni in universes:
            if norm[uni] <= 0:
                continue
            sub = fund_returns[
                (fund_returns["universe"] == uni) &
                (fund_returns["method"] == alloc_methods[uni])
            ].set_index("date")["daily_return"].rename(uni)
            blended_parts.append(sub)

        if blended_parts:
            aligned = pd.concat(blended_parts, axis=1, join="inner").dropna()
            active_weights = pd.Series({u: norm[u] for u in aligned.columns})
            active_weights = active_weights / active_weights.sum()
            blended = aligned.mul(active_weights, axis=1).sum(axis=1)
            blended_growth = (1 + blended).cumprod()
            total_ret = float(blended_growth.iloc[-1] - 1)
            n_days = len(blended)
            annual_factor = 365 if active_universes == ["Crypto"] else 252
            ann_ret = float((1 + total_ret) ** (annual_factor / max(n_days, 1)) - 1)
            ann_vol = float(blended.std() * np.sqrt(annual_factor))
            sharpe = (float(blended.mean() / blended.std() * np.sqrt(annual_factor))
                      if blended.std() > 0 else 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{total_ret*100:.1f}%")
            m2.metric("Ann. Return", f"{ann_ret*100:.1f}%")
            m3.metric("Ann. Volatility", f"{ann_vol*100:.1f}%")
            m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

            fig, ax = clean_fig(10, 3.4)
            ax.plot(blended_growth.index, blended_growth.values, color=TEAL_DARK,
                    lw=2.3, label="Your historical blend", zorder=3)
            ax.fill_between(blended_growth.index, 1, blended_growth.values,
                            color=TEAL, alpha=0.08, zorder=1)
            ax.axhline(1, color=GREY_300, lw=0.8, ls="--")
            ax.set_ylabel("Value of $1", fontsize=9, color=GREY_500)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption(f"Historical illustration over {n_days:,} aligned observations. "
                       "Backtested performance is not a guarantee of future results.")

    st.markdown("---")

    # === FUND COMPARISON TOOL ===
    st.markdown('<span id="compare" class="section-anchor"></span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Compare</div>'
        '<div class="section-title">Fund Comparison Tool</div>'
        '<div class="section-subtitle">Pick any two strategies side-by-side. '
        'Overlaid growth curves, drawdown profiles, and a head-to-head metrics table.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">How to use this tool</div>
        <div class="guide-text">
            Select any two fund-strategy combinations from the dropdowns below. The tool overlays their growth curves
            and drawdown profiles, then generates an automated insight comparing their correlation, risk-adjusted returns,
            and volatility profiles. Try comparing Equity vs Crypto strategies to see how different asset classes behave.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Build list of all available fund keys
    all_fund_keys = []
    for uni in universes:
        for meth in base_methods:
            all_fund_keys.append(f"{uni} — {meth}")
        if uni == "Equity":
            all_fund_keys.append("Equity — MinVar + sentiment")

    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        fund_a = st.selectbox("Fund A", all_fund_keys, index=0, key="comp_a")
    with comp_c2:
        fund_b = st.selectbox("Fund B", all_fund_keys,
                              index=min(1, len(all_fund_keys) - 1), key="comp_b")

    def _parse_fund_key(key):
        parts = key.split(" — ", 1)
        return parts[0], parts[1]

    uni_a, meth_a = _parse_fund_key(fund_a)
    uni_b, meth_b = _parse_fund_key(fund_b)

    fr_a = fund_returns[
        (fund_returns["universe"] == uni_a) & (fund_returns["method"] == meth_a)
    ].sort_values("date").set_index("date")
    fr_b = fund_returns[
        (fund_returns["universe"] == uni_b) & (fund_returns["method"] == meth_b)
    ].sort_values("date").set_index("date")

    if not fr_a.empty and not fr_b.empty:
        # Align dates
        common_idx = fr_a.index.intersection(fr_b.index)
        ga = fr_a.loc[common_idx, "growth_of_1"]
        gb = fr_b.loc[common_idx, "growth_of_1"]
        ra = fr_a.loc[common_idx, "daily_return"]
        rb = fr_b.loc[common_idx, "daily_return"]

        comp_chart1, comp_chart2 = st.columns(2)
        with comp_chart1:
            st.markdown('<div class="chart-card"><div class="chart-card-title">'
                        'Growth Overlay</div>', unsafe_allow_html=True)
            fig, ax = clean_fig(7, 4)
            ax.plot(ga.index, ga.values, color=TEAL_DARK, lw=2.2, label=fund_a, zorder=3)
            ax.plot(gb.index, gb.values, color="#E63946", lw=2.2, label=fund_b, zorder=3)
            ax.axhline(1, color=GREY_300, lw=0.7, ls="--", alpha=0.5, zorder=2)
            ax.set_ylabel("Value of $1", fontsize=9, color=GREY_500)
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with comp_chart2:
            st.markdown('<div class="chart-card"><div class="chart-card-title">'
                        'Drawdown Overlay</div>', unsafe_allow_html=True)
            fig, ax = clean_fig(7, 4)
            for label, g, color in [(fund_a, ga, TEAL_DARK), (fund_b, gb, "#E63946")]:
                running_max = np.maximum.accumulate(g.values)
                dd = g.values / running_max - 1.0
                ax.fill_between(g.index, dd, 0, alpha=0.12, color=color, zorder=1)
                ax.plot(g.index, dd, color=color, lw=1.8, label=label, zorder=3)
            ax.axhline(0, color=GREY_300, lw=0.7, zorder=2)
            ax.set_ylabel("Drawdown", fontsize=9, color=GREY_500)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.legend(fontsize=7.5, frameon=False, loc="lower left")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # Head-to-head metrics
        def _quick_metrics(ret_series, label):
            r = ret_series.dropna()
            n = len(r)
            total = float((1 + r).prod() - 1)
            ann_ret = float((1 + total) ** (252 / max(n, 1)) - 1)
            ann_vol = float(r.std() * np.sqrt(252))
            sharpe_v = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
            wealth = (1 + r).cumprod()
            max_dd = float((wealth / wealth.cummax() - 1).min())
            return {
                "Fund": label,
                "Ann. Return": f"{ann_ret*100:.1f}%",
                "Ann. Vol": f"{ann_vol*100:.1f}%",
                "Sharpe": f"{sharpe_v:.2f}",
                "Max DD": f"{max_dd*100:.1f}%",
                "Total Return": f"{total*100:.1f}%",
            }

        comp_table = pd.DataFrame([
            _quick_metrics(ra, fund_a),
            _quick_metrics(rb, fund_b),
        ])
        st.markdown(
            styled_html_table(comp_table.reset_index(drop=True),
                              highlight_col="Sharpe", highlight_max=True),
            unsafe_allow_html=True)

        # --- Smart commentary ---
        corr_ab = float(ra.corr(rb))
        sharpe_a = float(ra.mean() / ra.std() * np.sqrt(252)) if ra.std() > 0 else 0
        sharpe_b = float(rb.mean() / rb.std() * np.sqrt(252)) if rb.std() > 0 else 0
        vol_a = float(ra.std() * np.sqrt(252))
        vol_b = float(rb.std() * np.sqrt(252))
        winner = fund_a if sharpe_a >= sharpe_b else fund_b
        loser = fund_b if sharpe_a >= sharpe_b else fund_a

        insights = []
        if corr_ab > 0.90:
            insights.append(
                f"These two funds are **highly correlated** (r = {corr_ab:.2f}), "
                "meaning they tend to move together. Holding both provides "
                "**minimal diversification benefit** — consider replacing one with "
                "a less correlated strategy.")
        elif corr_ab > 0.60:
            insights.append(
                f"Moderate correlation (r = {corr_ab:.2f}) — combining these funds "
                "offers **some diversification** but they still share common risk drivers.")
        elif corr_ab > 0.0:
            insights.append(
                f"Low correlation (r = {corr_ab:.2f}) — a **strong diversification pair**. "
                "Blending these funds would meaningfully reduce portfolio volatility.")
        else:
            insights.append(
                f"Negative correlation (r = {corr_ab:.2f}) — an **excellent hedge pair**. "
                "These funds tend to offset each other's losses, offering natural "
                "downside protection.")

        sharpe_gap = abs(sharpe_a - sharpe_b)
        if sharpe_gap > 0.3:
            insights.append(
                f"**{winner}** delivers a substantially higher risk-adjusted return "
                f"(Sharpe gap of {sharpe_gap:.2f}). The lower-ranked fund "
                f"({loser}) would need to offer meaningful diversification to "
                "justify its inclusion.")
        elif sharpe_gap < 0.05:
            insights.append(
                "Both funds have **nearly identical Sharpe ratios** — the choice "
                "between them comes down to volatility preference and drawdown tolerance "
                "rather than raw performance.")

        if abs(vol_a - vol_b) / max(vol_a, vol_b, 0.01) > 0.4:
            lower_vol = fund_a if vol_a < vol_b else fund_b
            insights.append(
                f"**{lower_vol}** is significantly less volatile — better suited "
                "for investors with shorter horizons or lower drawdown tolerance.")

        st.markdown(
            '<div style="background:#F0FDFA; border-left:3px solid #2A9D8F; '
            'border-radius:0 8px 8px 0; padding:14px 18px; margin:8px 0 16px;">'
            '<div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
            'letter-spacing:0.08em; color:#2A9D8F; margin-bottom:6px;">'
            'Comparison Insight</div>'
            '<div style="font-size:0.82rem; color:#374151; line-height:1.65;">'
            + " ".join(insights) +
            '</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # === CORRELATION EXPLORER ===
    st.markdown('<span id="correlation" class="section-anchor"></span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Relationships</div>'
        '<div class="section-title">Correlation Explorer</div>'
        '<div class="section-subtitle">Strategy-level return correlations. '
        'Use the date range slider to examine regime-dependent co-movement.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">Reading the correlation heatmap</div>
        <div class="guide-text">
            Each cell shows how closely two strategies move together. Values near <strong>+1.0</strong> (dark red) mean
            they rise and fall in sync — holding both gives little diversification. Values near <strong>0.0</strong>
            (white) are uncorrelated — great for building a diversified portfolio. Negative values mean they tend to
            move in opposite directions, offering natural hedging. Drag the date slider to see how correlations shift
            across bull and bear markets.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Build wide returns matrix: all (universe, method) combos
    fr_wide_parts = []
    for uni in universes:
        for meth in base_methods:
            sub = fund_returns[
                (fund_returns["universe"] == uni) & (fund_returns["method"] == meth)
            ].set_index("date")["daily_return"]
            fr_wide_parts.append(sub.rename(f"{uni[:3]}:{meth[:8]}"))
    # Add fusion
    fus = fund_returns[
        (fund_returns["universe"] == "Equity") &
        (fund_returns["method"] == "MinVar + sentiment")
    ].set_index("date")["daily_return"]
    if not fus.empty:
        fr_wide_parts.append(fus.rename("Eq:Fusion"))

    fr_wide = pd.concat(fr_wide_parts, axis=1).dropna()

    if not fr_wide.empty:
        min_date = fr_wide.index.min().to_pydatetime()
        max_date = fr_wide.index.max().to_pydatetime()
        date_range = st.slider(
            "Date Range", min_value=min_date, max_value=max_date,
            value=(min_date, max_date), format="MMM YYYY", key="corr_dates")
        fr_filtered = fr_wide.loc[date_range[0]:date_range[1]]

        if len(fr_filtered) > 20:
            corr_matrix = fr_filtered.corr()

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
            fig.patch.set_facecolor(WHITE)
            im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1,
                           aspect="auto")
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, fontsize=7.5, rotation=45, ha="right")
            ax.set_yticks(range(len(corr_matrix.index)))
            ax.set_yticklabels(corr_matrix.index, fontsize=7.5)
            # Annotate cells
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    val = corr_matrix.iloc[i, j]
                    text_color = WHITE if abs(val) > 0.6 else GREY_900
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6.5, color=text_color, fontweight="bold")
            ax.tick_params(colors=GREY_500, length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            st.caption(f"Computed over {len(fr_filtered):,} trading days "
                       f"({date_range[0].strftime('%b %Y')} – {date_range[1].strftime('%b %Y')}).")

            # --- Smart commentary ---
            # Find most/least correlated pair (off-diagonal)
            mask = np.ones(corr_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            corr_vals = corr_matrix.where(mask)
            max_pair = corr_vals.stack().idxmax()
            min_pair = corr_vals.stack().idxmin()
            max_corr = corr_vals.stack().max()
            min_corr = corr_vals.stack().min()

            # Average cross-universe correlation
            eq_cols = [c for c in corr_matrix.columns if c.startswith("Eq:")]
            cr_cols = [c for c in corr_matrix.columns if c.startswith("Cry")]
            cross_corr = corr_matrix.loc[eq_cols, cr_cols].values.mean() if eq_cols and cr_cols else None

            corr_insights = []
            corr_insights.append(
                f"**Strongest link:** {max_pair[0]} and {max_pair[1]} "
                f"(r = {max_corr:.2f}) — these strategies share the most common "
                "risk exposure.")
            corr_insights.append(
                f"**Best diversifier:** {min_pair[0]} and {min_pair[1]} "
                f"(r = {min_corr:.2f}) — combining these offers the greatest "
                "volatility reduction.")
            if cross_corr is not None:
                if cross_corr < 0.3:
                    corr_insights.append(
                        f"Equity–Crypto cross-correlation averages just "
                        f"**{cross_corr:.2f}** — confirming that crypto adds "
                        "genuine diversification to an equity portfolio.")
                else:
                    corr_insights.append(
                        f"Equity–Crypto cross-correlation is {cross_corr:.2f} — "
                        "higher than expected, suggesting macro factors are driving "
                        "both asset classes in this period.")

            st.markdown(
                '<div style="background:#F0FDFA; border-left:3px solid #2A9D8F; '
                'border-radius:0 8px 8px 0; padding:14px 18px; margin:8px 0 16px;">'
                '<div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
                'letter-spacing:0.08em; color:#2A9D8F; margin-bottom:6px;">'
                'Correlation Insight</div>'
                '<div style="font-size:0.82rem; color:#374151; line-height:1.65;">'
                + " ".join(corr_insights) +
                '</div></div>', unsafe_allow_html=True)
        else:
            st.warning("Not enough data in selected range for correlation analysis.")

    st.markdown("---")

    # === RISK PROFILER QUIZ ===
    st.markdown('<span id="risk-scenario" class="section-anchor"></span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Interactive</div>'
        '<div class="section-title">Risk Scenario Explorer</div>'
        '<div class="section-subtitle">Answer three questions to explore an educational '
        'allocation scenario based on broad risk preferences.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">Educational scenario—not personal advice</div>
        <div class="guide-text">
            Answer three quick questions about your risk tolerance, investment horizon, and growth preference.
            The quiz maps your answers to one of four investor profiles (Conservative, Moderate, Growth, Aggressive),
            each with an illustrative fund allocation and method. The result does not consider your financial position,
            objectives, needs, tax circumstances, or capacity for loss.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("risk_quiz"):
        q1 = st.select_slider(
            "1. How would you react to a 20% drop in your portfolio?",
            options=["Sell everything", "Sell some", "Hold steady",
                     "Buy a little more", "Buy aggressively"],
            value="Hold steady", key="rq1")
        q2 = st.select_slider(
            "2. What is your investment horizon?",
            options=["< 1 year", "1–3 years", "3–5 years",
                     "5–10 years", "10+ years"],
            value="3–5 years", key="rq2")
        q3 = st.select_slider(
            "3. How important is growth vs stability?",
            options=["Stability only", "Mostly stable", "Balanced",
                     "Mostly growth", "Maximum growth"],
            value="Balanced", key="rq3")
        submitted = st.form_submit_button("Explore My Scenario",
                                          width="stretch")

    if submitted:
        # Score each answer 1–5
        score_map_q1 = {"Sell everything": 1, "Sell some": 2, "Hold steady": 3,
                        "Buy a little more": 4, "Buy aggressively": 5}
        score_map_q2 = {"< 1 year": 1, "1–3 years": 2, "3–5 years": 3,
                        "5–10 years": 4, "10+ years": 5}
        score_map_q3 = {"Stability only": 1, "Mostly stable": 2, "Balanced": 3,
                        "Mostly growth": 4, "Maximum growth": 5}
        total_score = score_map_q1[q1] + score_map_q2[q2] + score_map_q3[q3]

        # Map to allocation
        if total_score <= 5:
            profile = "Conservative"
            alloc_rec = {"Equity": 80, "Crypto": 0, "Combined": 20}
            strategy_rec = "Minimum-variance"
            desc = ("You prefer stability over growth. A minimum-variance equity "
                    "fund minimises portfolio volatility while maintaining market "
                    "exposure. No crypto allocation given your risk sensitivity.")
        elif total_score <= 8:
            profile = "Moderate"
            alloc_rec = {"Equity": 60, "Crypto": 5, "Combined": 35}
            strategy_rec = "Risk parity"
            desc = ("You seek balanced risk-adjusted returns. Risk parity equalises "
                    "each asset's contribution to portfolio risk, offering "
                    "diversification without concentration. Small crypto allocation "
                    "adds asymmetric upside potential.")
        elif total_score <= 11:
            profile = "Growth"
            alloc_rec = {"Equity": 40, "Crypto": 15, "Combined": 45}
            strategy_rec = "Max-Sharpe (tangency)"
            desc = ("You're comfortable with volatility in pursuit of higher returns. "
                    "The tangency portfolio targets the highest risk-adjusted return "
                    "on the efficient frontier. Meaningful crypto exposure adds growth "
                    "potential with higher volatility.")
        else:
            profile = "Aggressive"
            alloc_rec = {"Equity": 25, "Crypto": 30, "Combined": 45}
            strategy_rec = "Equal-weight (1/N)"
            desc = ("You want maximum growth and can tolerate large drawdowns. "
                    "Equal-weight avoids concentration risk and benefits from mean-"
                    "reversion across a broad basket. Heavy crypto exposure maximises "
                    "growth potential.")

        # Display result
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #1E2B1E 0%, #2A9D8F 100%);
             border-radius:14px; padding:24px 28px; color:white; margin:12px 0;">
            <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:0.1em; color:#4EE4C0; margin-bottom:6px;">
                Illustrative Risk Scenario</div>
            <div style="font-size:1.3rem; font-weight:700; margin-bottom:6px;">
                {profile} Investor</div>
            <div style="font-size:0.85rem; opacity:0.85; margin-bottom:16px;
                 line-height:1.6;">{desc}</div>
            <div style="display:flex; gap:20px; flex-wrap:wrap;">
                <div><span style="font-size:0.68rem; text-transform:uppercase;
                     letter-spacing:0.08em; opacity:0.6; font-weight:600;">
                     Scenario Method</span><br>
                     <span style="font-size:1.1rem; font-weight:700;">
                     {strategy_rec}</span></div>
                <div><span style="font-size:0.68rem; text-transform:uppercase;
                     letter-spacing:0.08em; opacity:0.6; font-weight:600;">
                     Equity</span><br>
                     <span style="font-size:1.1rem; font-weight:700;">
                     {alloc_rec['Equity']}%</span></div>
                <div><span style="font-size:0.68rem; text-transform:uppercase;
                     letter-spacing:0.08em; opacity:0.6; font-weight:600;">
                     Crypto</span><br>
                     <span style="font-size:1.1rem; font-weight:700;">
                     {alloc_rec['Crypto']}%</span></div>
                <div><span style="font-size:0.68rem; text-transform:uppercase;
                     letter-spacing:0.08em; opacity:0.6; font-weight:600;">
                     Combined</span><br>
                     <span style="font-size:1.1rem; font-weight:700;">
                     {alloc_rec['Combined']}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show what this allocation would have returned
        norm_r = {k: v / 100 for k, v in alloc_rec.items()}
        quiz_parts = []
        for uni in universes:
            if norm_r[uni] <= 0:
                continue
            sub = fund_returns[
                (fund_returns["universe"] == uni) &
                (fund_returns["method"] == strategy_rec)
            ].set_index("date")["daily_return"]
            if not sub.empty:
                quiz_parts.append(sub * norm_r[uni])
        if quiz_parts:
            quiz_blend = pd.concat(quiz_parts, axis=1).sum(axis=1).dropna()
            quiz_growth = (1 + quiz_blend).cumprod()
            quiz_total = float(quiz_growth.iloc[-1] - 1)
            quiz_ann = float((1 + quiz_total) ** (252 / max(len(quiz_blend), 1)) - 1)
            quiz_vol = float(quiz_blend.std() * np.sqrt(252))
            quiz_sharpe = (float(quiz_blend.mean() / quiz_blend.std() * np.sqrt(252))
                           if quiz_blend.std() > 0 else 0)
            qm1, qm2, qm3, qm4 = st.columns(4)
            qm1.metric("Backtest Total Return", f"{quiz_total*100:.1f}%")
            qm2.metric("Ann. Return", f"{quiz_ann*100:.1f}%")
            qm3.metric("Ann. Volatility", f"{quiz_vol*100:.1f}%")
            qm4.metric("Sharpe Ratio", f"{quiz_sharpe:.2f}")

            # --- Smart commentary ---
            quiz_wealth = (1 + quiz_blend).cumprod()
            quiz_dd = float((quiz_wealth / quiz_wealth.cummax() - 1).min())
            quiz_insights = []
            if quiz_sharpe > 0.7:
                quiz_insights.append(
                    f"This allocation would have delivered a **strong** risk-adjusted "
                    f"return (Sharpe {quiz_sharpe:.2f}), meaning each unit of "
                    "volatility was well compensated by returns.")
            elif quiz_sharpe > 0.3:
                quiz_insights.append(
                    f"A **reasonable** risk-adjusted outcome (Sharpe {quiz_sharpe:.2f}). "
                    "Returns compensated for the risk taken, though there's room "
                    "for improvement through strategy diversification.")
            else:
                quiz_insights.append(
                    f"The backtested Sharpe of {quiz_sharpe:.2f} is **modest** — "
                    "the returns didn't fully compensate for the volatility. "
                    "Consider diversifying across multiple strategies.")

            if quiz_dd < -0.25:
                quiz_insights.append(
                    f"The worst drawdown was **{quiz_dd*100:.0f}%** — a significant "
                    "peak-to-trough loss. Make sure this aligns with your actual "
                    "comfort level before committing capital.")
            elif quiz_dd < -0.10:
                quiz_insights.append(
                    f"Maximum drawdown of {quiz_dd*100:.0f}% is manageable for most "
                    "investors with a multi-year horizon.")

            if alloc_rec["Crypto"] > 0:
                quiz_insights.append(
                    f"Your {alloc_rec['Crypto']}% crypto allocation introduces "
                    "higher tail risk — crypto drawdowns can exceed 70% in bear markets. "
                    "The diversification benefit is real but comes with episodic volatility.")

            st.markdown(
                '<div style="background:#F0FDFA; border-left:3px solid #2A9D8F; '
                'border-radius:0 8px 8px 0; padding:14px 18px; margin:8px 0 16px;">'
                '<div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
                'letter-spacing:0.08em; color:#2A9D8F; margin-bottom:6px;">'
                'What This Means For You</div>'
                '<div style="font-size:0.82rem; color:#374151; line-height:1.65;">'
                + " ".join(quiz_insights) +
                '</div></div>', unsafe_allow_html=True)

            st.caption("Educational scenario only. It does not consider your objectives, financial "
                       "situation or needs. Based on out-of-sample backtest data; past performance "
                       "is not a reliable indicator of future performance.")


# =========================================================================
# TAB 2: SENTIMENT
# =========================================================================
elif page == "Sentiment":

    sentiment = load_sector_sentiment()
    headlines = load_headline_panel()
    _headlines_available = headlines is not None
    headline_count = len(headlines) if _headlines_available else load_app_stats()["headlines"]
    headline_label = f"{headline_count:,}" if headline_count else "Historical"

    sectors = sorted(sentiment["sector"].unique())

    # Welcome guide
    st.markdown("""
    <div class="welcome-banner">
        <div class="wb-title">Welcome to Sentiment Analytics</div>
        <div class="wb-text">
            This page shows how financial news sentiment varies across 10 equity sectors.
            We score 146K+ headlines using finVADER (a finance-tuned VADER model) and convert raw scores
            into a Fear & Greed index (0–100). Explore the sector snapshot, time series, heatmap,
            and the What-If Tilt Slider to see how sentiment can reshape portfolio weights.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Headline sentiment is a noisy historical proxy, not a forecast of prices or returns. "
               "Scores should be interpreted alongside—not instead of—market and risk evidence.")
    latest_sent = (sentiment.sort_values("date")
                   .groupby("sector", as_index=False).tail(1)
                   [["date", "sector", "fear_greed", "sentiment"]]
                   .sort_values("fear_greed", ascending=False))
    overall_fg = latest_sent["fear_greed"].mean()
    overall_label = "Greed" if overall_fg > 55 else ("Fear" if overall_fg < 45 else "Neutral")
    bull = latest_sent.iloc[0]
    bear = latest_sent.iloc[-1]

    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">Sentiment Analytics</div>
        <div class="hero-title">Sector News Sentiment — finVADER</div>
        <div class="kpi-row">
            <div class="kpi-item">
                <div class="kpi-lbl">Overall F&G</div>
                <div class="kpi-val">{overall_fg:.0f} <span style="font-size:0.8rem;opacity:0.7;">{overall_label}</span></div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">Most Bullish</div>
                <div class="kpi-val">{bull['sector']}</div>
                <div class="kpi-sub">Score: {bull['fear_greed']:.0f}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">Most Bearish</div>
                <div class="kpi-val">{bear['sector']}</div>
                <div class="kpi-sub">Score: {bear['fear_greed']:.0f}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">Headlines Scored</div>
                <div class="kpi-val">{headline_label}</div>
                <div class="kpi-sub">VADER + SentiBignomics + Henry</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sector snapshot
    st.markdown(
        '<div class="section-label">Latest</div>'
        '<div class="section-title">Sector Snapshot</div>'
        '<div class="section-subtitle">Choose a sector to inspect its latest available signal. '
        '0 = extreme fear, 100 = extreme greed.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">What do these scores mean?</div>
        <div class="guide-text">
            The snapshot shows a sector's Fear & Greed score from 0 to 100. Scores below 45 indicate
            <strong>Fear</strong> — negative news tone dominates. Scores above 55 indicate <strong>Greed</strong> —
            positive sentiment prevails. The 45–55 range is <strong>Neutral</strong>. Green deltas mean the sector
            is in greed territory; red means fear. These scores are computed from actual news headlines, not market prices.
        </div>
    </div>
    """, unsafe_allow_html=True)

    snapshot_sector = st.selectbox(
        "Explore a sector", sectors, index=sectors.index("Tech") if "Tech" in sectors else 0,
        key="snapshot_sector")
    sector_history = (sentiment[sentiment["sector"] == snapshot_sector]
                      .sort_values("date").reset_index(drop=True))
    sector_latest = sector_history.iloc[-1]
    sector_previous = sector_history.iloc[-2] if len(sector_history) > 1 else sector_latest
    fg = float(sector_latest["fear_greed"])
    fg_change = fg - float(sector_previous["fear_greed"])
    fg_30 = float(sector_history["fear_greed"].tail(30).mean())
    label = "Greed" if fg > 55 else ("Fear" if fg < 45 else "Neutral")
    sector_headlines = None
    if _headlines_available and "sector" in headlines.columns:
        sector_headlines = int((headlines["sector"] == snapshot_sector).sum())

    snap1, snap2, snap3, snap4 = st.columns(4)
    snap1.metric(f"{snapshot_sector} score", f"{fg:.1f}",
                 delta=f"{fg_change:+.1f} vs prior observation")
    snap2.metric("Signal", label)
    snap3.metric("30-observation average", f"{fg_30:.1f}",
                 delta=f"{fg - fg_30:+.1f} current vs average")
    snap4.metric("Headlines analysed", f"{sector_headlines:,}" if sector_headlines is not None else "—")
    st.caption(f"Latest {snapshot_sector} observation: "
               f"{sector_latest['date'].strftime('%Y-%m-%d')}. Sector coverage dates differ, "
               "so each sector uses its own latest available observation.")

    st.markdown("---")

    # F&G time series
    st.markdown('<span id="sentiment-trends" class="section-anchor"></span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Time Series</div>'
        '<div class="section-title">Fear & Greed Index by Sector</div>'
        '<div class="section-subtitle">21-day rolling average. Select or deselect '
        'sectors to compare trends over time.</div>',
        unsafe_allow_html=True)

    sel_sectors = st.multiselect("Select Sectors", sectors, default=sectors,
                                 key="sent_sectors")
    if sel_sectors:
        st.markdown('<div class="chart-card"><div class="chart-card-title">'
                    'Fear & Greed Rolling Average</div>', unsafe_allow_html=True)
        fig, ax = clean_fig(10, 4.5)
        for sector in sel_sectors:
            color = SECTOR_COLORS.get(sector, GREY_500)
            data = sentiment[sentiment["sector"] == sector].sort_values("date")
            smoothed = data["fear_greed"].rolling(21, min_periods=1).mean()
            ax.plot(data["date"].values, smoothed.values, color=color,
                    lw=1.8, label=sector, zorder=3)
        ax.axhline(50, color=GREY_300, lw=0.8, ls="--", alpha=0.6, zorder=2)
        ax.fill_between(sentiment["date"].sort_values().unique(),
                        45, 55, alpha=0.04, color=GREY_500, zorder=1)
        ax.set_ylabel("Fear & Greed (0–100)", fontsize=9, color=GREY_500)
        ax.set_ylim(35, 75)
        ax.legend(fontsize=7.5, frameon=False, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, 1.12), handlelength=1.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Heatmap
    st.markdown(
        '<div class="section-label">Distribution</div>'
        '<div class="section-title">Sector Sentiment Heatmap</div>'
        '<div class="section-subtitle">Monthly average Fear & Greed. Colour scale '
        'centred on cross-sector median for maximum contrast.</div>',
        unsafe_allow_html=True)

    sent_monthly = sentiment.copy()
    sent_monthly["month"] = sent_monthly["date"].dt.to_period("M").astype(str)
    heatmap_data = sent_monthly.pivot_table(
        index="sector", columns="month", values="fear_greed", aggfunc="mean")
    if heatmap_data.shape[1] > 24:
        heatmap_data = heatmap_data[list(heatmap_data.columns[::2])]

    median_val = np.nanmedian(heatmap_data.values)
    spread = max(np.nanstd(heatmap_data.values) * 2.5, 3)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.5), 4.5),
                           dpi=120)
    fig.patch.set_facecolor(WHITE)
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn",
                   vmin=median_val - spread, vmax=median_val + spread)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8.5)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, fontsize=6.5, rotation=45, ha="right")
    ax.tick_params(colors=GREY_500, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, label="Fear & Greed", shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Headlines
    st.markdown(
        '<div class="section-label">News Feed</div>'
        '<div class="section-title">Headline Browser</div>'
        '<div class="section-subtitle">Browse raw headlines scored by finVADER. '
        'Filter by sector and ticker to explore sentiment drivers.</div>',
        unsafe_allow_html=True)

    if _headlines_available:
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            hl_sector = st.selectbox("Sector", ["All"] + sectors, key="hl_sector")
        with col_h2:
            hl_tickers = sorted(headlines["ticker"].dropna().unique())
            hl_ticker = st.selectbox("Ticker", ["All"] + list(hl_tickers), key="hl_ticker")
        with col_h3:
            n_headlines = st.slider("Show", 10, 100, 25, key="n_hl")

        hl = headlines.copy()
        if hl_sector != "All":
            hl = hl[hl["sector"] == hl_sector]
        if hl_ticker != "All":
            hl = hl[hl["ticker"] == hl_ticker]

        hl = hl.sort_values("trading_date", ascending=False).head(n_headlines)
        display_hl = hl[["trading_date", "ticker", "sector", "title"]].copy().reset_index(drop=True)
        display_hl["trading_date"] = display_hl["trading_date"].dt.strftime("%Y-%m-%d")
        display_hl.columns = ["Date", "Ticker", "Sector", "Headline"]
        st.markdown(
            styled_html_table(display_hl, ticker_col="Ticker", sector_col="Sector",
                              max_height=500),
            unsafe_allow_html=True)
    else:
        st.info(
            "Headline data is not available in this deployment. "
            "The headline_panel.csv (28 MB) is too large for the GitHub repo. "
            "Run locally after `python scripts/run_part_b.py` to browse headlines."
        )

    st.markdown("---")

    # Fusion with What-If Tilt Slider
    st.markdown('<span id="sentiment-fusion" class="section-anchor"></span>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Innovation</div>'
        '<div class="section-title">Sentiment Fusion — What-If Tilt Slider</div>'
        '<div class="section-subtitle">Explore how different tilt strengths affect '
        'the equity min-variance fund. Drag the slider to interpolate between '
        'the base fund (0.0) and the tested sentiment tilt (0.3).</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-bubble">
        <div class="guide-label">How the tilt slider works</div>
        <div class="guide-text">
            The slider controls how much the portfolio tilts toward sectors with improving news sentiment.
            At <strong>0.0</strong>, the fund is pure minimum-variance (no sentiment input).
            At <strong>0.3</strong>, it matches our precomputed sentiment-tilted fund.
            The chart compares the tilted fund (teal) against the base fund (grey), while the metrics on the right
            update in real time as you drag.
        </div>
    </div>
    """, unsafe_allow_html=True)

    fr = load_fund_returns()
    base = fr[(fr["universe"] == "Equity") & (fr["method"] == "Minimum-variance")].sort_values("date")
    tilted = fr[(fr["universe"] == "Equity") & (fr["method"] == "MinVar + sentiment")].sort_values("date")

    if not base.empty and not tilted.empty:
        # Tilt strength slider — interpolates daily returns between base and tilted
        # The precomputed tilted fund uses tilt_strength=0.3
        # We linearly interpolate: r_blend = r_base + (strength/0.3) * (r_tilted - r_base)
        tilt_strength = st.slider(
            "Tilt Strength", 0.0, 0.3, 0.3, 0.05, key="tilt_slider",
            help="0.0 = pure base fund; 0.3 = the precomputed and tested tilt")

        # Align on common dates
        base_aligned = base.set_index("date")["daily_return"]
        tilted_aligned = tilted.set_index("date")["daily_return"]
        common = base_aligned.index.intersection(tilted_aligned.index)
        r_base = base_aligned.loc[common]
        r_tilted = tilted_aligned.loc[common]

        # Interpolate only within the tested range; do not invent stronger portfolios.
        blend_factor = tilt_strength / 0.3 if tilt_strength > 0 else 0.0
        r_blend = r_base + blend_factor * (r_tilted - r_base)
        g_base = (1 + r_base).cumprod()
        g_blend = (1 + r_blend).cumprod()

        col_fc, col_fm = st.columns([2, 1])
        with col_fc:
            strength_label = f"tilt = {tilt_strength:.2f}"
            st.markdown('<div class="chart-card"><div class="chart-card-title">'
                        f'Growth Comparison — {strength_label}</div>',
                        unsafe_allow_html=True)
            fig, ax = clean_fig(8, 4)
            ax.fill_between(g_base.index, 1, g_base.values,
                            alpha=0.06, color=GREY_500, zorder=1)
            ax.plot(g_base.index, g_base.values, color=GREY_300, lw=2.2,
                    label="Base (no sentiment)", zorder=3)
            ax.fill_between(g_blend.index, 1, g_blend.values,
                            alpha=0.08, color=TEAL, zorder=1)
            ax.plot(g_blend.index, g_blend.values, color=TEAL, lw=2.2,
                    label=f"Sentiment-tilted ({strength_label})", zorder=3)
            ax.axhline(1, color=GREY_300, lw=0.7, ls="--", alpha=0.5, zorder=2)
            ax.set_ylabel("Value of $1", fontsize=9, color=GREY_500)
            ax.legend(fontsize=8.5, frameon=False, handlelength=1.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_fm:
            # Compute live metrics for the blended series
            n_b = len(r_blend)
            total_b = float((1 + r_blend).prod() - 1)
            ann_ret_b = float((1 + total_b) ** (252 / max(n_b, 1)) - 1)
            ann_vol_b = float(r_blend.std() * np.sqrt(252))
            sharpe_b = (float(r_blend.mean() / r_blend.std() * np.sqrt(252))
                        if r_blend.std() > 0 else 0)

            total_base = float((1 + r_base).prod() - 1)
            ann_ret_base = float((1 + total_base) ** (252 / max(len(r_base), 1)) - 1)
            sharpe_base = (float(r_base.mean() / r_base.std() * np.sqrt(252))
                           if r_base.std() > 0 else 0)

            st.metric("Base Sharpe", f"{sharpe_base:.2f}")
            st.metric("Tilted Sharpe", f"{sharpe_b:.2f}",
                       delta=f"{sharpe_b - sharpe_base:+.3f}")
            st.metric("Base Return", f"{ann_ret_base*100:.1f}%")
            st.metric("Tilted Return", f"{ann_ret_b*100:.1f}%",
                       delta=f"{(ann_ret_b - ann_ret_base)*100:+.2f}%")
            st.metric("Tilted Volatility", f"{ann_vol_b*100:.1f}%")

        # --- Smart commentary ---
        sharpe_delta = sharpe_b - sharpe_base
        ret_delta = ann_ret_b - ann_ret_base
        tilt_insights = []

        if tilt_strength == 0.0:
            tilt_insights.append(
                "At **tilt = 0.0** the fund is pure minimum-variance with no "
                "sentiment overlay. Drag the slider right to see how news "
                "sentiment reshapes the portfolio.")
        elif sharpe_delta > 0.02:
            tilt_insights.append(
                f"At this tilt strength, sentiment **improves** the Sharpe ratio "
                f"by {sharpe_delta:+.3f}. The signal is adding value — likely by "
                "rotating toward sectors with improving sentiment before the "
                "market fully prices the shift.")
        elif sharpe_delta > -0.02:
            tilt_insights.append(
                f"Sharpe impact is **negligible** ({sharpe_delta:+.3f}). At this "
                "tilt level, sentiment neither helps nor hurts meaningfully — "
                "the signal and noise roughly cancel out.")
        else:
            tilt_insights.append(
                f"Sentiment **detracts** from risk-adjusted returns "
                f"(Sharpe {sharpe_delta:+.3f}). In this sample, headline tone was "
                "a noisy signal that tilted toward sectors just before they "
                "mean-reverted. This is an honest negative result — the value "
                "is in the methodology, not guaranteed alpha.")

        if tilt_strength > 0 and abs(ret_delta) > 0.005:
            direction = "higher" if ret_delta > 0 else "lower"
            tilt_insights.append(
                f"Annualised return is **{abs(ret_delta)*100:.2f}% {direction}** "
                "than the base. Consider whether the return change justifies "
                "the added model complexity and parameter risk.")

        st.markdown(
            '<div style="background:#F0FDFA; border-left:3px solid #2A9D8F; '
            'border-radius:0 8px 8px 0; padding:14px 18px; margin:8px 0 16px;">'
            '<div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; '
            'letter-spacing:0.08em; color:#2A9D8F; margin-bottom:6px;">'
            'Tilt Insight</div>'
            '<div style="font-size:0.82rem; color:#374151; line-height:1.65;">'
            + " ".join(tilt_insights) +
            '</div></div>', unsafe_allow_html=True)


# =========================================================================
# TAB 3: DATA EXPLORER
# =========================================================================
elif page == "Data Explorer":

    st.markdown('<span id="data-browser" class="section-anchor"></span>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="hero" style="padding:20px 28px;">
        <div class="hero-label">Data Explorer</div>
        <div class="hero-title">Browse & Download Underlying Data</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-banner">
        <div class="wb-title">Full Transparency — Browse Every Dataset</div>
        <div class="wb-text">
            Everything powering the Quantise dashboard is available here for download. Select a dataset
            from the tabs below, apply filters, preview the first 100 rows, then hit the download button
            to export the full CSV. This is how we keep our research reproducible and verifiable.
        </div>
    </div>
    """, unsafe_allow_html=True)

    data_tab = st.radio("Dataset",
        ["Fund Returns", "Fund Weights", "Performance Metrics",
         "Sector Sentiment", "Headlines"], horizontal=True)
    st.markdown("---")

    if data_tab == "Fund Returns":
        fr = load_fund_returns()
        st.markdown(
            '<div class="section-title">Fund Returns</div>'
            '<div class="section-subtitle">Daily out-of-sample returns for each '
            f'strategy and fund family. <strong>{len(fr):,} rows</strong>.</div>',
            unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            uni_filter = st.multiselect("Universe", fr["universe"].unique(),
                                        default=list(fr["universe"].unique()), key="de_uni")
        with col2:
            meth_filter = st.multiselect("Method", fr["method"].unique(),
                                         default=list(fr["method"].unique()), key="de_meth")
        badges = " ".join(universe_badge(u) for u in uni_filter)
        st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)

        filtered = fr[fr["universe"].isin(uni_filter) & fr["method"].isin(meth_filter)]
        filtered = apply_sort_controls(
            filtered,
            {"Date": "date", "Daily return": "daily_return", "Growth of $1": "growth_of_1"},
            "returns_sort", "Date")
        preview = filtered.head(100).copy().reset_index(drop=True)
        preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
        for c in ["daily_return", "growth_of_1"]:
            if c in preview.columns:
                preview[c] = preview[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        st.markdown(
            styled_html_table(preview, universe_col="universe", max_height=450),
            unsafe_allow_html=True)
        st.caption(f"Showing first 100 of {len(filtered):,} rows.")
        st.download_button("Download Full CSV", filtered.to_csv(index=False),
                           "fund_returns.csv", "text/csv")

    elif data_tab == "Fund Weights":
        fw = load_fund_weights()
        st.markdown(
            '<div class="section-title">Fund Weights</div>'
            '<div class="section-subtitle">Monthly rebalance weight snapshots. '
            f'<strong>{len(fw):,} rows</strong>.</div>',
            unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            uni_f = st.selectbox("Universe", fw["universe"].unique(), key="fw_uni")
        with col2:
            meth_f = st.selectbox("Method",
                                  fw[fw["universe"] == uni_f]["method"].unique(), key="fw_meth")
        st.markdown(f'<div class="badge-row">{universe_badge(uni_f)}</div>',
                    unsafe_allow_html=True)

        filtered = fw[(fw["universe"] == uni_f) & (fw["method"] == meth_f)]
        filtered = apply_sort_controls(
            filtered, {"Date": "date", "Portfolio weight": "weight", "Ticker": "ticker"},
            "weights_sort", "Date")
        preview = filtered.head(100).copy().reset_index(drop=True)
        preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
        if "weight" in preview.columns:
            preview["weight"] = preview["weight"].apply(lambda x: f"{x*100:.2f}%")
        st.markdown(
            styled_html_table(preview, ticker_col="ticker", universe_col="universe",
                              max_height=450, compact=True),
            unsafe_allow_html=True)
        st.caption(f"Showing first 100 of {len(filtered):,} rows.")
        st.download_button("Download Full CSV", filtered.to_csv(index=False),
                           "fund_weights.csv", "text/csv")

    elif data_tab == "Performance Metrics":
        pm = load_performance_metrics()
        st.markdown(
            '<div class="section-title">Performance Metrics</div>'
            '<div class="section-subtitle">Full scorecard across all fund families, '
            f'strategies, and cost scenarios. <strong>{len(pm)} rows</strong>.</div>',
            unsafe_allow_html=True)

        sorted_pm = apply_sort_controls(
            pm,
            {"Annual return": "ann_return", "Sharpe ratio": "sharpe",
             "Annual volatility": "ann_vol", "Maximum drawdown": "max_drawdown",
             "Total return": "total_return"},
            "metrics_sort", "Annual return")
        metric_cols = ["universe", "method", "tx_cost_bps", "ann_return", "ann_vol",
                       "sharpe", "sortino", "max_drawdown", "total_return"]
        show_pm = sorted_pm[metric_cols].copy().reset_index(drop=True)
        for c in ["ann_return", "ann_vol", "max_drawdown", "total_return"]:
            if c in show_pm.columns:
                show_pm[c] = show_pm[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        for c in ["var_95", "es_95"]:
            if c in show_pm.columns:
                show_pm[c] = show_pm[c].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
        show_pm = show_pm.rename(columns={
            "universe": "Fund", "method": "Strategy", "tx_cost_bps": "Cost (bps)",
            "ann_return": "Ann. Return", "ann_vol": "Ann. Volatility",
            "sharpe": "Sharpe", "sortino": "Sortino",
            "max_drawdown": "Max Drawdown", "total_return": "Total Return",
        })
        st.markdown(
            styled_html_table(show_pm, highlight_col="Sharpe", highlight_max=True,
                              universe_col="Fund", fmt2_cols=["Sharpe", "Sortino"],
                              max_height=500),
            unsafe_allow_html=True)
        st.download_button("Download CSV", pm.to_csv(index=False),
                           "performance_metrics.csv", "text/csv")

    elif data_tab == "Sector Sentiment":
        si = load_sector_sentiment()
        st.markdown(
            '<div class="section-title">Sector Sentiment</div>'
            '<div class="section-subtitle">Daily sector-level Fear & Greed index. '
            f'<strong>{len(si):,} rows</strong>.</div>',
            unsafe_allow_html=True)
        sector_f = st.multiselect("Sector", si["sector"].unique(),
                                  default=list(si["sector"].unique()), key="si_sec")
        filtered = si[si["sector"].isin(sector_f)]
        filtered = apply_sort_controls(
            filtered,
            {"Date": "date", "Fear & Greed": "fear_greed", "Raw sentiment": "sentiment"},
            "sentiment_sort", "Date")
        preview = filtered.head(100).copy().reset_index(drop=True)
        preview["date"] = preview["date"].dt.strftime("%Y-%m-%d")
        for c in ["sentiment", "fear_greed", "sentiment_lagged"]:
            if c in preview.columns:
                preview[c] = preview[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        st.markdown(
            styled_html_table(preview, sector_col="sector", max_height=450),
            unsafe_allow_html=True)
        st.caption(f"Showing first 100 of {len(filtered):,} rows.")
        st.download_button("Download Full CSV", filtered.to_csv(index=False),
                           "sector_sentiment.csv", "text/csv")

    elif data_tab == "Headlines":
        hl = load_headline_panel()
        if hl is not None:
            st.markdown(
                '<div class="section-title">Headlines</div>'
                '<div class="section-subtitle">News headlines scored by finVADER across '
                f'50 equities and 10 sectors. <strong>{len(hl):,} rows</strong>.</div>',
                unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                sec_f = st.selectbox("Sector",
                                     ["All"] + sorted(hl["sector"].dropna().unique()),
                                     key="hl_de_sec")
            with col2:
                tick_f = st.selectbox("Ticker",
                                     ["All"] + sorted(hl["ticker"].dropna().unique()),
                                     key="hl_de_tick")
            filtered = hl.copy()
            if sec_f != "All":
                filtered = filtered[filtered["sector"] == sec_f]
            if tick_f != "All":
                filtered = filtered[filtered["ticker"] == tick_f]

            filtered = apply_sort_controls(
                filtered, {"Date": "trading_date", "Ticker": "ticker", "Sector": "sector"},
                "headlines_sort", "Date")
            preview = filtered.head(100).copy().reset_index(drop=True)
            show_cols = ["trading_date", "ticker", "sector", "title"]
            if "publisher" in preview.columns:
                show_cols.append("publisher")
            preview = preview[show_cols]
            preview["trading_date"] = preview["trading_date"].dt.strftime("%Y-%m-%d")
            preview.columns = ["Date", "Ticker", "Sector", "Headline"] + (
                ["Publisher"] if "publisher" in show_cols else [])
            st.markdown(
                styled_html_table(preview, ticker_col="Ticker", sector_col="Sector",
                                  max_height=500),
                unsafe_allow_html=True)
            st.caption(f"Showing first 100 of {len(filtered):,} matching headlines.")
        else:
            st.info(
                "Headline data is not available in this deployment. "
                "The headline_panel.csv (28 MB) is too large for the GitHub repo. "
                "Run locally to browse individual headlines."
            )
