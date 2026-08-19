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

import pathlib

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
# Dark anchors
DARK_BG      = "#1E2B1E"   # sidebar / hero gradient start
DARK_GREEN   = "#0D1F0D"   # deepest dark
# Primary teal family
TEAL         = "#4ABEB2"   # primary accent
TEAL_DARK    = "#2A9D8F"   # hover / emphasis
TEAL_LIGHT   = "#5BC5B6"   # softer accent
CYAN         = "#4EE4C0"   # bright highlight / gradients
CYAN_PALE    = "#F1FAF8"   # very light tint for card bg
# Neutrals
WHITE        = "#FFFFFF"
GREY_50      = "#F9FAFB"
GREY_100     = "#F3F4F6"
GREY_200     = "#E5E7EB"
GREY_300     = "#D1D5DB"
GREY_500     = "#6B7280"
GREY_700     = "#374151"
GREY_900     = "#111827"
# Semantic
SUCCESS      = "#10B981"
DANGER       = "#EF4444"
WARN         = "#F59E0B"
INFO_BLUE    = "#3B82F6"

# Category colour tags (for tables / badges)
UNIVERSE_COLORS = {
    "Equity":   {"bg": "#DBEAFE", "text": "#1E40AF"},  # blue badge
    "Crypto":   {"bg": "#FDE68A", "text": "#92400E"},  # amber badge
    "Combined": {"bg": "#D1FAE5", "text": "#065F46"},  # green badge
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
STRATEGY_LABELS_SHORT = {
    "Equal-weight (1/N)":    "EW",
    "Minimum-variance":      "MinVar",
    "Max-Sharpe (tangency)": "MaxSharpe",
    "Risk parity":           "RiskPar",
    "MinVar + sentiment":    "MinVar+S",
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


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------
def clean_fig(width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=GREY_500, labelsize=8.5)
    ax.grid(True, axis="y", color=GREY_200, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


# ---------------------------------------------------------------------------
# HTML helpers for colour-tagged badges
# ---------------------------------------------------------------------------
def universe_badge(uni: str) -> str:
    c = UNIVERSE_COLORS.get(uni, {"bg": GREY_200, "text": GREY_700})
    return (f'<span style="background:{c["bg"]};color:{c["text"]};'
            f'padding:2px 10px;border-radius:10px;font-size:0.72rem;'
            f'font-weight:600;letter-spacing:0.02em;">{uni}</span>')

def sector_dot(sector: str) -> str:
    c = SECTOR_COLORS.get(sector, GREY_500)
    return (f'<span style="display:inline-block;width:8px;height:8px;'
            f'border-radius:50%;background:{c};margin-right:6px;'
            f'vertical-align:middle;"></span>'
            f'<span style="vertical-align:middle;">{sector}</span>')


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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #111827;
    }
    .main { background-color: #FFFFFF; }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E2B1E 0%, #0D1F0D 100%);
    }
    section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
    section[data-testid="stSidebar"] .stRadio label {
        border-radius: 8px;
        padding: 6px 10px;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(78, 228, 192, 0.12);
    }

    /* Typography */
    h1 {
        color: #111827 !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        font-size: 1.9rem !important;
    }
    h2, h3 {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    /* Section label */
    .section-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4ABEB2;
        margin-bottom: 4px;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 12px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #4ABEB2;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #6B7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #111827 !important;
    }

    /* DataFrames — cleaner */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Tabs / radio pill */
    div[data-testid="stHorizontalBlock"] .stRadio > div {
        gap: 4px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Dividers */
    hr { border: none; border-top: 1px solid #E5E7EB; margin: 1.5rem 0; }

    /* Download button */
    .stDownloadButton > button {
        background: #1E2B1E;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 6px 18px;
    }
    .stDownloadButton > button:hover { background: #2A9D8F; color: white; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1E2B1E 0%, #2A9D8F 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero .hero-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #4EE4C0;
        margin-bottom: 6px;
    }
    .hero .hero-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 18px;
    }
    .hero .kpi-row {
        display: flex;
        gap: 24px;
        flex-wrap: wrap;
    }
    .hero .kpi-item {
        flex: 1;
        min-width: 140px;
    }
    .hero .kpi-item .kpi-val {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .hero .kpi-item .kpi-lbl {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
        margin-bottom: 2px;
        font-weight: 600;
    }
    .hero .kpi-item .kpi-sub {
        font-size: 0.72rem;
        opacity: 0.55;
        margin-top: 2px;
    }

    /* Card wrapper */
    .card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 1rem;
    }

    /* Badge row */
    .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }

    /* Info box */
    div[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.markdown(
    "<div style='padding:8px 0;'>"
    "<span style='font-size:1.5rem;font-weight:800;letter-spacing:-0.04em;"
    "color:#4EE4C0 !important;'>Quantise</span><br>"
    "<span style='font-size:0.78rem;opacity:0.6;'>Systematic Multi-Asset Funds</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Funds", "Sentiment", "Data Explorer"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.7rem; opacity:0.45; line-height:1.7;'>"
    "FINS3645 FinTech<br>z5488988<br><br>"
    "Walk-forward OOS backtests<br>"
    "Long-only · Monthly rebalance<br>"
    "Equity &radic;252 &nbsp;|&nbsp; Crypto &radic;365"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================================
# TAB 1: FUNDS
# =========================================================================
if page == "Funds":

    fund_returns = load_fund_returns()
    fund_weights = load_fund_weights()
    metrics = load_performance_metrics()

    universes = ["Equity", "Crypto", "Combined"]
    base_methods = ["Equal-weight (1/N)", "Minimum-variance",
                    "Max-Sharpe (tangency)", "Risk parity"]

    # --- Hero banner ---
    base_m = metrics[
        (metrics["tx_cost_bps"] == 0) &
        (~metrics["method"].str.contains("sentiment|base", case=False, na=False))
    ]
    eq_best = base_m[base_m["universe"] == "Equity"].nlargest(1, "sharpe").iloc[0]
    comb_best = base_m[base_m["universe"] == "Combined"].nlargest(1, "sharpe").iloc[0]
    oos_start = fund_returns["date"].min().strftime("%b %Y")
    oos_end = fund_returns["date"].max().strftime("%b %Y")

    st.markdown(f"""
    <div class="hero">
        <div class="hero-label">Fund Dashboard</div>
        <div class="hero-title">Systematic Multi-Asset Portfolio Analytics</div>
        <div class="kpi-row">
            <div class="kpi-item">
                <div class="kpi-lbl">Total Assets</div>
                <div class="kpi-val">60</div>
                <div class="kpi-sub">50 equities + 10 crypto</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">Strategies</div>
                <div class="kpi-val">4 + Fusion</div>
                <div class="kpi-sub">+ sentiment overlay</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-lbl">Best Equity Sharpe</div>
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
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_universe = st.selectbox("Fund Family", universes, index=0)
    with col_sel2:
        available_methods = base_methods.copy()
        if selected_universe == "Equity":
            available_methods.append("MinVar + sentiment")
        selected_methods = st.multiselect("Strategies", available_methods,
                                          default=available_methods)

    if not selected_methods:
        st.warning("Please select at least one strategy.")
        st.stop()

    # Universe badge
    st.markdown(f'<div class="badge-row">{universe_badge(selected_universe)}</div>',
                unsafe_allow_html=True)

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
        k1.metric("Best Sharpe", f"{best['sharpe']:.2f}", delta=best["method"])
        k2.metric("Ann. Return", f"{best['ann_return']*100:.1f}%")
        k3.metric("Ann. Volatility", f"{best['ann_vol']*100:.1f}%")
        k4.metric("Max Drawdown", f"{best['max_drawdown']*100:.1f}%")

    st.markdown("---")

    # --- Charts side-by-side ---
    st.markdown('<div class="section-label">Performance</div>'
                '<div class="section-title">Growth of $1 & Drawdowns</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = clean_fig(7, 4)
        for method in selected_methods:
            sub = fr[fr["method"] == method].sort_values("date")
            if sub.empty:
                continue
            color = STRATEGY_COLORS.get(method, GREY_500)
            ax.plot(sub["date"], sub["growth_of_1"], label=method, color=color, lw=2)
        ax.set_ylabel("Value of $1", fontsize=9, color=GREY_500)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = clean_fig(7, 4)
        for method in selected_methods:
            sub = fr[fr["method"] == method].sort_values("date")
            if sub.empty:
                continue
            wealth = sub["growth_of_1"].values
            running_max = np.maximum.accumulate(wealth)
            dd = wealth / running_max - 1.0
            color = STRATEGY_COLORS.get(method, GREY_500)
            ax.fill_between(sub["date"], dd, 0, alpha=0.15, color=color)
            ax.plot(sub["date"], dd, color=color, lw=1.3, label=method)
        ax.set_ylabel("Drawdown", fontsize=9, color=GREY_500)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # --- Performance table with coloured universe/method tags ---
    st.markdown('<div class="section-label">Metrics</div>'
                '<div class="section-title">Performance Scorecard</div>',
                unsafe_allow_html=True)

    sel_base = metrics[
        (metrics["universe"] == selected_universe) &
        (metrics["tx_cost_bps"] == 0) &
        (metrics["method"].isin(selected_methods))
    ].copy()

    if not sel_base.empty:
        display_cols = {
            "method": "Strategy",
            "ann_return": "Ann. Return",
            "ann_vol": "Ann. Vol",
            "sharpe": "Sharpe",
            "sortino": "Sortino",
            "max_drawdown": "Max DD",
            "var_95": "VaR 95%",
            "es_95": "ES 95%",
            "total_return": "Total Return",
        }
        show = sel_base[list(display_cols.keys())].rename(columns=display_cols).reset_index(drop=True)
        for col in ["Ann. Return", "Ann. Vol", "Max DD", "Total Return"]:
            show[col] = show[col].apply(lambda x: f"{x*100:.1f}%")
        for col in ["VaR 95%", "ES 95%"]:
            show[col] = show[col].apply(lambda x: f"{x*100:.2f}%")

        # Highlight best Sharpe row
        def highlight_best(row):
            try:
                sharpe_val = float(row["Sharpe"])
                max_sharpe = sel_base["sharpe"].max()
                if abs(sharpe_val - max_sharpe) < 0.001:
                    return ["background-color: #F1FAF8; font-weight: 600"] * len(row)
            except (ValueError, TypeError):
                pass
            return [""] * len(row)

        styled = show.style.apply(highlight_best, axis=1).format(
            {"Sharpe": "{:.2f}", "Sortino": "{:.2f}"}, na_rep="—"
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- TX cost expander ---
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
            st.dataframe(tc_compare, use_container_width=True, hide_index=True)
            st.caption("Sharpe change after deducting 10 bps per unit of turnover at each monthly rebalance.")

    st.markdown("---")

    # --- Holdings ---
    st.markdown('<div class="section-label">Composition</div>'
                '<div class="section-title">Current Holdings</div>',
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
            top = latest.head(15)
            fig, ax = clean_fig(6, 5)
            # Gradient bar colours
            n_bars = len(top)
            bar_colors = [plt.cm.Greens(0.4 + 0.5 * i / max(n_bars - 1, 1))
                          for i in range(n_bars)]
            ax.barh(top["ticker"].values[::-1], top["weight"].values[::-1],
                    color=bar_colors[::-1], edgecolor="white", height=0.6)
            ax.set_xlabel("Weight", fontsize=8.5, color=GREY_500)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_table:
            disp = latest[["ticker", "weight"]].copy()
            disp["weight"] = disp["weight"].apply(lambda x: f"{x*100:.2f}%")
            disp = disp.rename(columns={"ticker": "Ticker", "weight": "Weight"})
            st.dataframe(disp, use_container_width=True, hide_index=True, height=400)

    st.markdown("---")

    # --- Allocation simulator ---
    st.markdown('<div class="section-label">Simulator</div>'
                '<div class="section-title">Set Your Allocation</div>',
                unsafe_allow_html=True)
    st.caption("Drag sliders to allocate across fund families. Results use each family's highest-Sharpe strategy.")

    alloc_cols = st.columns(3)
    allocs = {}
    for i, uni in enumerate(universes):
        with alloc_cols[i]:
            allocs[uni] = st.slider(f"{uni} (%)", 0, 100,
                                    33 if uni != "Combined" else 34, key=f"alloc_{uni}")

    total_alloc = sum(allocs.values())
    if total_alloc == 0:
        st.info("Set at least one allocation above 0%.")
    else:
        norm = {k: v / total_alloc for k, v in allocs.items()}
        best_strategy = {}
        for uni in universes:
            uni_m = metrics[
                (metrics["universe"] == uni) & (metrics["tx_cost_bps"] == 0) &
                (metrics["method"].isin(base_methods))
            ]
            if not uni_m.empty:
                best_strategy[uni] = uni_m.loc[uni_m["sharpe"].idxmax(), "method"]

        badges = " ".join(f"{universe_badge(uni)} {best_strategy.get(uni,'—')}"
                          for uni in universes if allocs[uni] > 0)
        st.markdown(f'<div style="margin-bottom:12px;">{badges}</div>',
                    unsafe_allow_html=True)

        blended_parts = []
        for uni in universes:
            if norm[uni] <= 0 or uni not in best_strategy:
                continue
            sub = fund_returns[
                (fund_returns["universe"] == uni) &
                (fund_returns["method"] == best_strategy[uni])
            ].set_index("date")["daily_return"]
            blended_parts.append(sub * norm[uni])

        if blended_parts:
            blended = pd.concat(blended_parts, axis=1).sum(axis=1).dropna()
            blended_growth = (1 + blended).cumprod()
            total_ret = float(blended_growth.iloc[-1] - 1)
            n_days = len(blended)
            ann_ret = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
            ann_vol = float(blended.std() * np.sqrt(252))
            sharpe = (float(blended.mean() / blended.std() * np.sqrt(252))
                      if blended.std() > 0 else 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{total_ret*100:.1f}%")
            m2.metric("Ann. Return", f"{ann_ret*100:.1f}%")
            m3.metric("Ann. Volatility", f"{ann_vol*100:.1f}%")
            m4.metric("Sharpe Ratio", f"{sharpe:.2f}")


# =========================================================================
# TAB 2: SENTIMENT
# =========================================================================
elif page == "Sentiment":

    sentiment = load_sector_sentiment()
    headlines = load_headline_panel()
    _headlines_available = headlines is not None

    sectors = sorted(sentiment["sector"].unique())
    latest_date = sentiment["date"].max()
    latest_sent = (sentiment[sentiment["date"] == latest_date]
                   [["sector", "fear_greed", "sentiment"]]
                   .sort_values("fear_greed", ascending=False))
    overall_fg = latest_sent["fear_greed"].mean()
    overall_label = "Greed" if overall_fg > 55 else ("Fear" if overall_fg < 45 else "Neutral")
    bull = latest_sent.iloc[0]
    bear = latest_sent.iloc[-1]

    # --- Hero ---
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
                <div class="kpi-val">146,830</div>
                <div class="kpi-sub">VADER + SentiBignomics + Henry</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Sector snapshot with colour-coded cards ---
    st.markdown('<div class="section-label">Latest</div>'
                '<div class="section-title">Sector Snapshot</div>',
                unsafe_allow_html=True)
    st.caption(f"As of {latest_date.strftime('%Y-%m-%d')}")

    row1 = st.columns(5)
    row2 = st.columns(5)
    all_cols = row1 + row2

    for i, (_, row) in enumerate(latest_sent.iterrows()):
        if i >= 10:
            break
        with all_cols[i]:
            fg = row["fear_greed"]
            label = "Greed" if fg > 55 else ("Fear" if fg < 45 else "Neutral")
            delta_color = "normal" if fg >= 50 else "inverse"
            st.metric(row["sector"], f"{fg:.0f}", delta=label, delta_color=delta_color)

    st.markdown("---")

    # --- Fear & Greed chart ---
    st.markdown('<div class="section-label">Time Series</div>'
                '<div class="section-title">Fear & Greed Index by Sector</div>',
                unsafe_allow_html=True)
    st.caption("0 = extreme fear, 50 = neutral, 100 = extreme greed. 21-day rolling mean.")

    sel_sectors = st.multiselect("Select Sectors", sectors, default=sectors,
                                 key="sent_sectors")

    if sel_sectors:
        fig, ax = clean_fig(10, 4.5)
        for sector in sel_sectors:
            color = SECTOR_COLORS.get(sector, GREY_500)
            data = sentiment[sentiment["sector"] == sector].sort_values("date")
            smoothed = data["fear_greed"].rolling(21, min_periods=1).mean()
            ax.plot(data["date"].values, smoothed.values, color=color,
                    lw=1.5, label=sector)

        ax.axhline(50, color=GREY_300, lw=0.8, ls="--", alpha=0.7)
        ax.set_ylabel("Fear & Greed (0–100)", fontsize=9, color=GREY_500)
        ax.set_ylim(35, 75)
        ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, 1.12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # --- Heatmap ---
    st.markdown('<div class="section-label">Distribution</div>'
                '<div class="section-title">Sector Sentiment Heatmap</div>',
                unsafe_allow_html=True)
    st.caption("Monthly average Fear & Greed. Scale centred on cross-sector median.")

    sent_monthly = sentiment.copy()
    sent_monthly["month"] = sent_monthly["date"].dt.to_period("M").astype(str)
    heatmap_data = sent_monthly.pivot_table(
        index="sector", columns="month", values="fear_greed", aggfunc="mean")

    if heatmap_data.shape[1] > 24:
        heatmap_data = heatmap_data[list(heatmap_data.columns[::2])]

    median_val = np.nanmedian(heatmap_data.values)
    spread = max(np.nanstd(heatmap_data.values) * 2.5, 3)

    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.45), 4.5))
    fig.patch.set_facecolor(WHITE)
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn",
                   vmin=median_val - spread, vmax=median_val + spread)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, fontsize=6, rotation=45, ha="right")
    ax.tick_params(colors=GREY_500)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, label="Fear & Greed", shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # --- Headlines ---
    st.markdown('<div class="section-label">News Feed</div>'
                '<div class="section-title">Headline Browser</div>',
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
        display_hl = hl[["trading_date", "ticker", "sector", "title"]].copy()
        display_hl["trading_date"] = display_hl["trading_date"].dt.strftime("%Y-%m-%d")
        display_hl = display_hl.rename(columns={
            "trading_date": "Date", "ticker": "Ticker",
            "sector": "Sector", "title": "Headline",
        })
        st.dataframe(display_hl, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Headline data is not available in this deployment. "
            "The headline_panel.csv (28 MB) is too large for the GitHub repo. "
            "Run the app locally with `streamlit run streamlit_app.py` after "
            "`python scripts/run_part_b.py` to browse individual headlines."
        )

    st.markdown("---")

    # --- Fusion ---
    st.markdown('<div class="section-label">Innovation</div>'
                '<div class="section-title">Sentiment Fusion — Before vs After</div>',
                unsafe_allow_html=True)
    st.caption("Equity min-var fund with and without sector sentiment tilt (strength=0.3, lagged 1 day).")

    fr = load_fund_returns()
    base = fr[(fr["universe"] == "Equity") & (fr["method"] == "Minimum-variance")].sort_values("date")
    tilted = fr[(fr["universe"] == "Equity") & (fr["method"] == "MinVar + sentiment")].sort_values("date")

    if not base.empty and not tilted.empty:
        col_fc, col_fm = st.columns([2, 1])

        with col_fc:
            fig, ax = clean_fig(8, 4)
            ax.plot(base["date"], base["growth_of_1"], color=GREY_300, lw=2.2,
                    label="Base (no sentiment)")
            ax.plot(tilted["date"], tilted["growth_of_1"], color=TEAL, lw=2.2,
                    label="Sentiment-tilted")
            ax.set_ylabel("Value of $1", fontsize=9, color=GREY_500)
            ax.legend(fontsize=8.5, frameon=False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_fm:
            met = load_performance_metrics()
            base_m = met[(met["method"] == "MinVar (base)") & (met["tx_cost_bps"] == 0)]
            tilt_m = met[(met["method"] == "MinVar + sentiment") & (met["tx_cost_bps"] == 0)]

            if not base_m.empty and not tilt_m.empty:
                bm = base_m.iloc[0]
                tm = tilt_m.iloc[0]
                st.metric("Base Sharpe", f"{bm['sharpe']:.2f}")
                st.metric("Tilted Sharpe", f"{tm['sharpe']:.2f}",
                           delta=f"{tm['sharpe'] - bm['sharpe']:+.2f}")
                st.metric("Base Return", f"{bm['ann_return']*100:.1f}%")
                st.metric("Tilted Return", f"{tm['ann_return']*100:.1f}%",
                           delta=f"{(tm['ann_return'] - bm['ann_return'])*100:+.1f}%")

        st.info(
            "The sentiment tilt slightly underperformed the base fund in this sample. "
            "Headline sentiment is a noisy signal — the tilt adds value in risk-off "
            "episodes but dilutes returns in steady markets. This is an honest negative "
            "result: the innovation is in the methodology and evidenced evaluation, "
            "not in guaranteed outperformance."
        )


# =========================================================================
# TAB 3: DATA EXPLORER
# =========================================================================
elif page == "Data Explorer":

    st.markdown("""
    <div class="hero" style="padding:20px 28px;">
        <div class="hero-label">Data Explorer</div>
        <div class="hero-title">Browse & Download Underlying Data</div>
    </div>
    """, unsafe_allow_html=True)

    data_tab = st.radio(
        "Dataset",
        ["Fund Returns", "Fund Weights", "Performance Metrics",
         "Sector Sentiment", "Headlines"],
        horizontal=True,
    )

    st.markdown("---")

    if data_tab == "Fund Returns":
        fr = load_fund_returns()
        st.markdown(f"**{len(fr):,} rows** — daily out-of-sample returns.")

        col1, col2 = st.columns(2)
        with col1:
            uni_filter = st.multiselect("Universe", fr["universe"].unique(),
                                        default=list(fr["universe"].unique()), key="de_uni")
        with col2:
            meth_filter = st.multiselect("Method", fr["method"].unique(),
                                         default=list(fr["method"].unique()), key="de_meth")

        # Show universe badges
        badges = " ".join(universe_badge(u) for u in uni_filter)
        st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)

        filtered = fr[fr["universe"].isin(uni_filter) & fr["method"].isin(meth_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
        st.download_button("Download CSV", filtered.to_csv(index=False),
                           "fund_returns.csv", "text/csv")

    elif data_tab == "Fund Weights":
        fw = load_fund_weights()
        st.markdown(f"**{len(fw):,} rows** — monthly rebalance weight snapshots.")

        col1, col2 = st.columns(2)
        with col1:
            uni_f = st.selectbox("Universe", fw["universe"].unique(), key="fw_uni")
        with col2:
            meth_f = st.selectbox("Method",
                                  fw[fw["universe"] == uni_f]["method"].unique(), key="fw_meth")

        st.markdown(f'<div class="badge-row">{universe_badge(uni_f)}</div>',
                    unsafe_allow_html=True)

        filtered = fw[(fw["universe"] == uni_f) & (fw["method"] == meth_f)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
        st.download_button("Download CSV", filtered.to_csv(index=False),
                           "fund_weights.csv", "text/csv")

    elif data_tab == "Performance Metrics":
        pm = load_performance_metrics()
        st.markdown(f"**{len(pm)} rows** — full scorecard.")
        st.dataframe(pm, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", pm.to_csv(index=False),
                           "performance_metrics.csv", "text/csv")

    elif data_tab == "Sector Sentiment":
        si = load_sector_sentiment()
        st.markdown(f"**{len(si):,} rows** — daily sector sentiment (Fear & Greed 0–100).")

        sector_f = st.multiselect("Sector", si["sector"].unique(),
                                  default=list(si["sector"].unique()), key="si_sec")
        filtered = si[si["sector"].isin(sector_f)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
        st.download_button("Download CSV", filtered.to_csv(index=False),
                           "sector_sentiment.csv", "text/csv")

    elif data_tab == "Headlines":
        hl = load_headline_panel()
        if hl is not None:
            st.markdown(f"**{len(hl):,} rows** — headlines for 50 equities, 10 sectors.")

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

            st.dataframe(
                filtered[["trading_date", "ticker", "sector", "title", "publisher"]].head(500),
                use_container_width=True, hide_index=True, height=400,
            )
            st.caption(f"Showing first 500 of {len(filtered):,} matching headlines.")
        else:
            st.info(
                "Headline data is not available in this deployment. "
                "The headline_panel.csv (28 MB) is too large for the GitHub repo. "
                "Run the app locally to browse individual headlines."
            )
