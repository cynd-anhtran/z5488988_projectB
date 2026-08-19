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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TABLES = ROOT / "results" / "tables"

# ---------------------------------------------------------------------------
# Design system — modern fintech palette
# ---------------------------------------------------------------------------
PRIMARY      = "#1B4332"   # deep forest green
PRIMARY_LT   = "#2D6A4F"   # lighter green for accents
ACCENT       = "#40916C"   # mid green
ACCENT_LT    = "#74C69D"   # light green
BG_MAIN      = "#FFFFFF"
BG_CARD      = "#F8FAF9"   # very light green-grey
BG_SIDEBAR   = "#1B4332"
TEXT_DARK     = "#1A1A2E"
TEXT_MID      = "#4A4A5A"
TEXT_LIGHT    = "#8E8E9E"
BORDER        = "#E8EDE9"
SUCCESS       = "#2D6A4F"
DANGER        = "#C0392B"
GOLD          = "#D4A853"

STRATEGY_COLORS = {
    "Equal-weight (1/N)":      "#1A1A2E",
    "Minimum-variance":        PRIMARY_LT,
    "Max-Sharpe (tangency)":   "#C0392B",
    "Risk parity":             "#2563EB",
    "MinVar + sentiment":      GOLD,
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
# Chart helper — clean modern style
# ---------------------------------------------------------------------------
def modern_fig(width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(BG_MAIN)
    ax.set_facecolor(BG_MAIN)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_MID, labelsize=9)
    ax.grid(True, color="#F0F0F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


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
# Global CSS — modern fintech design system
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import clean font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1B4332;
    }
    section[data-testid="stSidebar"] * {
        color: #E8EDE9 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(116, 198, 157, 0.15);
        border-radius: 6px;
    }

    /* Headers */
    h1 {
        color: #1B4332 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        font-size: 1.85rem !important;
    }
    h2, .stSubheader {
        color: #1A1A2E !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        margin-top: 1.8rem !important;
    }
    h3 {
        color: #1A1A2E !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #F8FAF9;
        border: 1px solid #E8EDE9;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #8E8E9E !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1A1A2E !important;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Selectboxes and inputs */
    div[data-baseweb="select"] {
        border-radius: 8px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        color: #1B4332 !important;
    }

    /* Divider replacement */
    hr {
        border: none;
        border-top: 1px solid #E8EDE9;
        margin: 1.5rem 0;
    }

    /* Caption */
    .stCaption {
        color: #8E8E9E !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #1B4332;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background-color: #2D6A4F;
        color: white;
    }

    /* Info box */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Hero KPI row */
    .hero-kpi {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
        border-radius: 14px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-kpi h2 { color: white !important; margin: 0 !important; font-size: 1.1rem !important; }
    .hero-kpi .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-top: 16px;
    }
    .hero-kpi .kpi-item .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.75;
        font-weight: 500;
    }
    .hero-kpi .kpi-item .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .hero-kpi .kpi-item .kpi-sub {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 2px;
    }

    /* Card container */
    .card {
        background: #F8FAF9;
        border: 1px solid #E8EDE9;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 1rem;
    }
    .card h3 { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.markdown(
    "<h1 style='font-size:1.6rem; font-weight:700; letter-spacing:-0.03em; "
    "margin-bottom:0; color:white !important;'>Quantise</h1>"
    "<p style='font-size:0.82rem; opacity:0.7; margin-top:4px;'>Systematic Multi-Asset Funds</p>",
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
    "<div style='font-size:0.72rem; opacity:0.55; line-height:1.6;'>"
    "FINS3645 FinTech Project<br>z5488988<br><br>"
    "Out-of-sample backtests<br>"
    "Long-only, monthly rebalance<br>"
    "Equity ann. factor &radic;252<br>"
    "Crypto ann. factor &radic;365"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================================
# TAB 1: FUNDS
# =========================================================================
if page == "Funds":

    # --- Header ---
    st.title("Fund Dashboard")
    st.caption("Compare systematically managed funds, review performance, and set your allocation.")

    fund_returns = load_fund_returns()
    fund_weights = load_fund_weights()
    metrics = load_performance_metrics()

    universes = ["Equity", "Crypto", "Combined"]
    base_methods = ["Equal-weight (1/N)", "Minimum-variance", "Max-Sharpe (tangency)", "Risk parity"]

    # --- Hero KPI banner ---
    base_m = metrics[
        (metrics["tx_cost_bps"] == 0) &
        (~metrics["method"].str.contains("sentiment|base", case=False, na=False))
    ]
    eq_best = base_m[base_m["universe"] == "Equity"].nlargest(1, "sharpe").iloc[0]
    combined_best = base_m[base_m["universe"] == "Combined"].nlargest(1, "sharpe").iloc[0]

    oos_start = fund_returns["date"].min().strftime("%b %Y")
    oos_end = fund_returns["date"].max().strftime("%b %Y")

    st.markdown(f"""
    <div class="hero-kpi">
        <h2>Portfolio Overview</h2>
        <div class="kpi-grid">
            <div class="kpi-item">
                <div class="kpi-label">Total Assets</div>
                <div class="kpi-value">60</div>
                <div class="kpi-sub">50 equities + 10 crypto</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Strategies</div>
                <div class="kpi-value">4</div>
                <div class="kpi-sub">+ sentiment fusion</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Best Equity Sharpe</div>
                <div class="kpi-value">{eq_best['sharpe']:.2f}</div>
                <div class="kpi-sub">{eq_best['method']}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">OOS Period</div>
                <div class="kpi-value">{oos_start} – {oos_end}</div>
                <div class="kpi-sub">Walk-forward backtest</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Fund selector ---
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_universe = st.selectbox("Fund Family", universes, index=0)
    with col_sel2:
        available_methods = base_methods.copy()
        if selected_universe == "Equity":
            available_methods.append("MinVar + sentiment")
        selected_methods = st.multiselect(
            "Strategies",
            available_methods,
            default=available_methods,
        )

    if not selected_methods:
        st.warning("Please select at least one strategy.")
        st.stop()

    # Filter data
    fr = fund_returns[
        (fund_returns["universe"] == selected_universe) &
        (fund_returns["method"].isin(selected_methods))
    ].copy()

    # --- Performance KPIs for selected fund ---
    sel_metrics = metrics[
        (metrics["universe"] == selected_universe) &
        (metrics["tx_cost_bps"] == 0) &
        (metrics["method"].isin(selected_methods))
    ]
    if not sel_metrics.empty:
        best = sel_metrics.loc[sel_metrics["sharpe"].idxmax()]
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Best Sharpe", f"{best['sharpe']:.2f}", delta=best["method"])
        kpi_cols[1].metric("Ann. Return", f"{best['ann_return']*100:.1f}%")
        kpi_cols[2].metric("Ann. Volatility", f"{best['ann_vol']*100:.1f}%")
        kpi_cols[3].metric("Max Drawdown", f"{best['max_drawdown']*100:.1f}%")

    st.markdown("---")

    # --- Growth of $1 & Drawdown side by side ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Growth of $1 — Out-of-Sample")
        fig, ax = modern_fig(7, 4)
        for method in selected_methods:
            sub = fr[fr["method"] == method].sort_values("date")
            if sub.empty:
                continue
            color = STRATEGY_COLORS.get(method, TEXT_LIGHT)
            ax.plot(sub["date"], sub["growth_of_1"], label=method, color=color, lw=1.8)
        ax.set_ylabel("Value of $1", fontsize=10, color=TEXT_MID)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with chart_col2:
        st.markdown("#### Drawdowns")
        fig, ax = modern_fig(7, 4)
        for method in selected_methods:
            sub = fr[fr["method"] == method].sort_values("date")
            if sub.empty:
                continue
            wealth = sub["growth_of_1"].values
            running_max = np.maximum.accumulate(wealth)
            dd = wealth / running_max - 1.0
            color = STRATEGY_COLORS.get(method, TEXT_LIGHT)
            ax.fill_between(sub["date"], dd, 0, alpha=0.2, color=color)
            ax.plot(sub["date"], dd, color=color, lw=1.2, label=method)
        ax.set_ylabel("Drawdown", fontsize=10, color=TEXT_MID)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # --- Performance metrics table ---
    st.markdown("#### Performance Metrics")

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
        st.dataframe(show, use_container_width=True, hide_index=True)

    # --- Transaction cost impact ---
    tc_data = metrics[
        (metrics["universe"] == selected_universe) &
        (metrics["method"].isin(base_methods))
    ].copy()
    tc_0 = tc_data[tc_data["tx_cost_bps"] == 0][["method", "sharpe"]].rename(columns={"sharpe": "Sharpe (0 bps)"})
    tc_10 = tc_data[tc_data["tx_cost_bps"] == 10][["method", "sharpe"]].rename(columns={"sharpe": "Sharpe (10 bps)"})

    if not tc_0.empty and not tc_10.empty:
        tc_compare = tc_0.merge(tc_10, on="method")
        tc_compare["Impact"] = tc_compare.apply(
            lambda r: f"{r['Sharpe (10 bps)'] - r['Sharpe (0 bps)']:+.2f}", axis=1
        )
        tc_compare = tc_compare.rename(columns={"method": "Strategy"})
        with st.expander("Transaction Cost Impact (10 bps one-way)"):
            st.dataframe(tc_compare, use_container_width=True, hide_index=True)
            st.caption("Sharpe change after deducting 10 bps per unit of turnover at each monthly rebalance.")

    st.markdown("---")

    # --- Holdings ---
    st.markdown("#### Current Holdings")

    hold_col1, hold_col2 = st.columns([1, 2])
    with hold_col1:
        holdings_method = st.selectbox("Strategy", selected_methods, key="holdings_strategy")

    fw_sel = fund_weights[
        (fund_weights["universe"] == selected_universe) &
        (fund_weights["method"] == holdings_method) &
        (fund_weights["weight"] > 0.001)
    ]

    if not fw_sel.empty:
        latest_date = fw_sel["date"].max()
        latest = fw_sel[fw_sel["date"] == latest_date].sort_values("weight", ascending=False)
        st.caption(f"As of {latest_date.strftime('%Y-%m-%d')} (last rebalance) · {len(latest)} positions")

        col_chart, col_table = st.columns([1.3, 1])
        with col_chart:
            top = latest.head(15)
            fig, ax = modern_fig(6, 4.5)
            colors_bar = [PRIMARY_LT if i == 0 else ACCENT_LT for i in range(len(top))]
            ax.barh(
                top["ticker"].values[::-1],
                top["weight"].values[::-1],
                color=colors_bar[::-1],
                edgecolor="white",
                height=0.65,
            )
            ax.set_xlabel("Weight", fontsize=9, color=TEXT_MID)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_table:
            disp = latest[["ticker", "weight"]].copy()
            disp["weight"] = disp["weight"].apply(lambda x: f"{x*100:.2f}%")
            disp = disp.rename(columns={"ticker": "Ticker", "weight": "Weight"})
            st.dataframe(disp, use_container_width=True, hide_index=True, height=380)

    st.markdown("---")

    # --- Allocation simulator ---
    st.markdown("#### Allocation Simulator")
    st.caption("Drag sliders to allocate capital across fund families and see blended performance.")

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
                (metrics["universe"] == uni) &
                (metrics["tx_cost_bps"] == 0) &
                (metrics["method"].isin(base_methods))
            ]
            if not uni_m.empty:
                best_strategy[uni] = uni_m.loc[uni_m["sharpe"].idxmax(), "method"]

        st.caption(
            "Blended using each family's highest-Sharpe strategy: "
            + ", ".join(f"{uni}: {best_strategy.get(uni, 'N/A')}" for uni in universes)
        )

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
            sharpe = float(blended.mean() / blended.std() * np.sqrt(252)) if blended.std() > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{total_ret*100:.1f}%")
            m2.metric("Ann. Return", f"{ann_ret*100:.1f}%")
            m3.metric("Ann. Volatility", f"{ann_vol*100:.1f}%")
            m4.metric("Sharpe Ratio", f"{sharpe:.2f}")


# =========================================================================
# TAB 2: SENTIMENT
# =========================================================================
elif page == "Sentiment":
    st.title("Sector Sentiment Analytics")
    st.caption(
        "News-sentiment index across 10 equity sectors, scored with "
        "finVADER (VADER + SentiBignomics + Henry finance lexicons, ~7,500 finance-specific terms)."
    )

    sentiment = load_sector_sentiment()
    headlines = load_headline_panel()
    _headlines_available = headlines is not None

    # --- Sentiment hero KPIs ---
    latest_date = sentiment["date"].max()
    latest_sent = sentiment[sentiment["date"] == latest_date][["sector", "fear_greed", "sentiment"]].sort_values("fear_greed", ascending=False)
    overall_fg = latest_sent["fear_greed"].mean()
    overall_label = "Greed" if overall_fg > 55 else ("Fear" if overall_fg < 45 else "Neutral")
    most_bullish = latest_sent.iloc[0]
    most_bearish = latest_sent.iloc[-1]

    st.markdown(f"""
    <div class="hero-kpi">
        <h2>Market Sentiment Overview</h2>
        <div class="kpi-grid">
            <div class="kpi-item">
                <div class="kpi-label">Overall Sentiment</div>
                <div class="kpi-value">{overall_fg:.0f}</div>
                <div class="kpi-sub">{overall_label}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Most Bullish Sector</div>
                <div class="kpi-value">{most_bullish['sector']}</div>
                <div class="kpi-sub">F&G: {most_bullish['fear_greed']:.0f}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Most Bearish Sector</div>
                <div class="kpi-value">{most_bearish['sector']}</div>
                <div class="kpi-sub">F&G: {most_bearish['fear_greed']:.0f}</div>
            </div>
            <div class="kpi-item">
                <div class="kpi-label">Headlines Scored</div>
                <div class="kpi-value">146,830</div>
                <div class="kpi-sub">finVADER compound score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Fear & Greed line chart ---
    st.markdown("#### Fear & Greed Index by Sector")
    st.caption("0 = extreme fear, 50 = neutral, 100 = extreme greed. 21-day rolling mean.")

    sectors = sorted(sentiment["sector"].unique())
    sel_sectors = st.multiselect("Select Sectors", sectors, default=sectors, key="sent_sectors")

    if sel_sectors:
        fig, ax = modern_fig(10, 4.5)
        cmap = plt.cm.Set2(np.linspace(0, 1, len(sel_sectors)))

        for i, sector in enumerate(sel_sectors):
            data = sentiment[sentiment["sector"] == sector].sort_values("date")
            smoothed = data["fear_greed"].rolling(21, min_periods=1).mean()
            ax.plot(data["date"].values, smoothed.values, color=cmap[i], lw=1.4, label=sector)

        ax.axhline(50, color=TEXT_LIGHT, lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel("Fear & Greed (0-100)", fontsize=10, color=TEXT_MID)
        ax.set_ylim(35, 75)
        ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # --- Sector snapshot cards ---
    st.markdown("#### Sector Snapshot")
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

    # --- Sentiment heatmap ---
    st.markdown("#### Sector Sentiment Heatmap")
    st.caption("Monthly average Fear & Greed score. Colour scale centred on the cross-sector median.")

    sent_monthly = sentiment.copy()
    sent_monthly["month"] = sent_monthly["date"].dt.to_period("M").astype(str)
    heatmap_data = sent_monthly.pivot_table(
        index="sector", columns="month", values="fear_greed", aggfunc="mean"
    )

    if heatmap_data.shape[1] > 24:
        cols_to_show = list(heatmap_data.columns[::2])
        heatmap_data = heatmap_data[cols_to_show]

    median_val = np.nanmedian(heatmap_data.values)
    spread = max(np.nanstd(heatmap_data.values) * 2.5, 3)
    vmin_heat = median_val - spread
    vmax_heat = median_val + spread

    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.45), 4.5))
    fig.patch.set_facecolor(BG_MAIN)
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn",
                   vmin=vmin_heat, vmax=vmax_heat)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=8.5)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, fontsize=6.5, rotation=45, ha="right")
    ax.tick_params(colors=TEXT_MID)
    fig.colorbar(im, ax=ax, label="Fear & Greed", shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # --- Headline feed ---
    st.markdown("#### Headline Feed")

    if _headlines_available:
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            hl_sector = st.selectbox("Sector", ["All"] + sectors, key="hl_sector")
        with col_h2:
            hl_tickers = sorted(headlines["ticker"].dropna().unique())
            hl_ticker = st.selectbox("Ticker", ["All"] + list(hl_tickers), key="hl_ticker")
        with col_h3:
            n_headlines = st.slider("Headlines to show", 10, 100, 25, key="n_hl")

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

    # --- Fusion comparison ---
    st.markdown("#### Sentiment Fusion — Before vs After")
    st.caption(
        "Equity min-var fund with and without sector sentiment tilt "
        "(tilt_strength=0.3, lagged 1 day)."
    )

    fr = load_fund_returns()
    base = fr[(fr["universe"] == "Equity") & (fr["method"] == "Minimum-variance")].sort_values("date")
    tilted = fr[(fr["universe"] == "Equity") & (fr["method"] == "MinVar + sentiment")].sort_values("date")

    if not base.empty and not tilted.empty:
        col_fusion_chart, col_fusion_metrics = st.columns([2, 1])

        with col_fusion_chart:
            fig, ax = modern_fig(8, 4)
            ax.plot(base["date"], base["growth_of_1"], color=TEXT_LIGHT, lw=2, label="Base (no sentiment)")
            ax.plot(tilted["date"], tilted["growth_of_1"], color=PRIMARY, lw=2, label="Sentiment-tilted")
            ax.set_ylabel("Value of $1", fontsize=10, color=TEXT_MID)
            ax.legend(fontsize=9, frameon=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_fusion_metrics:
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
    st.title("Data Explorer")
    st.caption("Browse and download the underlying data powering Quantise.")

    data_tab = st.radio(
        "Dataset",
        ["Fund Returns", "Fund Weights", "Performance Metrics", "Sector Sentiment", "Headlines"],
        horizontal=True,
    )

    st.markdown("---")

    if data_tab == "Fund Returns":
        fr = load_fund_returns()
        st.markdown(f"**{len(fr):,} rows** — daily out-of-sample returns for each fund/strategy combination.")

        col1, col2 = st.columns(2)
        with col1:
            uni_filter = st.multiselect("Universe", fr["universe"].unique(),
                                        default=list(fr["universe"].unique()), key="de_uni")
        with col2:
            meth_filter = st.multiselect("Method", fr["method"].unique(),
                                         default=list(fr["method"].unique()), key="de_meth")

        filtered = fr[fr["universe"].isin(uni_filter) & fr["method"].isin(meth_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
        st.download_button("Download CSV", filtered.to_csv(index=False), "fund_returns.csv", "text/csv")

    elif data_tab == "Fund Weights":
        fw = load_fund_weights()
        st.markdown(f"**{len(fw):,} rows** — portfolio weight snapshots at each monthly rebalance.")

        col1, col2 = st.columns(2)
        with col1:
            uni_f = st.selectbox("Universe", fw["universe"].unique(), key="fw_uni")
        with col2:
            meth_f = st.selectbox("Method", fw[fw["universe"] == uni_f]["method"].unique(), key="fw_meth")

        filtered = fw[(fw["universe"] == uni_f) & (fw["method"] == meth_f)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
        st.download_button("Download CSV", filtered.to_csv(index=False), "fund_weights.csv", "text/csv")

    elif data_tab == "Performance Metrics":
        pm = load_performance_metrics()
        st.markdown(f"**{len(pm)} rows** — full scorecard for every fund/strategy/cost combination.")
        st.dataframe(pm, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", pm.to_csv(index=False), "performance_metrics.csv", "text/csv")

    elif data_tab == "Sector Sentiment":
        si = load_sector_sentiment()
        st.markdown(f"**{len(si):,} rows** — daily sector-level sentiment index (Fear & Greed 0-100).")

        sector_f = st.multiselect("Sector", si["sector"].unique(),
                                  default=list(si["sector"].unique()), key="si_sec")
        filtered = si[si["sector"].isin(sector_f)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)
        st.download_button("Download CSV", filtered.to_csv(index=False), "sector_sentiment.csv", "text/csv")

    elif data_tab == "Headlines":
        hl = load_headline_panel()
        if hl is not None:
            st.markdown(f"**{len(hl):,} rows** — news headlines aligned to trading days for 50 equities across 10 sectors.")

            col1, col2 = st.columns(2)
            with col1:
                sec_f = st.selectbox("Sector", ["All"] + sorted(hl["sector"].dropna().unique()), key="hl_de_sec")
            with col2:
                tick_f = st.selectbox("Ticker", ["All"] + sorted(hl["ticker"].dropna().unique()), key="hl_de_tick")

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
