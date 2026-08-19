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
# FT colour palette
# ---------------------------------------------------------------------------
FT_CREAM   = "#FDF1E6"
FT_MAROON  = "#990F3D"
FT_BLUE    = "#0F5499"
FT_TEAL    = "#2F7F73"
FT_GREY    = "#6B625C"
FT_DARK    = "#262A33"
FT_GOLD    = "#C9922A"

STRATEGY_COLORS = {
    "Equal-weight (1/N)":      "#1A1A1A",
    "Minimum-variance":        FT_TEAL,
    "Max-Sharpe (tangency)":   FT_MAROON,
    "Risk parity":             FT_BLUE,
    "MinVar + sentiment":      FT_GOLD,
}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_fund_returns():
    df = pd.read_csv(DATA / "fund_returns.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_fund_weights():
    df = pd.read_csv(DATA / "fund_weights.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_performance_metrics():
    df = pd.read_csv(TABLES / "performance_metrics.csv")
    return df

@st.cache_data
def load_sector_sentiment():
    df = pd.read_csv(DATA / "sector_sentiment_index.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_headline_panel():
    df = pd.read_csv(
        DATA / "headline_panel.csv",
        parse_dates=["trading_date"],
        dtype={"publisher": str},
        low_memory=False,
    )
    return df

@st.cache_data
def load_tx_cost_comparison():
    path = TABLES / "tx_cost_comparison.csv"
    if path.exists():
        return pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    return None


# ---------------------------------------------------------------------------
# Helper: matplotlib figure in FT style
# ---------------------------------------------------------------------------
def ft_fig(width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(FT_CREAM)
    ax.set_facecolor(FT_CREAM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(True, color="#E2D8CF", zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="Quantise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polish
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    h1 { color: #990F3D !important; }
    .stMetric label { font-size: 0.85rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.title("Quantise")
st.sidebar.caption("Systematic Multi-Asset Funds")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Funds", "Sentiment", "Data Explorer"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#6B625C;'>"
    "FINS3645 FinTech Project &mdash; z5488988<br>"
    "Out-of-sample, long-only, monthly rebalance<br>"
    "Equity &radic;252 &nbsp;|&nbsp; Crypto &radic;365"
    "</small>",
    unsafe_allow_html=True,
)


# =========================================================================
# TAB 1: FUNDS
# =========================================================================
if page == "Funds":
    st.title("Fund Dashboard")
    st.markdown("Compare our systematically managed funds, review fact sheets, and set your allocation.")

    fund_returns = load_fund_returns()
    fund_weights = load_fund_weights()
    metrics = load_performance_metrics()

    # --- Fund selector ---
    universes = ["Equity", "Crypto", "Combined"]
    base_methods = ["Equal-weight (1/N)", "Minimum-variance", "Max-Sharpe (tangency)", "Risk parity"]

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_universe = st.selectbox("Select Fund Family", universes, index=0)
    with col_sel2:
        available_methods = base_methods.copy()
        if selected_universe == "Equity":
            available_methods.append("MinVar + sentiment")
        selected_methods = st.multiselect(
            "Select Strategies",
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

    # --- Growth of $1 chart ---
    st.subheader("Growth of $1 — Out-of-Sample")

    fig, ax = ft_fig(10, 5)
    for method in selected_methods:
        sub = fr[fr["method"] == method].sort_values("date")
        if sub.empty:
            continue
        color = STRATEGY_COLORS.get(method, FT_GREY)
        ax.plot(sub["date"], sub["growth_of_1"], label=method, color=color, lw=1.8)

    ax.set_ylabel("Value of $1 invested", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Drawdown chart ---
    st.subheader("Drawdowns — Peak-to-Trough Losses")

    fig, ax = ft_fig(10, 4)
    for method in selected_methods:
        sub = fr[fr["method"] == method].sort_values("date")
        if sub.empty:
            continue
        wealth = sub["growth_of_1"].values
        running_max = np.maximum.accumulate(wealth)
        dd = wealth / running_max - 1.0
        color = STRATEGY_COLORS.get(method, FT_GREY)
        ax.fill_between(sub["date"], dd, 0, alpha=0.25, color=color)
        ax.plot(sub["date"], dd, color=color, lw=1.2, label=method)

    ax.set_ylabel("Drawdown", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Performance metrics table ---
    st.subheader("Performance Metrics")

    base_metrics = metrics[
        (metrics["universe"] == selected_universe) &
        (metrics["tx_cost_bps"] == 0) &
        (metrics["method"].isin(selected_methods))
    ].copy()

    if not base_metrics.empty:
        display_cols = {
            "method": "Strategy",
            "ann_return": "Ann. Return",
            "ann_vol": "Ann. Volatility",
            "sharpe": "Sharpe",
            "sortino": "Sortino",
            "max_drawdown": "Max Drawdown",
            "var_95": "VaR (95%)",
            "es_95": "ES (95%)",
            "total_return": "Total Return",
        }
        show = base_metrics[list(display_cols.keys())].rename(columns=display_cols).reset_index(drop=True)

        # Format percentages
        for col in ["Ann. Return", "Ann. Volatility", "Max Drawdown", "Total Return"]:
            show[col] = show[col].apply(lambda x: f"{x*100:.1f}%")
        for col in ["VaR (95%)", "ES (95%)"]:
            show[col] = show[col].apply(lambda x: f"{x*100:.2f}%")

        st.dataframe(show, use_container_width=True, hide_index=True)

    # --- Transaction cost impact (innovation) ---
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
            st.caption("Sharpe ratio change after deducting 10 basis points per unit of turnover at each monthly rebalance.")

    # --- Current holdings ---
    st.subheader("Current Holdings")

    # Show the latest weight snapshot for selected strategy
    holdings_method = st.selectbox(
        "Strategy",
        selected_methods,
        key="holdings_strategy",
    )
    fw_sel = fund_weights[
        (fund_weights["universe"] == selected_universe) &
        (fund_weights["method"] == holdings_method) &
        (fund_weights["weight"] > 0.001)
    ]

    if not fw_sel.empty:
        latest_date = fw_sel["date"].max()
        latest = fw_sel[fw_sel["date"] == latest_date].sort_values("weight", ascending=False)
        st.caption(f"Weights as of {latest_date.strftime('%Y-%m-%d')} (last rebalance)")

        col_chart, col_table = st.columns([1.2, 1])
        with col_chart:
            top = latest.head(15)
            fig, ax = ft_fig(6, 4)
            bars = ax.barh(
                top["ticker"][::-1],
                top["weight"].values[::-1],
                color=FT_TEAL,
                edgecolor="white",
                height=0.7,
            )
            ax.set_xlabel("Weight", fontsize=10)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_table:
            disp = latest[["ticker", "weight"]].copy()
            disp["weight"] = disp["weight"].apply(lambda x: f"{x*100:.2f}%")
            disp = disp.rename(columns={"ticker": "Ticker", "weight": "Weight"})
            st.dataframe(disp, use_container_width=True, hide_index=True, height=350)

    # --- Allocation simulator ---
    st.subheader("Set Your Allocation")
    st.markdown("Drag the sliders to allocate capital across fund families and see your blended performance.")

    alloc_cols = st.columns(3)
    allocs = {}
    for i, uni in enumerate(universes):
        with alloc_cols[i]:
            allocs[uni] = st.slider(f"{uni} (%)", 0, 100, 33 if uni != "Combined" else 34, key=f"alloc_{uni}")

    total_alloc = sum(allocs.values())
    if total_alloc == 0:
        st.info("Set at least one allocation above 0%.")
    else:
        # Normalise
        norm = {k: v / total_alloc for k, v in allocs.items()}

        # Use best Sharpe strategy for each universe (from base methods)
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

        # Build blended daily returns
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
    st.markdown(
        "News-sentiment index across 10 equity sectors, scored with "
        "**finVADER** (VADER + SentiBignomics + Henry finance lexicons, ~7,500 finance-specific terms)."
    )

    sentiment = load_sector_sentiment()
    headlines = load_headline_panel()

    # --- Fear & Greed overview ---
    st.subheader("Fear & Greed Index by Sector")
    st.caption("0 = extreme fear, 50 = neutral, 100 = extreme greed. 21-day rolling mean for readability.")

    sectors = sorted(sentiment["sector"].unique())

    # Sector selector
    sel_sectors = st.multiselect("Select Sectors", sectors, default=sectors, key="sent_sectors")

    if sel_sectors:
        fig, ax = ft_fig(10, 5)
        colors = plt.cm.tab10(np.linspace(0, 1, len(sel_sectors)))

        for i, sector in enumerate(sel_sectors):
            data = sentiment[sentiment["sector"] == sector].sort_values("date")
            smoothed = data["fear_greed"].rolling(21, min_periods=1).mean()
            ax.plot(data["date"].values, smoothed.values, color=colors[i], lw=1.2, label=sector)

        ax.axhline(50, color=FT_GREY, lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel("Fear & Greed (0-100)", fontsize=11)
        ax.set_ylim(30, 70)
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Latest sentiment snapshot ---
    st.subheader("Latest Sentiment Snapshot")

    latest_date = sentiment["date"].max()
    latest_sent = sentiment[sentiment["date"] == latest_date][["sector", "fear_greed", "sentiment"]].sort_values("fear_greed", ascending=False)

    cols_snap = st.columns(5)
    for i, (_, row) in enumerate(latest_sent.iterrows()):
        with cols_snap[i % 5]:
            fg = row["fear_greed"]
            label = "Greed" if fg > 55 else ("Fear" if fg < 45 else "Neutral")
            delta_color = "normal" if fg >= 50 else "inverse"
            st.metric(
                row["sector"],
                f"{fg:.0f}",
                delta=label,
                delta_color=delta_color,
            )

    st.caption(f"As of {latest_date.strftime('%Y-%m-%d')}")

    # --- Sentiment heatmap ---
    st.subheader("Sector Sentiment Heatmap")
    st.caption("Monthly average Fear & Greed score across sectors.")

    sent_monthly = sentiment.copy()
    sent_monthly["month"] = sent_monthly["date"].dt.to_period("M").astype(str)
    heatmap_data = sent_monthly.pivot_table(
        index="sector", columns="month", values="fear_greed", aggfunc="mean"
    )

    # Subsample columns if too many
    if heatmap_data.shape[1] > 24:
        cols_to_show = list(heatmap_data.columns[::2])  # every other month
        heatmap_data = heatmap_data[cols_to_show]

    fig, ax = plt.subplots(figsize=(max(12, len(heatmap_data.columns) * 0.4), 5))
    fig.patch.set_facecolor(FT_CREAM)
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn", vmin=35, vmax=65)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=9)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, fontsize=7, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="Fear & Greed", shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Headline feed ---
    st.subheader("Headline Feed")

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
        "trading_date": "Date",
        "ticker": "Ticker",
        "sector": "Sector",
        "title": "Headline",
    })
    st.dataframe(display_hl, use_container_width=True, hide_index=True)

    # --- Fusion comparison ---
    st.subheader("Sentiment Fusion — Before vs After")
    st.markdown(
        "Equity minimum-variance fund with and without sector sentiment tilt "
        "(tilt_strength=0.3, lagged 1 day)."
    )

    fr = load_fund_returns()
    base = fr[(fr["universe"] == "Equity") & (fr["method"] == "Minimum-variance")].sort_values("date")
    tilted = fr[(fr["universe"] == "Equity") & (fr["method"] == "MinVar + sentiment")].sort_values("date")

    if not base.empty and not tilted.empty:
        fig, ax = ft_fig(10, 4)
        ax.plot(base["date"], base["growth_of_1"], color=FT_GREY, lw=2, label="Base (no sentiment)")
        ax.plot(tilted["date"], tilted["growth_of_1"], color=FT_MAROON, lw=2, label="Sentiment-tilted")
        ax.set_ylabel("Value of $1 invested", fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        met = load_performance_metrics()
        base_m = met[(met["method"] == "MinVar (base)") & (met["tx_cost_bps"] == 0)]
        tilt_m = met[(met["method"] == "MinVar + sentiment") & (met["tx_cost_bps"] == 0)]

        if not base_m.empty and not tilt_m.empty:
            c1, c2, c3, c4 = st.columns(4)
            bm = base_m.iloc[0]
            tm = tilt_m.iloc[0]
            c1.metric("Base Sharpe", f"{bm['sharpe']:.2f}")
            c2.metric("Tilted Sharpe", f"{tm['sharpe']:.2f}", delta=f"{tm['sharpe'] - bm['sharpe']:+.2f}")
            c3.metric("Base Ann. Return", f"{bm['ann_return']*100:.1f}%")
            c4.metric("Tilted Ann. Return", f"{tm['ann_return']*100:.1f}%", delta=f"{(tm['ann_return'] - bm['ann_return'])*100:+.1f}%")

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
    st.markdown("Explore the underlying data powering Quantise.")

    data_tab = st.radio(
        "Dataset",
        ["Fund Returns", "Fund Weights", "Performance Metrics", "Sector Sentiment", "Headlines"],
        horizontal=True,
    )

    if data_tab == "Fund Returns":
        fr = load_fund_returns()
        st.markdown(f"**{len(fr):,} rows** — daily out-of-sample returns for each fund/strategy combination.")

        col1, col2 = st.columns(2)
        with col1:
            uni_filter = st.multiselect("Universe", fr["universe"].unique(), default=list(fr["universe"].unique()), key="de_uni")
        with col2:
            meth_filter = st.multiselect("Method", fr["method"].unique(), default=list(fr["method"].unique()), key="de_meth")

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

        sector_f = st.multiselect("Sector", si["sector"].unique(), default=list(si["sector"].unique()), key="si_sec")
        filtered = si[si["sector"].isin(sector_f)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=400)

        st.download_button("Download CSV", filtered.to_csv(index=False), "sector_sentiment.csv", "text/csv")

    elif data_tab == "Headlines":
        hl = load_headline_panel()
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
            use_container_width=True,
            hide_index=True,
            height=400,
        )
        st.caption(f"Showing first 500 of {len(filtered):,} matching headlines.")