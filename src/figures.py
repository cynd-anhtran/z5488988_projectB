"""Part B figures — FT-style charts for the funds and sentiment analytics.

All figures are self-contained with caption, labelled axes, units, and sample period.
Uses the Financial Times palette from the course.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# FT palette
FT_CREAM = "#FDF1E6"
FT_MAROON = "#990F3D"
FT_BLUE = "#0F5499"
FT_TEAL = "#2F7F73"
FT_GREY = "#6B625C"
FT_DARK = "#262A33"

# Strategy colours
STRATEGY_COLORS = {
    "Equal-weight (1/N)": "#1A1A1A",
    "Minimum-variance": FT_TEAL,
    "Max-Sharpe (tangency)": FT_MAROON,
    "Risk parity": FT_BLUE,
}

UNIVERSE_COLORS = {
    "Equity": FT_BLUE,
    "Crypto": FT_MAROON,
    "Combined": FT_TEAL,
}


def _ft_style():
    """Apply FT-style rcParams."""
    plt.rcParams.update({
        "figure.facecolor": FT_CREAM, "axes.facecolor": FT_CREAM,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
        "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
    })


def _ft_header(fig, title, subtitle, source):
    """Write FT-style title/subtitle/source onto figure."""
    fig.text(0.012, 0.96, title, fontsize=15, fontweight="bold", color=FT_DARK)
    fig.text(0.012, 0.91, subtitle, fontsize=11, color=FT_GREY)
    fig.text(0.012, 0.01, source, fontsize=8, color=FT_GREY)
    fig.subplots_adjust(top=0.86, bottom=0.12)


def _reset():
    plt.rcParams.update(plt.rcParamsDefault)


# ---------------------------------------------------------------------------
# Fund figures
# ---------------------------------------------------------------------------

def plot_growth_of_one(
    oos_returns: dict[str, pd.DataFrame],
    save_path: Path,
    title: str = "Growth of $1 — out-of-sample",
):
    """Growth of $1 for each strategy, one panel per universe.

    oos_returns: {universe_name: DataFrame(date x strategy)}
    """
    _ft_style()
    n = len(oos_returns)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (uni, df) in zip(axes, oos_returns.items()):
        for col in df.columns:
            wealth = (1.0 + df[col]).cumprod()
            color = STRATEGY_COLORS.get(col, FT_GREY)
            ax.plot(wealth.index, wealth.values, label=col, color=color, lw=1.8)
        ax.set_title(uni, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value of $1 invested")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    _ft_header(fig, title,
               "Walk-forward, long-only, monthly rebalance, expanding window",
               "Source: project data bundle | out-of-sample returns, rf = 0")
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_drawdowns(
    oos_returns: dict[str, pd.DataFrame],
    save_path: Path,
):
    """Drawdown chart for each universe."""
    _ft_style()
    n = len(oos_returns)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (uni, df) in zip(axes, oos_returns.items()):
        for col in df.columns:
            wealth = (1.0 + df[col]).cumprod()
            dd = wealth / wealth.cummax() - 1.0
            color = STRATEGY_COLORS.get(col, FT_GREY)
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
            ax.plot(dd.index, dd.values, color=color, lw=1.2, label=col)
        ax.set_title(uni, fontsize=11, fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=8, frameon=False, loc="lower left")

    _ft_header(fig, "Drawdowns — peak-to-trough losses",
               "How deep each strategy falls from its high-water mark",
               "Source: project data bundle | out-of-sample, long-only")
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_sharpe_comparison(
    metrics_df: pd.DataFrame,
    save_path: Path,
):
    """Grouped bar chart of Sharpe ratios across universes and methods."""
    _ft_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    universes = metrics_df["universe"].unique()
    methods = metrics_df["method"].unique()
    x = np.arange(len(universes))
    width = 0.8 / len(methods)

    for j, method in enumerate(methods):
        vals = []
        for uni in universes:
            row = metrics_df[(metrics_df["universe"] == uni) & (metrics_df["method"] == method)]
            vals.append(row["sharpe"].iloc[0] if len(row) > 0 else 0)
        color = STRATEGY_COLORS.get(method, FT_GREY)
        bars = ax.bar(x + (j - len(methods)/2 + 0.5) * width, vals, width,
                      color=color, label=method)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width()/2, v),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(universes)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper left")
    ax.set_ylabel("Out-of-sample Sharpe ratio")

    _ft_header(fig, "Sharpe ratio comparison across funds and methods",
               "Long-only, out-of-sample, monthly rebalance, rf = 0",
               "Source: project data bundle | annualised with sqrt(periods_per_year)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_weights_over_time(
    weights: pd.DataFrame,
    top_n: int = 10,
    save_path: Path | None = None,
    title: str = "Portfolio weights over time",
    subtitle: str = "",
):
    """Stacked area chart of portfolio weights, showing top N tickers."""
    _ft_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Show top N tickers by average weight, group the rest as "Other"
    avg_w = weights.mean().sort_values(ascending=False)
    top_tickers = avg_w.head(top_n).index.tolist()
    other = weights.drop(columns=top_tickers, errors="ignore").sum(axis=1)

    plot_data = weights[top_tickers].copy()
    if other.sum() > 0:
        plot_data["Other"] = other

    ax.stackplot(plot_data.index, plot_data.T.values,
                 labels=plot_data.columns, alpha=0.85)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)

    _ft_header(fig, title, subtitle,
               "Source: project data bundle | out-of-sample, long-only, monthly rebalance")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


# ---------------------------------------------------------------------------
# Sentiment figures
# ---------------------------------------------------------------------------

def plot_sector_sentiment(
    sector_idx: pd.DataFrame,
    save_path: Path,
):
    """Time series of sector sentiment index (fear & greed scale)."""
    _ft_style()
    sectors = sector_idx["sector"].unique()
    n_sectors = len(sectors)
    ncols = 2
    nrows = (n_sectors + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=True)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_sectors))

    for i, sector in enumerate(sorted(sectors)):
        ax = axes[i]
        data = sector_idx[sector_idx["sector"] == sector].sort_values("date")
        # Rolling 21-day average for readability
        smoothed = data["fear_greed"].rolling(21, min_periods=1).mean()
        ax.plot(data["date"].values, smoothed.values, color=colors[i], lw=1.2)
        ax.axhline(50, color=FT_GREY, lw=0.6, ls="--")
        ax.set_title(sector, fontsize=9, fontweight="bold")
        ax.set_ylim(30, 70)
        ax.set_ylabel("F&G", fontsize=8)

    # Hide unused axes
    for j in range(n_sectors, len(axes)):
        axes[j].set_visible(False)

    _ft_header(fig, "Sector sentiment index — Fear & Greed (0–100)",
               "finVADER compound score, equal-weight across tickers, 21-day rolling mean",
               "Source: project news headlines | 0 = extreme fear, 50 = neutral, 100 = extreme greed")
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_fusion_comparison(
    base_returns: pd.Series,
    tilted_returns: pd.Series,
    save_path: Path,
    base_label: str = "Base (no sentiment)",
    tilted_label: str = "Sentiment-tilted",
):
    """Before vs after: base fund vs sentiment-augmented fund."""
    _ft_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    base_wealth = (1.0 + base_returns).cumprod()
    tilted_wealth = (1.0 + tilted_returns).cumprod()

    ax.plot(base_wealth.index, base_wealth.values, color=FT_GREY, lw=2, label=base_label)
    ax.plot(tilted_wealth.index, tilted_wealth.values, color=FT_MAROON, lw=2, label=tilted_label)
    ax.set_ylabel("Value of $1 invested")
    ax.legend(fontsize=10, frameon=False)

    _ft_header(fig, "Sentiment fusion — before vs after",
               "Equity minimum-variance fund with and without sector sentiment tilt",
               "Source: project data bundle | out-of-sample, lagged sentiment, tilt_strength=0.3")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()