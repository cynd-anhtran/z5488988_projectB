"""Part B figures — Quantise teal/cyan palette charts for funds and sentiment.

All figures are self-contained with caption, labelled axes, units, and sample period.
Uses the Quantise teal/cyan design system (white background).
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Quantise teal/cyan palette
Q_WHITE    = "#FFFFFF"
Q_TEAL     = "#00bfb2"   # Tech Teal — primary accent
Q_PINE     = "#04998f"   # Pine Green — secondary
Q_DARK     = "#023532"   # Brunswick Green — dark text/elements
Q_CYAN     = "#03b5aa"   # Brunswick Green lighter variant
Q_SKY      = "#8bfdf6"   # Light Cyan — highlights
Q_GREEN_DK = "#027770"   # Mid green — contrast line
Q_GREY     = "#6B7280"   # Neutral grey for labels/source
Q_DARK_TXT = "#111827"   # Near-black for titles

# Strategy colours — distinct within the teal family + contrast
STRATEGY_COLORS = {
    "Equal-weight (1/N)": Q_DARK,        # darkest — benchmark
    "Minimum-variance":   Q_TEAL,        # primary teal
    "Max-Sharpe (tangency)": "#E63946",  # warm red for contrast
    "Risk parity":        "#3B82F6",     # blue for differentiation
}

UNIVERSE_COLORS = {
    "Equity":   "#3B82F6",   # blue
    "Crypto":   "#F59E0B",   # amber
    "Combined": Q_TEAL,      # teal
}


def _q_style():
    """Apply Quantise white-background style rcParams."""
    plt.rcParams.update({
        "figure.facecolor": Q_WHITE, "axes.facecolor": Q_WHITE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#CCCCCC", "axes.grid": True, "grid.color": "#E5E7EB",
        "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
    })


def _q_header(fig, title, subtitle, source):
    """Write Quantise-style title/subtitle/source onto figure."""
    fig.text(0.012, 0.96, title, fontsize=15, fontweight="bold", color=Q_DARK_TXT)
    fig.text(0.012, 0.91, subtitle, fontsize=11, color=Q_GREY)
    fig.text(0.012, 0.01, source, fontsize=8, color=Q_GREY)
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
    _q_style()
    n = len(oos_returns)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (uni, df) in zip(axes, oos_returns.items()):
        for col in df.columns:
            wealth = (1.0 + df[col]).cumprod()
            color = STRATEGY_COLORS.get(col, Q_GREY)
            ax.plot(wealth.index, wealth.values, label=col, color=color, lw=1.8)
        ax.set_title(uni, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value of $1 invested")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0, labelsize=9)

    # Single shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False,
               ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, 0.04))

    _q_header(fig, title,
               "Walk-forward, long-only, monthly rebalance, expanding window",
               "Source: project data bundle | out-of-sample returns, rf = 0")
    fig.tight_layout(rect=[0, 0.08, 1, 0.86])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_drawdowns(
    oos_returns: dict[str, pd.DataFrame],
    save_path: Path,
):
    """Drawdown chart for each universe."""
    _q_style()
    n = len(oos_returns)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (uni, df) in zip(axes, oos_returns.items()):
        for col in df.columns:
            wealth = (1.0 + df[col]).cumprod()
            dd = wealth / wealth.cummax() - 1.0
            color = STRATEGY_COLORS.get(col, Q_GREY)
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color)
            ax.plot(dd.index, dd.values, color=color, lw=1.2, label=col)
        ax.set_title(uni, fontsize=11, fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0, labelsize=9)

    # Single shared legend below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False,
               ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, 0.04))

    _q_header(fig, "Drawdowns — peak-to-trough losses",
               "How deep each strategy falls from its high-water mark",
               "Source: project data bundle | out-of-sample, long-only")
    fig.tight_layout(rect=[0, 0.08, 1, 0.86])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_sharpe_comparison(
    metrics_df: pd.DataFrame,
    save_path: Path,
):
    """Grouped bar chart of Sharpe ratios across universes and methods."""
    _q_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    universes = metrics_df["universe"].unique()
    methods = metrics_df["method"].unique()
    x = np.arange(len(universes))
    width = 0.8 / len(methods)

    for j, method in enumerate(methods):
        vals = []
        for uni in universes:
            row = metrics_df[(metrics_df["universe"] == uni) & (metrics_df["method"] == method)]
            vals.append(row["sharpe"].iloc[0] if len(row) > 0 else 0)
        color = STRATEGY_COLORS.get(method, Q_GREY)
        bars = ax.bar(x + (j - len(methods)/2 + 0.5) * width, vals, width,
                      color=color, label=method)
        for bar, v in zip(bars, vals):
            # Place label above positive bars, below negative bars
            offset = 4 if v >= 0 else -4
            va = "bottom" if v >= 0 else "top"
            ax.annotate(f"{v:.2f}",
                        xy=(bar.get_x() + bar.get_width()/2, v),
                        xytext=(0, offset), textcoords="offset points",
                        ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(universes, fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color="#CCCCCC", lw=0.8)  # zero baseline
    # Place legend below the chart to avoid overlap
    ax.legend(fontsize=9, frameon=False, ncol=len(methods),
              loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_ylabel("Out-of-sample Sharpe ratio")
    # Give headroom for labels on both positive and negative sides
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=min(ymin, 0) * 1.25, top=ymax * 1.15)

    _q_header(fig, "Sharpe ratio comparison across funds and methods",
               "Long-only, out-of-sample, monthly rebalance, rf = 0",
               "Source: project data bundle | annualised with sqrt(periods_per_year)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_weights_over_time(
    weights: pd.DataFrame,
    top_n: int = 8,
    save_path: Path | None = None,
    title: str = "Portfolio weights over time",
    subtitle: str = "",
):
    """Stacked area chart of portfolio weights, showing top N tickers."""
    _q_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show top N tickers by average weight, group the rest as "Other"
    avg_w = weights.mean().sort_values(ascending=False)
    top_tickers = avg_w.head(top_n).index.tolist()
    other = weights.drop(columns=top_tickers, errors="ignore").sum(axis=1)

    plot_data = weights[top_tickers].copy()
    if other.sum() > 0:
        plot_data["Other"] = other

    # Use a perceptually distinct colormap for stacked areas
    n_colors = len(plot_data.columns)
    cmap = plt.cm.tab20
    colors = [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]

    ax.stackplot(plot_data.index, plot_data.T.values,
                 labels=plot_data.columns, colors=colors, alpha=0.85)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    # Place legend outside the chart area to avoid overlap
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), ncol=1, borderaxespad=0)

    _q_header(fig, title, subtitle,
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
    _q_style()
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
        ax.axhline(50, color=Q_GREY, lw=0.6, ls="--")
        ax.set_title(sector, fontsize=9, fontweight="bold")
        ax.set_ylim(30, 70)
        ax.set_ylabel("F&G", fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0, labelsize=8)

    # Hide unused axes
    for j in range(n_sectors, len(axes)):
        axes[j].set_visible(False)

    _q_header(fig, "Sector sentiment index — Fear & Greed (0–100)",
               "finVADER compound score, equal-weight across tickers, 21-day rolling mean",
               "Source: project news headlines | 0 = extreme fear, 50 = neutral, 100 = extreme greed")
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()


def plot_sector_sentiment_heatmap(
    sector_idx: pd.DataFrame,
    save_path: Path,
):
    """Heatmap of sector sentiment index (fear & greed scale).

    Compact alternative to the multi-panel line chart — shows all sectors
    in one figure with colour encoding.
    """
    _q_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    sectors = sorted(sector_idx["sector"].unique())

    # Pivot to date x sector, resample to monthly average
    pivot = sector_idx.pivot_table(
        index="date", columns="sector", values="fear_greed"
    )
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot[sectors]
    pivot_monthly = pivot.resample("MS").mean()

    # Centre colour scale on cross-sector median for maximum contrast
    median_val = pivot_monthly.values[~np.isnan(pivot_monthly.values)].mean()
    vspan = max(median_val - pivot_monthly.min().min(),
                pivot_monthly.max().max() - median_val, 3)

    im = ax.pcolormesh(
        pivot_monthly.index, range(len(sectors)), pivot_monthly.T.values,
        cmap="RdYlGn", vmin=median_val - vspan, vmax=median_val + vspan,
        shading="auto",
    )
    ax.set_yticks(range(len(sectors)))
    ax.set_yticklabels(sectors, fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", which="minor", labelsize=7, pad=2)
    ax.tick_params(axis="x", which="major", labelsize=11, pad=15)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
    cbar.set_label("Fear & Greed", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    _q_header(fig,
              "Sector sentiment heatmap — Fear & Greed (0–100)",
              "Monthly average Fear & Greed. Colour scale centred on "
              "cross-sector median for maximum contrast.",
              "Source: project news headlines | "
              "0 = extreme fear, 50 = neutral, 100 = extreme greed")
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
    _q_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    base_wealth = (1.0 + base_returns).cumprod()
    tilted_wealth = (1.0 + tilted_returns).cumprod()

    ax.plot(base_wealth.index, base_wealth.values, color=Q_GREY, lw=2, label=base_label)
    ax.plot(tilted_wealth.index, tilted_wealth.values, color=Q_PINE, lw=2, label=tilted_label)
    ax.set_ylabel("Value of $1 invested")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.legend(fontsize=10, frameon=False)

    _q_header(fig, "Sentiment fusion — before vs after",
               "Equity minimum-variance fund with and without sector sentiment tilt",
               "Source: project data bundle | out-of-sample, lagged sentiment, tilt_strength=0.3")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _reset()