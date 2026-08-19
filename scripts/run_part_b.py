"""Reproduce all Part B results. Run from the project root:

    python scripts/run_part_b.py

Pipeline:
  1. Load clean equity and crypto prices through data_access and reuse the
     Part A headline panel when available
  2. Compute Equity and Crypto returns on their native calendars, then build
     Combined by aligning already-computed crypto returns to equity dates
  3. Run out-of-sample backtests (4 methods x 3 universes)
  4. Score headlines with finVADER, build sector sentiment index
  5. Apply sentiment fusion to equity fund
  6. Save all required outputs to results/
  7. Generate all figures
"""
import sys
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.portfolios import (
    run_oos_backtest, performance_metrics, growth_of_one,
    WEIGHT_FUNCS, EQUITY_DAYS, CRYPTO_DAYS,
)
from src.sentiment import score_headlines, sector_sentiment_index
from src.fusion import apply_sentiment, backtest_with_fusion
from src.figures import (
    plot_growth_of_one, plot_drawdowns, plot_sharpe_comparison,
    plot_weights_over_time, plot_weights_across_methods,
    plot_sector_sentiment, plot_sector_sentiment_heatmap, plot_fusion_comparison,
)

# Output directories
RESULTS = ROOT / "results"
DATA_DIR = RESULTS / "data"
TABLE_DIR = RESULTS / "tables"
FIG_DIR = RESULTS / "figures"
for d in (DATA_DIR, TABLE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading: compute each return panel on its native calendar
# ---------------------------------------------------------------------------

def _find_part_a_headlines():
    """Find the Part A headline panel, if it is available locally."""
    search_paths = [
        ROOT / "results" / "data",               # if Part A results copied into Part B
        ROOT.parent / "z5488988_projectA" / "results" / "data",  # sibling folder
        pathlib.Path("/mnt/user-data/uploads/z5488988_projectA/results/data"),  # staged
    ]
    for p in search_paths:
        if (p / "headline_panel.csv").exists():
            return p / "headline_panel.csv"
    return None


def _load_from_data_access():
    """Build all return universes correctly from the provided price data."""
    from src.etl import load_clean_equities, load_clean_crypto, load_clean_news
    from src.features import (
        build_return_universes, assemble_headline_panel,
    )

    eq, _ = load_clean_equities()
    cr, _ = load_clean_crypto()
    universes = build_return_universes(eq, cr)

    headline_csv = _find_part_a_headlines()
    if headline_csv is not None:
        print(f"  Reusing Part A headline panel: {headline_csv}")
        headline_panel = pd.read_csv(
            headline_csv,
            parse_dates=["trading_date", "original_date"],
            dtype={"publisher": str},
            low_memory=False,
        )
    else:
        print("  Part A headline panel not found; rebuilding it from source data")
        news, _ = load_clean_news()
        headline_panel = assemble_headline_panel(news, eq)

    # Sector map
    sector_map = eq[["ticker", "sector"]].drop_duplicates().reset_index(drop=True)

    return universes, headline_panel, sector_map


def _turnover_metrics(audit: pd.DataFrame, strategy: str) -> dict:
    """Summarise drift-adjusted traded notional for one strategy.

    The first row is the initial purchase from cash.  Average rebalance
    turnover excludes that initial construction so a constant-target strategy
    such as 1/N still reports the trading caused by weight drift.
    """
    rows = audit[audit["strategy"] == strategy].sort_values("first_holding_date")
    if rows.empty:
        return {
            "initial_turnover": np.nan,
            "avg_rebalance_turnover": np.nan,
            "total_turnover": np.nan,
            "total_tx_cost": np.nan,
        }
    recurring = rows["turnover"].iloc[1:]
    return {
        "initial_turnover": float(rows["turnover"].iloc[0]),
        "avg_rebalance_turnover": float(recurring.mean()) if len(recurring) else 0.0,
        "total_turnover": float(rows["turnover"].sum()),
        "total_tx_cost": float(rows["tx_cost"].sum()),
    }


def main():
    t0 = time.time()

    # ==================================================================
    # STAGE 1: Load data
    # ==================================================================
    print("=" * 60)
    print("STAGE 1: Loading data...")
    print("=" * 60)

    universes, headline_panel, sector_map = _load_from_data_access()
    equity_ret = universes["Equity"]
    crypto_ret = universes["Crypto"]
    combined_ret = universes["Combined"]

    crypto_weekend_days = int((crypto_ret.index.dayofweek >= 5).sum())
    if crypto_weekend_days == 0:
        raise RuntimeError("Crypto return panel lost its native weekend observations.")
    if not combined_ret.index.equals(equity_ret.index):
        raise RuntimeError("Combined return panel is not aligned to the equity calendar.")

    print(f"  Equity universe:   {equity_ret.shape[0]} days x {equity_ret.shape[1]} assets")
    print(f"  Crypto universe:   {crypto_ret.shape[0]} days x {crypto_ret.shape[1]} assets")
    print(f"  Combined universe: {combined_ret.shape[0]} days x {combined_ret.shape[1]} assets")
    print(f"  Headlines:         {len(headline_panel):,} rows, "
          f"{headline_panel['sector'].nunique()} sectors")

    UNIVERSES = {
        "Equity": (equity_ret, EQUITY_DAYS),
        "Crypto": (crypto_ret, CRYPTO_DAYS),
        "Combined": (combined_ret, EQUITY_DAYS),
    }

    # ==================================================================
    # STAGE 2: Out-of-sample backtests
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Running out-of-sample backtests...")
    print("=" * 60)

    all_oos = {}       # {universe: DataFrame(date x strategy)}
    all_weights = {}   # {universe: {strategy: DataFrame}}
    all_metrics = []

    # Also run with transaction costs for the innovation comparison
    TX_COST = 10  # 10 bps one-way

    for uni_name, (ret, ann_days) in UNIVERSES.items():
        print(f"\n  --- {uni_name} ({ret.shape[1]} assets, ann={ann_days}) ---")

        # Run with zero tx cost (base)
        oos, weights, audit = run_oos_backtest(ret, WEIGHT_FUNCS, init_days=ann_days)
        all_oos[uni_name] = oos
        all_weights[uni_name] = weights

        rebalance_count = audit["decision_month"].nunique()
        print(f"    OOS period: {oos.index.min():%Y-%m-%d} to {oos.index.max():%Y-%m-%d} "
              f"({rebalance_count} monthly rebalances)")

        for method in oos.columns:
            m = performance_metrics(oos[method], ann_days)
            m.update(_turnover_metrics(audit, method))
            m["universe"] = uni_name
            m["method"] = method
            m["tx_cost_bps"] = 0
            all_metrics.append(m)
            print(f"    {method:25s}  Sharpe {m['sharpe']:5.2f}  "
                  f"AnnRet {m['ann_return']*100:6.1f}%  "
                  f"Vol {m['ann_vol']*100:5.1f}%  "
                  f"MaxDD {m['max_drawdown']*100:6.1f}%")

        # Run with transaction costs (innovation)
        oos_tc, _, audit_tc = run_oos_backtest(
            ret, WEIGHT_FUNCS, init_days=ann_days, tx_cost_bps=TX_COST
        )
        for method in oos_tc.columns:
            m = performance_metrics(oos_tc[method], ann_days)
            m.update(_turnover_metrics(audit_tc, method))
            m["universe"] = uni_name
            m["method"] = method
            m["tx_cost_bps"] = TX_COST
            all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)

    # ==================================================================
    # STAGE 3: Sentiment scoring
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Scoring headlines with finVADER...")
    print("=" * 60)

    scored = score_headlines(headline_panel)
    print(f"  Scored: {len(scored):,} headlines")
    print(f"  Compound score: mean={scored['compound'].mean():.4f}, "
          f"std={scored['compound'].std():.4f}")

    # Neutral rate
    neutral = (scored["compound"] == 0).mean() * 100
    print(f"  Neutral headlines: {neutral:.1f}%")

    # Build sector sentiment index
    sector_idx = sector_sentiment_index(scored)
    print(f"  Sector index: {len(sector_idx):,} sector-day rows, "
          f"{sector_idx['sector'].nunique()} sectors")

    # ==================================================================
    # STAGE 4: Sentiment fusion
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 4: Applying sentiment fusion to equity fund...")
    print("=" * 60)

    # Use the equity minimum-variance weights as the base for fusion
    eq_minvar_weights = all_weights["Equity"]["Minimum-variance"]

    tilted_weights = apply_sentiment(
        eq_minvar_weights, sector_idx, sector_map, tilt_strength=0.3
    )

    base_ret, tilted_ret = backtest_with_fusion(
        equity_ret, eq_minvar_weights, tilted_weights
    )

    base_m = performance_metrics(base_ret, EQUITY_DAYS)
    tilted_m = performance_metrics(tilted_ret, EQUITY_DAYS)

    print(f"  Base (no sentiment):  Sharpe {base_m['sharpe']:.2f}  "
          f"AnnRet {base_m['ann_return']*100:.1f}%  Vol {base_m['ann_vol']*100:.1f}%")
    print(f"  Sentiment-tilted:     Sharpe {tilted_m['sharpe']:.2f}  "
          f"AnnRet {tilted_m['ann_return']*100:.1f}%  Vol {tilted_m['ann_vol']*100:.1f}%")

    # Add fusion metrics to the table
    base_m.update({"universe": "Equity", "method": "MinVar (base)", "tx_cost_bps": 0})
    tilted_m.update({"universe": "Equity", "method": "MinVar + sentiment", "tx_cost_bps": 0})
    all_metrics.extend([base_m, tilted_m])
    metrics_df = pd.DataFrame(all_metrics)

    # ==================================================================
    # STAGE 5: Save required outputs
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 5: Saving outputs...")
    print("=" * 60)

    # --- fund_returns.csv ---
    fund_returns_parts = []
    for uni, df in all_oos.items():
        for method in df.columns:
            tmp = df[[method]].rename(columns={method: "daily_return"}).copy()
            tmp["universe"] = uni
            tmp["method"] = method
            tmp["growth_of_1"] = growth_of_one(df[method]).values
            fund_returns_parts.append(tmp)
    # Add fusion
    fusion_df = pd.DataFrame({
        "daily_return": tilted_ret.values,
        "universe": "Equity",
        "method": "MinVar + sentiment",
        "growth_of_1": growth_of_one(tilted_ret).values,
    }, index=tilted_ret.index)
    fund_returns_parts.append(fusion_df)

    fund_returns = pd.concat(fund_returns_parts)
    fund_returns.index.name = "date"
    fund_returns.to_csv(DATA_DIR / "fund_returns.csv")
    print(f"  Saved fund_returns.csv ({len(fund_returns):,} rows)")

    # --- fund_weights.csv ---
    weight_parts = []
    for uni, strat_dict in all_weights.items():
        for method, wdf in strat_dict.items():
            wdf_long = wdf.stack().reset_index()
            wdf_long.columns = ["date", "ticker", "weight"]
            wdf_long["universe"] = uni
            wdf_long["method"] = method
            weight_parts.append(wdf_long)
    # Add tilted weights
    tw_long = tilted_weights.stack().reset_index()
    tw_long.columns = ["date", "ticker", "weight"]
    tw_long["universe"] = "Equity"
    tw_long["method"] = "MinVar + sentiment"
    weight_parts.append(tw_long)

    fund_weights = pd.concat(weight_parts, ignore_index=True)
    fund_weights.to_csv(DATA_DIR / "fund_weights.csv", index=False)
    print(f"  Saved fund_weights.csv ({len(fund_weights):,} rows)")

    # --- sector_sentiment_index.csv ---
    sector_idx.to_csv(DATA_DIR / "sector_sentiment_index.csv", index=False)
    print(f"  Saved sector_sentiment_index.csv ({len(sector_idx):,} rows)")

    # --- headline_panel.csv (for the Streamlit headline feed) ---
    headline_panel.to_csv(DATA_DIR / "headline_panel.csv", index=False)
    print(f"  Saved headline_panel.csv ({len(headline_panel):,} rows)")

    # --- performance_metrics.csv ---
    metrics_df.to_csv(TABLE_DIR / "performance_metrics.csv", index=False)
    print(f"  Saved performance_metrics.csv ({len(metrics_df)} rows)")

    # --- Transaction cost comparison table (innovation) ---
    tc_compare = metrics_df[metrics_df["tx_cost_bps"].isin([0, TX_COST])].copy()
    tc_compare = tc_compare[~tc_compare["method"].str.contains("sentiment|base", case=False, na=False)]
    tc_pivot = tc_compare.pivot_table(
        index=["universe", "method"], columns="tx_cost_bps",
        values=["sharpe", "ann_return", "ann_vol", "max_drawdown"]
    ).round(4)
    tc_pivot.to_csv(TABLE_DIR / "tx_cost_comparison.csv")
    print(f"  Saved tx_cost_comparison.csv")

    # ==================================================================
    # STAGE 6: Generate figures
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 6: Generating figures...")
    print("=" * 60)

    # 1. Growth of $1
    plot_growth_of_one(all_oos, FIG_DIR / "growth_of_one.png")
    print("  Saved growth_of_one.png")

    # 2. Drawdowns
    plot_drawdowns(all_oos, FIG_DIR / "drawdowns.png")
    print("  Saved drawdowns.png")

    # 3. Sharpe comparison (bar chart)
    base_metrics = metrics_df[
        (metrics_df["tx_cost_bps"] == 0) &
        (~metrics_df["method"].str.contains("sentiment|base", case=False, na=False))
    ]
    plot_sharpe_comparison(base_metrics, FIG_DIR / "sharpe_comparison.png")
    print("  Saved sharpe_comparison.png")

    # 4. Weights over time across all four Equity methods
    plot_weights_across_methods(
        all_weights["Equity"],
        save_path=FIG_DIR / "weights_equity_methods.png",
        universe="Equity",
        top_n=6,
    )
    print("  Saved weights_equity_methods.png")

    # 4b. Weights over time (Combined Min-Var as example)
    plot_weights_over_time(
        all_weights["Combined"]["Minimum-variance"],
        top_n=10,
        save_path=FIG_DIR / "weights_combined_minvar.png",
        title="Combined fund — Minimum-variance holdings",
        subtitle="Top 10 positions by average weight, monthly rebalance",
    )
    print("  Saved weights_combined_minvar.png")

    # 4c. Weights over time (Equity Min-Var)
    plot_weights_over_time(
        all_weights["Equity"]["Minimum-variance"],
        top_n=10,
        save_path=FIG_DIR / "weights_equity_minvar.png",
        title="Equity fund — Minimum-variance holdings",
        subtitle="Top 10 positions by average weight, monthly rebalance",
    )
    print("  Saved weights_equity_minvar.png")

    # 6. Sector sentiment index
    plot_sector_sentiment(sector_idx, FIG_DIR / "sector_sentiment.png")
    print("  Saved sector_sentiment.png")

    # 6b. Sector sentiment heatmap (compact alternative)
    plot_sector_sentiment_heatmap(sector_idx, FIG_DIR / "sector_sentiment_heatmap.png")
    print("  Saved sector_sentiment_heatmap.png")

    # 7. Fusion comparison
    plot_fusion_comparison(
        base_ret, tilted_ret,
        FIG_DIR / "fusion_comparison.png",
    )
    print("  Saved fusion_comparison.png")

    # ==================================================================
    # Done
    # ==================================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.0f} seconds.")
    print(f"  fund_returns.csv:           {DATA_DIR / 'fund_returns.csv'}")
    print(f"  fund_weights.csv:           {DATA_DIR / 'fund_weights.csv'}")
    print(f"  sector_sentiment_index.csv: {DATA_DIR / 'sector_sentiment_index.csv'}")
    print(f"  performance_metrics.csv:    {TABLE_DIR / 'performance_metrics.csv'}")
    print(f"  Figures:                    {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
