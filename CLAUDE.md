# CLAUDE.md — Agent Instructions for FINS3645 Part B (Quantise)

## What this project is and where the data comes from

Quantise is a prototype FinTech investment app (FINS3645, Part B — Stations 3 & 4)
that offers an investor several systematically managed funds backed by out-of-sample
backtested portfolios, plus a news-sentiment analytics layer across 10 equity sectors.

Part A (Stations 1-2) already built the clean data foundation: equity prices (50 US
large-cap stocks across 10 sectors, daily 2020-2023), crypto prices (10 coins with
-USD suffix, daily 2020-2023, capped at 2023-12-31), and news headlines (~146,836
rows after dedup on ticker+date+title). Part B reuses that foundation to build the
funds, score sentiment, and deploy the app.

Data source: all raw data loads through the provided `src/data_access.py` (do not
edit). It downloads one ZIP of three Parquet files from a hosted Google Drive URL,
caches it, and exposes `load_equity_prices()`, `load_crypto_prices()`,
`load_news_headlines()`. For offline work, set `FINS_DATA_ZIP` env var to a local
copy. Part A's derived CSVs (`combined_returns_panel.csv`, `headline_panel.csv`) can
also be placed in `results/data/` or in a sibling `z5488988_projectA/results/data/`
folder — the pipeline script searches both locations before falling back to a fresh
download.

Never commit raw data files. Only commit precomputed artifacts under `results/`.

## Coding conventions and folder layout

```
streamlit_app.py       — Quantise dashboard entry point (Station 4)
.streamlit/            — Streamlit theme config (FT colour palette)
requirements.txt       — deploy-time deps only (streamlit, pandas, numpy, scipy, etc.)
requirements-dev.txt   — build/repro deps (nltk, finvader — NOT in deployed app)

src/
  data_access.py       — provided data loader (DO NOT EDIT)
  etl.py               — cleaning functions ported from Part A
  features.py          — returns, combined panel, headline panel assembly
  portfolios.py        — 4 optimisation methods + walk-forward backtest engine
  sentiment.py         — finVADER headline scoring + sector sentiment index
  fusion.py            — sentiment tilt on equity portfolio weights
  figures.py           — FT-style chart generators (7 required exhibits)

scripts/
  run_part_b.py        — full pipeline: load data -> backtest -> sentiment -> fusion
                         -> save outputs -> generate figures
  check_handin.py      — submission checker (fix every [FAIL] before hand-in)

results/
  data/                — precomputed CSVs the app reads (fund_returns, fund_weights,
                         sector_sentiment_index, headline_panel)
  tables/              — performance_metrics.csv, tx_cost_comparison.csv
  figures/             — PNG exhibits (7 required + any extras)

report/                — written report (author in Word, submit as report.pdf)
ai/                    — prompt logs and AI-use notes
context/               — provided guides: DATA_GUIDE.md, project_context.md,
                         verify_ai_output.md (do not edit these)
```

Conventions:
- All data loads go through `src/data_access.py` — no direct file I/O of raw data.
- Compute returns within each price panel BEFORE any cross-panel merge.
- Left-merge crypto returns onto the equity trading calendar (drops weekend-only
  crypto moves — this is intended per the brief and DATA_GUIDE.md).
- Annualise: equities with sqrt(252), crypto with sqrt(365), combined with sqrt(252).
  State the annualisation factor in every exhibit.
- Keep raw headline text intact — VADER needs stopwords, casing, and punctuation.
- The deployed app (`streamlit_app.py`) loads precomputed CSVs from `results/` only.
  It must NOT import nltk, finvader, or recompute backtests — Streamlit Community
  Cloud's free tier cannot handle it, and the brief explicitly forbids it.
- News deduplication: on (ticker, date, title) — NOT on (ticker, date) alone.
- Timezone: normalise news UTC dates to tz-naive before merging with price dates.

## Rules the assistant must follow

### Anti-look-ahead (critical — marks are lost here)
- Walk-forward backtests only: weights at time t are formed from an expanding
  window of data up to t-1. The OOS period starts after the initial estimation
  window (init_days), not on the first date in the data.
- Sentiment lag: day-t trading decisions use only headlines from day t-1 or earlier.
  Implemented via `groupby("sector")["sentiment"].shift(1)`.
- Never use future price information to form current weights.

### Accuracy and honesty (see context/verify_ai_output.md)
- Never invent a citation, statistic, or source.
- Flag any claim that cannot be verified from the data or a computation I can re-run.
- Show working for every number: which function, which column, which date range.
- If a function raises an error, explain the root cause rather than silently patching.
- An honest negative result (e.g. sentiment tilt underperforming the base fund) is
  acceptable — the brief says "marks are for evidenced work, not for outperformance."
- Remind me to run the code and verify outputs before putting them in the report.

### Portfolio constraints
- All portfolios are long-only, fully invested (weights >= 0, sum to 1).
- Rebalance monthly (first trading day of each month).
- Four methods: equal-weight (1/N), minimum-variance (SLSQP), max-Sharpe/tangency
  (convex transform), risk parity (Maillard et al. 2010, L-BFGS-B).
- Three fund families: Equity (50 stocks), Crypto (10 coins), Combined (60 assets).

### Innovations (two implemented)
1. **finVADER extended finance lexicon**: merges ~7,295 SentiBignomics terms (scaled
   by 0.1) and ~189 Henry terms into VADER's SentimentIntensityAnalyzer. This adds
   finance-specific sentiment words (e.g. "bullish", "downgrade", "outperform") that
   plain VADER misses. Reduces neutral rate from ~50% to ~17% on this corpus.
2. **Transaction cost model**: deducts cost proportional to absolute weight turnover
   at each monthly rebalance (default 10 bps one-way). Shows which strategies are
   most resilient to real-world trading friction.

## How I check and correct the assistant's output

1. Run `python scripts/run_part_b.py` end to end — all 6 stages must complete
   without error.
2. Verify output file row counts against expectations:
   - fund_returns.csv: ~9,293 rows (3 universes x 4-5 methods x daily returns)
   - fund_weights.csv: ~18,840 rows (monthly weight snapshots)
   - sector_sentiment_index.csv: ~9,832 rows (10 sectors x ~983 trading days)
   - performance_metrics.csv: 26 rows (methods x universes x cost scenarios)
3. Verify no look-ahead: the OOS period must start after init_days of training.
   Equity first live date should be ~2021-01-04 (after 252 training days from
   2020-01-02). Crypto first live date ~2021-07-01 (after 365 days).
4. Confirm sentiment lag: `sentiment_lagged` column should have exactly 10 NaN
   values (one per sector — the first observation in each group after shift(1)).
5. Confirm weights sum to 1.0 for every (universe, method, date) combination.
6. Check Sharpe ratios for economic plausibility: equity ~0.5-0.9 is reasonable,
   crypto near zero or negative in the 2021-2023 bear market.
7. Visually inspect all 7 figures — each must be self-contained with caption,
   labelled axes, units, and sample period.
8. Run `streamlit run streamlit_app.py` and confirm all tabs load without errors.
9. Run `python scripts/check_handin.py` and fix every [FAIL].
10. Before any number goes into the report, trace it to the function that produced
    it and re-run that function to confirm.

## Required output filenames (exact — app and markers depend on them)

- `results/data/fund_returns.csv`
- `results/data/fund_weights.csv`
- `results/data/sector_sentiment_index.csv`
- `results/tables/performance_metrics.csv`
- `results/tables/tx_cost_comparison.csv` (innovation)
- 7 figures in `results/figures/`: growth_of_one.png, drawdowns.png,
  sharpe_comparison.png, weights_combined_minvar.png, weights_equity_minvar.png,
  sector_sentiment.png, fusion_comparison.png