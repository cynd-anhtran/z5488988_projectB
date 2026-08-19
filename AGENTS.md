# AGENTS.md — FINS3645 Part B (Quantise)

## Project purpose

Quantise is an academic Streamlit research prototype for FINS3645 Part B. It
compares twelve systematic portfolios across Equity, Crypto, and Combined asset
families, adds a sector news-sentiment layer, and exposes the saved results through
an interactive investor journey. It is not an investment product or a source of
personal financial advice.

The public app must load precomputed files from `results/`. Heavy data preparation,
sentiment scoring, optimisation, and backtesting belong in the local pipeline, not
in the deployed Streamlit process.

## Non-negotiable technical rules

- Load source data through `src/data_access.py`; do not commit raw datasets.
- Calculate returns inside each asset panel before aligning asset classes. For the
  Combined universe, select already-calculated Crypto returns on Equity trading
  dates. Do not merge price levels first because this can compress weekend moves
  into a false one-day return.
- Keep each standalone calendar intact: Equity and Combined use 252-day
  annualisation; Crypto retains weekends and uses 365-day annualisation.
- Use a monthly expanding-window walk-forward design. A rebalance may use only
  observations strictly earlier than the period whose return it earns.
- Set target weights once at the monthly rebalance. Between rebalances, calculate
  fund returns from pre-return holdings and let weights drift with asset returns.
  Do not apply the month-start target weights afresh every day.
- Calculate turnover from drifted pre-trade weights to the new target. Charge the
  initial investment and each rebalance at 10 bps one way. Exclude the initial
  purchase from average recurring turnover but include it in total cost.
- Require optimiser success and validate finite, non-negative, fully invested
  weights. Also check the transformed constraint for Maximum-Sharpe and the
  equal-risk-contribution condition for Risk parity. Invalid solutions must fail
  visibly rather than enter the results.
- Preserve headline casing, punctuation, negation, and intensifiers for VADER.
- Lag sector sentiment within sector before it can affect a portfolio decision.
  The first observation in each sector should therefore have no lagged value.
- Treat the 0–100 Fear & Greed scale as a display transformation of compound
  sentiment, not as a probability, forecast, or trading signal.

## Verified sample and interpretation

- Equity and Combined OOS: 2021-01-04 to 2023-12-29, 753 observations.
- Crypto OOS: 2021-01-01 to 2023-12-31, 1,095 observations.
- Thirty-six monthly rebalances are expected for each fund.
- Combined Minimum-variance can match Equity Minimum-variance because the optimiser
  gives Crypto an economically negligible weight. Confirm this from saved weights
  before treating identical rounded results as duplication.
- The finVADER extension broadens finance-vocabulary coverage. Without manually
  labelled sentiment or a formal prediction test, do not claim that it is more
  accurate or more predictive than base VADER.
- The implemented sentiment tilt underperforms its directly comparable Equity
  Minimum-variance baseline in this sample. Report the negative result rather than
  reframing it as investment improvement.

## App and investor-journey rules

- Use the Quantise wordmark and the established cyan, teal, green-blue palette
  consistently. Decorative typography may support the brand, but analytical text
  must remain easy to read.
- Organise the experience as Discover → Evaluate → Build → Compare. Journey cards,
  home-page calls to action, and feature cards must link to their actual pages or
  tools; do not present clickable-looking elements that do nothing.
- Avoid repeated navigation bars and duplicate calls to action. Each section should
  have one clear next action.
- Explain the question answered by each interactive feature: selectors define the
  fund under study; linked charts show return, risk, and holdings; comparison and
  correlation tools test alternatives; sentiment views connect summary scores to
  the underlying headline evidence.
- Keep the Data Explorer sortable, filterable, and downloadable. It reads saved
  results and must not trigger model recomputation.
- Maintain disclosures beside relevant outputs: historical OOS results, illustrative
  scenarios, simplified transaction costs, sentiment as language tone, and no
  personal financial advice.
- Check desktop and narrow/mobile layouts. Long tables need bounded scrolling and
  legible headers; navigation and content must retain sufficient contrast.

## Verification workflow

Run these checks after changes to the analytical pipeline:

1. `python -m pytest -q`
2. `python scripts/run_part_b.py`
3. Confirm the three OOS calendars, row counts, and 36 rebalance dates.
4. Confirm every target-weight vector is finite, non-negative, and sums to one.
5. Check that equal-weight has non-zero recurring turnover after holdings drift.
6. Reconcile `performance_metrics.csv` and `tx_cost_comparison.csv` with the report.
7. Visually inspect all generated figures, including dates, units, legends, and
   captions.
8. Run the Streamlit app and test every navigation control and interactive tool.
9. Run `python scripts/check_handin.py` and address submission-package warnings.

Do not copy a number into the report until it has been traced to a saved output and
the function that produced it.

## Repository map

```text
streamlit_app.py       Streamlit entry point; reads precomputed results
.streamlit/            Theme configuration
src/                   Portfolio, sentiment, fusion, and figure modules
scripts/               Pipeline and submission checks
tests/                 Regression and technical-integrity tests
results/data/          Saved returns, weights, and sentiment records
results/tables/        Performance and transaction-cost tables
results/figures/       Report and app figures
report/                Word report and final exported PDF
ai/                    Curated AI prompt and correction log
```
