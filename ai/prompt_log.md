# AI Prompt and Correction Log — Quantise Part B

This is a condensed record of the AI-assisted work. Repeated styling requests,
routine command output, and ideas that were not implemented have been removed. The
entries retain the decisions, risks, corrections, and verification steps that
materially affected the submitted system.

## 1. Translate the brief into a testable system

**Prompt focus:** Review the project files and brief, identify the required
portfolio, sentiment, innovation, app, and report outputs, and assess what was still
missing.

**Assistant contribution:** Mapped the work into separate layers: data access,
portfolio optimisation and walk-forward backtesting, sentiment scoring and
aggregation, sentiment fusion, transaction-cost sensitivity, saved exhibits, and a
lightweight Streamlit presentation layer.

**Risk identified:** A visually complete app would not demonstrate a correct model
if the saved results came from inconsistent calendars or backtest mechanics.

**Decision and verification:** Kept calculation modules separate from the app and
made the app consume only saved CSVs. Later checks traced every report table and
figure back to the regenerated pipeline outputs.

## 2. Implement finance-aware headline scoring

**Prompt focus:** Build the required sector sentiment index and investigate whether
finVADER could provide a meaningful extension to base VADER.

**Assistant contribution:** Inspected the package behaviour and avoided repeated
wrapper calls by constructing one VADER analyser with the SentiBignomics and Henry
finance lexicons. Headline scores were aggregated first within ticker-day and then
equally across available ticker-days within sector-day. The sector series was lagged
before portfolio use.

**What was wrong or risky:** The finVADER wrapper path was unreliable and repeated
calls would have made scoring unnecessarily slow. An early description also said the
extension improved sentiment “accuracy,” although the project had no manually
labelled validation set.

**Correction:** Used the underlying lexicons directly and limited the conclusion to
broader finance-vocabulary coverage and different score distributions. The report
now states that predictive or classification superiority was not established.

**Evidence:** The final pipeline scores 146,830 headlines and produces 9,832
sector-day observations. Deterministic tests check sector ordering and lagging.

## 3. Detect and correct the return-calendar problem

**Prompt focus:** Audit the technical layers before trusting the final figures,
especially the Equity, Crypto, and Combined return calendars.

**How the issue was spotted:** The reported Crypto period and some performance
figures did not agree with a 365-day standalone calendar. This prompted a trace from
price panels to return panels and then into the OOS slices.

**What was wrong or risky:** Aligning price levels before calculating returns can
compress several weekend Crypto moves into the next Equity date. It also makes the
standalone and Combined conventions difficult to interpret consistently.

**Correction:** Returns are now calculated inside each original asset panel. The
Combined universe selects the already-calculated Crypto returns on Equity trading
dates, while the standalone Crypto fund retains all calendar days.

**Evidence:** Final OOS samples are Equity/Combined 2021-01-04 to 2023-12-29 with
753 observations and Crypto 2021-01-01 to 2023-12-31 with 1,095 observations. The
annualisation factors are 252, 365, and 252 respectively.

## 4. Detect and correct the holding-period and turnover inconsistency

**Prompt focus:** Correct the return calculations before trusting the final results,
compute pre-trade drifted weights, and check why equal-weight turnover appeared
unresolved.

**How the issue was spotted:** The transaction-cost code assumed that holdings drifted
between monthly rebalances, but daily gross fund returns were still calculated with
the fixed month-start target vector. These two parts described different portfolios.
The near-zero recurring turnover initially shown for 1/N was another warning sign:
even an equal-weight target requires trades after its holdings drift.

**What was wrong or risky:** Reapplying target weights each day silently creates daily
rebalancing. Measuring costs from one target vector to the next understates trades
and can incorrectly give equal-weight zero turnover.

**Correction:** At each month start, the target is set once. Each day uses the current
pre-return holdings, then updates those holdings self-financingly after asset returns.
The ending weights become the next rebalance’s pre-trade holdings. Costs are charged
on the absolute trade from those weights to the new target; the initial purchase is
charged separately.

**Evidence:** A regression test with a two-asset example confirms that the second
day’s fund return uses drifted rather than original weights. Final recurring monthly
turnover for Equity 1/N is 5.45%, not zero. The corrected pipeline regenerated all
performance, cost, weight, and figure files.

## 5. Investigate identical Combined and Equity Minimum-variance results

**Prompt focus:** Check whether identical rounded Combined and Equity
Minimum-variance outputs indicated duplicated weights or a pipeline bug.

**Assistant contribution:** Compared saved weight histories by universe and inspected
the Combined Crypto sleeve rather than relying only on rounded performance metrics.

**Finding:** The Combined optimiser assigns an economically negligible weight to the
higher-volatility Crypto assets. Its material allocation therefore converges to the
Equity Minimum-variance portfolio, so the rounded returns and risk measures match.

**Correction to interpretation:** The report now explains this as an optimisation
outcome, not duplicated data. The weight figures show both portfolios so the reader
can verify the result.

## 6. Add optimiser checks and reproducibility controls

**Prompt focus:** Prevent silent optimisation failures and make the environment and
figure generation reproducible.

**Assistant contribution:** Added checks for solver success, finite weights,
non-negativity, full investment, the transformed Maximum-Sharpe constraint, and the
Risk-parity contribution condition. Added deterministic regression tests and a
headless Matplotlib backend for pipeline execution.

**Risk addressed:** An optimiser can return a vector even when convergence or a
constraint fails. Without explicit checks, invalid weights may enter tables and
charts unnoticed. Figure generation also previously depended on the local display
environment.

**Evidence:** The final automated suite passes eight tests, the six-stage pipeline
completes, and all saved figures render in a headless environment.

## 7. Research and redesign the app around the user journey

**Prompt focus:** Make the Streamlit experience more professional and interactive;
use the supplied fintech references for hierarchy and typography while retaining the
cyan/teal/green-blue identity.

**Research approach:** Reviewed the supplied landing-page and dashboard references,
then compared their recurring patterns with the app’s analytical tasks: a strong
brand-led opening, restrained metric summaries, clear cards, one primary action per
section, progressive disclosure, and direct navigation from overview content to the
underlying tool. Browser screenshots were used to inspect the implementation at
desktop and narrow widths.

**Problems found through iteration:**

- Several home-page call-to-action rows repeated the same destinations.
- Feature cards and the investor-journey strip looked interactive but initially did
  not take users to the corresponding analysis.
- The first sentiment snapshot showed only one sector, limiting comparison.
- Long result tables needed sorting, filtering, and bounded scrolling.
- The mobile sidebar had poor contrast and the fund-family cards exposed raw HTML in
  two columns.
- The brand name was not prominent or consistent across pages.

**Design decisions:**

- Established the Quantise wordmark and a consistent cyan/teal/blue system. A
  contrasting type style is used selectively for the brand, not for dense analysis.
- Organised the experience as Discover → Evaluate → Build → Compare and made the
  journey controls navigate to the relevant tools.
- Removed duplicate calls to action and linked home feature cards to Funds,
  Sentiment, Data Explorer, Fund Comparison, Correlation Explorer, and scenario
  tools.
- Added a sector selector and contextual sentiment views so users can move from a
  summary score to a time series, heatmap, and available headlines.
- Made analytical tables sortable, filterable, scrollable, and downloadable without
  rerunning the model.
- Added disclosures close to historical results and illustrative scenarios.

**Why these interactions were retained:** Each one answers a user question rather
than adding decoration. Selectors define the object of study; linked return,
drawdown, and holdings views explain the same fund from different angles; comparison
and correlation tools evaluate alternatives; sentiment controls connect the summary
index to its underlying evidence.

**Verification boundary:** The interface was functionally and visually checked, but
no formal user study was conducted. The report therefore describes the intended
journey without claiming proven improvements in comprehension or suitability.

**Final verification:**

- Eight technical tests pass.
- The project checker passes all nineteen substantive checks.
- Report tables agree with the regenerated CSVs.
- All seven embedded figures were regenerated from the corrected results.
- All eleven rendered report pages were inspected for clipping, broken tables,
  unreadable figures, and caption problems.

Remaining hand-in tasks are packaging only: export the Word report to PDF and omit
`.DS_Store`, cache files, and the local virtual environment from the submitted ZIP.
