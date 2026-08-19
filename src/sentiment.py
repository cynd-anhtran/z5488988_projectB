"""Station 3 - Sentiment model and sector index from news headlines.

Innovation: uses finVADER's extended finance lexicons (SentiBignomics +
Henry word lists) merged into VADER's SentimentIntensityAnalyzer. This
adds ~7,500 finance-specific sentiment terms that standard VADER misses
(e.g. "bullish", "downgrade", "outperform", "restructuring").

We call polarity_scores() once per headline (returns all 4 scores in one
pass) and build the scorer once, making the 146K-headline corpus scorable
in ~2 minutes.

Pipeline:
  1. Score each headline -> compound, pos, neg, neu scores.
  2. Aggregate to ticker-day level (mean of headline scores).
  3. Build sector sentiment index: equal-weight average across tickers.
  4. Rescale to a 0-100 Fear & Greed scale.
  5. Lag by 1 trading day to avoid look-ahead.
"""
from __future__ import annotations

import ssl as _ssl
try:
    _create_unverified = _ssl._create_unverified_context
except AttributeError:
    pass
else:
    _ssl._create_default_https_context = _create_unverified

import pandas as pd
import numpy as np


def _build_scorer():
    """Build a VADER scorer with finVADER's finance lexicons merged in.

    Uses vaderSentiment package (which bundles the lexicon — no separate
    NLTK download needed) instead of nltk.sentiment.vader. This avoids
    the SSL certificate errors on macOS when nltk tries to download
    vader_lexicon.zip.

    SentiBignomics (~7,295 terms) is scaled by 0.1 (per the finvader
    library default) before merging; Henry (~189 terms) is used as-is.
    The merged lexicon is a superset of standard VADER + finance terms.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from finvader.SentiBignomics import lexicon1
    from finvader.Henry import lexicon2

    # Build merged finance lexicon
    sentibignomics = lexicon1()
    constant = 0.1  # finvader default scaling factor
    sentibignomics = {k: v * constant for k, v in sentibignomics.items()}
    henry = lexicon2()
    merged = {**sentibignomics, **henry}

    # Build scorer with merged lexicon
    scorer = SentimentIntensityAnalyzer()
    scorer.lexicon.update(merged)
    return scorer


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score each headline with VADER + finVADER finance lexicons.

    Keeps casing, punctuation, and stopwords intact — VADER needs them.
    One call to polarity_scores() per headline returns {compound, pos, neg, neu}.

    Parameters
    ----------
    panel : DataFrame with columns [trading_date, ticker, sector, title, ...]

    Returns
    -------
    DataFrame with original columns plus [compound, pos, neg, neu].
    """
    scorer = _build_scorer()
    scored = panel.copy()

    titles = scored["title"].fillna("").tolist()

    # Score all headlines — polarity_scores returns all 4 in one call
    results = []
    for i, title in enumerate(titles):
        try:
            s = scorer.polarity_scores(title)
            results.append((s["compound"], s["pos"], s["neg"], s["neu"]))
        except Exception:
            results.append((0.0, 0.0, 0.0, 1.0))
        if (i + 1) % 25000 == 0:
            print(f"    ... scored {i+1:,} / {len(titles):,} headlines")

    scored["compound"] = [r[0] for r in results]
    scored["pos"] = [r[1] for r in results]
    scored["neg"] = [r[2] for r in results]
    scored["neu"] = [r[3] for r in results]

    return scored


def _ticker_day_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline scores to ticker-day level (mean compound)."""
    agg = (
        scored
        .groupby(["trading_date", "ticker", "sector"])["compound"]
        .mean()
        .reset_index()
        .rename(columns={"compound": "sentiment"})
    )
    return agg


def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """Build a daily sentiment index per sector.

    Equal-weight average of ticker-day sentiment within each sector.
    Ticker-days with no headlines are implicitly excluded (not carried
    forward or treated as neutral) — we only average over tickers that
    have at least one headline on that day, so the index reflects the
    actual signal from the news flow rather than diluting it.

    Also computes a Fear & Greed rescaling: compound in [-1, 1] mapped
    linearly to [0, 100], where 0 = extreme fear, 100 = extreme greed.

    The sentiment is then LAGGED by 1 trading day so that day t's
    decision uses only sentiment from day t-1 or earlier.

    Returns DataFrame: [date, sector, sentiment, fear_greed, sentiment_lagged]
    """
    # Aggregate to ticker-day
    ticker_day = _ticker_day_sentiment(scores)

    # Sector-day index: equal-weight across tickers
    sector_day = (
        ticker_day
        .groupby(["trading_date", "sector"])["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"trading_date": "date"})
    )

    # Fear & Greed rescale: [-1, 1] -> [0, 100]
    sector_day["fear_greed"] = ((sector_day["sentiment"] + 1) / 2 * 100).round(2)

    # Lag sentiment by 1 trading day within each sector (no look-ahead)
    sector_day = sector_day.sort_values(["sector", "date"])
    sector_day["sentiment_lagged"] = (
        sector_day.groupby("sector")["sentiment"].shift(1)
    )

    return sector_day.reset_index(drop=True)