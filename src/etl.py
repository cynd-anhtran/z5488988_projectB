"""Station 1 - ETL: load and clean all three datasets.

Ported from Part A (z5488988_projectA) with the same integrity checks:
  - Missing-date audit (equity & crypto)
  - Duplicate check  (prices: ticker+date; news: ticker+date+title)
  - Outlier / extreme-value screen on daily returns (flagged, not dropped)
  - Calendar alignment: crypto capped at 2023-12-31
  - UTC timezone normalisation on news dates before any merge
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src import data_access


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equity_trading_days(eq: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the full set of equity trading dates in the dataset."""
    return pd.DatetimeIndex(sorted(eq["date"].unique()))


# ---------------------------------------------------------------------------
# Equity loader
# ---------------------------------------------------------------------------

def load_clean_equities() -> tuple[pd.DataFrame, dict]:
    """Load equity prices and run Station 1 integrity checks.

    Returns (clean DataFrame, integrity report dict).
    Outliers are flagged and kept — genuine market events.
    """
    df = data_access.load_equity_prices()
    report = {}

    # 1. Missing-date audit
    date_counts = df.groupby("ticker")["date"].count()
    full_count = date_counts.max()
    missing_dates = full_count - date_counts
    report["eq_missing_dates"] = {
        "max_trading_days": int(full_count),
        "tickers_with_gaps": int((missing_dates > 0).sum()),
        "total_missing_obs": int(missing_dates.sum()),
        "worst_ticker": missing_dates.idxmax() if missing_dates.max() > 0 else "none",
        "worst_count": int(missing_dates.max()),
    }

    # 2. Duplicate check (ticker + date)
    dupes = df.duplicated(subset=["ticker", "date"], keep=False)
    report["eq_duplicates"] = {
        "duplicate_rows": int(dupes.sum()),
        "action": "none found" if dupes.sum() == 0 else "dropped keep=first",
    }
    if dupes.sum() > 0:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # 3. Outlier screen (|z| > 5 within ticker)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["_ret"] = df.groupby("ticker")["adjClose"].pct_change()
    stats = df.groupby("ticker")["_ret"].agg(["mean", "std"])
    df = df.join(stats, on="ticker", rsuffix="_grp")
    df["_z"] = (df["_ret"] - df["mean"]) / df["std"]
    outliers = df[df["_z"].abs() > 5].copy()

    report["eq_outliers"] = {
        "threshold": "±5 std dev within ticker",
        "outlier_rows": int(len(outliers)),
        "action": "flagged and kept — genuine market events",
    }

    df = df.drop(columns=["_ret", "mean", "std", "_z", "mean_grp", "std_grp"], errors="ignore")
    return df, report


# ---------------------------------------------------------------------------
# Crypto loader
# ---------------------------------------------------------------------------

def load_clean_crypto() -> tuple[pd.DataFrame, dict]:
    """Load crypto prices (365-day calendar), capped at 2023-12-31.

    Returns (clean DataFrame, integrity report dict).
    """
    df = data_access.load_crypto_prices()
    report = {}

    # Cap at 2023-12-31 (10 stray 2024-01-01 rows)
    stray = (df["date"] > pd.Timestamp("2023-12-31")).sum()
    df = df[df["date"] <= pd.Timestamp("2023-12-31")].copy()
    report["cr_stray_rows_dropped"] = int(stray)

    # Missing-date audit
    date_counts = df.groupby("ticker")["date"].count()
    full_count = date_counts.max()
    missing_dates = full_count - date_counts
    report["cr_missing_dates"] = {
        "max_calendar_days": int(full_count),
        "tickers_with_gaps": int((missing_dates > 0).sum()),
        "total_missing_obs": int(missing_dates.sum()),
    }

    # Duplicate check
    dupes = df.duplicated(subset=["ticker", "date"], keep=False)
    report["cr_duplicates"] = {
        "duplicate_rows": int(dupes.sum()),
        "action": "none found" if dupes.sum() == 0 else "dropped keep=first",
    }
    if dupes.sum() > 0:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # Outlier screen
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["_ret"] = df.groupby("ticker")["adjClose"].pct_change()
    stats = df.groupby("ticker")["_ret"].agg(["mean", "std"])
    df = df.join(stats, on="ticker", rsuffix="_grp")
    df["_z"] = (df["_ret"] - df["mean"]) / df["std"]
    outliers = df[df["_z"].abs() > 5].copy()

    report["cr_outliers"] = {
        "threshold": "±5 std dev within ticker",
        "outlier_rows": int(len(outliers)),
        "action": "flagged and kept — crypto volatility events",
    }

    df = df.drop(columns=["_ret", "mean", "std", "_z", "mean_grp", "std_grp"], errors="ignore")
    return df, report


# ---------------------------------------------------------------------------
# News loader
# ---------------------------------------------------------------------------

def load_clean_news() -> tuple[pd.DataFrame, dict]:
    """Load and clean news headlines.

    Deduplication is on ticker + date + title (NOT ticker+date alone).
    Timezone normalised from UTC to tz-naive.
    """
    df = data_access.load_news_headlines()
    report = {}

    # Timezone normalisation
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["date"] = df["date"].astype("datetime64[ns]")

    # Deduplicate on ticker + date + title
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date", "title"], keep="first")
    after = len(df)

    report["news_duplicates"] = {
        "rows_before": before,
        "rows_dropped": before - after,
        "rows_after": after,
        "note": "deduplicated on ticker+date+title (NOT ticker+date alone)",
    }

    # Publisher blank rate
    blank_publisher = df["publisher"].isna() | (df["publisher"].str.strip() == "")
    report["news_blank_publisher_pct"] = round(blank_publisher.mean() * 100, 1)

    return df, report