"""Station 2 - Feature Engineering & Text Assembly.

Ported from Part A (z5488988_projectA):
  - daily_returns()          : simple daily returns per ticker (adjClose.pct_change)
  - combined_returns_panel() : equity + crypto returns left-merged on equity calendar
  - assemble_headline_panel(): headlines aligned to equity trading days (NO scoring)

Annualisation: equities sqrt(252), crypto sqrt(365).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute simple daily returns per ticker, keeping long format.

    Returns DataFrame with columns [ticker, date, return].
    """
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["return"] = prices.groupby("ticker")[price_col].pct_change()
    result = prices[["ticker", "date", "return"]].dropna(subset=["return"])
    return result.reset_index(drop=True)


def returns_panel_wide(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
) -> pd.DataFrame:
    """Compute returns on the source calendar and pivot to date x ticker.

    This helper deliberately calculates returns *before* any cross-asset-class
    calendar alignment.  In particular, crypto weekend observations remain in
    the standalone crypto panel.
    """
    returns = daily_returns(prices, price_col=price_col)
    panel = returns.pivot(index="date", columns="ticker", values="return")
    panel.index = pd.DatetimeIndex(panel.index)
    panel.index.name = "date"
    return panel.sort_index()


def combined_returns_panel(
    eq_prices: pd.DataFrame,
    cr_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build combined equity + crypto returns panel, left-merging on equity calendar.

    Correct order:
      1. Compute each panel's returns on its own calendar.
      2. Left-merge crypto returns onto equity trading dates.

    Returns wide DataFrame: index=date, columns=all tickers.
    """
    eq_wide = returns_panel_wide(eq_prices)
    cr_wide = returns_panel_wide(cr_prices)
    combined = eq_wide.join(cr_wide, how="left")
    combined.index.name = "date"
    combined = combined.sort_index()
    return combined


def build_return_universes(
    eq_prices: pd.DataFrame,
    cr_prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build clean Equity, Crypto and Combined return universes.

    Equity and Crypto retain their native calendars.  Combined uses the equity
    trading calendar, but its crypto returns have already been calculated from
    consecutive crypto calendar days.  This prevents Friday-to-Monday price
    changes from being manufactured by differencing price levels after a merge.
    """
    equity = returns_panel_wide(eq_prices).dropna(how="any")
    crypto = returns_panel_wide(cr_prices).dropna(how="any")
    combined = equity.join(crypto, how="left")

    missing_crypto = int(combined[crypto.columns].isna().sum().sum())
    if missing_crypto:
        raise ValueError(
            "Combined panel has missing crypto returns on equity trading dates "
            f"({missing_crypto} cells); inspect the native crypto calendar."
        )

    if not equity.index.equals(combined.index):
        raise ValueError("Combined return panel must retain the equity calendar exactly.")

    return {
        "Equity": equity,
        "Crypto": crypto,
        "Combined": combined,
    }


def price_panel_wide(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Pivot prices to wide format (date x ticker), used for backtest universe building."""
    return prices.pivot(index="date", columns="ticker", values=price_col).sort_index()


# ---------------------------------------------------------------------------
# Text panel assembly (from Part A)
# ---------------------------------------------------------------------------

def assemble_headline_panel(
    headlines: pd.DataFrame,
    eq_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Align headlines to equity trading days.

    Rules:
    - If headline date is an equity trading day -> keep it on that date.
    - If NOT a trading day -> advance to next trading day (merge_asof forward).
    - Raw headline text is KEPT as-is. NO stopword stripping (VADER needs them).
    - NO sentiment scoring here — that is Station 3.

    Returns: [trading_date, original_date, ticker, sector, title, url, publisher]
    """
    trading_days = pd.DatetimeIndex(sorted(eq_prices["date"].unique()))

    headlines = headlines.copy()
    if headlines["date"].dt.tz is not None:
        headlines["date"] = headlines["date"].dt.tz_localize(None)
    headlines["date"] = headlines["date"].astype("datetime64[ns]")

    trading_df = pd.DataFrame({"trading_date": trading_days.astype("datetime64[ns]")})
    trading_df = trading_df.sort_values("trading_date")

    headlines = headlines.sort_values("date")
    merged = pd.merge_asof(
        headlines.rename(columns={"date": "original_date"}),
        trading_df,
        left_on="original_date",
        right_on="trading_date",
        direction="forward",
    )

    merged = merged.dropna(subset=["trading_date"])
    merged["trading_date"] = pd.to_datetime(merged["trading_date"])

    cols = ["trading_date", "original_date", "ticker", "sector", "title", "url", "publisher"]
    return merged[cols].sort_values(["trading_date", "ticker"]).reset_index(drop=True)
