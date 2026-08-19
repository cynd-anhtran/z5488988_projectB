"""Station 3 (extension) - Fuse sentiment into the equity funds.

Sentiment tilt: at each rebalance, adjust EQUITY-ONLY portfolio weights
based on the lagged sector sentiment. Positive sentiment tilts toward
that sector; negative tilts away. Only applies to equity tickers (crypto
has no news data).

The tilt is proportional and mild (controlled by `tilt_strength`) so that
the fused portfolio remains well-diversified. The fusion is look-ahead safe:
it uses only the lagged sentiment available at the rebalance date.

An honest negative result is fine — the marks are for evidenced work, not
for outperformance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    sector_map: pd.DataFrame,
    tilt_strength: float = 0.3,
) -> pd.DataFrame:
    """Tilt equity portfolio weights by lagged sector sentiment.

    Parameters
    ----------
    weights : DataFrame
        Monthly weight snapshots (rows = rebalance dates, columns = tickers).
    sentiment : DataFrame
        Sector sentiment index with columns [date, sector, sentiment_lagged].
    sector_map : DataFrame
        Ticker -> sector mapping with columns [ticker, sector].
    tilt_strength : float
        Controls how strongly sentiment tilts the weights. 0 = no tilt,
        1 = maximum tilt. Default 0.3 (moderate).

    Returns
    -------
    DataFrame with the same shape as `weights`, containing tilted weights.
    The tilt is: w_tilted_i = w_i * (1 + tilt_strength * z_sector_i),
    then renormalised to sum to 1.
    """
    ticker_to_sector = dict(zip(sector_map["ticker"], sector_map["sector"]))
    tilted = weights.copy()

    for idx in tilted.index:
        rebal_date = idx
        w = tilted.loc[idx].copy()

        # Find the most recent lagged sentiment for each sector
        avail = sentiment[
            (sentiment["date"] <= rebal_date) & sentiment["sentiment_lagged"].notna()
        ]
        if avail.empty:
            continue  # no sentiment available yet, keep original weights

        latest = avail.sort_values("date").groupby("sector").last()["sentiment_lagged"]

        # Standardise sentiment across sectors (z-score) for this date
        if latest.std() > 0:
            z_scores = (latest - latest.mean()) / latest.std()
        else:
            z_scores = latest * 0  # all equal -> no tilt

        # Apply tilt to each ticker
        for ticker in w.index:
            sector = ticker_to_sector.get(ticker)
            if sector and sector in z_scores.index:
                multiplier = 1.0 + tilt_strength * z_scores[sector]
                w[ticker] *= max(multiplier, 0.01)  # floor to avoid negatives

        # Renormalise to sum to 1
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum
        tilted.loc[idx] = w

    return tilted


def backtest_with_fusion(
    returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    tilted_weights: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Backtest both the base and sentiment-tilted funds for comparison.

    Parameters
    ----------
    returns : wide returns DataFrame (date x tickers)
    base_weights : monthly weight snapshots (no tilt)
    tilted_weights : monthly weight snapshots (sentiment tilt)

    Returns
    -------
    (base_returns, tilted_returns) : daily return Series for each
    """
    weight_dates = sorted(base_weights.index)

    base_rets = []
    tilted_rets = []

    for i, wdate in enumerate(weight_dates):
        # Holding period: from this rebalance to the next (or end)
        if i + 1 < len(weight_dates):
            mask = (returns.index >= wdate) & (returns.index < weight_dates[i + 1])
        else:
            mask = returns.index >= wdate

        block = returns.loc[mask]
        if block.empty:
            continue

        # Align tickers
        common = base_weights.columns.intersection(block.columns)
        bw = base_weights.loc[wdate, common].values
        tw = tilted_weights.loc[wdate, common].values
        ret_block = block[common].values

        base_rets.append(pd.Series(ret_block @ bw, index=block.index))
        tilted_rets.append(pd.Series(ret_block @ tw, index=block.index))

    base_daily = pd.concat(base_rets) if base_rets else pd.Series(dtype=float)
    tilted_daily = pd.concat(tilted_rets) if tilted_rets else pd.Series(dtype=float)

    return base_daily, tilted_daily