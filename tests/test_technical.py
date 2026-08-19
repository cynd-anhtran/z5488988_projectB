"""Deterministic tests for calendars, trading costs and portfolio constraints."""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.features import build_return_universes  # noqa: E402
from src.sentiment import sector_sentiment_index  # noqa: E402
from src.portfolios import (  # noqa: E402
    PortfolioOptimisationError,
    drifted_weights,
    holding_period_returns,
    minimum_variance_weights,
    risk_parity_weights,
    run_oos_backtest,
    tangency_weights,
)


class CalendarTests(unittest.TestCase):
    def test_crypto_returns_are_computed_before_equity_alignment(self):
        equity_dates = pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-10"])
        crypto_dates = pd.date_range("2023-01-06", "2023-01-10", freq="D")

        equity = pd.DataFrame({
            "ticker": "EQ",
            "date": equity_dates,
            "adjClose": [100.0, 102.0, 103.0],
        })
        crypto = pd.DataFrame({
            "ticker": "CR-USD",
            "date": crypto_dates,
            "adjClose": [100.0, 110.0, 121.0, 133.1, 146.41],
        })

        universes = build_return_universes(equity, crypto)

        self.assertTrue((universes["Crypto"].index.dayofweek >= 5).any())
        self.assertTrue(universes["Combined"].index.equals(universes["Equity"].index))
        self.assertAlmostEqual(
            universes["Combined"].loc[pd.Timestamp("2023-01-09"), "CR-USD"],
            0.10,
        )

    def test_crypto_365_observations_start_live_period_in_january_2021(self):
        dates = pd.date_range("2020-01-02", "2021-01-31", freq="D")
        returns = pd.DataFrame(
            {"A": 0.001, "B": 0.0005}, index=dates,
        )
        equal_only = {"Equal": lambda mu, cov: np.array([0.5, 0.5])}

        oos, _, _ = run_oos_backtest(
            returns, equal_only, init_days=365,
        )

        self.assertEqual(oos.index.min(), pd.Timestamp("2021-01-01"))


class SentimentLagTests(unittest.TestCase):
    def test_sector_lag_uses_previous_available_observation(self):
        scores = pd.DataFrame({
            "trading_date": pd.to_datetime([
                "2023-01-03", "2023-01-02", "2023-01-02", "2023-01-03",
            ]),
            "ticker": ["A", "A", "B", "B"],
            "sector": ["Tech", "Tech", "Energy", "Energy"],
            "compound": [0.4, 0.1, -0.2, -0.1],
        })

        index = sector_sentiment_index(scores)
        for _, group in index.groupby("sector"):
            expected = group["sentiment"].shift(1)
            pd.testing.assert_series_equal(
                group["sentiment_lagged"].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
            )


class TransactionCostTests(unittest.TestCase):
    def test_pretrade_weights_drift_with_asset_returns(self):
        returns = pd.DataFrame([[0.10, 0.00]], columns=["A", "B"])
        drifted = drifted_weights(np.array([0.5, 0.5]), returns)
        np.testing.assert_allclose(drifted, [1.1 / 2.1, 1.0 / 2.1])

    def test_holding_period_returns_use_drifting_weights(self):
        returns = pd.DataFrame(
            [[0.10, 0.00], [0.10, 0.00]],
            index=pd.to_datetime(["2023-02-01", "2023-02-02"]),
            columns=["A", "B"],
        )
        realised, ending = holding_period_returns(np.array([0.5, 0.5]), returns)

        self.assertAlmostEqual(realised.iloc[0], 0.05)
        self.assertAlmostEqual(realised.iloc[1], (1.1 / 2.1) * 0.10)
        self.assertAlmostEqual(float((1.0 + realised).prod()), (1.21 + 1.0) / 2.0)
        np.testing.assert_allclose(ending, [1.21 / 2.21, 1.0 / 2.21])

    def test_costs_include_initial_trade_and_drifted_rebalance(self):
        dates = pd.to_datetime([
            "2023-01-02", "2023-01-03",
            "2023-02-01", "2023-02-02",
            "2023-03-01", "2023-03-02",
        ])
        returns = pd.DataFrame({
            "A": [0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
            "B": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        }, index=dates)
        equal_only = {"Equal": lambda mu, cov: np.array([0.5, 0.5])}

        gross, _, _ = run_oos_backtest(
            returns, equal_only, init_days=2, tx_cost_bps=0,
        )
        net, _, audit = run_oos_backtest(
            returns, equal_only, init_days=2, tx_cost_bps=100,
        )

        self.assertAlmostEqual(
            gross.loc[pd.Timestamp("2023-02-01"), "Equal"]
            - net.loc[pd.Timestamp("2023-02-01"), "Equal"],
            0.01,
        )
        expected_turnover = 2 * abs(0.5 - 1.1 / 2.1)
        self.assertAlmostEqual(
            gross.loc[pd.Timestamp("2023-03-01"), "Equal"]
            - net.loc[pd.Timestamp("2023-03-01"), "Equal"],
            expected_turnover * 0.01,
        )
        march = audit[audit["decision_month"] == "2023-03"].iloc[0]
        self.assertAlmostEqual(march["turnover"], expected_turnover)


class OptimisationTests(unittest.TestCase):
    def test_all_optimisers_return_valid_long_only_weights(self):
        cov = np.array([
            [0.040, 0.006, 0.004],
            [0.006, 0.090, 0.010],
            [0.004, 0.010, 0.160],
        ])
        mean = np.array([0.010, 0.014, 0.018])
        solutions = [
            minimum_variance_weights(cov),
            tangency_weights(mean, cov),
            risk_parity_weights(cov),
        ]
        for weights in solutions:
            self.assertTrue(np.isfinite(weights).all())
            self.assertGreaterEqual(weights.min(), -1e-10)
            self.assertAlmostEqual(weights.sum(), 1.0)

    def test_invalid_covariance_fails_explicitly(self):
        with self.assertRaises(PortfolioOptimisationError):
            minimum_variance_weights(np.array([[1.0, np.nan], [np.nan, 1.0]]))


if __name__ == "__main__":
    unittest.main()
