"""Station 3 - Funds: optimal portfolios + out-of-sample backtest.

Walk-forward out-of-sample backtest engine adapted from the course
dff_oos_helpers.py lecture scripts. No look-ahead: at each monthly
rebalance, weights are formed from an expanding window of PAST data only.

Methods implemented:
  - Equal-weight (1/N)
  - Minimum-variance (long-only, SLSQP)
  - Maximum-Sharpe / tangency (long-only, convex transform)
  - Risk parity (equal-risk-contribution, L-BFGS-B)

Innovation: transaction cost model — deducts cost proportional to turnover
at each rebalance.

Annualisation: equities sqrt(252), crypto sqrt(365), combined sqrt(252).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Annualisation constants
# ---------------------------------------------------------------------------
EQUITY_DAYS = 252
CRYPTO_DAYS = 365


class PortfolioOptimisationError(RuntimeError):
    """Raised when an optimiser fails or returns an invalid portfolio."""


def _validated_covariance(cov: np.ndarray, method: str) -> np.ndarray:
    """Return a finite, symmetric covariance matrix or fail explicitly."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise PortfolioOptimisationError(f"{method}: covariance matrix must be square.")
    if not np.isfinite(cov).all():
        raise PortfolioOptimisationError(f"{method}: covariance matrix contains non-finite values.")
    return 0.5 * (cov + cov.T)


def _validated_weights(weights: np.ndarray, n: int, method: str) -> np.ndarray:
    """Normalise a long-only solution and verify the portfolio constraints."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all():
        raise PortfolioOptimisationError(f"{method}: optimiser returned invalid weights.")
    if weights.min() < -1e-8:
        raise PortfolioOptimisationError(
            f"{method}: long-only constraint violated (minimum={weights.min():.3g})."
        )
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        raise PortfolioOptimisationError(f"{method}: weights have a non-positive sum.")
    weights /= total
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise PortfolioOptimisationError(f"{method}: weights do not sum to one.")
    return weights


def _require_solver_success(result, method: str) -> None:
    """Turn SciPy's silent failure state into a clear, reproducible error."""
    if not result.success:
        raise PortfolioOptimisationError(
            f"{method} failed (status={result.status}): {result.message}"
        )


# ---------------------------------------------------------------------------
# Weight functions (all long-only, fully invested, weights sum to 1)
# ---------------------------------------------------------------------------

def equal_weights(n: int) -> np.ndarray:
    """1/N benchmark: same fraction in every asset."""
    return np.ones(n) / n


def minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only minimum variance: minimise w' Sigma w s.t. sum(w)=1, w>=0.

    Uses SLSQP with analytic gradient 2*Sigma*w.
    """
    cov = _validated_covariance(cov, "Minimum-variance")
    n = cov.shape[0]
    result = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        jac=lambda w: 2 * cov @ w,
        bounds=[(0.0, None)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
            "jac": lambda w: np.ones(n),
        }],
        method="SLSQP",
        options={"maxiter": 300, "ftol": 1e-14},
    )
    _require_solver_success(result, "Minimum-variance")
    return _validated_weights(result.x, n, "Minimum-variance")


def tangency_weights(mean: np.ndarray, cov: np.ndarray, risk_free: float = 0.0) -> np.ndarray:
    """Long-only maximum-Sharpe (tangency) portfolio via convex transform.

    Minimise y' Sigma y s.t. (mu - rf)' y = 1, y>=0, then w = y / sum(y).
    Falls back to equal weights if no asset has positive excess return.
    """
    mean = np.asarray(mean, dtype=float)
    cov = _validated_covariance(cov, "Maximum-Sharpe")
    n = len(mean)
    if mean.shape != (cov.shape[0],) or not np.isfinite(mean).all():
        raise PortfolioOptimisationError("Maximum-Sharpe: invalid mean-return vector.")
    excess = mean - risk_free
    if (excess <= 0).all():
        return equal_weights(n)
    result = minimize(
        lambda y: y @ cov @ y,
        np.ones(n) / n,
        jac=lambda y: 2 * cov @ y,
        bounds=[(0.0, None)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda y: excess @ y - 1.0,
            "jac": lambda y: excess,
        }],
        method="SLSQP",
        options={"maxiter": 300, "ftol": 1e-12},
    )
    _require_solver_success(result, "Maximum-Sharpe")
    if not np.isclose(excess @ result.x, 1.0, atol=1e-6):
        raise PortfolioOptimisationError(
            "Maximum-Sharpe: transformed excess-return constraint was not satisfied."
        )
    return _validated_weights(result.x, n, "Maximum-Sharpe")


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only risk-parity (equal-risk-contribution) weights.

    Solved with the convex form: minimise 0.5 w'Sigma w - (1/n) sum(log w_i)
    over w>0, then rescale to sum to 1 (Maillard, Roncalli & Teiletche, 2010).
    """
    cov = _validated_covariance(cov, "Risk parity")
    n = cov.shape[0]
    result = minimize(
        lambda w: 0.5 * w @ cov @ w - np.mean(np.log(w)),
        np.ones(n) / n,
        jac=lambda w: cov @ w - 1.0 / (n * w),
        bounds=[(1e-9, None)] * n,
        method="L-BFGS-B",
        options={"maxiter": 800},
    )
    _require_solver_success(result, "Risk parity")
    w = _validated_weights(result.x, n, "Risk parity")

    portfolio_variance = float(w @ cov @ w)
    if portfolio_variance > 1e-16:
        contributions = w * (cov @ w)
        target = portfolio_variance / n
        relative_error = np.max(np.abs(contributions - target)) / target
        if relative_error > 1e-2:
            raise PortfolioOptimisationError(
                "Risk parity: equal-risk-contribution condition was not satisfied "
                f"(maximum relative error={relative_error:.2%})."
            )
    return w


# ---------------------------------------------------------------------------
# Weight function registry
# ---------------------------------------------------------------------------

WEIGHT_FUNCS = {
    "Equal-weight (1/N)": lambda mu, cov: equal_weights(len(mu)),
    "Minimum-variance": lambda mu, cov: minimum_variance_weights(cov),
    "Max-Sharpe (tangency)": lambda mu, cov: tangency_weights(mu, cov, 0.0),
    "Risk parity": lambda mu, cov: risk_parity_weights(cov),
}


def drifted_weights(weights: np.ndarray, asset_returns: pd.DataFrame) -> np.ndarray:
    """Carry target weights through a holding period to pre-trade weights.

    The calculation is self-financing: each position grows by its realised asset
    return and the resulting notionals are divided by the portfolio's ending
    value.  These are the weights that exist immediately before the next trade.
    """
    _, ending_weights = holding_period_returns(weights, asset_returns)
    return ending_weights


def holding_period_returns(
    weights: np.ndarray,
    asset_returns: pd.DataFrame,
) -> tuple[pd.Series, np.ndarray]:
    """Return a monthly buy-and-hold path and its ending pre-trade weights.

    Target weights are set once at the rebalance.  Thereafter each asset
    position compounds with its own realised return, so portfolio weights drift
    until the next scheduled rebalance.  This keeps the return calculation and
    the turnover calculation on the same self-financing convention.
    """
    current = _validated_weights(
        weights, asset_returns.shape[1], "Holding-period weights"
    )
    values = asset_returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot apply non-finite asset returns to a portfolio.")
    if (values <= -1.0).any():
        raise ValueError("Asset returns at or below -100% are not supported.")

    realised = np.empty(len(asset_returns), dtype=float)
    for row_number, row in enumerate(values):
        portfolio_return = float(current @ row)
        if not np.isfinite(portfolio_return) or portfolio_return <= -1.0:
            raise ValueError("Portfolio value became non-positive or non-finite.")
        realised[row_number] = portfolio_return
        current = current * (1.0 + row) / (1.0 + portfolio_return)

    return pd.Series(realised, index=asset_returns.index), current


# ---------------------------------------------------------------------------
# Out-of-sample backtest engine
# ---------------------------------------------------------------------------

def run_oos_backtest(
    returns: pd.DataFrame,
    weight_funcs: dict | None = None,
    init_days: int = EQUITY_DAYS,
    tx_cost_bps: float = 0.0,
    charge_initial_cost: bool = True,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Walk-forward out-of-sample backtest with no look-ahead.

    At each month-end, if at least `init_days` of past data exist:
      - Estimate mu and Sigma from ALL past returns (expanding window).
      - Solve each strategy's weights.
      - Apply weights to the next month's daily returns.

    Parameters
    ----------
    returns : DataFrame
        Wide daily returns (date x tickers), the backtest universe.
    weight_funcs : dict
        {strategy name -> function(mu, cov) -> weights}.
        Defaults to WEIGHT_FUNCS (all four methods).
    init_days : int
        Minimum training observations before the first live month.
    tx_cost_bps : float
        One-way transaction cost in basis points. Deducted proportionally
        to absolute traded notional at each rebalance. Pre-trade weights are
        the previous target weights drifted through realised asset returns.
    charge_initial_cost : bool
        If True, charge the initial purchase from cash to the first target.

    Returns
    -------
    oos_returns : DataFrame (date x strategy)
    weights : dict of {strategy: DataFrame of weight snapshots}
    audit : DataFrame of rebalance audit trail
    """
    if weight_funcs is None:
        weight_funcs = WEIGHT_FUNCS

    strategies = list(weight_funcs)
    period = returns.index.to_period("M")
    months = sorted(period.unique())

    oos = {s: [] for s in strategies}
    weights_log = {s: [] for s in strategies}
    pretrade_weights = {s: None for s in strategies}
    audit = []
    train_blocks = []
    train_days = 0

    tx_frac = tx_cost_bps / 10_000  # convert bps to decimal

    for month in months:
        block = returns[period == month]
        if train_days >= init_days:
            train = pd.concat(train_blocks)
            mu = train.mean().to_numpy()
            cov = np.cov(train.to_numpy(), rowvar=False, ddof=1)

            for strategy in strategies:
                w = weight_funcs[strategy](mu, cov)

                # Trade from drifted pre-trade holdings, not the previous target.
                if pretrade_weights[strategy] is None:
                    turnover = float(np.abs(w).sum())
                    cost = turnover * tx_frac if charge_initial_cost else 0.0
                else:
                    turnover = float(
                        np.abs(w - pretrade_weights[strategy]).sum()
                    )
                    cost = turnover * tx_frac

                # Hold the target positions without intermediate rebalancing.
                # The returned ending weights are the actual pre-trade holdings
                # from which the next month's rebalance is charged.
                port_ret, ending_weights = holding_period_returns(w, block)
                # Deduct tx cost on the first day of the month
                if cost > 0 and len(port_ret) > 0:
                    port_ret.iloc[0] -= cost

                oos[strategy].append(port_ret)
                weights_log[strategy].append(
                    pd.Series(w, index=returns.columns, name=block.index[0])
                )
                pretrade_weights[strategy] = ending_weights

                audit.append({
                    "decision_month": str(month),
                    "strategy": strategy,
                    "train_days": int(len(train)),
                    "first_holding_date": block.index[0],
                    "n_assets": int(returns.shape[1]),
                    "turnover": turnover,
                    "tx_cost": cost,
                    "tx_cost_bps": float(tx_cost_bps),
                })

        train_blocks.append(block)
        train_days += len(block)

    oos_returns = pd.DataFrame({s: pd.concat(v) for s, v in oos.items()})
    weights = {s: pd.DataFrame(v) for s, v in weights_log.items()}
    audit_df = pd.DataFrame(audit)

    return oos_returns, weights, audit_df


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def growth_of_one(returns: pd.Series) -> pd.Series:
    """Growth of $1: running product of (1 + r_t)."""
    return (1.0 + returns).cumprod()


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = EQUITY_DAYS) -> dict:
    """Compute a full scorecard of risk-return metrics.

    Returns dict with: total_return, ann_return (CAGR), ann_vol, sharpe,
    sortino, var_95, es_95, max_drawdown, n_obs, start, end.
    """
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {k: np.nan for k in [
            "total_return", "ann_return", "ann_vol", "sharpe", "sortino",
            "var_95", "es_95", "max_drawdown", "n_obs"
        ]}

    # Total return
    total = float((1.0 + r).prod() - 1.0)

    # Geometric annualised return (CAGR)
    ann_ret = float((1.0 + total) ** (periods_per_year / n) - 1.0)

    # Annualised volatility
    ann_vol = float(r.std() * np.sqrt(periods_per_year))

    # Sharpe ratio (rf = 0)
    sharpe = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan

    # Sortino ratio (rf = 0)
    downside = np.sqrt((np.minimum(r.values, 0.0) ** 2).mean())
    sortino = float(r.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan

    # Historical VaR at 95%
    var_95 = float(-r.quantile(0.05))

    # Historical Expected Shortfall (CVaR) at 95%
    tail = r[r <= r.quantile(0.05)]
    es_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    # Maximum drawdown
    wealth = growth_of_one(r)
    drawdown = float((wealth / wealth.cummax() - 1.0).min())

    return {
        "total_return": round(total, 4),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "var_95": round(var_95, 4),
        "es_95": round(es_95, 4),
        "max_drawdown": round(drawdown, 4),
        "n_obs": n,
        "start": str(r.index.min().date()) if hasattr(r.index.min(), "date") else str(r.index.min()),
        "end": str(r.index.max().date()) if hasattr(r.index.max(), "date") else str(r.index.max()),
    }
