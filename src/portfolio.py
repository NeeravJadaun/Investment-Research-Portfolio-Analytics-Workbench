"""Portfolio construction and risk/return analytics.

All results here are backtested against the project's synthetic monthly price
series -- never live trading, never real performance. Functions are pure and
take DataFrames/dicts in, so results are reproducible and traceable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TOLERANCE = 1e-6


class WeightValidationError(ValueError):
    pass


def validate_weights(weights: dict[str, float]) -> None:
    """Weights must be non-negative and sum to 1.0 (within TOLERANCE)."""
    if not weights:
        raise WeightValidationError("No weights provided.")
    total = sum(weights.values())
    if any(w < 0 for w in weights.values()):
        raise WeightValidationError("Weights must be non-negative (no short positions in this workbench).")
    if abs(total - 1.0) > TOLERANCE:
        raise WeightValidationError(f"Weights must sum to 100%, got {total * 100:.4f}%.")


def build_return_matrix(prices: pd.DataFrame, tickers: list[str], return_col: str = "total_return") -> pd.DataFrame:
    """Pivot the long market_prices table into a wide date x ticker return matrix."""
    sub = prices[prices["ticker"].isin(tickers)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    wide = sub.pivot(index="date", columns="ticker", values=return_col).sort_index()
    return wide[tickers]


def equal_weight_benchmark(tickers: list[str]) -> dict[str, float]:
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


@dataclass
class PortfolioReturns:
    weights: dict[str, float]
    period_returns: pd.Series          # portfolio return per period (rebalanced to target weights each period)
    cumulative_growth: pd.Series       # growth of $1
    drifted_weights: pd.DataFrame      # weights at the end of each period before rebalance (buy & hold drift)


def compute_portfolio_returns(
    return_matrix: pd.DataFrame, weights: dict[str, float], rebalance: str = "monthly"
) -> PortfolioReturns:
    """Compute portfolio period returns under a simple rebalance assumption.

    rebalance='monthly': weights reset to target every period (return = sum(w_i * r_i)).
    rebalance='none': buy-and-hold -- weights drift with relative performance from
    the initial allocation, no rebalancing trades occur.
    """
    tickers = list(weights.keys())
    validate_weights(weights)
    rm = return_matrix[tickers].fillna(0.0)
    w0 = np.array([weights[t] for t in tickers])

    if rebalance == "monthly":
        period_returns = rm.dot(w0)
        period_returns.name = "portfolio_return"
        drifted = pd.DataFrame([w0] * len(rm), index=rm.index, columns=tickers)
    elif rebalance == "none":
        current = w0.copy()
        rows = []
        period_returns_list = []
        for dt, r in rm.iterrows():
            port_ret = float(np.dot(current, r.values))
            period_returns_list.append(port_ret)
            grown = current * (1 + r.values)
            current = grown / grown.sum() if grown.sum() != 0 else current
            rows.append(current.copy())
        period_returns = pd.Series(period_returns_list, index=rm.index, name="portfolio_return")
        drifted = pd.DataFrame(rows, index=rm.index, columns=tickers)
    else:
        raise ValueError("rebalance must be 'monthly' or 'none'")

    cumulative_growth = (1 + period_returns).cumprod()
    cumulative_growth.name = "cumulative_growth_of_1"

    return PortfolioReturns(
        weights=weights, period_returns=period_returns,
        cumulative_growth=cumulative_growth, drifted_weights=drifted,
    )


@dataclass
class RiskReturnMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    active_return_annualized: float


def compute_risk_return_metrics(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02
) -> RiskReturnMetrics:
    """
    total_return: geometric compounding over the full period.
    annualized_return: (1 + total_return)^(12/n_periods) - 1.
    annualized_volatility: std(monthly returns) * sqrt(12).
    max_drawdown: min over time of (cum/cummax - 1).
    sharpe_ratio: (annualized_return - risk_free_rate) / annualized_volatility.
    beta: cov(portfolio, benchmark) / var(benchmark), monthly.
    active_return_annualized: portfolio annualized_return - benchmark annualized_return.
    """
    pr = portfolio_returns.dropna()
    br = benchmark_returns.reindex(pr.index).dropna()
    pr = pr.reindex(br.index)

    n = len(pr)
    total_return = float((1 + pr).prod() - 1)
    annualized_return = float((1 + total_return) ** (MONTHS_PER_YEAR / n) - 1) if n > 0 else np.nan
    annualized_vol = float(pr.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if n > 1 else np.nan

    cum = (1 + pr).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_drawdown = float(drawdown.min()) if n > 0 else np.nan

    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol else np.nan

    cov = float(np.cov(pr.values, br.values, ddof=1)[0, 1]) if n > 1 else np.nan
    var_b = float(np.var(br.values, ddof=1)) if n > 1 else np.nan
    beta = cov / var_b if var_b else np.nan

    bench_total = float((1 + br).prod() - 1)
    bench_annualized = float((1 + bench_total) ** (MONTHS_PER_YEAR / n) - 1) if n > 0 else np.nan
    active_return = annualized_return - bench_annualized

    return RiskReturnMetrics(
        total_return=total_return, annualized_return=annualized_return,
        annualized_volatility=annualized_vol, max_drawdown=max_drawdown,
        sharpe_ratio=sharpe, beta=beta, active_return_annualized=active_return,
    )


def sector_concentration(weights: dict[str, float], security_master: pd.DataFrame) -> pd.Series:
    w = pd.Series(weights)
    sectors = security_master.set_index("ticker")["sector"].reindex(w.index)
    return w.groupby(sectors).sum().sort_values(ascending=False)


def security_concentration(weights: dict[str, float]) -> pd.Series:
    return pd.Series(weights).sort_values(ascending=False)


def contribution_to_return(return_matrix: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Approximate multi-period contribution: for each security, w_i * cumulative
    growth of $1 invested in that security over the period, normalized to sum
    to total portfolio contribution (additive attribution on total return)."""
    tickers = list(weights.keys())
    rm = return_matrix[tickers].fillna(0.0)
    w = pd.Series(weights)
    security_cum_return = (1 + rm).prod() - 1
    contribution = w * security_cum_return
    return contribution.sort_values(ascending=False)


def contribution_to_risk(return_matrix: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Contribution to portfolio variance: w_i * (Cov * w)_i / portfolio_variance,
    summing to 1.0 across securities (standard risk-contribution decomposition)."""
    tickers = list(weights.keys())
    rm = return_matrix[tickers].fillna(0.0)
    w = np.array([weights[t] for t in tickers])
    cov = rm.cov().values
    port_var = w @ cov @ w
    marginal = cov @ w
    contrib = w * marginal / port_var if port_var else np.zeros_like(w)
    return pd.Series(contrib, index=tickers).sort_values(ascending=False)


def dividend_income(weights: dict[str, float], return_matrix_dividends: pd.DataFrame, portfolio_value: float = 100.0) -> float:
    """Total simulated dividend income over the period for a $portfolio_value notional portfolio."""
    tickers = list(weights.keys())
    dr = return_matrix_dividends[tickers].fillna(0.0)
    w = pd.Series(weights)
    weighted_div_return = (dr * w).sum(axis=1)
    cum_div_return = weighted_div_return.sum()
    return float(cum_div_return * portfolio_value)


def turnover(weights_t0: dict[str, float], weights_t1: dict[str, float]) -> float:
    """One-way turnover = 0.5 * sum(|w1_i - w0_i|) across the union of tickers."""
    all_tickers = set(weights_t0) | set(weights_t1)
    diffs = [abs(weights_t1.get(t, 0.0) - weights_t0.get(t, 0.0)) for t in all_tickers]
    return 0.5 * sum(diffs)
