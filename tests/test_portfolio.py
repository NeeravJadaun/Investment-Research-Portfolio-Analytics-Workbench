import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    validate_weights, WeightValidationError, equal_weight_benchmark,
    compute_portfolio_returns, compute_risk_return_metrics, sector_concentration,
    contribution_to_risk, turnover,
)


def test_validate_weights_rejects_non_100_pct():
    with pytest.raises(WeightValidationError):
        validate_weights({"A": 0.5, "B": 0.3})


def test_validate_weights_rejects_negative():
    with pytest.raises(WeightValidationError):
        validate_weights({"A": 1.2, "B": -0.2})


def test_validate_weights_accepts_100_pct():
    validate_weights({"A": 0.4, "B": 0.6})  # should not raise


def test_equal_weight_benchmark_sums_to_one():
    bench = equal_weight_benchmark(["A", "B", "C", "D"])
    assert sum(bench.values()) == pytest.approx(1.0)
    assert all(w == pytest.approx(0.25) for w in bench.values())


def _synthetic_return_matrix():
    dates = pd.date_range("2023-01-31", periods=12, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "A": rng.normal(0.01, 0.03, 12),
            "B": rng.normal(0.005, 0.02, 12),
            "C": rng.normal(0.02, 0.05, 12),
        },
        index=dates,
    )


def test_portfolio_return_matches_manual_weighted_sum():
    rm = _synthetic_return_matrix()
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    result = compute_portfolio_returns(rm, weights, rebalance="monthly")
    manual = rm["A"] * 0.5 + rm["B"] * 0.3 + rm["C"] * 0.2
    pd.testing.assert_series_equal(result.period_returns, manual, check_names=False)


def test_cumulative_growth_is_reproducible():
    rm = _synthetic_return_matrix()
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    r1 = compute_portfolio_returns(rm, weights)
    r2 = compute_portfolio_returns(rm, weights)
    pd.testing.assert_series_equal(r1.cumulative_growth, r2.cumulative_growth)


def test_risk_return_metrics_reproducible_and_sane():
    rm = _synthetic_return_matrix()
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    bench_weights = equal_weight_benchmark(["A", "B", "C"])
    port = compute_portfolio_returns(rm, weights)
    bench = compute_portfolio_returns(rm, bench_weights)

    m1 = compute_risk_return_metrics(port.period_returns, bench.period_returns)
    m2 = compute_risk_return_metrics(port.period_returns, bench.period_returns)
    assert m1 == m2  # reproducible (dataclass equality)

    assert m1.annualized_volatility > 0
    assert -1.0 <= m1.max_drawdown <= 0.0


def test_max_drawdown_known_series():
    dates = pd.date_range("2023-01-31", periods=4, freq="ME")
    # growth path: 1 -> 1.10 -> 0.88 (-20% from peak) -> 0.99
    returns = pd.Series([0.10, -0.20, 0.125, 0.0], index=dates)
    metrics = compute_risk_return_metrics(returns, returns)
    assert metrics.max_drawdown == pytest.approx(-0.20, abs=1e-6)


def test_beta_of_series_against_itself_is_one():
    rm = _synthetic_return_matrix()
    returns = rm["A"]
    metrics = compute_risk_return_metrics(returns, returns)
    assert metrics.beta == pytest.approx(1.0, rel=1e-6)


def test_sector_concentration_sums_to_portfolio_weight():
    master = pd.DataFrame({"ticker": ["A", "B", "C"], "sector": ["Tech", "Tech", "Health"]})
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}
    conc = sector_concentration(weights, master)
    assert conc.sum() == pytest.approx(1.0)
    assert conc["Tech"] == pytest.approx(0.7)


def test_contribution_to_risk_sums_to_one():
    rm = _synthetic_return_matrix()
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    contrib = contribution_to_risk(rm, weights)
    assert contrib.sum() == pytest.approx(1.0, rel=1e-6)


def test_turnover_zero_for_identical_weights():
    weights = {"A": 0.5, "B": 0.5}
    assert turnover(weights, weights) == pytest.approx(0.0)


def test_turnover_full_replacement():
    w0 = {"A": 1.0}
    w1 = {"B": 1.0}
    assert turnover(w0, w1) == pytest.approx(1.0)
