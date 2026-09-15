import pandas as pd
import pytest

from src.valuation import (
    DCFAssumptions, run_dcf, comparable_valuation, reit_nav, reit_cap_rate_check,
    dcf_sensitivity, dcf_exit_multiple_sensitivity, reit_cap_rate_sensitivity,
)


def _base_dcf_assumptions(**overrides) -> DCFAssumptions:
    defaults = dict(
        base_revenue=100, revenue_growth=[0.10], ebitda_margin=[0.30], tax_rate=0.25,
        capex_pct_revenue=0.05, da_pct_revenue=0.05, nwc_pct_revenue_change=0.0,
        discount_rate=0.10, terminal_growth=0.02, net_debt=50, shares_outstanding=10,
        projection_years=1,
    )
    defaults.update(overrides)
    return DCFAssumptions(**defaults)


def test_dcf_matches_hand_calculation():
    """Hand-calculated 1-year DCF:
    revenue=110, ebitda=33, da=5.5, ebit=27.5, tax=6.875, capex=5.5, fcf=20.625
    terminal_value = 20.625*1.02/0.08 = 262.96875; pv_terminal = 239.0625
    pv_fcf = 18.75; EV=257.8125; equity=207.8125; value/share=20.78125
    """
    result = run_dcf(_base_dcf_assumptions())
    assert result.enterprise_value == pytest.approx(257.8125, rel=1e-6)
    assert result.equity_value == pytest.approx(207.8125, rel=1e-6)
    assert result.value_per_share == pytest.approx(20.78125, rel=1e-6)


def test_dcf_exit_multiple_terminal_value():
    """Exit multiple terminal value = final EBITDA * multiple, undiscounted.
    ebitda=33, multiple=9 -> terminal_value = 297."""
    a = _base_dcf_assumptions(terminal_method="exit_multiple", exit_multiple=9.0, terminal_growth=0.0)
    result = run_dcf(a)
    assert result.terminal_value == pytest.approx(297.0, rel=1e-6)


def test_dcf_assumption_direction_higher_discount_rate_lowers_value():
    low_rate = run_dcf(_base_dcf_assumptions(discount_rate=0.08))
    high_rate = run_dcf(_base_dcf_assumptions(discount_rate=0.14))
    assert high_rate.value_per_share < low_rate.value_per_share


def test_dcf_assumption_direction_higher_terminal_growth_raises_value():
    low_g = run_dcf(_base_dcf_assumptions(terminal_growth=0.01))
    high_g = run_dcf(_base_dcf_assumptions(terminal_growth=0.03))
    assert high_g.value_per_share > low_g.value_per_share


def test_dcf_assumption_direction_higher_revenue_growth_raises_value():
    low_growth = run_dcf(_base_dcf_assumptions(revenue_growth=[0.02]))
    high_growth = run_dcf(_base_dcf_assumptions(revenue_growth=[0.20]))
    assert high_growth.value_per_share > low_growth.value_per_share


def test_dcf_requires_discount_rate_above_terminal_growth():
    with pytest.raises(ValueError):
        _base_dcf_assumptions(discount_rate=0.02, terminal_growth=0.05)


def test_dcf_sensitivity_table_monotonic():
    base = _base_dcf_assumptions(projection_years=5, revenue_growth=[0.08] * 5, ebitda_margin=[0.25] * 5, discount_rate=0.09)
    grid = dcf_sensitivity(base, [0.08, 0.10, 0.12], [0.01, 0.02, 0.03])
    # value/share should decrease as discount rate rises (rows), increase as terminal growth rises (cols)
    assert grid.loc[0.08, 0.02] > grid.loc[0.12, 0.02]
    assert grid.loc[0.10, 0.03] > grid.loc[0.10, 0.01]


def test_dcf_exit_multiple_sensitivity_monotonic():
    base = _base_dcf_assumptions(projection_years=5, revenue_growth=[0.08] * 5, ebitda_margin=[0.25] * 5, discount_rate=0.09)
    grid = dcf_exit_multiple_sensitivity(base, [0.08, 0.10], [7, 11])
    assert grid.loc[0.08, 11] > grid.loc[0.08, 7]


def test_comparable_valuation_ev_ebitda_hand_calculation():
    """peer multiples [8, 9, 10] -> median 9; target EBITDA=33 -> implied EV=297;
    equity = 297 - net_debt(50) = 247; value/share = 24.7 (shares=10)."""
    peers = pd.Series([8.0, 9.0, 10.0])
    result = comparable_valuation(33, peers, "ev_ebitda", shares_outstanding=10, net_debt=50)
    assert result.peer_multiple_median == pytest.approx(9.0)
    assert result.implied_enterprise_or_equity_value == pytest.approx(297.0)


def test_reit_nav_hand_calculation():
    """NAV = property_value - net_debt = 1000 - 400 = 600; NAV/share = 600/50 = 12.0"""
    result = reit_nav(property_value=1000, net_debt=400, shares_outstanding=50)
    assert result.nav == pytest.approx(600.0)
    assert result.nav_per_share == pytest.approx(12.0)


def test_reit_cap_rate_check_hand_calculation():
    """implied_cap_rate = NOI/property_value = 60/1000 = 0.06;
    implied property value at a 5% reference rate = 60/0.05 = 1200."""
    result = reit_cap_rate_check(noi=60, property_value=1000, reference_cap_rate=0.05)
    assert result.implied_cap_rate == pytest.approx(0.06)
    assert result.implied_property_value_at_reference_rate == pytest.approx(1200.0)


def test_reit_cap_rate_sensitivity_direction():
    grid = reit_cap_rate_sensitivity(noi=120, net_debt=800, shares_outstanding=60, cap_rates=[0.05, 0.07], noi_growth_rates=[-0.05, 0.05])
    # higher cap rate -> lower NAV/share; higher NOI growth -> higher NAV/share
    assert grid.loc[0.05, -0.05] > grid.loc[0.07, -0.05]
    assert grid.loc[0.05, 0.05] > grid.loc[0.05, -0.05]
