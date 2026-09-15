"""Scenario analysis: bull / base / bear / user-defined shocks applied to
valuation and portfolio assumptions.

A scenario is a set of deltas (not absolute overrides) applied on top of a
base case, so the same scenario definition can be reused across different
securities and portfolios. Every scenario result is reported alongside the
base case and the change versus base, and is explicitly a simulated /
backtested output, not a forecast guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from src.valuation import DCFAssumptions, run_dcf, reit_nav, reit_cap_rate_check


@dataclass
class Scenario:
    name: str
    revenue_growth_delta: float = 0.0       # additive, applied to each projection year
    ebitda_margin_delta: float = 0.0        # additive
    discount_rate_delta: float = 0.0        # additive (interest-rate sensitivity proxy)
    terminal_growth_delta: float = 0.0      # additive
    cap_rate_delta: float = 0.0             # additive, REIT cap-rate shock
    leverage_delta_pct: float = 0.0         # % change applied to net_debt / net_debt_mm
    sector_price_shock_pct: float = 0.0     # % change applied to a sector's share prices


BULL = Scenario(
    name="Bull",
    revenue_growth_delta=0.03,
    ebitda_margin_delta=0.02,
    discount_rate_delta=-0.01,
    terminal_growth_delta=0.005,
    cap_rate_delta=-0.005,
    leverage_delta_pct=-0.05,
    sector_price_shock_pct=0.10,
)

BASE = Scenario(name="Base")

BEAR = Scenario(
    name="Bear",
    revenue_growth_delta=-0.04,
    ebitda_margin_delta=-0.03,
    discount_rate_delta=0.015,
    terminal_growth_delta=-0.005,
    cap_rate_delta=0.01,
    leverage_delta_pct=0.08,
    sector_price_shock_pct=-0.15,
)


def apply_scenario_to_dcf(base_assumptions: DCFAssumptions, scenario: Scenario) -> DCFAssumptions:
    """Return a new DCFAssumptions with the scenario's deltas applied on top of base."""
    new_growth = [g + scenario.revenue_growth_delta for g in base_assumptions.revenue_growth]
    new_margin = [max(m + scenario.ebitda_margin_delta, 0.01) for m in base_assumptions.ebitda_margin]
    new_discount = max(base_assumptions.discount_rate + scenario.discount_rate_delta, 0.01)
    new_terminal = base_assumptions.terminal_growth + scenario.terminal_growth_delta
    new_net_debt = base_assumptions.net_debt * (1 + scenario.leverage_delta_pct)

    if new_discount <= new_terminal:
        new_discount = new_terminal + 0.01

    return replace(
        base_assumptions,
        revenue_growth=new_growth,
        ebitda_margin=new_margin,
        discount_rate=new_discount,
        terminal_growth=new_terminal,
        net_debt=new_net_debt,
    )


def run_dcf_scenario_suite(
    base_assumptions: DCFAssumptions, scenarios: list[Scenario] | None = None
) -> pd.DataFrame:
    """Run base + bull/bear (+ any extra user-defined scenarios) and report
    value_per_share, enterprise_value, and the change vs. the Base scenario."""
    scenarios = scenarios or [BEAR, BASE, BULL]
    rows = []
    base_result = None
    for scenario in scenarios:
        adj = apply_scenario_to_dcf(base_assumptions, scenario)
        result = run_dcf(adj)
        if scenario.name == "Base":
            base_result = result
        rows.append(
            {
                "scenario": scenario.name,
                "revenue_growth_y1": adj.revenue_growth[0],
                "ebitda_margin_y1": adj.ebitda_margin[0],
                "discount_rate": adj.discount_rate,
                "terminal_growth": adj.terminal_growth,
                "net_debt": adj.net_debt,
                "enterprise_value": result.enterprise_value,
                "equity_value": result.equity_value,
                "value_per_share": result.value_per_share,
            }
        )
    df = pd.DataFrame(rows)
    if base_result is not None:
        df["value_per_share_change_vs_base"] = df["value_per_share"] - base_result.value_per_share
        df["value_per_share_pct_change_vs_base"] = (
            df["value_per_share"] / base_result.value_per_share - 1
        )
    return df


def apply_scenario_to_reit(
    property_value: float, noi: float, net_debt: float, shares_outstanding: float, scenario: Scenario
) -> dict:
    """Apply a scenario's cap-rate and leverage shocks to a REIT's NAV and cap-rate
    cross-check. property_value is re-derived from NOI at the shocked cap rate,
    which is how a cap-rate shock actually transmits to REIT valuations."""
    base_cap_rate = noi / property_value if property_value else float("nan")
    shocked_cap_rate = max(base_cap_rate + scenario.cap_rate_delta, 0.01)
    shocked_property_value = noi / shocked_cap_rate
    shocked_net_debt = net_debt * (1 + scenario.leverage_delta_pct)

    nav_result = reit_nav(shocked_property_value, shocked_net_debt, shares_outstanding)
    cap_check = reit_cap_rate_check(noi, shocked_property_value, reference_cap_rate=base_cap_rate)

    return {
        "scenario": scenario.name,
        "base_cap_rate": base_cap_rate,
        "shocked_cap_rate": shocked_cap_rate,
        "shocked_property_value": shocked_property_value,
        "shocked_net_debt": shocked_net_debt,
        "nav": nav_result.nav,
        "nav_per_share": nav_result.nav_per_share,
        "implied_cap_rate_check": cap_check.implied_cap_rate,
    }


def run_reit_scenario_suite(
    property_value: float, noi: float, net_debt: float, shares_outstanding: float,
    scenarios: list[Scenario] | None = None,
) -> pd.DataFrame:
    scenarios = scenarios or [BEAR, BASE, BULL]
    rows = [apply_scenario_to_reit(property_value, noi, net_debt, shares_outstanding, s) for s in scenarios]
    df = pd.DataFrame(rows)
    base_row = df[df["scenario"] == "Base"]
    if not base_row.empty:
        base_nav_ps = base_row["nav_per_share"].iloc[0]
        df["nav_per_share_change_vs_base"] = df["nav_per_share"] - base_nav_ps
        df["nav_per_share_pct_change_vs_base"] = df["nav_per_share"] / base_nav_ps - 1
    return df


def apply_sector_price_shock(
    prices: pd.Series, sector_of_ticker: pd.Series, sector: str, scenario: Scenario
) -> pd.Series:
    """Apply a scenario's sector price shock to the latest prices of securities in
    that sector; other sectors are left unchanged. Used by the portfolio scenario
    lab to re-price a portfolio's holdings under a shock scenario."""
    shocked = prices.copy()
    mask = sector_of_ticker.reindex(prices.index) == sector
    shocked[mask] = prices[mask] * (1 + scenario.sector_price_shock_pct)
    return shocked
