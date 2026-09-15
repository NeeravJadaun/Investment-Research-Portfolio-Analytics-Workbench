"""Valuation methods: DCF, trading comparables, REIT NAV, cap-rate cross-check,
and generic sensitivity tables.

Every function documents its assumptions in its docstring/dataclass and returns
the full calculation trail (not just a headline number) so the workbench can
show its work rather than hard-coding a result.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Discounted Cash Flow
# --------------------------------------------------------------------------
@dataclass
class DCFAssumptions:
    """All inputs required to run a revenue/EBITDA-driven unlevered FCF DCF.

    revenue_growth / ebitda_margin: one value per projection year (length =
    projection_years). tax_rate, capex/D&A/NWC are expressed as % of revenue.
    discount_rate is the WACC applied to unlevered free cash flow.
    """

    base_revenue: float
    revenue_growth: list[float]
    ebitda_margin: list[float]
    tax_rate: float
    capex_pct_revenue: float
    da_pct_revenue: float
    nwc_pct_revenue_change: float
    discount_rate: float
    terminal_growth: float
    net_debt: float
    shares_outstanding: float
    projection_years: int = 5
    terminal_method: str = "gordon"  # 'gordon' or 'exit_multiple'
    exit_multiple: float | None = None  # EV/EBITDA multiple applied to final-year EBITDA when terminal_method='exit_multiple'

    def __post_init__(self):
        if len(self.revenue_growth) != self.projection_years:
            raise ValueError("revenue_growth must have one entry per projection year")
        if len(self.ebitda_margin) != self.projection_years:
            raise ValueError("ebitda_margin must have one entry per projection year")
        if self.terminal_method not in ("gordon", "exit_multiple"):
            raise ValueError("terminal_method must be 'gordon' or 'exit_multiple'")
        if self.terminal_method == "gordon" and self.discount_rate <= self.terminal_growth:
            raise ValueError("discount_rate must exceed terminal_growth for the Gordon growth terminal value to be finite/positive")
        if self.terminal_method == "exit_multiple" and not self.exit_multiple:
            raise ValueError("exit_multiple must be set when terminal_method='exit_multiple'")


@dataclass
class DCFResult:
    assumptions: DCFAssumptions
    cash_flow_trail: pd.DataFrame
    pv_of_fcf: float
    terminal_value: float
    pv_of_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float


def run_dcf(a: DCFAssumptions) -> DCFResult:
    """Unlevered FCF DCF: Revenue -> EBITDA -> EBIT -> NOPAT -> unlevered FCF.

    unlevered_fcf = ebitda - da - taxes_on_ebit - capex - change_in_nwc

    Terminal value, two supported methods (a.terminal_method):
    - 'gordon': fcf_final_year * (1 + g) / (discount_rate - g)
    - 'exit_multiple': ebitda_final_year * exit_multiple  (implied EV at exit)

    enterprise_value = sum(discounted fcf) + discounted terminal_value
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares_outstanding
    """
    years = list(range(1, a.projection_years + 1))
    revenue, ebitda, da, ebit, taxes, capex, nwc_change, fcf, discount_factor, pv_fcf = ([] for _ in range(10))

    rev = a.base_revenue
    prev_rev = a.base_revenue
    for i in range(a.projection_years):
        rev = rev * (1 + a.revenue_growth[i])
        ebd = rev * a.ebitda_margin[i]
        d_a = rev * a.da_pct_revenue
        ebt = ebd - d_a
        tax = max(ebt, 0) * a.tax_rate
        cpx = rev * a.capex_pct_revenue
        nwc = (rev - prev_rev) * a.nwc_pct_revenue_change
        f = ebd - tax - cpx - nwc
        df_ = 1 / (1 + a.discount_rate) ** (i + 1)

        revenue.append(rev); ebitda.append(ebd); da.append(d_a); ebit.append(ebt)
        taxes.append(tax); capex.append(cpx); nwc_change.append(nwc); fcf.append(f)
        discount_factor.append(df_); pv_fcf.append(f * df_)
        prev_rev = rev

    trail = pd.DataFrame(
        {
            "year": years,
            "revenue": revenue,
            "ebitda": ebitda,
            "d_and_a": da,
            "ebit": ebit,
            "taxes_on_ebit": taxes,
            "capex": capex,
            "nwc_change": nwc_change,
            "unlevered_fcf": fcf,
            "discount_factor": discount_factor,
            "pv_of_fcf": pv_fcf,
        }
    )

    if a.terminal_method == "exit_multiple":
        terminal_value = ebitda[-1] * a.exit_multiple
    else:
        terminal_value = fcf[-1] * (1 + a.terminal_growth) / (a.discount_rate - a.terminal_growth)
    pv_terminal = terminal_value * discount_factor[-1]
    pv_fcf_sum = sum(pv_fcf)
    ev = pv_fcf_sum + pv_terminal
    equity_value = ev - a.net_debt
    value_per_share = equity_value / a.shares_outstanding if a.shares_outstanding else np.nan

    return DCFResult(
        assumptions=a,
        cash_flow_trail=trail,
        pv_of_fcf=pv_fcf_sum,
        terminal_value=terminal_value,
        pv_of_terminal_value=pv_terminal,
        enterprise_value=ev,
        equity_value=equity_value,
        value_per_share=value_per_share,
    )


# --------------------------------------------------------------------------
# 2. Trading comparables
# --------------------------------------------------------------------------
@dataclass
class ComparablesResult:
    method: str
    peer_multiple_median: float
    peer_multiple_mean: float
    target_metric: float
    implied_enterprise_or_equity_value: float
    implied_value_per_share: float
    peers_used: pd.DataFrame


def comparable_valuation(
    target_metric_value: float,
    peer_multiples: pd.Series,
    method: str,
    shares_outstanding: float,
    net_debt: float = 0.0,
    peers_table: pd.DataFrame | None = None,
) -> ComparablesResult:
    """Apply a peer median/mean multiple to a target's own metric.

    method in {'ev_ebitda', 'pe', 'pb', 'p_ffo'}.
    - ev_ebitda: implied EV = median(EV/EBITDA peers) * target EBITDA; equity value = EV - net_debt
    - pe / pb / p_ffo: implied equity value per share = median(multiple) * target per-share metric
      (target_metric_value is expected to already be per-share for pe/pb/p_ffo)
    """
    peer_multiples = peer_multiples.dropna()
    if peer_multiples.empty:
        raise ValueError(f"No peer multiples available for method '{method}'")

    median_mult = float(peer_multiples.median())
    mean_mult = float(peer_multiples.mean())

    if method == "ev_ebitda":
        implied_ev = median_mult * target_metric_value
        implied_equity = implied_ev - net_debt
        implied_value = implied_ev
        value_per_share = implied_equity / shares_outstanding if shares_outstanding else np.nan
    elif method in ("pe", "pb", "p_ffo"):
        implied_value = median_mult * target_metric_value
        value_per_share = implied_value
    else:
        raise ValueError(f"Unknown comparables method: {method}")

    return ComparablesResult(
        method=method,
        peer_multiple_median=median_mult,
        peer_multiple_mean=mean_mult,
        target_metric=target_metric_value,
        implied_enterprise_or_equity_value=implied_value,
        implied_value_per_share=value_per_share,
        peers_used=peers_table if peers_table is not None else pd.DataFrame({"multiple": peer_multiples}),
    )


# --------------------------------------------------------------------------
# 3. REIT NAV
# --------------------------------------------------------------------------
@dataclass
class NAVResult:
    property_value: float
    net_debt: float
    shares_outstanding: float
    nav: float
    nav_per_share: float


def reit_nav(property_value: float, net_debt: float, shares_outstanding: float) -> NAVResult:
    """NAV = property_value - net_debt; NAV per share = NAV / shares_outstanding."""
    nav = property_value - net_debt
    nav_ps = nav / shares_outstanding if shares_outstanding else np.nan
    return NAVResult(
        property_value=property_value,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
        nav=nav,
        nav_per_share=nav_ps,
    )


# --------------------------------------------------------------------------
# 4. REIT capitalization-rate cross-check
# --------------------------------------------------------------------------
@dataclass
class CapRateCheckResult:
    noi: float
    property_value: float
    implied_cap_rate: float
    reference_cap_rate: float | None
    implied_property_value_at_reference_rate: float | None


def reit_cap_rate_check(noi: float, property_value: float, reference_cap_rate: float | None = None) -> CapRateCheckResult:
    """Cross-check reported property value against NOI-derived cap rate.

    implied_cap_rate = NOI / property_value.
    If a reference (e.g. sector-average) cap rate is supplied, also back into
    an implied property value = NOI / reference_cap_rate for comparison.
    """
    implied_cap_rate = noi / property_value if property_value else np.nan
    implied_pv = noi / reference_cap_rate if reference_cap_rate else None
    return CapRateCheckResult(
        noi=noi,
        property_value=property_value,
        implied_cap_rate=implied_cap_rate,
        reference_cap_rate=reference_cap_rate,
        implied_property_value_at_reference_rate=implied_pv,
    )


# --------------------------------------------------------------------------
# 5. Generic 2D sensitivity table
# --------------------------------------------------------------------------
def sensitivity_table(
    calc_fn,
    row_param: str,
    row_values: list[float],
    col_param: str,
    col_values: list[float],
    base_kwargs: dict,
    result_attr: str,
) -> pd.DataFrame:
    """Build a 2D grid of `calc_fn(**base_kwargs, row_param=r, col_param=c)` results.

    `calc_fn` must accept keyword overrides matching row_param/col_param and
    return an object (or dataclass) exposing `result_attr`.
    """
    grid = pd.DataFrame(index=row_values, columns=col_values, dtype=float)
    for r in row_values:
        for c in col_values:
            kwargs = dict(base_kwargs)
            kwargs[row_param] = r
            kwargs[col_param] = c
            result = calc_fn(**kwargs)
            grid.loc[r, c] = getattr(result, result_attr)
    grid.index.name = row_param
    grid.columns.name = col_param
    return grid


def dcf_sensitivity(
    base: DCFAssumptions,
    discount_rates: list[float],
    terminal_growths: list[float],
    result_attr: str = "value_per_share",
) -> pd.DataFrame:
    """Convenience wrapper: sensitivity of DCF value_per_share to discount rate x terminal growth."""

    def _run(discount_rate, terminal_growth):
        kwargs = asdict(base)
        kwargs["discount_rate"] = discount_rate
        kwargs["terminal_growth"] = terminal_growth
        return run_dcf(DCFAssumptions(**kwargs))

    return sensitivity_table(
        _run, "discount_rate", discount_rates, "terminal_growth", terminal_growths,
        base_kwargs={}, result_attr=result_attr,
    )


def dcf_exit_multiple_sensitivity(
    base: DCFAssumptions,
    discount_rates: list[float],
    exit_multiples: list[float],
    result_attr: str = "value_per_share",
) -> pd.DataFrame:
    """Convenience wrapper: sensitivity of DCF value_per_share to discount rate x
    EV/EBITDA exit multiple, using the exit-multiple terminal value method."""

    def _run(discount_rate, exit_multiple):
        kwargs = asdict(base)
        kwargs["discount_rate"] = discount_rate
        kwargs["terminal_method"] = "exit_multiple"
        kwargs["exit_multiple"] = exit_multiple
        return run_dcf(DCFAssumptions(**kwargs))

    return sensitivity_table(
        _run, "discount_rate", discount_rates, "exit_multiple", exit_multiples,
        base_kwargs={}, result_attr=result_attr,
    )


def reit_cap_rate_sensitivity(
    noi: float, net_debt: float, shares_outstanding: float,
    cap_rates: list[float], noi_growth_rates: list[float],
) -> pd.DataFrame:
    """Sensitivity of REIT NAV-per-share to capitalization rate x NOI growth.

    For each grid cell: shocked_noi = noi * (1 + noi_growth); implied_property_value
    = shocked_noi / cap_rate (the cap-rate cross-check run in reverse); NAV per
    share = (implied_property_value - net_debt) / shares_outstanding.
    """
    grid = pd.DataFrame(index=cap_rates, columns=noi_growth_rates, dtype=float)
    for cr in cap_rates:
        for g in noi_growth_rates:
            shocked_noi = noi * (1 + g)
            implied_pv = shocked_noi / cr
            nav_result = reit_nav(implied_pv, net_debt, shares_outstanding)
            grid.loc[cr, g] = nav_result.nav_per_share
    grid.index.name = "cap_rate"
    grid.columns.name = "noi_growth_rate"
    return grid
