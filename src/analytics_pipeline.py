"""Assembles derived analytics (valuations, factor screen, default portfolio,
scenarios) on top of the processed tables written by `ingest.py`.

This module is the bridge between raw processed data and the two consumers
that need a consistent, precomputed view of it: `scripts/run_pipeline.py`
(which feeds `reporting.py`) and `app.py` (the Streamlit workbench, which
calls the same functions live so nothing is hard-coded).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.valuation import (
    DCFAssumptions, run_dcf, comparable_valuation, reit_nav, reit_cap_rate_check,
    dcf_sensitivity, reit_cap_rate_sensitivity,
)
from src.portfolio import (
    build_return_matrix, equal_weight_benchmark, compute_portfolio_returns,
    compute_risk_return_metrics, sector_concentration, security_concentration,
    contribution_to_return, contribution_to_risk, dividend_income, turnover,
)
from src.scenarios import BULL, BASE, BEAR, run_dcf_scenario_suite, run_reit_scenario_suite
from src.screen import build_factor_table, score_factor_table, DEFAULT_FACTOR_WEIGHTS, ResearchNoteInputs

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

RISK_FREE_RATE = 0.04
EQUITY_RISK_PREMIUM = 0.05
DEFAULT_TERMINAL_GROWTH = 0.02
PROJECTION_YEARS = 5

# Comp peer groups: Financial and Infrastructure are grouped together because
# the dataset only has 3 members across the two -- too few to peer separately.
PEER_GROUP_MAP = {"Company": "Company", "REIT": "REIT", "Financial": "Financial/Infrastructure", "Infrastructure": "Financial/Infrastructure"}


def load_processed_tables(processed_dir: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    names = [
        "security_master", "company_financials_annual", "company_financials_quarterly",
        "reit_financials_annual", "reit_financials_quarterly", "peer_multiples",
        "market_prices", "company_events", "data_quality", "data_dictionary",
    ]
    tables = db.read_tables(names, db_path=processed_dir / "research.db")
    for name in ("security_master", "company_events"):
        tables[name]["is_synthetic"] = tables[name]["is_synthetic"].astype(bool)
    return tables


def capm_discount_rate(beta: float, risk_free_rate: float = RISK_FREE_RATE, erp: float = EQUITY_RISK_PREMIUM) -> float:
    """discount_rate = risk_free_rate + beta * equity_risk_premium (simple CAPM cost of equity, used as a WACC proxy)."""
    return risk_free_rate + beta * erp


def build_dcf_assumptions_from_financials(
    ticker: str, master: pd.DataFrame, annual: pd.DataFrame,
    projection_years: int = PROJECTION_YEARS, terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
) -> DCFAssumptions:
    """Derive DCF assumptions for `ticker` from its historical annual financials.

    - base_revenue: latest fiscal year revenue.
    - revenue_growth: flat at the historical average YoY growth rate (clipped to [-10%, 25%]).
    - ebitda_margin: flat at the latest fiscal year's margin.
    - discount_rate: CAPM cost of equity using the security's beta (see capm_discount_rate).
    - tax_rate: flat 24%. capex/D&A: latest fiscal year's % of revenue. NWC: 5% of revenue change.
    - net_debt: latest debt - cash. shares_outstanding: latest reported.
    """
    hist = annual[annual["ticker"] == ticker].sort_values("fiscal_year")
    latest = hist.iloc[-1]
    beta = float(master.loc[master["ticker"] == ticker, "beta"].iloc[0])

    yoy_growth = hist["revenue_mm"].pct_change().dropna()
    avg_growth = float(np.clip(yoy_growth.mean(), -0.10, 0.25)) if len(yoy_growth) else 0.05

    ebitda_margin = float(latest["ebitda_mm"] / latest["revenue_mm"]) if latest["revenue_mm"] else 0.15
    capex_pct = float(latest["capex_mm"] / latest["revenue_mm"]) if latest["revenue_mm"] else 0.05
    da_pct = float((latest["ebitda_mm"] - latest["ebit_mm"]) / latest["revenue_mm"]) if latest["revenue_mm"] else 0.05

    discount_rate = capm_discount_rate(beta)
    if discount_rate <= terminal_growth:
        discount_rate = terminal_growth + 0.02

    return DCFAssumptions(
        base_revenue=float(latest["revenue_mm"]),
        revenue_growth=[avg_growth] * projection_years,
        ebitda_margin=[ebitda_margin] * projection_years,
        tax_rate=0.24,
        capex_pct_revenue=max(capex_pct, 0.01),
        da_pct_revenue=max(da_pct, 0.01),
        nwc_pct_revenue_change=0.05,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        net_debt=float(latest["debt_mm"] - latest["cash_mm"]),
        shares_outstanding=float(latest["shares_outstanding_mm"]),
        projection_years=projection_years,
    )


def run_dcf_for_all_companies(master: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Base-case DCF for every non-REIT security. Returns one row per ticker with
    the full assumption set alongside the resulting valuation."""
    non_reit = master[master["security_type"] != "REIT"]
    rows = []
    for ticker in non_reit["ticker"]:
        assumptions = build_dcf_assumptions_from_financials(ticker, master, annual)
        result = run_dcf(assumptions)
        latest_price = annual[(annual["ticker"] == ticker)].sort_values("fiscal_year").iloc[-1]["share_price"]
        rows.append(
            {
                "ticker": ticker,
                "current_price": latest_price,
                "revenue_growth_assumption": assumptions.revenue_growth[0],
                "ebitda_margin_assumption": assumptions.ebitda_margin[0],
                "discount_rate_assumption": assumptions.discount_rate,
                "terminal_growth_assumption": assumptions.terminal_growth,
                "net_debt": assumptions.net_debt,
                "shares_outstanding_mm": assumptions.shares_outstanding,
                "pv_of_explicit_fcf": result.pv_of_fcf,
                "terminal_value": result.terminal_value,
                "pv_of_terminal_value": result.pv_of_terminal_value,
                "enterprise_value": result.enterprise_value,
                "equity_value": result.equity_value,
                "dcf_value_per_share": result.value_per_share,
                "upside_vs_current_price": result.value_per_share / latest_price - 1 if latest_price else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_comps_for_all(master: pd.DataFrame, peer_multiples_latest: pd.DataFrame) -> pd.DataFrame:
    """Trading-comparables valuation for every security using peers from the same
    broad peer group (Company / REIT / Financial-Infrastructure), excluding self."""
    df = peer_multiples_latest.merge(master[["ticker", "security_type"]], on="ticker", suffixes=("", "_m"))
    df["peer_group"] = df["security_type"].map(PEER_GROUP_MAP)

    rows = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        group = row["peer_group"]
        shares = master.loc[master["ticker"] == ticker, "shares_outstanding_mm"].iloc[0]
        peers = df[(df["peer_group"] == group) & (df["ticker"] != ticker)]

        if not peers.empty:
            ev_comp = comparable_valuation(
                row["ebitda_mm"], peers["ev_ebitda_multiple"], "ev_ebitda", shares_outstanding=shares,
            )
            implied_ev = ev_comp.implied_enterprise_or_equity_value
            peer_median_ev_ebitda = ev_comp.peer_multiple_median
        else:
            implied_ev, peer_median_ev_ebitda = np.nan, np.nan

        rows.append(
            {
                "ticker": ticker,
                "peer_group": group,
                "peer_count": len(peers),
                "peer_median_ev_ebitda": peer_median_ev_ebitda,
                "own_ebitda_mm": row["ebitda_mm"],
                "implied_ev_from_comps": implied_ev,
                "peer_median_pe": peers["pe_multiple"].median() if not peers.empty else np.nan,
                "peer_median_pb": peers["pb_multiple"].median() if not peers.empty else np.nan,
                "peer_median_p_ffo": peers["p_ffo_multiple"].median() if not peers.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_comps_valuation_per_share(comps_row: pd.Series, master: pd.DataFrame, annual: pd.DataFrame) -> float:
    """Convert an implied EV from comps into an implied value per share for one ticker."""
    ticker = comps_row["ticker"]
    latest = annual[annual["ticker"] == ticker].sort_values("fiscal_year").iloc[-1]
    shares = master.loc[master["ticker"] == ticker, "shares_outstanding_mm"].iloc[0]
    net_debt = latest["debt_mm"] - latest["cash_mm"]
    implied_ev = comps_row["implied_ev_from_comps"]
    if pd.isna(implied_ev) or not shares:
        return np.nan
    return (implied_ev - net_debt) / shares


def run_nav_and_cap_rate_for_reits(master: pd.DataFrame, reit_annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reit_tickers = master[master["security_type"] == "REIT"]["ticker"]
    for ticker in reit_tickers:
        latest = reit_annual[reit_annual["ticker"] == ticker].sort_values("fiscal_year").iloc[-1]
        nav = reit_nav(latest["property_value_mm"], latest["net_debt_mm"], latest["shares_outstanding_mm"])
        cap_check = reit_cap_rate_check(latest["noi_mm"], latest["property_value_mm"], reference_cap_rate=latest["cap_rate"])
        rows.append(
            {
                "ticker": ticker,
                "fiscal_year": int(latest["fiscal_year"]),
                "property_value_mm": latest["property_value_mm"],
                "net_debt_mm": latest["net_debt_mm"],
                "noi_mm": latest["noi_mm"],
                "ffo_mm": latest["ffo_mm"],
                "affo_mm": latest["affo_mm"],
                "occupancy_pct": latest["occupancy_pct"],
                "shares_outstanding_mm": latest["shares_outstanding_mm"],
                "reported_cap_rate": latest["cap_rate"],
                "nav_mm": nav.nav,
                "nav_per_share": nav.nav_per_share,
                "implied_cap_rate_from_noi": cap_check.implied_cap_rate,
            }
        )
    return pd.DataFrame(rows)


def dcf_sensitivity_for_ticker(
    ticker: str, master: pd.DataFrame, annual: pd.DataFrame,
    discount_rate_spread: float = 0.02, terminal_growth_spread: float = 0.01, steps: int = 5,
) -> pd.DataFrame:
    """Discount-rate x terminal-growth sensitivity grid of DCF value/share, centered
    on the ticker's own CAPM-derived base-case assumptions."""
    base = build_dcf_assumptions_from_financials(ticker, master, annual)
    rates = list(np.linspace(base.discount_rate - discount_rate_spread, base.discount_rate + discount_rate_spread, steps))
    growths = list(np.linspace(max(base.terminal_growth - terminal_growth_spread, 0.0), base.terminal_growth + terminal_growth_spread, steps))
    return dcf_sensitivity(base, [round(r, 4) for r in rates], [round(g, 4) for g in growths])


def reit_cap_rate_sensitivity_for_ticker(ticker: str, reit_annual: pd.DataFrame, steps: int = 5) -> pd.DataFrame:
    """Cap-rate x NOI-growth sensitivity grid of NAV/share for a REIT ticker."""
    latest = reit_annual[reit_annual["ticker"] == ticker].sort_values("fiscal_year").iloc[-1]
    base_cap_rate = float(latest["cap_rate"])
    cap_rates = [round(r, 4) for r in np.linspace(max(base_cap_rate - 0.02, 0.02), base_cap_rate + 0.02, steps)]
    noi_growths = [round(g, 4) for g in np.linspace(-0.10, 0.10, steps)]
    return reit_cap_rate_sensitivity(
        noi=float(latest["noi_mm"]), net_debt=float(latest["net_debt_mm"]),
        shares_outstanding=float(latest["shares_outstanding_mm"]), cap_rates=cap_rates, noi_growth_rates=noi_growths,
    )


def dcf_scenarios_for_ticker(ticker: str, master: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    base = build_dcf_assumptions_from_financials(ticker, master, annual)
    return run_dcf_scenario_suite(base, [BEAR, BASE, BULL])


def reit_scenarios_for_ticker(ticker: str, reit_annual: pd.DataFrame) -> pd.DataFrame:
    latest = reit_annual[reit_annual["ticker"] == ticker].sort_values("fiscal_year").iloc[-1]
    return run_reit_scenario_suite(
        property_value=float(latest["property_value_mm"]), noi=float(latest["noi_mm"]),
        net_debt=float(latest["net_debt_mm"]), shares_outstanding=float(latest["shares_outstanding_mm"]),
        scenarios=[BEAR, BASE, BULL],
    )


def comps_valuation_summary(master: pd.DataFrame, annual: pd.DataFrame, peer_multiples: pd.DataFrame) -> pd.DataFrame:
    """Comps table (run_comps_for_all) enriched with an implied value-per-share column."""
    latest_year = annual["fiscal_year"].max()
    peers_latest = peer_multiples[peer_multiples["fiscal_year"] == latest_year]
    comps = run_comps_for_all(master, peers_latest)
    comps["implied_value_per_share"] = comps.apply(lambda r: run_comps_valuation_per_share(r, master, annual), axis=1)
    return comps


def build_factor_screen(processed: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    master = processed["security_master"]
    annual = processed["company_financials_annual"]
    peers = processed["peer_multiples"]
    prices = processed["market_prices"].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    latest_year = annual["fiscal_year"].max()
    prior_year = latest_year - 1
    latest = annual[annual["fiscal_year"] == latest_year]
    prior = annual[annual["fiscal_year"] == prior_year]
    peers_latest = peers[peers["fiscal_year"] == latest_year]

    momentum_12m = (
        prices.sort_values("date").groupby("ticker")
        .apply(lambda g: (1 + g.tail(12)["total_return"]).prod() - 1, include_groups=False)
    )

    factor_table = build_factor_table(master, latest, prior, peers_latest, momentum_12m)
    return score_factor_table(factor_table, weights or DEFAULT_FACTOR_WEIGHTS)


def build_default_portfolios(processed: dict[str, pd.DataFrame], factor_scores: pd.DataFrame) -> dict:
    """Model Portfolio: top 10 factor-ranked securities, weighted by (shifted,
    normalized) composite score so no name goes to zero or negative weight.
    Benchmarks: equal-weight across all 20 securities, and a market-cap-weighted
    custom benchmark across all 20."""
    master = processed["security_master"]
    annual = processed["company_financials_annual"]
    all_tickers = master["ticker"].tolist()

    top10 = factor_scores.head(10).copy()
    shifted = top10["composite_score"] - top10["composite_score"].min() + 0.1
    model_weights = dict(zip(top10["ticker"], shifted / shifted.sum()))

    eq_bench = equal_weight_benchmark(all_tickers)

    latest_year = annual["fiscal_year"].max()
    latest = annual[annual["fiscal_year"] == latest_year].set_index("ticker")
    market_cap = (latest["share_price"] * latest["shares_outstanding_mm"]).reindex(all_tickers)
    custom_bench = dict((market_cap / market_cap.sum()).fillna(0))

    return {"model_portfolio": model_weights, "equal_weight_benchmark": eq_bench, "custom_benchmark": custom_bench}


def compute_portfolio_analytics(processed: dict[str, pd.DataFrame], weights: dict[str, float], benchmark_weights: dict[str, float], rebalance: str = "monthly") -> dict:
    prices = processed["market_prices"]
    master = processed["security_master"]

    all_tickers = sorted(set(weights) | set(benchmark_weights))
    rm = build_return_matrix(prices, all_tickers)
    div_rm = build_return_matrix(prices, all_tickers, return_col="dividend_return")

    port = compute_portfolio_returns(rm, weights, rebalance=rebalance)
    bench = compute_portfolio_returns(rm, benchmark_weights, rebalance=rebalance)
    metrics = compute_risk_return_metrics(port.period_returns, bench.period_returns)

    sector_conc = sector_concentration(weights, master)
    security_conc = security_concentration(weights)
    contrib_return = contribution_to_return(rm, weights)
    contrib_risk = contribution_to_risk(rm, weights)
    div_income = dividend_income(weights, div_rm, portfolio_value=100.0)
    to = turnover(equal_weight_benchmark(list(weights.keys())), weights)

    return {
        "portfolio_returns": port, "benchmark_returns": bench, "metrics": metrics,
        "sector_concentration": sector_conc, "security_concentration": security_conc,
        "contribution_to_return": contrib_return, "contribution_to_risk": contrib_risk,
        "dividend_income_per_100": div_income, "turnover_vs_equal_weight": to,
    }


def portfolio_valuation_scenario_impact(processed: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Recompute the portfolio's weighted-average fair-value change under Bear/Base/
    Bull scenarios: each non-REIT holding re-run through the DCF scenario suite,
    each REIT holding through the REIT NAV scenario suite, then blended by
    portfolio weight. This is a valuation-level scenario (fair value change), not
    a price-return backtest.
    """
    master = processed["security_master"]
    annual = processed["company_financials_annual"]
    reit_annual = processed["reit_financials_annual"]
    reit_tickers = set(master[master["security_type"] == "REIT"]["ticker"])

    per_ticker_pct_change = {}
    for ticker in weights:
        if ticker in reit_tickers:
            df = reit_scenarios_for_ticker(ticker, reit_annual)
            per_ticker_pct_change[ticker] = df.set_index("scenario")["nav_per_share_pct_change_vs_base"]
        else:
            df = dcf_scenarios_for_ticker(ticker, master, annual)
            per_ticker_pct_change[ticker] = df.set_index("scenario")["value_per_share_pct_change_vs_base"]

    scenario_names = ["Bear", "Base", "Bull"]
    rows = []
    for scenario in scenario_names:
        weighted_change = sum(weights[t] * per_ticker_pct_change[t].get(scenario, 0.0) for t in weights)
        rows.append({"scenario": scenario, "weighted_avg_fair_value_change_pct": weighted_change})
    return pd.DataFrame(rows)


def portfolio_sector_shock_impact(processed: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Mark-to-market impact of applying each scenario's sector_price_shock_pct to
    the portfolio's single largest sector weight, holding all other holdings'
    prices flat -- a simple, transparent sector-shock stress test."""
    master = processed["security_master"]
    sector_conc = sector_concentration(weights, master)
    shocked_sector = sector_conc.index[0]
    sector_of = master.set_index("ticker")["sector"]

    rows = []
    for scenario in (BEAR, BASE, BULL):
        impact = sum(
            w * scenario.sector_price_shock_pct for t, w in weights.items() if sector_of.get(t) == shocked_sector
        )
        rows.append(
            {
                "scenario": scenario.name,
                "shocked_sector": shocked_sector,
                "shocked_sector_weight": sector_conc.iloc[0],
                "sector_price_shock_pct": scenario.sector_price_shock_pct,
                "portfolio_market_value_impact_pct": impact,
            }
        )
    return pd.DataFrame(rows)


def build_research_note_inputs(
    ticker: str, processed: dict[str, pd.DataFrame], dcf_all: pd.DataFrame,
    comps: pd.DataFrame, dcf_scenarios: pd.DataFrame, factor_scores: pd.DataFrame,
) -> ResearchNoteInputs:
    """Assemble a ResearchNoteInputs for `ticker` entirely from already-computed
    tables -- every field traces back to stored data or an assumption table."""
    master = processed["security_master"]
    events = processed["company_events"]
    sec = master[master["ticker"] == ticker].iloc[0]
    dcf_row = dcf_all[dcf_all["ticker"] == ticker].iloc[0]
    comp_row = comps[comps["ticker"] == ticker]
    comp_vps = float(comp_row["implied_value_per_share"].iloc[0]) if not comp_row.empty else float("nan")

    scen = dcf_scenarios.set_index("scenario")["value_per_share"]

    ticker_events = events[events["ticker"] == ticker].sort_values("event_date", ascending=False)
    catalysts = [
        f"{r['event_date']}: {r['description']}"
        for _, r in ticker_events[ticker_events["event_type"].isin(["Guidance Change", "Acquisition"])].head(3).iterrows()
    ]
    latest_earnings = ticker_events[ticker_events["event_type"] == "Earnings"].head(1)
    if not latest_earnings.empty:
        catalysts.append(f"Next/most recent earnings event: {latest_earnings.iloc[0]['event_date']}")

    risks = [
        f"Net debt of ${dcf_row['net_debt']:.0f}mm vs. discount rate assumption of {dcf_row['discount_rate_assumption']:.1%} -- leverage amplifies valuation sensitivity to rates.",
        f"DCF fair value implies {dcf_row['upside_vs_current_price']:+.1%} vs. current price -- a meaningful re-rating would be required for this thesis to play out.",
        "All financials, prices, and events are synthetic; this note demonstrates methodology, not a real market view.",
    ]

    rank_row = factor_scores[factor_scores["ticker"] == ticker].iloc[0]

    return ResearchNoteInputs(
        ticker=ticker, name=sec["name"], sector=sec["sector"], security_type=sec["security_type"],
        country=sec["country"], latest_price=float(dcf_row["current_price"]),
        dcf_value_per_share=float(dcf_row["dcf_value_per_share"]), comp_value_per_share=comp_vps,
        bull_value_per_share=float(scen.get("Bull", float("nan"))),
        base_value_per_share=float(scen.get("Base", float("nan"))),
        bear_value_per_share=float(scen.get("Bear", float("nan"))),
        key_assumptions={
            "Revenue growth (annual)": f"{dcf_row['revenue_growth_assumption']:.1%}",
            "EBITDA margin": f"{dcf_row['ebitda_margin_assumption']:.1%}",
            "Discount rate (CAPM)": f"{dcf_row['discount_rate_assumption']:.1%}",
            "Terminal growth": f"{dcf_row['terminal_growth_assumption']:.1%}",
            "Net debt ($mm)": f"{dcf_row['net_debt']:.1f}",
        },
        catalysts=catalysts, risks=risks,
        factor_rank=int(rank_row["rank"]), factor_total=len(factor_scores),
    )
