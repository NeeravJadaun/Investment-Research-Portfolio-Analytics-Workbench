"""Repeatable end-to-end pipeline:

1. Generate (or regenerate) the synthetic raw dataset with a fixed seed.
2. Ingest and validate it -- dedupe, flag data-quality issues, write the
   data-quality report and data dictionary.
3. Run DCF, comparables, REIT NAV/cap-rate valuation for every security.
4. Run the factor screen and build the default Model Portfolio + benchmarks.
5. Run portfolio analytics and Bull/Base/Bear scenario analysis.
6. Write output/investment_research_pack.xlsx and a one-page Markdown
   research note for the top-ranked security.

Run with: python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src import generate_data, ingest, reporting
from src.analytics_pipeline import (
    load_processed_tables, run_dcf_for_all_companies, comps_valuation_summary,
    run_nav_and_cap_rate_for_reits, build_factor_screen, build_default_portfolios,
    compute_portfolio_analytics, dcf_sensitivity_for_ticker, dcf_scenarios_for_ticker,
    reit_scenarios_for_ticker, portfolio_valuation_scenario_impact,
    portfolio_sector_shock_impact, build_research_note_inputs,
    RISK_FREE_RATE, EQUITY_RISK_PREMIUM, DEFAULT_TERMINAL_GROWTH, PROJECTION_YEARS,
    PEER_GROUP_MAP,
)
from src.scenarios import BULL, BASE, BEAR
from src.screen import render_research_note, DEFAULT_FACTOR_WEIGHTS

OUTPUT_DIR = PROJECT_ROOT / "output"


def log(msg: str) -> None:
    print(f"[run_pipeline] {msg}")


def build_assumptions_table() -> pd.DataFrame:
    rows = [
        {"assumption": "Risk-free rate", "value": f"{RISK_FREE_RATE:.1%}", "used_in": "DCF discount rate (CAPM)"},
        {"assumption": "Equity risk premium", "value": f"{EQUITY_RISK_PREMIUM:.1%}", "used_in": "DCF discount rate (CAPM)"},
        {"assumption": "Discount rate formula", "value": "risk_free_rate + beta * equity_risk_premium", "used_in": "DCF"},
        {"assumption": "Terminal growth (base case)", "value": f"{DEFAULT_TERMINAL_GROWTH:.1%}", "used_in": "DCF Gordon-growth terminal value"},
        {"assumption": "Tax rate", "value": "24.0%", "used_in": "DCF unlevered FCF"},
        {"assumption": "Projection horizon", "value": f"{PROJECTION_YEARS} years", "used_in": "DCF"},
        {"assumption": "Comparable peer groups", "value": str(PEER_GROUP_MAP), "used_in": "Trading comparables"},
        {"assumption": "Bull scenario", "value": str(BULL), "used_in": "Scenario analysis"},
        {"assumption": "Base scenario", "value": str(BASE), "used_in": "Scenario analysis"},
        {"assumption": "Bear scenario", "value": str(BEAR), "used_in": "Scenario analysis"},
        {"assumption": "Factor screen default weights", "value": str(DEFAULT_FACTOR_WEIGHTS), "used_in": "Investment thesis / factor screen"},
        {"assumption": "Model Portfolio construction", "value": "Top 10 factor-ranked securities, weighted by shifted/normalized composite score", "used_in": "Portfolio Analytics"},
        {"assumption": "Rebalance assumption", "value": "Monthly rebalance to target weights (see src/portfolio.py for the buy-and-hold 'none' alternative)", "used_in": "Portfolio Analytics"},
        {"assumption": "Risk-free rate for Sharpe ratio", "value": "2.0%", "used_in": "Portfolio Sharpe ratio"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    log("Step 1/6: generating synthetic raw dataset (fixed seed)...")
    generate_data.generate_all()

    log("Step 2/6: ingesting, validating, and flagging data-quality issues...")
    ingest_result = ingest.run_ingestion()
    log(f"  -> {len(ingest_result['issues'])} data-quality issues detected and logged to data/processed/data_quality.csv")

    processed = load_processed_tables()
    master = processed["security_master"]
    annual = processed["company_financials_annual"]
    reit_annual = processed["reit_financials_annual"]
    peers = processed["peer_multiples"]

    log("Step 3/6: running DCF, comparables, and REIT NAV/cap-rate valuation for every security...")
    dcf_all = run_dcf_for_all_companies(master, annual)
    comps = comps_valuation_summary(master, annual, peers)
    nav_summary = run_nav_and_cap_rate_for_reits(master, reit_annual)

    log("Step 4/6: running the factor screen and building the Model Portfolio...")
    screen = build_factor_screen(processed)
    portfolios = build_default_portfolios(processed, screen)
    model_weights = portfolios["model_portfolio"]

    reit_tickers = set(master[master["security_type"] == "REIT"]["ticker"])
    featured_company = next(t for t in screen["ticker"] if t not in reit_tickers)
    featured_reit = next(t for t in screen["ticker"] if t in reit_tickers)
    log(f"  -> featured company for sensitivity/research note: {featured_company}; featured REIT: {featured_reit}")

    log("Step 5/6: running portfolio analytics and Bull/Base/Bear scenarios...")
    pa_eq = compute_portfolio_analytics(processed, model_weights, portfolios["equal_weight_benchmark"])
    pa_custom = compute_portfolio_analytics(processed, model_weights, portfolios["custom_benchmark"])

    dcf_sensitivity = dcf_sensitivity_for_ticker(featured_company, master, annual)
    dcf_scenarios = dcf_scenarios_for_ticker(featured_company, master, annual)
    reit_scenarios = reit_scenarios_for_ticker(featured_reit, reit_annual)
    port_val_impact = portfolio_valuation_scenario_impact(processed, model_weights)
    port_sector_shock = portfolio_sector_shock_impact(processed, model_weights)

    log("Step 6/6: writing Excel research pack and Markdown research note...")

    metrics_eq = pd.DataFrame([{"metric": k, "vs_equal_weight_benchmark": v} for k, v in vars(pa_eq["metrics"]).items()])
    metrics_custom = pd.DataFrame([{"metric": k, "vs_custom_benchmark": v} for k, v in vars(pa_custom["metrics"]).items()])
    portfolio_metrics = metrics_eq.merge(metrics_custom, on="metric")

    weights_df = pd.DataFrame(sorted(model_weights.items(), key=lambda kv: -kv[1]), columns=["ticker", "weight"])
    sector_conc_df = pa_eq["sector_concentration"].reset_index()
    sector_conc_df.columns = ["sector", "weight"]

    contrib_df = pd.DataFrame(
        {
            "contribution_to_return": pa_eq["contribution_to_return"],
            "contribution_to_risk": pa_eq["contribution_to_risk"],
        }
    ).reset_index().rename(columns={"index": "ticker"})

    growth_df = pd.DataFrame(
        {
            "date": pa_eq["portfolio_returns"].cumulative_growth.index,
            "model_portfolio_growth": pa_eq["portfolio_returns"].cumulative_growth.values,
            "equal_weight_benchmark_growth": pa_eq["benchmark_returns"].cumulative_growth.values,
        }
    )
    growth_df["date"] = growth_df["date"].dt.strftime("%Y-%m-%d")

    top_ideas = screen.head(10)[["rank", "ticker", "sector", "security_type", "composite_score"]]

    summary = {
        "Securities in universe": len(master),
        "Companies": int((master["security_type"] == "Company").sum()),
        "REITs": int((master["security_type"] == "REIT").sum()),
        "Financial / Infrastructure": int(master["security_type"].isin(["Financial", "Infrastructure"]).sum()),
        "Data-quality issues detected": len(ingest_result["issues"]),
        "Model Portfolio holdings": len(model_weights),
        "Featured company (DCF sensitivity/research note)": featured_company,
        "Featured REIT (NAV sensitivity)": featured_reit,
    }

    sheets_data = {
        "summary": summary,
        "top_ideas": top_ideas,
        "company_financials_annual": annual,
        "reit_financials_annual": reit_annual,
        "reit_nav_summary": nav_summary,
        "dcf_all": dcf_all,
        "dcf_sensitivity": dcf_sensitivity,
        "sensitivity_ticker": featured_company,
        "comps": comps,
        "peer_multiples_latest": peers[peers["fiscal_year"] == annual["fiscal_year"].max()],
        "portfolio_weights": weights_df,
        "portfolio_metrics": portfolio_metrics,
        "sector_concentration": sector_conc_df,
        "contribution": contrib_df,
        "growth_series": growth_df,
        "portfolio_other": {
            "Dividend income per $100 invested (period total)": round(pa_eq["dividend_income_per_100"], 4),
            "Turnover vs. equal-weight starting point": round(pa_eq["turnover_vs_equal_weight"], 4),
        },
        "dcf_scenarios": dcf_scenarios,
        "reit_scenarios": reit_scenarios,
        "portfolio_valuation_impact": port_val_impact,
        "portfolio_sector_shock": port_sector_shock,
        "featured_reit": featured_reit,
        "data_quality": processed["data_quality"],
        "data_dictionary": processed["data_dictionary"],
        "assumptions": build_assumptions_table(),
    }

    reporting.build_excel_pack(OUTPUT_DIR / "investment_research_pack.xlsx", sheets_data)
    log(f"  -> wrote {OUTPUT_DIR / 'investment_research_pack.xlsx'}")

    note_inputs = build_research_note_inputs(featured_company, processed, dcf_all, comps, dcf_scenarios, screen)
    note_md = render_research_note(note_inputs)
    reporting.write_research_note_file(OUTPUT_DIR / f"research_note_{featured_company}.md", note_md)
    log(f"  -> wrote {OUTPUT_DIR / f'research_note_{featured_company}.md'}")

    log("Pipeline complete.")


if __name__ == "__main__":
    main()
