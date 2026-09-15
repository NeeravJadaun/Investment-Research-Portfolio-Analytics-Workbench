"""Independent verification of the Investment Research & Portfolio Analytics
Workbench.

Runs the pipeline, runs the test suite, checks that every required output
file exists, and independently recomputes a DCF value, a REIT NAV, and a
portfolio Sharpe ratio from raw formulas (NOT by calling src/valuation.py or
src/portfolio.py) to cross-check the pipeline's own numbers.

Run with: python scripts/verify_project.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3

import numpy as np
import pandas as pd
from openpyxl import load_workbook

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = PROCESSED_DIR / "research.db"

REQUIRED_FILES = [
    DB_PATH,
    PROCESSED_DIR / "data_quality.csv",
    PROCESSED_DIR / "data_dictionary.csv",
    PROCESSED_DIR / "security_master.csv",
    PROCESSED_DIR / "company_financials_annual.csv",
    PROCESSED_DIR / "reit_financials_annual.csv",
    PROCESSED_DIR / "market_prices.csv",
    OUTPUT_DIR / "investment_research_pack.xlsx",
]

REQUIRED_DB_TABLES = [
    "security_master", "company_financials_annual", "company_financials_quarterly",
    "reit_financials_annual", "reit_financials_quarterly", "peer_multiples",
    "market_prices", "company_events", "data_quality", "data_dictionary",
]

REQUIRED_SHEETS = [
    "Research Summary", "Company Financials", "REIT Metrics", "DCF Valuation",
    "Comparable Companies", "Portfolio Analytics", "Scenario Analysis",
    "Data Quality", "Assumptions and Data Dictionary",
]


def fail(message: str) -> None:
    print(f"VERIFY FAILED: {message}")
    sys.exit(1)


def step(message: str) -> None:
    print(f"\n=== {message} ===")


def run_pipeline() -> None:
    step("Running scripts/run_pipeline.py")
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline.py")], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        fail("run_pipeline.py exited non-zero")
    print("Pipeline ran successfully.")


def run_tests() -> None:
    step("Running pytest")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        fail("pytest reported failures")
    print("All tests passed.")


def check_required_files() -> None:
    step("Checking required output files")
    for f in REQUIRED_FILES:
        if not f.exists():
            fail(f"missing required file: {f.relative_to(PROJECT_ROOT)}")
        print(f"  found: {f.relative_to(PROJECT_ROOT)}")

    conn = sqlite3.connect(DB_PATH)
    try:
        existing_tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    for table in REQUIRED_DB_TABLES:
        if table not in existing_tables:
            fail(f"research.db missing required normalized table: {table}")
    print(f"  research.db has all {len(REQUIRED_DB_TABLES)} required tables.")

    wb = load_workbook(OUTPUT_DIR / "investment_research_pack.xlsx")
    for sheet in REQUIRED_SHEETS:
        if sheet not in wb.sheetnames:
            fail(f"Excel workbook missing required sheet: {sheet}")
    print(f"  Excel workbook has all {len(REQUIRED_SHEETS)} required sheets.")

    notes = list(OUTPUT_DIR.glob("research_note_*.md"))
    if not notes:
        fail("no research_note_*.md file found in output/")
    print(f"  found research note: {notes[0].relative_to(PROJECT_ROOT)}")


def independently_recompute_dcf() -> None:
    """Recompute a 1-year DCF value/share from scratch, by hand, and compare
    against src/valuation.py's run_dcf -- without importing/calling it."""
    step("Independently recomputing a DCF value/share (not via src/valuation.py)")

    revenue0, growth, margin, tax_rate = 100.0, 0.10, 0.30, 0.25
    capex_pct, da_pct, discount_rate, terminal_growth = 0.05, 0.05, 0.10, 0.02
    net_debt, shares = 50.0, 10.0

    revenue = revenue0 * (1 + growth)
    ebitda = revenue * margin
    da = revenue * da_pct
    ebit = ebitda - da
    tax = ebit * tax_rate
    capex = revenue * capex_pct
    fcf = ebitda - tax - capex
    terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    discount_factor = 1 / (1 + discount_rate)
    ev = fcf * discount_factor + terminal_value * discount_factor
    equity_value = ev - net_debt
    value_per_share = equity_value / shares

    print(f"  independently computed value/share: {value_per_share:.6f}")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.valuation import DCFAssumptions, run_dcf

    pipeline_result = run_dcf(
        DCFAssumptions(
            base_revenue=revenue0, revenue_growth=[growth], ebitda_margin=[margin], tax_rate=tax_rate,
            capex_pct_revenue=capex_pct, da_pct_revenue=da_pct, nwc_pct_revenue_change=0.0,
            discount_rate=discount_rate, terminal_growth=terminal_growth, net_debt=net_debt,
            shares_outstanding=shares, projection_years=1,
        )
    )
    print(f"  src/valuation.py value/share:        {pipeline_result.value_per_share:.6f}")

    if abs(value_per_share - pipeline_result.value_per_share) > 1e-6:
        fail("independent DCF recomputation does not match src/valuation.py")
    print("  MATCH.")


def independently_recompute_reit_nav() -> None:
    step("Independently recomputing a REIT NAV/share (not via src/valuation.py)")
    conn = sqlite3.connect(DB_PATH)
    try:
        reit_annual = pd.read_sql_query("SELECT * FROM reit_financials_annual", conn)
    finally:
        conn.close()
    row = reit_annual.sort_values("fiscal_year").groupby("ticker").tail(1).iloc[0]

    nav_per_share = (row["property_value_mm"] - row["net_debt_mm"]) / row["shares_outstanding_mm"]
    print(f"  ticker={row['ticker']}  independently computed NAV/share: {nav_per_share:.6f}")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.valuation import reit_nav

    pipeline_result = reit_nav(row["property_value_mm"], row["net_debt_mm"], row["shares_outstanding_mm"])
    print(f"  src/valuation.py NAV/share:        {pipeline_result.nav_per_share:.6f}")

    if abs(nav_per_share - pipeline_result.nav_per_share) > 1e-6:
        fail("independent REIT NAV recomputation does not match src/valuation.py")
    print("  MATCH.")


def independently_recompute_portfolio_sharpe() -> None:
    """Recompute the Model Portfolio's Sharpe ratio from raw monthly returns,
    without calling src/portfolio.py, and compare against the Excel pack."""
    step("Independently recomputing the Model Portfolio Sharpe ratio (not via src/portfolio.py)")

    wb = load_workbook(OUTPUT_DIR / "investment_research_pack.xlsx", data_only=True)
    ws = wb["Portfolio Analytics"]
    weights = {}
    header_row = None
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        values = [c.value for c in row]
        if values[:2] == ["ticker", "weight"]:
            header_row = row[0].row
            break
    if header_row is None:
        fail("could not locate the Model Portfolio weights table in the Excel pack")
    r = header_row + 1
    while True:
        ticker = ws.cell(row=r, column=1).value
        weight = ws.cell(row=r, column=2).value
        if ticker is None or not isinstance(weight, (int, float)):
            break
        weights[ticker] = weight
        r += 1

    conn = sqlite3.connect(DB_PATH)
    try:
        prices = pd.read_sql_query("SELECT * FROM market_prices", conn)
    finally:
        conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    sub = prices[prices["ticker"].isin(weights)].pivot(index="date", columns="ticker", values="total_return").sort_index()
    sub = sub[list(weights.keys())].fillna(0.0)

    w = np.array([weights[t] for t in sub.columns])
    port_returns = sub.values @ w

    annualized_return = (1 + port_returns).prod() ** (12 / len(port_returns)) - 1
    annualized_vol = port_returns.std(ddof=1) * np.sqrt(12)
    risk_free = 0.02
    sharpe = (annualized_return - risk_free) / annualized_vol
    print(f"  independently computed Sharpe ratio: {sharpe:.6f}")

    metric_row = None
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        if row[0].value == "sharpe_ratio":
            metric_row = row
            break
    if metric_row is None:
        fail("could not find sharpe_ratio row in the Excel pack")
    pipeline_sharpe = metric_row[1].value
    print(f"  Excel pack Sharpe ratio:             {pipeline_sharpe:.6f}")

    if abs(sharpe - pipeline_sharpe) > 1e-6:
        fail("independent Sharpe ratio recomputation does not match the Excel pack")
    print("  MATCH.")


def main() -> None:
    run_pipeline()
    run_tests()
    check_required_files()
    independently_recompute_dcf()
    independently_recompute_reit_nav()
    independently_recompute_portfolio_sharpe()
    print("\nVERIFY PASSED: Investment Research & Portfolio Analytics Workbench")


if __name__ == "__main__":
    main()
