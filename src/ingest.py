"""Ingestion, validation, and data-quality reporting.

Reads the raw synthetic CSVs written by `generate_data.py`, checks required
columns/types, deduplicates records, flags missing and anomalous values with
independent statistical/rule-based checks (this module never reads the
planted-issue manifest -- detection has to stand on its own), and writes the
cleaned tables plus a data-quality report to `data/processed/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from src import db

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# --------------------------------------------------------------------------
# Required schema per raw table: {column: expected pandas dtype kind}
# kind: 'O' object/string, 'f' float, 'i' int, 'b' bool
# --------------------------------------------------------------------------
SCHEMAS: dict[str, dict[str, str]] = {
    "security_master": {
        "ticker": "O", "name": "O", "sector": "O", "security_type": "O",
        "country": "O", "beta": "f", "shares_outstanding_mm": "f", "is_synthetic": "b",
    },
    "company_financials_annual": {
        "ticker": "O", "period_type": "O", "fiscal_year": "i",
        "revenue_mm": "f", "ebitda_mm": "f", "ebit_mm": "f", "net_income_mm": "f",
        "operating_cash_flow_mm": "f", "capex_mm": "f", "total_assets_mm": "f",
        "total_liabilities_mm": "f", "book_equity_mm": "f", "cash_mm": "f",
        "debt_mm": "f", "dividends_paid_mm": "f", "shares_outstanding_mm": "f",
        "share_price": "f",
    },
    "company_financials_quarterly": {
        "ticker": "O", "period_type": "O", "fiscal_year": "i", "fiscal_quarter": "i",
        "revenue_mm": "f", "ebitda_mm": "f", "ebit_mm": "f", "net_income_mm": "f",
        "operating_cash_flow_mm": "f", "capex_mm": "f", "total_assets_mm": "f",
        "total_liabilities_mm": "f", "book_equity_mm": "f", "cash_mm": "f",
        "debt_mm": "f", "dividends_paid_mm": "f", "shares_outstanding_mm": "f",
        "share_price": "f",
    },
    "reit_financials_annual": {
        "ticker": "O", "period_type": "O", "fiscal_year": "i",
        "property_value_mm": "f", "occupancy_pct": "f", "noi_mm": "f", "ffo_mm": "f",
        "affo_mm": "f", "net_debt_mm": "f", "shares_outstanding_mm": "f", "cap_rate": "f",
    },
    "reit_financials_quarterly": {
        "ticker": "O", "period_type": "O", "fiscal_year": "i", "fiscal_quarter": "i",
        "property_value_mm": "f", "occupancy_pct": "f", "noi_mm": "f", "ffo_mm": "f",
        "affo_mm": "f", "net_debt_mm": "f", "shares_outstanding_mm": "f", "cap_rate": "f",
    },
    "peer_multiples": {
        "ticker": "O", "fiscal_year": "i", "sector": "O", "security_type": "O",
        "enterprise_value_mm": "f", "ebitda_mm": "f", "ev_ebitda_multiple": "f",
    },
    "market_prices": {
        "ticker": "O", "date": "O", "price": "f", "monthly_return": "f",
        "dividend_return": "f", "total_return": "f", "benchmark_return": "f",
    },
    "company_events": {
        "event_id": "i", "ticker": "O", "event_date": "O", "event_type": "O",
        "description": "O", "is_synthetic": "b",
    },
}

# fields that must never be negative (business rule)
NON_NEGATIVE_FIELDS: dict[str, list[str]] = {
    "company_financials_annual": ["revenue_mm", "total_assets_mm", "shares_outstanding_mm", "share_price"],
    "company_financials_quarterly": ["revenue_mm", "total_assets_mm", "shares_outstanding_mm", "share_price"],
    "reit_financials_annual": ["property_value_mm", "occupancy_pct", "shares_outstanding_mm"],
    "reit_financials_quarterly": ["property_value_mm", "occupancy_pct", "shares_outstanding_mm"],
}

# primary-key columns used to detect duplicate observations
PRIMARY_KEYS: dict[str, list[str]] = {
    "security_master": ["ticker"],
    "company_financials_annual": ["ticker", "fiscal_year"],
    "company_financials_quarterly": ["ticker", "fiscal_year", "fiscal_quarter"],
    "reit_financials_annual": ["ticker", "fiscal_year"],
    "reit_financials_quarterly": ["ticker", "fiscal_year", "fiscal_quarter"],
    "peer_multiples": ["ticker", "fiscal_year"],
    "market_prices": ["ticker", "date"],
    "company_events": ["event_id"],
}

# numeric columns checked for unit-mismatch style outliers (z-score on log scale)
OUTLIER_FIELDS: dict[str, list[str]] = {
    "company_financials_annual": ["revenue_mm", "ebitda_mm"],
    "company_financials_quarterly": ["revenue_mm", "ebitda_mm"],
}

DATA_DICTIONARY: list[dict[str, str]] = [
    {"table": "security_master", "field": "ticker", "description": "Fictional security ticker symbol, unique identifier."},
    {"table": "security_master", "field": "sector", "description": "GICS-like sector classification, synthetic."},
    {"table": "security_master", "field": "security_type", "description": "Company, REIT, Financial, or Infrastructure."},
    {"table": "security_master", "field": "beta", "description": "Assumed market beta used to simulate correlated monthly returns."},
    {"table": "company_financials_annual/quarterly", "field": "revenue_mm", "description": "Total revenue, USD millions, synthetic."},
    {"table": "company_financials_annual/quarterly", "field": "ebitda_mm", "description": "Earnings before interest, tax, depreciation & amortization, USD millions."},
    {"table": "company_financials_annual/quarterly", "field": "ebit_mm", "description": "EBITDA less depreciation & amortization, USD millions."},
    {"table": "company_financials_annual/quarterly", "field": "net_income_mm", "description": "Net income after modeled interest expense and a flat 24% tax rate."},
    {"table": "company_financials_annual/quarterly", "field": "operating_cash_flow_mm", "description": "Simulated cash from operations, USD millions."},
    {"table": "company_financials_annual/quarterly", "field": "capex_mm", "description": "Capital expenditure, USD millions, sector-scaled % of revenue."},
    {"table": "company_financials_annual/quarterly", "field": "total_assets_mm", "description": "Total assets, USD millions (book_equity + debt + working capital proxy)."},
    {"table": "company_financials_annual/quarterly", "field": "total_liabilities_mm", "description": "total_assets_mm - book_equity_mm, by construction."},
    {"table": "company_financials_annual/quarterly", "field": "book_equity_mm", "description": "Book value of equity, USD millions, rolled forward by retained earnings."},
    {"table": "company_financials_annual/quarterly", "field": "cash_mm", "description": "Cash and equivalents, USD millions."},
    {"table": "company_financials_annual/quarterly", "field": "debt_mm", "description": "Total debt, USD millions."},
    {"table": "company_financials_annual/quarterly", "field": "dividends_paid_mm", "description": "Total dividends paid in the period, USD millions."},
    {"table": "company_financials_annual/quarterly", "field": "share_price", "description": "Period-end share price, synthetic random walk."},
    {"table": "reit_financials_annual/quarterly", "field": "property_value_mm", "description": "Gross investment property value, USD millions."},
    {"table": "reit_financials_annual/quarterly", "field": "occupancy_pct", "description": "Portfolio occupancy rate, 0-1."},
    {"table": "reit_financials_annual/quarterly", "field": "noi_mm", "description": "Net operating income, USD millions."},
    {"table": "reit_financials_annual/quarterly", "field": "ffo_mm", "description": "Funds From Operations = net income + depreciation & amortization."},
    {"table": "reit_financials_annual/quarterly", "field": "affo_mm", "description": "Adjusted FFO = FFO - recurring/maintenance capex."},
    {"table": "reit_financials_annual/quarterly", "field": "net_debt_mm", "description": "Total debt less cash, USD millions."},
    {"table": "reit_financials_annual/quarterly", "field": "cap_rate", "description": "NOI / property_value_mm, implied capitalization rate."},
    {"table": "peer_multiples", "field": "enterprise_value_mm", "description": "Market cap + debt - cash, USD millions."},
    {"table": "peer_multiples", "field": "ev_ebitda_multiple", "description": "enterprise_value_mm / ebitda_mm."},
    {"table": "peer_multiples", "field": "pe_multiple", "description": "share_price / (net_income_mm / shares_outstanding_mm)."},
    {"table": "peer_multiples", "field": "pb_multiple", "description": "share_price / (book_equity_mm / shares_outstanding_mm)."},
    {"table": "peer_multiples", "field": "p_ffo_multiple", "description": "REIT-only: share_price / (FFO / shares_outstanding_mm)."},
    {"table": "peer_multiples", "field": "p_nav_multiple", "description": "REIT-only: share_price / ((property_value - net_debt) / shares_outstanding_mm)."},
    {"table": "market_prices", "field": "monthly_return", "description": "Simulated monthly price return, driven by beta * benchmark_return + idiosyncratic noise."},
    {"table": "market_prices", "field": "dividend_return", "description": "Simulated monthly dividend yield contribution (paid quarterly)."},
    {"table": "market_prices", "field": "total_return", "description": "monthly_return + dividend_return."},
    {"table": "market_prices", "field": "benchmark_return", "description": "Synthetic composite benchmark (SYN-COMP) monthly return."},
    {"table": "company_events", "field": "event_type", "description": "Earnings, Guidance Change, Acquisition, or Sector Event -- all fictional."},
    {"table": "data_quality", "field": "issue_type", "description": "missing_value, duplicate_observation, unit_mismatch, or negative_value_anomaly."},
]


class DQIssue(BaseModel):
    model_config = ConfigDict(frozen=True)
    table: str
    issue_type: str
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[float] = None
    field: Optional[str] = None
    detail: str


def _read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name}.csv")


def validate_schema(name: str, df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable schema problems (missing columns or bad dtype kind)."""
    problems = []
    schema = SCHEMAS.get(name, {})
    for col, kind in schema.items():
        if col not in df.columns:
            problems.append(f"{name}: missing required column '{col}'")
            continue
        actual_kind = df[col].dropna().dtype.kind
        if kind == "f" and actual_kind not in ("f", "i"):
            problems.append(f"{name}.{col}: expected numeric, found dtype kind '{actual_kind}'")
        elif kind == "i" and actual_kind not in ("i", "f"):
            problems.append(f"{name}.{col}: expected integer-like, found dtype kind '{actual_kind}'")
        elif kind == "O" and actual_kind not in ("O",):
            problems.append(f"{name}.{col}: expected string, found dtype kind '{actual_kind}'")
        elif kind == "b" and actual_kind not in ("b",):
            problems.append(f"{name}.{col}: expected boolean, found dtype kind '{actual_kind}'")
    return problems


def find_duplicates(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[DQIssue]]:
    keys = PRIMARY_KEYS.get(name)
    if not keys:
        return df, []
    issues = []
    dup_mask = df.duplicated(subset=keys, keep="first")
    for _, row in df[dup_mask].iterrows():
        issues.append(
            DQIssue(
                table=name,
                issue_type="duplicate_observation",
                ticker=row.get("ticker"),
                fiscal_year=row.get("fiscal_year"),
                fiscal_quarter=row.get("fiscal_quarter") if "fiscal_quarter" in row else None,
                detail=f"Duplicate primary key {dict((k, row[k]) for k in keys)}; kept first occurrence.",
            )
        )
    deduped = df[~dup_mask].reset_index(drop=True)
    return deduped, issues


def find_missing_values(name: str, df: pd.DataFrame) -> list[DQIssue]:
    issues = []
    schema = SCHEMAS.get(name, {})
    numeric_cols = [c for c, k in schema.items() if k in ("f", "i") and c in df.columns]
    for col in numeric_cols:
        missing = df[df[col].isna()]
        for _, row in missing.iterrows():
            issues.append(
                DQIssue(
                    table=name,
                    issue_type="missing_value",
                    ticker=row.get("ticker"),
                    fiscal_year=row.get("fiscal_year") if "fiscal_year" in row else None,
                    fiscal_quarter=row.get("fiscal_quarter") if "fiscal_quarter" in row else None,
                    field=col,
                    detail=f"Null value in required field '{col}'.",
                )
            )
    return issues


def find_negative_anomalies(name: str, df: pd.DataFrame) -> list[DQIssue]:
    issues = []
    for col in NON_NEGATIVE_FIELDS.get(name, []):
        if col not in df.columns:
            continue
        bad = df[df[col] < 0]
        for _, row in bad.iterrows():
            issues.append(
                DQIssue(
                    table=name,
                    issue_type="negative_value_anomaly",
                    ticker=row.get("ticker"),
                    fiscal_year=row.get("fiscal_year") if "fiscal_year" in row else None,
                    fiscal_quarter=row.get("fiscal_quarter") if "fiscal_quarter" in row else None,
                    field=col,
                    detail=f"Field '{col}' cannot be negative; found {row[col]}.",
                )
            )
    return issues


def find_unit_mismatches(name: str, df: pd.DataFrame, z_threshold: float = 3.0) -> list[DQIssue]:
    """Flag values that are extreme outliers vs. the same ticker's own history on a
    log scale -- the signature of a unit mismatch (e.g. thousands vs. millions)."""
    issues = []
    for col in OUTLIER_FIELDS.get(name, []):
        if col not in df.columns or "ticker" not in df.columns:
            continue
        for ticker, grp in df.groupby("ticker"):
            vals = grp[col].dropna()
            positive = vals[vals > 0]
            if len(positive) < 4:
                continue
            log_vals = np.log(positive)
            mu, sigma = log_vals.mean(), log_vals.std(ddof=0)
            if sigma == 0 or np.isnan(sigma):
                continue
            z = (log_vals - mu) / sigma
            outliers = z[z.abs() > z_threshold]
            for idx in outliers.index:
                row = df.loc[idx]
                issues.append(
                    DQIssue(
                        table=name,
                        issue_type="unit_mismatch",
                        ticker=ticker,
                        fiscal_year=row.get("fiscal_year") if "fiscal_year" in row else None,
                        fiscal_quarter=row.get("fiscal_quarter") if "fiscal_quarter" in row else None,
                        field=col,
                        detail=(
                            f"Value {row[col]:,.2f} is a {z[idx]:.1f}-sigma outlier vs. {ticker}'s own "
                            f"log-scale history in '{col}' -- likely a unit mismatch (e.g. thousands vs. millions)."
                        ),
                    )
                )
    return issues


def run_ingestion(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, db_path: Path | None = None) -> dict:
    processed_dir.mkdir(parents=True, exist_ok=True)
    if db_path is None:
        db_path = processed_dir / "research.db"

    tables = {name: pd.read_csv(raw_dir / f"{name}.csv") for name in SCHEMAS}

    all_issues: list[DQIssue] = []
    schema_problems: list[str] = []
    cleaned: dict[str, pd.DataFrame] = {}

    for name, df in tables.items():
        schema_problems.extend(validate_schema(name, df))

        deduped, dup_issues = find_duplicates(name, df)
        all_issues.extend(dup_issues)

        all_issues.extend(find_missing_values(name, deduped))
        all_issues.extend(find_negative_anomalies(name, deduped))
        all_issues.extend(find_unit_mismatches(name, deduped))

        cleaned[name] = deduped
        deduped.to_csv(processed_dir / f"{name}.csv", index=False)

    dq_report = pd.DataFrame([i.model_dump() for i in all_issues])
    if dq_report.empty:
        dq_report = pd.DataFrame(
            columns=["table", "issue_type", "ticker", "fiscal_year", "fiscal_quarter", "field", "detail"]
        )
    dq_report.to_csv(processed_dir / "data_quality.csv", index=False)

    data_dictionary_df = pd.DataFrame(DATA_DICTIONARY)
    data_dictionary_df.to_csv(processed_dir / "data_dictionary.csv", index=False)

    if schema_problems:
        with open(processed_dir / "_schema_problems.txt", "w") as f:
            f.write("\n".join(schema_problems))

    db_tables = {**cleaned, "data_quality": dq_report, "data_dictionary": data_dictionary_df}
    db.write_tables(db_tables, PRIMARY_KEYS, db_path=db_path)

    return {
        "tables": cleaned,
        "issues": all_issues,
        "dq_report": dq_report,
        "schema_problems": schema_problems,
    }


if __name__ == "__main__":
    result = run_ingestion()
    print(f"Ingested {len(result['tables'])} tables.")
    print(f"Data-quality issues found: {len(result['issues'])}")
    for issue in result["issues"]:
        print(f"  [{issue.issue_type}] {issue.table} ticker={issue.ticker} field={issue.field}: {issue.detail}")
    if result["schema_problems"]:
        print("Schema problems:")
        for p in result["schema_problems"]:
            print(f"  {p}")
