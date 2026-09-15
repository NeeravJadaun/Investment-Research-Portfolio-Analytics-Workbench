"""Synthetic data generator for the Investment Research & Portfolio Analytics Workbench.

Everything produced here is fabricated for demonstration purposes using a fixed
random seed so repeated runs are byte-for-byte reproducible. No real company,
security, price, or event data is used anywhere in this project.

Deliberately planted data-quality issues (see PLANTED_ISSUES at the bottom of
`generate_all`) exist so the ingestion pipeline in `ingest.py` has something
real to catch. Nothing in `ingest.py` reads the planted-issue manifest this
module writes -- that file exists only so tests can check the pipeline found
the issues on its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ANNUAL_YEARS = [2022, 2023, 2024]
QUARTERS = [(y, q) for y in ANNUAL_YEARS for q in (1, 2, 3, 4)]
PRICE_MONTHS = pd.date_range("2022-01-31", "2024-12-31", freq="ME")

BENCHMARK_TICKER = "SYN-COMP"

# --------------------------------------------------------------------------
# Security master: 12 general companies, 5 REITs, 3 financial/infrastructure
# --------------------------------------------------------------------------
SECURITIES = [
    # ticker, name, sector, security_type, country, beta
    ("NRVX", "Nervalux Technologies Inc.", "Technology", "Company", "United States", 1.35),
    ("BRKF", "Brookfen Industrial Corp.", "Industrials", "Company", "United States", 1.10),
    ("QNTA", "Quintara Consumer Group", "Consumer Discretionary", "Company", "Canada", 1.20),
    ("STPL", "Staple & Co Foods Ltd.", "Consumer Staples", "Company", "United States", 0.65),
    ("MRDH", "Meridian Health Sciences", "Healthcare", "Company", "United States", 0.85),
    ("VOLX", "Voltarion Energy PLC", "Energy", "Company", "United Kingdom", 1.25),
    ("CPRM", "Coppermine Materials Ltd.", "Materials", "Company", "Australia", 1.15),
    ("SGNL", "Signalworks Communications", "Communication Services", "Company", "United States", 1.05),
    ("AQUU", "AquaUtility Corp.", "Utilities", "Company", "Germany", 0.55),
    ("FLXW", "Flexware Systems Inc.", "Technology", "Company", "United States", 1.45),
    ("HRZI", "Horizon Industrial Group", "Industrials", "Company", "Canada", 1.00),
    ("NMBS", "Nimbus Consumer Brands", "Consumer Discretionary", "Company", "United States", 1.10),
    ("RETP", "Retailpoint REIT", "Retail REIT", "REIT", "United States", 0.95),
    ("OFCT", "Office Realty Trust", "Office REIT", "REIT", "United States", 1.05),
    ("INDR", "Industrial Depot REIT", "Industrial REIT", "REIT", "United States", 0.90),
    ("RESH", "Residentia Homes REIT", "Residential REIT", "REIT", "United States", 0.80),
    ("HLTP", "HealthTrust Properties REIT", "Healthcare REIT", "REIT", "United States", 0.75),
    ("CVBK", "Civic Bancorp", "Financials", "Financial", "United States", 1.15),
    ("ASSR", "Assuretto Insurance Group", "Financials", "Financial", "United Kingdom", 0.90),
    ("GRDN", "Gridline Infrastructure Partners", "Infrastructure", "Infrastructure", "United States", 0.70),
]

TICKERS = [s[0] for s in SECURITIES]
REIT_TICKERS = [s[0] for s in SECURITIES if s[3] == "REIT"]

SECTOR_PROFILE = {
    # sector: (base_revenue_mm, revenue_growth_mean, ebitda_margin, capex_pct_rev, div_payout)
    "Technology": (450, 0.14, 0.28, 0.05, 0.05),
    "Industrials": (900, 0.06, 0.18, 0.06, 0.30),
    "Consumer Discretionary": (600, 0.08, 0.15, 0.04, 0.25),
    "Consumer Staples": (1200, 0.04, 0.20, 0.04, 0.45),
    "Healthcare": (700, 0.09, 0.24, 0.05, 0.20),
    "Energy": (1500, 0.05, 0.32, 0.10, 0.35),
    "Materials": (800, 0.05, 0.22, 0.08, 0.30),
    "Communication Services": (950, 0.06, 0.30, 0.07, 0.35),
    "Utilities": (1100, 0.03, 0.35, 0.12, 0.55),
    "Financials": (500, 0.06, 0.45, 0.02, 0.35),
    "Infrastructure": (650, 0.05, 0.40, 0.15, 0.50),
}

REIT_PROFILE = {
    # sector: (base_property_value_mm, cap_rate, occupancy_mean)
    "Retail REIT": (2200, 0.065, 0.93),
    "Office REIT": (2600, 0.070, 0.89),
    "Industrial REIT": (1900, 0.055, 0.96),
    "Residential REIT": (2400, 0.050, 0.95),
    "Healthcare REIT": (1700, 0.060, 0.91),
}


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Security master
# --------------------------------------------------------------------------
def build_security_master(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for ticker, name, sector, sec_type, country, beta in SECURITIES:
        shares = round(rng.uniform(40, 260), 2)  # millions
        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "security_type": sec_type,
                "country": country,
                "beta": beta,
                "shares_outstanding_mm": shares,
                "is_synthetic": True,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Company / REIT financials (annual + quarterly)
# --------------------------------------------------------------------------
def build_company_financials(rng: np.random.Generator, master: pd.DataFrame):
    annual_rows = []
    quarterly_rows = []

    for _, sec in master.iterrows():
        ticker = sec["ticker"]
        sector = sec["sector"]
        sec_type = sec["security_type"]
        shares = sec["shares_outstanding_mm"]

        if sec_type == "REIT":
            base_rev, growth_mean, ebitda_margin, capex_pct, payout = (
                180, 0.05, 0.60, 0.03, 0.80,
            )
        else:
            base_rev, growth_mean, ebitda_margin, capex_pct, payout = SECTOR_PROFILE[sector]

        growth_path = [max(growth_mean + rng.normal(0, 0.03), -0.05) for _ in ANNUAL_YEARS]
        revenue = base_rev
        book_equity = base_rev * rng.uniform(0.9, 1.6)
        debt = base_rev * rng.uniform(0.4, 1.3)
        cash = base_rev * rng.uniform(0.15, 0.45)
        # per-ticker market pricing multiple: price = book value per share * this
        # multiple (with per-year noise). Ties the synthetic share price to the
        # company's own fundamentals so DCF/comps fair values land in a plausible
        # range around the market price instead of an unrelated random level.
        pb_multiple = rng.uniform(1.1, 2.8)

        for yi, year in enumerate(ANNUAL_YEARS):
            revenue = revenue * (1 + growth_path[yi]) if yi > 0 else revenue
            margin_noise = rng.normal(0, 0.015)
            ebitda = revenue * max(ebitda_margin + margin_noise, 0.03)
            da = revenue * rng.uniform(0.03, 0.07)
            ebit = ebitda - da
            interest_exp = debt * rng.uniform(0.03, 0.06)
            pretax_income = ebit - interest_exp
            tax = pretax_income * 0.24
            net_income = pretax_income - tax
            capex = revenue * capex_pct * rng.uniform(0.85, 1.15)
            ocf = ebitda - interest_exp - tax * 0.9 + rng.normal(0, revenue * 0.01)
            dividends_paid = max(net_income, 0) * payout * rng.uniform(0.9, 1.1)

            debt = max(debt * (1 + rng.normal(0, 0.05)), revenue * 0.1)
            cash = max(cash + ocf * 0.1 - capex * 0.05 + rng.normal(0, revenue * 0.02), revenue * 0.05)
            book_equity = book_equity + net_income - dividends_paid + rng.normal(0, revenue * 0.01)
            total_assets = book_equity + debt + cash * 0.2 + revenue * 0.35
            total_liabilities = total_assets - book_equity

            bvps = book_equity / shares if shares else 0.0
            share_price = round(max(bvps * pb_multiple * (1 + rng.normal(0, 0.12)), 1.0), 2)

            annual_rows.append(
                {
                    "ticker": ticker,
                    "period_type": "annual",
                    "fiscal_year": year,
                    "fiscal_quarter": np.nan,
                    "revenue_mm": round(revenue, 2),
                    "ebitda_mm": round(ebitda, 2),
                    "ebit_mm": round(ebit, 2),
                    "net_income_mm": round(net_income, 2),
                    "operating_cash_flow_mm": round(ocf, 2),
                    "capex_mm": round(capex, 2),
                    "total_assets_mm": round(total_assets, 2),
                    "total_liabilities_mm": round(total_liabilities, 2),
                    "book_equity_mm": round(book_equity, 2),
                    "cash_mm": round(cash, 2),
                    "debt_mm": round(debt, 2),
                    "dividends_paid_mm": round(dividends_paid, 2),
                    "shares_outstanding_mm": shares,
                    "share_price": share_price,
                }
            )

            # split the annual figures into 4 quarters with seasonal noise
            q_weights = rng.dirichlet(np.array([1.0, 1.0, 1.0, 1.0]) * 6)
            for qi, quarter in enumerate((1, 2, 3, 4)):
                q_rev = revenue * q_weights[qi]
                q_ebitda = ebitda * q_weights[qi] * rng.uniform(0.95, 1.05)
                q_ebit = ebit * q_weights[qi] * rng.uniform(0.95, 1.05)
                q_ni = net_income * q_weights[qi] * rng.uniform(0.9, 1.1)
                q_ocf = ocf * q_weights[qi] * rng.uniform(0.9, 1.1)
                q_capex = capex * q_weights[qi] * rng.uniform(0.9, 1.1)
                quarterly_rows.append(
                    {
                        "ticker": ticker,
                        "period_type": "quarterly",
                        "fiscal_year": year,
                        "fiscal_quarter": quarter,
                        "revenue_mm": round(q_rev, 2),
                        "ebitda_mm": round(q_ebitda, 2),
                        "ebit_mm": round(q_ebit, 2),
                        "net_income_mm": round(q_ni, 2),
                        "operating_cash_flow_mm": round(q_ocf, 2),
                        "capex_mm": round(q_capex, 2),
                        "total_assets_mm": round(total_assets, 2),
                        "total_liabilities_mm": round(total_liabilities, 2),
                        "book_equity_mm": round(book_equity, 2),
                        "cash_mm": round(cash, 2),
                        "debt_mm": round(debt, 2),
                        "dividends_paid_mm": round(dividends_paid / 4, 2),
                        "shares_outstanding_mm": shares,
                        "share_price": share_price,
                    }
                )

    return pd.DataFrame(annual_rows), pd.DataFrame(quarterly_rows)


# --------------------------------------------------------------------------
# REIT-specific financials
# --------------------------------------------------------------------------
def build_reit_financials(rng: np.random.Generator, master: pd.DataFrame):
    annual_rows = []
    quarterly_rows = []
    reits = master[master["security_type"] == "REIT"]

    for _, sec in reits.iterrows():
        ticker = sec["ticker"]
        sector = sec["sector"]
        shares = sec["shares_outstanding_mm"]
        base_pv, cap_rate_mean, occ_mean = REIT_PROFILE[sector]

        property_value = base_pv
        net_debt = property_value * rng.uniform(0.35, 0.55)

        for yi, year in enumerate(ANNUAL_YEARS):
            growth = rng.uniform(0.02, 0.06)
            property_value = property_value * (1 + growth) if yi > 0 else property_value
            occupancy = min(max(occ_mean + rng.normal(0, 0.02), 0.70), 0.99)
            cap_rate = max(cap_rate_mean + rng.normal(0, 0.003), 0.03)
            noi = property_value * cap_rate * (occupancy / occ_mean)
            da = property_value * rng.uniform(0.02, 0.03)
            interest_exp = net_debt * rng.uniform(0.035, 0.055)
            net_income = noi - da - interest_exp
            ffo = net_income + da
            recurring_capex = property_value * rng.uniform(0.006, 0.012)
            affo = ffo - recurring_capex
            net_debt = max(net_debt * (1 + rng.normal(0, 0.04)), property_value * 0.15)

            annual_rows.append(
                {
                    "ticker": ticker,
                    "period_type": "annual",
                    "fiscal_year": year,
                    "fiscal_quarter": np.nan,
                    "property_value_mm": round(property_value, 2),
                    "occupancy_pct": round(occupancy, 4),
                    "noi_mm": round(noi, 2),
                    "ffo_mm": round(ffo, 2),
                    "affo_mm": round(affo, 2),
                    "net_debt_mm": round(net_debt, 2),
                    "shares_outstanding_mm": shares,
                    "cap_rate": round(cap_rate, 4),
                }
            )

            q_weights = rng.dirichlet(np.array([1.0, 1.0, 1.0, 1.0]) * 8)
            for qi, quarter in enumerate((1, 2, 3, 4)):
                quarterly_rows.append(
                    {
                        "ticker": ticker,
                        "period_type": "quarterly",
                        "fiscal_year": year,
                        "fiscal_quarter": quarter,
                        "property_value_mm": round(property_value, 2),
                        "occupancy_pct": round(min(max(occupancy + rng.normal(0, 0.01), 0.70), 0.99), 4),
                        "noi_mm": round(noi * q_weights[qi] * 4 * 0.25 * (1 + rng.normal(0, 0.03)), 2),
                        "ffo_mm": round(ffo * q_weights[qi] * 4 * 0.25 * (1 + rng.normal(0, 0.03)), 2),
                        "affo_mm": round(affo * q_weights[qi] * 4 * 0.25 * (1 + rng.normal(0, 0.03)), 2),
                        "net_debt_mm": round(net_debt, 2),
                        "shares_outstanding_mm": shares,
                        "cap_rate": round(max(cap_rate_mean + rng.normal(0, 0.003), 0.03), 4),
                    }
                )

    return pd.DataFrame(annual_rows), pd.DataFrame(quarterly_rows)


# --------------------------------------------------------------------------
# Peer / multiples table
# --------------------------------------------------------------------------
def build_peer_multiples(
    master: pd.DataFrame, company_annual: pd.DataFrame, reit_annual: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for _, sec in master.iterrows():
        ticker = sec["ticker"]
        sector = sec["sector"]
        sec_type = sec["security_type"]
        cfa = company_annual[company_annual["ticker"] == ticker]
        for _, r in cfa.iterrows():
            year = r["fiscal_year"]
            price = r["share_price"]
            shares = r["shares_outstanding_mm"]
            market_cap = price * shares
            ev = market_cap + r["debt_mm"] - r["cash_mm"]
            eps = r["net_income_mm"] / shares if shares else np.nan
            bvps = r["book_equity_mm"] / shares if shares else np.nan
            ffo_mm = np.nan
            nav_mm = np.nan
            p_ffo = np.nan
            p_nav = np.nan
            if sec_type == "REIT":
                rrow = reit_annual[(reit_annual["ticker"] == ticker) & (reit_annual["fiscal_year"] == year)]
                if not rrow.empty:
                    rrow = rrow.iloc[0]
                    ffo_mm = rrow["ffo_mm"]
                    nav_mm = rrow["property_value_mm"] - rrow["net_debt_mm"]
                    ffo_ps = ffo_mm / shares if shares else np.nan
                    nav_ps = nav_mm / shares if shares else np.nan
                    p_ffo = price / ffo_ps if ffo_ps else np.nan
                    p_nav = price / nav_ps if nav_ps else np.nan

            rows.append(
                {
                    "ticker": ticker,
                    "fiscal_year": year,
                    "sector": sector,
                    "security_type": sec_type,
                    "enterprise_value_mm": round(ev, 2),
                    "ebitda_mm": round(r["ebitda_mm"], 2),
                    "ffo_mm": round(ffo_mm, 2) if pd.notna(ffo_mm) else np.nan,
                    "nav_mm": round(nav_mm, 2) if pd.notna(nav_mm) else np.nan,
                    "ev_ebitda_multiple": round(ev / r["ebitda_mm"], 2) if r["ebitda_mm"] else np.nan,
                    "pe_multiple": round(price / eps, 2) if eps and eps > 0 else np.nan,
                    "pb_multiple": round(price / bvps, 2) if bvps and bvps > 0 else np.nan,
                    "p_ffo_multiple": round(p_ffo, 2) if pd.notna(p_ffo) else np.nan,
                    "p_nav_multiple": round(p_nav, 2) if pd.notna(p_nav) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Monthly market prices + benchmark
# --------------------------------------------------------------------------
def build_market_prices(rng: np.random.Generator, master: pd.DataFrame, company_annual: pd.DataFrame) -> pd.DataFrame:
    """Monthly price path per ticker, beta-correlated to a single synthetic
    benchmark (SYN-COMP), constructed as a Brownian bridge across each fiscal
    year so month-end December prices land exactly on that year's
    `company_financials_annual.share_price` -- i.e. the monthly series and the
    annual fundamentals-derived price are the same number, not two independent
    random draws.
    """
    n_months = len(PRICE_MONTHS)
    benchmark_returns = rng.normal(0.006, 0.035, n_months)
    months_per_segment = 12

    rows = []
    for _, sec in master.iterrows():
        ticker = sec["ticker"]
        beta = sec["beta"]
        sector = sec["sector"]
        base_div_yield = 0.045 if sec["security_type"] == "REIT" else SECTOR_PROFILE.get(sector, (0, 0, 0, 0, 0.2))[4] * 0.06

        ticker_annual = company_annual[company_annual["ticker"] == ticker].sort_values("fiscal_year")
        year_end_price = {int(r["fiscal_year"]): float(r["share_price"]) for _, r in ticker_annual.iterrows()}
        pre_start_price = year_end_price[ANNUAL_YEARS[0]] * (1 + rng.normal(0, 0.06))
        segment_anchors = [pre_start_price] + [year_end_price[y] for y in ANNUAL_YEARS]

        monthly_returns_all = []
        price_path = []
        for yi, year in enumerate(ANNUAL_YEARS):
            start_price = segment_anchors[yi]
            end_price = segment_anchors[yi + 1]
            seg_start = yi * months_per_segment
            seg_benchmark = benchmark_returns[seg_start: seg_start + months_per_segment]
            idio = rng.normal(0.0, 0.03, months_per_segment)
            raw_simple_returns = np.clip(beta * seg_benchmark + idio, -0.6, 5.0)
            raw_log_returns = np.log1p(raw_simple_returns)

            target_log_return = np.log(end_price) - np.log(start_price)
            correction = (target_log_return - raw_log_returns.sum()) / months_per_segment
            adjusted_log_returns = raw_log_returns + correction

            price = start_price
            for lr in adjusted_log_returns:
                price = price * np.exp(lr)
                price_path.append(price)
                monthly_returns_all.append(np.exp(lr) - 1)
            # force exact reconciliation to the annual anchor at year-end (guards
            # against floating-point drift from the log/exp round trip)
            price_path[-1] = end_price

        for i, dt in enumerate(PRICE_MONTHS):
            monthly_div_yield = base_div_yield / 12 if (i % 3 == 2) else 0.0  # paid quarterly
            dividend_return = max(monthly_div_yield * (1 + rng.normal(0, 0.1)), 0.0)
            rows.append(
                {
                    "ticker": ticker,
                    "date": dt.strftime("%Y-%m-%d"),
                    "price": round(price_path[i], 2),
                    "monthly_return": round(monthly_returns_all[i], 6),
                    "dividend_return": round(dividend_return, 6),
                    "total_return": round(monthly_returns_all[i] + dividend_return, 6),
                    "benchmark_ticker": BENCHMARK_TICKER,
                    "benchmark_return": round(benchmark_returns[i], 6),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Synthetic company events
# --------------------------------------------------------------------------
EVENT_TYPES = ["Earnings", "Guidance Change", "Acquisition", "Sector Event"]


def build_company_events(rng: np.random.Generator, master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_id = 1

    # quarterly earnings dates, ~40 days after quarter end
    quarter_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    for _, sec in master.iterrows():
        ticker = sec["ticker"]
        for year, q in QUARTERS:
            base = pd.Timestamp(f"{year}-{quarter_end[q]}") + pd.Timedelta(days=int(rng.integers(28, 48)))
            rows.append(
                {
                    "event_id": event_id,
                    "ticker": ticker,
                    "event_date": base.strftime("%Y-%m-%d"),
                    "event_type": "Earnings",
                    "description": f"[SYNTHETIC] {ticker} reports Q{q} FY{year} results.",
                    "is_synthetic": True,
                }
            )
            event_id += 1

    # sparse guidance changes
    guidance_sample = rng.choice(TICKERS, size=8, replace=False)
    for ticker in guidance_sample:
        year = int(rng.choice(ANNUAL_YEARS))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))
        direction = rng.choice(["raised", "lowered"])
        rows.append(
            {
                "event_id": event_id,
                "ticker": ticker,
                "event_date": f"{year}-{month:02d}-{day:02d}",
                "event_type": "Guidance Change",
                "description": f"[SYNTHETIC] {ticker} management {direction} full-year guidance.",
                "is_synthetic": True,
            }
        )
        event_id += 1

    # a handful of fictional acquisitions
    acq_pairs = rng.choice(TICKERS, size=(4, 2), replace=False)
    for acquirer, target in acq_pairs:
        if acquirer == target:
            continue
        year = int(rng.choice(ANNUAL_YEARS))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))
        rows.append(
            {
                "event_id": event_id,
                "ticker": acquirer,
                "event_date": f"{year}-{month:02d}-{day:02d}",
                "event_type": "Acquisition",
                "description": f"[SYNTHETIC] {acquirer} announces fictional acquisition of {target}.",
                "is_synthetic": True,
            }
        )
        event_id += 1

    # sector-wide fictional events
    sectors = sorted(set(s[2] for s in SECURITIES))
    sector_sample = rng.choice(sectors, size=5, replace=False)
    for sector in sector_sample:
        sector_tickers = [s[0] for s in SECURITIES if s[2] == sector]
        rep_ticker = rng.choice(sector_tickers)
        year = int(rng.choice(ANNUAL_YEARS))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))
        rows.append(
            {
                "event_id": event_id,
                "ticker": rep_ticker,
                "event_date": f"{year}-{month:02d}-{day:02d}",
                "event_type": "Sector Event",
                "description": f"[SYNTHETIC] Fictional {sector} sector-wide regulatory/demand shift noted.",
                "is_synthetic": True,
            }
        )
        event_id += 1

    df = pd.DataFrame(rows).sort_values(["event_date", "ticker"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Plant deliberate data-quality issues
# --------------------------------------------------------------------------
def plant_data_quality_issues(rng: np.random.Generator, company_quarterly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = company_quarterly.copy().reset_index(drop=True)
    manifest: dict = {}

    non_reit_tickers = [s[0] for s in SECURITIES if s[3] != "REIT"]

    # 1. one missing quarterly value (revenue)
    t1 = non_reit_tickers[2]
    idx1 = df[(df["ticker"] == t1) & (df["fiscal_year"] == 2023) & (df["fiscal_quarter"] == 2)].index[0]
    df.loc[idx1, "revenue_mm"] = np.nan
    manifest["missing_value"] = {"ticker": t1, "fiscal_year": 2023, "fiscal_quarter": 2, "field": "revenue_mm"}

    # 2. one duplicate observation (exact duplicate row appended)
    t2 = non_reit_tickers[5]
    dup_row = df[(df["ticker"] == t2) & (df["fiscal_year"] == 2023) & (df["fiscal_quarter"] == 4)]
    df = pd.concat([df, dup_row], ignore_index=True)
    manifest["duplicate_observation"] = {"ticker": t2, "fiscal_year": 2023, "fiscal_quarter": 4}

    # 3. one unit mismatch (revenue reported in thousands instead of millions -> 1000x too large)
    t3 = non_reit_tickers[7]
    idx3 = df[(df["ticker"] == t3) & (df["fiscal_year"] == 2024) & (df["fiscal_quarter"] == 1)].index[0]
    df.loc[idx3, "revenue_mm"] = df.loc[idx3, "revenue_mm"] * 1000
    manifest["unit_mismatch"] = {"ticker": t3, "fiscal_year": 2024, "fiscal_quarter": 1, "field": "revenue_mm"}

    # 4. one negative-value anomaly (revenue cannot be negative)
    t4 = non_reit_tickers[9]
    idx4 = df[(df["ticker"] == t4) & (df["fiscal_year"] == 2024) & (df["fiscal_quarter"] == 3)].index[0]
    df.loc[idx4, "revenue_mm"] = -abs(df.loc[idx4, "revenue_mm"])
    manifest["negative_anomaly"] = {"ticker": t4, "fiscal_year": 2024, "fiscal_quarter": 3, "field": "revenue_mm"}

    return df, manifest


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def generate_all(out_dir: Path = RAW_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = _rng()

    master = build_security_master(rng)
    company_annual, company_quarterly = build_company_financials(rng, master)
    reit_annual, reit_quarterly = build_reit_financials(rng, master)
    peer_multiples = build_peer_multiples(master, company_annual, reit_annual)
    market_prices = build_market_prices(rng, master, company_annual)
    company_events = build_company_events(rng, master)

    company_quarterly_dirty, manifest = plant_data_quality_issues(rng, company_quarterly)

    master.to_csv(out_dir / "security_master.csv", index=False)
    company_annual.to_csv(out_dir / "company_financials_annual.csv", index=False)
    company_quarterly_dirty.to_csv(out_dir / "company_financials_quarterly.csv", index=False)
    reit_annual.to_csv(out_dir / "reit_financials_annual.csv", index=False)
    reit_quarterly.to_csv(out_dir / "reit_financials_quarterly.csv", index=False)
    peer_multiples.to_csv(out_dir / "peer_multiples.csv", index=False)
    market_prices.to_csv(out_dir / "market_prices.csv", index=False)
    company_events.to_csv(out_dir / "company_events.csv", index=False)

    with open(out_dir / "_planted_data_quality_issues.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "security_master": master,
        "company_financials_annual": company_annual,
        "company_financials_quarterly": company_quarterly_dirty,
        "reit_financials_annual": reit_annual,
        "reit_financials_quarterly": reit_quarterly,
        "peer_multiples": peer_multiples,
        "market_prices": market_prices,
        "company_events": company_events,
        "planted_issues": manifest,
    }


if __name__ == "__main__":
    result = generate_all()
    for name, obj in result.items():
        if isinstance(obj, pd.DataFrame):
            print(f"{name}: {obj.shape}")
    print("Planted data-quality issues:", json.dumps(result["planted_issues"], indent=2))
