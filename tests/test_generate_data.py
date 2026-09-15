import pandas as pd

from src.generate_data import generate_all, SECURITIES


def test_seed_reproducibility():
    """The fixed seed must produce byte-identical DataFrames across repeated runs."""
    r1 = generate_all()
    r2 = generate_all()
    for key, obj in r1.items():
        if isinstance(obj, pd.DataFrame):
            assert obj.equals(r2[key]), f"{key} is not reproducible across runs"


def test_security_counts():
    result = generate_all()
    master = result["security_master"]
    assert len(master) == 20
    assert (master["security_type"] == "Company").sum() == 12
    assert (master["security_type"] == "REIT").sum() == 5
    assert master["security_type"].isin(["Financial", "Infrastructure"]).sum() == 3
    assert len(SECURITIES) == 20


def test_at_least_24_months_of_data():
    result = generate_all()
    prices = result["market_prices"]
    n_months = prices.groupby("ticker")["date"].nunique().min()
    assert n_months >= 24

    annual = result["company_financials_annual"]
    n_years = annual.groupby("ticker")["fiscal_year"].nunique().min()
    assert n_years >= 2

    quarterly = result["company_financials_quarterly"]
    n_quarters = quarterly[quarterly["ticker"] != quarterly["ticker"].iloc[0]].groupby("ticker").size().min()
    assert n_quarters >= 8


def test_planted_issues_are_present_in_raw_data():
    result = generate_all()
    manifest = result["planted_issues"]
    quarterly = result["company_financials_quarterly"]

    missing = manifest["missing_value"]
    row = quarterly[
        (quarterly["ticker"] == missing["ticker"])
        & (quarterly["fiscal_year"] == missing["fiscal_year"])
        & (quarterly["fiscal_quarter"] == missing["fiscal_quarter"])
    ]
    assert row[missing["field"]].isna().all()

    dup = manifest["duplicate_observation"]
    dup_rows = quarterly[
        (quarterly["ticker"] == dup["ticker"])
        & (quarterly["fiscal_year"] == dup["fiscal_year"])
        & (quarterly["fiscal_quarter"] == dup["fiscal_quarter"])
    ]
    assert len(dup_rows) == 2

    neg = manifest["negative_anomaly"]
    neg_row = quarterly[
        (quarterly["ticker"] == neg["ticker"])
        & (quarterly["fiscal_year"] == neg["fiscal_year"])
        & (quarterly["fiscal_quarter"] == neg["fiscal_quarter"])
    ]
    assert (neg_row[neg["field"]] < 0).all()


def test_reit_specific_fields_present():
    result = generate_all()
    reit_annual = result["reit_financials_annual"]
    for col in ["property_value_mm", "occupancy_pct", "noi_mm", "ffo_mm", "affo_mm", "net_debt_mm", "cap_rate"]:
        assert col in reit_annual.columns
    assert reit_annual["ticker"].nunique() == 5


def test_annual_and_monthly_prices_reconcile_at_year_end():
    result = generate_all()
    annual = result["company_financials_annual"]
    prices = result["market_prices"].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    for year in annual["fiscal_year"].unique():
        dec_prices = prices[prices["date"] == pd.Timestamp(f"{year}-12-31")].set_index("ticker")["price"]
        annual_prices = annual[annual["fiscal_year"] == year].set_index("ticker")["share_price"]
        diff = (dec_prices - annual_prices).abs()
        assert diff.max() < 1e-6
