from src.analytics_pipeline import build_factor_screen, build_dcf_assumptions_from_financials
from src.scenarios import BULL, BASE, BEAR, run_dcf_scenario_suite, run_reit_scenario_suite


def test_factor_screen_ranking_changes_with_weights(processed):
    default_screen = build_factor_screen(processed)
    tilted_screen = build_factor_screen(processed, weights={"valuation_discount": 0.0, "growth": 0.0, "leverage": 0.0, "profitability": 0.0, "momentum": 1.0, "dividend_yield": 0.0})

    default_order = default_screen.sort_values("ticker")["rank"].tolist()
    tilted_order = tilted_screen.sort_values("ticker")["rank"].tolist()
    assert default_order != tilted_order, "changing factor weights should change the ranking"


def test_factor_screen_composite_score_ranks_descending(processed):
    screen = build_factor_screen(processed)
    scores = screen["composite_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert screen["rank"].tolist() == list(range(1, len(screen) + 1))


def test_dcf_scenario_suite_ordering_bear_base_bull(processed):
    master = processed["security_master"]
    annual = processed["company_financials_annual"]
    ticker = master[master["security_type"] != "REIT"]["ticker"].iloc[0]
    base_assumptions = build_dcf_assumptions_from_financials(ticker, master, annual)

    suite = run_dcf_scenario_suite(base_assumptions, [BEAR, BASE, BULL])
    values = suite.set_index("scenario")["value_per_share"]
    assert values["Bear"] < values["Base"] < values["Bull"]


def test_reit_scenario_suite_ordering_bear_base_bull(processed):
    reit_annual = processed["reit_financials_annual"]
    row = reit_annual.sort_values("fiscal_year").groupby("ticker").tail(1).iloc[0]

    suite = run_reit_scenario_suite(
        property_value=row["property_value_mm"], noi=row["noi_mm"],
        net_debt=row["net_debt_mm"], shares_outstanding=row["shares_outstanding_mm"],
        scenarios=[BEAR, BASE, BULL],
    )
    values = suite.set_index("scenario")["nav_per_share"]
    assert values["Bear"] < values["Base"] < values["Bull"]
