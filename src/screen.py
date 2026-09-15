"""Transparent, configurable multi-factor screen and research-note generator.

Every factor is a simple, inspectable calculation over the processed data --
no black-box scoring. Weights are user-configurable and the ranking updates
live. The research note template pulls every number from stored data or the
valuation/scenario modules; nothing is invented.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FACTOR_WEIGHTS = {
    "valuation_discount": 0.25,
    "growth": 0.20,
    "leverage": 0.15,
    "profitability": 0.15,
    "momentum": 0.15,
    "dividend_yield": 0.10,
}


def _zscore(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def build_factor_table(
    master: pd.DataFrame,
    latest_annual: pd.DataFrame,
    prior_annual: pd.DataFrame,
    peer_multiples_latest: pd.DataFrame,
    momentum_12m: pd.Series,
) -> pd.DataFrame:
    """Assemble one row per ticker with the six raw factor inputs.

    valuation_discount: -1 * EV/EBITDA (or P/FFO for REITs) so a LOWER multiple
        (cheaper) scores higher after z-scoring.
    growth: YoY revenue growth (FFO growth for REITs), latest vs. prior annual period.
    leverage: -1 * net_debt/EBITDA (or net_debt/NOI for REITs) so lower leverage scores higher.
    profitability: EBITDA margin (or NOI margin proxy for REITs).
    momentum: trailing 12-month total price return.
    dividend_yield: dividends_paid / market cap, latest annual period.
    """
    df = master[["ticker", "sector", "security_type"]].copy().set_index("ticker")

    latest = latest_annual.set_index("ticker")
    prior = prior_annual.set_index("ticker")
    peers = peer_multiples_latest.set_index("ticker")

    growth = (latest["revenue_mm"] - prior["revenue_mm"]) / prior["revenue_mm"].replace(0, np.nan)
    df["growth_raw"] = growth.reindex(df.index)

    ev_ebitda = peers["ev_ebitda_multiple"].reindex(df.index)
    p_ffo = peers["p_ffo_multiple"].reindex(df.index)
    valuation_multiple = p_ffo.where(df["security_type"] == "REIT", ev_ebitda)
    df["valuation_discount_raw"] = -1 * valuation_multiple

    net_debt = (latest["debt_mm"] - latest["cash_mm"]).reindex(df.index)
    ebitda = latest["ebitda_mm"].reindex(df.index)
    leverage_ratio = net_debt / ebitda.replace(0, np.nan)
    df["leverage_raw"] = -1 * leverage_ratio

    margin = (latest["ebitda_mm"] / latest["revenue_mm"].replace(0, np.nan)).reindex(df.index)
    df["profitability_raw"] = margin

    df["momentum_raw"] = momentum_12m.reindex(df.index)

    market_cap = latest["share_price"] * latest["shares_outstanding_mm"]
    div_yield = (latest["dividends_paid_mm"] / market_cap.replace(0, np.nan)).reindex(df.index)
    df["dividend_yield_raw"] = div_yield

    return df.reset_index()


def score_factor_table(factor_table: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Z-score each raw factor cross-sectionally, apply the (user-configurable)
    weights, and rank descending by composite score. Weights need not sum to 1
    -- they're relative importances; the composite is a weighted average."""
    weights = weights or DEFAULT_FACTOR_WEIGHTS
    df = factor_table.copy()

    raw_to_factor = {
        "valuation_discount_raw": "valuation_discount",
        "growth_raw": "growth",
        "leverage_raw": "leverage",
        "profitability_raw": "profitability",
        "momentum_raw": "momentum",
        "dividend_yield_raw": "dividend_yield",
    }

    total_weight = sum(weights.get(f, 0.0) for f in raw_to_factor.values()) or 1.0
    composite = pd.Series(0.0, index=df.index)
    for raw_col, factor_name in raw_to_factor.items():
        z = _zscore(df[raw_col].fillna(df[raw_col].median()))
        df[f"{factor_name}_zscore"] = z
        composite = composite + z * weights.get(factor_name, 0.0)

    df["composite_score"] = composite / total_weight
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


@dataclass
class ResearchNoteInputs:
    ticker: str
    name: str
    sector: str
    security_type: str
    country: str
    latest_price: float
    dcf_value_per_share: float
    comp_value_per_share: float
    bull_value_per_share: float
    base_value_per_share: float
    bear_value_per_share: float
    key_assumptions: dict
    catalysts: list[str]
    risks: list[str]
    factor_rank: int
    factor_total: int
    primary_valuation_method: str = "DCF (base case)"


SECURITY_TYPE_DESCRIPTOR = {
    "Company": "company",
    "REIT": "real estate investment trust (REIT)",
    "Financial": "financial institution",
    "Infrastructure": "infrastructure company",
}


def render_research_note(inp: ResearchNoteInputs) -> str:
    upside_base = inp.base_value_per_share / inp.latest_price - 1 if inp.latest_price else float("nan")
    assumptions_lines = "\n".join(f"- **{k}**: {v}" for k, v in inp.key_assumptions.items())
    catalysts_lines = "\n".join(f"- {c}" for c in inp.catalysts) or "- None identified."
    risks_lines = "\n".join(f"- {r}" for r in inp.risks) or "- None identified."
    descriptor = SECURITY_TYPE_DESCRIPTOR.get(inp.security_type, inp.security_type.lower())

    return f"""# Research Note: {inp.name} ({inp.ticker})

> **Simulated analysis.** This note is generated entirely from synthetic data
> produced by this project for portfolio-demonstration purposes. It is not
> investment advice, not a real research report, and does not reference any
> real company, security, or market data.

## Business & Sector Summary
{inp.name} ({inp.ticker}) is a fictional {descriptor} in the
{inp.sector} sector, domiciled in {inp.country}, used here to demonstrate a
valuation and portfolio-analytics workflow.

## Valuation Overview
| Method | Value per share |
|---|---|
| Current price | {inp.latest_price:,.2f} |
| {inp.primary_valuation_method} | {inp.dcf_value_per_share:,.2f} |
| Trading comparables | {inp.comp_value_per_share:,.2f} |

Implied upside/downside vs. {inp.primary_valuation_method.lower()}: **{upside_base:+.1%}**

## Factor Screen Position
Ranked **#{inp.factor_rank} of {inp.factor_total}** securities in the research
universe under the current factor weights (see Research Universe view for the
live, user-adjustable ranking).

## Catalysts
{catalysts_lines}

## Risks & Downside Cases
{risks_lines}

## Key Assumptions
{assumptions_lines}

## Bull / Base / Bear Target Values
| Case | Value per share | vs. current price |
|---|---|---|
| Bull | {inp.bull_value_per_share:,.2f} | {(inp.bull_value_per_share/inp.latest_price - 1):+.1%} |
| Base | {inp.base_value_per_share:,.2f} | {(inp.base_value_per_share/inp.latest_price - 1):+.1%} |
| Bear | {inp.bear_value_per_share:,.2f} | {(inp.bear_value_per_share/inp.latest_price - 1):+.1%} |

---
*Every figure above is computed from the project's synthetic dataset and
valuation functions (see `src/valuation.py`, `src/scenarios.py`). Simulated
analysis -- not investment advice.*
"""
