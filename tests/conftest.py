import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from scripts import run_pipeline
from src.analytics_pipeline import load_processed_tables


@pytest.fixture(scope="session", autouse=True)
def full_pipeline_run():
    """Run the full pipeline once per test session so every test can rely on
    data/processed/*.csv and output/investment_research_pack.xlsx existing."""
    run_pipeline.main()
    yield


@pytest.fixture(scope="session")
def processed(full_pipeline_run):
    return load_processed_tables()
