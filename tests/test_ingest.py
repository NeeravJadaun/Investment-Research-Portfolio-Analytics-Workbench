from src.generate_data import generate_all
from src.ingest import run_ingestion


def test_all_four_planted_issues_are_detected(tmp_path):
    """The pipeline must find the missing value, duplicate, unit mismatch, and
    negative-value anomaly on its own -- it never reads the planted-issue
    manifest, so this test is an independent check that detection actually works."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    result = generate_all(out_dir=raw_dir)
    manifest = result["planted_issues"]

    ingestion = run_ingestion(raw_dir=raw_dir, processed_dir=processed_dir)
    issue_types_found = {i.issue_type for i in ingestion["issues"]}

    assert "missing_value" in issue_types_found
    assert "duplicate_observation" in issue_types_found
    assert "unit_mismatch" in issue_types_found
    assert "negative_value_anomaly" in issue_types_found

    missing = next(i for i in ingestion["issues"] if i.issue_type == "missing_value")
    assert missing.ticker == manifest["missing_value"]["ticker"]

    dup = next(i for i in ingestion["issues"] if i.issue_type == "duplicate_observation")
    assert dup.ticker == manifest["duplicate_observation"]["ticker"]

    unit = next(i for i in ingestion["issues"] if i.issue_type == "unit_mismatch")
    assert unit.ticker == manifest["unit_mismatch"]["ticker"]

    neg = next(i for i in ingestion["issues"] if i.issue_type == "negative_value_anomaly")
    assert neg.ticker == manifest["negative_anomaly"]["ticker"]


def test_deduplication_removes_duplicate_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    generate_all(out_dir=raw_dir)
    ingestion = run_ingestion(raw_dir=raw_dir, processed_dir=processed_dir)

    quarterly = ingestion["tables"]["company_financials_quarterly"]
    dup_count = quarterly.duplicated(subset=["ticker", "fiscal_year", "fiscal_quarter"]).sum()
    assert dup_count == 0


def test_data_quality_csv_and_data_dictionary_written(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    generate_all(out_dir=raw_dir)
    run_ingestion(raw_dir=raw_dir, processed_dir=processed_dir)

    assert (processed_dir / "data_quality.csv").exists()
    assert (processed_dir / "data_dictionary.csv").exists()

    import pandas as pd
    dq = pd.read_csv(processed_dir / "data_quality.csv")
    assert len(dq) == 4
    assert set(dq.columns) >= {"table", "issue_type", "ticker", "field", "detail"}


def test_schema_validation_passes_on_clean_generated_data(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    generate_all(out_dir=raw_dir)
    ingestion = run_ingestion(raw_dir=raw_dir, processed_dir=processed_dir)
    assert ingestion["schema_problems"] == []
