"""SQLite-backed normalized data store.

`data/processed/research.db` is the project's normalized data store (per the
build spec's "SQLite for the normalized data store" requirement). `ingest.py`
(re)builds it from the cleaned tables on every pipeline run; `analytics_pipeline.py`
(via `load_processed_tables`), `app.py`, the test suite, and
`scripts/verify_project.py` all read from it. The equivalent CSVs under
`data/processed/` are still written alongside it for the Excel pack's
raw-data feel, but this database is the source of truth.

Table and column definitions are derived from the cleaned DataFrames at write
time (not hand-duplicated here), so the schema can never drift from what
`ingest.py` actually produces. Primary keys come from the caller (`ingest.py`
owns that business knowledge); a `ticker` foreign key into `security_master`
is added automatically to any other table that has a `ticker` column.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DB_PATH = PROCESSED_DIR / "research.db"

_SQL_TYPE_BY_DTYPE_KIND = {"i": "INTEGER", "f": "REAL", "b": "INTEGER", "O": "TEXT"}


def _create_table_sql(name: str, df: pd.DataFrame, primary_key: list[str] | None, ticker_fk: bool) -> str:
    col_defs = [f'"{col}" {_SQL_TYPE_BY_DTYPE_KIND.get(df[col].dtype.kind, "TEXT")}' for col in df.columns]
    if primary_key:
        col_defs.append(f'PRIMARY KEY ({", ".join(primary_key)})')
    if ticker_fk:
        col_defs.append('FOREIGN KEY ("ticker") REFERENCES "security_master" ("ticker")')
    return f'CREATE TABLE "{name}" (\n  ' + ",\n  ".join(col_defs) + "\n)"


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def write_tables(tables: dict[str, pd.DataFrame], primary_keys: dict[str, list[str]], db_path: Path = DB_PATH) -> None:
    """(Re)build the normalized SQLite store from a dict of cleaned DataFrames.

    Every table is dropped and recreated so re-running the pipeline always
    leaves the DB in sync with the freshly ingested data -- this is a batch
    pipeline, not an incrementally-updated warehouse. Foreign-key enforcement
    is left off for the bulk load itself (so table/insert order don't matter)
    but the constraints are still declared in the schema.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for name in tables:
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        for name, df in tables.items():
            ticker_fk = name != "security_master" and "ticker" in df.columns and "security_master" in tables
            conn.execute(_create_table_sql(name, df, primary_keys.get(name), ticker_fk))
        conn.commit()

        for name, df in tables.items():
            write_df = df.copy()
            for col in write_df.columns:
                if write_df[col].dtype.kind == "b":
                    write_df[col] = write_df[col].astype(int)
            write_df.to_sql(name, conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def read_tables(names: list[str], db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    conn = connect(db_path)
    try:
        return {name: pd.read_sql_query(f'SELECT * FROM "{name}"', conn) for name in names}
    finally:
        conn.close()
