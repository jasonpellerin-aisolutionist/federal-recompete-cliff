"""DuckDB connection and SQL-file runner shared by the pipeline steps."""

from __future__ import annotations

import duckdb

from recompete import DUCKDB_PATH, RAW_DIR, SQL_DIR


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def run_sql(con: duckdb.DuckDBPyConnection, name: str, **params: str) -> None:
    """Run sql/<name>, substituting {raw_dir} and any keyword params into the text."""
    text = (SQL_DIR / name).read_text()
    con.execute(text.format(raw_dir=RAW_DIR.as_posix(), **params))
