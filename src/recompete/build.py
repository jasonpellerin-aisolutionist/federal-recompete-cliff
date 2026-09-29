"""Load the raw award files into DuckDB as one typed, deduplicated `awards` table."""

from __future__ import annotations

from recompete.db import connect, run_sql


def main() -> None:
    con = connect()
    run_sql(con, "00_awards.sql")
    n, lo, hi, vendors = con.execute(
        "SELECT count(*), min(base_date), max(base_date), count(DISTINCT vendor_id) FROM awards"
    ).fetchone()
    print(f"awards: {n:,} contracts signed {lo} to {hi}, {vendors:,} distinct vendors")


if __name__ == "__main__":
    main()
