"""Build point-in-time feature tables for training (ended contracts) and the FY27 watchlist.

Training rows are predecessors from `follow_on`, observed 180 days before they ended (a
realistic capture lead time). Watchlist rows are $5M+ contracts whose current end date falls
in FY2027 (Oct 1 2026 to Sep 30 2027), observed as of today. Both go through the same SQL
(sql/20_features.sql), then get the Fed-Spend heuristic score recomputed from the same inputs.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from recompete.db import connect, run_sql

LEAD_DAYS = 180
FY27 = (date(2026, 10, 1), date(2027, 9, 30))
COMPETED = {"A", "D", "F", "CDO"}
NOT_COMPETED = {"B", "C", "G", "NDO"}


def fedspend_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """Fed-Spend's production vulnerability score, re-implemented.

    Weighted mean of 0-100 votes over the signals that resolve. CPARS, value growth, and the
    set-aside change signal need data USAspending does not publish, so they abstain here.
    """
    votes, weights = [], []
    num = ["offers", "projected_ceiling_use", "contract_protests", "office_protests_3y",
           "office_protests_sustained_3y", "actions_per_year", "duration_years"]  # fmt: skip
    df = df.assign(**{c: df[c].astype(float) for c in num})

    offers = df["offers"]
    s = np.select([offers >= 5, offers >= 3, offers == 2], [85, 65, 45], 25).astype(float)
    votes.append(np.where(offers.notna(), s, np.nan))
    weights.append(3.0)

    pct = df["projected_ceiling_use"] * 100
    votes.append(np.select([pct >= 95, pct >= 85, pct <= 25], [85, 70, 25], 52).astype(float))
    weights.append(2.5)

    n = df["contract_protests"]
    votes.append(np.where(n > 0, np.clip(50 + np.minimum(40, n * 3), 0, 100), np.nan))
    weights.append(2.5)

    t, sus = df["office_protests_3y"], df["office_protests_sustained_3y"]
    climate = np.select([t >= 12, t >= 5], [68, 60], 54) + np.minimum(12, sus * 6)
    votes.append(np.where(t >= 2, np.clip(climate, 0, 100), np.nan))
    weights.append(1.0)

    per_year = df["actions_per_year"]
    mods = np.select(
        [per_year > 30, per_year > 15, per_year > 8, per_year > 3], [80, 72, 62, 52], 42
    )
    votes.append(np.where((per_year > 0) & (df["duration_years"] >= 1), mods, np.nan).astype(float))
    weights.append(1.0)

    code = df["extent_competed"].fillna("").str.upper()
    posture = np.where(code.isin(COMPETED), 70.0, np.where(code.isin(NOT_COMPETED), 30.0, np.nan))
    votes.append(posture)
    weights.append(1.5)

    v = np.vstack(votes).T
    w = np.array(weights)
    resolved = ~np.isnan(v)
    total_w = (resolved * w).sum(axis=1)
    weighted = np.nansum(v * w, axis=1)
    score = np.where(total_w > 0, np.round(weighted / np.where(total_w > 0, total_w, 1)), 50)
    return pd.DataFrame(
        {"heuristic_score": np.clip(score, 0, 100), "heuristic_signals": resolved.sum(axis=1)},
        index=df.index,
    )


def build(con, base_sql: str, table: str) -> pd.DataFrame:
    con.execute(f"CREATE OR REPLACE TABLE score_base AS {base_sql}")
    run_sql(con, "20_features.sql")
    df = con.execute("SELECT * FROM features").df()
    df = pd.concat([df, fedspend_heuristic(df)], axis=1)
    con.register("feat_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM feat_df")
    con.unregister("feat_df")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    con = connect()
    run_sql(con, "05_protests.sql")

    train = build(
        con,
        f"""SELECT f.pred_key AS award_key, a.end_date - INTERVAL {LEAD_DAYS} DAY AS as_of
            FROM follow_on f JOIN awards a ON a.award_key = f.pred_key""",
        "train_features",
    )
    print(f"train_features: {len(train):,} rows, {train.shape[1]} columns")

    watch = build(
        con,
        f"""SELECT award_key, DATE '{args.asof}' AS as_of FROM awards
            WHERE obligated >= 5e6
              AND end_date BETWEEN DATE '{FY27[0]}' AND DATE '{FY27[1]}'
              AND NOT regexp_matches(coalesce(psc, ''), '^[YZ]')
              AND date_diff('day', coalesce(start_date, base_date), end_date) >= 180""",
        "watch_features",
    )
    print(f"watch_features: {len(watch):,} FY27 contracts as of {args.asof}")


if __name__ == "__main__":
    main()
