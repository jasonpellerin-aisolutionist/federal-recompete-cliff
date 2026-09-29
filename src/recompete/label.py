"""Match each ended contract to its follow-on award and label whether the incumbent kept the work.

Candidates come from sql/10_candidates.sql (same office, same PSC or NAICS, started near the
predecessor's end, similar size, not under the predecessor's own vehicle). A candidate is only
eligible when (see docs/LABEL_AUDIT.md for the audits behind each rule):

* both descriptions are informative (at least three distinctive words, not "LABOR" or
  "RESEARCH AND DEVELOPMENT"),
* the descriptions share real content (TF-IDF cosine >= 0.30), and
* for services, the place of performance is the same state when both are known. For
  products, place of performance is the vendor's plant, so it is ignored: using it would let
  vendor identity leak into the pick.

Eligible candidates are scored on requirement similarity only:

    4.0 * description cosine + 1.0 * same PSC + 0.5 * same NAICS
    + 0.5 * same place-of-performance state (services only)
    - 0.75 * |log size ratio| / log 10 - 0.5 * |gap - 30 days| / 365

The best candidate is the follow-on. Vendor identity is read only after the pick, so the
matcher cannot favor the incumbent. Outcomes: `retained`, `changed`, or `no_follow_on`.
A match is `high` confidence when it scores >= 3.0 and beats the best candidate with the
opposite outcome by >= 0.75, `medium` when it scores >= 2.0, otherwise no follow-on.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recompete import ROOT
from recompete.db import connect, run_sql

MIN_COSINE = 0.30
MIN_WORDS = 3
MIN_SCORE = 2.0
HIGH_SCORE = 3.0
HIGH_MARGIN = 0.75
BOILERPLATE = re.compile(r"IGF::[\w,]+::IGF|\bIGF\b|\bOT\b|\bCL\b|\bCT\b", re.IGNORECASE)
GENERIC = set("""
    a an and the of for to in on at by with from under fy base year option period award awards
    new task order orders delivery call contract contracts labor support services service costs
    cost program programs project research development funding funds price clin item items
    requirement requirements purpose provide provides mod modification incremental other
    function technical management professional
""".split())  # fmt: skip
AUDIT_PATH = ROOT / "docs" / "label_audit_sample.csv"


def clean(text: str | None) -> str:
    return BOILERPLATE.sub(" ", text if isinstance(text, str) else "").lower()


def informative(text: str | None) -> bool:
    words = {w for w in re.findall(r"[a-z][a-z0-9&/-]{2,}", clean(text)) if w not in GENERIC}
    return len(words) >= MIN_WORDS


def eligible(c: pd.DataFrame, informative_keys: set[str]) -> pd.DataFrame:
    ok = (
        c["pred_key"].isin(informative_keys)
        & c["succ_key"].isin(informative_keys)
        & (c["desc_cosine"] >= MIN_COSINE)
        & ~(c["is_service"] & c["pop_conflict"])
    )
    return c[ok].copy()


def description_cosine(pairs: pd.DataFrame, descriptions: pd.Series) -> np.ndarray:
    vec = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True,
                          stop_words="english")  # fmt: skip
    matrix = vec.fit_transform(descriptions.map(clean))
    index = pd.Series(np.arange(len(descriptions)), index=descriptions.index)
    a = matrix[index.loc[pairs["pred_key"]].to_numpy()]
    b = matrix[index.loc[pairs["succ_key"]].to_numpy()]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def score(c: pd.DataFrame) -> pd.Series:
    return (
        4.0 * c["desc_cosine"]
        + 1.0 * c["psc_match"]
        + 0.5 * c["naics_match"]
        + 0.5 * (c["pop_match"] & c["is_service"])
        - 0.75 * c["size_logratio"].abs() / np.log(10)
        - 0.5 * (c["gap_days"] - 30).abs() / 365
    )


def pick_follow_ons(c: pd.DataFrame) -> pd.DataFrame:
    c = c.sort_values(["pred_key", "score"], ascending=[True, False])
    best = c.groupby("pred_key", sort=False).head(1).set_index("pred_key")
    rival = (
        c.merge(best[["same_vendor"]], left_on="pred_key", right_index=True, suffixes=("", "_best"))
        .query("same_vendor != same_vendor_best")
        .groupby("pred_key")["score"]
        .max()
    )
    best["rival_score"] = rival
    best["margin"] = best["score"] - best["rival_score"].fillna(best["score"] - 10)
    best["n_candidates"] = c.groupby("pred_key").size()
    return best.reset_index()


def main() -> None:
    con = connect()
    run_sql(con, "10_candidates.sql")
    cand = con.execute("SELECT * FROM candidates").df()
    keys = pd.unique(pd.concat([cand["pred_key"], cand["succ_key"]]))
    desc = (
        con.execute(
            "SELECT award_key, description FROM awards WHERE award_key IN (SELECT * FROM UNNEST(?))",
            [keys.tolist()],
        )  # fmt: skip
        .df()
        .set_index("award_key")["description"]
    )
    cand["desc_cosine"] = description_cosine(cand, desc)
    cand = eligible(cand, set(desc.index[desc.map(informative)]))
    cand["score"] = score(cand)
    best = pick_follow_ons(cand)

    pred = con.execute("SELECT award_key AS pred_key FROM predecessors").df()
    lab = pred.merge(best, on="pred_key", how="left")
    matched = (lab["score"] >= MIN_SCORE).fillna(False).astype(bool)
    same = lab["same_vendor"].astype("boolean").fillna(False).astype(bool)
    lab["outcome"] = np.where(~matched, "no_follow_on", np.where(same, "retained", "changed"))
    lab["confidence"] = np.select(
        [matched & (lab["score"] >= HIGH_SCORE) & (lab["margin"] >= HIGH_MARGIN), matched],
        ["high", "medium"],
        default="none",
    )
    lab.loc[~matched, ["succ_key"]] = None
    cols = ["pred_key", "succ_key", "outcome", "confidence", "score", "margin", "rival_score",
            "n_candidates", "desc_cosine", "psc_match", "naics_match", "gap_days", "size_logratio"]  # fmt: skip
    con.register("lab_df", lab[cols])
    con.execute("CREATE OR REPLACE TABLE follow_on AS SELECT * FROM lab_df")

    summary = con.execute("""
        SELECT outcome, confidence, count(*) AS n
        FROM follow_on GROUP BY ALL ORDER BY outcome, confidence
    """).df()
    print(summary.to_string(index=False))
    if "--audit-sample" in sys.argv:
        write_audit_sample(con)


def write_audit_sample(con, n_per_outcome: int = 50, seed: int = 317) -> None:
    sample = con.execute(f"""
        WITH s AS (
            SELECT f.*, row_number() OVER (PARTITION BY outcome ORDER BY hash(pred_key || '{seed}')) AS rk
            FROM follow_on f WHERE outcome IN ('retained', 'changed')
        )
        SELECT s.outcome, s.confidence, round(s.score, 2) AS score, round(s.desc_cosine, 2) AS desc_cosine,
               s.gap_days, p.office, p.piid AS pred_piid, p.end_date AS pred_end,
               p.recipient_name AS pred_vendor, round(p.obligated / 1e6, 1) AS pred_obligated_m,
               p.psc AS pred_psc, left(p.description, 160) AS pred_description,
               n.piid AS succ_piid, n.base_date AS succ_start, n.recipient_name AS succ_vendor,
               round(n.obligated / 1e6, 1) AS succ_obligated_m, n.psc AS succ_psc,
               left(n.description, 160) AS succ_description,
               NULL AS reviewer_verdict, NULL AS reviewer_note
        FROM s
        JOIN awards p ON p.award_key = s.pred_key
        JOIN awards n ON n.award_key = s.succ_key
        WHERE rk <= {n_per_outcome}
        ORDER BY s.outcome, s.pred_key
    """).df()
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(AUDIT_PATH, index=False)
    print(f"audit sample: {len(sample)} pairs -> {AUDIT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
