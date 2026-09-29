"""Pull enrichment data from the Fed-Spend API (https://fed-spend.com).

Two pulls, both cached to data/raw/:

* fedspend_protests.parquet: every GAO bid protest decision Fed-Spend has indexed (case,
  solicitation number, decision date, outcome). Used for a point-in-time feature: GAO protest
  decisions at the contracting office in the three years before a contract ended.
* fedspend_recompete.parquet: Fed-Spend's recompete radar (contracts expiring within a year)
  with its production vulnerability score. Kept local for comparison; never published.

Needs FEDSPEND_API_KEY and FEDSPEND_API_BASE in the environment (scripts/with-secrets.sh).
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import requests

from recompete import RAW_DIR

AGENCIES = [
    "DOD", "VA", "HHS", "DHS", "GSA", "DOE", "NASA", "DOJ", "USDA", "DOS", "DOT",
    "TREAS", "DOI", "DOC", "ED", "EPA", "SSA", "DOL", "HUD", "SBA", "OPM", "USAID",
]  # fmt: skip


def session() -> requests.Session:
    key = os.environ.get("FEDSPEND_API_KEY")
    if not key or not os.environ.get("FEDSPEND_API_BASE"):
        raise SystemExit("FEDSPEND_API_KEY and FEDSPEND_API_BASE must be set; see with-secrets.sh")
    s = requests.Session()
    s.headers.update({"x-api-key": key, "User-Agent": "federal-recompete-cliff/0.1"})
    return s


def get(s: requests.Session, path: str, **params) -> dict:
    base = os.environ["FEDSPEND_API_BASE"].rstrip("/")
    for attempt in range(4):
        r = s.get(f"{base}/{path}", params=params, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def pull_protests(s: requests.Session, page: int = 100) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        body = get(s, "protests", limit=page, offset=offset)
        batch = body.get("data") or []
        rows.extend(batch)
        total = (body.get("meta") or {}).get("total", 0)
        offset += page
        if not batch or offset >= total:
            break
    df = pd.DataFrame(rows)
    keep = ["caseNumber", "protester", "agency", "contractNumber", "solicitationNumber",
            "decisionDate", "filingDate", "outcome", "gaoUrl"]  # fmt: skip
    return df[[c for c in keep if c in df.columns]].drop_duplicates("caseNumber")


def pull_recompete(s: requests.Session) -> pd.DataFrame:
    rows = []
    for agency in [None, *AGENCIES]:
        params = {"maxDays": 365, "limit": 200, "sortBy": "recompeteScore"}
        if agency:
            params["agency"] = agency
        contracts = (get(s, "recompete", **params).get("data") or {}).get("contracts") or []
        for c in contracts:
            c["query_agency"] = agency or "ALL"
        rows.extend(contracts)
        print(f"  recompete {agency or 'ALL'}: {len(contracts)}", flush=True)
    df = pd.json_normalize(rows, sep="_")
    return df.drop_duplicates("contract") if len(df) else df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["protests", "recompete"])
    args = parser.parse_args()
    s = session()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "protests"):
        p = pull_protests(s)
        p.to_parquet(RAW_DIR / "fedspend_protests.parquet", index=False)
        print(f"protests: {len(p):,} GAO decisions")
    if args.only in (None, "recompete"):
        r = pull_recompete(s)
        r["pulled_at"] = pd.Timestamp.now(tz="UTC")
        r.to_parquet(RAW_DIR / "fedspend_recompete.parquet", index=False)
        print(f"recompete: {len(r):,} unique contracts with Fed-Spend vulnerability scores")


if __name__ == "__main__":
    main()
