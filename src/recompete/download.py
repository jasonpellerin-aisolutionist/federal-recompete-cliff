"""Download federal prime contract award summaries from USAspending, one file per fiscal year.

Uses the public Advanced Search download API (no key): contracts (award types A-D) signed in
the fiscal year with a current total obligation of at least $1M. Each year becomes
data/raw/awards_fy{YYYY}.parquet with only the columns this project uses. Re-running skips
years already on disk unless --force is passed, so a refresh only re-pulls the open years.
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from functools import partial

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from recompete import RAW_DIR

API = "https://api.usaspending.gov/api/v2"
AWARD_TYPES = ["A", "B", "C", "D"]
MIN_OBLIGATION = 1_000_000
BACKFILL_WINDOW_DAYS = 90
POLL_SECONDS = 10
TIMEOUT_SECONDS = 60 * 60
# USAspending often marks a finished HTTP request as failed with "An error occurred."
# Re-queue that job. Do not retry a timeout: one hour is already the whole budget for a slice.
JOB_ATTEMPTS = 3
JOB_RETRY_SLEEP = 60

KEEP = [
    "contract_award_unique_key",
    "award_id_piid",
    "parent_award_id_piid",
    "parent_award_agency_id",
    "award_type_code",
    "award_type",
    "total_obligated_amount",
    "current_total_value_of_award",
    "potential_total_value_of_award",
    "award_base_action_date",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    "period_of_performance_potential_end_date",
    "last_modified_date",
    "awarding_agency_code",
    "awarding_agency_name",
    "awarding_sub_agency_code",
    "awarding_sub_agency_name",
    "awarding_office_code",
    "awarding_office_name",
    "funding_office_code",
    "recipient_uei",
    "recipient_name",
    "recipient_parent_uei",
    "recipient_parent_name",
    "recipient_state_code",
    "primary_place_of_performance_state_code",
    "naics_code",
    "naics_description",
    "product_or_service_code",
    "product_or_service_code_description",
    "extent_competed_code",
    "extent_competed",
    "solicitation_procedures_code",
    "other_than_full_and_open_competition_code",
    "number_of_offers_received",
    "type_of_set_aside_code",
    "type_of_set_aside",
    "type_of_contract_pricing_code",
    "solicitation_identifier",
    "multiple_or_single_award_idv_code",
    "type_of_idc_code",
    "commercial_item_acquisition_procedures_code",
    "contracting_officers_determination_of_business_size",
    "c8a_program_participant",
    "service_disabled_veteran_owned_business",
    "woman_owned_business",
    "historically_underutilized_business_zone_hubzone_firm",
    "national_interest_action_code",
    "parent_award_type_code",
    "parent_award_single_or_multiple_code",
    "fair_opportunity_limited_sources_code",
    "contract_bundling_code",
    "consolidated_contract_code",
    "subcontracting_plan_code",
    "performance_based_service_acquisition_code",
    "number_of_actions",
    "total_outlayed_amount",
    "prime_award_base_transaction_description",
    "usaspending_permalink",
]


def _session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


# Retries POST too: a repeated download request only queues a duplicate job.
HTTP = _session()


def fy_bounds(fy: int) -> tuple[str, str]:
    return f"{fy - 1}-10-01", f"{fy}-09-30"


def request_download(start: str, end: str, date_type: str = "date_signed") -> dict:
    body = {
        "filters": {
            "award_type_codes": AWARD_TYPES,
            "time_period": [{"start_date": start, "end_date": end, "date_type": date_type}],
            "award_amounts": [{"lower_bound": MIN_OBLIGATION}],
        },
        "file_format": "csv",
    }
    r = HTTP.post(f"{API}/download/awards/", json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def wait_for(status_url: str, fy: int | str) -> dict:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        s = HTTP.get(status_url, timeout=60).json()
        if s["status"] == "finished":
            return s
        if s["status"] == "failed":
            raise RuntimeError(f"{fy} download failed: {s.get('message')}")
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"{fy} download did not finish in {TIMEOUT_SECONDS}s")


def fetch_status(start: str, end: str, date_type: str, label: str) -> dict:
    """Queue a download and wait. A failed job is re-queued. A timeout is not."""
    last: RuntimeError | None = None
    for attempt in range(1, JOB_ATTEMPTS + 1):
        job = request_download(start, end, date_type=date_type)
        try:
            return wait_for(job["status_url"], label)
        except RuntimeError as exc:
            last = exc
            print(f"{label}: attempt {attempt} of {JOB_ATTEMPTS} failed ({exc})", flush=True)
            if attempt == JOB_ATTEMPTS:
                break
            time.sleep(JOB_RETRY_SLEEP)
    assert last is not None
    raise last


def read_awards(zip_bytes: bytes) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in sorted(z.namelist()):
            if not name.startswith("Contracts_PrimeAwardSummaries") or not name.endswith(".csv"):
                continue
            with z.open(name) as fh:
                df = pd.read_csv(fh, dtype=str, keep_default_na=False, na_values=[""])
            frames.append(df[[c for c in KEEP if c in df.columns]])
    if not frames:
        raise RuntimeError("download zip had no Contracts_PrimeAwardSummaries file")
    df = pd.concat(frames, ignore_index=True)
    return df[df["contract_award_unique_key"].str.startswith("CONT_AWD_", na=False)]


def download_year(fy: int, force: bool = False) -> str:
    out = RAW_DIR / f"awards_fy{fy}.parquet"
    if out.exists() and not force:
        return f"FY{fy}: cached ({out.name})"
    started = time.monotonic()
    status = fetch_status(*fy_bounds(fy), "date_signed", f"FY{fy}")
    blob = HTTP.get(status["file_url"], timeout=600).content
    df = read_awards(blob)
    df["source_fy_signed"] = fy
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return f"FY{fy}: {len(df):,} awards in {time.monotonic() - started:,.0f}s"


def download_modified(days: int) -> str:
    """Pull every award modified in the last `days` days, whatever year it was signed.

    Extensions and new obligations land on old awards, so this is how end dates and totals stay
    current between full pulls. `build` keeps the newest version of each award key.
    """
    end = date.today()
    start = end - timedelta(days=days)
    label = f"modified {start}..{end}"
    status = fetch_status(start.isoformat(), end.isoformat(), "last_modified_date", label)
    df = read_awards(HTTP.get(status["file_url"], timeout=600).content)
    df["source_fy_signed"] = None
    out = RAW_DIR / f"awards_modified_{end:%Y%m%d}.parquet"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return f"modified {start} to {end}: {len(df):,} awards -> {out.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2012, 2027)))
    parser.add_argument(
        "--modified-days", type=int, help="pull awards modified in the last N days instead"
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        help="pull awards modified in the last N days in 90-day slices (active pre-FY2012 contracts)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.modified_days:
        print(download_modified(args.modified_days), flush=True)
        return
    if args.backfill_days:
        jobs = {w[0]: partial(download_window, *w, args.force) for w in windows(args.backfill_days)}
    else:
        jobs = {fy: partial(download_year, fy, args.force) for fy in args.years}
    run_all(jobs, args.workers)


def windows(days: int, width: int = BACKFILL_WINDOW_DAYS) -> list[tuple[date, date]]:
    end = date.today()
    starts = [end - timedelta(days=d) for d in range(days, 0, -width)]
    return [(s, min(s + timedelta(days=width - 1), end)) for s in starts]


def download_window(start: date, end: date, force: bool = False) -> str:
    """One slice of the backfill. A single two-year export does not finish within the timeout."""
    out = RAW_DIR / f"awards_backfill_{start:%Y%m%d}.parquet"
    if out.exists() and not force:
        return f"backfill {start}: cached ({out.name})"
    started = time.monotonic()
    label = f"backfill {start}..{end}"
    status = fetch_status(start.isoformat(), end.isoformat(), "last_modified_date", label)
    df = read_awards(HTTP.get(status["file_url"], timeout=600).content)
    df["source_fy_signed"] = None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return f"backfill {start}..{end}: {len(df):,} awards in {time.monotonic() - started:,.0f}s"


def run_all(jobs: dict, workers: int) -> None:
    """Run every job, retry failures one at a time, and fail if any piece is still missing."""
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): key for key, fn in jobs.items()}
        for fut in as_completed(futures):
            try:
                print(fut.result(), flush=True)
            except Exception as exc:  # noqa: BLE001 - report every piece, keep the others going
                print(f"{futures[fut]}: FAILED {exc}", flush=True)
                failed.append(futures[fut])
    for key in list(failed):
        time.sleep(60)
        try:
            print(jobs[key](), "(sequential retry)", flush=True)
            failed.remove(key)
        except Exception as exc:  # noqa: BLE001
            print(f"{key}: FAILED again {exc}", flush=True)
    if failed:
        raise SystemExit(
            f"missing {sorted(map(str, failed))}; refusing to continue on partial data"
        )


if __name__ == "__main__":
    main()
