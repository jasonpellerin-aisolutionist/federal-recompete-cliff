"""Compare this week's scored watchlist with last week's and post a digest to n8n.

Last week's file comes from git (`HEAD:app/data/watchlist_fy27.parquet`) so the comparison is
against exactly what the app was showing. The digest lists contracts that newly entered the
top 5%, the biggest risk moves, and top-band contracts ending within 90 days. It is written to
reports/weekly_digest.json and, when N8N_DIGEST_WEBHOOK_URL is set, posted to that webhook.
"""

from __future__ import annotations

import io
import json
import os
import subprocess

import pandas as pd
import requests

from recompete import APP_DATA_DIR, ROOT

WATCHLIST = APP_DATA_DIR / "watchlist_fy27.parquet"
APP_URL = "https://fy27-recompete-cliff.streamlit.app"


def previous() -> pd.DataFrame | None:
    rel = WATCHLIST.relative_to(ROOT).as_posix()
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return pd.read_parquet(io.BytesIO(blob))


def row_view(r: pd.Series) -> dict:
    return {
        "piid": r["piid"],
        "incumbent": r["recipient_name"],
        "agency": r["agency"],
        "office": r["office"],
        "end_date": f"{pd.Timestamp(r['end_date']):%Y-%m-%d}",
        "obligated_m": round(float(r["obligated"]) / 1e6, 1),
        "p_change": round(float(r["p_change"]), 3),
        "top_driver": r["reason_1"],
        "usaspending": r["permalink"],
    }


def build_digest(now: pd.DataFrame, before: pd.DataFrame | None) -> dict:
    meta = json.loads((APP_DATA_DIR / "meta.json").read_text())
    top = now[now["risk_band"] == "Top 5%"]
    soon = top[top["days_to_end"].between(0, 90)].nsmallest(10, "days_to_end")
    digest = {
        "asof": meta["asof"],
        "app_url": APP_URL,
        "contracts": len(now),
        "obligated_b": round(float(now["obligated"].sum()) / 1e9, 1),
        "expected_changes": round(float(now["p_change"].sum())),
        "top_band_ending_90d": [row_view(r) for _, r in soon.iterrows()],
        "new_to_top_band": [],
        "biggest_moves": [],
    }
    if before is not None:
        was_top = set(before.loc[before["risk_band"] == "Top 5%", "piid"])
        new_top = top[~top["piid"].isin(was_top)].nlargest(10, "p_change")
        digest["new_to_top_band"] = [row_view(r) for _, r in new_top.iterrows()]
        moved = now.merge(before[["piid", "p_change"]], on="piid", suffixes=("", "_before"))
        moved["delta"] = moved["p_change"] - moved["p_change_before"]
        big = moved.reindex(moved["delta"].abs().sort_values(ascending=False).index).head(10)
        digest["biggest_moves"] = [
            {**row_view(r), "delta": round(float(r["delta"]), 3)}
            for _, r in big.iterrows()
            if abs(r["delta"]) >= 0.02
        ]
        digest["dropped_off"] = int((~before["piid"].isin(now["piid"])).sum())
        digest["added"] = int((~now["piid"].isin(before["piid"])).sum())
    return digest


def main() -> None:
    digest = build_digest(pd.read_parquet(WATCHLIST), previous())
    out = ROOT / "reports" / "weekly_digest.json"
    out.write_text(json.dumps(digest, indent=2))
    print(f"digest: {len(digest['new_to_top_band'])} new top-band contracts, "
          f"{len(digest['top_band_ending_90d'])} top-band contracts ending within 90 days")  # fmt: skip
    url = os.environ.get("N8N_DIGEST_WEBHOOK_URL")
    if url:
        headers = {"X-Digest-Token": os.environ.get("N8N_DIGEST_TOKEN", "")}
        r = requests.post(url, json=digest, headers=headers, timeout=60)
        r.raise_for_status()
        print(f"posted digest to n8n: HTTP {r.status_code}")


if __name__ == "__main__":
    main()
