"""Social and blog cover image (1200x630) from the published watchlist and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from recompete.style import BODY, INK, NAVY, SLATE, TEAL, apply_style

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "figures" / "cover.png"


def main() -> None:
    apply_style()
    watch = pd.read_parquet(ROOT / "app" / "data" / "watchlist_fy27.parquet")
    meta = json.loads((ROOT / "app" / "data" / "meta.json").read_text())
    month = pd.to_datetime(watch["end_date"]).dt.to_period("M")
    by_month = watch.groupby(month)["obligated"].sum() / 1e9

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.text(0.05, 0.84, "The FY27 Recompete Cliff", fontsize=30, weight="bold", color=INK)
    fig.text(
        0.05,
        0.76,
        "Which federal contracts ending Oct 2026 to Sep 2027 will change hands?",
        fontsize=14,
        color=BODY,
    )
    stats = [
        (f"{len(watch):,}", "contracts ending in FY27"),
        (f"${watch['obligated'].sum() / 1e9:,.1f}B", "lifetime obligations"),
        (f"{meta['test_metrics']['auc']:.3f}", "held-out AUC, FY2025"),
        (f"{meta['test_metrics']['top_decile_precision']:.0%}", "top-decile precision"),
    ]
    for i, (value, label) in enumerate(stats):
        y = 0.60 - i * 0.13
        fig.text(0.05, y, value, fontsize=24, weight="bold", color=TEAL)
        fig.text(0.05, y - 0.045, label, fontsize=11, color=SLATE)

    ax = fig.add_axes([0.47, 0.14, 0.49, 0.54])
    ax.bar(range(len(by_month)), by_month.values, color=NAVY, width=0.72)
    ax.set_xticks(range(len(by_month)))
    ax.set_xticklabels([p.strftime("%b\n%y") for p in by_month.index], fontsize=9)
    ax.set_ylabel("Obligated, $B")
    ax.set_title("Dollars reaching their end date, by month", loc="left", fontsize=12)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_ylim(0, by_month.max() * 1.1)

    fig.text(
        0.05,
        0.04,
        "Jason Pellerin  |  USAspending.gov prime contracts, as of "
        f"{meta['asof']}  |  "
        "github.com/jasonpellerin-aisolutionist/federal-recompete-cliff",
        fontsize=9,
        color=SLATE,
    )
    fig.savefig(OUT, dpi=100)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
