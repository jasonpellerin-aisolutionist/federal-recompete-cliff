"""Score the FY27 watchlist with the evaluated model and write the app and Tableau extracts.

Each contract gets a calibrated probability that the incumbent loses the follow-on, the
Fed-Spend heuristic score recomputed from the same inputs, Fed-Spend's live production score
where its recompete radar covers the same PIID, and the three features that push the
estimate up the most (per-row SHAP values). Outputs are aggregates plus public award facts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap

from recompete import APP_DATA_DIR, EXPORT_DIR, MODEL_DIR, RAW_DIR, ROOT
from recompete.db import connect
from recompete.model import add_rates, prepare

MIN_KEEP_SHARE = 0.8

REASONS = {
    "offers": "number of offers on the last award",
    "extent_competed": "how the last award was competed",
    "inc_retain_rate_prior": "incumbent's past recompete record",
    "office_change_rate_prior": "how often this office changes vendors",
    "subagency_change_rate_prior": "how often this sub-agency changes vendors",
    "inc_office_share_3y": "incumbent's share of this office's spend",
    "inc_office_awards_3y": "incumbent's other awards at this office",
    "inc_log_obligated_3y": "incumbent's total federal revenue",
    "inc_awards_3y": "incumbent's federal award count",
    "inc_offices_3y": "number of offices the incumbent serves",
    "projected_ceiling_use": "projected use of the contract ceiling",
    "log_spend_per_year": "annual spend on the contract",
    "log_ceiling": "contract ceiling",
    "duration_years": "contract length",
    "actions_per_year": "modification rate",
    "set_aside_group": "set-aside type",
    "pricing_group": "pricing type",
    "vehicle": "single- or multiple-award vehicle",
    "naics_hhi_3y": "vendor concentration in the industry",
    "naics_vendors_3y": "number of vendors in the industry",
    "office_sector_vendors_3y": "vendors this office uses in the sector",
    "office_awards_3y": "office buying volume",
    "small_business": "incumbent is a small business",
    "is_8a": "incumbent is an 8(a) firm",
    "is_sdvosb": "incumbent is service-disabled veteran-owned",
    "is_wosb": "incumbent is woman-owned",
    "is_hubzone": "incumbent is a HUBZone firm",
    "national_interest": "national interest action",
    "office_recompetes_prior": "office's recompete history",
    "subagency_recompetes_prior": "sub-agency's recompete history",
    "inc_recompetes_prior": "incumbent's recompete history",
    "offers_known": "offers reported",
    "commercial_item": "commercial item",
    "performance_based": "performance-based acquisition",
    "subcontracting_plan": "subcontracting plan requirement",
    "agency": "agency",
    "naics2": "industry sector",
    "psc_group": "product or service category",
    "award_type": "award type",
}

OUT_COLUMNS = [
    "piid", "award_key", "agency", "sub_agency", "office", "recipient_name", "vendor_id",
    "naics", "naics_description", "psc", "psc_description", "end_date", "days_to_end",
    "obligated", "potential_value", "offers", "extent_competed", "set_aside_group",
    "p_change", "risk_band", "heuristic_score", "contract_protests",
    "office_protests_3y", "reason_1",
    "reason_2", "reason_3", "description", "permalink",
]  # fmt: skip


def reason_text(row_values: np.ndarray, features: list[str], k: int = 3) -> list[str]:
    order = np.argsort(-row_values)[:k]
    return [REASONS.get(features[i], features[i]) if row_values[i] > 0 else "" for i in order]


def main() -> None:
    bundle = joblib.load(MODEL_DIR / "incumbent_loss.joblib")
    model, iso, features = bundle["model"], bundle["calibrator"], bundle["features"]
    con = connect()
    watch = con.execute("""
        SELECT w.*, a.piid, a.sub_agency, a.office, a.recipient_name, a.naics_description,
               a.psc_description, a.end_date, a.obligated, a.potential_value, a.description, a.permalink
        FROM watch_features w JOIN awards a USING (award_key)
    """).df()
    watch = add_rates(watch, bundle["prior"])
    X, _ = prepare(watch, bundle["categories"])
    watch["score_raw"] = model.predict_proba(X[features])[:, 1]
    watch["p_change"] = iso.predict(watch["score_raw"])
    values = shap.TreeExplainer(model).shap_values(X[features])
    values = values[1] if isinstance(values, list) else values
    reasons = np.array([reason_text(v, features) for v in values])
    watch[["reason_1", "reason_2", "reason_3"]] = reasons

    # Isotonic calibration is a step function, so rank and band on the raw score.
    s = watch["score_raw"]
    q = s.quantile([0.5, 0.8, 0.95]).to_numpy()
    watch["risk_band"] = np.select(
        [s >= q[2], s >= q[1], s >= q[0]],
        ["Top 5%", "Elevated", "Typical"], "Low",
    )  # fmt: skip
    asof = pd.to_datetime(watch["as_of"]).iloc[0]
    watch["days_to_end"] = (pd.to_datetime(watch["end_date"]) - asof).dt.days

    # Fed-Spend's production score is proprietary: count the overlap, never publish the values.
    live = RAW_DIR / "fedspend_recompete.parquet"
    radar = set(pd.read_parquet(live)["contract"]) if live.exists() else set()
    overlap = int(watch["piid"].isin(radar).sum())

    out = watch.sort_values("score_raw", ascending=False)[OUT_COLUMNS]
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    check_not_shrunk(len(out), APP_DATA_DIR / "watchlist_fy27.parquet")
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(APP_DATA_DIR / "watchlist_fy27.parquet", index=False)
    out.drop(columns=["description"]).to_csv(EXPORT_DIR / "watchlist_fy27.csv", index=False)
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    audit = pd.read_csv(ROOT / "docs" / "label_audit_sample.csv")["reviewer_verdict"]
    meta = {
        "asof": f"{asof:%B %d, %Y}",
        "n_labeled": sum(metrics["n"].values()),
        "change_rate": metrics["base_rate_train"],
        "test_metrics": metrics["models"]["LightGBM (calibrated)"],
        "logistic_metrics": metrics["models"]["Logistic regression"],
        "heuristic_metrics": metrics["models"]["Fed-Spend heuristic (as shipped, score/100)"],
        "high_confidence_metrics": metrics["high_confidence_only"],
        "audit": {k: int(v) for k, v in audit.value_counts().items()},
    }
    (APP_DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(
        f"watchlist: {len(out):,} FY27 contracts, ${out['obligated'].sum() / 1e9:,.1f}B obligated; "
        f"mean P(change) {out['p_change'].mean():.3f}; {overlap} also on the Fed-Spend radar"
    )


def check_not_shrunk(n_new: int, published: Path, floor: float = MIN_KEEP_SHARE) -> None:
    """Contracts leave the list only as they end (a few percent a week), so a large drop means
    missing input data, not a real change. Pass --allow-shrink to publish anyway."""
    if not published.exists() or "--allow-shrink" in sys.argv:
        return
    n_old = pq.read_metadata(published).num_rows
    if n_new < floor * n_old:
        raise SystemExit(
            f"watchlist shrank from {n_old:,} to {n_new:,} contracts; "
            "not publishing (check the raw downloads, or pass --allow-shrink)"
        )


if __name__ == "__main__":
    main()
