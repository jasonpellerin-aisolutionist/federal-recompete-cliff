import numpy as np
import pandas as pd

from recompete.features import fedspend_heuristic


def row(**kw) -> dict:
    base = {
        "offers": np.nan,
        "projected_ceiling_use": 0.5,
        "contract_protests": 0,
        "office_protests_3y": 0,
        "office_protests_sustained_3y": 0,
        "actions_per_year": 0.0,
        "duration_years": 3.0,
        "extent_competed": None,
    }
    base.update(kw)
    return base


def test_weighted_mean_over_resolved_signals():
    # offers 85 (w3), ceiling 85 (w2.5), posture 70 (w1.5); protests and churn abstain.
    df = pd.DataFrame([row(offers=6, projected_ceiling_use=0.96, extent_competed="A")])
    out = fedspend_heuristic(df).iloc[0]
    assert out["heuristic_score"] == round((85 * 3 + 85 * 2.5 + 70 * 1.5) / 7)
    assert out["heuristic_signals"] == 3


def test_sole_source_single_offer_reads_as_locked_in():
    df = pd.DataFrame([row(offers=1, projected_ceiling_use=0.2, extent_competed="C")])
    assert fedspend_heuristic(df).iloc[0]["heuristic_score"] < 40


def test_office_protest_climate_needs_two_filings():
    one = fedspend_heuristic(pd.DataFrame([row(office_protests_3y=1)])).iloc[0]
    two = fedspend_heuristic(pd.DataFrame([row(office_protests_3y=2)])).iloc[0]
    assert two["heuristic_signals"] == one["heuristic_signals"] + 1


def test_churn_abstains_under_one_year():
    short = fedspend_heuristic(pd.DataFrame([row(actions_per_year=40, duration_years=0.5)])).iloc[0]
    long = fedspend_heuristic(pd.DataFrame([row(actions_per_year=40, duration_years=2)])).iloc[0]
    assert long["heuristic_signals"] == short["heuristic_signals"] + 1
