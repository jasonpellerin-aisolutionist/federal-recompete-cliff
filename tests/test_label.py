import numpy as np
import pandas as pd

from recompete.label import eligible, informative, pick_follow_ons, score


def candidates(same_vendor: list[bool]) -> pd.DataFrame:
    c = pd.DataFrame({
        "pred_key": ["P1", "P1", "P1", "P2", "P2"],
        "succ_key": ["S1", "S2", "S3", "S4", "S5"],
        "psc_match": [True, True, False, True, True],
        "naics_match": [True, False, True, True, True],
        "desc_cosine": [0.8, 0.2, 0.9, 0.1, 0.6],
        "pop_match": [True, True, False, False, True],
        "is_service": [True] * 5,
        "size_logratio": [0.1, 0.0, 1.5, 0.2, 0.3],
        "gap_days": [20, 40, 300, 10, 60],
        "same_vendor": same_vendor,
    })  # fmt: skip
    c["score"] = score(c)
    return c


def test_matcher_never_uses_vendor_identity():
    a = pick_follow_ons(candidates([True, False, False, False, True]))
    b = pick_follow_ons(candidates([False, True, True, True, False]))
    assert a["succ_key"].tolist() == b["succ_key"].tolist() == ["S1", "S5"]


def test_margin_is_against_best_opposite_outcome():
    best = pick_follow_ons(candidates([True, False, True, False, False])).set_index("pred_key")
    c = candidates([True, False, True, False, False])
    s2 = c.loc[c["succ_key"] == "S2", "score"].item()
    assert np.isclose(best.loc["P1", "margin"], best.loc["P1", "score"] - s2)


def test_generic_descriptions_are_not_informative():
    assert not informative("IGF::OT::IGF RDT&E  LABOR")
    assert not informative("NEW TASK ORDER - SUPPORT COSTS")
    assert informative("ORACLE SOFTWARE AND HARDWARE MAINTENANCE SUPPORT SERVICES")


def test_pop_conflict_blocks_services_but_not_products():
    c = candidates([False] * 5).assign(
        is_service=[True, False, True, True, True],
        pop_conflict=[True, True, False, False, False],
    )
    kept = eligible(c, set(c["pred_key"]) | set(c["succ_key"]))["succ_key"].tolist()
    assert "S1" not in kept  # service at a different site
    assert "S2" not in kept  # product, dropped for low description similarity only
    assert kept == ["S3", "S5"]


def test_description_similarity_raises_score():
    c = candidates([False] * 5)
    hi = c.iloc[[0]].assign(desc_cosine=0.9)
    lo = c.iloc[[0]].assign(desc_cosine=0.1)
    assert score(hi).item() > score(lo).item()
