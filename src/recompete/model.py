"""Back-test the Fed-Spend heuristic and train the incumbent-loss model with a temporal split.

Target: 1 if the follow-on went to a different vendor (`changed`), 0 if the incumbent kept it.
Split by the fiscal year the predecessor ended: FY2019-2023 train, FY2024 validation (early
stopping and isotonic calibration), FY2025 test (touched once, for the numbers we report).
The model that scores the FY27 watchlist is exactly the evaluated model; it is not refit.
"""

from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recompete import FIGURES_DIR, MODEL_DIR, ROOT
from recompete.db import connect
from recompete.style import apply_style, palette

SMOOTH = 10
SEED = 2027
NUMERIC = [
    "log_spend_per_year", "log_ceiling", "projected_ceiling_use", "duration_years",
    "actions_per_year", "offers", "inc_log_obligated_3y", "inc_awards_3y", "inc_offices_3y",
    "inc_office_awards_3y", "inc_office_share_3y", "office_awards_3y", "office_sector_vendors_3y",
    "naics_hhi_3y", "naics_vendors_3y", "office_recompetes_prior", "office_change_rate_prior",
    "subagency_recompetes_prior", "subagency_change_rate_prior", "inc_recompetes_prior",
    "inc_retain_rate_prior", "small_business", "is_8a", "is_sdvosb", "is_wosb", "is_hubzone",
    "national_interest",
]  # fmt: skip
CATEGORICAL = [
    "extent_competed", "set_aside_group", "pricing_group", "award_type", "vehicle",
    "commercial_item", "performance_based", "subcontracting_plan", "naics2", "psc_group", "agency",
]  # fmt: skip
LABEL_HISTORY = [
    "office_recompetes_prior", "office_change_rate_prior", "subagency_recompetes_prior",
    "subagency_change_rate_prior", "inc_recompetes_prior", "inc_retain_rate_prior",
]  # fmt: skip


def fiscal_year(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d)
    return d.dt.year + (d.dt.month >= 10).astype(int)


def add_rates(df: pd.DataFrame, prior: float) -> pd.DataFrame:
    df = df.copy()
    df["office_change_rate_prior"] = (df["office_changes_prior"] + SMOOTH * prior) / (
        df["office_recompetes_prior"] + SMOOTH
    )
    df["subagency_change_rate_prior"] = (df["subagency_changes_prior"] + SMOOTH * prior) / (
        df["subagency_recompetes_prior"] + SMOOTH
    )
    df["inc_retain_rate_prior"] = (df["inc_retained_prior"] + SMOOTH * (1 - prior)) / (
        df["inc_recompetes_prior"] + SMOOTH
    )
    for c in ["small_business", "is_8a", "is_sdvosb", "is_wosb", "is_hubzone", "national_interest"]:
        df[c] = df[c].fillna(False).astype(int)
    return df


def prepare(df: pd.DataFrame, categories: dict[str, list[str]] | None = None):
    X = df[NUMERIC + CATEGORICAL].copy()
    X["offers"] = X["offers"].astype(float)
    if categories is None:
        categories = {}
        for c in CATEGORICAL:
            counts = X[c].fillna("NA").value_counts()
            categories[c] = sorted(counts[counts >= 50].index.tolist())
    for c in CATEGORICAL:
        v = X[c].fillna("NA")
        X[c] = pd.Categorical(
            v.where(v.isin(categories[c]), "OTHER"), categories=[*categories[c], "OTHER"]
        )
    return X, categories


def load_labeled(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT t.*, f.outcome, f.confidence, a.end_date, a.piid, a.recipient_name, a.office
        FROM train_features t
        JOIN follow_on f ON f.pred_key = t.award_key
        JOIN awards a ON a.award_key = t.award_key
        JOIN awards s ON s.award_key = f.succ_key
        WHERE f.outcome IN ('retained', 'changed')
          -- A follow-on awarded before the observation date is already decided, not forecast.
          AND s.base_date >= t.as_of
    """).df()
    df["y"] = (df["outcome"] == "changed").astype(int)
    df["fy"] = fiscal_year(df["end_date"])
    return df


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))  # fmt: skip


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    k = max(1, int(len(p) * 0.1))
    top = np.argsort(-p)[:k]
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "ece": ece(y, p),
        "top_decile_precision": float(y[top].mean()),
        "top_decile_lift": float(y[top].mean() / y.mean()),
    }


def bootstrap_auc_gap(y, p_a, p_b, n: int = 1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    gaps = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        gaps.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(np.mean(gaps)), float(lo), float(hi)


def fit_lgbm(Xtr, ytr, Xva, yva, features):
    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=SEED, verbose=-1,
    )  # fmt: skip
    model.fit(Xtr[features], ytr, eval_set=[(Xva[features], yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])  # fmt: skip
    return model


def fit_logistic(Xtr, ytr):
    num = [c for c in NUMERIC]
    pre = ColumnTransformer([
        ("num", make_pipeline(StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])  # fmt: skip
    model = make_pipeline(pre, LogisticRegression(C=0.5, max_iter=3000))
    Xn = Xtr.copy()
    Xn[num] = Xn[num].fillna(-1)
    for c in CATEGORICAL:
        Xn[c] = Xn[c].astype(str)
    model.fit(Xn, ytr)
    return model


def logistic_predict(model, X):
    Xn = X.copy()
    Xn[NUMERIC] = Xn[NUMERIC].fillna(-1)
    for c in CATEGORICAL:
        Xn[c] = Xn[c].astype(str)
    return model.predict_proba(Xn)[:, 1]


def subgroup_table(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    t = test.assign(p=p)
    t["size_band"] = pd.cut(np.exp(t["log_ceiling"]), [0, 25e6, 100e6, 1e9, np.inf],
                            labels=["$5-25M", "$25-100M", "$100M-1B", "$1B+"])  # fmt: skip
    t["incumbent_small"] = np.where(t["small_business"] == 1, "small business", "other than small")
    top_agencies = t["agency"].value_counts().head(8).index
    t["agency_band"] = t["agency"].where(t["agency"].isin(top_agencies), "All other agencies")
    rows = []
    for dim in ["agency_band", "incumbent_small", "set_aside_group", "size_band", "confidence"]:
        for g, d in t.groupby(dim, observed=True):
            if len(d) < 100 or d["y"].nunique() < 2:
                continue
            rows.append({"dimension": dim, "group": str(g), "n": len(d), "observed_change_rate": d["y"].mean(),
                         "mean_predicted": d["p"].mean(), "auc": roc_auc_score(d["y"], d["p"])})  # fmt: skip
    return pd.DataFrame(rows)


def reliability_figure(y, curves: dict[str, np.ndarray], path) -> None:
    apply_style()
    colors = palette()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1, ls="--", label="Perfect calibration")
    for (name, p), color in zip(curves.items(), colors, strict=False):
        q = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).mean()
        ax.plot(g["p"], g["y"], marker="o", color=color, label=name)
    ax.set_xlabel("Predicted probability the incumbent loses")
    ax.set_ylabel("Observed share that changed hands")
    ax.set_title("Calibration on FY2025 recompetes (held out)")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def shap_figure(model, X, features, path) -> pd.DataFrame:
    apply_style()
    sample = X[features].sample(min(5000, len(X)), random_state=SEED)
    values = shap.TreeExplainer(model).shap_values(sample)
    values = values[1] if isinstance(values, list) else values
    imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.barh(top["feature"], top["mean_abs_shap"], color=palette()[0])
    ax.set_xlabel("Mean |SHAP| (log-odds of the incumbent losing)")
    ax.set_title("What drives the model")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return imp


def main() -> None:
    con = connect()
    df = load_labeled(con)
    train, valid, test = (
        df[df["fy"].between(2019, 2023)],
        df[df["fy"] == 2024],
        df[df["fy"] == 2025],
    )
    prior = float(train["y"].mean())
    train, valid, test = (add_rates(d, prior) for d in (train, valid, test))
    Xtr, cats = prepare(train)
    Xva, _ = prepare(valid, cats)
    Xte, _ = prepare(test, cats)
    ytr, yva, yte = train["y"].to_numpy(), valid["y"].to_numpy(), test["y"].to_numpy()
    features = NUMERIC + CATEGORICAL
    print(
        f"rows train {len(train):,} valid {len(valid):,} test {len(test):,}; base rate {prior:.3f}"
    )

    gbm = fit_lgbm(Xtr, ytr, Xva, yva, features)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(gbm.predict_proba(Xva[features])[:, 1], yva)
    p_gbm_raw = gbm.predict_proba(Xte[features])[:, 1]
    p_gbm = iso.predict(p_gbm_raw)

    logit = fit_logistic(Xtr, ytr)
    p_logit = logistic_predict(logit, Xte)

    heur_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    heur_iso.fit(valid["heuristic_score"] / 100, yva)
    p_heur_raw = (test["heuristic_score"] / 100).to_numpy()
    p_heur = heur_iso.predict(p_heur_raw)

    no_hist = [f for f in features if f not in LABEL_HISTORY]
    gbm_no_hist = fit_lgbm(Xtr, ytr, Xva, yva, no_hist)
    p_no_hist = gbm_no_hist.predict_proba(Xte[no_hist])[:, 1]

    results = {
        "base_rate_train": prior,
        "test_change_rate": float(yte.mean()),
        "n": {"train": len(train), "valid": len(valid), "test": len(test)},
        "best_iteration": int(gbm.best_iteration_),
        "models": {
            "Fed-Spend heuristic (as shipped, score/100)": metrics(yte, p_heur_raw),
            "Fed-Spend heuristic (recalibrated)": metrics(yte, p_heur),
            "Logistic regression": metrics(yte, p_logit),
            "LightGBM (calibrated)": metrics(yte, p_gbm),
            "LightGBM without label-history features (leakage check)": metrics(yte, p_no_hist),
        },
        "auc_gap_lgbm_vs_heuristic": bootstrap_auc_gap(yte, p_gbm, p_heur_raw),
        "auc_gap_lgbm_vs_logistic": bootstrap_auc_gap(yte, p_gbm, p_logit),
        "high_confidence_only": metrics(
            yte[test["confidence"].eq("high").to_numpy()],
            p_gbm[test["confidence"].eq("high").to_numpy()],
        ),  # fmt: skip
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reliability_figure(yte, {"Fed-Spend heuristic (as shipped)": p_heur_raw, "Logistic regression": p_logit,
                             "LightGBM (calibrated)": p_gbm}, FIGURES_DIR / "calibration_fy2025.png")  # fmt: skip
    imp = shap_figure(gbm, Xte, features, FIGURES_DIR / "shap_importance.png")
    subgroups = subgroup_table(test, p_gbm)

    reports = ROOT / "reports"
    (reports / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    imp.to_csv(reports / "feature_importance.csv", index=False)
    subgroups.to_csv(reports / "subgroup_metrics.csv", index=False)
    test.assign(p_model=p_gbm, p_heuristic=p_heur_raw)[
        ["award_key", "piid", "fy", "y", "confidence", "p_model", "p_heuristic", "agency"]
    ].to_parquet(reports / "test_predictions.parquet", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": gbm, "calibrator": iso, "features": features, "categories": cats,
                 "prior": prior, "trained_on": "FY2019-FY2023", "calibrated_on": "FY2024"},
                MODEL_DIR / "incumbent_loss.joblib")  # fmt: skip

    for name, m in results["models"].items():
        print(
            f"{name:52s} AUC {m['auc']:.3f}  Brier {m['brier']:.3f}  ECE {m['ece']:.3f}  top-10% {m['top_decile_precision']:.2f}"
        )
    g = results["auc_gap_lgbm_vs_heuristic"]
    print(f"AUC gain over heuristic: {g[0]:+.3f} (95% CI {g[1]:+.3f} to {g[2]:+.3f})")


if __name__ == "__main__":
    main()
