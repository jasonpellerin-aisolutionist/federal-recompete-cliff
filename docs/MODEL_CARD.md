# Model card: FY27 incumbent-loss model

Structured after *Model Cards for Model Reporting* (Mitchell et al., 2019) and mapped to the
four functions of the NIST AI Risk Management Framework (AI RMF 1.0: Govern, Map, Measure,
Manage). Numbers are from `reports/metrics.json`, produced by `make train`.

## Summary

| | |
|---|---|
| Model | LightGBM binary classifier, isotonic calibration (`models/incumbent_loss.joblib`) |
| Predicts | Probability that a federal contract's follow-on award goes to a different vendor than the incumbent, given the requirement is bought again as a new award |
| Unit | One prime contract or order of $5M+ obligated, at least 180 days long, not construction |
| Observed | 180 days before the contract's end date (training); the run date (watchlist) |
| Trained | 4,478 recompetes that ended FY2019 to FY2023 |
| Calibrated | 1,279 recompetes that ended FY2024 (early stopping and isotonic fit) |
| Tested | 1,126 recompetes that ended FY2025, scored once |
| Owner | Jason Pellerin, [jasonpellerin.com](https://www.jasonpellerin.com) |

## Map (context and intended use)

**Intended use.** Prioritizing research: a capture analyst, small business, or agency
market-research team asking which expiring contracts are most likely to be genuinely open
next year. The output is a ranked list with reasons, not a decision.

**Out of scope.**

- Predicting any specific agency's source-selection decision or evaluating any vendor's
  performance. The model sees no CPARS ratings, proposals, prices, or protest merits.
- Construction projects, contracts under $5M, and requirements that move to a different
  office or product code (not represented in the labels).
- Adverse action against an incumbent. A high score says the award pattern resembles past
  vendor changes; it says nothing about the incumbent's quality.

**Stakeholders.** Incumbent contractors (a public "likely to lose" list can affect
reputations), challengers, contracting offices, and the public. Mitigations: only public
award facts appear; the app states that scores are estimates of patterns, not judgments; the
Fed-Spend heuristic is shown alongside for contrast.

## Measure (performance and evaluation)

Held-out FY2025 test set, 1,126 recompetes, 50.4% changed hands.

| Model | AUC | PR AUC | Brier | ECE | Top-decile precision |
|---|---|---|---|---|---|
| Fed-Spend heuristic, as shipped (score / 100) | 0.585 | 0.559 | 0.266 | 0.151 | 53% |
| Fed-Spend heuristic, recalibrated on FY2024 | 0.597 | 0.565 | 0.246 | 0.058 | 54% |
| Logistic regression (baseline) | 0.743 | 0.722 | 0.209 | 0.059 | 80% |
| **LightGBM, calibrated (shipped)** | **0.754** | **0.725** | **0.204** | **0.053** | **88%** |
| LightGBM without label-history features | 0.756 | 0.751 | 0.202 | 0.052 | 90% |
| Shipped model, high-confidence labels only (n = 705) | 0.793 | 0.728 | 0.186 | 0.045 | 87% |

- **Versus the heuristic:** AUC +0.169, 95% bootstrap CI +0.131 to +0.207 (1,000 resamples).
  The heuristic here is a faithful reimplementation of Fed-Spend's production formula on the
  signals public data supports (offers, ceiling use, modification churn, competition posture);
  CPARS, value growth, set-aside change, and protest signals abstain, as they do in production
  when that data is missing. The comparison is to the heuristic as it scores these contracts,
  not to a version with CPARS access.
- **Versus logistic regression:** AUC +0.012, 95% CI -0.002 to +0.025. The gain is not
  statistically distinguishable from zero; LightGBM is shipped for its better top-decile
  precision and calibration, but a transparent logistic model is nearly as good.
- **Leakage check.** Removing every feature derived from past labels (office, sub-agency, and
  incumbent change rates) does not reduce performance, so the back-test is not being carried
  by a label-history leak. That model was not chosen after seeing the test result: the shipped
  model was fixed before the test run.
- **Calibration.** [Reliability curve](../reports/figures/calibration_fy2025.png). The
  shipped heuristic is over-confident at the top (contracts it scores near 80 changed hands
  about half the time).
- **Drivers.** [SHAP importance](../reports/figures/shap_importance.png): how the last award
  was competed, the office's history of changing vendors, single versus multiple-award
  vehicle, the incumbent's share of the office's spend, and the incumbent's record.

**Subgroups** (`reports/subgroup_metrics.csv`, groups under 100 test rows omitted):

| Group | n | Observed change rate | Mean predicted | AUC |
|---|---|---|---|---|
| Department of Defense | 658 | 51% | 47% | 0.786 |
| All other agencies (outside top 8) | 108 | 52% | 53% | 0.604 |
| Incumbent other than small | 644 | 43% | 44% | 0.775 |
| Incumbent small business | 482 | 61% | 60% | 0.693 |
| Contract $5-25M | 874 | 51% | 53% | 0.757 |
| Contract $25-100M | 194 | 52% | 45% | 0.731 |
| High-confidence labels | 705 | 45% | 50% | 0.793 |
| Medium-confidence labels | 421 | 59% | 52% | 0.687 |

The model is weaker for small-business incumbents and for smaller agencies, and under-predicts
change on $25-100M contracts. These are reported, not tuned away.

## Data (datasheet)

- **Source.** USAspending.gov bulk prime award summaries, contracts only (award types A, B,
  C, D), $1M+ obligated, signed FY2012 to FY2026: 554,649 awards, 46,843 vendors after parent
  rollup, plus every award modified in the last two years (active contracts signed before
  FY2012; used for scoring, not in the evaluated training set). Public domain; no personal
  data beyond business names and UEIs.
- **Labels.** Built by the follow-on matcher in `label.py`; see
  [METHODOLOGY.md](METHODOLOGY.md) and [LABEL_AUDIT.md](LABEL_AUDIT.md). Final blind audit:
  66% same requirement, 17% related, 17% wrong match; 28% of `changed` labels and 6% of
  `retained` labels are wrong matches.
- **Protests.** Fed-Spend's GAO protest feed is a live docket (1,525 of 1,605 cases undated),
  so it is watchlist context only, never a model input.
- **Known gaps.** No CPARS (not public). Award summaries carry final values, so contract size
  enters as rates (spend per year, projected ceiling use). DoD reports to FPDS with a 90-day
  delay, so FY2025 follow-ons awarded in the last quarter may be missing, which can shift
  late FY2025 labels toward `no_follow_on`. Parent-entity rollup misses some post-merger names.

## Govern (accountability and change control)

- **Versioning.** The model is a committed artifact. The weekly refresh rescores with it and
  never retrains; a new model is a reviewed pull request that reruns `make train` and updates
  this card.
- **Reproducibility.** `make download build label features train score` rebuilds everything
  from public data. Secrets (the Fed-Spend API key) are injected at runtime from Infisical or
  GitHub Actions secrets and never written to disk.
- **Transparency.** Every watchlist row carries its top three drivers in plain language and a
  link to the public award record.

## Manage (monitoring and response)

- **Weekly.** GitHub Actions pulls the last 14 days of modified awards, rescores, and posts a
  digest (new top-band entries, biggest moves, top-band contracts ending within 90 days) to an
  n8n webhook that emails it. n8n also alerts if no digest arrives within eight days.
- **Drift.** Each refresh changes the as-of date; contracts whose end dates slip out of FY2027
  drop off and are counted in the digest.
- **Retraining trigger.** When FY2026 recompetes are labelable (follow-ons visible through
  about Q2 FY2027), retrain on FY2019 to FY2024, calibrate on FY2025, test on FY2026, and
  compare against this card before replacing the artifact.
- **Known failure modes.** Foreign military sales buys for different countries, different
  task orders under the same multiple-award IDIQ, and vendor mergers can produce wrong labels;
  scores for contracts resembling those patterns deserve extra skepticism.
