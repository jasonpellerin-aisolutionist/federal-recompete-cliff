# Methodology

The question: of the federal contracts whose current period of performance ends in FY2027
(October 1, 2026 to September 30, 2027), which are most likely to change hands when they are
recompeted? Federal records do not say "this award replaced that one," so the project first has
to build its own ground truth, then test whether anything observable six months before a
contract ends predicts the outcome.

## 1. Data

| Source | What it gives | Access |
|---|---|---|
| USAspending.gov bulk award download (`/api/v2/download/awards/`) | Every prime contract (award types A, B, C, D) of $1M+ signed FY2012 to FY2026: 60 of the 286 summary columns (dates, values, office, NAICS, PSC, competition, offers, set-aside, recipient and parent UEI, description) | Public, no key |
| Fed-Spend API (key-authenticated) | 1,605 GAO bid protest cases with solicitation and contract numbers, filing and decision dates, outcome | Key held in Infisical, injected at runtime |
| Fed-Spend API (key-authenticated) | Fed-Spend's recompete radar with its production vulnerability score (local comparison only, not published) | Same |

Raw pulls are cached as parquet under `data/raw/` (not committed; `make download` rebuilds them)
and loaded into DuckDB by `sql/00_awards.sql`, which types every column, normalizes
solicitation ids, keeps the latest record per award key, and builds a `vendor_id` that rolls
subsidiaries up to their parent UEI.

The bulk file returns assistance awards alongside contracts when filtered by the newer
prime-and-sub award type parameter; `download.py` requests `award_type_codes` and keeps only
`CONT_AWD_` keys, which reconciles each fiscal year to the count endpoint exactly (FY2012:
35,475 of 35,475).

Signing-date pulls cannot see contracts signed before FY2012 that are still running. A
backfill (`make backfill`) pulls every award modified in the last two years, in nine 90-day
slices because a single two-year export does not finish, and the weekly refresh adds each
fortnight's modifications. The backfill added 61 FY27 contracts carrying $176.1B, mostly
long-running Energy laboratory, Defense, and NASA contracts. The model and its test metrics
were produced before the backfill and were not retrained on it.

## 2. Ground truth: who won the follow-on?

For every predecessor contract ($5M+ obligated, at least 180 days long, ended FY2019 to
FY2025, not construction), `sql/10_candidates.sql` collects candidate follow-ons: awards from
the **same contracting office**, for the **same PSC or NAICS**, whose base date falls between
270 days before and 365 days after the predecessor ended, whose size is within a factor of
ten, and that were **not issued under the predecessor's own parent vehicle** (the next order on
the same vehicle is a continuation, and 99% of them went to the same vendor).

`label.py` keeps only candidates whose descriptions are informative (three or more distinctive
words) and similar (TF-IDF cosine of at least 0.30), and, for services, performed in the same
state. It then scores them on requirement similarity only:

```
4.0 * TF-IDF description cosine + 1.0 * same PSC + 0.5 * same NAICS
+ 0.5 * same place-of-performance state (services only)
- 0.75 * |log size ratio| / log 10 - 0.5 * |gap - 30 days| / 365
```

For products, place of performance is the vendor's plant, so it is never used: it would let
vendor identity leak into the pick. Each of these rules came out of a blind hand audit; see
[LABEL_AUDIT.md](LABEL_AUDIT.md). The top-scoring candidate is the follow-on. **Vendor identity is never an input to the pick**
(a unit test flips every candidate's vendor flag and asserts identical matches), so the matcher
cannot manufacture "retained" labels by preferring the incumbent. Only after the pick is the
outcome read: `retained` if the follow-on went to the same parent entity, UEI, or normalized
name; `changed` otherwise; `no_follow_on` if no candidate scores at least 2.0. A follow-on
awarded before the observation date (below) was already decided, so those rows are dropped
from training rather than "forecast".

Each label carries a confidence: `high` when the match scores 3.0 or more and beats the best
candidate with the opposite outcome by 0.75 or more, `medium` otherwise. A stratified sample
of 100 pairs (50 retained, 50 changed) is written to `docs/label_audit_sample.csv` for manual
review; results are in [LABEL_AUDIT.md](LABEL_AUDIT.md).

Only `retained` and `changed` rows train the model. `no_follow_on` covers work that was
cancelled, insourced, consolidated into a larger vehicle, or recompeted under a different
office or code; it is reported, not modeled.

## 3. Point-in-time features

Every feature is computed **as of 180 days before the predecessor ended**, a realistic capture
lead time. `sql/20_features.sql` takes `(award_key, asof)` pairs, so training rows and the live
FY2027 watchlist (as of the run date) go through identical SQL. History features only count
events whose outcome was knowable before `asof`:

- **Contract**: value, ceiling use, duration, actions per year, pricing, competition extent,
  number of offers, set-aside, vehicle type, business size, socioeconomic flags.
- **Incumbent**: three-year award count and dollars, share of the office's spend, prior
  recompete record (smoothed rate).
- **Office and sub-agency**: three-year spend and vendor count, historical change rate
  (smoothed toward the global mean with m = 10 pseudo-observations).
- **Market**: NAICS concentration (HHI) over the prior three years.
- **Protests (watchlist context only)**: GAO filings against the same solicitation or office.
  The Fed-Spend protest feed is a live docket: 1,525 of its 1,605 cases carry no filing or
  decision date and every dated case is from 2026. Counting them for historical contracts would
  either be all zeros or leak today's docket into the past, so protests are excluded from the
  model and shown only on the live watchlist.

The smoothed historical rates are the most likely leakage path (a rate built from the future
would inflate the back-test), so the model is also trained without them as a check. It scores
the same (AUC 0.756 versus 0.754), so the result does not depend on them; see
[MODEL_CARD.md](MODEL_CARD.md).

## 4. Baseline: Fed-Spend's production heuristic

`features.fedspend_heuristic` reimplements Fed-Spend's production vulnerability score: a
weighted mean of 0-100 votes from offers, ceiling use, contract
protests, office protest climate, modification churn, and competition posture. CPARS, value
growth, and set-aside change need data USAspending does not publish, so those three signals
abstain, exactly as they do in production when data is missing; the two protest signals also
abstain in the back-test for the reason above. Scoring the heuristic on the
same labeled back-test is the honest question this project exists to answer: does the score
Fed-Spend already ships separate winners from losers?

## 5. Model and evaluation

- **Temporal split**: train on predecessors that ended FY2019 to FY2023, validate on FY2024
  (early stopping and isotonic calibration), test once on FY2025. No random shuffling across
  years.
- **Models**: L2 logistic regression (one-hot, standardized) as the transparent baseline,
  LightGBM with native categoricals as the candidate, the Fed-Spend heuristic as the incumbent
  method.
- **Metrics**: ROC AUC with 1,000-sample bootstrap confidence intervals, PR AUC, Brier score,
  log loss, expected calibration error, and precision and lift in the top decile (the list a
  capture team would actually work).
- **Fairness and robustness**: metrics by agency, business size, set-aside status, and
  contract size band, with small groups flagged rather than hidden.
- **Explanations**: SHAP values for global importance and per-contract top three drivers,
  shown in plain language in the app.

The evaluated model is the shipped model. The weekly job rescoring the watchlist never
retrains; a new model is a reviewed code change.

## 6. Watchlist and refresh

`score.py` scores every $5M+ contract whose current end date falls in FY2027, assigns a risk
band by quantile, and attaches the top three SHAP drivers. Fed-Spend's own production scores
are used locally for comparison and are not published. Every Monday a GitHub Action pulls the last 14 days of modified awards and
the Fed-Spend feeds, rebuilds features, rescores, commits the app data, and posts a digest of
what moved to an n8n webhook that emails it. n8n also runs a Tuesday watchdog that alerts if no
digest arrived in eight days.
