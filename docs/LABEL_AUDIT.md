# Label audit

Every label in this project comes from a matcher that guesses which later award replaced an
ended contract. If the matcher is wrong, the model learns noise and the back-test grades it
against noise. So the matcher was audited by hand, four times, and each round changed the rules.

**Protocol.** After each labeling run, `label.py` draws a stratified random sample of 100
matched pairs (50 `retained`, 50 `changed`) with a new seed. The pairs are shuffled and
reviewed **with vendor names hidden**, showing only product codes, the time gap, and both
descriptions, so the verdict cannot be swayed by who won. Each pair gets one verdict:

- `same_requirement`: the successor is clearly the same work bought again.
- `partial`: related work (a later phase, a consolidation, the same program but a different
  line item) where "follow-on" is arguable.
- `wrong_match`: different work that merely shares an office and a product code.

The review was done during the build by the project author using this protocol; the final
round's verdicts are in [label_audit_sample.csv](label_audit_sample.csv) (column
`reviewer_verdict`) and rounds 2 and 3 in `label_audit_round2.csv` and `label_audit_round3.csv`.
Anyone can re-review the same rows.

## Rounds

| Round | Rule change | Same requirement | Partial | Wrong | Wrong among `changed` |
|---|---|---|---|---|---|
| 1 | Office + PSC/NAICS + timing + size, description as a soft bonus | not tallied | | | about 2 in 3 of the 30 `changed` pairs read |
| 2 | Exclude next orders under the same parent vehicle; require description cosine >= 0.25 | 53 | 22 | 25 | 17 of 50 |
| 3 | Require informative descriptions (3+ distinctive words), cosine >= 0.30, same state for services | 58 | 19 | 23 | 12 of 50 |
| 4 | Exclude construction (PSC Y and Z) | **66** | **17** | **17** | **14 of 50** |

What each round found:

1. **Round 1.** Office plus product code alone clears the match threshold, so bearing
   prognostics matched composites research and FBI training venues matched fire alarms. The
   same run also showed 13,052 "retained" labels that were just the next order under the same
   single-award vehicle: a continuation, not a recompete.
2. **Round 2.** The remaining errors were generic descriptions ("LABOR", "RESEARCH AND
   DEVELOPMENT", "SUPPORT COSTS") and parallel sites (guard services in Oregon versus
   Washington, court security circuit 9 versus 8). It also exposed a subtle leak: for products,
   place of performance is the *vendor's* plant, so a "same state" bonus quietly favored the
   incumbent. The bonus now applies to services only.
3. **Round 3.** Eleven of the 23 wrong matches were construction: two repair projects at the
   same base look alike but neither is a follow-on. Construction is one-time work, so it was
   scoped out.
4. **Round 4** is a fresh sample under the final rules.

## Final precision and what it means

On the final sample, 66% of matches are clearly the same requirement and 83% are at least
related. The errors are **asymmetric**: 28% of `changed` labels are wrong matches versus 6% of
`retained` labels, because a random wrong match is almost always a different vendor. High
confidence labels are cleaner (8 of 66 wrong) than medium (9 of 34).

Consequences, all carried into the [model card](MODEL_CARD.md):

- Observed change rates lean high. The 48% training change rate is an upper-leaning estimate.
- Label noise attenuates measured discrimination. On high-confidence test labels the model's
  AUC is 0.793 versus 0.754 on all labels.
- Coverage is the price of precision: 7,332 of 59,888 ended non-construction contracts get a
  label, and 6,883 of those were still undecided at the observation date and train the model.
  Contracts with boilerplate descriptions, and requirements that moved to a different office
  or code, are not labeled. The model is only claimed to apply to contracts like the labeled
  ones.

Residual error patterns worth a future rule: foreign military sales buys for different
countries, different task orders under the same multiple-award IDIQ, and vendor rollups after
mergers (L3 Technologies to L3Harris).
