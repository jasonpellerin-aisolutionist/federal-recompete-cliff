-- Predecessors: contracts of $5M+ that ran at least six months and ended in FY2019 to FY2025.
-- Construction (PSC Y and Z) is out of scope: those are one-time projects, not recompeted
-- requirements, and they were the largest source of false matches in the label audit.
-- Candidates: later awards from the same contracting office for the same kind of work that
-- started between nine months before and twelve months after the predecessor ended, and that
-- were not issued under the predecessor's own parent vehicle.
-- Vendor identity is carried along for labeling but never used to pick the match.

CREATE OR REPLACE TABLE predecessors AS
SELECT *
FROM awards
WHERE obligated >= 5e6
  AND end_date BETWEEN DATE '2018-10-01' AND DATE '2025-09-30'
  AND NOT regexp_matches(coalesce(psc, ''), '^[YZ]')
  AND coalesce(start_date, base_date) IS NOT NULL
  AND date_diff('day', coalesce(start_date, base_date), end_date) >= 180;

CREATE OR REPLACE TABLE candidates AS
SELECT
    p.award_key                                             AS pred_key,
    s.award_key                                             AS succ_key,
    date_diff('day', p.end_date, s.base_date)               AS gap_days,
    s.psc = p.psc                                           AS psc_match,
    s.naics = p.naics                                       AS naics_match,
    regexp_matches(coalesce(p.psc, ''), '^[A-Z]')           AS is_service,
    coalesce(s.pop_state = p.pop_state, false)              AS pop_match,
    coalesce(s.pop_state <> p.pop_state, false)             AS pop_conflict,
    ln(greatest(coalesce(s.potential_value, 0), s.obligated, 1)
       / greatest(coalesce(p.potential_value, 0), p.obligated, 1)) AS size_logratio,
    coalesce(s.vendor_id = p.vendor_id, false)
        OR coalesce(s.recipient_uei = p.recipient_uei, false)
        OR coalesce(length(s.vendor_name_norm) > 3 AND s.vendor_name_norm = p.vendor_name_norm, false) AS same_vendor
FROM predecessors p
JOIN awards s
  ON s.office_code = p.office_code
 AND s.award_key <> p.award_key
 AND s.base_date BETWEEN p.end_date - INTERVAL 270 DAY AND p.end_date + INTERVAL 365 DAY
 AND s.base_date > coalesce(p.base_date, p.start_date) + INTERVAL 180 DAY
 AND (s.psc = p.psc OR s.naics = p.naics)
 -- The next order under the same parent vehicle is a continuation, not a recompete (99% of
 -- same-vehicle "follow-ons" in FY2019-2025 went to the same vendor).
 AND NOT coalesce(s.parent_piid = p.parent_piid, false)
 AND NOT coalesce(s.parent_piid = p.piid, false)
WHERE abs(ln(greatest(coalesce(s.potential_value, 0), s.obligated, 1)
             / greatest(coalesce(p.potential_value, 0), p.obligated, 1))) <= ln(10);
