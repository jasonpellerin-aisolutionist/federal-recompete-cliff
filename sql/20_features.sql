-- Point-in-time features for every row of `score_base(award_key, as_of)`.
--
-- Contract size and churn enter as rates (spend per year, projected ceiling use, actions per
-- year) so a finished contract and a live one read the same way. Every history feature
-- (incumbent, office, market, protests, past outcomes) only counts events before `as_of`.

CREATE OR REPLACE TABLE features AS
WITH base AS (
    SELECT
        b.award_key,
        b.as_of,
        a.*  EXCLUDE (award_key),
        greatest(date_diff('day', coalesce(a.start_date, a.base_date), a.end_date), 1) / 365.25 AS duration_years,
        greatest(date_diff('day', coalesce(a.start_date, a.base_date), least(a.end_date, b.as_of + INTERVAL 180 DAY)), 90) / 365.25 AS elapsed_years
    FROM score_base b
    JOIN awards a USING (award_key)
),
contract AS (
    SELECT
        award_key, as_of, vendor_id, office_code, sub_agency_code, naics, psc, solicitation_id,
        ln(greatest(obligated / elapsed_years, 1))                                   AS log_spend_per_year,
        ln(greatest(coalesce(potential_value, obligated), 1))                        AS log_ceiling,
        least(obligated / elapsed_years * duration_years
              / greatest(coalesce(potential_value, obligated), 1), 3)                AS projected_ceiling_use,
        duration_years,
        coalesce(n_actions, 0) / elapsed_years                                       AS actions_per_year,
        offers,
        coalesce(extent_competed, 'UNK')                                             AS extent_competed,
        CASE
            WHEN set_aside IS NULL OR set_aside IN ('NONE', 'N/A') THEN 'none'
            WHEN set_aside LIKE '8A%' OR set_aside LIKE '8AN%' THEN '8a'
            WHEN set_aside LIKE 'SDVOSB%' THEN 'sdvosb'
            WHEN set_aside LIKE 'HZ%' THEN 'hubzone'
            WHEN set_aside LIKE 'WOSB%' OR set_aside LIKE 'EDWOSB%' THEN 'wosb'
            WHEN set_aside LIKE 'SBA%' OR set_aside LIKE 'SBP%' THEN 'small_business'
            ELSE 'other'
        END                                                                          AS set_aside_group,
        CASE
            WHEN pricing IN ('J', 'K', 'L', 'M') THEN 'fixed_price'
            WHEN pricing IN ('R', 'S', 'T', 'U', 'V') THEN 'cost_plus'
            WHEN pricing IN ('Y', 'Z') THEN 'time_materials'
            ELSE 'other'
        END                                                                          AS pricing_group,
        award_type,
        coalesce(parent_single_multiple, 'NONE')                                     AS vehicle,
        coalesce(commercial_item, 'UNK')                                             AS commercial_item,
        coalesce(performance_based, 'UNK')                                           AS performance_based,
        coalesce(subcontracting_plan, 'UNK')                                         AS subcontracting_plan,
        national_interest IS NOT NULL                                                AS national_interest,
        left(naics, 2)                                                               AS naics2,
        CASE WHEN regexp_matches(left(psc, 1), '[0-9]') THEN 'product' ELSE left(psc, 1) END AS psc_group,
        agency,
        business_size = 'SMALL BUSINESS'                                             AS small_business,
        coalesce(is_8a, false)                                                       AS is_8a,
        coalesce(is_sdvosb, false)                                                   AS is_sdvosb,
        coalesce(is_wosb, false)                                                     AS is_wosb,
        coalesce(is_hubzone, false)                                                  AS is_hubzone
    FROM base
),
incumbent AS (
    SELECT
        c.award_key,
        ln(1 + coalesce(sum(h.obligated), 0))                                        AS inc_log_obligated_3y,
        count(h.award_key)                                                           AS inc_awards_3y,
        count(DISTINCT h.office_code)                                                AS inc_offices_3y,
        count(h.award_key) FILTER (WHERE h.office_code = c.office_code)              AS inc_office_awards_3y,
        coalesce(sum(h.obligated) FILTER (WHERE h.office_code = c.office_code), 0)   AS inc_office_obligated_3y
    FROM contract c
    LEFT JOIN awards h
      ON h.vendor_id = c.vendor_id
     AND h.base_date >= c.as_of - INTERVAL 3 YEAR
     AND h.base_date < c.as_of
     AND h.award_key <> c.award_key
    GROUP BY c.award_key
),
office AS (
    SELECT
        c.award_key,
        count(h.award_key)                                                           AS office_awards_3y,
        coalesce(sum(h.obligated), 0)                                                AS office_obligated_3y,
        count(DISTINCT h.vendor_id) FILTER (WHERE left(h.naics, 2) = c.naics2)       AS office_sector_vendors_3y
    FROM contract c
    LEFT JOIN awards h
      ON h.office_code = c.office_code
     AND h.base_date >= c.as_of - INTERVAL 3 YEAR
     AND h.base_date < c.as_of
    GROUP BY c.award_key
),
market AS (
    -- Vendor concentration (HHI, 0-10000) in the contract's 6-digit NAICS over the prior 3 years.
    SELECT award_key, coalesce(sum(share * share) * 10000, 10000) AS naics_hhi_3y, count(*) AS naics_vendors_3y
    FROM (
        SELECT c.award_key, h.vendor_id,
               sum(h.obligated) / sum(sum(h.obligated)) OVER (PARTITION BY c.award_key) AS share
        FROM contract c
        JOIN awards h
          ON h.naics = c.naics
         AND h.base_date >= c.as_of - INTERVAL 3 YEAR
         AND h.base_date < c.as_of
        GROUP BY c.award_key, h.vendor_id
    )
    GROUP BY award_key
),
events AS (
    -- Past recompete outcomes, each visible only once its follow-on had started.
    SELECT p.office_code, p.sub_agency_code, p.vendor_id, s.base_date AS known_at,
           (f.outcome = 'changed')::INTEGER AS changed
    FROM follow_on f
    JOIN awards p ON p.award_key = f.pred_key
    JOIN awards s ON s.award_key = f.succ_key
    WHERE f.outcome IN ('retained', 'changed')
),
office_hist AS (
    SELECT c.award_key, count(*) AS office_recompetes_prior, sum(e.changed) AS office_changes_prior
    FROM contract c JOIN events e ON e.office_code = c.office_code AND e.known_at < c.as_of
    GROUP BY c.award_key
),
subagency_hist AS (
    SELECT c.award_key, count(*) AS subagency_recompetes_prior, sum(e.changed) AS subagency_changes_prior
    FROM contract c JOIN events e ON e.sub_agency_code = c.sub_agency_code AND e.known_at < c.as_of
    GROUP BY c.award_key
),
vendor_hist AS (
    SELECT c.award_key, count(*) AS inc_recompetes_prior, count(*) - sum(e.changed) AS inc_retained_prior
    FROM contract c JOIN events e ON e.vendor_id = c.vendor_id AND e.known_at < c.as_of
    GROUP BY c.award_key
),
protests AS (
    SELECT
        c.award_key,
        count(g.caseNumber) FILTER (WHERE g.office6 = c.office_code)                                      AS office_protests_3y,
        count(g.caseNumber) FILTER (WHERE g.office6 = c.office_code AND g.outcome = 'sustained')          AS office_protests_sustained_3y,
        count(g.caseNumber) FILTER (WHERE g.solicitation_id = c.solicitation_id)                          AS contract_protests
    FROM contract c
    CROSS JOIN protest_feed f
    LEFT JOIN protests g
      ON (g.decision_date >= c.as_of - INTERVAL 3 YEAR AND g.decision_date < c.as_of
          OR g.decision_date IS NULL AND c.as_of >= f.feed_start)
     AND (g.office6 = c.office_code OR g.solicitation_id = c.solicitation_id)
    GROUP BY c.award_key
)
SELECT
    c.*,
    i.* EXCLUDE (award_key),
    CASE WHEN o.office_obligated_3y > 0 THEN i.inc_office_obligated_3y / o.office_obligated_3y ELSE 0 END AS inc_office_share_3y,
    o.* EXCLUDE (award_key),
    coalesce(m.naics_hhi_3y, 10000) AS naics_hhi_3y,
    coalesce(m.naics_vendors_3y, 0) AS naics_vendors_3y,
    coalesce(oh.office_recompetes_prior, 0)    AS office_recompetes_prior,
    coalesce(oh.office_changes_prior, 0)       AS office_changes_prior,
    coalesce(sh.subagency_recompetes_prior, 0) AS subagency_recompetes_prior,
    coalesce(sh.subagency_changes_prior, 0)    AS subagency_changes_prior,
    coalesce(vh.inc_recompetes_prior, 0)       AS inc_recompetes_prior,
    coalesce(vh.inc_retained_prior, 0)         AS inc_retained_prior,
    coalesce(g.office_protests_3y, 0)          AS office_protests_3y,
    coalesce(g.office_protests_sustained_3y, 0) AS office_protests_sustained_3y,
    coalesce(g.contract_protests, 0)           AS contract_protests
FROM contract c
JOIN incumbent i USING (award_key)
JOIN office o USING (award_key)
LEFT JOIN market m USING (award_key)
LEFT JOIN office_hist oh USING (award_key)
LEFT JOIN subagency_hist sh USING (award_key)
LEFT JOIN vendor_hist vh USING (award_key)
LEFT JOIN protests g USING (award_key);
