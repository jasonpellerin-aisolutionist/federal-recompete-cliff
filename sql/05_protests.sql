-- GAO bid protests from the Fed-Spend API, keyed the way FPDS keys awards.
-- The first six characters of a solicitation number are the issuing office (DoDAAC style).
-- The feed is a live docket, not a history: most cases carry no filing or decision date and the
-- dated ones start in 2026. Undated cases therefore only count for as-of dates on or after the
-- feed's first dated case (see 20_features.sql), and protests are watchlist context, not model
-- inputs.

CREATE OR REPLACE TABLE protests AS
WITH p AS (
    SELECT
        caseNumber,
        lower(outcome) AS outcome,
        TRY_CAST(left(decisionDate, 10) AS DATE) AS decision_date,
        nullif(upper(regexp_replace(coalesce(solicitationNumber, contractNumber, ''), '[^A-Za-z0-9]', '', 'g')), '') AS solicitation_id
    FROM read_parquet('{raw_dir}/fedspend_protests.parquet')
)
SELECT *, CASE WHEN length(solicitation_id) >= 6 THEN left(solicitation_id, 6) END AS office6
FROM p;

CREATE OR REPLACE TABLE protest_feed AS
SELECT min(decision_date) AS feed_start FROM protests;
