-- One typed row per prime contract award, deduplicated on the award key.
-- Source: data/raw/awards_*.parquet (USAspending Advanced Search downloads, obligation >= $1M):
-- one file per signed fiscal year plus weekly "modified in the last N days" pulls. The newest
-- version of each award key wins, so extensions and new obligations replace stale rows.

CREATE OR REPLACE TABLE awards AS
WITH raw AS (
    SELECT * FROM read_parquet('{raw_dir}/awards_*.parquet', union_by_name = true)
),
typed AS (
    SELECT
        contract_award_unique_key                                   AS award_key,
        award_id_piid                                               AS piid,
        nullif(parent_award_id_piid, '')                            AS parent_piid,
        award_type_code                                             AS award_type,
        TRY_CAST(total_obligated_amount AS DOUBLE)                  AS obligated,
        TRY_CAST(current_total_value_of_award AS DOUBLE)            AS current_value,
        TRY_CAST(potential_total_value_of_award AS DOUBLE)          AS potential_value,
        TRY_CAST(award_base_action_date AS DATE)                    AS base_date,
        TRY_CAST(period_of_performance_start_date AS DATE)          AS start_date,
        TRY_CAST(period_of_performance_current_end_date AS DATE)    AS end_date,
        TRY_CAST(left(period_of_performance_potential_end_date, 10) AS DATE) AS potential_end_date,
        TRY_CAST(left(last_modified_date, 19) AS TIMESTAMP)         AS last_modified,
        awarding_agency_code                                        AS agency_code,
        awarding_agency_name                                        AS agency,
        awarding_sub_agency_code                                    AS sub_agency_code,
        awarding_sub_agency_name                                    AS sub_agency,
        upper(awarding_office_code)                                 AS office_code,
        awarding_office_name                                        AS office,
        nullif(upper(recipient_uei), '')                            AS recipient_uei,
        recipient_name,
        nullif(upper(recipient_parent_uei), '')                     AS parent_uei,
        recipient_parent_name                                       AS parent_name,
        primary_place_of_performance_state_code                     AS pop_state,
        naics_code                                                  AS naics,
        naics_description,
        upper(product_or_service_code)                              AS psc,
        product_or_service_code_description                         AS psc_description,
        upper(extent_competed_code)                                 AS extent_competed,
        solicitation_procedures_code                                AS solicitation_procedures,
        other_than_full_and_open_competition_code                   AS not_competed_reason,
        TRY_CAST(number_of_offers_received AS INTEGER)              AS offers,
        nullif(type_of_set_aside_code, '')                          AS set_aside,
        type_of_contract_pricing_code                               AS pricing,
        nullif(upper(regexp_replace(solicitation_identifier, '[^A-Za-z0-9]', '', 'g')), '') AS solicitation_id,
        parent_award_type_code                                      AS parent_type,
        parent_award_single_or_multiple_code                        AS parent_single_multiple,
        fair_opportunity_limited_sources_code                       AS fair_opportunity_limited,
        commercial_item_acquisition_procedures_code                 AS commercial_item,
        contracting_officers_determination_of_business_size         AS business_size,
        c8a_program_participant = 't'                               AS is_8a,
        service_disabled_veteran_owned_business = 't'               AS is_sdvosb,
        woman_owned_business = 't'                                  AS is_wosb,
        historically_underutilized_business_zone_hubzone_firm = 't' AS is_hubzone,
        nullif(national_interest_action_code, 'NONE')               AS national_interest,
        contract_bundling_code                                      AS bundling,
        consolidated_contract_code                                  AS consolidated,
        subcontracting_plan_code                                    AS subcontracting_plan,
        performance_based_service_acquisition_code                  AS performance_based,
        TRY_CAST(number_of_actions AS INTEGER)                      AS n_actions,
        prime_award_base_transaction_description                    AS description,
        usaspending_permalink                                       AS permalink,
        source_fy_signed
    FROM raw
)
SELECT *
FROM typed
WHERE award_key IS NOT NULL
  AND obligated IS NOT NULL
  AND end_date IS NOT NULL
QUALIFY row_number() OVER (PARTITION BY award_key ORDER BY last_modified DESC NULLS LAST) = 1;

-- Vendor identity: the parent UEI when present, else the recipient UEI. UEIs replaced DUNS in
-- April 2022 and USAspending backfilled them, but a normalized name is the fallback for gaps.
CREATE OR REPLACE MACRO norm_name(n) AS
    trim(regexp_replace(
        regexp_replace(upper(coalesce(n, '')), '[^A-Z0-9 ]', ' ', 'g'),
        '\b(INC|INCORPORATED|LLC|L L C|CORP|CORPORATION|CO|COMPANY|LTD|LP|LLP|PLLC|PC|THE|DBA|JV)\b', '', 'g'
    ));

ALTER TABLE awards ADD COLUMN vendor_id VARCHAR;
UPDATE awards SET vendor_id = coalesce(parent_uei, recipient_uei, 'NAME:' || norm_name(recipient_name));
ALTER TABLE awards ADD COLUMN vendor_name_norm VARCHAR;
UPDATE awards SET vendor_name_norm = norm_name(coalesce(nullif(parent_name, ''), recipient_name));
