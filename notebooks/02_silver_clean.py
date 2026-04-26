# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver clean
# MAGIC * Typo-fix `farmacy → pharmacy` (raw kept in `facility_type_raw`).
# MAGIC * Parse JSON-array strings to `ARRAY<STRING>` via deterministic `from_json`
# MAGIC   (verified: 100% of non-null rows in specialties/procedure/equipment/capability
# MAGIC   are valid JSON arrays — no LLM fallback needed).
# MAGIC * Geo-validate; normalise the 162 dirty `state` values via `state_aliases.json`
# MAGIC   (covers ~396 rows like 'Jammu And Kashmir', 'Punjab Region', 'Tamilnadu',
# MAGIC   'Up', 'Mh', city-as-state values like 'Mumbai'/'Pune'/'Chennai').
# MAGIC * Source `city` from `address_city` (the dataset has no `address_district`).
# MAGIC * Derive `recency_months` from `recency_of_page_update` for R8.

# COMMAND ----------
import sys, json, pathlib
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA

BR = f"{CATALOG}.{SCHEMA}.bronze_facilities_raw"
SC = f"{CATALOG}.{SCHEMA}.silver_facilities_clean"
CL = f"{CATALOG}.{SCHEMA}.silver_claims_long"
REF_BBOX    = f"{CATALOG}.{SCHEMA}.ref_state_bbox"
REF_ALIASES = f"{CATALOG}.{SCHEMA}.ref_state_aliases"

# COMMAND ----------
# Reference: state bbox (used by R3 and the Verifier)
state_bbox = json.loads(pathlib.Path("../data/reference/india_state_bbox.json").read_text(encoding="utf-8"))
(spark.createDataFrame([{"state": k, **v} for k, v in state_bbox.items()])
    .write.mode("overwrite").saveAsTable(REF_BBOX))

# Reference: state aliases (162 dirty values → canonical state names)
aliases = json.loads(pathlib.Path("../data/reference/state_aliases.json").read_text(encoding="utf-8"))
alias_rows = []
for raw, canon in aliases.items():
    if raw.startswith("_"):
        continue
    alias_rows.append({"alias": raw.lower(), "state": canon, "kind": "alias"})
for raw, canon in (aliases.get("_city_to_state") or {}).items():
    alias_rows.append({"alias": raw.lower(), "state": canon, "kind": "city"})
(spark.createDataFrame(alias_rows)
    .write.mode("overwrite").saveAsTable(REF_ALIASES))
print(f"loaded {len(alias_rows)} state-alias rows")

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {SC}
COMMENT 'Silver facilities — typo-fixed, JSON arrays parsed, geo-validated, state names normalized.'
AS
WITH base AS (
  SELECT
    CAST(facility_id AS STRING) AS facility_id,
    name,
    LOWER(facilityTypeId)        AS facility_type_raw,
    CASE WHEN LOWER(facilityTypeId) = 'farmacy' THEN 'pharmacy'
         ELSE LOWER(facilityTypeId) END AS facility_type,
    address_stateOrRegion        AS state_raw,
    address_city                 AS city,
    CAST(address_zipOrPostcode AS STRING) AS pin,
    CAST(latitude  AS DOUBLE) AS latitude,
    CAST(longitude AS DOUBLE) AS longitude,
    description,
    CAST(capacity        AS INT) AS capacity,
    CAST(numberDoctors   AS INT) AS number_doctors,
    CAST(yearEstablished AS INT) AS year_established,
    facebookLink, twitterLink, instagramLink, linkedinLink,
    websites, officialWebsite,
    try_cast(recency_of_page_update AS TIMESTAMP) AS recency_ts,
    specialties AS specialties_raw,
    procedure   AS procedure_raw,
    equipment   AS equipment_raw,
    capability  AS capability_raw
  FROM {BR}
),
norm AS (
  SELECT b.*,
         coalesce(
           bbox.state,
           a1.state,
           a2.state
         ) AS state
  FROM base b
  LEFT JOIN {REF_BBOX} bbox ON bbox.state = b.state_raw
  LEFT JOIN {REF_ALIASES} a1 ON a1.alias = lower(coalesce(b.state_raw, '')) AND a1.kind = 'alias'
  LEFT JOIN {REF_ALIASES} a2 ON a2.alias = lower(coalesce(b.state_raw, '')) AND a2.kind = 'city'
),
parsed AS (
  SELECT *,
    coalesce(
      from_json(specialties_raw, 'array<string>'),
      CASE WHEN specialties_raw IS NOT NULL AND length(trim(specialties_raw)) > 0
           THEN split(regexp_replace(specialties_raw, '[\\\\[\\\\]"]', ''), '\\\\s*[,|]\\\\s*')
           ELSE array() END
    ) AS specialties,
    coalesce(
      from_json(procedure_raw, 'array<string>'),
      CASE WHEN procedure_raw IS NOT NULL AND length(trim(procedure_raw)) > 0
           THEN split(regexp_replace(procedure_raw, '[\\\\[\\\\]"]', ''), '\\\\s*[,|]\\\\s*')
           ELSE array() END
    ) AS procedure,
    coalesce(
      from_json(equipment_raw, 'array<string>'),
      CASE WHEN equipment_raw IS NOT NULL AND length(trim(equipment_raw)) > 0
           THEN split(regexp_replace(equipment_raw, '[\\\\[\\\\]"]', ''), '\\\\s*[,|]\\\\s*')
           ELSE array() END
    ) AS equipment,
    coalesce(
      from_json(capability_raw, 'array<string>'),
      CASE WHEN capability_raw IS NOT NULL AND length(trim(capability_raw)) > 0
           THEN split(regexp_replace(capability_raw, '[\\\\[\\\\]"]', ''), '\\\\s*[,|]\\\\s*')
           ELSE array() END
    ) AS capability
  FROM norm
)
SELECT
  facility_id, name, facility_type_raw, facility_type,
  state_raw, state, city, pin,
  latitude, longitude, description, capacity, number_doctors, year_established,
  facebookLink, twitterLink, instagramLink, linkedinLink, websites, officialWebsite,
  recency_ts,
  CAST(months_between(current_timestamp(), recency_ts) AS INT) AS recency_months,
  specialties, procedure, equipment, capability
FROM parsed
""")

# COMMAND ----------
# Per-column comments — Genie quality is dominated by these
COMMENTS = {
    "facility_id":       "Unique facility identifier (STRING). Example: 'F000123'.",
    "name":              "Facility display name. May be unstructured.",
    "facility_type_raw": "Original facilityTypeId before typo-correction (e.g. 'farmacy').",
    "facility_type":     "Normalised type: hospital, clinic, dentist, doctor, pharmacy. 'farmacy' typo mapped to 'pharmacy'.",
    "state_raw":         "Original address_stateOrRegion verbatim. Includes ~400 rows of dirty values (cities, abbreviations, mixed strings).",
    "state":             "Canonical state name resolved via ref_state_bbox / ref_state_aliases. NULL when unresolvable.",
    "city":              "Sourced from address_city. The dataset has no address_district column, so city ≈ district HQ for most rows.",
    "pin":               "Postal Index Number, STRING with leading zeros preserved. Example: '855107'.",
    "latitude":          "WGS84 latitude in decimal degrees. India range 6.5–35.5.",
    "longitude":         "WGS84 longitude in decimal degrees. India range 68.0–97.5.",
    "description":       "Free-form facility description. Known to contain fabricated awards (e.g. 'W.HO award').",
    "capacity":          "Bed/seat capacity (INT). 99% null in source.",
    "number_doctors":    "Number of doctors (INT). 94% null in source.",
    "year_established":  "Year founded (INT). 92% null in source.",
    "recency_ts":        "Timestamp of last page update; null for ~95% of rows.",
    "recency_months":    "Months since last page update. Used by R8 (stale-page rule, threshold 24 months).",
    "specialties":       "ARRAY<STRING> of clinical specialties (camelCase tokens like 'cardiology', 'medicaloncology', 'orthopedicsurgery').",
    "procedure":         "ARRAY<STRING> of procedures performed. Example: ['root canal treatment','dental implants'].",
    "equipment":         "ARRAY<STRING> of equipment available. Example: ['ecg machine','ct scanner','dental chair'].",
    "capability":        "ARRAY<STRING> of declared capabilities. Free-form phrases like 'open 24/7', 'multispeciality hospital', 'has 1 doctor on staff'.",
}
for c, txt in COMMENTS.items():
    safe = txt.replace("'", "''")
    spark.sql(f"ALTER TABLE {SC} ALTER COLUMN {c} COMMENT '{safe}'")

# COMMAND ----------
# Claims long: one row per (facility_id, claim_type, claim_value) for cross-source joins
spark.sql(f"""
CREATE OR REPLACE TABLE {CL}
COMMENT 'One row per (facility_id, claim_type, claim_value) — used for cross-source consistency joins.'
AS
SELECT facility_id, 'specialty' AS claim_type, lower(trim(c)) AS claim_value FROM {SC}
  LATERAL VIEW explode(specialties) t AS c WHERE c IS NOT NULL AND length(trim(c)) > 0
UNION ALL SELECT facility_id, 'procedure',  lower(trim(c)) FROM {SC}
  LATERAL VIEW explode(procedure)  t AS c WHERE c IS NOT NULL AND length(trim(c)) > 0
UNION ALL SELECT facility_id, 'equipment',  lower(trim(c)) FROM {SC}
  LATERAL VIEW explode(equipment)  t AS c WHERE c IS NOT NULL AND length(trim(c)) > 0
UNION ALL SELECT facility_id, 'capability', lower(trim(c)) FROM {SC}
  LATERAL VIEW explode(capability) t AS c WHERE c IS NOT NULL AND length(trim(c)) > 0
""")

# COMMAND ----------
display(spark.sql(f"SELECT facility_type, count(*) FROM {SC} GROUP BY facility_type ORDER BY 2 DESC"))
display(spark.sql(f"SELECT claim_type, count(*) FROM {CL} GROUP BY claim_type"))
display(spark.sql(f"SELECT count(*) AS unresolved_states FROM {SC} WHERE state IS NULL"))
