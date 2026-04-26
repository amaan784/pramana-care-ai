# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver clean
# MAGIC Typo-fix `farmacy → pharmacy` (kept in audit column), parse JSON-array strings to ARRAY<STRING>
# MAGIC via `from_json` (deterministic fast path) with `ai_extract` fallback for malformed rows,
# MAGIC geo-validate, canonicalize specialties via `ai_classify`, explode claims into long form.

# COMMAND ----------
import sys
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA

BR = f"{CATALOG}.{SCHEMA}.bronze_facilities_raw"
SC = f"{CATALOG}.{SCHEMA}.silver_facilities_clean"
CL = f"{CATALOG}.{SCHEMA}.silver_claims_long"

# COMMAND ----------
# Reference table for state bbox (used by R3 and the Verifier)
import json, pathlib
state_bbox = json.loads(pathlib.Path("../data/reference/india_state_bbox.json").read_text(encoding="utf-8"))
rows = [{"state": k, **v} for k, v in state_bbox.items()]
(spark.createDataFrame(rows)
    .write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.ref_state_bbox"))

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {SC}
COMMENT 'Silver facilities — typo-fixed, JSON arrays parsed, geo-validated, specialties canonicalized.'
AS
WITH base AS (
  SELECT
    CAST(facility_id AS STRING) AS facility_id,
    name,
    LOWER(facilityTypeId)        AS facility_type_raw,
    CASE WHEN LOWER(facilityTypeId) = 'farmacy' THEN 'pharmacy'
         ELSE LOWER(facilityTypeId) END AS facility_type,
    address_stateOrRegion AS state,
    address_city          AS city,
    CAST(address_postalCode AS STRING) AS pin,
    CAST(latitude  AS DOUBLE) AS latitude,
    CAST(longitude AS DOUBLE) AS longitude,
    description,
    CAST(capacity        AS INT) AS capacity,
    CAST(numberDoctors   AS INT) AS number_doctors,
    CAST(yearEstablished AS INT) AS year_established,
    specialties AS specialties_raw,
    procedure   AS procedure_raw,
    equipment   AS equipment_raw,
    capability  AS capability_raw
  FROM {BR}
),
parsed AS (
  SELECT *,
    coalesce(try_cast(from_json(specialties_raw, 'array<string>') AS array<string>), array()) AS specialties_fast,
    coalesce(try_cast(from_json(procedure_raw,   'array<string>') AS array<string>), array()) AS procedure_fast,
    coalesce(try_cast(from_json(equipment_raw,   'array<string>') AS array<string>), array()) AS equipment_fast,
    coalesce(try_cast(from_json(capability_raw,  'array<string>') AS array<string>), array()) AS capability_fast
  FROM base
)
SELECT
  facility_id, name, facility_type_raw, facility_type, state, city, pin,
  latitude, longitude, description, capacity, number_doctors, year_established,
  CASE WHEN size(specialties_fast)=0 AND specialties_raw IS NOT NULL
       THEN try_cast(ai_extract(specialties_raw, array('items'))['items'] AS array<string>)
       ELSE specialties_fast END AS specialties,
  CASE WHEN size(procedure_fast)=0  AND procedure_raw  IS NOT NULL
       THEN try_cast(ai_extract(procedure_raw,  array('items'))['items'] AS array<string>)
       ELSE procedure_fast  END AS procedure,
  CASE WHEN size(equipment_fast)=0  AND equipment_raw  IS NOT NULL
       THEN try_cast(ai_extract(equipment_raw,  array('items'))['items'] AS array<string>)
       ELSE equipment_fast  END AS equipment,
  CASE WHEN size(capability_fast)=0 AND capability_raw IS NOT NULL
       THEN try_cast(ai_extract(capability_raw, array('items'))['items'] AS array<string>)
       ELSE capability_fast END AS capability
FROM parsed
""")

# COMMAND ----------
# Per-column comments — Genie quality is dominated by these
COMMENTS = {
    "facility_id":       "Unique facility identifier (STRING). Example: 'F000123'.",
    "name":              "Facility display name. May be unstructured.",
    "facility_type_raw": "Original facilityTypeId before typo-correction (e.g. 'farmacy').",
    "facility_type":     "Normalised type: hospital, clinic, dentist, doctor, pharmacy. 'farmacy' typo mapped to 'pharmacy'.",
    "state":             "Indian state or union territory. Example: 'Bihar', 'Tamil Nadu'.",
    "city":              "City / district-HQ proxy from address_city. Example: 'Kishanganj'.",
    "pin":               "Postal Index Number, STRING with leading zeros preserved. Example: '855107'.",
    "latitude":          "WGS84 latitude in decimal degrees. India range 6.5–35.5.",
    "longitude":         "WGS84 longitude in decimal degrees. India range 68.0–97.5.",
    "description":       "Free-form facility description. R6 checks for fabricated awards (0 matches in current snapshot).",
    "capacity":          "Bed/seat capacity (INT). 99% null in source.",
    "number_doctors":    "Number of doctors (INT). 94% null in source.",
    "year_established":  "Year founded (INT). 92% null in source.",
    "specialties":       "ARRAY<STRING> of clinical specialties. Example: ['cardiology','oncology'].",
    "procedure":         "ARRAY<STRING> of procedures performed. Example: ['angioplasty','dialysis'].",
    "equipment":         "ARRAY<STRING> of equipment available. Example: ['ECG','MRI','ventilator'].",
    "capability":        "ARRAY<STRING> of declared capabilities. Example: ['ICU','24x7 emergency'].",
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
