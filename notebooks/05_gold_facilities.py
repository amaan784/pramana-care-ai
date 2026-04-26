# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Gold facilities
# MAGIC Join silver.facilities_clean ⊕ silver.contradictions ⊕ silver.trust + add
# MAGIC `h3_6`, `h3_8`, `st_geom_wkt`. Powers the map and Genie.
# MAGIC
# MAGIC Geometry is persisted as WKT (a STRING) rather than as the native GEOMETRY
# MAGIC type because:
# MAGIC   1. serverless `display()` cannot render GEOMETRY columns,
# MAGIC   2. Genie and downstream `SELECT *` queries break on it,
# MAGIC   3. the lat/lon doubles already cover 99% of needs; `tools/geo.py` builds
# MAGIC      `ST_Point(longitude, latitude)` on the fly anyway.
# MAGIC Hydrate with `ST_GeomFromText(st_geom_wkt)` when geometry is needed.

# COMMAND ----------
import sys
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA

SC = f"{CATALOG}.{SCHEMA}.silver_facilities_clean"
CT = f"{CATALOG}.{SCHEMA}.silver_contradictions"
TS = f"{CATALOG}.{SCHEMA}.silver_trust"
GD = f"{CATALOG}.{SCHEMA}.gold_facilities"

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {GD}
COMMENT 'Gold facility table: clean fields + trust score + contradiction flags + H3 cells + ST geometry. Powers the map UI and the Genie space.'
AS
WITH agg AS (
  SELECT facility_id,
         collect_list(struct(rule_id, severity, message, evidence, citation_column)) AS flags
  FROM {CT} GROUP BY facility_id
)
SELECT
  s.facility_id,
  s.name,
  s.facility_type_raw,
  s.facility_type,
  s.state_raw,
  s.state,
  s.city,
  s.pin,
  s.latitude,
  s.longitude,
  s.description,
  s.capacity,
  s.number_doctors,
  s.year_established,
  s.specialties,
  s.procedure,
  s.equipment,
  s.capability,
  coalesce(t.trust_score, 100) AS trust_score,
  coalesce(a.flags, array())   AS flags,
  CASE WHEN s.latitude IS NOT NULL AND s.longitude IS NOT NULL
       THEN h3_h3tostring(h3_longlatash3(s.longitude, s.latitude, 6)) END AS h3_6,
  CASE WHEN s.latitude IS NOT NULL AND s.longitude IS NOT NULL
       THEN h3_h3tostring(h3_longlatash3(s.longitude, s.latitude, 8)) END AS h3_8,
  CASE WHEN s.latitude IS NOT NULL AND s.longitude IS NOT NULL
       THEN ST_AsText(ST_Point(s.longitude, s.latitude)) END AS st_geom_wkt
FROM {SC} s
LEFT JOIN {TS}  t USING (facility_id)
LEFT JOIN agg   a USING (facility_id)
""")

# COMMAND ----------
COMMENTS = {
    "trust_score": "0-100 derived score: 100 - 35*HIGH - 15*MED - 5*LOW. Lower = more contradictions.",
    "flags":       "Array of struct{rule_id, severity, message, evidence, citation_column}.",
    "h3_6":        "H3 hex id (string) at resolution 6. Use for state/city-level density maps.",
    "h3_8":        "H3 hex id (string) at resolution 8. Use for radius search and street-level maps.",
    "st_geom_wkt": "WKT (well-known-text) representation of ST_Point(longitude, latitude). Hydrate with ST_GeomFromText() when geometry is needed; the doubles latitude/longitude are usually enough.",
}
for c, txt in COMMENTS.items():
    safe = txt.replace("'", "''")
    spark.sql(f"ALTER TABLE {GD} ALTER COLUMN {c} COMMENT '{safe}'")

display(spark.sql(f"SELECT trust_score, count(*) FROM {GD} GROUP BY trust_score ORDER BY 1"))
display(spark.sql(f"SELECT * FROM {GD} ORDER BY trust_score ASC LIMIT 10"))
