# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Silver text representation
# MAGIC One concatenated free-text field per facility, used by Vector Search.

# COMMAND ----------
import sys
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA

SC = f"{CATALOG}.{SCHEMA}.silver_facilities_clean"
TX = f"{CATALOG}.{SCHEMA}.silver_facilities_text"

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TABLE {TX}
TBLPROPERTIES (delta.enableChangeDataFeed = true)
COMMENT 'One row per facility — flattened free-text representation used as the embedding source.'
AS
SELECT
  facility_id,
  concat_ws(' | ',
    concat('name: ', coalesce(name, '')),
    concat('type: ', coalesce(facility_type, '')),
    concat('state: ', coalesce(state, '')),
    concat('district: ', coalesce(district, '')),
    concat('specialties: ', concat_ws(', ', coalesce(specialties, array()))),
    concat('procedure: ',   concat_ws(', ', coalesce(procedure,   array()))),
    concat('equipment: ',   concat_ws(', ', coalesce(equipment,   array()))),
    concat('capability: ',  concat_ws(', ', coalesce(capability,  array()))),
    concat('description: ', coalesce(description, ''))
  ) AS facility_text,
  name, facility_type, state, district, description
FROM {SC}
""")

spark.sql(f"ALTER TABLE {TX} ALTER COLUMN facility_id   COMMENT 'Primary key, joins to silver_facilities_clean.'")
spark.sql(f"ALTER TABLE {TX} ALTER COLUMN facility_text COMMENT 'Concatenated text used as embedding input.'")
display(spark.sql(f"SELECT * FROM {TX} LIMIT 5"))
