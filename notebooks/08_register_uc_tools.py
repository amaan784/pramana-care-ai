# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Register UC tools
# MAGIC Registers the 4 Unity Catalog verifier functions used by the agent + Genie.

# COMMAND ----------
# MAGIC %pip install -q unitycatalog-langchain[databricks]>=0.3.0 databricks-vectorsearch>=0.50
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import sys
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA
from pramana.tools.registration import register_all
from pramana.config import CATALOG, SCHEMA

NS = f"{CATALOG}.{SCHEMA}"

created = register_all(spark)
for f in created:
    print("registered:", f)

# COMMAND ----------
# Smoke test each one
import json
sample = spark.sql(f"SELECT facility_id FROM {NS}.gold_facilities ORDER BY trust_score ASC LIMIT 1").collect()[0][0]

print("score_claim_consistency:")
print(spark.sql(f"SELECT {NS}.score_claim_consistency('{sample}') AS r").collect()[0][0][:600])

print("\ncross_source_disagree:")
print(spark.sql(f"SELECT {NS}.cross_source_disagree('{sample}', 'cardiac surgery icu') AS r").collect()[0][0][:600])

print("\ngeo_radius:")
print(spark.sql(f"SELECT {NS}.geo_radius(25.59, 85.13, 25.0, 'cardiology', 5) AS r").collect()[0][0][:600])

print("\nparse_messy_field:")
print(spark.sql(f"SELECT {NS}.parse_messy_field('24x7 emergency, ICU, MRI, Dr. Sharma cardiology, W.HO award 2019') AS r").collect()[0][0][:600])
