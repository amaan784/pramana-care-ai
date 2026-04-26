# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Silver contradictions
# MAGIC Apply the 8 rules from `pramana.tools.consistency.evaluate_facility` row-by-row
# MAGIC and persist `silver.contradictions`.

# COMMAND ----------
import sys, json
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA
from pramana.tools.consistency import evaluate_facility, trust_score

from pyspark.sql import functions as F, types as T

SC = f"{CATALOG}.{SCHEMA}.silver_facilities_clean"
CT = f"{CATALOG}.{SCHEMA}.silver_contradictions"
TS = f"{CATALOG}.{SCHEMA}.silver_trust"

# COMMAND ----------
bbox_rows = spark.table(f"{CATALOG}.{SCHEMA}.ref_state_bbox").collect()
bbox = {r["state"]: {"min_lat": r["min_lat"], "max_lat": r["max_lat"],
                       "min_lon": r["min_lon"], "max_lon": r["max_lon"]} for r in bbox_rows}
bbox_b = spark.sparkContext.broadcast(bbox)

flag_schema = T.ArrayType(T.StructType([
    T.StructField("rule_id",         T.StringType()),
    T.StructField("severity",        T.StringType()),
    T.StructField("message",         T.StringType()),
    T.StructField("evidence",        T.StringType()),
    T.StructField("citation_column", T.StringType()),
]))

@F.udf(returnType=flag_schema)
def _flags_udf(row_json: str):
    if not row_json:
        return []
    return evaluate_facility(json.loads(row_json), bbox_b.value)

@F.udf(returnType=T.IntegerType())
def _trust_udf(flags):
    return trust_score([dict(f.asDict()) if hasattr(f, "asDict") else dict(f) for f in (flags or [])])

# COMMAND ----------
src = (spark.table(SC)
       .withColumn("__row", F.to_json(F.struct(*[F.col(c) for c in spark.table(SC).columns]))))

flagged = src.withColumn("flags", _flags_udf(F.col("__row"))) \
             .withColumn("trust_score", _trust_udf(F.col("flags"))) \
             .drop("__row")

(flagged.select("facility_id", "trust_score", "flags")
        .write.mode("overwrite").saveAsTable(TS))
spark.sql(f"COMMENT ON TABLE {TS} IS 'Per-facility trust score (0-100) and contradiction flags array.'")

(flagged.select("facility_id", F.explode_outer("flags").alias("flag"))
        .selectExpr("facility_id",
                     "flag.rule_id        AS rule_id",
                     "flag.severity       AS severity",
                     "flag.message        AS message",
                     "flag.evidence       AS evidence",
                     "flag.citation_column AS citation_column")
        .where("rule_id IS NOT NULL")
        .write.mode("overwrite").saveAsTable(CT))
spark.sql(f"COMMENT ON TABLE {CT} IS 'Long form: one row per (facility_id, contradiction flag).'")

# COMMAND ----------
display(spark.sql(f"SELECT severity, count(*) FROM {CT} GROUP BY severity ORDER BY 2 DESC"))
display(spark.sql(f"SELECT rule_id, count(*) FROM {CT} GROUP BY rule_id ORDER BY rule_id"))
display(spark.sql(f"""
  SELECT
    percentile_approx(trust_score, 0.10) AS p10,
    percentile_approx(trust_score, 0.50) AS p50,
    percentile_approx(trust_score, 0.90) AS p90,
    avg(trust_score)                     AS mean_score
  FROM {TS}
"""))
