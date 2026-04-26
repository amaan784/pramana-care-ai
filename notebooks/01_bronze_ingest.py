# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze ingest
# MAGIC Read VF_Hackathon_Dataset_India_Large.xlsx from a Unity Catalog volume and write
# MAGIC `workspace.pramana.bronze_facilities_raw` by default. PIN codes preserved as STRING (leading zeros).

# COMMAND ----------
# MAGIC %pip install -q openpyxl==3.1.5 pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import sys, pandas as pd
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA, RAW_XLSX, VOLUME_PATH

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")
print(f"Reading {RAW_XLSX}")

# COMMAND ----------
pdf = pd.read_excel(RAW_XLSX, dtype=str, engine="openpyxl")
print(f"rows={len(pdf):,} cols={len(pdf.columns)}")
print(pdf.columns.tolist())

# COMMAND ----------
pdf.columns = [c.strip().replace(" ", "_") for c in pdf.columns]
if "facility_id" not in pdf.columns:
    pid_col = next((c for c in pdf.columns if c.lower() in {"facilityid", "id", "facility"}), None)
    pdf["facility_id"] = pdf[pid_col] if pid_col else [f"F{i:06d}" for i in range(len(pdf))]
pdf["facility_id"] = pdf["facility_id"].astype(str).str.strip()

for c in [c for c in pdf.columns if "pin" in c.lower() or "postal" in c.lower() or "zip" in c.lower()]:
    pdf[c] = pdf[c].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

for c in ("latitude", "longitude", "capacity", "numberDoctors", "yearEstablished"):
    if c in pdf.columns:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

# COMMAND ----------
sdf = spark.createDataFrame(pdf)
target = f"{CATALOG}.{SCHEMA}.bronze_facilities_raw"
(sdf.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(target))

spark.sql(f"COMMENT ON TABLE {target} IS 'Raw VillageFinder Indian healthcare facility dataset, "
          "10K rows, 41 cols. JSON-array fields preserved as STRING. PIN codes as STRING with leading zeros.'")
display(spark.sql(f"SELECT count(*) AS n FROM {target}"))
