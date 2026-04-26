# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Vector Search index
# MAGIC Idempotent: create endpoint if missing, then create a Delta-Sync index over
# MAGIC `silver.facilities_text` with managed embeddings (`databricks-gte-large-en`).

# COMMAND ----------
# MAGIC %pip install -q databricks-vectorsearch>=0.50
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import sys, time
sys.path.insert(0, "../src")
from pramana.config import CATALOG, SCHEMA, VS_ENDPOINT, INDEX, EMBED, SOURCE_TEXT_TABLE

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------
existing = {e["name"] for e in vsc.list_endpoints().get("endpoints", []) or []}
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    while True:
        s = vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state")
        print("endpoint state:", s)
        if s == "ONLINE":
            break
        time.sleep(20)

# COMMAND ----------
spark.sql(f"ALTER TABLE {SOURCE_TEXT_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

try:
    vsc.delete_index(endpoint_name=VS_ENDPOINT, index_name=INDEX)
except Exception as e:
    print("delete (ok if not found):", e)

vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT,
    index_name=INDEX,
    source_table_name=SOURCE_TEXT_TABLE,
    pipeline_type="TRIGGERED",
    primary_key="facility_id",
    embedding_source_column="facility_text",
    embedding_model_endpoint_name=EMBED,
)
print(f"Created index {INDEX}")

# COMMAND ----------
idx = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX)
idx.sync()
while True:
    st = idx.describe().get("status", {})
    ready = st.get("ready", False) or st.get("indexed_row_count", 0) > 0
    print("indexed:", st.get("indexed_row_count"), "ready:", ready)
    if ready:
        break
    time.sleep(15)

# COMMAND ----------
res = idx.similarity_search(
    query_text="cardiac surgery hospital with cath lab in Bihar",
    columns=["facility_id", "name", "district", "state", "facility_type", "description"],
    num_results=5, query_type="HYBRID")
display(res)
