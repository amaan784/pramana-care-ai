"""Single source of truth for catalog / schema / endpoint names.

These constants are imported everywhere (notebooks, tools, agent, app).
Override at runtime via env vars when needed (e.g. dev vs prod).
"""
from __future__ import annotations
import os

CATALOG: str = os.getenv("PRAMANA_CATALOG", "workspace")
SCHEMA: str = os.getenv("PRAMANA_SCHEMA", "pramana")
VOLUME: str = os.getenv("PRAMANA_VOLUME", "raw")

VOLUME_PATH: str = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_XLSX: str = f"{VOLUME_PATH}/VF_Hackathon_Dataset_India_Large.xlsx"

VS_ENDPOINT: str = os.getenv("PRAMANA_VS_ENDPOINT", "pramana_vs")
INDEX: str = os.getenv("PRAMANA_INDEX", f"{CATALOG}.{SCHEMA}.facilities_idx")
SOURCE_TEXT_TABLE: str = f"{CATALOG}.{SCHEMA}.silver_facilities_text"

LLM: str = os.getenv("PRAMANA_LLM", "databricks-meta-llama-3-3-70b-instruct")
JUDGE_LLM: str = os.getenv("PRAMANA_JUDGE_LLM", "databricks-claude-sonnet-4-5")
BATCH_LLM: str = os.getenv("PRAMANA_BATCH_LLM", "databricks-gpt-oss-20b")
EMBED: str = os.getenv("PRAMANA_EMBED", "databricks-gte-large-en")

GENIE_SPACE_ID: str = os.getenv("GENIE_SPACE_ID", "")
WAREHOUSE_ID: str = os.getenv("WAREHOUSE_ID", "")
SERVING_ENDPOINT_NAME: str = os.getenv("SERVING_ENDPOINT_NAME", "pramana-agent")

REGISTERED_MODEL: str = f"{CATALOG}.{SCHEMA}.pramana_agent"
EXPERIMENT_PATH: str = os.getenv(
    "PRAMANA_EXPERIMENT", "/Users/me@example.com/pramana-traces"
)

INDIA_BBOX = (6.5, 35.5, 68.0, 97.5)
MAX_VERIFIER_ITER: int = 8

UC_TOOLS = [
    f"{CATALOG}.{SCHEMA}.parse_messy_field",
    f"{CATALOG}.{SCHEMA}.score_claim_consistency",
    f"{CATALOG}.{SCHEMA}.geo_radius",
    f"{CATALOG}.{SCHEMA}.cross_source_disagree",
]
