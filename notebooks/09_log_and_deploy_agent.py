# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Log + register + deploy the Pramana agent
# MAGIC `mlflow.pyfunc.log_model(code_paths=["../src/pramana"], python_model="../src/pramana/agent/agent.py")`,
# MAGIC register to UC, then `databricks.agents.deploy(...)` with `scale_to_zero=True`.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os, sys, mlflow
sys.path.insert(0, "../src")
from pramana.config import (
    REGISTERED_MODEL, INDEX, UC_TOOLS, LLM, EXPERIMENT_PATH, SERVING_ENDPOINT_NAME,
)

# IMPORTANT: experiment must NOT live inside a Git folder, or live tracing breaks.
mlflow.set_experiment(EXPERIMENT_PATH)

# Surface source-doc links + groundedness to the AI Playground / judges.
mlflow.models.set_retriever_schema(
    name="search_facilities",
    primary_key="facility_id",
    text_column="description",
    doc_uri="facility_id",
)

# COMMAND ----------
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksFunction,
    DatabricksGenieSpace,
)

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM),
    DatabricksVectorSearchIndex(index_name=INDEX),
    *[DatabricksFunction(function_name=f) for f in UC_TOOLS],
]
genie_space = os.environ.get("GENIE_SPACE_ID")
if genie_space:
    resources.append(DatabricksGenieSpace(genie_space_id=genie_space))

input_example = {
    "input": [{"role": "user",
                "content": "Is District Hospital Kishanganj actually equipped for cardiac surgery?"}]
}

with mlflow.start_run(run_name="pramana-agent"):
    # Avoid MLflow's log-time input-example validation path. On Databricks this
    # can trip a tracing/contextmanager bug ("generator didn't stop after
    # throw()") before deployment. We validate the live endpoint after
    # agents.deploy() instead.
    os.environ["PRAMANA_DISABLE_LANGCHAIN_AUTOLOG"] = "1"
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../src/pramana/agent/agent.py",
        code_paths=["../src/pramana"],
        pip_requirements="../requirements.txt",
        resources=resources,
    )
    print("logged:", info.model_uri)

# COMMAND ----------
import mlflow
mlflow.set_registry_uri("databricks-uc")
reg = mlflow.register_model(model_uri=info.model_uri, name=REGISTERED_MODEL)
print("registered version:", reg.version)

# COMMAND ----------
from databricks import agents
from mlflow import MlflowClient

# Defensively resolve the latest existing registered version. If notebook cells
# are rerun out of order (or older versions are deleted in the UI), a stale
# `reg.version` can point at a deleted model version and make deploy fail.
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name = '{REGISTERED_MODEL}'")
if not versions:
    raise RuntimeError(f"No registered versions found for {REGISTERED_MODEL}")

latest_version = max(versions, key=lambda v: int(v.version)).version
print("deploying registered model version:", latest_version)

dep = agents.deploy(
    model_name=REGISTERED_MODEL,
    model_version=latest_version,
    endpoint_name=SERVING_ENDPOINT_NAME,
    scale_to_zero=True,
    tags={"project": "pramana", "stage": "demo"},
    environment_vars={
        "GENIE_SPACE_ID": os.environ.get("GENIE_SPACE_ID", ""),
        "WAREHOUSE_ID": os.environ.get("WAREHOUSE_ID", ""),
        "PRAMANA_CATALOG": os.environ.get("PRAMANA_CATALOG", "workspace"),
        "PRAMANA_SCHEMA": os.environ.get("PRAMANA_SCHEMA", "pramana"),
        "PRAMANA_INDEX": os.environ.get(
            "PRAMANA_INDEX",
            "workspace.pramana.facilities_idx",
        ),
        "SERVING_ENDPOINT_NAME": os.environ.get(
            "SERVING_ENDPOINT_NAME",
            "pramana-agent",
        ),
    },
)
print("Endpoint:", dep.endpoint_name)
print("Review App:", dep.review_app_url)
