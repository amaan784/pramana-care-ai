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
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../src/pramana/agent/agent.py",
        code_paths=["../src/pramana"],
        pip_requirements="../requirements.txt",
        resources=resources,
        input_example=input_example,
    )
    print("logged:", info.model_uri)

# COMMAND ----------
import mlflow
mlflow.set_registry_uri("databricks-uc")
reg = mlflow.register_model(model_uri=info.model_uri, name=REGISTERED_MODEL)
print("registered version:", reg.version)

# COMMAND ----------
from databricks import agents
dep = agents.deploy(
    model_name=REGISTERED_MODEL,
    model_version=reg.version,
    endpoint_name=SERVING_ENDPOINT_NAME,
    scale_to_zero=True,
    tags={"project": "pramana", "stage": "demo"},
    environment_vars={"GENIE_SPACE_ID": os.environ.get("GENIE_SPACE_ID", "")},
)
print("Endpoint:", dep.endpoint_name)
print("Review App:", dep.review_app_url)
