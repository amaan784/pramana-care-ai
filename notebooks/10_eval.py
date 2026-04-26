# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Evaluation
# MAGIC Baseline (bare Llama 3.3 70B, no tools) vs Intervention (full Pramana agent).
# MAGIC Produces the headline delta on RetrievalGroundedness, Correctness, and our custom judges.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import sys, json, mlflow, pandas as pd
sys.path.insert(0, "../src")
from pathlib import Path

from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Correctness
from databricks_langchain import ChatDatabricks

from pramana.config import LLM, EXPERIMENT_PATH, SERVING_ENDPOINT_NAME
from pramana.eval.custom_judges import PRAMANA_JUDGES
from pramana.eval.generate_golden import seed_from_corpus, merge_with_handwritten

mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------
seed = seed_from_corpus(n=10)
golden = merge_with_handwritten(seed, "../eval/golden_questions.jsonl")
golden = golden.head(25).reset_index(drop=True)
print(f"golden rows: {len(golden)}")
display(golden.head(5))

# COMMAND ----------
def baseline_predict_fn(question: str) -> str:
    llm = ChatDatabricks(endpoint=LLM, max_tokens=600, temperature=0.1)
    msg = llm.invoke([{"role": "system", "content":
                       "You are a healthcare facility assistant. Answer briefly with confidence label."},
                       {"role": "user", "content": question}])
    return msg.content

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from openai import OpenAI
w = WorkspaceClient()
client = OpenAI(api_key=w.config.oauth_token().access_token,
                base_url=f"{w.config.host}/serving-endpoints")

def pramana_predict_fn(question: str) -> str:
    r = client.chat.completions.create(
        model=SERVING_ENDPOINT_NAME,
        messages=[{"role": "user", "content": question}],
        timeout=120,
    )
    return r.choices[0].message.content

# COMMAND ----------
scorers = [RetrievalGroundedness(), RelevanceToQuery(), Safety(), Correctness(), *PRAMANA_JUDGES]

with mlflow.start_run(run_name="baseline_llama_no_tools"):
    base_res = mlflow.genai.evaluate(
        data=golden,
        predict_fn=baseline_predict_fn,
        scorers=scorers,
    )
    base_metrics = base_res.metrics if hasattr(base_res, "metrics") else {}
    print("baseline metrics:", base_metrics)

with mlflow.start_run(run_name="pramana_agent_full"):
    int_res = mlflow.genai.evaluate(
        data=golden,
        predict_fn=pramana_predict_fn,
        scorers=scorers,
    )
    int_metrics = int_res.metrics if hasattr(int_res, "metrics") else {}
    print("intervention metrics:", int_metrics)

# COMMAND ----------
keys = sorted(set(list(base_metrics.keys()) + list(int_metrics.keys())))
delta = pd.DataFrame([
    {"metric": k,
     "baseline": base_metrics.get(k),
     "pramana":  int_metrics.get(k),
     "delta":    (int_metrics.get(k, 0) or 0) - (base_metrics.get(k, 0) or 0)}
    for k in keys
])
display(delta)
