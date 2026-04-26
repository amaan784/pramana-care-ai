# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Evaluation
# MAGIC Baseline (bare Llama 3.3 70B, no tools) vs Intervention (full Pramana agent).
# MAGIC Produces the headline delta on RetrievalGroundedness, Correctness, and our custom judges.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os

os.environ["PRAMANA_EXPERIMENT"] = "/Users/amaan784@gmail.com/pramana-traces"
os.environ["PRAMANA_CATALOG"] = "workspace"
os.environ["PRAMANA_SCHEMA"] = "pramana"
os.environ["SERVING_ENDPOINT_NAME"] = "pramana-agent"

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

if "question" not in golden.columns:
    if "inputs" in golden.columns:
        golden["question"] = golden["inputs"].apply(
            lambda x: x.get("question") if isinstance(x, dict) else x
        )
    elif "input" in golden.columns:
        golden["question"] = golden["input"]
    elif "request" in golden.columns:
        golden["question"] = golden["request"]
    else:
        raise ValueError(f"No question-like column found. Columns: {list(golden.columns)}")

if "expected_facts" not in golden.columns:
    golden["expected_facts"] = [[] for _ in range(len(golden))]

golden = golden[golden["question"].notna()].reset_index(drop=True)
golden["inputs"] = golden["question"].apply(lambda q: {"question": q})
golden["expectations"] = golden["expected_facts"].apply(
    lambda facts: {
        "expected_facts": facts if isinstance(facts, list) else [str(facts)]
    }
)
eval_data = golden[["inputs", "expectations"]].copy()

print(f"golden rows: {len(golden)}")
display(golden[["question", "expected_facts"]].head(5))

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

token = None
try:
    token = w.config.token
except Exception:
    token = None

if not token:
    token = (
        dbutils.notebook.entry_point
        .getDbutils()
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )

client = OpenAI(
    api_key=token,
    base_url=f"{w.config.host}/serving-endpoints",
)

def pramana_predict_fn(question: str) -> str:
    # The deployed model is an MLflow ResponsesAgent, so the endpoint schema is
    # `input`, not OpenAI chat-completions `messages`.
    r = client.responses.create(
        model=SERVING_ENDPOINT_NAME,
        input=[{"role": "user", "content": question}],
        timeout=120,
    )
    if getattr(r, "output_text", None):
        return r.output_text
    chunks = []
    for item in getattr(r, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            text = getattr(c, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks) or str(r)

# COMMAND ----------
scorers = [RetrievalGroundedness(), RelevanceToQuery(), Safety(), Correctness(), *PRAMANA_JUDGES]

with mlflow.start_run(run_name="baseline_llama_no_tools"):
    base_res = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=baseline_predict_fn,
        scorers=scorers,
    )
    base_metrics = base_res.metrics if hasattr(base_res, "metrics") else {}
    print("baseline metrics:", base_metrics)

with mlflow.start_run(run_name="pramana_agent_full"):
    int_res = mlflow.genai.evaluate(
        data=eval_data,
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
