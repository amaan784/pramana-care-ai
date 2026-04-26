"""Seed a golden-set with `databricks.agents.evals.generate_evals_df`, merge with hand-curated."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd


def seed_from_corpus(text_table: str = "workspace.pramana.silver_facilities_text",
                      n: int = 15) -> pd.DataFrame:
    from pyspark.sql import SparkSession
    from databricks.agents.evals import generate_evals_df

    spark = SparkSession.builder.getOrCreate()
    docs = (spark.table(text_table)
            .selectExpr("facility_id AS doc_uri", "facility_text AS content")
            .limit(200).toPandas())

    agent_description = (
        "Pramana verifies whether listed Indian healthcare facility capabilities are "
        "backed by equipment, geo, and structured fill-rate. Cites facility_id."
    )
    question_guidelines = (
        "- Mix discovery (e.g. nearest oncology in Bihar), verification "
        "(does X actually have ICU?), and audit (find facilities with bad coords) questions.\n"
        "- Target known data issues: 'farmacy' typo, ghost hospitals, specialty-without-equipment, and cross-state coordinate mismatches. "
        "For fabricated awards, expect 0 matches in this snapshot."
    )
    return generate_evals_df(
        docs=docs,
        num_evals=n,
        agent_description=agent_description,
        question_guidelines=question_guidelines,
    )


def merge_with_handwritten(seed_df: pd.DataFrame, jsonl_path: str) -> pd.DataFrame:
    extra = []
    p = Path(jsonl_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                extra.append(json.loads(line))
    extra_df = pd.DataFrame(extra)
    return pd.concat([seed_df, extra_df], ignore_index=True)
