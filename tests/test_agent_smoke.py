"""End-to-end smoke test: hit the deployed serving endpoint with one demo question.

Skipped automatically unless `SERVING_ENDPOINT_NAME`, `DATABRICKS_HOST`, and
`DATABRICKS_TOKEN` are present (so it's safe to run in CI).
"""
from __future__ import annotations
import os
import pytest


pytestmark = pytest.mark.skipif(
    not (os.environ.get("SERVING_ENDPOINT_NAME")
         and os.environ.get("DATABRICKS_HOST")
         and os.environ.get("DATABRICKS_TOKEN")),
    reason="Requires deployed endpoint + Databricks credentials.",
)


def test_endpoint_round_trip():
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DATABRICKS_TOKEN"],
        base_url=f"{os.environ['DATABRICKS_HOST'].rstrip('/')}/serving-endpoints",
    )
    resp = client.chat.completions.create(
        model=os.environ["SERVING_ENDPOINT_NAME"],
        messages=[{"role": "user",
                    "content": "How many entries have the 'farmacy' typo? Answer briefly."}],
        timeout=180,
    )
    text = resp.choices[0].message.content or ""
    assert text.strip(), "empty response"
    low = text.lower()
    assert "farmacy" in low or "pharmacy" in low or "typo" in low
    assert "confidence:" in low, "answer must end with a Confidence: line"
