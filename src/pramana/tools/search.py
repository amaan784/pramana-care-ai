"""Hybrid vector search wrapper.

The agent uses `databricks_langchain.VectorSearchRetrieverTool` directly
(see agent/graph.py); this thin Python wrapper exists for the *batch*
notebook code path and as a UC function for ad-hoc Genie/SQL use.
"""
from __future__ import annotations
import json


def search_facilities(query: str, k: int = 8) -> str:
    """Return top-k facilities matching free-text `query` as JSON list.

    Registered as UC function `main.pramana.search_facilities`.
    """
    from databricks.vector_search.client import VectorSearchClient
    client = VectorSearchClient(disable_notice=True)
    idx = client.get_index(index_name="main.pramana.facilities_idx")
    res = idx.similarity_search(
        query_text=query,
        columns=["facility_id", "name", "city", "state", "facility_type", "description"],
        num_results=int(k),
        query_type="HYBRID",
    )
    rows = (res or {}).get("result", {}).get("data_array", []) or []
    cols = [c["name"] for c in (res or {}).get("manifest", {}).get("columns", [])]
    out = [dict(zip(cols, r)) for r in rows]
    return json.dumps({"query": query, "k": int(k), "results": out})
