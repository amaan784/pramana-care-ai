"""Hybrid vector search wrapper.

The agent uses `databricks_langchain.VectorSearchRetrieverTool` directly
(see agent/agent.py); this thin Python wrapper exists for local/notebook
experiments only. It is **not** registered as a Unity Catalog Python function:
UC Python UDF sandboxes do not inherit notebook `%pip` packages like
`databricks-vectorsearch`.
"""
from __future__ import annotations
import json


def search_facilities(query: str, k: int) -> str:
    """Return top-k facilities matching free-text ``query`` as a JSON list.

    Performs a hybrid (vector + BM25) search against the Delta-Sync index
    ``facilities_idx`` over ``silver_facilities_text``.

    Args:
        query: Natural-language query describing the facility, capability or
            location of interest. Examples: ``"cardiac surgery in Bihar"``,
            ``"primary health centre near Patna"``, ``"oncology hospital
            Maharashtra"``.
        k: Number of top hits to return, ordered by hybrid score. Pass 8 if
            the user did not specify how many results they want; reasonable
            range is 1 to 50.

    Returns:
        JSON string with keys ``query``, ``k`` and ``results`` (a list of dicts
        with ``facility_id``, ``name``, ``city``, ``state``, ``facility_type``
        and ``description``).
    """
    import os

    from databricks.vector_search.client import VectorSearchClient

    cat = os.environ.get("PRAMANA_CATALOG", "workspace")
    sch = os.environ.get("PRAMANA_SCHEMA", "pramana")
    index = os.environ.get("PRAMANA_INDEX", f"{cat}.{sch}.facilities_idx")

    client = VectorSearchClient(disable_notice=True)
    idx = client.get_index(index_name=index)
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
