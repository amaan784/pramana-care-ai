"""Wrap the Genie Conversation API as a LangChain tool.

Genie spaces can't be created via SDK on Free Edition without service-principal
gymnastics, so we reference an existing space by id (env var GENIE_SPACE_ID).
"""
from __future__ import annotations
import os
from typing import Optional

from langchain_core.tools import tool


@tool("genie_query", return_direct=False)
def genie_query(question: str) -> str:
    """Run a natural-language question against the `pramana_facilities` Genie space and
    return its answer (text + the SQL it generated, when available)."""
    space_id = os.environ.get("GENIE_SPACE_ID")
    if not space_id:
        return "Genie not configured (GENIE_SPACE_ID missing). Falling back to search."
    try:
        from databricks.sdk import WorkspaceClient
    except Exception as e:
        return f"databricks-sdk not available: {e}"

    w = WorkspaceClient()
    try:
        res = w.genie.start_conversation_and_wait(space_id=space_id, content=question)
    except Exception as e:
        # Genie can fail a message when it generates invalid SQL (for example,
        # stale space instructions referring to `main.pramana` or `district`).
        # Return a tool result instead of crashing the whole serving endpoint.
        return (
            "Genie query failed before completion. Treat this as low-confidence "
            f"and do not fabricate an aggregate answer. Error: {type(e).__name__}: {e}"
        )
    text_parts: list[str] = []
    sql_text: Optional[str] = None

    attachments = getattr(res, "attachments", None) or []
    for a in attachments:
        text = getattr(getattr(a, "text", None), "content", None)
        if text:
            text_parts.append(text)
        q = getattr(a, "query", None)
        if q is not None:
            sql_text = getattr(q, "query", None) or getattr(q, "description", None)

    out = "\n".join(text_parts) if text_parts else (getattr(res, "content", "") or "")
    if sql_text:
        out += f"\n\n```sql\n{sql_text}\n```"
    return out or "Genie returned no answer."
