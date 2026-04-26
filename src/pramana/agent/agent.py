"""Pramana ResponsesAgent — the artifact that gets logged & deployed."""
from __future__ import annotations
import os
from typing import Generator

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
    UCFunctionToolkit,
)
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from pramana.agent.graph import build_graph
from pramana.agent.genie_tool import genie_query
from pramana.config import LLM, INDEX, UC_TOOLS

if os.getenv("PRAMANA_DISABLE_LANGCHAIN_AUTOLOG", "").lower() not in {"1", "true", "yes"}:
    mlflow.langchain.autolog()


def _to_chat_messages(items):
    """Convert ResponsesAgent input items to chat-completion message dicts."""
    out = []
    for it in items:
        d = it.model_dump() if hasattr(it, "model_dump") else dict(it)
        role = d.get("role") or ("assistant" if d.get("type") == "message" else "user")
        content = d.get("content")
        if isinstance(content, list):
            content = "".join(
                (c.get("text") or "") if isinstance(c, dict) else str(c) for c in content
            )
        out.append({"role": role, "content": content or ""})
    return out


class PramanaAgent(ResponsesAgent):
    def __init__(self):
        llm = ChatDatabricks(endpoint=LLM, max_tokens=2000, temperature=0.1)

        retriever = VectorSearchRetrieverTool(
            index_name=INDEX,
            num_results=8,
            query_type="HYBRID",
            tool_name="search_facilities",
            tool_description="Search Indian medical facilities by free-text claim or specialty.",
            columns=["facility_id", "name", "district", "state",
                      "facility_type", "description"],
        )

        uc_tools = UCFunctionToolkit(function_names=UC_TOOLS).tools

        self.tools = [retriever, genie_query, *uc_tools]
        self.graph = build_graph(llm, self.tools)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        msgs = _to_chat_messages(request.input)
        result = self.graph.invoke({"messages": msgs, "iter": 0})
        final = result["messages"][-1]
        text = getattr(final, "content", "") or ""
        item = {
            "type": "message",
            "id": "msg_pramana",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
        return ResponsesAgentResponse(output=[item], custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        msgs = _to_chat_messages(request.input)
        last_text = ""
        for _, evts in self.graph.stream(
            {"messages": msgs, "iter": 0}, stream_mode=["updates"]
        ):
            for nd in evts.values():
                ms = nd.get("messages") if isinstance(nd, dict) else None
                if not ms:
                    continue
                m = ms[-1]
                txt = getattr(m, "content", "") or ""
                if txt and txt != last_text:
                    last_text = txt
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item={
                            "type": "message",
                            "id": "msg_pramana",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": txt}],
                        },
                    )


set_model(PramanaAgent())
