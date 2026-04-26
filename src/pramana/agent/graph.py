"""LangGraph supervisor: Planner (ReAct w/ tools) → Verifier → (loop|Synthesizer)."""
from __future__ import annotations
from typing import Any, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from pramana.agent.prompts import SYSTEM_PROMPT, SYNTHESIZER_PROMPT
from pramana.agent.verifier import verifier_node
from pramana.config import MAX_VERIFIER_ITER


class GraphState(TypedDict, total=False):
    messages: list[AnyMessage]
    iter: int
    needs_refine: bool
    allowed_facility_ids: list[str]


def build_graph(llm, tools: list):
    """Wire planner → verifier → synthesizer with refine loop. `tools` is a list of LangChain tools."""

    tools_by_name = {getattr(t, "name", str(i)): t for i, t in enumerate(tools)}

    planner = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )

    synthesizer = create_react_agent(
        model=llm,
        tools=[],
        prompt=SystemMessage(content=SYNTHESIZER_PROMPT),
    )

    g = StateGraph(GraphState)

    def _planner(state: GraphState) -> dict:
        out = planner.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "iter": state.get("iter", 0)}

    def _verifier(state: GraphState) -> dict:
        return verifier_node(state, llm, tools_by_name)

    def _synth(state: GraphState) -> dict:
        out = synthesizer.invoke({"messages": state["messages"]})
        return {"messages": out["messages"]}

    g.add_node("planner", _planner)
    g.add_node("verifier", _verifier)
    g.add_node("synthesizer", _synth)

    g.set_entry_point("planner")
    g.add_edge("planner", "verifier")

    def _route(state: GraphState):
        if state.get("needs_refine") and state.get("iter", 0) <= MAX_VERIFIER_ITER:
            return "planner"
        return "synthesizer"

    g.add_conditional_edges("verifier", _route, {"planner": "planner", "synthesizer": "synthesizer"})
    g.add_edge("synthesizer", END)

    return g.compile()
