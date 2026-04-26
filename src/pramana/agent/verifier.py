"""Verifier / Critic node.

CRAG-style self-grade: extracts factual claims from the latest assistant draft,
calls cross_source_disagree on each, and returns either:
  - a normal AIMessage  (accept)  → graph proceeds to Synthesizer
  - an AIMessage flagged with `needs_refine=True` in additional_kwargs → graph loops to Planner

Hard step counter prevents infinite loops (max MAX_VERIFIER_ITER).
"""
from __future__ import annotations
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from pramana.config import MAX_VERIFIER_ITER

_FACILITY_ID_RE = re.compile(r"\b(?:facility[_ ]id[:= ]?|\[)?([A-Za-z0-9][\w\-]{4,})\]?", re.I)
_BRACKET_ID_RE  = re.compile(r"\[([A-Za-z0-9][\w\-]{4,})\]")


def _extract_facility_ids(text: str, allowed: set[str] | None = None) -> list[str]:
    ids = set(_BRACKET_ID_RE.findall(text or ""))
    if allowed:
        ids &= allowed
    return sorted(ids)


def _extract_claims(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in sents if 20 < len(s) < 400]


def verifier_node(state: dict, llm: Any, tools_by_name: dict) -> dict:
    """Return updated state with either a refine instruction or an accept signal.

    `state` keys used: messages, iter, allowed_facility_ids (optional).
    """
    iters = int(state.get("iter", 0)) + 1
    msgs = state.get("messages", [])
    allowed = set(state.get("allowed_facility_ids") or [])

    last_ai = next((m for m in reversed(msgs) if getattr(m, "type", "") == "ai"), None)
    draft = (getattr(last_ai, "content", "") if last_ai else "") or ""

    if iters > MAX_VERIFIER_ITER:
        return {
            "messages": [AIMessage(
                content=draft + "\n\nConfidence: low (verifier iteration cap reached).",
                additional_kwargs={"verified": True, "confidence": "low"},
            )],
            "iter": iters,
        }

    fids = _extract_facility_ids(draft, allowed or None)
    claims = _extract_claims(draft)
    cross = tools_by_name.get("cross_source_disagree")
    score = tools_by_name.get("score_claim_consistency")

    disagreements: list[dict] = []
    if cross and fids and claims:
        for fid in fids[:5]:
            for claim in claims[:3]:
                try:
                    raw = cross.invoke({"facility_id": fid, "claim": claim})
                    res = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(res, dict) and not res.get("agree", True):
                        disagreements.append({"fid": fid, "claim": claim,
                                               "supporting": res.get("sources_supporting", [])})
                except Exception:
                    continue

    high_flags: list[dict] = []
    if score and fids:
        for fid in fids[:3]:
            try:
                raw = score.invoke({"facility_id": fid})
                res = json.loads(raw) if isinstance(raw, str) else raw
                for f in (res or {}).get("flags", []):
                    if f.get("severity") == "HIGH":
                        high_flags.append({"fid": fid, **f})
            except Exception:
                continue

    if disagreements or high_flags:
        reason_parts = []
        if disagreements:
            reason_parts.append(f"{len(disagreements)} cross-source disagreements: " +
                                 "; ".join(f"{d['fid']}: {d['claim'][:60]}" for d in disagreements[:3]))
        if high_flags:
            reason_parts.append(f"{len(high_flags)} HIGH contradiction flags: " +
                                 "; ".join(f"{h['fid']}/{h['rule_id']}" for h in high_flags[:3]))
        reason = " | ".join(reason_parts)
        return {
            "messages": [HumanMessage(content=(
                "VERIFIER_REFINE: " + reason +
                "\nReturn an updated answer that either (a) removes the unsupported claims, "
                "(b) attaches the contradiction flags as 'Truth-gap flags', or (c) downgrades confidence."
            ))],
            "iter": iters,
            "needs_refine": True,
        }

    return {
        "messages": [AIMessage(
            content=draft,
            additional_kwargs={"verified": True, "confidence": "high" if fids else "medium"},
        )],
        "iter": iters,
        "needs_refine": False,
    }
