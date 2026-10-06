"""Scenario planner: Nemotron 3 Ultra turns a task into a stress-test plan.

Tavily (optional) pulls known failure modes for similar tasks from the web.
The planner may only pick axes the backend supports, and its ranges are
clipped to the backend's limits, so a model mistake can't break the simulator.
"""

from __future__ import annotations

import json

import requests

from .config import Settings
from .llm import TokenFactory
from .schemas import Axis, CampaignSpec, StressPlan

PLANNER_SYSTEM = """You are a robotics test engineer designing a stress test for a learned robot policy.
You receive the task, the robot, and the ONLY perturbation axes the simulator supports, with their maximum ranges.
Choose the axes most likely to expose failures for this task (at least 4), and for each choose a test range
inside the allowed range. Prefer ranges that include the nominal value and extend into realistic deployment
conditions. Respect any user constraints.
Reply with JSON only:
{"task_summary": str,
 "axes": [{"name": str, "low": float, "high": float, "why": str}],
 "known_failure_modes": [str],
 "rationale": str}"""


def tavily_failure_research(settings: Settings, spec: CampaignSpec, max_results: int = 5) -> tuple[list[str], list[str]]:
    """Return (snippets, source urls) about known failure modes for similar tasks."""
    if not settings.tavily_enabled:
        return [], []
    query = (
        f"failure modes of learned robot manipulation policies for: {spec.task.instruction} "
        f"({spec.policy.embodiment}); sim-to-real robustness lighting, pose, friction, clutter"
    )
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={"api_key": settings.tavily_api_key, "query": query, "max_results": max_results, "search_depth": "advanced"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as err:
        print(f"[tavily] search failed: {err}")
        return [], []
    snippets = [f"{r.get('title', '')}: {r.get('content', '')[:500]}" for r in results]
    urls = [r.get("url", "") for r in results if r.get("url")]
    return snippets, urls


def _clip_axes(proposed: list[dict], allowed: list[Axis]) -> list[Axis]:
    by_name = {a.name: a for a in allowed}
    axes: list[Axis] = []
    for item in proposed:
        base = by_name.get(str(item.get("name", "")))
        if base is None or any(a.name == base.name for a in axes):
            continue
        try:
            low = max(base.low, min(base.high, float(item["low"])))
            high = max(base.low, min(base.high, float(item["high"])))
        except (KeyError, TypeError, ValueError):
            low, high = base.low, base.high
        if high - low < 1e-9:
            low, high = base.low, base.high
        axes.append(base.model_copy(update={"low": low, "high": high, "description": item.get("why") or base.description}))
    return axes


def make_plan(spec: CampaignSpec, allowed: list[Axis], settings: Settings, tf: TokenFactory) -> StressPlan:
    snippets, sources = tavily_failure_research(settings, spec)
    user = json.dumps(
        {
            "task": spec.task.model_dump(),
            "robot": spec.policy.embodiment,
            "policy": spec.policy.label or spec.policy.source,
            "allowed_axes": [a.model_dump(include={"name", "kind", "low", "high", "unit", "nominal", "description"}) for a in allowed],
            "web_research": snippets,
        },
        indent=1,
    )
    reply = tf.chat_json(settings.models.ultra, PLANNER_SYSTEM, user, purpose="plan stress test")
    if isinstance(reply, dict):
        axes = _clip_axes(reply.get("axes", []), allowed)
        if len(axes) >= 2:
            return StressPlan(
                task_summary=str(reply.get("task_summary", spec.task.instruction)),
                axes=axes,
                rationale=str(reply.get("rationale", "")),
                known_failure_modes=[str(x) for x in reply.get("known_failure_modes", [])][:10],
                sources=sources,
                planner_model=settings.models.ultra,
            )
    # Offline or failed: test every supported axis over its full range.
    return StressPlan(
        task_summary=spec.task.instruction,
        axes=[a.model_copy() for a in allowed],
        rationale="Offline default: every supported axis over its full range.",
        known_failure_modes=[],
        sources=sources,
    )
