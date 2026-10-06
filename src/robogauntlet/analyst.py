"""Failure analyst: finds what the failures have in common and steers the next round.

Statistics are computed in code (sensitivity per axis, failure clusters with
their parameter bounds). Nemotron 3 Super then reads those numbers, writes the
plain-language summary and proposes regions to probe next; Nemotron Nano names
each failure cluster. Offline, regions come from the clusters and the most
sensitive axes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from .config import Settings
from .llm import TokenFactory
from .schemas import AnalystReport, Axis, AxisSensitivity, FailureCluster, FailureLabel, Region, Scenario

IGNORED_FOR_MODES = {"timeout", "none"}

ANALYST_SYSTEM = """You analyse stress-test results for a robot policy. You receive the tested axes, failure rate by
value bin for each axis, and failure clusters with the parameter bounds where each failure type occurred.
1. Write a 2-4 sentence summary a robotics engineer can act on: what breaks the policy and under which conditions.
2. Propose up to 3 regions to test next round. Each region constrains 1-3 axes (real units, inside the axis ranges).
   Mix (a) tightening the boundary of known failure clusters and (b) plausible combinations NOT yet covered,
   based on physical reasoning (e.g. glare plus camera error). Explain each with a one-line hypothesis.
Reply with JSON only:
{"summary": str, "regions": [{"bounds": {"axis_name": [low, high]}, "hypothesis": str}]}"""

NAMER_SYSTEM = """Give each failure cluster a short plain-language title (max 9 words) describing the condition and the
failure, e.g. "Heavy objects slip on low-friction grippers". Reply with JSON only: {"titles": {"<id>": str}}"""


def sensitivities(axes: list[Axis], scenarios: list[Scenario], failed: np.ndarray, bins: int = 5) -> list[AxisSensitivity]:
    out = []
    for a in axes:
        vals = np.array([s.params[a.name] for s in scenarios])
        edges = np.linspace(a.low, a.high, bins + 1)
        idx = np.clip(np.digitize(vals, edges[1:-1]), 0, bins - 1)
        rates: list[float | None] = []
        for b in range(bins):
            m = idx == b
            rates.append(float(failed[m].mean()) if m.sum() >= 3 else None)
        known = [r for r in rates if r is not None]
        out.append(AxisSensitivity(axis=a.name, bin_edges=[float(e) for e in edges], failure_rate_by_bin=rates,
                                   spread=float(max(known) - min(known)) if len(known) >= 2 else 0.0))
    return sorted(out, key=lambda s: -s.spread)


def failure_clusters(axes: list[Axis], scenarios: list[Scenario], labels: list[FailureLabel],
                     max_coverage: float = 0.30, min_size: int = 3) -> list[FailureCluster]:
    """One cluster per failure category, bounded on the axes that explain it.

    An axis explains a cluster when the cluster's failures sit in a value range
    that holds only a small share of everything tested (max_coverage). This
    compares against what was actually sampled, so adaptive oversampling of a
    region doesn't make an irrelevant axis look concentrated.
    """
    by_id = {s.id: s for s in scenarios}
    tested = {a.name: np.array([s.params[a.name] for s in scenarios]) for a in axes}
    groups: dict[str, list[Scenario]] = defaultdict(list)
    for lab in labels:
        if lab.failed:
            groups[lab.category].append(by_id[lab.scenario_id])
    clusters = []
    for k, (cat, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        bounds = {}
        if len(members) >= min_size:
            for a in axes:
                v = np.array([m.params[a.name] for m in members])
                lo, hi = float(v.min()), float(v.max())
                coverage = float(((tested[a.name] >= lo) & (tested[a.name] <= hi)).mean())
                if coverage <= max_coverage:
                    bounds[a.name] = (lo, hi)
        clusters.append(FailureCluster(id=k, size=len(members), categories=dict(Counter([cat] * len(members))),
                                       bounds=bounds, example_ids=[m.id for m in members[:6]]))
    return clusters


def _offline_regions(axes: list[Axis], clusters: list[FailureCluster], sens: list[AxisSensitivity]) -> list[Region]:
    by_name = {a.name: a for a in axes}
    regions = []
    for c in clusters:
        if not c.bounds or set(c.categories) <= IGNORED_FOR_MODES:
            continue
        b = {}
        for name, (lo, hi) in c.bounds.items():
            pad = 0.15 * (by_name[name].high - by_name[name].low)
            b[name] = (max(by_name[name].low, lo - pad), min(by_name[name].high, hi + pad))
        regions.append(Region(bounds=b, hypothesis=f"Map the boundary of the {next(iter(c.categories))} cluster"))
    # Combine the worst bins of the two most sensitive axes: an untested interaction.
    top = [s for s in sens if s.spread > 0][:2]
    if len(top) == 2:
        b = {}
        for s in top:
            rates = [r if r is not None else -1 for r in s.failure_rate_by_bin]
            i = int(np.argmax(rates))
            b[s.axis] = (s.bin_edges[i], s.bin_edges[i + 1])
        regions.append(Region(bounds=b, hypothesis=f"Interaction of {top[0].axis} and {top[1].axis} at their worst values"))
    return regions[:3]


def _validate_regions(raw: list, axes: list[Axis]) -> list[Region]:
    by_name = {a.name: a for a in axes}
    regions = []
    for r in raw or []:
        b = {}
        for name, rng in (r.get("bounds") or {}).items():
            a = by_name.get(name)
            if a is None or not isinstance(rng, (list, tuple)) or len(rng) != 2:
                continue
            lo, hi = sorted(float(x) for x in rng)
            lo, hi = max(a.low, lo), min(a.high, hi)
            if hi > lo:
                b[name] = (lo, hi)
        if b:
            regions.append(Region(bounds=b, hypothesis=str(r.get("hypothesis", ""))[:200]))
    return regions[:3]


def _offline_summary(sens: list[AxisSensitivity], clusters: list[FailureCluster], n: int, n_fail: int) -> str:
    parts = [f"{n_fail} of {n} rollouts failed so far."]
    real = [c for c in clusters if not set(c.categories) <= IGNORED_FOR_MODES]
    for c in real[:4]:
        cond = " and ".join(f"{k} {lo:.3g} to {hi:.3g}" for k, (lo, hi) in c.bounds.items()) or "no single region yet"
        parts.append(f"{next(iter(c.categories)).replace('_', ' ')} ({c.size} runs): {cond}.")
    if sens and sens[0].spread > 0:
        parts.append(f"Most sensitive axis: {sens[0].axis}.")
    return " ".join(parts)


def analyse(round_idx: int, axes: list[Axis], scenarios: list[Scenario], labels: list[FailureLabel],
            settings: Settings, tf: TokenFactory) -> AnalystReport:
    lab_by_id = {l.scenario_id: l for l in labels}
    failed = np.array([lab_by_id[s.id].failed for s in scenarios])
    sens = sensitivities(axes, scenarios, failed)
    clusters = failure_clusters(axes, scenarios, labels)

    payload = json.dumps({
        "axes": [a.model_dump(include={"name", "unit", "low", "high"}) for a in axes],
        "rollouts": len(scenarios),
        "failures": int(failed.sum()),
        "sensitivity": [s.model_dump() for s in sens],
        "clusters": [c.model_dump(exclude={"example_ids"}) for c in clusters],
    })
    reply = tf.chat_json(settings.models.super, ANALYST_SYSTEM, payload, purpose="analyse failures, steer next round")
    if isinstance(reply, dict) and reply.get("summary"):
        summary = str(reply["summary"])
        regions = _validate_regions(reply.get("regions", []), axes) or _offline_regions(axes, clusters, sens)
        model = settings.models.super
    else:
        summary = _offline_summary(sens, clusters, len(scenarios), int(failed.sum()))
        regions = _offline_regions(axes, clusters, sens)
        model = "offline-default"
    return AnalystReport(round=round_idx, summary=summary, sensitivities=sens, clusters=clusters,
                         next_regions=regions, analyst_model=model)


def name_clusters(clusters: list[FailureCluster], settings: Settings, tf: TokenFactory) -> dict[int, str]:
    """Short titles for the report, from Nemotron Nano (cheap, one call)."""
    def offline(c: FailureCluster) -> str:
        cat = next(iter(c.categories)).replace("_", " ")
        axes = " + ".join(c.bounds) if c.bounds else "scattered conditions"
        return f"{cat.capitalize()} under {axes.replace('_', ' ')}"

    titles = {c.id: offline(c) for c in clusters}
    reply = tf.chat_json(settings.models.nano, NAMER_SYSTEM,
                         json.dumps([c.model_dump(exclude={"example_ids"}) for c in clusters]),
                         purpose="name failure clusters", max_tokens=800)
    if isinstance(reply, dict):
        for k, v in (reply.get("titles") or {}).items():
            try:
                titles[int(k)] = str(v)[:80]
            except ValueError:
                pass
    return titles
