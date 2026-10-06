"""Orchestrates a full stress-test campaign: plan, sanity check, adaptive rounds, random baseline."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analyst import IGNORED_FOR_MODES, analyse, failure_clusters, name_clusters
from .backends import get_backend
from .config import Settings
from .judge import CosmosJudge
from .llm import TokenFactory, UsageLedger
from .planner import make_plan
from .runner import LocalRunner
from .schemas import AnalystReport, Axis, CampaignSpec, FailureLabel, Region, RolloutResult, Scenario, StressPlan
from .search import propose_adaptive, uniform


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class StrategyTrace:
    scenarios: list[Scenario] = field(default_factory=list)
    results: list[RolloutResult] = field(default_factory=list)
    labels: list[FailureLabel] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)

    def add(self, sc: list[Scenario], res: list[RolloutResult], lab: list[FailureLabel], round_idx: int) -> None:
        self.scenarios += sc
        self.results += res
        self.labels += lab
        gpu_h = sum(r.gpu_seconds for r in self.results) / 3600
        modes = sorted({l.category for l in self.labels if l.failed and l.category not in IGNORED_FOR_MODES})
        self.curve.append({
            "round": round_idx,
            "rollouts": len(self.results),
            "gpu_hours": gpu_h,
            "failures": sum(l.failed for l in self.labels),
            "distinct_modes": len(modes),
            "modes": modes,
        })


def _to_scenarios(U: np.ndarray, origins: list[str], axes: list[Axis], nominal: dict[str, float],
                  round_idx: int, prefix: str, strategy: str, rng: np.random.Generator) -> list[Scenario]:
    out = []
    for i, (u, origin) in enumerate(zip(U, origins)):
        params = dict(nominal)
        params.update({a.name: a.denormalize(float(np.clip(uj, 0, 1))) for a, uj in zip(axes, u)})
        out.append(Scenario(id=f"{prefix}{round_idx}-{i:03d}", params=params, round=round_idx, origin=origin,
                            seed=int(rng.integers(0, 2**31 - 1)), strategy=strategy))
    return out


def _unit(scenarios: list[Scenario], axes: list[Axis]) -> np.ndarray:
    return np.array([[a.normalize(s.params[a.name]) for a in axes] for s in scenarios]).reshape(-1, len(axes))


class Campaign:
    def __init__(self, spec: CampaignSpec, settings: Settings, backend: str = "mock", rounds: int = 4,
                 budget: int = 40, compare_random: bool = True, seed: int = 0, workers: int = 4,
                 run_dir: Path | None = None, runner=None, sanity_runs: int = 5, verbose: bool = True) -> None:
        self.spec = spec
        self.settings = settings
        self.backend_name = backend
        self.backend = get_backend(backend)
        self.rounds = rounds
        self.budget = budget
        self.compare_random = compare_random
        self.seed = seed
        self.run_dir = run_dir or Path("runs") / time.strftime("%Y%m%d-%H%M%S")
        self.runner = runner or LocalRunner(backend, workers=workers)
        self.sanity_runs = sanity_runs
        self.verbose = verbose
        self.ledger = UsageLedger()
        self.tf = TokenFactory(settings, self.ledger)
        self.judge = CosmosJudge(settings, self.ledger)

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _execute(self, scenarios: list[Scenario]) -> tuple[list[RolloutResult], list[FailureLabel]]:
        results = self.runner.run(scenarios, self.spec, self.run_dir / "frames")
        order = {s.id: i for i, s in enumerate(scenarios)}
        results.sort(key=lambda r: order[r.scenario_id])
        labels = self.judge.judge(results, self.spec.task.instruction)
        return results, labels

    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.seed)
        allowed = self.backend.supported_axes(self.spec)
        nominal = self.backend.nominal_params(self.spec)

        self.log("1/4 Planning stress test" + (" with Nemotron Ultra" if self.tf.enabled else " (offline default plan)"))
        plan: StressPlan = make_plan(self.spec, allowed, self.settings, self.tf)
        axes = plan.axes
        self.log(f"    axes: {', '.join(a.name for a in axes)}")

        self.log(f"2/4 Sanity check: {self.sanity_runs} rollouts in the unperturbed scene")
        sanity = [Scenario(id=f"s0-{i:03d}", params=dict(nominal), round=0, origin="nominal",
                           seed=int(rng.integers(0, 2**31 - 1)), strategy="sanity") for i in range(self.sanity_runs)]
        s_res, s_lab = self._execute(sanity)
        sanity_rate = sum(r.success for r in s_res) / max(1, len(s_res))
        self.log(f"    nominal success rate {sanity_rate:.0%}")
        if sanity_rate < 0.6:
            summary = {"aborted": True, "reason": (
                f"Policy succeeds only {sanity_rate:.0%} of the time in its own nominal scene. "
                "Check the observation/action spec and normalization stats before stress testing.")}
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
            self.log("    " + summary["reason"])
            return summary

        self.log(f"3/4 Adaptive search: {self.rounds} rounds x {self.budget} rollouts")
        adaptive = StrategyTrace()
        reports: list[AnalystReport] = []
        regions: list[Region] = []
        for r in range(self.rounds):
            U, origins = propose_adaptive(_unit(adaptive.scenarios, axes),
                                          np.array([l.failed for l in adaptive.labels], dtype=bool),
                                          self.budget, rng, axes, regions, r)
            sc = _to_scenarios(U, origins, axes, nominal, r + 1, "a", "adaptive", rng)
            res, lab = self._execute(sc)
            adaptive.add(sc, res, lab, r + 1)
            report = analyse(r + 1, axes, adaptive.scenarios, adaptive.labels, self.settings, self.tf)
            reports.append(report)
            regions = report.next_regions
            c = adaptive.curve[-1]
            self.log(f"    round {r + 1}: {c['failures']} failures, {c['distinct_modes']} failure types {c['modes']}")

        baseline = StrategyTrace()
        if self.compare_random:
            self.log(f"4/4 Random baseline with the same budget ({self.rounds * self.budget} rollouts)")
            for r in range(self.rounds):
                U = uniform(self.budget, len(axes), rng)
                sc = _to_scenarios(U, ["random"] * self.budget, axes, nominal, r + 1, "r", "random", rng)
                res, lab = self._execute(sc)
                baseline.add(sc, res, lab, r + 1)
            c = baseline.curve[-1]
            self.log(f"    random: {c['failures']} failures, {c['distinct_modes']} failure types {c['modes']}")

        # Robustness is estimated on uniform samples only; adaptive samples are biased toward failures.
        if self.compare_random:
            est_from, est_res = "random baseline", baseline.results
        else:
            est_from = "round-1 space-filling sample"
            est_res = [res for res, s in zip(adaptive.results, adaptive.scenarios) if s.round == 1]
        p, lo, hi = wilson(sum(x.success for x in est_res), len(est_res))

        all_sc = adaptive.scenarios + baseline.scenarios
        all_lab = adaptive.labels + baseline.labels
        clusters = failure_clusters(axes, all_sc, all_lab)
        titles = name_clusters(clusters, self.settings, self.tf)

        summary = {
            "aborted": False,
            "spec": self.spec.model_dump(),
            "backend": self.backend_name,
            "settings": {"llm": self.tf.enabled, "judge": self.judge._client is not None,
                         "tavily": self.settings.tavily_enabled, "models": vars(self.settings.models)},
            "sanity_success_rate": sanity_rate,
            "robustness": {"success_rate": p, "ci95": [lo, hi], "n": len(est_res), "estimated_from": est_from},
            "curves": {"adaptive": adaptive.curve, "random": baseline.curve},
            "clusters": [dict(c.model_dump(), title=titles.get(c.id, "")) for c in clusters],
            "ledger": self.ledger.to_dict(),
        }
        self._save(plan, sanity, s_res, s_lab, adaptive, baseline, reports, summary)
        self.log(f"Done. Robustness {p:.0%} (95% CI {lo:.0%}-{hi:.0%}). Results in {self.run_dir}")
        return summary

    def _save(self, plan, sanity, s_res, s_lab, adaptive, baseline, reports, summary) -> None:
        d = self.run_dir
        (d / "plan.json").write_text(plan.model_dump_json(indent=1))
        with open(d / "rollouts.jsonl", "w") as f:
            for sc, res, lab in zip(sanity + adaptive.scenarios + baseline.scenarios,
                                    s_res + adaptive.results + baseline.results,
                                    s_lab + adaptive.labels + baseline.labels):
                f.write(json.dumps({"scenario": sc.model_dump(), "result": res.model_dump(), "label": lab.model_dump()}) + "\n")
        (d / "rounds.json").write_text(json.dumps([r.model_dump() for r in reports], indent=1))
        (d / "summary.json").write_text(json.dumps(summary, indent=1))
