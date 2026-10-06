"""Command line: robogauntlet run | report | benchmark | jobs-dry-run"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .campaign import Campaign
from .config import Settings
from .report import build_report
from .schemas import CampaignSpec


def _load_spec(path: str) -> CampaignSpec:
    return CampaignSpec.model_validate_json(Path(path).read_text())


def cmd_run(args: argparse.Namespace) -> None:
    spec = _load_spec(args.spec)
    settings = Settings(offline=args.offline)
    run_dir = Path(args.out) if args.out else None
    summary = Campaign(spec, settings, backend=args.backend, rounds=args.rounds, budget=args.budget,
                       compare_random=not args.no_random, seed=args.seed, workers=args.workers, run_dir=run_dir).run()
    if not summary.get("aborted"):
        out_dir = run_dir or max(Path("runs").iterdir(), key=lambda p: p.stat().st_mtime)
        print(f"Report: {build_report(out_dir)}")


def cmd_report(args: argparse.Namespace) -> None:
    print(build_report(Path(args.run_dir)))


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Adaptive vs random over several seeds (mock backend), to report averages instead of one lucky run."""
    os.environ["RG_MOCK_RENDER"] = "0"
    spec = _load_spec(args.spec)
    rows = []
    for seed in range(args.seeds):
        s = Campaign(spec, Settings(offline=args.offline), backend="mock", rounds=args.rounds, budget=args.budget,
                     seed=seed, workers=args.workers, run_dir=Path(args.out) / f"seed{seed}", verbose=False).run()
        a, r = s["curves"]["adaptive"], s["curves"]["random"]
        total = a[-1]["distinct_modes"]

        def first(curve: list[dict], k: int) -> int | None:
            return next((p["rollouts"] for p in curve if p["distinct_modes"] >= k), None)

        rows.append({
            "seed": seed,
            "adaptive_failures": a[-1]["failures"], "random_failures": r[-1]["failures"],
            "adaptive_types": a[-1]["distinct_modes"], "random_types": r[-1]["distinct_modes"],
            "adaptive_fail_per_gpuh": a[-1]["failures"] / a[-1]["gpu_hours"],
            "random_fail_per_gpuh": r[-1]["failures"] / r[-1]["gpu_hours"],
            "adaptive_rollouts_to_all_types": first(a, 4), "random_rollouts_to_all_types": first(r, 4),
        })
        print(json.dumps(rows[-1]))
    keys = [k for k in rows[0] if k != "seed"]
    print("\nmean over", args.seeds, "seeds")
    for k in keys:
        vals = [row[k] for row in rows if row[k] is not None]
        missing = sum(row[k] is None for row in rows)
        print(f"  {k:34s} {np.mean(vals) if vals else float('nan'):8.2f}" + (f"   (not reached in {missing} seeds)" if missing else ""))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "benchmark.json").write_text(json.dumps(rows, indent=1))


def cmd_jobs_dry_run(args: argparse.Namespace) -> None:
    """Print the Nebius Serverless Jobs commands one round would launch, without launching anything."""
    from .runner import NebiusJobsRunner
    from .schemas import Scenario

    spec = _load_spec(args.spec)
    scen = [Scenario(id=f"a1-{i:03d}", params={}, round=1, origin="space_filling", seed=i) for i in range(args.budget)]
    NebiusJobsRunner(image=args.image, subnet_id=args.subnet_id, volume=args.volume, dry_run=True,
                     local_mount=Path(args.local_mount)).run(scen, spec, Path("."), tag="demo")


def main() -> None:
    ap = argparse.ArgumentParser(prog="robogauntlet", description="Adaptive stress testing for robot policies")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("run", help="run a full campaign and build the report")
    p.add_argument("spec")
    p.add_argument("--backend", default="mock", choices=["mock", "isaac_arena"])
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--budget", type=int, default=40, help="rollouts per round")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-random", action="store_true", help="skip the random baseline")
    p.add_argument("--offline", action="store_true", help="no Token Factory, Cosmos or Tavily calls")
    p.add_argument("--out")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="rebuild the HTML report for a run directory")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("benchmark", help="adaptive vs random over several seeds on the mock backend")
    p.add_argument("spec")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out", default="runs/benchmark")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("jobs-dry-run", help="print the Nebius Serverless Jobs commands for one round")
    p.add_argument("spec")
    p.add_argument("--image", default="<registry>/robogauntlet-worker:latest")
    p.add_argument("--subnet-id", default="<subnet_id>")
    p.add_argument("--volume", default="<filesystem_id>:/data")
    p.add_argument("--local-mount", default="./nebius_volume")
    p.add_argument("--budget", type=int, default=16)
    p.set_defaults(func=cmd_jobs_dry_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
