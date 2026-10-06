"""Entry point inside each Nebius Serverless Job.

Reads a shard of scenarios from the mounted volume, runs them on the GPU,
and writes <shard>.results.json next to it for the orchestrator to collect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robogauntlet.backends import get_backend
from robogauntlet.schemas import CampaignSpec, Scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--backend", default="isaac_arena")
    args = ap.parse_args()

    shard = Path(args.shard)
    payload = json.loads(shard.read_text())
    spec = CampaignSpec.model_validate(payload["spec"])
    backend = get_backend(args.backend)
    out_dir = shard.parent / "artifacts"
    results = []
    for raw in payload["scenarios"]:
        scenario = Scenario.model_validate(raw)
        results.append(backend.run(scenario, spec, out_dir).model_dump())
        print(f"done {scenario.id} success={results[-1]['success']}", flush=True)
    tmp = shard.with_suffix(".results.json.tmp")
    tmp.write_text(json.dumps(results))
    tmp.rename(shard.with_suffix(".results.json"))  # atomic: the orchestrator never reads half a file


if __name__ == "__main__":
    main()
