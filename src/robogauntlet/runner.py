"""Running a batch of scenarios: locally in parallel, or fanned out as Nebius Serverless Jobs."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .backends import get_backend
from .schemas import CampaignSpec, RolloutResult, Scenario


def _run_one(args: tuple[str, str, str, str]) -> str:
    backend_name, scenario_json, spec_json, out_dir = args
    backend = get_backend(backend_name)
    result = backend.run(Scenario.model_validate_json(scenario_json), CampaignSpec.model_validate_json(spec_json), Path(out_dir))
    return result.model_dump_json()


class LocalRunner:
    """Process pool on this machine. Good for the mock backend and single-GPU dev boxes."""

    def __init__(self, backend_name: str, workers: int = 4) -> None:
        self.backend_name = backend_name
        self.workers = workers

    def run(self, scenarios: list[Scenario], spec: CampaignSpec, out_dir: Path) -> list[RolloutResult]:
        spec_json = spec.model_dump_json()
        jobs = [(self.backend_name, s.model_dump_json(), spec_json, str(out_dir)) for s in scenarios]
        if self.workers <= 1:
            return [RolloutResult.model_validate_json(_run_one(j)) for j in jobs]
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            return [RolloutResult.model_validate_json(r) for r in pool.map(_run_one, jobs)]


@dataclass
class NebiusJobsRunner:
    """One Serverless Job per shard of scenarios.

    Shards and results go through a shared volume (a Nebius filesystem or an
    Object Storage bucket mounted into every job at `mount_path`). The container
    runs jobs/rollout_worker.py, which reads its shard and writes results back.

    Set dry_run=True to print the exact CLI commands without launching anything.
    """

    image: str
    subnet_id: str
    volume: str  # e.g. "<filesystem_id>:/data" as accepted by `nebius ai job create --volume`
    mount_path: str = "/data"
    local_mount: Path = Path("./nebius_volume")  # where this machine sees the same volume
    platform: str = "gpu-l40s-a"
    preset: str = "1gpu-8vcpu-32gb"
    backend_name: str = "isaac_arena"
    shard_size: int = 8
    timeout: str = "2h"
    env: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    poll_s: float = 20.0

    def build_command(self, job_name: str, shard_rel: str) -> list[str]:
        cmd = [
            "nebius", "ai", "job", "create",
            "--name", job_name,
            "--image", self.image,
            "--platform", self.platform,
            "--preset", self.preset,
            "--subnet-id", self.subnet_id,
            "--volume", self.volume,
            "--timeout", self.timeout,
            "--container-command", "python",
            "--args", f"/app/jobs/rollout_worker.py --shard {self.mount_path}/{shard_rel} --backend {self.backend_name}",
        ]
        for k, v in self.env.items():
            cmd += ["--env", f"{k}={v}"]
        return cmd

    def run(self, scenarios: list[Scenario], spec: CampaignSpec, out_dir: Path, tag: str = "rg") -> list[RolloutResult]:
        shard_dir = self.local_mount / "shards" / tag
        shard_dir.mkdir(parents=True, exist_ok=True)
        pending: dict[str, Path] = {}
        for k in range(0, len(scenarios), self.shard_size):
            chunk = scenarios[k : k + self.shard_size]
            name = f"{tag}-shard-{k // self.shard_size:03d}"
            shard = shard_dir / f"{name}.json"
            shard.write_text(json.dumps({"spec": spec.model_dump(), "scenarios": [s.model_dump() for s in chunk]}))
            cmd = self.build_command(name, f"shards/{tag}/{shard.name}")
            print("$", " ".join(shlex.quote(c) for c in cmd))
            if not self.dry_run:
                subprocess.run(cmd, check=True)
            pending[name] = shard.with_suffix(".results.json")
        if self.dry_run:
            return []
        results: list[RolloutResult] = []
        while pending:
            for name, res_path in list(pending.items()):
                if res_path.exists():
                    results += [RolloutResult.model_validate(r) for r in json.loads(res_path.read_text())]
                    del pending[name]
            if pending:
                time.sleep(self.poll_s)
        return results
