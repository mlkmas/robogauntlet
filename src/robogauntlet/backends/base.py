"""Interface every simulator backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schemas import Axis, CampaignSpec, RolloutResult, Scenario


class SimBackend(ABC):
    """Runs one closed-loop rollout of a policy in one perturbed scenario.

    The planner may only choose axes the backend says it can apply, so the
    language model can never invent a perturbation the simulator can't do.
    """

    name: str = "base"
    gpus_per_rollout: float = 1.0

    @abstractmethod
    def supported_axes(self, spec: CampaignSpec) -> list[Axis]:
        """Axes this backend can perturb, with their widest safe ranges."""

    @abstractmethod
    def run(self, scenario: Scenario, spec: CampaignSpec, out_dir: Path) -> RolloutResult:
        """Run one episode and write any frames or video under out_dir."""

    def nominal_params(self, spec: CampaignSpec) -> dict[str, float]:
        """Unperturbed scene, used by the sanity check."""
        return {a.name: (a.nominal if a.nominal is not None else (a.low + a.high) / 2) for a in self.supported_axes(spec)}
