"""Isaac Lab-Arena backend: the real simulator, run inside a Nebius Serverless Job.

STATUS: interface only. It needs a GPU machine with Isaac Lab-Arena installed
(an RTX-class GPU such as L40S, since camera rendering uses RT cores) and has
not been run yet. Each method lists exactly what to wire up.
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import Axis, CampaignSpec, RolloutResult, Scenario
from .base import SimBackend

# Perturbations Isaac Lab exposes through event terms / scene config.
ARENA_AXES = [
    Axis(name="light_intensity", kind="visual", low=0.2, high=1.5, unit="x", nominal=1.0,
         description="Dome light intensity multiplier"),
    Axis(name="light_azimuth", kind="visual", low=0.0, high=360.0, unit="deg", nominal=45.0,
         description="Key light direction"),
    Axis(name="object_x_offset", kind="scene", low=-0.15, high=0.15, unit="m", nominal=0.0,
         description="Object spawn offset along x"),
    Axis(name="object_yaw", kind="scene", low=0.0, high=90.0, unit="deg", nominal=0.0,
         description="Object spawn yaw"),
    Axis(name="object_mass", kind="physics", low=0.1, high=1.2, unit="kg", nominal=0.3,
         description="Rigid body mass"),
    Axis(name="friction", kind="physics", low=0.2, high=1.2, unit="mu", nominal=0.8,
         description="Physics material static friction"),
    Axis(name="camera_yaw_offset", kind="sensor", low=-15.0, high=15.0, unit="deg", nominal=0.0,
         description="Camera mount yaw error"),
    Axis(name="distractor_count", kind="scene", low=0, high=8, unit="objects", nominal=0, integer=True,
         description="Clutter objects sampled from the asset library"),
]


class IsaacArenaBackend(SimBackend):
    name = "isaac_arena"
    gpus_per_rollout = 1.0

    def supported_axes(self, spec: CampaignSpec) -> list[Axis]:
        return [a.model_copy() for a in ARENA_AXES]

    def run(self, scenario: Scenario, spec: CampaignSpec, out_dir: Path) -> RolloutResult:
        raise NotImplementedError(
            "Wire this up on a GPU box:\n"
            "1. Build the Arena env for spec.task.scene and apply scenario.params "
            "(light, spawn pose, mass, friction, camera offset, clutter).\n"
            "2. Load the policy: GR00T checkpoint from spec.policy.source, or call "
            "spec.policy.source as an inference endpoint when spec.policy.type == 'endpoint'.\n"
            "3. Step closed-loop at spec.action.control_hz until success or spec.task.timeout_s.\n"
            "4. Record the front camera to out_dir/<scenario.id>.mp4 and return RolloutResult "
            "with the Arena success flag and measured wall-clock GPU seconds."
        )
