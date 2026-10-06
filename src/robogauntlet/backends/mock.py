"""A fast synthetic backend for developing the pipeline without a GPU.

It fakes a pick-and-place policy with a few small, hidden failure modes, the
kind real policies have: they only appear when two conditions combine. Random
sampling rarely lands in them, which is exactly what the adaptive search must
beat. Results from this backend are for testing the pipeline only; they say
nothing about any real policy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..schemas import Axis, CampaignSpec, RolloutResult, Scenario
from .base import SimBackend

AXES = [
    Axis(name="light_intensity", kind="visual", low=0.2, high=1.5, unit="x", nominal=1.0,
         description="Scene brightness relative to training"),
    Axis(name="light_azimuth", kind="visual", low=0.0, high=360.0, unit="deg", nominal=45.0,
         description="Direction of the key light around the table"),
    Axis(name="object_x_offset", kind="scene", low=-0.15, high=0.15, unit="m", nominal=0.0,
         description="Object displacement from its trained position"),
    Axis(name="object_yaw", kind="scene", low=0.0, high=90.0, unit="deg", nominal=0.0,
         description="Object rotation about the vertical axis"),
    Axis(name="object_mass", kind="physics", low=0.1, high=1.2, unit="kg", nominal=0.3,
         description="Mass of the manipulated object"),
    Axis(name="friction", kind="physics", low=0.2, high=1.2, unit="mu", nominal=0.8,
         description="Gripper-object friction coefficient"),
    Axis(name="camera_yaw_offset", kind="sensor", low=-15.0, high=15.0, unit="deg", nominal=0.0,
         description="Front camera mounting error"),
    Axis(name="distractor_count", kind="scene", low=0, high=8, unit="objects", nominal=0, integer=True,
         description="Clutter objects placed near the target"),
]


def _hidden_modes(p: dict[str, float]) -> list[tuple[str, float]]:
    """(category, failure probability) for each hidden mode the point falls in."""
    modes = []
    az = p["light_azimuth"] % 360
    # Each mode covers roughly 1-3% of the parameter space.
    if 160 <= az <= 205 and p["light_intensity"] > 1.3:
        modes.append(("perception_failure", 0.92))  # backlight glare
    if p["object_mass"] > 0.95 and p["friction"] < 0.35:
        modes.append(("grasp_slip", 0.9))
    if p["object_yaw"] > 70 and p["camera_yaw_offset"] > 10 and p["light_intensity"] < 0.6:
        modes.append(("missed_grasp", 0.88))  # rotated object, miscalibrated camera, dim light
    if p["distractor_count"] >= 7 and p["object_x_offset"] < -0.10:
        modes.append(("collision", 0.85))
    return modes


class MockBackend(SimBackend):
    name = "mock"
    gpus_per_rollout = 1.0

    def __init__(self, render: bool = True) -> None:
        self.render = render

    def supported_axes(self, spec: CampaignSpec) -> list[Axis]:
        return [a.model_copy() for a in AXES]

    def run(self, scenario: Scenario, spec: CampaignSpec, out_dir: Path) -> RolloutResult:
        rng = np.random.default_rng(scenario.seed)
        p = {a.name: scenario.params.get(a.name, a.nominal) for a in AXES}
        modes = _hidden_modes(p)
        hint = None
        for category, prob in modes:
            if rng.random() < prob:
                hint = category
                break
        if hint is None and rng.random() < 0.03:
            hint = "timeout"  # background noise, as in any real policy
        success = hint is None
        sim_time = float(rng.uniform(12, 18) if success else spec.task.timeout_s * rng.uniform(0.4, 1.0))
        # Wall-clock GPU cost of one Isaac Lab rollout, roughly 4x slower than real time.
        gpu_seconds = sim_time * 4.0 * self.gpus_per_rollout
        frames: list[str] = []
        if self.render:
            out_dir.mkdir(parents=True, exist_ok=True)
            frames.append(str(_render_topdown(scenario, p, hint, out_dir)))
        return RolloutResult(
            scenario_id=scenario.id,
            success=success,
            sim_time_s=sim_time,
            gpu_seconds=gpu_seconds,
            metrics={"final_dist_to_target_m": 0.01 if success else float(rng.uniform(0.08, 0.3))},
            frames=frames,
            sim_failure_hint=hint,
        )


def _render_topdown(scenario: Scenario, p: dict[str, float], hint: str | None, out_dir: Path) -> Path:
    """A schematic top-down keyframe so the report gallery has something to show."""
    from PIL import Image, ImageDraw

    w, h = 240, 180
    shade = int(max(40, min(250, 170 * p["light_intensity"])))
    img = Image.new("RGB", (w, h), (shade, shade, int(shade * 0.95)))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, w - 20, h - 20], outline=(90, 80, 70), width=2, fill=(min(255, shade + 20),) * 3)
    # light direction arrow
    ang = math.radians(p["light_azimuth"])
    cx, cy = w / 2, h / 2
    d.line([cx + 80 * math.cos(ang), cy + 60 * math.sin(ang), cx + 55 * math.cos(ang), cy + 40 * math.sin(ang)],
           fill=(230, 180, 40), width=3)
    # distractors
    rng = np.random.default_rng(scenario.seed + 7)
    for _ in range(int(p["distractor_count"])):
        x, y = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(120, 120, 140))
    # target object, rotated square
    ox = cx + p["object_x_offset"] * 500
    yaw = math.radians(p["object_yaw"])
    pts = [(ox + 12 * math.cos(yaw + k * math.pi / 2), cy + 12 * math.sin(yaw + k * math.pi / 2)) for k in range(4)]
    d.polygon(pts, fill=(200, 60, 50))
    # gripper end position
    gx = ox + (0 if hint is None else 35)
    gy = cy + (0 if hint is None else -25)
    d.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], outline=(20, 20, 20), width=2)
    d.text((26, 24), "mock render", fill=(60, 60, 60))
    path = out_dir / f"{scenario.id}.png"
    img.save(path)
    return path
