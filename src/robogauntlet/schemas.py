"""Data model shared by every stage of the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# What the user uploads
# --------------------------------------------------------------------------


class PolicySpec(BaseModel):
    """The policy under test: a checkpoint we run, or an endpoint we call."""

    type: Literal["checkpoint", "endpoint", "mock"] = "checkpoint"
    source: str = Field(description="hf://org/repo, a local path, or an https:// inference URL")
    embodiment: str = "franka_panda"
    label: str | None = Field(default=None, description="Display name in the report")


class ObservationSpec(BaseModel):
    cameras: list[str] = ["front_rgb", "wrist_rgb"]
    resolution: tuple[int, int] = (224, 224)
    state: list[str] = ["joint_pos", "gripper_pos"]


class ActionSpec(BaseModel):
    type: Literal["joint_position", "ee_delta", "joint_velocity"] = "joint_position"
    chunk_size: int = 16
    control_hz: int = 30


class TaskSpec(BaseModel):
    instruction: str
    scene: str = "arena://kitchen_shelf_v1"
    success: str = "object_at_target AND upright"
    timeout_s: float = 30.0
    constraints: list[str] = []


class CampaignSpec(BaseModel):
    """Everything in one upload file (see examples/mug_task.json)."""

    policy: PolicySpec
    observation: ObservationSpec = ObservationSpec()
    action: ActionSpec = ActionSpec()
    task: TaskSpec


# --------------------------------------------------------------------------
# Test plan produced by the planner
# --------------------------------------------------------------------------

AxisKind = Literal["physics", "visual", "sensor", "scene"]


class Axis(BaseModel):
    """One perturbation dimension, e.g. object mass between 0.1 and 1.2 kg."""

    name: str
    kind: AxisKind
    low: float
    high: float
    unit: str = ""
    nominal: float | None = None
    integer: bool = False
    description: str = ""

    def denormalize(self, u: float) -> float:
        v = self.low + float(u) * (self.high - self.low)
        return float(round(v)) if self.integer else v

    def normalize(self, v: float) -> float:
        span = self.high - self.low
        return 0.0 if span == 0 else (float(v) - self.low) / span


class StressPlan(BaseModel):
    task_summary: str
    axes: list[Axis]
    rationale: str = ""
    known_failure_modes: list[str] = []
    sources: list[str] = []
    planner_model: str = "offline-default"


class Region(BaseModel):
    """A box in parameter space the analyst wants probed next round."""

    bounds: dict[str, tuple[float, float]]
    hypothesis: str = ""


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

Origin = Literal["random", "space_filling", "exploit", "explore", "llm_region", "nominal"]


class Scenario(BaseModel):
    id: str
    params: dict[str, float]
    round: int
    origin: Origin
    seed: int
    strategy: Literal["adaptive", "random", "sanity"] = "adaptive"


class RolloutResult(BaseModel):
    scenario_id: str
    success: bool
    sim_time_s: float
    gpu_seconds: float
    metrics: dict[str, float] = {}
    frames: list[str] = Field(default_factory=list, description="Paths to keyframes or a video")
    sim_failure_hint: str | None = None


FailureCategory = Literal[
    "missed_grasp",
    "grasp_slip",
    "collision",
    "perception_failure",
    "unsafe_motion",
    "wrong_target",
    "timeout",
    "other",
    "none",
]


class FailureLabel(BaseModel):
    scenario_id: str
    failed: bool
    category: FailureCategory
    timestamp_s: float | None = None
    explanation: str = ""
    source: Literal["cosmos_reason", "sim"] = "sim"


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


class AxisSensitivity(BaseModel):
    axis: str
    bin_edges: list[float]
    failure_rate_by_bin: list[float | None]
    spread: float


class FailureCluster(BaseModel):
    id: int
    size: int
    categories: dict[str, int]
    bounds: dict[str, tuple[float, float]]
    example_ids: list[str]


class AnalystReport(BaseModel):
    round: int
    summary: str
    sensitivities: list[AxisSensitivity]
    clusters: list[FailureCluster]
    next_regions: list[Region]
    analyst_model: str = "offline-default"
