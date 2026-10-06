import json
from pathlib import Path

import numpy as np
import pytest

from robogauntlet.analyst import _validate_regions, failure_clusters, sensitivities
from robogauntlet.backends.mock import AXES
from robogauntlet.campaign import Campaign, wilson
from robogauntlet.config import Settings
from robogauntlet.llm import extract_json
from robogauntlet.planner import _clip_axes
from robogauntlet.report import build_report
from robogauntlet.runner import NebiusJobsRunner
from robogauntlet.schemas import CampaignSpec, FailureLabel, Scenario
from robogauntlet.search import latin_hypercube, propose_adaptive

SPEC = CampaignSpec.model_validate_json((Path(__file__).parent.parent / "examples" / "mug_task.json").read_text())


def test_extract_json_handles_reasoning_and_fences():
    assert extract_json('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"b": [1, 2]} hope that helps') == {"b": [1, 2]}


def test_planner_cannot_invent_axes_or_exceed_ranges():
    axes = _clip_axes([
        {"name": "object_mass", "low": -5, "high": 99},      # clipped to backend limits
        {"name": "gravity", "low": 0, "high": 20},           # unsupported: dropped
        {"name": "friction", "low": 0.3, "high": 0.6},
    ], AXES)
    assert [a.name for a in axes] == ["object_mass", "friction"]
    assert (axes[0].low, axes[0].high) == (0.1, 1.2)
    assert (axes[1].low, axes[1].high) == (0.3, 0.6)


def test_analyst_regions_are_validated():
    regions = _validate_regions([
        {"bounds": {"friction": [0.9, 0.1], "made_up": [0, 1]}, "hypothesis": "x"},
        {"bounds": {"object_mass": [5, 9]}},  # entirely out of range: dropped
    ], AXES)
    assert len(regions) == 1 and regions[0].bounds == {"friction": (0.2, 0.9)}


def test_latin_hypercube_stratifies_each_axis():
    U = latin_hypercube(20, 3, np.random.default_rng(0))
    for j in range(3):
        assert sorted(np.floor(U[:, j] * 20).astype(int)) == list(range(20))


def test_adaptive_search_concentrates_near_failures():
    # 6 axes, failures only where axes 0 and 1 are both high (4% of the space).
    from robogauntlet.schemas import Axis
    rng = np.random.default_rng(0)
    X = rng.random((200, 6))
    y = (X[:, 0] > 0.8) & (X[:, 1] > 0.8)
    axes = [Axis(name=f"a{i}", kind="scene", low=0, high=1) for i in range(6)]
    U, origins = propose_adaptive(X, y, 40, rng, axes, [], round_idx=1)
    exploit = U[[o == "exploit" for o in origins]]
    near = ((exploit[:, 0] > 0.6) & (exploit[:, 1] > 0.6)).mean()
    assert near > 3 * 0.16  # uniform sampling would land there 16% of the time


def test_clusters_find_the_explanatory_axes():
    rng = np.random.default_rng(1)
    scen, labels = [], []
    for i in range(400):
        p = {a.name: a.denormalize(rng.random()) for a in AXES}
        fail = p["object_mass"] > 0.95 and p["friction"] < 0.35
        scen.append(Scenario(id=str(i), params=p, round=1, origin="random", seed=i))
        labels.append(FailureLabel(scenario_id=str(i), failed=fail, category="grasp_slip" if fail else "none"))
    (c,) = failure_clusters(AXES, scen, labels)
    assert set(c.bounds) == {"object_mass", "friction"}
    sens = sensitivities(AXES, scen, np.array([l.failed for l in labels]))
    assert {sens[0].axis, sens[1].axis} == {"object_mass", "friction"}


def test_wilson_interval():
    p, lo, hi = wilson(90, 100)
    assert p == 0.9 and lo < 0.9 < hi and hi - lo < 0.15


def test_nebius_job_command():
    cmd = NebiusJobsRunner(image="cr.example/rg:1", subnet_id="sub-1", volume="fs-1:/data").build_command("j1", "shards/x.json")
    assert cmd[:4] == ["nebius", "ai", "job", "create"]
    assert "gpu-l40s-a" in cmd and "/app/jobs/rollout_worker.py --shard /data/shards/x.json --backend isaac_arena" in cmd


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    Campaign(SPEC, Settings(offline=True), rounds=3, budget=30, seed=3, workers=1, run_dir=d, verbose=False).run()
    return d


def test_campaign_end_to_end(run_dir):
    s = json.loads((run_dir / "summary.json").read_text())
    assert not s["aborted"]
    assert len(s["curves"]["adaptive"]) == 3 and s["curves"]["adaptive"][-1]["rollouts"] == 90
    assert s["curves"]["adaptive"][-1]["failures"] > s["curves"]["random"][-1]["failures"]
    assert 0 < s["robustness"]["success_rate"] < 1
    html = build_report(run_dir).read_text()
    assert "Stress test report" in html and "Synthetic mock backend" in html


def test_sanity_check_aborts_on_broken_policy(tmp_path, monkeypatch):
    import robogauntlet.backends.mock as mock

    monkeypatch.setattr(mock, "_hidden_modes", lambda p: [("missed_grasp", 1.0)])
    s = Campaign(SPEC, Settings(offline=True), rounds=1, budget=5, workers=1, run_dir=tmp_path, verbose=False).run()
    assert s["aborted"] and "nominal scene" in s["reason"]
