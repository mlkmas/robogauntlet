"""Failure judge: Cosmos Reason 2 watches each rollout and says how it failed.

The simulator says *whether* the task succeeded. Cosmos Reason 2, served with
vLLM on a Nebius Serverless Endpoint (OpenAI-compatible), says *how* it failed,
and can also flag runs that "succeeded" unsafely. Without an endpoint, labels
fall back to the simulator's own failure hint.
"""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Settings
from .llm import UsageLedger, extract_json
from .schemas import FailureLabel, RolloutResult

CATEGORIES = ["missed_grasp", "grasp_slip", "collision", "perception_failure", "unsafe_motion",
              "wrong_target", "timeout", "other", "none"]

JUDGE_PROMPT = """You are reviewing a recorded robot manipulation episode.
Task given to the robot: "{instruction}".
The simulator reports the task as {outcome}.
Watch the episode and answer with JSON only:
{{"failed": bool, "category": one of {categories},
 "timestamp_s": number or null (when the problem first becomes visible),
 "explanation": one sentence describing what went wrong, or "none"}}
Mark "unsafe_motion" if the robot succeeded but moved dangerously (hit objects, swept near people, dropped items)."""


def _media_part(path: str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    if path.endswith(".mp4"):
        return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{data}"}}
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _fallback(r: RolloutResult) -> FailureLabel:
    return FailureLabel(
        scenario_id=r.scenario_id,
        failed=not r.success,
        category="none" if r.success else (r.sim_failure_hint if r.sim_failure_hint in CATEGORIES else "other"),
        explanation="" if r.success else f"Simulator reported failure ({r.sim_failure_hint or 'unknown'}).",
        source="sim",
    )


class CosmosJudge:
    def __init__(self, settings: Settings, ledger: UsageLedger | None = None, workers: int = 8) -> None:
        self.settings = settings
        self.ledger = ledger or UsageLedger()
        self.workers = workers
        self._client = None
        if settings.judge_enabled:
            from openai import OpenAI

            self._client = OpenAI(base_url=settings.cosmos_reason_url, api_key=settings.cosmos_reason_key or "EMPTY")

    def _judge_one(self, r: RolloutResult, instruction: str) -> FailureLabel:
        if self._client is None or not r.frames:
            return _fallback(r)
        prompt = JUDGE_PROMPT.format(instruction=instruction, outcome="SUCCESS" if r.success else "FAILURE",
                                     categories=json.dumps(CATEGORIES))
        content = [_media_part(f) for f in r.frames[:4]] + [{"type": "text", "text": prompt}]
        try:
            resp = self._client.chat.completions.create(
                model=self.settings.cosmos_reason_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1024,
                temperature=0.1,
            )
            self.ledger.record(self.settings.cosmos_reason_model, "judge rollout video", resp.usage)
            ans = extract_json(resp.choices[0].message.content or "")
            category = ans.get("category") if ans.get("category") in CATEGORIES else "other"
            failed = bool(ans.get("failed", not r.success)) or not r.success
            return FailureLabel(
                scenario_id=r.scenario_id,
                failed=failed,
                category=category if failed else "none",
                timestamp_s=ans.get("timestamp_s"),
                explanation=str(ans.get("explanation", ""))[:300],
                source="cosmos_reason",
            )
        except Exception as err:
            print(f"[cosmos-reason] {r.scenario_id}: {err}; falling back to sim label")
            return _fallback(r)

    def judge(self, results: list[RolloutResult], instruction: str, success_sample_every: int = 5) -> list[FailureLabel]:
        """Spend VLM calls where they add information: every failure, plus every Nth success
        (to catch unsafe successes). Everything else gets the simulator label."""
        if self._client is None:
            return [_fallback(r) for r in results]
        successes_seen = 0
        selected: list[bool] = []
        for r in results:
            if r.success:
                selected.append(successes_seen % success_sample_every == 0)
                successes_seen += 1
            else:
                selected.append(True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            judged = list(pool.map(lambda r: self._judge_one(r, instruction),
                                   [r for r, s in zip(results, selected) if s]))
        it = iter(judged)
        return [next(it) if s else _fallback(r) for r, s in zip(results, selected)]
