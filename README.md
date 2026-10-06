# RoboGauntlet

**Adaptive stress testing for robot policies.** Upload a policy and a task; RoboGauntlet plans a stress test, runs hundreds of perturbed simulations in parallel on Nebius Serverless Jobs, has Cosmos Reason 2 explain every failure, and steers each new round toward the policy's weak spots. You get a robustness score, the conditions that break the policy, and every failing scenario as a replayable test case.

Built for the Nebius x NVIDIA Global AI Hackathon, Physical AI track.

## Why

Robot policies pass in the lab and fail in deployment, usually under combinations nobody tested: a heavy object on worn gripper pads, a backlit scene with a slightly misaligned camera. Testing every combination is impossible, and random testing wastes most of its GPU time on cases the policy already handles. RoboGauntlet searches for failures the way an engineer would, but at cluster scale.

## How it works

```
Policy + task spec
      │
      ▼
Scenario planner ─── Nemotron 3 Ultra (+ Tavily research on known failure modes)
      │
      ├── Physics variants (Isaac Lab-Arena)     ├── Visual variants (Cosmos Transfer 2.5, planned)
      ▼
Rollout farm ─────── Nebius Serverless Jobs, one job per shard, fanned out in parallel
      │
      ▼
Failure judge ────── Cosmos Reason 2 on a Nebius Serverless Endpoint (vLLM)
      │
      ▼
Failure analyst ──── statistics in code + Nemotron 3 Super write-up ──┐
      │                                                               │ next round targets weak spots
      ▼                                                               │
Robustness report ◄───────────────────────────────────────────────────┘
```

1. **Plan.** Nemotron 3 Ultra reads the task and picks which perturbation axes to test and over what ranges. It can only choose axes the simulator supports, and ranges are clipped to the simulator's limits, so a model mistake can't break a run.
2. **Sanity check.** A few rollouts in the unperturbed scene. If the policy fails its own training conditions, the run stops and tells you to check the observation/action spec, instead of burning GPU hours.
3. **Adaptive rounds.** Round 1 is a space-filling Latin hypercube. Each later round splits its budget three ways: *exploit* (high estimated failure probability, kept spread out to map boundaries), *explore* (the least-tested parts of the space), and *regions* the analyst asked to probe.
4. **Judge.** The simulator says whether the task succeeded; Cosmos Reason 2 watches the video and says how it failed, including runs that "succeeded" unsafely. Every failure and a sample of successes are judged.
5. **Analyse.** Per-axis sensitivity and failure clusters are computed in code; Nemotron 3 Super turns them into a summary and proposes regions for the next round; Nemotron Nano names each failure cluster.
6. **Report.** A self-contained HTML page: robustness score with confidence interval, adaptive versus random comparison, failure clusters with the exact parameter ranges that trigger them, a sensitivity heatmap, and the round-by-round reasoning.

The robustness score is estimated only from uniformly sampled rollouts (the random baseline), because adaptive sampling deliberately over-represents failures.

## Model routing

| Job | Model | Calls |
|---|---|---|
| Plan the stress test | `nvidia/Nemotron-3-Ultra-550b-a55b` | 1 per campaign |
| Analyse failures, steer the next round | `nvidia/nemotron-3-super-120b-a12b` | 1 per round |
| Name failure clusters | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 1 per campaign |
| Judge rollout video | `nvidia/Cosmos-Reason2-8B` (vLLM) | 1 per failure + sampled successes |
| Policy under test | Isaac GR00T N1.6 (+ SONIC for humanoids) | every control step |

All Nemotron calls go through Nebius Token Factory's OpenAI-compatible API. Model IDs can be overridden with `RG_MODEL_ULTRA`, `RG_MODEL_SUPER` and `RG_MODEL_NANO`. Every call is counted per model in the report.

## Quickstart

```bash
pip install -e ".[dev]"

# Full pipeline on the synthetic mock backend, no keys needed
robogauntlet run examples/mug_task.json --offline --out runs/demo
open runs/demo/report.html

# With Nemotron on Token Factory, Tavily and Cosmos Reason 2
cp .env.example .env   # fill in keys, then: set -a; source .env; set +a
robogauntlet run examples/mug_task.json --out runs/live

# Adaptive vs random averaged over 10 seeds
robogauntlet benchmark examples/mug_task.json --offline --seeds 10

# Print the Nebius Serverless Jobs commands for one round, without launching
robogauntlet jobs-dry-run examples/mug_task.json --image <registry>/robogauntlet-worker:latest \
    --subnet-id <subnet_id> --volume <filesystem_id>:/data

pytest
```

## What the upload looks like

See `examples/mug_task.json`: the policy (a checkpoint, or an inference endpoint URL for teams that can't share weights), the robot embodiment, observation and action specs, and the task with its success criteria.

## Results so far (mock backend)

The mock backend fakes a pick-and-place policy with four small hidden failure modes, each triggered only by a combination of two or three conditions and covering 1 to 3% of the parameter space. Over 10 seeds, 4 rounds of 40 rollouts each:

| | Adaptive | Random (same budget) |
|---|---|---|
| Failing scenarios found | 43.4 | 19.6 |
| Failures per GPU-hour | 14.7 | 7.0 |
| Rollouts to find all 4 failure types | 84 | 115 (missed one type in 2 of 10 seeds) |

The failure clusters recover the hidden conditions, e.g. grasp slip at object mass 0.95 to 1.2 kg with friction 0.21 to 0.35. These numbers validate the search and analysis pipeline only; they say nothing about a real policy.

## Project status

| Component | Status |
|---|---|
| Planner, adaptive search, analyst, report, CLI | Working, tested |
| Nemotron calls via Token Factory | Working against an OpenAI-compatible server; needs a run with real keys |
| Tavily research | Implemented; needs a real key |
| Cosmos Reason 2 judge | Implemented for an OpenAI-compatible vLLM endpoint; needs a deployed endpoint |
| Nebius Serverless Jobs runner + worker | Implemented with dry run; needs a Nebius project to launch |
| Isaac Lab-Arena backend | Interface only; to be wired on an L40S machine |
| Cosmos Transfer 2.5 visual variants | Planned |
| Web app on a Serverless Endpoint | Planned (the report is the UI today) |

## Repository layout

```
src/robogauntlet/
  schemas.py      data model for every stage
  planner.py      Nemotron Ultra test plan + Tavily research
  search.py       Latin hypercube, adaptive exploit/explore/region sampling
  backends/       simulator interface, mock backend, Isaac Lab-Arena backend
  runner.py       local process pool and Nebius Serverless Jobs fan-out
  judge.py        Cosmos Reason 2 failure labels
  analyst.py      sensitivity, clusters, Nemotron Super/Nano write-up
  campaign.py     orchestration
  report.py       HTML report
jobs/             rollout worker and Dockerfile for Serverless Jobs
tests/
```

## Feedback on Nebius and NVIDIA tools

To be filled in during the build.

## License

Apache 2.0, see `LICENSE`.
