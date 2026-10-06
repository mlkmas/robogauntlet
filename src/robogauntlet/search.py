"""Choosing which scenarios to run next.

All sampling happens in the unit cube [0, 1]^d; axes map it back to real units.

Random search: uniform samples, the baseline and the unbiased robustness estimate.
Adaptive search, per round:
  - round 1 is a space-filling Latin hypercube;
  - later rounds split the budget between
      exploit  - candidates with high estimated failure probability, kept diverse,
      explore  - points farthest from anything tested so far,
      regions  - samples inside boxes the analyst (Nemotron) asked to probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import Axis, Region


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        cut[:, j] = cut[rng.permutation(n), j]
    return cut


def uniform(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, d))


def _min_dist(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if len(ref) == 0:
        return np.full(len(points), np.sqrt(points.shape[1]))
    diff = points[:, None, :] - ref[None, :, :]
    return np.sqrt((diff**2).sum(-1)).min(1)


def failure_probability(cands: np.ndarray, X: np.ndarray, y: np.ndarray, bandwidth: float = 0.18,
                        prior: float = 0.05, prior_weight: float = 0.5) -> np.ndarray:
    """Kernel-smoothed failure rate around each candidate (Nadaraya-Watson with a prior)."""
    if len(X) == 0:
        return np.full(len(cands), prior)
    d2 = ((cands[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * bandwidth**2))
    return (w @ y.astype(float) + prior * prior_weight) / (w.sum(1) + prior_weight)


def _greedy_diverse(scores: np.ndarray, cands: np.ndarray, k: int, min_sep: float, taken: np.ndarray) -> list[int]:
    order = np.argsort(-scores)
    chosen: list[int] = []
    picked = taken.copy() if len(taken) else np.empty((0, cands.shape[1]))
    for i in order:
        if len(chosen) >= k:
            break
        if len(picked) and _min_dist(cands[i : i + 1], picked)[0] < min_sep:
            continue
        chosen.append(int(i))
        picked = np.vstack([picked, cands[i]])
    return chosen


def _farthest_points(cands: np.ndarray, X: np.ndarray, k: int) -> list[int]:
    chosen: list[int] = []
    ref = X.copy() if len(X) else np.empty((0, cands.shape[1]))
    dist = _min_dist(cands, ref)
    for _ in range(k):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.sqrt(((cands - cands[i]) ** 2).sum(1)))
    return chosen


def region_to_unit(region: Region, axes: list[Axis]) -> np.ndarray:
    """Convert a region in real units to [lo, hi] rows in the unit cube."""
    box = np.tile(np.array([0.0, 1.0]), (len(axes), 1))
    for j, a in enumerate(axes):
        if a.name in region.bounds:
            lo, hi = region.bounds[a.name]
            u = sorted((np.clip(a.normalize(lo), 0, 1), np.clip(a.normalize(hi), 0, 1)))
            if u[1] - u[0] > 1e-6:
                box[j] = u
    return box


@dataclass
class AdaptiveConfig:
    exploit_frac: float = 0.45
    explore_frac: float = 0.30
    region_frac: float = 0.25
    candidates: int = 4000
    min_sep: float = 0.12


def propose_adaptive(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
    axes: list[Axis],
    regions: list[Region],
    round_idx: int,
    cfg: AdaptiveConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (unit-cube points, origin label per point)."""
    cfg = cfg or AdaptiveConfig()
    d = len(axes)
    if round_idx == 0 or len(X) == 0:
        return latin_hypercube(n, d, rng), ["space_filling"] * n

    n_region = int(round(n * cfg.region_frac)) if regions else 0
    n_explore = int(round(n * cfg.explore_frac))
    n_exploit = n - n_region - n_explore

    pts: list[np.ndarray] = []
    origins: list[str] = []

    # Regions requested by the analyst
    if n_region:
        boxes = [region_to_unit(r, axes) for r in regions]
        for i in range(n_region):
            box = boxes[i % len(boxes)]
            pts.append(box[:, 0] + rng.random(d) * (box[:, 1] - box[:, 0]))
            origins.append("llm_region")

    cands = rng.random((cfg.candidates, d))
    taken = np.vstack(pts) if pts else np.empty((0, d))

    # Exploit: likely failures, spread out so we map boundaries instead of hammering one spot
    p = failure_probability(cands, X, y)
    idx = _greedy_diverse(p, cands, n_exploit, cfg.min_sep, taken)
    for i in idx:
        pts.append(cands[i])
        origins.append("exploit")

    # Explore: the least-tested parts of the space
    tested = np.vstack([X, np.vstack(pts)]) if pts else X
    idx = _farthest_points(cands, tested, n - len(pts))
    for i in idx:
        pts.append(cands[i])
        origins.append("explore")

    return np.vstack(pts)[:n], origins[:n]
