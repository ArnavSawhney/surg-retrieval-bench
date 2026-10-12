"""Paired video-level bootstrap (pre-registration section 4.6) and its fast metrics.

A bootstrap resample draws the evaluated videos with replacement; a video drawn m
times contributes its frames m times, and every metric is recomputed on the pooled
frames. Re-sorting ~90k frames for each of 70 queries and 10,000 resamples is slow, so
this module computes the **same numbers** another way:

* each query's frames are sorted once (descending score) and split into groups of
  exactly equal score;
* in a resample, a frame of video v appears m_v times. Copies of a frame have equal
  scores, so they join that frame's tied group, and a group's size and relevant count
  become weighted sums: g = sum of m_v over its frames, r = sum over its relevant frames;
* the tie-neutral metrics (``srb.metrics``) depend only on those per-group counts, in
  score order.

So the result is identical to materialising the resampled frame list and calling
``srb.metrics`` on it; ``tests/test_resample.py`` checks this exactly on random cases.

AP of the groups uses the closed form of the per-group sum in ``srb.metrics``
(McSherry & Najork 2008): for a group of g slots, r relevant, after n_b frames of which
r_b are relevant, with c = (r-1)/(g-1) (0 if g = 1) and dH = H(n_b+g) - H(n_b),

    sum_{j=1..g} (r/g)(r_b + 1 + (j-1)c)/(n_b + j) = (r/g) [ (r_b+1) dH + c (g - (n_b+1) dH) ]

using sum_j 1/(n_b+j) = dH and sum_j (j-1)/(n_b+j) = g - (n_b+1) dH.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma

BOOT_SEED = 20261011  # pre-registration 4.6
B_CONFIRMATORY = 10_000
B_DESCRIPTIVE = 2_000  # the first 2,000 resamples of the same stream


def resample_counts(n_videos: int, B: int, seed: int = BOOT_SEED) -> np.ndarray:
    """(B, n_videos) int multiplicities. Row b is resample b of one fixed stream, so the
    first 2,000 rows of a B = 10,000 draw equal a B = 2,000 draw (row-major order)."""
    idx = np.random.default_rng(seed).integers(0, n_videos, size=(B, n_videos))
    return np.stack([np.bincount(row, minlength=n_videos) for row in idx])


class SortedQuery:
    """One query's ranking over the full evaluated set, sorted once.

    ``video`` holds each frame's position (0..V-1) in the sorted list of evaluated
    videos. Scores must be float32 or wider (fp16 ties are refused upstream).
    """

    def __init__(self, scores: np.ndarray, relevant: np.ndarray, video: np.ndarray):
        s = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-s, kind="stable")
        s = s[order]
        new = np.empty(len(s), dtype=bool)
        new[0] = True
        new[1:] = s[1:] != s[:-1]
        self.group = np.cumsum(new) - 1  # group id per sorted position
        self.n_groups = int(self.group[-1]) + 1
        self.rel = np.asarray(relevant, dtype=bool)[order]
        self.video = np.asarray(video)[order]

    def group_counts(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-group (g, r) for video multiplicities ``m``, empty groups dropped."""
        w = m[self.video].astype(np.float64)
        g = np.bincount(self.group, weights=w, minlength=self.n_groups)
        r = np.bincount(self.group, weights=w * self.rel, minlength=self.n_groups)
        keep = g > 0
        return g[keep], r[keep]


def ap_from_groups(g: np.ndarray, r: np.ndarray) -> float:
    """Tie-neutral AP from per-group sizes and relevant counts in score order."""
    R = r.sum()
    if R == 0:
        return float("nan")
    n_b = np.cumsum(g) - g
    r_b = np.cumsum(r) - r
    k = r > 0
    g, r, n_b, r_b = g[k], r[k], n_b[k], r_b[k]
    dH = digamma(n_b + g + 1) - digamma(n_b + 1)
    c = np.where(g > 1, (r - 1) / np.maximum(g - 1, 1), 0.0)
    return float(((r / g) * ((r_b + 1) * dH + c * (g - (n_b + 1) * dH))).sum() / R)


def hits_at_from_groups(g: np.ndarray, r: np.ndarray, k: float) -> float:
    """Expected relevant frames in the top k (a straddling group adds t * r / g)."""
    end = np.cumsum(g)
    i = int((end <= k).sum())  # groups entirely inside the top k
    hits = r[:i].sum()
    t = k - (end[i - 1] if i else 0.0)  # slots of the next group inside the top k
    if i < len(g) and t > 0:
        hits += t * r[i] / g[i]
    return float(hits)


def r_precision_from_groups(g: np.ndarray, r: np.ndarray) -> float:
    R = r.sum()
    return float("nan") if R == 0 else hits_at_from_groups(g, r, R) / R


def percentile_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval over resamples (axis 0), NaN resamples ignored."""
    a = (1 - level) / 2 * 100
    lo, hi = np.nanpercentile(x, [a, 100 - a], axis=0)
    return lo, hi


def bootstrap_p(diff: np.ndarray) -> float:
    """Two-sided p = 2 * min(share <= 0, share >= 0), capped at 1 (pre-registration 4.6)."""
    d = diff[~np.isnan(diff)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni: name -> (adjusted p, rejected at family-wise alpha)."""
    names = sorted(pvalues, key=pvalues.get)
    k, running, out = len(names), 0.0, {}
    for i, name in enumerate(names):
        running = max(running, min(1.0, (k - i) * pvalues[name]))
        out[name] = (running, running <= alpha)
    return out
