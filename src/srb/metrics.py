"""Retrieval metrics for text-to-frame search on surgical video.

Pure NumPy (plus ``fractions`` for the exact mode). No torch, no model code, no I/O.

Why this metric set
-------------------
Each text query ("clipping the cystic duct") has *thousands* of relevant frames in
the Cholec80 test split, because relevance is defined by the ground-truth phase or
tool label of a 1 fps frame. With R that large:

* ``Recall@k`` is capped at ``k / R`` (Recall@10 <= 0.0003 for a perfect model), so
  it measures R, not the model.
* ``Hit@k`` is ~0.99 for a *random* ranker on the two long phases.
* ``P@10`` and ``nDCG@10`` saturate at 1.0 for weak models, strong models and for a
  degenerate "one-clip shortcut" ranker alike.

So the primary metric is full-ranking **average precision (AP)**, macro-averaged over
queries (**mAP**), always reported next to the **random-ranker baseline** (the
prevalence ``R / N``) and the lift ``AP / prevalence``. **R-Precision** is the
secondary metric. ``P@10`` / ``P@50`` are descriptive only and never used to rank
models; ``hit_at_k`` is kept for debugging. See ``docs/metrics.md`` and
``docs/metric_rationale.py`` for the numbers behind this.

Conventions (all functions)
---------------------------
Ties: tie-neutral expected values
    Frames with exactly equal scores form a *tied group*. Every metric returns its
    **expected value under a uniformly random ordering within each tied group**
    (McSherry & Najork, "Computing information retrieval performance measures
    efficiently in the presence of tied scores", ECIR 2008). Groups are processed
    in descending score order. For a group of ``g`` frames, ``r`` of them relevant,
    preceded by ``n_b`` frames of which ``r_b`` are relevant:

    * AP: the group contributes
      ``sum_{j=1..g} (r/g) * (r_b + 1 + (j-1)(r-1)/(g-1)) / (n_b + j)``
      (with ``(r-1)/(g-1) := 0`` when ``g == 1``); the total is divided by ``R``.
      Exact by linearity of expectation: ``r/g`` is the chance that slot ``j`` holds
      a relevant frame, and given that, the expected number of relevant frames in
      slots ``1..j`` is ``r_b + 1 + (j-1)(r-1)/(g-1)``.
    * P@k and R-Precision: a group straddling the cutoff with ``t`` of its ``g``
      slots inside the top ``k`` contributes ``t * r / g`` relevant frames.
    * Hit@k: if no relevant frame precedes the straddling group,
      ``P(hit) = 1 - C(g - r, t) / C(g, t)``.

    Why: ties are *not* rare here. Embeddings are stored in fp16, real data has true
    duplicates (black and out-of-body frames), and a frame-index tie-break would let
    video order leak into the score. On tie-free scores every metric equals the plain
    (sorted) definition, and AP equals ``sklearn.metrics.average_precision_score``.
    With ties, sklearn differs: it assigns every frame in a tied group the precision
    at the end of the group, which is not the expectation.

    Every metric takes ``exact=True``, which runs the same formulas in
    ``fractions.Fraction`` arithmetic and returns a ``Fraction``. That mode is slow
    and exists so the tests can compare against brute-force enumeration with exact
    equality.

Score precision
    Scores are cast to float64 before grouping. A **float16 score array raises**:
    fp16 has so few distinct values near a cosine similarity of ~0.2-0.3 that most
    frames would tie (a 98.5k x 1152-d fp16 simulation gave 3,818 distinct scores for
    98.5k frames). Compute similarities in float32 or higher, *after* upcasting the
    stored fp16 embeddings. Use ``count_ties`` to report how many ties remain.

Undefined values
    A query with **zero** relevant frames has no defined AP or R-Precision. These
    functions return ``nan`` and emit a ``RuntimeWarning``. They never silently
    return 0.0, which would look like a real (bad) score.

Negative / control queries
    Queries such as "a car on a road" have no relevant frames by construction. They
    are **excluded** from mAP and evaluated separately as a score-calibration check
    (a later week). Do not feed them to ``mean_ap``.

Inputs
    ``scores``: 1-D, finite, float-castable (not float16), one score per frame (higher
    = more similar). ``relevant``: 1-D, same length, boolean or 0/1 ground truth.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np

__all__ = [
    "rank_order",
    "count_ties",
    "average_precision",
    "r_precision",
    "precision_at_k",
    "hit_at_k",
    "random_baseline_ap",
    "mean_ap",
]


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _as_scores(scores) -> np.ndarray:
    """Validate scores -> 1-D finite float64. Refuses float16 (see module docstring)."""
    raw = np.asarray(scores)
    if raw.dtype == np.float16:
        raise TypeError(
            "scores are float16: fp16 similarities tie massively and would distort "
            "every metric. Upcast the embeddings and compute scores in float32 or higher."
        )
    scores = raw.astype(np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contains nan or inf; fix the encoder, do not rank nans")
    return scores


def _check(scores, relevant) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalise (scores, relevant) -> (float64 scores, bool relevant)."""
    scores = _as_scores(scores)
    relevant = np.asarray(relevant)

    if relevant.ndim != 1:
        raise ValueError(f"relevant must be 1-D, got shape {relevant.shape}")
    if scores.shape != relevant.shape:
        raise ValueError(
            f"scores and relevant must have the same length, "
            f"got {scores.shape[0]} and {relevant.shape[0]}"
        )
    if scores.size == 0:
        raise ValueError("scores is empty: nothing to rank")

    if relevant.dtype == bool:
        rel = relevant
    else:
        rel_num = np.asarray(relevant, dtype=np.float64)
        if not np.all(np.isin(rel_num, (0.0, 1.0))):
            raise ValueError("relevant must be boolean or 0/1 (relevance here is binary)")
        rel = rel_num.astype(bool)
    return scores, rel


def _check_k(k: int, n: int) -> int:
    """Validate k and clamp it to the number of ranked frames."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        warnings.warn(
            f"k={k} exceeds the number of frames N={n}; using k={n}. "
            "The denominator is N, not k, so the value is not comparable to a "
            "P@k measured on a larger candidate pool.",
            RuntimeWarning,
            stacklevel=3,
        )
        return n
    return k


def _warn_undefined(name: str) -> float:
    warnings.warn(
        f"{name} is undefined for a query with 0 relevant frames; returning nan. "
        "Negative/control queries belong in the calibration check, not in mAP.",
        RuntimeWarning,
        stacklevel=3,
    )
    return float("nan")


# --------------------------------------------------------------------------- #
# ranking and tied groups
# --------------------------------------------------------------------------- #
def rank_order(scores) -> np.ndarray:
    """Indices of ``scores`` from most to least similar, for *display* only.

    Descending score; ties broken by ascending frame index via a stable sort, so the
    same scores always give the same list. **No metric depends on this tie-break**:
    the metrics are expectations over the order within each tied group.
    """
    scores = _as_scores(scores)
    return np.argsort(-scores, kind="stable")


def _groups(scores: np.ndarray, rel: np.ndarray):
    """Tied groups in descending score order.

    Returns int64 arrays ``(g, r, n_b, r_b)``: group size, relevant frames in the
    group, frames before the group, relevant frames before the group.
    """
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    g = np.diff(np.r_[starts, s.size])
    r = np.add.reduceat(rel[order].astype(np.int64), starts)
    n_b = starts.astype(np.int64)
    r_b = np.cumsum(r) - r
    return g, r, n_b, r_b


def count_ties(scores) -> tuple[int, int]:
    """``(n_tied_frames, n_groups)``: frames that share their score with at least one
    other frame, and the number of such tied groups (groups of size >= 2).

    ``(0, 0)`` means the scores are tie-free. Report this per query in every
    retrieval output.
    """
    scores = _as_scores(scores)
    _, counts = np.unique(scores, return_counts=True)
    tied = counts[counts > 1]
    return int(tied.sum()), int(tied.size)


def _cutoff(g, r, n_b, k: int):
    """Relevant frames fully above cutoff ``k``, plus the straddling group (or None).

    Returns ``(r_full, straddle)`` where ``straddle = (t, g_i, r_i)`` with ``t`` the
    straddling group's slots inside the top ``k`` (0 < t < g_i), or ``None`` when the
    cutoff falls exactly on a group boundary.
    """
    ends = n_b + g
    i = int(np.searchsorted(ends, k, side="left"))  # first group ending at or after k
    if ends[i] == k:
        return int(r[: i + 1].sum()), None
    return int(r[:i].sum()), (int(k - n_b[i]), int(g[i]), int(r[i]))


# --------------------------------------------------------------------------- #
# primary metric
# --------------------------------------------------------------------------- #
def average_precision(scores, relevant, *, exact: bool = False):
    """Full-ranking average precision for one query, tie-neutral.

    ``AP = (1 / R) * sum over relevant ranks i of P@i``. With ties, this is the
    expected AP over uniformly random orderings within each tied group (formula in
    the module docstring). Every relevant frame contributes, so AP is not capped by
    a cutoff and cannot be saturated by getting one clip right.

    Returns ``nan`` (with a ``RuntimeWarning``) if ``R == 0``. ``exact=True``
    returns a ``Fraction``.
    """
    scores, rel = _check(scores, relevant)
    R = int(rel.sum())
    if R == 0:
        return _warn_undefined("average_precision")
    g, r, n_b, r_b = _groups(scores, rel)

    if exact:
        total = Fraction(0)
        for gi, ri, nbi, rbi in zip(g.tolist(), r.tolist(), n_b.tolist(), r_b.tolist()):
            if ri == 0:
                continue
            c = Fraction(ri - 1, gi - 1) if gi > 1 else Fraction(0)
            p_rel = Fraction(ri, gi)
            total += sum(p_rel * (rbi + 1 + (j - 1) * c) / (nbi + j) for j in range(1, gi + 1))
        return total / R

    # Vectorised over frames: expand the group statistics to one entry per slot.
    keep = r > 0
    g, r, n_b, r_b = g[keep], r[keep], n_b[keep], r_b[keep]
    c = np.where(g > 1, (r - 1) / np.maximum(g - 1, 1), 0.0)
    rep = lambda a: np.repeat(a, g)  # noqa: E731
    j = np.arange(1, g.sum() + 1) - rep(np.cumsum(g) - g)  # 1..g within each group
    terms = rep(r / g) * (rep(r_b) + 1 + (j - 1) * rep(c)) / (rep(n_b) + j)
    return float(terms.sum() / R)


# --------------------------------------------------------------------------- #
# secondary metric
# --------------------------------------------------------------------------- #
def _expected_hits_at(scores, rel, k: int, exact: bool):
    g, r, n_b, _ = _groups(scores, rel)
    r_full, straddle = _cutoff(g, r, n_b, k)
    if straddle is None:
        return Fraction(r_full) if exact else float(r_full)
    t, gi, ri = straddle
    return r_full + (Fraction(t * ri, gi) if exact else t * ri / gi)


def r_precision(scores, relevant, *, exact: bool = False):
    """Expected precision in the top ``R``, where ``R`` is the number of relevant frames.

    Self-normalising: a perfect ranker scores 1.0 whatever R is, and a random
    ranker scores about the prevalence. Tie-neutral (see module docstring).
    Returns ``nan`` (with a ``RuntimeWarning``) if ``R == 0``.
    """
    scores, rel = _check(scores, relevant)
    R = int(rel.sum())
    if R == 0:
        return _warn_undefined("r_precision")
    hits = _expected_hits_at(scores, rel, R, exact)
    return hits / R if exact else float(hits / R)


# --------------------------------------------------------------------------- #
# descriptive / debugging
# --------------------------------------------------------------------------- #
def precision_at_k(scores, relevant, k: int, *, exact: bool = False):
    """Expected fraction of the top ``k`` frames that are relevant.

    **Descriptive only.** With thousands of relevant frames per query this
    saturates at 1.0 for weak and strong models alike, so never rank models by it.
    ``k`` is clamped to ``N`` with a warning if ``k > N``. Tie-neutral.
    """
    scores, rel = _check(scores, relevant)
    k = _check_k(k, rel.size)
    hits = _expected_hits_at(scores, rel, k, exact)
    return hits / k if exact else float(hits / k)


def hit_at_k(scores, relevant, k: int, *, exact: bool = False):
    """Probability that at least one relevant frame is in the top ``k``.

    Without ties this is 1.0 or 0.0. If the cutoff splits a tied group with ``t`` of
    its ``g`` slots inside the top ``k`` and no relevant frame ranks above that
    group, it is ``1 - C(g - r, t) / C(g, t)``.

    **Sanity check only.** A random ranker gets ~0.99 on the long Cholec80 phases,
    so it cannot discriminate between models.
    """
    scores, rel = _check(scores, relevant)
    k = _check_k(k, rel.size)
    g, r, n_b, _ = _groups(scores, rel)
    r_full, straddle = _cutoff(g, r, n_b, k)
    one = Fraction(1) if exact else 1.0
    if r_full > 0:
        return one
    if straddle is None:
        return 0 * one
    t, gi, ri = straddle
    if t > gi - ri:  # more slots than irrelevant frames: a hit is certain
        return one
    if exact:
        return 1 - Fraction(math.comb(gi - ri, t), math.comb(gi, t))
    # C(g-r, t) / C(g, t) = prod_{i<t} (g-r-i) / (g-i), summed in log space
    i = np.arange(t)
    return float(-np.expm1(np.log((gi - ri - i) / (gi - i)).sum()))


# --------------------------------------------------------------------------- #
# baseline and aggregation
# --------------------------------------------------------------------------- #
def random_baseline_ap(relevant) -> float:
    """Reference AP of a ranker that ignores the query: the prevalence ``R / N``.

    Report this next to every AP, and the lift ``AP / prevalence``. An AP at or
    below prevalence means the model carries no signal for that query.

    Strictly this is the large-``N`` limit of the expected AP of a uniformly random
    ranker; the exact expectation is marginally higher (``docs/metric_rationale.py``
    shows a simulated 0.3880 against a prevalence of 0.3873 at N ~ 98k). Treat it
    as a reference line, not as a significance test.

    Returns ``nan`` (with a ``RuntimeWarning``) if ``R == 0``, matching
    ``average_precision``, so the lift never becomes a division by zero.
    """
    _, rel = _check(np.zeros(np.shape(relevant)), relevant)
    R = int(rel.sum())
    if R == 0:
        warnings.warn(
            "random_baseline_ap is undefined for a query with 0 relevant frames "
            "(the AP it baselines is undefined too); returning nan.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(R / rel.size)


def mean_ap(aps, *, skip_nan: bool = True) -> float:
    """Macro-average of per-query APs (mAP): every query weighted equally.

    Macro, not micro, so a long phase like CalotTriangleDissection does not dominate.

    ``skip_nan=True`` (default) drops undefined queries (``R == 0``) with a
    ``RuntimeWarning`` naming how many were dropped -- they are never counted as
    0.0. ``skip_nan=False`` propagates ``nan`` so a caller can assert that every
    query was valid. Returns ``nan`` for an empty input.
    """
    aps = np.asarray(aps, dtype=np.float64)
    if aps.ndim != 1:
        raise ValueError(f"aps must be 1-D, got shape {aps.shape}")
    if aps.size == 0:
        warnings.warn("mean_ap got an empty list of APs; returning nan.", RuntimeWarning,
                      stacklevel=2)
        return float("nan")

    bad = np.isnan(aps)
    if bad.any():
        if not skip_nan:
            return float("nan")
        warnings.warn(
            f"mean_ap: dropping {int(bad.sum())} of {aps.size} queries with an "
            "undefined AP (0 relevant frames). They are excluded, not scored 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        aps = aps[~bad]
        if aps.size == 0:
            return float("nan")
    return float(aps.mean())
