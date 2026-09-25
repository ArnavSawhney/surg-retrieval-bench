"""Retrieval metrics for text-to-frame search on surgical video.

Pure NumPy. No torch, no model code, no I/O.

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
Ranking / ties
    Frames are ranked by ``np.argsort(-scores, kind="stable")``: descending score,
    and **ties broken by ascending frame index**. This is deterministic and
    reproducible, but it is not tie-neutral -- see the warning under `Ties` in
    ``docs/metrics.md``. It is also why ``average_precision`` only agrees with
    ``sklearn.metrics.average_precision_score`` when the scores contain no ties
    (sklearn groups tied scores at a single threshold, which averages over them).

Undefined values
    A query with **zero** relevant frames has no defined AP or R-Precision. These
    functions return ``nan`` and emit a ``RuntimeWarning``. They never silently
    return 0.0, which would look like a real (bad) score.

Negative / control queries
    Queries such as "a car on a road" have no relevant frames by construction. They
    are **excluded** from mAP and evaluated separately as a score-calibration check
    (a later week). Do not feed them to ``mean_ap``.

Inputs
    ``scores``: 1-D, finite, float-castable, one score per frame (higher = more
    similar). ``relevant``: 1-D, same length, boolean or 0/1 ground truth.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "rank_order",
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
def _check(scores, relevant) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalise (scores, relevant) -> (float64 scores, bool relevant)."""
    scores = np.asarray(scores, dtype=np.float64)
    relevant = np.asarray(relevant)

    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if relevant.ndim != 1:
        raise ValueError(f"relevant must be 1-D, got shape {relevant.shape}")
    if scores.shape != relevant.shape:
        raise ValueError(
            f"scores and relevant must have the same length, "
            f"got {scores.shape[0]} and {relevant.shape[0]}"
        )
    if scores.size == 0:
        raise ValueError("scores is empty: nothing to rank")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contains nan or inf; fix the encoder, do not rank nans")

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


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def rank_order(scores) -> np.ndarray:
    """Indices of ``scores`` from most to least similar.

    Descending score; ties broken by **ascending frame index** via a stable sort.
    Deterministic: the same scores always give the same ranking.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contains nan or inf; fix the encoder, do not rank nans")
    return np.argsort(-scores, kind="stable")


def _ranked_relevance(scores, relevant) -> np.ndarray:
    """Boolean relevance of each frame, in ranked order."""
    scores, rel = _check(scores, relevant)
    return rel[np.argsort(-scores, kind="stable")]


# --------------------------------------------------------------------------- #
# primary metric
# --------------------------------------------------------------------------- #
def average_precision(scores, relevant) -> float:
    """Full-ranking average precision for one query.

    ``AP = (1 / R) * sum over relevant ranks i of P@i``, where ``R`` is the total
    number of relevant frames and ``P@i`` is the precision in the top ``i``. Every
    relevant frame contributes, so AP is not capped by a cutoff and cannot be
    saturated by getting one clip right.

    Returns ``nan`` (with a ``RuntimeWarning``) if ``R == 0``.
    """
    r = _ranked_relevance(scores, relevant)
    R = int(r.sum())
    if R == 0:
        warnings.warn(
            "average_precision is undefined for a query with 0 relevant frames; "
            "returning nan. Negative/control queries belong in the calibration "
            "check, not in mAP.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    hit_positions = np.flatnonzero(r)                  # 0-based ranks of relevant frames
    hits_so_far = np.arange(1, R + 1)                  # 1st, 2nd, ... relevant frame
    precision_at_hits = hits_so_far / (hit_positions + 1)
    return float(precision_at_hits.mean())


# --------------------------------------------------------------------------- #
# secondary metric
# --------------------------------------------------------------------------- #
def r_precision(scores, relevant) -> float:
    """Precision in the top ``R``, where ``R`` is the number of relevant frames.

    Self-normalising: a perfect ranker scores 1.0 whatever R is, and a random
    ranker scores about the prevalence. Returns ``nan`` (with a
    ``RuntimeWarning``) if ``R == 0``.
    """
    r = _ranked_relevance(scores, relevant)
    R = int(r.sum())
    if R == 0:
        warnings.warn(
            "r_precision is undefined for a query with 0 relevant frames; returning nan.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(r[:R].mean())


# --------------------------------------------------------------------------- #
# descriptive / debugging
# --------------------------------------------------------------------------- #
def precision_at_k(scores, relevant, k: int) -> float:
    """Fraction of the top ``k`` frames that are relevant ("what a user sees first").

    **Descriptive only.** With thousands of relevant frames per query this
    saturates at 1.0 for weak and strong models alike, so never rank models by it.
    ``k`` is clamped to ``N`` with a warning if ``k > N``.
    """
    r = _ranked_relevance(scores, relevant)
    k = _check_k(k, r.size)
    return float(r[:k].mean())


def hit_at_k(scores, relevant, k: int) -> float:
    """1.0 if at least one relevant frame is in the top ``k``, else 0.0.

    **Sanity check only.** A random ranker gets ~0.99 on the long Cholec80 phases,
    so it cannot discriminate between models. Kept for debugging a ranking that
    looks broken.
    """
    r = _ranked_relevance(scores, relevant)
    k = _check_k(k, r.size)
    return float(r[:k].any())


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
