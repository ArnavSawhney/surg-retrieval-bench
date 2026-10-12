"""The fast bootstrap metrics equal the metrics on the materialised resample, exactly."""

from __future__ import annotations

import numpy as np
import pytest

from srb.metrics import average_precision, precision_at_k, r_precision
from srb.resample import (B_CONFIRMATORY, B_DESCRIPTIVE, SortedQuery, ap_from_groups,
                          bootstrap_p, hits_at_from_groups, holm, r_precision_from_groups,
                          resample_counts)


def _case(rng, n_videos=6, max_frames=12, n_levels=None):
    """Random frames over videos; coarse score levels force ties across videos."""
    sizes = rng.integers(1, max_frames + 1, n_videos)
    video = np.repeat(np.arange(n_videos), sizes)
    n = len(video)
    levels = n_levels or rng.integers(2, n + 2)
    scores = rng.integers(0, levels, n).astype(np.float32) / levels
    rel = rng.random(n) < rng.uniform(0.1, 0.9)
    return scores, rel, video


def _materialise(scores, rel, video, m):
    """The resample as a frame list: each video's frames repeated m_v times."""
    keep = np.concatenate([np.tile(np.flatnonzero(video == v), m[v]) for v in range(len(m))])
    return scores[keep].astype(np.float64), rel[keep]


@pytest.mark.parametrize("seed", range(200))
def test_weighted_groups_equal_materialised_resample(seed):
    rng = np.random.default_rng(seed)
    scores, rel, video = _case(rng)
    q = SortedQuery(scores, rel, video)
    m = rng.integers(0, 3, video.max() + 1)
    if m.sum() == 0:
        m[0] = 1
    s2, r2 = _materialise(scores, rel, video, m)
    g, r = q.group_counts(m)
    if r2.sum() == 0:
        assert np.isnan(ap_from_groups(g, r))
        return
    assert ap_from_groups(g, r) == pytest.approx(average_precision(s2, r2), abs=1e-12)
    assert r_precision_from_groups(g, r) == pytest.approx(r_precision(s2, r2), abs=1e-12)
    for k in (1, 3, len(s2)):
        assert hits_at_from_groups(g, r, k) / k == pytest.approx(
            precision_at_k(s2, r2, k), abs=1e-12)


def test_identity_resample_equals_full_metric_on_large_tie_free_case():
    rng = np.random.default_rng(1)
    video = np.repeat(np.arange(40), 2000)
    scores = rng.standard_normal(len(video)).astype(np.float32)
    rel = rng.random(len(video)) < 0.3
    g, r = SortedQuery(scores, rel, video).group_counts(np.ones(40, int))
    assert ap_from_groups(g, r) == pytest.approx(average_precision(scores, rel), abs=1e-10)


def test_all_tied_hand_case():
    """4 tied frames, relevance [1,1,0,0]: AP = 49/72 (docs/metrics.md)."""
    g, r = SortedQuery(np.zeros(4, np.float32), np.array([1, 1, 0, 0], bool),
                       np.zeros(4, int)).group_counts(np.array([1]))
    assert ap_from_groups(g, r) == pytest.approx(49 / 72, abs=1e-15)


def test_resample_stream_prefix_is_stable():
    """The first 2,000 resamples of B = 10,000 equal a B = 2,000 draw."""
    a = resample_counts(40, 50)
    b = resample_counts(40, 20)
    assert np.array_equal(a[:20], b)
    assert (a.sum(1) == 40).all()
    assert B_DESCRIPTIVE < B_CONFIRMATORY


def test_bootstrap_p_and_holm():
    assert bootstrap_p(np.array([1.0, 2.0, 3.0, -1.0])) == pytest.approx(0.5)
    assert bootstrap_p(np.ones(10)) == 0.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # sorted a .01, c .03, b .04 -> 3*.01=.03, max(.03, 2*.03)=.06, max(.06, .04)=.06
    assert adj["a"] == (pytest.approx(0.03), True)
    assert adj["c"] == (pytest.approx(0.06), False)
    assert adj["b"] == (pytest.approx(0.06), False)
