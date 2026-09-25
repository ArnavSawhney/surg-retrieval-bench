"""Tests for srb.metrics.

The hand-computed toy cases below are the ground truth. The sklearn cross-check is
a secondary confirmation, not the definition (and it only applies to tie-free
scores -- see ``test_ties_diverge_from_sklearn_and_that_is_expected``).

Note: ``pyproject.toml`` sets ``filterwarnings = ["error::RuntimeWarning"]``, so any
undefined-value warning that a test does not explicitly expect fails the suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from srb.metrics import (
    average_precision,
    hit_at_k,
    mean_ap,
    precision_at_k,
    r_precision,
    random_baseline_ap,
    rank_order,
)

# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #


def test_rank_order_descending_with_index_tiebreak():
    """scores = [0.1, 0.9, 0.5, 0.9].

    Descending by score: the two 0.9s (indices 1 and 3) come first, and the tie is
    broken by ascending index -> 1 then 3. Then 0.5 (index 2), then 0.1 (index 0).
    Expected order: [1, 3, 2, 0].
    """
    assert rank_order([0.1, 0.9, 0.5, 0.9]).tolist() == [1, 3, 2, 0]


def test_rank_order_is_deterministic_on_repeated_calls():
    rng = np.random.default_rng(0)
    scores = rng.choice([0.0, 0.5, 1.0], size=200)  # lots of ties on purpose
    first = rank_order(scores)
    for _ in range(5):
        assert np.array_equal(rank_order(scores), first)


# --------------------------------------------------------------------------- #
# average precision: hand-computed
# --------------------------------------------------------------------------- #


def test_ap_hand_computed_four_frames():
    """scores = [0.9, 0.8, 0.7, 0.6], relevant = [1, 0, 1, 0].

    Ranked relevance: [1, 0, 1, 0]. Relevant frames sit at ranks 1 and 3.
      P@1 = 1/1 = 1
      P@3 = 2/3 = 0.666...
      AP  = (1 + 2/3) / 2 = (5/3) / 2 = 5/6 = 0.8333...
    """
    assert average_precision([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx(5 / 6)


def test_ap_hand_computed_five_frames_unsorted_input():
    """scores = [0.2, 0.9, 0.4, 0.75, 0.6], relevant = [0, 1, 1, 0, 1].

    Ranked order by score: 0.9(i1), 0.75(i3), 0.6(i4), 0.4(i2), 0.2(i0).
    Ranked relevance:      1,       0,        1,       1,       0.
    Relevant at ranks 1, 3, 4:
      P@1 = 1/1 = 1
      P@3 = 2/3 = 0.666...
      P@4 = 3/4 = 0.75
      AP  = (1 + 2/3 + 3/4) / 3 = (29/12) / 3 = 29/36 = 0.80555...
    """
    ap = average_precision([0.2, 0.9, 0.4, 0.75, 0.6], [0, 1, 1, 0, 1])
    assert ap == pytest.approx(29 / 36)


def test_ap_perfect_ranking_is_one():
    """All 3 relevant frames ranked above all irrelevant ones.

    P@1 = 1/1, P@2 = 2/2, P@3 = 3/3 -> AP = (1 + 1 + 1) / 3 = 1.0.
    """
    assert average_precision([0.9, 0.8, 0.7, 0.2, 0.1], [1, 1, 1, 0, 0]) == 1.0


def test_ap_worst_ranking_hand_computed():
    """The single relevant frame is ranked last of 4.

    Ranked relevance [0, 0, 0, 1]; P@4 = 1/4 -> AP = 0.25.
    """
    assert average_precision([0.1, 0.9, 0.8, 0.7], [1, 0, 0, 0]) == pytest.approx(0.25)


def test_ap_single_frame_relevant():
    """N = 1, R = 1: the only frame is relevant at rank 1 -> P@1 = 1 -> AP = 1.0."""
    assert average_precision([0.42], [1]) == 1.0


def test_ap_all_relevant():
    """R = N = 4: every rank i has P@i = i/i = 1 -> AP = 1.0, whatever the scores."""
    assert average_precision([0.1, 0.9, 0.5, 0.3], [1, 1, 1, 1]) == 1.0


def test_ap_ties_use_ascending_index():
    """All four scores equal; relevant = [1, 1, 0, 0].

    The stable index tie-break ranks 0, 1, 2, 3, so ranked relevance is [1, 1, 0, 0]:
      P@1 = 1, P@2 = 1 -> AP = 1.0.
    Reverse the labels to [0, 0, 1, 1] and the same tie-break gives [0, 0, 1, 1]:
      P@3 = 1/3, P@4 = 2/4 -> AP = (1/3 + 1/2) / 2 = (5/6) / 2 = 5/12.
    Same scores, different AP: the tie-break is deterministic but not tie-neutral.
    """
    equal = [0.5, 0.5, 0.5, 0.5]
    assert average_precision(equal, [1, 1, 0, 0]) == 1.0
    assert average_precision(equal, [0, 0, 1, 1]) == pytest.approx(5 / 12)


# --------------------------------------------------------------------------- #
# R-Precision: hand-computed
# --------------------------------------------------------------------------- #


def test_r_precision_hand_computed():
    """scores = [0.9, 0.8, 0.7, 0.6, 0.5], relevant = [1, 0, 1, 0, 0]. R = 2.

    Top R = top 2 ranked frames have relevance [1, 0] -> R-Prec = 1/2 = 0.5.
    """
    assert r_precision([0.9, 0.8, 0.7, 0.6, 0.5], [1, 0, 1, 0, 0]) == pytest.approx(0.5)


def test_r_precision_perfect_and_worst():
    """R = 2 of N = 4.

    Perfect: top 2 both relevant -> 2/2 = 1.0.
    Worst:   top 2 both irrelevant -> 0/2 = 0.0.
    """
    assert r_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert r_precision([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_r_precision_all_relevant_is_one():
    """R = N: the top R frames are every frame, all relevant -> 1.0."""
    assert r_precision([0.3, 0.1, 0.2], [1, 1, 1]) == 1.0


# --------------------------------------------------------------------------- #
# P@k and Hit@k: hand-computed
# --------------------------------------------------------------------------- #


def test_precision_at_k_hand_computed():
    """Ranked relevance [1, 0, 1, 0, 0] (scores already descending).

    P@1 = 1/1 = 1.0
    P@2 = 1/2 = 0.5
    P@3 = 2/3 = 0.666...
    P@5 = 2/5 = 0.4
    """
    s, rel = [0.9, 0.8, 0.7, 0.6, 0.5], [1, 0, 1, 0, 0]
    assert precision_at_k(s, rel, 1) == pytest.approx(1.0)
    assert precision_at_k(s, rel, 2) == pytest.approx(0.5)
    assert precision_at_k(s, rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(s, rel, 5) == pytest.approx(0.4)


def test_hit_at_k_hand_computed():
    """Ranked relevance [0, 0, 1, 0]: no relevant frame until rank 3.

    Hit@1 = Hit@2 = 0.0, Hit@3 = Hit@4 = 1.0.
    """
    s, rel = [0.9, 0.8, 0.7, 0.6], [0, 0, 1, 0]
    assert hit_at_k(s, rel, 1) == 0.0
    assert hit_at_k(s, rel, 2) == 0.0
    assert hit_at_k(s, rel, 3) == 1.0
    assert hit_at_k(s, rel, 4) == 1.0


def test_hit_at_k_zero_relevant_is_zero_not_nan():
    """Hit@k is defined with R = 0: nothing relevant can be in the top k -> 0.0."""
    assert hit_at_k([0.9, 0.1], [0, 0], 2) == 0.0


def test_precision_at_k_zero_relevant_is_zero_not_nan():
    """P@k is defined with R = 0: 0 relevant in the top k -> 0.0."""
    assert precision_at_k([0.9, 0.1], [0, 0], 2) == 0.0


def test_k_greater_than_n_clamps_and_warns():
    """k = 10 with N = 3: clamp to k = 3. Ranked relevance [1, 0, 0] -> 1/3."""
    with pytest.warns(RuntimeWarning, match="exceeds the number of frames"):
        assert precision_at_k([0.9, 0.8, 0.7], [1, 0, 0], 10) == pytest.approx(1 / 3)
    with pytest.warns(RuntimeWarning, match="exceeds the number of frames"):
        assert hit_at_k([0.9, 0.8, 0.7], [1, 0, 0], 10) == 1.0


@pytest.mark.parametrize("bad_k", [0, -1, -10])
def test_k_below_one_raises(bad_k):
    with pytest.raises(ValueError, match="k must be >= 1"):
        precision_at_k([0.9, 0.1], [1, 0], bad_k)


@pytest.mark.parametrize("bad_k", [1.0, 2.5, "3", None, True])
def test_non_integer_k_raises(bad_k):
    with pytest.raises(TypeError, match="k must be an int"):
        precision_at_k([0.9, 0.1], [1, 0], bad_k)


# --------------------------------------------------------------------------- #
# random baseline
# --------------------------------------------------------------------------- #


def test_random_baseline_ap_is_prevalence():
    """R = 3 relevant of N = 10 -> prevalence = 3/10 = 0.3."""
    rel = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert random_baseline_ap(rel) == pytest.approx(0.3)


def test_random_baseline_ap_ignores_order():
    """Prevalence depends only on how many frames are relevant, not where they are."""
    assert random_baseline_ap([1, 0, 1, 0]) == random_baseline_ap([0, 0, 1, 1])


def test_random_ranker_ap_is_close_to_prevalence():
    """A uniformly random ranker should score AP ~ prevalence (the baseline's claim).

    N = 20000, R = 2000 -> prevalence 0.1. Averaged over 20 random rankings the AP
    should land within 0.01 of 0.1. (The exact expectation is marginally above
    prevalence, which is why this is a tolerance, not an equality.)
    """
    rng = np.random.default_rng(12345)
    n, r_count = 20_000, 2_000
    rel = np.zeros(n, dtype=bool)
    rel[:r_count] = True
    aps = [average_precision(rng.random(n), rel) for _ in range(20)]
    assert mean_ap(aps) == pytest.approx(r_count / n, abs=0.01)


# --------------------------------------------------------------------------- #
# mAP aggregation
# --------------------------------------------------------------------------- #


def test_mean_ap_is_macro_average():
    """APs [1.0, 0.5, 0.0] -> (1.0 + 0.5 + 0.0) / 3 = 0.5, each query weighted equally."""
    assert mean_ap([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_mean_ap_skips_nan_and_warns():
    """APs [1.0, nan, 0.5]: the undefined query is dropped, not scored 0.

    Dropped  -> (1.0 + 0.5) / 2 = 0.75.
    Scored 0 -> (1.0 + 0.0 + 0.5) / 3 = 0.5, which would be a fabricated penalty.
    """
    with pytest.warns(RuntimeWarning, match="dropping 1 of 3 queries"):
        assert mean_ap([1.0, float("nan"), 0.5]) == pytest.approx(0.75)


def test_mean_ap_can_propagate_nan():
    """skip_nan=False lets a caller assert that every query was valid."""
    assert np.isnan(mean_ap([1.0, float("nan")], skip_nan=False))


def test_mean_ap_all_nan_is_nan():
    with pytest.warns(RuntimeWarning):
        assert np.isnan(mean_ap([float("nan"), float("nan")]))


def test_mean_ap_empty_is_nan_and_warns():
    with pytest.warns(RuntimeWarning, match="empty list of APs"):
        assert np.isnan(mean_ap([]))


# --------------------------------------------------------------------------- #
# undefined queries (R = 0)
# --------------------------------------------------------------------------- #


def test_ap_zero_relevant_is_nan_and_warns():
    """A negative/control query ("a car on a road") has R = 0: AP is undefined.

    nan, never 0.0 -- 0.0 would look like a real (terrible) score in a results table.
    """
    with pytest.warns(RuntimeWarning, match="undefined for a query with 0 relevant"):
        assert np.isnan(average_precision([0.9, 0.5, 0.1], [0, 0, 0]))


def test_r_precision_zero_relevant_is_nan_and_warns():
    with pytest.warns(RuntimeWarning, match="undefined for a query with 0 relevant"):
        assert np.isnan(r_precision([0.9, 0.5, 0.1], [0, 0, 0]))


def test_random_baseline_zero_relevant_is_nan_and_warns():
    """The baseline is nan too, so lift = AP / baseline never divides by zero."""
    with pytest.warns(RuntimeWarning, match="undefined for a query with 0 relevant"):
        assert np.isnan(random_baseline_ap([0, 0, 0]))


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        average_precision([0.9, 0.5], [1, 0, 1])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="nothing to rank"):
        average_precision([], [])


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="must be 1-D"):
        average_precision([[0.9, 0.5]], [[1, 0]])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_raise(bad):
    """A nan similarity means the encoder is broken; ranking it would hide that."""
    with pytest.raises(ValueError, match="nan or inf"):
        average_precision([0.9, bad, 0.1], [1, 0, 0])


def test_non_binary_relevance_raises():
    """Relevance here is binary (phase/tool label match), so graded values are a bug."""
    with pytest.raises(ValueError, match="boolean or 0/1"):
        average_precision([0.9, 0.5, 0.1], [2, 0, 1])


def test_boolean_and_integer_relevance_agree():
    scores = [0.9, 0.8, 0.7, 0.6]
    assert average_precision(scores, [True, False, True, False]) == average_precision(
        scores, [1, 0, 1, 0]
    )


def test_float_relevance_of_zeros_and_ones_is_accepted():
    assert average_precision([0.9, 0.1], [1.0, 0.0]) == 1.0


# --------------------------------------------------------------------------- #
# sklearn cross-check (secondary confirmation, tie-free inputs only)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(10))
def test_ap_matches_sklearn_on_tie_free_scores(seed):
    """On continuous (tie-free) scores our AP must equal sklearn's to 1e-12."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 500))
    rel = rng.integers(0, 2, n)
    if rel.sum() == 0:  # keep the query defined
        rel[0] = 1
    scores = rng.normal(size=n) + 0.7 * rel  # continuous -> no exact ties
    assert len(np.unique(scores)) == n
    assert average_precision(scores, rel) == pytest.approx(
        average_precision_score(rel, scores), abs=1e-12
    )


def test_ap_matches_sklearn_on_the_hand_computed_cases():
    """The toy cases are the ground truth; sklearn should agree on them too."""
    for scores, rel, expected in [
        ([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0], 5 / 6),
        ([0.2, 0.9, 0.4, 0.75, 0.6], [0, 1, 1, 0, 1], 29 / 36),
        ([0.1, 0.9, 0.8, 0.7], [1, 0, 0, 0], 0.25),
    ]:
        assert average_precision(scores, rel) == pytest.approx(expected)
        assert average_precision_score(rel, scores) == pytest.approx(expected)


def test_ties_diverge_from_sklearn_and_that_is_expected():
    """Documented divergence: with ties, sklearn and this module answer different questions.

    All scores equal, relevant = [1, 1, 0, 0].
      ours    -> index tie-break gives ranked relevance [1, 1, 0, 0] -> AP = 1.0
      sklearn -> groups all four at one threshold, so precision = recall-weighted
                 prevalence -> AP = 0.5
    Neither is a bug. Ours is reproducible; sklearn's is tie-neutral. Logged as an
    open decision in LOG.md.
    """
    equal, rel = [0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]
    assert average_precision(equal, rel) == 1.0
    assert average_precision_score(rel, equal) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# the reason this metric set was chosen (regression test on the rationale)
# --------------------------------------------------------------------------- #


def test_p_at_10_saturates_where_ap_does_not():
    """The "one-clip shortcut" failure mode that motivates AP over P@10 / Hit@k.

    N = 50000 frames, R = 5000 relevant (prevalence 0.1). The shortcut ranker puts
    50 genuinely relevant frames on top and scores the other 49950 at random -- it
    has found one clip and learned nothing else.
      P@10  = 1.0     (indistinguishable from a perfect model)
      Hit@1 = 1.0     (likewise)
      AP    = 0.1144  (lift 1.14 over the 0.1 baseline: correctly judged near-worthless)

    The 50 perfect hits at the top contribute 50 near-1.0 terms to a mean over 5000
    relevant frames, which is the whole 0.0144 above prevalence. A model with real
    signal lifts AP several-fold (see docs/metric_rationale.py: gauss_2.0 reaches
    0.60 at prevalence 0.068), so AP separates them and P@10 / Hit@k do not.
    """
    rng = np.random.default_rng(0)
    n, r_count = 50_000, 5_000
    rel = np.zeros(n, dtype=bool)
    rel[:r_count] = True

    scores = rng.random(n)
    scores[rng.choice(np.flatnonzero(rel), 50, replace=False)] = 2.0

    assert precision_at_k(scores, rel, 10) == 1.0
    assert hit_at_k(scores, rel, 1) == 1.0

    prevalence = random_baseline_ap(rel)
    assert prevalence == pytest.approx(0.1)
    ap = average_precision(scores, rel)
    assert ap == pytest.approx(0.1144, abs=1e-4)
    assert ap / prevalence < 1.2  # lift barely above a query-blind ranker


def test_hit_at_10_cannot_discriminate_on_a_long_phase():
    """With prevalence 0.39 (CalotTriangleDissection), a random ranker gets Hit@10 = 1.0.

    1 - (1 - 0.39)^10 = 0.993, so Hit@10 carries essentially no information.
    """
    rng = np.random.default_rng(7)
    n = 98_520
    rel = np.zeros(n, dtype=bool)
    rel[: int(0.387 * n)] = True
    hits = [hit_at_k(rng.random(n), rel, 10) for _ in range(5)]
    assert all(h == 1.0 for h in hits)
