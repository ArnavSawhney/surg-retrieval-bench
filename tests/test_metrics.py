"""Tests for srb.metrics.

The hand-computed toy cases below are the ground truth. The sklearn cross-check is
a secondary confirmation, not the definition (and it only applies to tie-free
scores -- see ``test_ties_diverge_from_sklearn_and_that_is_expected``).

Note: ``pyproject.toml`` sets ``filterwarnings = ["error::RuntimeWarning"]``, so any
undefined-value warning that a test does not explicitly expect fails the suite.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from srb.metrics import (
    average_precision,
    count_ties,
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


def test_ap_ties_are_tie_neutral_49_over_72():
    """All four scores equal; relevant = [1, 1, 0, 0] and, reversed, [0, 0, 1, 1].

    One tied group: g = 4, r = 2, n_b = 0, r_b = 0, so (r-1)/(g-1) = 1/3 and each
    slot j is relevant with probability r/g = 1/2. Slot j contributes
    (1/2) * (0 + 1 + (j-1)/3) / j:
      j=1: (1/2) * 1       / 1 = 1/2   = 18/36
      j=2: (1/2) * (4/3)   / 2 = 1/3   = 12/36
      j=3: (1/2) * (5/3)   / 3 = 5/18  = 10/36
      j=4: (1/2) * 2       / 4 = 1/4   =  9/36
      sum = 49/36; divide by R = 2 -> AP = 49/72 = 0.68055...
    Cross-check by enumeration: of the C(4,2) = 6 placements of the two relevant
    frames, the APs are 1, 5/6, 3/4, 7/12, 1/2, 5/12 (positions {1,2}, {1,3}, {1,4},
    {2,3}, {2,4}, {3,4}); their mean is (12+10+9+7+6+5)/12/6 = 49/72.
    The same value for both labelings: the frame index no longer matters.
    """
    equal = [0.5, 0.5, 0.5, 0.5]
    assert average_precision(equal, [1, 1, 0, 0], exact=True) == Fraction(49, 72)
    assert average_precision(equal, [0, 0, 1, 1], exact=True) == Fraction(49, 72)
    assert average_precision(equal, [1, 1, 0, 0]) == pytest.approx(49 / 72, abs=1e-15)
    assert average_precision(equal, [0, 0, 1, 1]) == pytest.approx(49 / 72, abs=1e-15)


def test_ap_partial_tie_hand_computed():
    """scores = [0.9, 0.5, 0.5, 0.1], relevant = [0, 1, 0, 1]. R = 2.

    Group 1 (0.9): g=1, r=0 -> contributes nothing.
    Group 2 (0.5): g=2, r=1, n_b=1, r_b=0, (r-1)/(g-1) = 0.
      j=1: (1/2) * 1 / 2 = 1/4;  j=2: (1/2) * 1 / 3 = 1/6   -> 5/12
    Group 3 (0.1): g=1, r=1, n_b=3, r_b=1: 1 * 2 / 4 = 1/2  -> 6/12
    AP = (5/12 + 6/12) / 2 = 11/24.
    Check: the two orderings give AP (1/2 + 2/4)/2 = 1/2 and (1/3 + 2/4)/2 = 5/12;
    their mean is 11/24.
    """
    assert average_precision([0.9, 0.5, 0.5, 0.1], [0, 1, 0, 1], exact=True) == Fraction(
        11, 24
    )


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
    """Documented divergence: with ties, sklearn does not compute the expectation.

    All scores equal, relevant = [1, 1, 0, 0].
      ours    -> expected AP over the 6 equally likely placements of the 2 relevant frames = 49/72
      sklearn -> one threshold for the whole tied group: recall jumps 0 -> 1 with
                 precision 2/4 at the end of the group -> AP = 1 * 1/2 = 0.5
    sklearn interpolates at thresholds, so every relevant frame in a tied group gets
    the precision at the *end* of the group. That is a pessimistic step, not the
    mean over orderings. Both are order-invariant; only ours is the expectation.
    """
    equal, rel = [0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]
    assert average_precision(equal, rel) == pytest.approx(49 / 72)
    assert average_precision_score(rel, equal) == pytest.approx(0.5)
    assert average_precision(equal, rel) != pytest.approx(average_precision_score(rel, equal))


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


# --------------------------------------------------------------------------- #
# tie neutrality: brute force, invariance, tie-free equivalence
# --------------------------------------------------------------------------- #


def _old_ranked(scores, rel):
    """The Session 1 convention: stable sort, ties broken by ascending index."""
    return np.asarray(rel, dtype=bool)[rank_order(scores)]


def _old_ap(scores, rel):
    r = _old_ranked(scores, rel)
    pos = np.flatnonzero(r)
    return float((np.arange(1, r.sum() + 1) / (pos + 1)).mean())


def _metrics_of_ranking(ranked, k):
    """Plain (untied) metrics of one fully ordered 0/1 list, in exact arithmetic."""
    R = sum(ranked)
    hits, ap = 0, Fraction(0)
    for i, x in enumerate(ranked, start=1):
        if x:
            hits += 1
            ap += Fraction(hits, i)
    return {
        "ap": ap / R if R else None,
        "rprec": Fraction(sum(ranked[:R]), R) if R else None,
        "p@k": Fraction(sum(ranked[:k]), k),
        "hit@k": Fraction(int(any(ranked[:k]))),
    }


def _brute_force(scores, rel, k):
    """Exact expectation by enumerating every within-group permutation.

    Groups (equal scores) are laid out in descending score order; the product of all
    within-group permutations is enumerated and each full ordering weighted equally.
    """
    rel = [int(x) for x in rel]
    groups = [
        [rel[i] for i in range(len(rel)) if scores[i] == v]
        for v in sorted(set(scores), reverse=True)
    ]
    per_group = [list(itertools.permutations(grp)) for grp in groups]
    total = None
    count = 0
    for combo in itertools.product(*per_group):
        m = _metrics_of_ranking([x for grp in combo for x in grp], k)
        total = m if total is None else {
            key: (None if v is None else v + m[key]) for key, v in total.items()
        }
        count += 1
    return {key: (None if v is None else v / count) for key, v in total.items()}


def _random_tied_case(rng):
    n = int(rng.integers(1, 8))  # n <= 7
    n_levels = int(rng.integers(1, n + 1))  # few levels -> many ties
    scores = rng.integers(0, n_levels, n).astype(float) / 4
    rel = rng.integers(0, 2, n)
    k = int(rng.integers(1, n + 1))
    return scores, rel, k


@pytest.mark.parametrize("seed", range(300))
def test_exact_mode_equals_brute_force_enumeration(seed):
    """Every metric equals the brute-force expectation, with exact Fraction equality.

    Random small cases (n <= 7) drawn with few distinct score levels so that ties are
    the norm. The float path is checked against the exact value to 1e-12.
    """
    rng = np.random.default_rng(seed)
    scores, rel, k = _random_tied_case(rng)
    bf = _brute_force(list(scores), list(rel), k)

    assert precision_at_k(scores, rel, k, exact=True) == bf["p@k"]
    assert hit_at_k(scores, rel, k, exact=True) == bf["hit@k"]
    assert precision_at_k(scores, rel, k) == pytest.approx(float(bf["p@k"]), abs=1e-12)
    assert hit_at_k(scores, rel, k) == pytest.approx(float(bf["hit@k"]), abs=1e-12)
    if rel.sum() == 0:
        return
    assert average_precision(scores, rel, exact=True) == bf["ap"]
    assert r_precision(scores, rel, exact=True) == bf["rprec"]
    assert average_precision(scores, rel) == pytest.approx(float(bf["ap"]), abs=1e-12)
    assert r_precision(scores, rel) == pytest.approx(float(bf["rprec"]), abs=1e-12)


def test_brute_force_helper_reproduces_the_hand_computed_tie():
    """Guard the guard: the enumeration helper itself gives 49/72 on the 4-way tie."""
    assert _brute_force([0.5] * 4, [1, 1, 0, 0], 2)["ap"] == Fraction(49, 72)


def test_hit_at_k_straddling_group_hand_computed():
    """scores = [0.9, 0.5, 0.5, 0.5, 0.5], relevant = [0, 1, 0, 0, 0], k = 3.

    The top frame is irrelevant. The tied group has g = 4, r = 1 and t = 2 of its
    slots inside the top 3: P(hit) = 1 - C(3, 2) / C(4, 2) = 1 - 3/6 = 1/2.
    P@3 = (0 + t*r/g) / 3 = (2/4) / 3 = 1/6.
    """
    s, rel = [0.9, 0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0, 0]
    assert hit_at_k(s, rel, 3, exact=True) == Fraction(1, 2)
    assert hit_at_k(s, rel, 3) == pytest.approx(0.5)
    assert precision_at_k(s, rel, 3, exact=True) == Fraction(1, 6)


def test_hit_at_k_large_straddling_group_float_path():
    """The log-space float path matches exact comb() on a big group.

    g = 1000 tied, r = 3, t = 10: 1 - C(997, 10) / C(1000, 10).
    """
    s = np.full(1000, 0.3)
    rel = np.zeros(1000, dtype=bool)
    rel[[5, 500, 999]] = True
    expected = 1 - math.comb(997, 10) / math.comb(1000, 10)
    assert hit_at_k(s, rel, 10) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_metrics_are_invariant_to_frame_order(seed):
    """Shuffling frame indices (scores and labels together) changes no metric.

    Tied data on purpose (4 score levels over 200 frames): this would fail under the
    old index tie-break.
    """
    rng = np.random.default_rng(seed)
    n = 200
    scores = rng.integers(0, 4, n) / 4
    rel = rng.integers(0, 2, n)
    rel[0] = 1
    perm = rng.permutation(n)
    for fn in (average_precision, r_precision):
        assert fn(scores, rel) == fn(scores[perm], rel[perm])
    for k in (1, 10, 57):
        assert precision_at_k(scores, rel, k) == precision_at_k(scores[perm], rel[perm], k)
        assert hit_at_k(scores, rel, k) == hit_at_k(scores[perm], rel[perm], k)


@pytest.mark.parametrize("seed", range(10))
def test_tie_free_scores_match_the_old_sorted_definition(seed):
    """Without ties the expectation is the plain metric: equal to the Session 1 code.

    AP additionally matches sklearn to 1e-12 (see test_ap_matches_sklearn_...).
    """
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(20, 500))
    rel = rng.integers(0, 2, n)
    rel[0] = 1
    scores = rng.normal(size=n) + 0.7 * rel
    assert count_ties(scores) == (0, 0)

    r = _old_ranked(scores, rel)
    R = int(r.sum())
    assert average_precision(scores, rel) == pytest.approx(_old_ap(scores, rel), abs=1e-12)
    assert average_precision(scores, rel) == pytest.approx(
        average_precision_score(rel, scores), abs=1e-12
    )
    assert r_precision(scores, rel) == r[:R].mean()
    for k in (1, 5, 10, n):
        assert precision_at_k(scores, rel, k) == r[:k].mean()
        assert hit_at_k(scores, rel, k) == float(r[:k].any())


# --------------------------------------------------------------------------- #
# score precision and tie counting
# --------------------------------------------------------------------------- #


def test_float16_scores_raise():
    """fp16 similarities tie massively; refusing them is the guard for open decision #5."""
    s16 = np.array([0.9, 0.5, 0.1], dtype=np.float16)
    for call in (
        lambda: average_precision(s16, [1, 0, 0]),
        lambda: r_precision(s16, [1, 0, 0]),
        lambda: precision_at_k(s16, [1, 0, 0], 1),
        lambda: hit_at_k(s16, [1, 0, 0], 1),
        lambda: count_ties(s16),
        lambda: rank_order(s16),
    ):
        with pytest.raises(TypeError, match="float16"):
            call()


def test_float32_scores_are_accepted_and_upcast():
    s32 = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    assert average_precision(s32, [1, 0, 1, 0]) == pytest.approx(5 / 6)


def test_count_ties_hand_computed():
    """scores = [0.3, 0.1, 0.3, 0.2, 0.1, 0.3, 0.5].

    Groups: 0.3 x3, 0.1 x2, 0.2 x1, 0.5 x1 -> tied groups {0.3, 0.1}: 2 groups,
    3 + 2 = 5 tied frames.
    """
    assert count_ties([0.3, 0.1, 0.3, 0.2, 0.1, 0.3, 0.5]) == (5, 2)
    assert count_ties([0.4, 0.3, 0.2]) == (0, 0)
    assert count_ties([0.7] * 6) == (6, 1)
