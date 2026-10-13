"""Temporal models and metrics (pre-registration 4.4 and its 2026-10-13 clarification).
Hand-computed cases carry their arithmetic in the docstrings."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

import srb.temporal as tm

K = tm.K


def _onehot_post(y, eps=0.05):
    p = np.full((len(y), K), eps / (K - 1))
    p[np.arange(len(y)), y] = 1 - eps
    return p


# ---- moving average -----------------------------------------------------------------

def test_moving_average_hand_case():
    """Posteriors of class 0 over 4 frames: [1, 0, 1, 1] (others fill the rest).
    Causal w=2: t0 = 1 (window shortened to 1 frame), t1 = (1+0)/2, t2 = (0+1)/2,
    t3 = (1+1)/2. Centred w=3 (t-1..t+1): t0 = (1+0)/2 (shortened), t1 = 2/3, t2 = 2/3,
    t3 = (1+1)/2 (shortened). Centred w=2 covers t-1..t, i.e. exactly 2 frames."""
    p = np.zeros((4, K))
    p[:, 0] = [1, 0, 1, 1]
    p[:, 1] = 1 - p[:, 0]
    np.testing.assert_allclose(tm.moving_average(p, 2)[:, 0], [1, .5, .5, 1])
    np.testing.assert_allclose(tm.moving_average(p, 3, causal=False)[:, 0],
                               [.5, 2 / 3, 2 / 3, 1])
    np.testing.assert_allclose(tm.moving_average(p, 2, causal=False),
                               tm.moving_average(p, 2))
    np.testing.assert_allclose(tm.moving_average(p, 1), p)
    # a window longer than the video averages everything seen so far
    np.testing.assert_allclose(tm.moving_average(p, 120)[:, 0], [1, .5, 2 / 3, .75])


def test_select_window_uses_only_validation_videos():
    rng = np.random.default_rng(0)
    y = {v: np.repeat(np.arange(K), 30) for v in range(33, 41)}
    post = {v: _onehot_post(yy, eps=0.9) + rng.random((len(yy), K)) for v, yy in y.items()}
    w, scores = tm.select_window(post, y)
    assert w in tm.WINDOWS and len(scores) == len(tm.WINDOWS)
    for bad in (5, 41):
        p2 = dict(post)
        p2[bad] = post[33]
        with pytest.raises(ValueError, match="only accepts videos 33-40"):
            tm.select_window(p2, y | {bad: y[33]})


def test_select_window_ties_go_to_smaller_w():
    """Each video holds one phase throughout, so every w gives the same predictions and
    the same F1: the smallest w (5) is chosen."""
    rng = np.random.default_rng(0)
    y = {v: np.full(50, v % K) for v in range(33, 41)}
    post = {v: _onehot_post(yy, eps=0.3) + 0.01 * rng.random((50, K)) for v, yy in y.items()}
    w, scores = tm.select_window(post, y)
    assert len(set(scores)) == 1 and w == 5


# ---- HMM ----------------------------------------------------------------------------

def _toy_hmm():
    return tm.HMM(pi=np.array([.5, .3, .2]),
                  A=np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7]]),
                  prior=np.ones(3) / 3)


def test_forward_filter_hand_checked_three_states():
    """e0 = [.6, .3, .1], e1 = [.1, .6, .3].
    alpha0 ∝ pi*e0 = [.30, .09, .02] (sum .41) -> [30, 9, 2]/41.
    alpha0 @ A = [30*.8 + 9*.2 + 2*.1, 30*.1 + 9*.7 + 2*.2, 30*.1 + 9*.1 + 2*.7]/41
               = [26, 9.7, 5.3]/41;  * e1 = [2.6, 5.82, 1.59] (sum 10.01)
    alpha1 = [2.6, 5.82, 1.59]/10.01."""
    e = np.array([[.6, .3, .1], [.1, .6, .3]])
    a = tm.forward_filter(e, _toy_hmm())
    np.testing.assert_allclose(a[0], np.array([30, 9, 2]) / 41, rtol=1e-12)
    np.testing.assert_allclose(a[1], np.array([2.6, 5.82, 1.59]) / 10.01, rtol=1e-12)
    assert a.argmax(1).tolist() == [0, 1]


def test_viterbi_hand_checked_three_states():
    """delta0 = [.30, .09, .02]. delta1[j] = max_i delta0[i] A[i,j] * e1[j]:
    j=0: max(.24, .018, .002) * .1 = .024; j=1: max(.03, .063, .004) * .6 = .0378 (from 1);
    j=2: max(.03, .009, .014) * .3 = .009. Best end state 1, from 1: path [1, 1] - unlike
    filtering, which says 0 at t=0."""
    e = np.array([[.6, .3, .1], [.1, .6, .3]])
    assert tm.viterbi(e, _toy_hmm()).tolist() == [1, 1]


def test_fit_hmm_counts_with_add_one():
    """Video 1: [0,0,1,2,3,4,5,6]; video 2: [1,1].
    Starts: one 0, one 1, plus one each -> pi = (2,2,1,1,1,1,1)/9.
    Transitions: 0->0, 0->1, 1->2, ..., 5->6 (video 1) and 1->1 (video 2), plus one per
    cell -> row 0 = (2,2,1,1,1,1,1)/9, row 1 = (1,2,2,1,1,1,1)/9, row 6 = all 1/7.
    Frames: 0 twice, 1 three times, 2..6 once -> prior = (2,3,1,1,1,1,1)/10."""
    y = {1: np.array([0, 0, 1] + list(range(2, K))), 2: np.array([1, 1])}
    h = tm.fit_hmm(y)
    assert h.pi[0] == 2 / 9 and h.pi[1] == 2 / 9 and h.pi[2] == 1 / 9
    assert h.A[0, 0] == 2 / 9 and h.A[0, 1] == 2 / 9
    assert h.A[1, 1] == 2 / 9 and h.A[1, 2] == 2 / 9 and h.A[1, 0] == 1 / 9
    assert h.A[K - 1].tolist() == [1 / K] * K  # no transition out of the last phase
    np.testing.assert_allclose(h.A.sum(1), 1)
    assert h.prior[1] == 3 / 10 and h.prior[0] == 2 / 10


@pytest.mark.parametrize("bad", [33, 41, 80])
def test_fit_hmm_refuses_non_training_videos(bad):
    y = {1: np.arange(K), bad: np.arange(K)}
    with pytest.raises(ValueError, match="only accepts videos 1-32"):
        tm.fit_hmm(y)


def test_emissions_floor_and_count():
    p = np.array([[0.0, 1e-13, 1.0]])
    e, n = tm.emissions(p, np.array([.5, .25, .25]))
    assert n == 2
    np.testing.assert_allclose(e, [[2e-12, 4e-12, 4.0]])


# ---- causality ----------------------------------------------------------------------

def _changed_after(x, t, rng):
    y = x.copy()
    y[t + 1:] = rng.random(y[t + 1:].shape)
    return y


def test_causal_methods_ignore_the_future():
    rng = np.random.default_rng(1)
    post = rng.dirichlet(np.ones(K), 300)
    hmm = tm.HMM(rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K), K),
                 rng.dirichlet(np.ones(K)))
    for t in (0, 10, 150, 298):
        post2 = _changed_after(post, t, rng)
        for w in tm.WINDOWS:
            a = tm.moving_average(post, w)[:t + 1]
            assert np.array_equal(a, tm.moving_average(post2, w)[:t + 1])
        f1 = tm.forward_filter(tm.emissions(post, hmm.prior)[0], hmm)[:t + 1]
        f2 = tm.forward_filter(tm.emissions(post2, hmm.prior)[0], hmm)[:t + 1]
        assert np.array_equal(f1, f2)
    # the offline methods do look ahead (sanity check of the test itself)
    post2 = _changed_after(post, 150, rng)
    assert not np.array_equal(tm.moving_average(post, 60, causal=False)[:151],
                              tm.moving_average(post2, 60, causal=False)[:151])


@pytest.mark.parametrize("depth", tm.DEPTHS)
def test_tcn_is_causal_with_the_stated_receptive_field(depth):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = tm.make_tcn(12, depth)
    rf = tm.receptive_field(depth)
    assert rf == {2: 7, 3: 15, 4: 31}[depth]
    rng = np.random.default_rng(depth)
    X = rng.normal(size=(80, 12)).astype(np.float32)
    base = tm.tcn_logits(net, X)
    for t in (0, 20, 78):
        assert np.array_equal(base[:t + 1], tm.tcn_logits(net, _changed_after(X, t, rng))[:t + 1])
    t = 60  # frame t - rf + 1 is the oldest frame the output at t sees
    X2 = X.copy(); X2[t - rf + 1] += 5
    assert not np.allclose(base[t], tm.tcn_logits(net, X2)[t])
    X3 = X.copy(); X3[t - rf] += 5
    assert np.array_equal(base[t], tm.tcn_logits(net, X3)[t])


# ---- TCN training -------------------------------------------------------------------

def _tcn_data(videos, rng, T=40, dim=8):
    X, y = {}, {}
    for v in videos:
        yy = np.repeat(np.arange(K), T // K + 1)[:T]
        x = rng.normal(0, 1, (T, dim)).astype(np.float32)
        x[np.arange(T), yy % dim] += 2
        X[v], y[v] = x, yy
    return X, y


def test_train_tcn_is_deterministic_and_keeps_the_best_epoch():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    Xt, yt = _tcn_data(range(1, 5), rng)
    Xv, yv = _tcn_data(range(33, 35), rng)
    a = tm.train_tcn(Xt, yt, Xv, yv, depth=2, seed=0, epochs=4)
    b = tm.train_tcn(Xt, yt, Xv, yv, depth=2, seed=0, epochs=4)
    assert a.val_f1 == b.val_f1 and a.best_epoch == b.best_epoch
    assert all(np.array_equal(a.state[k].numpy(), b.state[k].numpy()) for k in a.state)
    assert a.best_epoch == int(np.argmax(a.val_f1)) + 1 and len(a.val_f1) == 4
    # the kept weights reproduce the best epoch's validation score
    net = tm.load_tcn(a.state, 8, 2)
    assert tm.macro_f1(yv, tm.tcn_predict(net, Xv)) == a.best_f1


@pytest.mark.parametrize("bad", [33, 41])
def test_train_tcn_refuses_non_training_videos(bad):
    rng = np.random.default_rng(0)
    Xt, yt = _tcn_data([1, bad], rng)
    Xv, yv = _tcn_data([33], rng)
    with pytest.raises(ValueError, match="only accepts videos 1-32"):
        tm.train_tcn(Xt, yt, Xv, yv, depth=2, seed=0, epochs=1)


@pytest.mark.parametrize("bad", [1, 41])
def test_tcn_selection_refuses_non_validation_videos(bad):
    rng = np.random.default_rng(0)
    Xt, yt = _tcn_data([1], rng)
    Xv, yv = _tcn_data([33, bad], rng)
    with pytest.raises(ValueError, match="only accepts videos 33-40"):
        tm.train_tcn(Xt, yt, Xv, yv, depth=2, seed=0, epochs=1)


def test_select_tcn_ties_go_to_smaller_depth_then_earlier_epoch():
    runs = [tm.TCNRun(4, 0, [.5, .7], 2, {}), tm.TCNRun(2, 0, [.7, .7], 1, {}),
            tm.TCNRun(3, 0, [.6, .6], 1, {})]
    best = tm.select_tcn(runs)
    assert best.depth == 2 and best.best_epoch == 1
    with pytest.raises(ValueError):
        tm.select_tcn(runs + [tm.TCNRun(2, 1, [.9], 1, {})])


# ---- metrics ------------------------------------------------------------------------

def test_edit_score_hand_cases():
    """truth segments [0,1,2]; pred [0,1,0,1,2] has 5 segments; Levenshtein = 2 (delete
    two) -> 100 (1 - 2/5) = 60. Boundary shifts do not matter: [0,0,0,1,1] vs
    [0,0,1,1,1] -> 100. A short spurious segment counts in full: [0,0,3,0,0] vs all 0
    -> segments [0,3,0] vs [0], distance 2 -> 100 (1 - 2/3)."""
    assert tm.edit_score(np.array([0, 1, 0, 1, 2]), np.array([0, 0, 1, 1, 2])) == pytest.approx(60)
    assert tm.edit_score(np.array([0, 0, 0, 1, 1]), np.array([0, 0, 1, 1, 1])) == 100
    assert tm.edit_score(np.array([0, 0, 3, 0, 0]), np.zeros(5, int)) == pytest.approx(100 / 3)
    assert tm.levenshtein([1, 2, 3], [2, 3, 4]) == 2
    assert tm.levenshtein([], [1, 2]) == 2


def test_jaccard_hand_cases_including_a_missing_phase():
    """truth [0,0,1,1], pred [0,1,1,1]: phase 0 = 1/2, phase 1 = 2/3, others nan.
    truth [0,0], pred [0,2]: phase 0 = 1/2; phase 2 is predicted but absent -> nan."""
    j = tm.jaccard_per_phase(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
    assert j[0] == .5 and j[1] == pytest.approx(2 / 3) and np.isnan(j[2:]).all()
    j = tm.jaccard_per_phase(np.array([0, 2]), np.array([0, 0]))
    assert j[0] == .5 and np.isnan(j[2])


def test_video_level_jaccard_averages_over_videos_where_the_phase_occurs():
    """Video 41: phase 0 J = 1/2, phase 1 J = 2/3. Video 42: only phase 0, J = 1.
    Phase 0 = (1/2 + 1)/2 = 3/4; phase 1 = 2/3 (one video); mean over phases = 17/24."""
    truth = {41: np.array([0, 0, 1, 1]), 42: np.array([0, 0])}
    pred = {41: np.array([0, 1, 1, 1]), 42: np.array([0, 0])}
    s = tm.video_stats(pred, truth)
    assert tm.point_metrics(s)["jaccard"] == pytest.approx(17 / 24)


def test_weighted_metrics_equal_the_materialised_resample():
    rng = np.random.default_rng(3)
    truth = {v: np.sort(rng.integers(0, K, rng.integers(20, 60))) for v in range(41, 47)}
    pred = {v: np.where(rng.random(len(t)) < .7, t, rng.integers(0, K, len(t)))
            for v, t in truth.items()}
    s = tm.video_stats(pred, truth)
    pt = tm.point_metrics(s)
    allt = np.concatenate([truth[v] for v in sorted(truth)])
    allp = np.concatenate([pred[v] for v in sorted(truth)])
    assert pt["accuracy"] == pytest.approx((allt == allp).mean(), abs=1e-15)
    assert pt["macro_f1"] == pytest.approx(
        f1_score(allt, allp, labels=range(K), average="macro", zero_division=0), abs=1e-12)
    counts = np.array([[2, 0, 1, 0, 3, 0], [1, 1, 1, 1, 1, 1]])
    w = tm.weighted_metrics(s, counts)
    for b, row in enumerate(counts):
        tr, pr, k = {}, {}, 0
        for v, c in zip(sorted(truth), row):
            for _ in range(c):  # a video drawn c times enters c times
                tr[1000 + k], pr[1000 + k] = truth[v], pred[v]
                k += 1
        mat = tm.point_metrics(tm.video_stats(pr, tr))
        for name in tm.METRICS:
            assert w[name][b] == pytest.approx(mat[name], abs=1e-12)
