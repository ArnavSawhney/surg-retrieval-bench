"""The registered evaluation on a synthetic pool (random embeddings, real query set)."""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from srb.datasets.cholec80 import PHASES, TOOL_COLUMNS
from srb.evaluation import (_f1_macro_from_confusion, compare_models, evaluate_model,
                            expected_distinct_videos, headline_resamples, summarise)
from srb.queries import cholec80_queries
from srb.resample import resample_counts

DIM = 16


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def pool():
    """4 validation videos, 60 frames each, every phase and tool present."""
    rng = np.random.default_rng(0)
    rows = []
    for v in (33, 34, 35, 36):
        for i in range(60):
            row = {"dataset": "cholec80", "video_id": v, "frame_idx_25fps": 25 * i,
                   "split": "val", "phase": PHASES[(i // 9) % 7]}
            row.update({c: bool(rng.random() < 0.3) for c in TOOL_COLUMNS})
            rows.append(row)
    manifest = pd.DataFrame(rows)
    emb = _unit(rng.standard_normal((len(manifest), DIM)))
    text = {q.text: _unit(rng.standard_normal(DIM)) for q in cholec80_queries()}
    return manifest, emb, text


@pytest.fixture(scope="module")
def evaluated(pool):
    manifest, emb, text = pool
    counts = np.vstack([np.ones((1, 4), int), resample_counts(4, 40)])  # row 0 = identity
    return evaluate_model("fake", emb, manifest, text, counts)


def test_shapes_and_aggregation(evaluated):
    ev = evaluated
    assert len(ev.per_query) == 77 and len(ev.per_target) == 14
    pos = ev.per_query[ev.per_query.kind != "negative"]
    assert ev.summary["overall_map"] == pytest.approx(pos.ap.mean())  # 14 x 5, equal weights
    assert ev.summary["phase_map"] == pytest.approx(pos[pos.kind == "phase"].ap.mean())
    assert ev.per_query[ev.per_query.kind == "negative"].ap.isna().all()
    assert (ev.per_target.per_video_n == 4).all()
    assert ev.summary["rq2_mean_sd"] == pytest.approx(ev.per_target.ap_sd.mean())


def test_identity_resample_reproduces_point_estimates(evaluated):
    ev = evaluated
    pos = ev.per_query[ev.per_query.kind != "negative"].set_index("qid")
    np.testing.assert_allclose(ev.boot["ap"][0], pos.loc[ev.boot["qids"]].ap, atol=1e-12)
    np.testing.assert_allclose(ev.boot["r_prec"][0], pos.loc[ev.boot["qids"]].r_prec,
                               atol=1e-12)
    res = headline_resamples(ev)
    for key in ("overall_map", "phase_ensemble_map", "zs_phase_accuracy",
                "zs_phase_macro_f1", "overall_per_video_ap", "rq2_mean_sd", "role_r3_map"):
        assert res[key][0] == pytest.approx(ev.summary[key], abs=1e-12), key


def test_classification_matches_sklearn(pool, evaluated):
    manifest, emb, text = pool
    qs = {q.qid: q for q in cholec80_queries()}
    t = np.stack([text[qs[f"phase_{p.lower()}_r2"].text] for p in PHASES])
    pred = np.argmax(t @ emb.T, 0)
    truth = pd.Categorical(manifest.phase, categories=PHASES).codes
    row = evaluated.classification.set_index("classifier").loc["r2"]
    assert row.accuracy == pytest.approx((pred == truth).mean())
    assert row.macro_f1 == pytest.approx(f1_score(truth, pred, average="macro"))


def test_f1_from_confusion_matches_sklearn_with_missing_class():
    truth = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 0])
    c = np.zeros((4, 4))
    np.add.at(c, (truth, pred), 1)  # class 3 never occurs
    assert _f1_macro_from_confusion(c) == pytest.approx(
        f1_score(truth, pred, labels=range(4), average="macro", zero_division=0))


def _brute_distinct(scores, video, k):
    """Exact expectation over all within-group orderings (small cases)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    groups = [list(g) for _, g in itertools.groupby(order, key=lambda i: scores[i])]
    total, n = Fraction(0), 0
    for perm in itertools.product(*[itertools.permutations(g) for g in groups]):
        ranking = [i for g in perm for i in g]
        total += len({video[i] for i in ranking[:k]})
        n += 1
    return total / n


@pytest.mark.parametrize("seed", range(30))
def test_expected_distinct_videos_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 7))
    scores = rng.integers(0, 3, n).astype(np.float64)
    video = rng.integers(0, 3, n)
    k = int(rng.integers(1, n + 1))
    assert expected_distinct_videos(scores, video, k) == pytest.approx(
        float(_brute_distinct(scores, video, k)), abs=1e-12)


def test_calibration_and_summary_table(evaluated):
    table = summarise(evaluated)
    row = table.set_index("metric").loc["overall_map"]
    assert row.n_boot == 41 and row.ci_low <= row.ci_high
    assert table.set_index("metric").loc["zs_phase_accuracy"].n_boot == 41
    cal = evaluated.calibration
    assert (cal.kind == "negative").sum() == 7 and (cal.kind == "positive target").sum() == 14
    assert cal[cal.kind == "negative"].below_every_positive_target_mean.notna().all()


def test_compare_models_holm_and_flips(pool):
    manifest, emb, text = pool
    rng = np.random.default_rng(5)
    counts = resample_counts(4, 30)
    evals = {}
    for name in ("a", "b", "c"):
        other = {k: _unit(v + 0.5 * rng.standard_normal(DIM)) for k, v in text.items()}
        evals[name] = evaluate_model(name, emb, manifest, other, counts)
    pairs, flips = compare_models(evals)
    assert list(pairs.pair) == ["a - b", "a - c", "b - c"]
    assert (pairs.p_holm >= pairs.p).all() and pairs.n_boot.eq(30).all()
    assert flips.order_changes_fraction.between(0, 1).all() and flips.targets.eq(14).all()


def test_refuses_test_videos(pool):
    manifest, emb, text = pool
    m = manifest.copy()
    m.loc[0, "video_id"] = 41
    with pytest.raises(PermissionError):
        evaluate_model("fake", emb, m, text, None)
