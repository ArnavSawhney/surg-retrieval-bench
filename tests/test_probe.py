"""Linear probe (pre-registration 4.3) on synthetic embeddings.

Pool: Cholec80-style videos 1-40 plus test videos 41-44, 70 frames each, every phase in
every video (blocks of 10 frames). Phase k adds +3 to feature k, tool t adds +3 to
feature 7 + t, so both tasks are linearly separable up to N(0, 0.5) noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import srb.probe as probe
from srb.datasets.cholec80 import PHASES, TOOL_COLUMNS
from srb.evaluation import evaluate_model
from srb.queries import cholec80_queries
from srb.resample import resample_counts

VIDEOS = list(range(1, 45))
FRAMES = 70
DIM = 16


def _pool(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for v in VIDEOS:
        for i in range(FRAMES):
            k = (i // 10) % len(PHASES)
            tools = rng.random(len(TOOL_COLUMNS)) < 0.3
            row = {"dataset": "cholec80", "video_id": v, "frame_idx_25fps": 25 * i,
                   "phase": PHASES[k]}
            row.update(dict(zip(TOOL_COLUMNS, tools.tolist())))
            rows.append(row)
            x = rng.normal(0, 0.5, DIM)
            x[k] += 3
            x[7:14] += 3 * tools
            X.append(x)
    return np.asarray(X, dtype=np.float32), pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pool():
    return _pool()


@pytest.fixture(scope="module")
def selected(pool):
    X, man = pool
    phase, pinfos = probe.select_phase_probe(X, man)
    tools, tinfos = probe.select_tool_probes(X, man)
    return phase, pinfos, tools, tinfos


def _mask(man, lo, hi):
    return man.video_id.between(lo, hi).to_numpy()


def test_separable_data_is_learned(pool, selected):
    X, man = pool
    phase, pinfos, tools, tinfos = selected
    assert len(pinfos) == len(probe.C_GRID)
    assert len(tinfos) == len(probe.C_GRID) * len(TOOL_COLUMNS)
    assert max(i.val_score for i in pinfos) > 0.99
    test = _mask(man, 41, 44)
    ev = probe.evaluate_probe("fake", phase, tools, X[test], man[test].reset_index(drop=True),
                              {"r1": np.zeros(test.sum(), int)}, resample_counts(4, 20))
    assert ev.point["phase_accuracy"] > 0.99
    assert ev.point["tool_map"] > 0.99
    assert ev.posteriors.shape == (test.sum(), len(PHASES))
    np.testing.assert_allclose(ev.posteriors.sum(1), 1.0)


def test_selection_scores_only_validation_rows(pool, monkeypatch):
    """Every selection score (macro-F1, AP) is computed on exactly the 33-40 rows."""
    X, man = pool
    n_val = int(_mask(man, 33, 40).sum())
    seen = []
    real_f1, real_ap = probe.f1_score, probe.average_precision
    monkeypatch.setattr(probe, "f1_score",
                        lambda y, p, **kw: seen.append(len(y)) or real_f1(y, p, **kw))
    monkeypatch.setattr(probe, "average_precision",
                        lambda s, r, **kw: seen.append(len(r)) or real_ap(s, r, **kw))
    probe.select_phase_probe(X, man)
    probe.select_tool_probes(X, man)
    assert len(seen) == len(probe.C_GRID) * (1 + len(TOOL_COLUMNS))
    assert set(seen) == {n_val}


def test_selection_ignores_test_rows(pool, selected):
    """Replacing test-video features and labels with noise changes nothing selected."""
    X, man = pool
    test = _mask(man, 41, 44)
    X2, man2 = X.copy(), man.copy()
    rng = np.random.default_rng(99)
    X2[test] = rng.normal(0, 5, (test.sum(), DIM))
    man2.loc[test, "phase"] = rng.choice(PHASES, test.sum())
    for t in TOOL_COLUMNS:
        man2.loc[test, t] = rng.random(test.sum()) < 0.5
    phase2, pinfos2 = probe.select_phase_probe(X2, man2)
    tools2, _ = probe.select_tool_probes(X2, man2)
    phase, pinfos, tools, _ = selected
    assert phase2.C == phase.C
    np.testing.assert_array_equal(phase2.coef, phase.coef)
    assert [i.val_score for i in pinfos2] == [i.val_score for i in pinfos]
    for t in TOOL_COLUMNS:
        assert tools2[t].C == tools[t].C
        np.testing.assert_array_equal(tools2[t].coef, tools[t].coef)


def test_selection_follows_validation_score(pool, monkeypatch):
    """The chosen C is the grid point with the best validation score (first on ties)."""
    X, man = pool
    fake = iter([0.1, 0.5, 0.9, 0.9, 0.2, 0.3])  # best at index 2 (tie with 3)
    monkeypatch.setattr(probe, "f1_score", lambda *a, **k: next(fake))
    phase, _ = probe.select_phase_probe(X, man)
    assert phase.C == probe.C_GRID[2]


def test_deterministic(pool, selected):
    X, man = pool
    phase, pinfos, tools, _ = selected
    again, pinfos2 = probe.select_phase_probe(X, man)
    np.testing.assert_array_equal(again.coef, phase.coef)
    np.testing.assert_array_equal(again.intercept, phase.intercept)
    assert [i.val_score for i in pinfos2] == [i.val_score for i in pinfos]


@pytest.mark.parametrize("video", [45, 80, 33])
def test_fit_refuses_non_training_videos(pool, video, monkeypatch):
    X, man = pool
    sub = man[_mask(man, 1, 3)].copy()
    sub.iloc[0, sub.columns.get_loc("video_id")] = video
    called = []
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="Test leakage" if video > 40 else "videos 1-32"):
        probe.fit_head(X[:len(sub)], probe.phase_labels(sub), sub, 1.0, "phase")
    assert not called


def test_leakage_guard_is_called(pool, monkeypatch):
    X, man = pool
    calls = []
    real = probe.assert_no_test_leakage
    monkeypatch.setattr(probe, "assert_no_test_leakage", lambda m: calls.append(len(m)) or real(m))
    probe.select_phase_probe(X, man)
    n_train = int(_mask(man, 1, 32).sum())
    assert calls.count(n_train) == len(probe.C_GRID)  # once per fit, on the training rows


def test_test_rows_never_reach_fit(pool, monkeypatch):
    X, man = pool
    train = _mask(man, 1, 32)
    fitted = []
    real_fit = LogisticRegression.fit

    def spy(self, Xf, yf, *a, **k):
        fitted.append(np.array(Xf))
        return real_fit(self, Xf, yf, *a, **k)

    monkeypatch.setattr(LogisticRegression, "fit", spy)
    probe.select_phase_probe(X, man)
    probe.select_tool_probes(X, man)
    assert len(fitted) == len(probe.C_GRID) * (1 + len(TOOL_COLUMNS))
    for Xf in fitted:
        np.testing.assert_array_equal(Xf, X[train])


def test_non_convergence_is_recorded_not_raised(pool, monkeypatch):
    X, man = pool
    monkeypatch.setattr(probe, "MAX_ITER", 1)
    _, infos = probe.select_phase_probe(X, man)
    assert all(not i.converged and i.n_iter == 1 for i in infos)


def test_bootstrap_identity_resample_equals_point(pool, selected):
    X, man = pool
    phase, _, tools, _ = selected
    val = _mask(man, 33, 40)
    Xv, mv = X[val], man[val].reset_index(drop=True)
    rng = np.random.default_rng(3)
    zs = {f"r{r}": rng.integers(0, 7, len(mv)) for r in range(1, 6)}
    counts = np.vstack([np.ones((1, 8), int), resample_counts(8, 10)])
    ev = probe.evaluate_probe("fake", phase, tools, Xv, mv, zs, counts)
    for k in ("phase_accuracy", "phase_macro_f1", "phase_per_video_accuracy", "tool_map",
              "zs_phase_accuracy", "gain_over_zs_accuracy"):
        assert ev.boot[k][0] == pytest.approx(ev.point[k], abs=1e-12), k
    truth = probe.phase_labels(mv)
    assert ev.point["zs_phase_accuracy"] == pytest.approx(
        np.mean([(p == truth).mean() for p in zs.values()]))


def test_summaries_and_spread(pool, selected):
    X, man = pool
    phase, _, tools, _ = selected
    val = _mask(man, 33, 40)
    Xv, mv = X[val], man[val].reset_index(drop=True)
    zs = {"r1": np.zeros(len(mv), int)}
    counts = resample_counts(8, 30)
    a = probe.evaluate_probe("a", phase, tools, Xv, mv, zs, counts)
    b = probe.evaluate_probe("b", phase, tools, Xv + 0.7, mv, zs, counts)
    summary, pairs = probe.summarise_probes({"a": a, "b": b})
    assert len(summary) == 2 * len(probe.HEADLINE)
    row = pairs[(pairs.pair == "a - b") & (pairs.metric == "phase_accuracy")].iloc[0]
    assert row["diff"] == pytest.approx(a.point["phase_accuracy"] - b.point["phase_accuracy"])
    spread = pairs[(pairs.pair == "best - worst (spread)") & (pairs.metric == "phase_accuracy")]
    assert spread["diff"].iloc[0] == pytest.approx(abs(row["diff"]))


def test_save_load_roundtrip(tmp_path, pool, selected):
    X, _ = pool
    phase, _, tools, _ = selected
    path = tmp_path / "probe.npz"
    probe.save_probes(path, phase, tools, {"model": "fake", "git_hash": "abc"})
    p2, t2, meta = probe.load_probes(path)
    assert meta["git_hash"] == "abc" and p2.C == phase.C
    np.testing.assert_array_equal(p2.decision(X), phase.decision(X))
    for t in TOOL_COLUMNS:
        np.testing.assert_array_equal(t2[t].decision(X), tools[t].decision(X))


def test_zero_shot_matches_registered_evaluation():
    """The probe's r1-r5 zero-shot classifiers equal evaluate_model's."""
    rng = np.random.default_rng(5)
    rows = [{"dataset": "cholec80", "video_id": v, "frame_idx_25fps": 25 * i,
             "split": "val", "phase": PHASES[(i // 9) % 7],
             **{c: bool(rng.random() < 0.3) for c in TOOL_COLUMNS}}
            for v in (33, 34) for i in range(63)]
    man = pd.DataFrame(rows)

    def unit(x):
        return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)

    emb = unit(rng.standard_normal((len(man), 12)))
    text = {q.text: unit(rng.standard_normal(12)) for q in cholec80_queries()}
    ref = evaluate_model("fake", emb, man, text, None).classification.set_index("classifier")
    truth = probe.phase_labels(man)
    for name, pred in probe.zero_shot_phase_predictions(emb, text).items():
        assert (pred == truth).mean() == pytest.approx(ref.loc[name, "accuracy"], abs=1e-15)


def test_sensitivity_grid_extends_registered_grid(pool, selected):
    """Extended grid (not registered): the registered fits are reproduced exactly, and a
    tool whose choice stays on the registered grid keeps its registered head."""
    X, man = pool
    _, _, tools, tinfos = selected
    assert probe.C_GRID_SENSITIVITY[:len(probe.C_GRID)] == probe.C_GRID
    tools_s, infos_s = probe.select_tool_probes(X, man, grid=probe.C_GRID_SENSITIVITY)
    assert len(infos_s) == len(probe.C_GRID_SENSITIVITY) * len(TOOL_COLUMNS)
    reg = {(i.head, i.C): i.val_score for i in tinfos}
    for i in infos_s:
        if i.C in probe.C_GRID:
            assert i.val_score == reg[(i.head, i.C)]
    for t in TOOL_COLUMNS:
        if tools_s[t].C in probe.C_GRID:
            assert tools_s[t].C == tools[t].C
            np.testing.assert_array_equal(tools_s[t].coef, tools[t].coef)


def test_evaluate_tools_matches_evaluate_probe(pool, selected):
    X, man = pool
    phase, _, tools, _ = selected
    val = _mask(man, 33, 40)
    Xv, mv = X[val], man[val].reset_index(drop=True)
    counts = resample_counts(8, 15)
    ev = probe.evaluate_probe("fake", phase, tools, Xv, mv, {"r1": np.zeros(len(mv), int)},
                              counts)
    per_tool, tmap, boot = probe.evaluate_tools(tools, Xv, mv, counts)
    pd.testing.assert_frame_equal(per_tool, ev.per_tool)
    assert tmap == ev.point["tool_map"]
    np.testing.assert_array_equal(boot, ev.boot["tool_map"])
