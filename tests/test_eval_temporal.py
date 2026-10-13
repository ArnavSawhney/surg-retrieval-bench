"""scripts/eval_temporal.py end to end on a synthetic project root: the val run selects
on 33-40 only, the test run never fits or selects, appends to results.csv once."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import srb.temporal as tm
from srb.datasets.cholec80 import PHASES
from srb.index import manifest_hash, save_index

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_temporal.py"
FRAMES, DIM, MODEL = 28, 6, "fake"
HEADER = ("model,metric,value,ci_low,ci_high,n_boot,setting,split,seed,git_hash,date,"
          "p,p_holm,significant_holm_0.05\n")
REGISTERED = "fake,phase_map,0.17565920732316115,0.1,0.2,2000,zero-shot,test,1,h,d,,,\n"


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_temporal", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_root(root: Path) -> None:
    rng = np.random.default_rng(0)
    for tag, videos, split in (("videos_1-40", range(1, 41), "val"),
                               ("videos_41-80", range(41, 81), "test")):
        rows, emb, post = [], [], []
        for v in videos:
            y = np.repeat(np.arange(len(PHASES)), FRAMES // len(PHASES))
            for i, k in enumerate(y):
                rows.append({"dataset": "cholec80", "video_id": v, "frame_idx_25fps": 25 * i,
                             "phase": PHASES[k], "path": f"v{v}/{i}.jpg"})
            x = rng.normal(0, 1, (len(y), DIM)).astype(np.float32)
            x[np.arange(len(y)), y % DIM] += 2
            emb.append(x)
            p = rng.dirichlet(np.ones(len(PHASES)), len(y))
            p[np.arange(len(y)), y] += rng.random(len(y))
            post.append((p / p.sum(1, keepdims=True)).astype(np.float32))
        man = pd.DataFrame(rows)
        save_index(root / "data" / "index" / MODEL / tag, np.concatenate(emb), man,
                   {"revision": "rev", "manifest_sha256": manifest_hash(man)})
        P = np.concatenate(post)
        (root / "data" / "probe" / MODEL).mkdir(parents=True, exist_ok=True)
        np.savez(root / "data" / "probe" / MODEL / f"posteriors_{tag}.npz", posterior=P,
                 video_id=man.video_id.to_numpy(), frame_idx_25fps=man.frame_idx_25fps.to_numpy(),
                 phases=np.array(PHASES))
        keep = man.video_id.isin(range(33, 41) if split == "val" else range(41, 81)).to_numpy()
        y = pd.Categorical(man.phase, categories=PHASES).codes
        acc = float((P[keep].argmax(1) == y[keep]).mean())
        path = root / ("results/dev/probe_val_20261013-105258/probe_summary.csv"
                       if split == "val" else "results/test/probe_summary.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model": MODEL, "metric": "phase_accuracy", "value": acc}]).to_csv(
            path, index=False)
    (root / "results" / "results.csv").write_text(HEADER + REGISTERED)


@pytest.fixture
def script(tmp_path, monkeypatch):
    _build_root(tmp_path)
    mod = _load_script()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "MODELS", (MODEL,))
    monkeypatch.setattr(mod, "_SPECS", {MODEL: SimpleNamespace(revision="rev")})
    monkeypatch.setattr(mod, "git_hash", lambda root: "cafe")
    monkeypatch.setattr(tm, "EPOCHS", 2)
    return mod


def _main(mod, monkeypatch, split):
    monkeypatch.setattr(sys, "argv", ["eval_temporal.py", "--split", split, "--jobs", "1",
                                      "--threads", "1"])
    mod.main()


def test_val_then_test_run(script, tmp_path, monkeypatch):
    seen = {"train": set(), "select": set()}
    real_train, real_sel = tm.train_tcn, tm.select_window

    def spy_train(Xt, yt, Xv, yv, *a, **k):
        seen["train"] |= set(Xt)
        seen["select"] |= set(Xv)
        return real_train(Xt, yt, Xv, yv, *a, **k)

    def spy_sel(post, truth, *a, **k):
        seen["select"] |= set(post)
        return real_sel(post, truth, *a, **k)

    monkeypatch.setattr(tm, "train_tcn", spy_train)
    monkeypatch.setattr(tm, "select_window", spy_sel)
    _main(script, monkeypatch, "val")
    assert seen["train"] == set(range(1, 33)) and seen["select"] == set(range(33, 41))
    sel = json.loads((tmp_path / "data/temporal/fake/selection.json").read_text())
    assert sel["w_causal"] in tm.WINDOWS and set(sel["tcn"]) == {"0", "1", "2"}
    dev = next((tmp_path / "results/dev").glob("temporal_val_*"))
    summ = pd.read_csv(dev / "temporal_summary.csv")
    assert summ.label.str.contains("NOT A RESULT").all()
    assert (tmp_path / "results/results.csv").read_text() == HEADER + REGISTERED

    # the test run never fits or selects anything
    def boom(*a, **k):
        raise AssertionError("fit or selection called in the test run")

    for name in ("train_tcn", "select_window", "fit_hmm", "select_tcn"):
        monkeypatch.setattr(tm, name, boom)
    _main(script, monkeypatch, "test")
    text = (tmp_path / "results/results.csv").read_text()
    assert text.startswith(HEADER + REGISTERED)  # appended, registered rows untouched
    res = pd.read_csv(tmp_path / "results/results.csv")
    new = res.iloc[1:]
    assert set(new.setting) == {"single-frame", "ma-causal", "ma-centred", "hmm-forward",
                                "hmm-viterbi", "tcn-causal"}
    tcn = new[(new.setting == "tcn-causal") & (new.metric == "temporal_accuracy")]
    assert sorted(tcn.seed.astype(str)) == ["0", "1", "2", "mean(0,1,2)"]
    assert (new.split == "test").all() and (new.git_hash == "cafe").all()
    t = pd.read_csv(tmp_path / "results/test/temporal_summary.csv")
    sf = t[(t.method == "single-frame") & (t.metric == "accuracy")].value.iloc[0]
    ref = pd.read_csv(tmp_path / "results/test/probe_summary.csv").value.iloc[0]
    assert sf == ref
    with pytest.raises(SystemExit, match="happens once"):
        _main(script, monkeypatch, "test")


def test_single_frame_mismatch_stops_the_run(script, tmp_path, monkeypatch):
    p = tmp_path / "results/dev/probe_val_20261013-105258/probe_summary.csv"
    df = pd.read_csv(p)
    df["value"] += 1e-9
    df.to_csv(p, index=False)
    monkeypatch.setattr(tm, "EPOCHS", 1)
    with pytest.raises(SystemExit, match="differs from the registered probe accuracy"):
        _main(script, monkeypatch, "val")


def test_test_run_needs_a_clean_tree(script, monkeypatch):
    monkeypatch.setattr(script, "git_hash", lambda root: "cafe-dirty")
    with pytest.raises(SystemExit, match="clean tree"):
        _main(script, monkeypatch, "test")
