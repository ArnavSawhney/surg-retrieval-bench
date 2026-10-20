"""Endoscapes CVS evaluation on a synthetic root: checks, check-only, the run's outputs,
append-only results.csv, and the shared resample stream. Random embeddings, real
query strings; no real Endoscapes frame or index is touched."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from srb.datasets.endoscapes import build_manifest
from srb.index import manifest_hash, save_index
from srb.models.registry import _SPECS
from srb.queries import endoscapes_queries
from srb.resample import resample_counts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_endoscapes  # noqa: E402

MODELS = ["clip-vit-l14", "peskavlp"]
VIDEOS = [162, 163, 164]
HEADER = ("model,metric,value,ci_low,ci_high,n_boot,setting,split,seed,git_hash,date,"
          "p,p_holm,significant_holm_0.05\n")
OLD_ROW = "x,overall_map,0.17565920732316115,,,,zero-shot,test,20261011,abc,2026-10-13,,,\n"


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def _meta_rows(rng):
    """25 keyframes per test video with random binary votes, plus non-keyframes and a
    train video, as in all_metadata.csv."""
    rows = []
    for v in [1, *VIDEOS]:
        for i in range(27):
            key = i < 25
            votes = rng.integers(0, 2, size=(3, 3)) if key else None
            mean = votes.mean(0).tolist() if key else [0.0, 0.0, 0.0]
            rows.append({"vid": v, "frame": 125 * (i + 1) + (0 if key else 25),
                         "avg_cvs": str(mean), "C1": mean[0], "C2": mean[1], "C3": mean[2],
                         "is_ds_keyframe": key,
                         **{f"cvs_annotator_{a + 1}": (str(votes[a].tolist()) if key
                                                       else float("nan")) for a in range(3)}})
    return pd.DataFrame(rows)


@pytest.fixture
def root(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    data = tmp_path / "data" / "endoscapes"
    labels = data / "labels"
    labels.mkdir(parents=True)
    _meta_rows(rng).to_csv(labels / "all_metadata.csv", index=False)
    for s, ids in {"train": [1], "val": [121], "test": VIDEOS}.items():
        (labels / f"{s}_vids.txt").write_text("".join(f"{float(i):.18e}\n" for i in ids))
    man = build_manifest(labels, "test")
    man.to_parquet(data / "manifest.parquet", index=False)
    for p in man.path:
        (data / p).parent.mkdir(parents=True, exist_ok=True)
        (data / p).write_bytes(b"jpeg")
    texts = [q.text for q in endoscapes_queries()]
    for m in MODELS:
        rev = _SPECS[m].revision
        emb = _unit(rng.standard_normal((len(man), 8)))
        save_index(tmp_path / "data" / "index" / m / "endoscapes_test", emb, man,
                   {"model": m, "revision": rev, "dataset": "endoscapes",
                    "manifest_sha256": manifest_hash(man), "device": "mps",
                    "compute_dtype": "float32", "git_hash": "synthetic"})
        cache = tmp_path / "data" / "text_emb" / f"{m}_{rev[:12]}_endoscapes.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, texts=np.array(texts), emb=_unit(rng.standard_normal((len(texts), 8))),
                 revision=rev)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "results.csv").write_text(HEADER + OLD_ROW)
    expected = {"n_frames": len(man), "video_ids": VIDEOS,
                "R": {c: int(man[c].sum()) for c in ("c1", "c2", "c3")},
                "R_unanimous": {c: int(man[f"{c}_unanimous"].sum()) for c in ("c1", "c2", "c3")}}
    monkeypatch.setattr(eval_endoscapes, "EXPECTED", expected)
    return tmp_path


def _run(root, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["eval_endoscapes.py", "--root", str(root),
                                      "--models", *MODELS, *extra])
    eval_endoscapes.main()


def test_check_only_scores_nothing(root, monkeypatch, capsys):
    _run(root, monkeypatch, "--check-only")
    assert "stopped before any score" in capsys.readouterr().out
    assert not (root / "results" / "test").exists()
    assert (root / "results" / "results.csv").read_text() == HEADER + OLD_ROW


def test_check_fails_on_wrong_relevance_counts(root, monkeypatch):
    bad = dict(eval_endoscapes.EXPECTED, R={"c1": -1, "c2": 0, "c3": 0})
    monkeypatch.setattr(eval_endoscapes, "EXPECTED", bad)
    with pytest.raises(SystemExit, match="CHECK FAILED: R"):
        _run(root, monkeypatch, "--check-only")


def test_check_fails_if_manifest_drifts_from_labels(root, monkeypatch):
    p = root / "data" / "endoscapes" / "manifest.parquet"
    man = pd.read_parquet(p)
    man.loc[0, "c1"] = not man.loc[0, "c1"]
    man.to_parquet(p, index=False)
    with pytest.raises(SystemExit, match="rebuilt from the label files"):
        _run(root, monkeypatch, "--check-only")


def test_check_fails_on_an_index_of_another_manifest(root, monkeypatch):
    d = root / "data" / "index" / "peskavlp" / "endoscapes_test"
    meta = json.loads((d / "meta.json").read_text())
    meta["dataset"] = "cholec80"
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="peskavlp: index is not of this Endoscapes"):
        _run(root, monkeypatch, "--check-only")


def test_dry_run_outputs(root, monkeypatch):
    _run(root, monkeypatch)
    out = root / "results" / "test"
    summary = pd.read_csv(out / "endoscapes_summary.csv")
    assert (summary.label == "DRY RUN on a synthetic root, NOT A RESULT").all()
    for m in MODELS:
        pt = pd.read_csv(out / f"endoscapes_{m}_per_target.csv")
        assert pt.target.tolist() == ["cvs_c1", "cvs_c2", "cvs_c3"]
        s = summary[(summary.model == m)].set_index("metric").value
        assert s["cvs_map"] == pytest.approx(pt.ap.mean())
        assert s["cvs_prevalence"] == pytest.approx(pt.prevalence.mean())
        assert s["rq2_mean_sd"] == pytest.approx(pt.ap_sd.mean())
        assert not any(k.startswith(("overall", "phase", "tool", "zs_phase")) for k in s.index)
        row = summary[(summary.model == m) & (summary.metric == "cvs_map")].iloc[0]
        assert row.n_boot == 2000 and row.ci_low <= row.value <= row.ci_high
        # unanimity uses the same strings and pool but the 3-of-3 columns
        upt = pd.read_csv(out / f"endoscapes_unanimity_{m}_per_target.csv")
        man = pd.read_parquet(root / "data" / "endoscapes" / "manifest.parquet")
        assert upt.n_relevant.tolist() == [int(man[f"c{k}_unanimous"].sum()) for k in (1, 2, 3)]
    pairs = pd.read_csv(out / "endoscapes_pairs.csv")
    assert "p" not in pairs and pairs.n_boot.eq(2000).all()   # descriptive, no p-values
    run = json.loads((out / "endoscapes_run.json").read_text())
    assert run["bootstrap_video_order"] == VIDEOS and run["boot"] == 2000

    text = (root / "results" / "results.csv").read_text()
    assert text.startswith(HEADER + OLD_ROW)                  # appended, prefix untouched
    new = pd.read_csv(root / "results" / "results.csv").iloc[1:]
    assert (new.split == "endoscapes-test").all()
    assert set(new.setting) == {"zero-shot", "zero-shot-unanimity-not-registered"}
    assert (new.metric == "cvs_map_difference").sum() == 2   # one pair x two analyses


def test_resamples_are_the_first_2000_of_the_cholec80_stream():
    """n = 40 for both datasets: the descriptive matrix is the first 2,000 rows of the
    B = 10,000 Cholec80 draw (Deviations, 2026-10-20)."""
    assert np.array_equal(resample_counts(40, 2000), resample_counts(40, 10_000)[:2000])
