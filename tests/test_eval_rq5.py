"""RQ5 per-tag analysis: accounting, multi-tag frames, drop sign, frozen-hash check."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from srb.resample import resample_counts
from srb.rq5 import (ARTEFACTS, OUT_OF_BODY_COLUMNS, POOL_COLUMNS, TAG_COLUMNS, TAGS,
                     apply_corrections, check_frozen, drops, mean_drop, per_tag_accuracy)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_rq5  # noqa: E402


def _tags(rows):
    """rows: (video_id, set of tags)."""
    return pd.DataFrame([{"tag_order": i, "video_id": v, "frame_idx_25fps": 25 * i,
                          **{t: int(t in ts) for t in TAGS}, "tagged_at": "t"}
                         for i, (v, ts) in enumerate(rows)])[TAG_COLUMNS]


def test_per_tag_accounting_multi_tag_and_drop_sign(monkeypatch):
    monkeypatch.setattr("srb.rq5.MIN_FRAMES_FOR_ACCURACY", 1)
    rows = [(1, {"clean"}), (1, {"clean"}), (2, {"clean"}), (2, {"clean"}),
            (1, {"blur", "smoke"}), (2, {"blur"}), (2, {"smoke"}),
            (1, {"glare"}), (1, {"blood"}), (2, {"occlusion"})]
    tags = _tags(rows)
    correct = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 1])
    videos = np.array([1, 2])
    counts = np.array([[1, 1], [2, 0], [0, 2]])
    acc = per_tag_accuracy(tags, correct, videos, counts)
    assert acc["clean"]["n"] == 4 and acc["clean"]["value"] == 0.75
    assert acc["blur"]["n"] == 2 and acc["blur"]["value"] == 0.5      # frame 4 counts in
    assert acc["smoke"]["n"] == 2 and acc["smoke"]["value"] == 0.5    # blur and in smoke
    assert acc["blur"]["n_videos"] == 2
    # resample 1 = video 1 twice: clean 2/2, blur 0/1; resample 2 = video 2 twice
    np.testing.assert_allclose(acc["clean"]["boot"], [0.75, 1.0, 0.5])
    np.testing.assert_allclose(acc["blur"]["boot"], [0.5, 0.0, 1.0])
    assert np.isnan(acc["glare"]["boot"][2])  # no glare frame drawn
    d = drops(acc)
    assert set(d) == set(ARTEFACTS)
    assert d["blur"]["value"] == 0.25  # clean 0.75 - blur 0.5: positive = worse on blur
    assert d["glare"]["value"] == -0.25 and d["blood"]["value"] == 0.75
    np.testing.assert_allclose(d["blur"]["boot"], [0.25, 1.0, -0.5])
    md = mean_drop(d)
    assert md["value"] == pytest.approx(np.mean([d[a]["value"] for a in ARTEFACTS]))


def test_zero_shot_headline_is_the_mean_over_classifiers(monkeypatch):
    monkeypatch.setattr("srb.rq5.MIN_FRAMES_FOR_ACCURACY", 1)
    tags = _tags([(1, {"clean"}), (1, {"clean"}), (1, {"blur"})])
    c = np.array([[1, 0], [1, 1], [0, 1]])
    acc = per_tag_accuracy(tags, c, np.array([1]), np.ones((2, 1)))
    assert acc["clean"]["value"] == pytest.approx((1.0 + 0.5) / 2)
    assert acc["blur"]["value"] == pytest.approx(0.5)


def test_fewer_than_20_frames_counts_only():
    tags = _tags([(1, {"clean"})] * 20 + [(1, {"blur"})] * 19)
    acc = per_tag_accuracy(tags, np.ones(39), np.array([1]), np.ones((3, 1)))
    assert acc["clean"]["value"] == 1.0
    assert acc["blur"]["n"] == 19 and np.isnan(acc["blur"]["value"])
    assert np.isnan(acc["blur"]["boot"]).all()


def test_analyse_tables_and_pairs(monkeypatch):
    monkeypatch.setattr("srb.rq5.MIN_FRAMES_FOR_ACCURACY", 1)
    rng = np.random.default_rng(0)
    rows = [(v, {t}) for v in (41, 42, 43) for t in TAGS for _ in range(3)]
    tags = _tags(rows)
    videos = np.array([41, 42, 43])
    counts = resample_counts(3, 50, 1)
    correct = {m: {"probe": rng.integers(0, 2, len(tags)),
                   "zs": rng.integers(0, 2, (len(tags), 5))}
               for m in ("clip-vit-l14", "siglip-so400m-384", "medsiglip-448", "peskavlp")}
    t = eval_rq5.analyse(tags, correct, counts, videos)
    pt = t["per_tag"]
    assert len(pt) == 4 * 2 * len(TAGS) and pt[pt.tag == "clean"]["drop"].isna().all()
    bm = t["backbone_mean"]
    blur = bm[(bm.tag == "blur") & (bm.metric == "probe_accuracy")]["drop"].iloc[0]
    assert blur == pytest.approx(pt[(pt.tag == "blur") & (pt.metric == "probe_accuracy")]
                                 ["drop"].mean())
    pairs = t["pairs"]
    md = t["mean_drop"].set_index(["model", "metric"])["mean_drop"]
    got = pairs[(pairs.pair == "peskavlp - clip-vit-l14")
                & (pairs.metric == "probe_accuracy")]["diff"].iloc[0]
    assert got == pytest.approx(md["peskavlp", "probe_accuracy"]
                                - md["clip-vit-l14", "probe_accuracy"])
    rr = eval_rq5.results_rows(t)
    assert not rr.duplicated(["model", "metric"]).any()
    assert "rq5_probe_accuracy_mean_drop_difference" in set(rr.metric)


def _frozen_dir(tmp_path):
    d = tmp_path / "rq5"
    d.mkdir()
    rows = [(41, {"clean"}), (41, {"occlusion"}), (42, {"blur"})]
    raw = _tags(rows)
    raw.to_csv(d / "tags.csv", index=False, lineterminator="\n")
    oob = pd.DataFrame([[1, 1, "review", "t"]], columns=OUT_OF_BODY_COLUMNS)
    oob.to_csv(d / "out_of_body.csv", index=False, lineterminator="\n")
    apply_corrections(raw, oob).to_csv(d / "tags_corrected.csv", index=False,
                                       lineterminator="\n")
    pool = raw.assign(draw_idx=raw.tag_order, path="x.jpg")[POOL_COLUMNS]
    pool.to_csv(tmp_path / "pool.csv", index=False, lineterminator="\n")
    sha = {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
           for n in ("tags.csv", "out_of_body.csv", "tags_corrected.csv")}
    return d, tmp_path / "pool.csv", sha


def test_hash_check_refuses_a_changed_file(tmp_path, monkeypatch):
    d, pool, sha = _frozen_dir(tmp_path)
    psha = hashlib.sha256(pool.read_bytes()).hexdigest()
    monkeypatch.setattr(eval_rq5, "stopping_status", lambda df: {"done": True})
    corr = eval_rq5.load_tags(d, pool, sha, psha)
    assert corr["occlusion"].tolist() == [0, 0, 0] and corr["out_of_body"].tolist() == [0, 1, 0]
    check_frozen(d, sha)
    with open(d / "tags.csv", "a") as f:
        f.write("3,42,75,0,0,0,0,0,1,t\n")
    with pytest.raises(ValueError, match="tags.csv"):
        check_frozen(d, sha)
    with pytest.raises(SystemExit, match="differs from the frozen"):
        eval_rq5.load_tags(d, pool, sha, psha)


def test_load_tags_refuses_inconsistent_corrections_and_incomplete_tagging(tmp_path):
    d, pool, sha = _frozen_dir(tmp_path)
    psha = hashlib.sha256(pool.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="not complete"):  # 3 frames: rule not met
        eval_rq5.load_tags(d, pool, sha, psha)
    c = pd.read_csv(d / "tags_corrected.csv")
    c.loc[1, "occlusion"] = 1
    c.to_csv(d / "tags_corrected.csv", index=False, lineterminator="\n")
    sha["tags_corrected.csv"] = hashlib.sha256((d / "tags_corrected.csv").read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="raw tags \\+ out-of-body"):
        eval_rq5.load_tags(d, pool, sha, psha)


def test_frozen_hashes_match_the_log():
    log = (Path(__file__).resolve().parents[1] / "LOG.md").read_text()
    from srb.rq5 import FROZEN_SHA256
    for sha in FROZEN_SHA256.values():
        assert sha in log
