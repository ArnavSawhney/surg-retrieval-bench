"""Endoscapes2023 label reader: relevance rule, splits, manifest, Cholec80 overlap.

Synthetic label files are hand-made; the last tests run on the real label files and
manifest only when they exist under data/endoscapes/ (gitignored, label-derived).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from srb.datasets import endoscapes as es
from srb.datasets.frames import resize_image
from srb.datasets.overlap import ENDOSCAPES_TO_CHOLEC80

DATA = Path(__file__).resolve().parents[1] / "data" / "endoscapes"


def meta_rows(rows):
    """rows: (vid, frame, keyframe, annotator votes a1, a2, a3 or None, C override)."""
    out = []
    for vid, frame, key, votes, c in rows:
        if key:
            ann = [str(list(v)) if v is not None else float("nan") for v in votes]
            mean = c if c is not None else [sum(v[j] for v in votes) / 3 for j in range(3)]
        else:
            ann, mean = [float("nan")] * 3, c if c is not None else [0.0, 0.0, 0.0]
        out.append({"vid": vid, "frame": frame, "avg_cvs": str(mean), "C1": mean[0],
                    "C2": mean[1], "C3": mean[2], "is_ds_keyframe": key,
                    "cvs_annotator_1": ann[0], "cvs_annotator_2": ann[1],
                    "cvs_annotator_3": ann[2]})
    return pd.DataFrame(out)


# --- relevance rule -----------------------------------------------------------------

def test_majority_and_unanimity_hand_cases():
    """Votes per criterion, summed over the three annotators:
    frame 125: C1 (1,1,0) = 2 -> relevant, not unanimous; C2 (1,0,0) = 1 -> not relevant;
               C3 (1,1,1) = 3 -> relevant and unanimous.
    frame 250: all 0 -> nothing relevant.
    frame 375: C1 (0,1,1) = 2 -> relevant (a 2-1 split, the order of raters is irrelevant)."""
    m = meta_rows([
        (5, 125, True, [(1, 1, 1), (1, 0, 1), (0, 0, 1)], None),
        (5, 250, True, [(0, 0, 0)] * 3, None),
        (5, 375, True, [(0, 0, 0), (1, 0, 0), (1, 0, 0)], None),
    ])
    lab = es.cvs_labels(m)
    assert lab[["c1_votes", "c2_votes", "c3_votes"]].values.tolist() == [[2, 1, 3], [0, 0, 0],
                                                                        [2, 0, 0]]
    assert lab[["c1", "c2", "c3"]].values.tolist() == [[True, False, True],
                                                      [False, False, False],
                                                      [True, False, False]]
    assert lab[["c1_unanimous", "c2_unanimous", "c3_unanimous"]].values.tolist() == [
        [False, False, True], [False, False, False], [False, False, False]]


def test_one_of_three_is_not_relevant_mean_one_third():
    """1 of 3 -> mean 1/3 < 0.5 -> not relevant; 2 of 3 -> 2/3 >= 0.5 -> relevant."""
    m = meta_rows([(1, 25, True, [(1, 1, 0), (0, 1, 0), (0, 0, 0)], None)])
    lab = es.cvs_labels(m)
    assert (lab.c1.item(), lab.c2.item(), lab.c3.item()) == (False, True, False)


def test_non_keyframes_are_ignored_even_with_derived_labels():
    """The archive's non-keyframes carry derived C values but no annotator votes;
    they must never enter the evaluated set."""
    m = meta_rows([(1, 25, False, None, [1.0, 1.0, 1.0]),
                   (1, 125, True, [(0, 0, 0)] * 3, None)])
    lab = es.cvs_labels(m)
    assert lab.frame.tolist() == [125]


def test_missing_annotator_raises():
    """Fewer than 3 raters on a keyframe: the rule cannot be applied literally."""
    m = meta_rows([(1, 25, True, [(1, 1, 1), (1, 1, 1), None], [1.0, 1.0, 1.0])])
    with pytest.raises(ValueError, match="missing annotator"):
        es.cvs_labels(m)


@pytest.mark.parametrize("cell", ["[1, 2, 0]", "[1, 0]", "[0.5, 0, 0]", "1"])
def test_non_binary_vote_raises(cell):
    with pytest.raises(ValueError):
        es.parse_votes(cell)


def test_released_mean_mismatch_raises():
    """If the released C columns ever disagree with the votes, stop rather than pick one."""
    m = meta_rows([(1, 25, True, [(1, 0, 0), (1, 0, 0), (0, 0, 0)], [0.0, 0.0, 0.0])])
    with pytest.raises(ValueError, match="annotator mean"):
        es.cvs_labels(m)


# --- splits and manifest ------------------------------------------------------------

def write_label_dir(tmp_path, splits, meta):
    for s, ids in splits.items():
        (tmp_path / f"{s}_vids.txt").write_text(
            "".join(f"{float(i):.18e}\n" for i in ids))
    meta.to_csv(tmp_path / "all_metadata.csv", index=False)
    return tmp_path


def test_split_ids_parse_floats(tmp_path):
    (tmp_path / "t.txt").write_text("1.620000000000000000e+02\n1.630000000000000000e+02\n")
    assert es.read_split_ids(tmp_path / "t.txt") == [162, 163]
    (tmp_path / "bad.txt").write_text("1.625e+02\n")
    with pytest.raises(ValueError):
        es.read_split_ids(tmp_path / "bad.txt")


def test_overlapping_splits_raise(tmp_path):
    write_label_dir(tmp_path, {"train": [1, 2], "val": [2], "test": [3]}, meta_rows([]))
    with pytest.raises(ValueError, match="shares"):
        es.read_splits(tmp_path)


def test_manifest_integrity(tmp_path):
    """Only the test split's keyframes, sorted by (video, frame), Cholec80 key columns
    first, paths and archive members named {vid}_{frame}.jpg."""
    meta = meta_rows([
        (163, 250, True, [(1, 1, 0)] * 3, None),
        (162, 500, True, [(1, 0, 0), (1, 0, 0), (0, 0, 0)], None),
        (162, 125, True, [(0, 0, 0)] * 3, None),
        (162, 150, False, None, [1.0, 1.0, 1.0]),   # derived, not annotated
        (7, 125, True, [(1, 1, 1)] * 3, None),      # train video
    ])
    d = write_label_dir(tmp_path, {"train": [7], "val": [121], "test": [162, 163]}, meta)
    man = es.build_manifest(d, "test")
    assert list(man.columns) == list(es.MANIFEST_COLUMNS)
    assert man[["video_id", "frame_idx_25fps"]].values.tolist() == [[162, 125], [162, 500],
                                                                    [163, 250]]
    assert (man.dataset == "endoscapes").all() and (man.split == "test").all()
    assert man.path.tolist() == ["frames/162/162_125.jpg", "frames/162/162_500.jpg",
                                 "frames/163/163_250.jpg"]
    assert man.member.tolist()[0] == "endoscapes/test/162_125.jpg"
    assert man.t_sec.tolist() == [5.0, 20.0, 10.0]
    assert man.c1.tolist() == [False, True, True] and man.c3.tolist() == [False, False, False]
    assert man.c1_unanimous.tolist() == [False, False, True]


def test_manifest_refuses_a_video_without_keyframes(tmp_path):
    meta = meta_rows([(162, 125, True, [(0, 0, 0)] * 3, None)])
    d = write_label_dir(tmp_path, {"train": [1], "val": [2], "test": [162, 163]}, meta)
    with pytest.raises(ValueError, match="without CVS keyframes"):
        es.build_manifest(d, "test")


# --- overlap with Cholec80 ----------------------------------------------------------

def test_no_endoscapes_test_video_is_a_cholec80_video():
    """Official test split = public IDs 162-201 (test_vids.txt); CAMMA's overlap table
    lists Endoscapes-test x Cholec80 (every split) as 0."""
    assert es.cholec80_overlap(range(162, 202)) == {}
    assert all(e < 162 for e in ENDOSCAPES_TO_CHOLEC80)


def test_overlap_reports_known_pairs():
    assert es.cholec80_overlap([1, 121, 162]) == {1: 67, 121: 66}


# --- preprocessing ------------------------------------------------------------------

def test_resize_image_short_side_448(tmp_path):
    src, dest = tmp_path / "a.jpg", tmp_path / "out" / "a.jpg"
    Image.new("RGB", (854, 480), (120, 60, 30)).save(src, quality=95)
    assert resize_image(src, dest) == (854, 480)
    with Image.open(dest) as im:
        assert im.size == (797, 448)


# --- the real files (skipped when absent) -------------------------------------------

LABELS = DATA / "labels"
needs_labels = pytest.mark.skipif(not (LABELS / "all_metadata.csv").exists(),
                                  reason="no local Endoscapes labels")


@needs_labels
def test_real_test_manifest_counts():
    """Counts from the label files (Deviations, 2026-10-20)."""
    man = es.build_manifest(LABELS, "test")
    assert len(man) == 1799
    assert sorted(man.video_id.unique()) == list(range(162, 202))
    assert [int(man[c].sum()) for c in ("c1", "c2", "c3")] == [423, 302, 501]
    assert [int(man[f"{c}_unanimous"].sum()) for c in ("c1", "c2", "c3")] == [77, 138, 222]
    assert es.cholec80_overlap(man.video_id.unique()) == {}


@needs_labels
def test_every_keyframe_has_three_binary_raters_and_majority_equals_mean():
    meta = pd.read_csv(LABELS / "all_metadata.csv")
    lab = es.cvs_labels(meta)   # raises on any violation
    assert len(lab) == 11090


@pytest.mark.skipif(not (DATA / "manifest.parquet").exists(), reason="no local manifest")
def test_real_frames_on_disk_match_the_manifest():
    man = pd.read_parquet(DATA / "manifest.parquet")
    assert man.equals(es.build_manifest(LABELS, "test"))
    for p in man.path:
        assert (DATA / p).exists(), p
    with Image.open(DATA / man.path.iloc[0]) as im:
        assert min(im.size) == 448
