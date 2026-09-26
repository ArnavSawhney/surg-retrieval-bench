"""Tests for srb.datasets.overlap against CAMMA's camma_dataset_overlaps README.

Expected counts are the "Overlap Count" column of the README Summary table at commit
8347b9f4cb02ebe739747903e6eada272ee9d25e, in *their* split convention (Cholec80
train 1-40, val 41-48, test 49-80), so that the transcription is checked cell by cell.
"""

from __future__ import annotations

import pandas as pd
import pytest

from srb.datasets.overlap import (
    CHOLECT50_IN_CHOLEC80,
    ENDOSCAPES_TO_CHOLEC80,
    M2CAI16_TOOL_OVERLAP,
    assert_no_test_leakage,
    cholec80_split,
    forbidden_for_training,
)

CAMMA_C80 = {"train": set(range(1, 41)), "val": set(range(41, 49)), "test": set(range(49, 81))}


@pytest.mark.parametrize(
    "c80_split,t50_split,count",
    [
        ("train", "train", 16), ("train", "val", 3), ("train", "test", 4),
        ("val", "train", 3), ("val", "val", 0), ("val", "test", 1),
        ("test", "train", 12), ("test", "val", 2), ("test", "test", 4),
    ],
)
def test_cholec80_x_cholect50_counts_match_readme(c80_split, t50_split, count):
    assert len(CAMMA_C80[c80_split] & CHOLECT50_IN_CHOLEC80[t50_split]) == count


def test_endoscapes_x_cholec80_counts_match_readme():
    """README: Endoscapes-train x Cholec80-test = 5 ([1,2,3,4,7] -> [67,68,70,71,72]),
    Endoscapes-val x Cholec80-test = 1 ([121] -> [66]); every other cell is 0."""
    train = {e: c for e, c in ENDOSCAPES_TO_CHOLEC80.items() if e != 121}
    assert sorted(train) == [1, 2, 3, 4, 7]
    assert sorted(train.values()) == [67, 68, 70, 71, 72]
    assert ENDOSCAPES_TO_CHOLEC80[121] == 66
    assert all(c in CAMMA_C80["test"] for c in ENDOSCAPES_TO_CHOLEC80.values())


def test_m2cai_counts_match_readme():
    """README: 10 (M2CAI-tool train 1-10 <-> Cholec80 67-76), 5 (test 11-15 <-> 61,62,64,65,66)."""
    assert [len(m) for m, _ in M2CAI16_TOOL_OVERLAP.values()] == [10, 5]
    assert all(len(m) == len(c) for m, c in M2CAI16_TOOL_OVERLAP.values())


def test_cholect50_covers_22_of_our_40_test_videos():
    """Mapped by video ID to *our* test split 41-80: 3+1 (their val) + 12+2+4 (their test)."""
    assert forbidden_for_training("cholect50") == {
        42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 60, 62, 65, 66, 68, 70, 73, 74, 75, 78, 79, 80
    }


def test_forbidden_endoscapes():
    assert forbidden_for_training("endoscapes") == {1, 2, 3, 4, 7, 121}


def test_forbidden_m2cai_and_cholec80():
    assert forbidden_for_training("m2cai16_tool") == set(range(1, 16))
    assert forbidden_for_training("Cholec80") == set(range(41, 81))


@pytest.mark.parametrize("ds", ["cholecseg8k", "cholec80_cvs", "cholect45"])
def test_unverified_datasets_raise(ds):
    """No mapping must never read as "no overlap"."""
    with pytest.raises(KeyError, match="No verified"):
        forbidden_for_training(ds)


def test_our_split_rule():
    assert [cholec80_split(v) for v in (1, 32, 33, 40, 41, 80)] == [
        "train", "train", "val", "val", "test", "test"]
    with pytest.raises(ValueError):
        cholec80_split(81)


def test_assert_no_test_leakage_passes_clean_manifest():
    m = pd.DataFrame({"dataset": ["cholec80"] * 3 + ["endoscapes", "cholect50"],
                      "video_id": [1, 5, 40, 5, 92]})
    assert_no_test_leakage(m)


@pytest.mark.parametrize("ds,vid", [("cholec80", 41), ("endoscapes", 121),
                                    ("cholect50", 80), ("m2cai16_tool", 3)])
def test_assert_no_test_leakage_catches_each_source(ds, vid):
    m = pd.DataFrame({"dataset": ["cholec80", ds], "video_id": [1, vid]})
    with pytest.raises(ValueError, match="Test leakage"):
        assert_no_test_leakage(m)


def test_assert_no_test_leakage_refuses_unknown_dataset():
    m = pd.DataFrame({"dataset": ["cholecseg8k"], "video_id": [1]})
    with pytest.raises(KeyError):
        assert_no_test_leakage(m)
