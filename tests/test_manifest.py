"""Integrity of the real Cholec80 frame manifest (open decision #3).

Runs only when data/cholec80/manifest.parquet exists (it is gitignored: it is
label-derived per-frame data). Checks that every extracted frame has exactly one
canonical phase and a full tool vector, that frame counts match the label counts
(with the single end-of-video tail sample accounted for), that files exist, and that
nothing from the test split was touched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from srb.datasets.cholec80 import PHASES, TOOL_COLUMNS, load_video_labels
from srb.datasets.overlap import CHOLEC80_TEST_IDS, cholec80_split

DATA = Path(__file__).resolve().parents[1] / "data" / "cholec80"
MANIFEST = DATA / "manifest.parquet"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="no local manifest")


@pytest.fixture(scope="module")
def manifest():
    return pd.read_parquet(MANIFEST)


def test_columns(manifest):
    assert list(manifest.columns) == ["dataset", "video_id", "split", "frame_idx_25fps",
                                      "t_sec", "path", "phase", *TOOL_COLUMNS]


def test_no_test_videos(manifest):
    assert not set(manifest.video_id) & CHOLEC80_TEST_IDS


def test_split_follows_our_rule(manifest):
    assert (manifest.split == manifest.video_id.map(cholec80_split)).all()


def test_one_canonical_phase_and_full_tool_vector(manifest):
    assert manifest.phase.isin(PHASES).all()
    assert manifest[list(TOOL_COLUMNS)].notna().all().all()
    assert (manifest[list(TOOL_COLUMNS)].dtypes == bool).all()
    assert not manifest.duplicated(["video_id", "frame_idx_25fps"]).any()


def test_frame_index_and_time_agree(manifest):
    assert (manifest.frame_idx_25fps % 25 == 0).all()
    assert (manifest.t_sec * 25 == manifest.frame_idx_25fps).all()


def test_counts_match_labels(manifest):
    """Frames per video = 1 fps samples in the phase file - the dropped tail sample."""
    for vid, g in manifest.groupby("video_id"):
        a = load_video_labels(DATA / "raw", int(vid))
        assert len(g) == len(a.labels) == a.n_samples_1fps - len(a.dropped_tail)
        assert g.frame_idx_25fps.tolist() == a.labels.frame_idx.tolist()
        assert (g.phase.to_numpy() == a.labels.phase.to_numpy()).all()


def test_frame_files_exist(manifest):
    missing = [p for p in manifest.path if not (DATA / p).is_file()]
    assert not missing, f"{len(missing)} frame files missing, e.g. {missing[:3]}"
