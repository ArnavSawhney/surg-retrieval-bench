"""Cholec80 label reader and 1 fps alignment (open decision #3).

Synthetic cases always run. The real-label checks run when videos 1-5's annotation
files are present under data/cholec80/raw (gitignored), and skip otherwise.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from srb.datasets.cholec80 import (
    PHASES,
    TOOL_COLUMNS,
    align_1fps,
    load_video_labels,
    read_phases,
    read_tools,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "cholec80" / "raw"
TOOL_HEADER = "Frame\tGrasper\tBipolar\tHook\tScissors\tClipper\tIrrigator\tSpecimenBag\n"


def phase_txt(n, phase="Preparation"):
    return "Frame\tPhase\n" + "".join(f"{i}\t{phase}\n" for i in range(n))


def tool_txt(n_rows, row="1\t0\t0\t0\t0\t0\t0"):
    return TOOL_HEADER + "".join(f"{i * 25}\t{row}\n" for i in range(n_rows))


def test_the_canonical_names():
    assert len(PHASES) == 7 and len(TOOL_COLUMNS) == 7


def test_align_with_real_end_pattern_drops_exactly_the_last_sample():
    """51 phase frames (0..50): 1 fps samples 0, 25, 50. Tools stop at 25.

    The sample at 50 has a phase and no tool vector -> dropped, and recorded.
    """
    a = align_1fps(read_phases(io.StringIO(phase_txt(51))), read_tools(io.StringIO(tool_txt(2))))
    assert a.labels["frame_idx"].tolist() == [0, 25]
    assert a.labels["t_sec"].tolist() == [0.0, 1.0]
    assert a.dropped_tail == [50]
    assert a.n_samples_1fps == 3 and a.n_phase_frames == 51


def test_align_complete_has_nothing_dropped():
    a = align_1fps(read_phases(io.StringIO(phase_txt(50))), read_tools(io.StringIO(tool_txt(2))))
    assert a.dropped_tail == [] and len(a.labels) == 2


def test_tool_index_missing_from_phases_raises():
    with pytest.raises(ValueError, match="absent from the phase labels"):
        align_1fps(read_phases(io.StringIO(phase_txt(30))), read_tools(io.StringIO(tool_txt(3))))


def test_gap_in_tools_before_the_end_raises():
    """Tools stop at 0 but samples 25 and 50 exist: more than the single tail sample."""
    with pytest.raises(ValueError, match="not the single final sample"):
        align_1fps(read_phases(io.StringIO(phase_txt(51))), read_tools(io.StringIO(tool_txt(1))))


def test_non_canonical_phase_raises():
    with pytest.raises(ValueError, match="non-canonical"):
        read_phases(io.StringIO(phase_txt(3, phase="Clipping")))


def test_non_contiguous_phase_frames_raise():
    with pytest.raises(ValueError, match="contiguous"):
        read_phases(io.StringIO("Frame\tPhase\n0\tPreparation\n2\tPreparation\n"))


def test_tool_indices_must_be_multiples_of_25():
    with pytest.raises(ValueError, match="0, 25, 50"):
        read_tools(io.StringIO(TOOL_HEADER + "0\t1\t0\t0\t0\t0\t0\t0\n24\t1\t0\t0\t0\t0\t0\t0\n"))


def test_non_binary_tool_raises():
    with pytest.raises(ValueError, match="0/1"):
        read_tools(io.StringIO(tool_txt(1, row="2\t0\t0\t0\t0\t0\t0")))


@pytest.mark.parametrize("vid", [1, 2, 3, 4, 5])
def test_real_labels_align(vid):
    if not (RAW / "phase_annotations" / f"video{vid:02d}-phase.txt").exists():
        pytest.skip("Cholec80 annotations not present locally")
    a = load_video_labels(RAW, vid)
    lab = a.labels
    # every 1 fps sample has exactly one canonical phase and a full tool vector
    assert lab["phase"].isin(PHASES).all()
    assert lab[list(TOOL_COLUMNS)].notna().all().all()
    assert lab["frame_idx"].is_unique
    # counts: samples = ceil(phase_frames / 25); labelled = samples - dropped tail
    assert a.n_samples_1fps == (a.n_phase_frames + 24) // 25
    assert len(lab) == a.n_samples_1fps - len(a.dropped_tail)
    assert len(a.dropped_tail) <= 1
