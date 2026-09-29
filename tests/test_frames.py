"""Frame extraction selects by decoded frame index, exactly.

A synthetic 25 fps video is generated where frame n has luma Y = 16 + 2n (inside
the limited 16-235 range). Converted to RGB, grey = (Y - 16) * 255/219 = 2n * 255/219
~ 2.33 n. After extraction the mean brightness of the frame written for index i must
match that to within 1 grey level, which fails if the selection were off by even one
frame (a 2.33-level step; lossless encode, flat JPEG).
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from srb.datasets.frames import extract_frames, probe, scaled_size

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_video(path, n=80, rate=25, size="64x48"):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"nullsrc=s={size}:r={rate},geq=lum='16+2*N':cb=128:cr=128",
         "-frames:v", str(n), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
         "-g", "12", "-bf", "2", str(path)],
        check=True,
    )
    return path


def grey(n):
    return 2 * n * 255 / 219


def test_probe(tmp_path):
    info = probe(make_video(tmp_path / "v.mp4"))
    assert (info.width, info.height, info.fps, info.n_packets) == (64, 48, 25, 80)


def test_frames_are_selected_by_index(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    paths = extract_frames(video, [0, 25, 50, 75], tmp_path / "out", short_side=48)
    means = [np.asarray(Image.open(p).convert("L")).mean() for p in paths]
    assert [p.name for p in paths] == ["000000.jpg", "000025.jpg", "000050.jpg", "000075.jpg"]
    assert means == pytest.approx([grey(i) for i in (0, 25, 50, 75)], abs=1.0)


def test_non_contiguous_indices(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    paths = extract_frames(video, [3, 40, 41], tmp_path / "out", short_side=48)
    means = [np.asarray(Image.open(p).convert("L")).mean() for p in paths]
    assert means == pytest.approx([grey(i) for i in (3, 40, 41)], abs=1.0)


def test_resize_short_side_and_quality(tmp_path):
    video = make_video(tmp_path / "v.mp4", size="96x54")
    (p,) = extract_frames(video, [0], tmp_path / "out", short_side=32)
    assert Image.open(p).size == scaled_size(96, 54, 32) == (57, 32)


def test_non_25fps_refused(tmp_path):
    video = make_video(tmp_path / "v30.mp4", rate=30)
    with pytest.raises(ValueError, match="expected 25 fps"):
        extract_frames(video, [0], tmp_path / "out")


def test_index_past_end_refused(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    with pytest.raises(ValueError, match="requested"):
        extract_frames(video, [0, 80], tmp_path / "out")


def test_scaled_size_landscape_and_portrait():
    assert scaled_size(854, 480, 448) == (797, 448)
    assert scaled_size(1920, 1080, 448) == (796, 448)
    assert scaled_size(480, 854, 448) == (448, 797)
