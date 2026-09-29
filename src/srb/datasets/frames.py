"""Frame extraction by 25 fps frame index (not timestamp).

ffmpeg decodes the video and keeps frames whose decoded index ``n`` is in the wanted
set (``select`` on ``n``, with ``-fps_mode passthrough`` so nothing is duplicated or
dropped to fit a frame rate); it scales them and pipes raw RGB to Python, which writes
each JPEG with Pillow at exactly quality 90. ffmpeg's own MJPEG encoder only exposes a
1-31 ``-q:v`` scale, which has no exact "quality 90" setting, hence the Pillow step.

Why ffmpeg and not OpenCV: ffmpeg is on the machine (``which ffmpeg``), its ``select``
filter works on the decoder's frame counter, and ``ffprobe`` gives the container's
frame rate and packet count without decoding. OpenCV's ``CAP_PROP_POS_FRAMES`` seeking
is known to be inexact on some H.264 streams.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED_FPS = Fraction(25)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    n_packets: int  # video packets counted by ffprobe (= frames for these H.264 files)


def probe(path: Path) -> VideoInfo:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found; install ffmpeg")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    return VideoInfo(int(s["width"]), int(s["height"]), Fraction(s["r_frame_rate"]),
                     int(s["nb_read_packets"]))


def scaled_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """(w, h) with the short side = ``short_side`` and the aspect ratio kept."""
    if width <= height:
        return short_side, round(height * short_side / width)
    return round(width * short_side / height), short_side


def extract_frames(video: Path, frame_indices, out_dir: Path, *, short_side: int = 448,
                   quality: int = 90, name=lambda i: f"{i:06d}.jpg") -> list[Path]:
    """Write the frames at the given 0-based decoded indices as JPEGs.

    Asserts the video is 25 fps and that exactly ``len(frame_indices)`` frames came out.
    Returns the paths in index order.
    """
    info = probe(video)
    if info.fps != EXPECTED_FPS:
        raise ValueError(f"{video}: frame rate is {info.fps}, expected 25 fps; the labels "
                         "are indexed at 25 fps, so refusing to guess an alignment")
    wanted = sorted(set(int(i) for i in frame_indices))
    if not wanted:
        raise ValueError("no frame indices requested")
    if wanted[-1] >= info.n_packets:
        raise ValueError(f"{video}: frame {wanted[-1]} requested, video has {info.n_packets}")

    # Every index is a multiple of 25 in our use; express the selection compactly
    # when it is a full arithmetic range, otherwise as an explicit OR of equalities.
    step = wanted[1] - wanted[0] if len(wanted) > 1 else 1
    if wanted == list(range(wanted[0], wanted[-1] + 1, step)):
        expr = f"not(mod(n-{wanted[0]}\\,{step}))*gte(n\\,{wanted[0]})*lte(n\\,{wanted[-1]})"
    else:
        expr = "+".join(f"eq(n\\,{i})" for i in wanted)

    w, h = scaled_size(info.width, info.height, short_side)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
           "-vf", f"select='{expr}',scale={w}:{h}:flags=bicubic",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_bytes = w * h * 3
    paths: list[Path] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for idx in wanted:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            p = out_dir / name(idx)
            Image.fromarray(arr).save(p, quality=quality)
            paths.append(p)
        extra = proc.stdout.read()
        err = proc.stderr.read().decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video}: {err[-2000:]}")
    if len(paths) != len(wanted) or extra:
        raise RuntimeError(f"{video}: expected {len(wanted)} frames, got {len(paths)} "
                           f"(+{len(extra) // frame_bytes} extra)")
    return paths
