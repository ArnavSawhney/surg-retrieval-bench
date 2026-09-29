#!/usr/bin/env python
"""Fetch selected Cholec80 videos from the zip (local path or official URL) into data/tmp/.

Resumable (curl range segments), CRC-32 verified against the zip's central directory,
and checks free disk before each video. ``extract_frames.py`` consumes the files and
deletes each video after its frames are written.

Hard rule this week: only training videos may be fetched. Anything in 41-80 is refused
until docs/preregistration.md is committed.

    python scripts/fetch_cholec80_videos.py --videos 1-5 --connections 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from srb.datasets.overlap import CHOLEC80_TEST_IDS
from srb.datasets.zipsource import fetch_member, open_zip, resolve_zip

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data" / "tmp"
MARGIN = 2 << 30  # keep 2 GiB free beyond what this video needs


def parse_videos(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def refuse_test_videos(videos: list[int]) -> None:
    """Hard stop on 41-80. Lift this deliberately, in a commit, after pre-registration."""
    bad = sorted(set(videos) & CHOLEC80_TEST_IDS)
    if bad:
        sys.exit(f"refusing test videos {bad}: nothing in 41-80 is touched before "
                 "docs/preregistration.md is committed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", required=True, help="e.g. 1-5 or 1,3,4")
    ap.add_argument("--connections", type=int, default=8)
    args = ap.parse_args()
    videos = parse_videos(args.videos)
    refuse_test_videos(videos)

    src = resolve_zip("cholec80")
    zf = open_zip(src)
    TMP.mkdir(parents=True, exist_ok=True)
    names = sorted((f"videos/video{v:02d}.mp4" for v in videos),
                   key=lambda n: zf.getinfo(n).compress_size)  # smallest first
    for name in names:
        dest = TMP / Path(name).name
        if dest.exists() and dest.stat().st_size == zf.getinfo(name).file_size:
            print(f"{name}: already fetched", flush=True)
            continue
        need = zf.getinfo(name).compress_size + zf.getinfo(name).file_size + MARGIN
        free = shutil.disk_usage(TMP).free
        print(f"{name}: {zf.getinfo(name).file_size / 1e9:.2f} GB, free {free / 1e9:.1f} GB",
              flush=True)
        if free < need:
            sys.exit(f"not enough disk for {name}: need {need / 1e9:.1f} GB")
        stats = fetch_member(src, zf, name, dest, connections=args.connections, workdir=TMP)
        print(f"{name}: OK {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
