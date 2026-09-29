#!/usr/bin/env python
"""Extract 1 fps frames + labels for fetched Cholec80 videos and build the manifest.

For each requested video whose mp4 is in data/tmp/ (see fetch_cholec80_videos.py):
  1. check free disk,
  2. read and align its labels (srb.datasets.cholec80),
  3. extract exactly the labelled 1 fps frames (25 fps indices 0, 25, 50, ...),
     short side 448 px, JPEG quality 90 (srb.datasets.frames),
  4. write a per-video manifest fragment and a stats row,
  5. delete the mp4 (unless --keep-video).
Then data/cholec80/manifest.parquet is rebuilt from all fragments.

Videos not fetched yet are skipped with a message; re-run when they arrive.

    python scripts/extract_frames.py --videos 1-5
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from srb.datasets.cholec80 import TOOL_COLUMNS, load_video_labels
from srb.datasets.frames import extract_frames, probe
from srb.datasets.overlap import cholec80_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_cholec80_videos import parse_videos, refuse_test_videos  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cholec80"
RAW, TMP = DATA / "raw", ROOT / "data" / "tmp"
FRAGMENTS, STATS = DATA / "manifests", DATA / "extraction_stats.csv"
MANIFEST = DATA / "manifest.parquet"
COLUMNS = ["dataset", "video_id", "split", "frame_idx_25fps", "t_sec", "path", "phase",
           *TOOL_COLUMNS]


def process(video_id: int, keep_video: bool) -> dict | None:
    mp4 = TMP / f"video{video_id:02d}.mp4"
    if not mp4.exists():
        print(f"video{video_id:02d}: not fetched yet, skipping", flush=True)
        return None
    free = shutil.disk_usage(DATA.parent).free
    if free < 5 << 30:
        sys.exit(f"only {free / 1e9:.1f} GB free; stopping before video{video_id:02d}")

    aligned = load_video_labels(RAW, video_id)
    info = probe(mp4)
    lab = aligned.labels
    out_dir = DATA / "frames" / f"video{video_id:02d}"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # never mix frames from two runs

    t0 = time.time()
    paths = extract_frames(mp4, lab["frame_idx"], out_dir)
    seconds = time.time() - t0

    frag = pd.DataFrame({
        "dataset": "cholec80",
        "video_id": video_id,
        "split": cholec80_split(video_id),
        "frame_idx_25fps": lab["frame_idx"].astype("int64"),
        "t_sec": lab["t_sec"],
        "path": [str(p.relative_to(DATA)) for p in paths],
        "phase": lab["phase"],
        **{c: lab[c] for c in TOOL_COLUMNS},
    })[COLUMNS]
    FRAGMENTS.mkdir(parents=True, exist_ok=True)
    frag.to_parquet(FRAGMENTS / f"video{video_id:02d}.parquet", index=False)

    stats = {
        "video_id": video_id,
        "src_width": info.width, "src_height": info.height,
        "video_frames": info.n_packets, "phase_rows": aligned.n_phase_frames,
        "video_minus_phase_rows": info.n_packets - aligned.n_phase_frames,
        "samples_1fps": aligned.n_samples_1fps, "dropped_tail": len(aligned.dropped_tail),
        "frames_written": len(paths),
        "frames_bytes": sum(p.stat().st_size for p in paths),
        "extract_seconds": round(seconds, 1),
        "video_bytes": mp4.stat().st_size,
    }
    if not keep_video:
        mp4.unlink()
    print(f"video{video_id:02d}: {stats}", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--keep-video", action="store_true")
    args = ap.parse_args()
    videos = parse_videos(args.videos)
    refuse_test_videos(videos)

    rows = [s for v in videos if (s := process(v, args.keep_video))]
    if rows:
        new = pd.DataFrame(rows)
        if STATS.exists():
            old = pd.read_csv(STATS)
            new = pd.concat([old[~old.video_id.isin(new.video_id)], new])
        new.sort_values("video_id").to_csv(STATS, index=False)

    frags = sorted(FRAGMENTS.glob("video*.parquet"))
    if frags:
        manifest = pd.concat([pd.read_parquet(f) for f in frags], ignore_index=True)
        manifest.to_parquet(MANIFEST, index=False)
        print(f"manifest: {len(manifest)} rows, videos "
              f"{sorted(manifest.video_id.unique().tolist())} -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
