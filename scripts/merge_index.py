#!/usr/bin/env python
"""Merge cached indexes of consecutive video ranges into one index.

    python scripts/merge_index.py --model clip-vit-l14 --parts 1-5 6-40
        -> data/index/clip-vit-l14/videos_1-40/

The probe trains on 1-32 and tunes on 33-40, so it needs one index over 1-40; the
1-5 indexes were built before the 6-40 ones. The merged index has the same manifest
hash a direct build of the union would have, so build_index.py reports it as cached
and eval_retrieval.py loads it unchanged. The parts stay on disk; meta.json lists
their metas (device, GPU, throughput, git hash) under "merged_from".
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from srb.index import git_hash, merge_indexes, save_index

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import DATA, ROOT, video_tag  # noqa: E402
from fetch_cholec80_videos import parse_videos, refuse_test_videos  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--parts", nargs="+", required=True, help="video ranges, in order")
    args = ap.parse_args()

    ranges = [parse_videos(p) for p in args.parts]
    videos = [v for r in ranges for v in r]
    refuse_test_videos(videos)
    base = ROOT / "data" / "index" / args.model
    full = pd.read_parquet(DATA / "manifest.parquet")
    expected = (full[full.video_id.isin(videos)]
                .sort_values(["video_id", "frame_idx_25fps"]).reset_index(drop=True))
    emb, manifest, meta = merge_indexes([base / video_tag(r) for r in ranges], expected)
    out_dir = base / video_tag(sorted(videos))
    if out_dir.exists():
        sys.exit(f"{out_dir.relative_to(ROOT)} already exists; remove it first")
    meta["git_hash"] = git_hash(ROOT)
    meta["created"] = dt.datetime.now().isoformat(timespec="seconds")
    save_index(out_dir, emb, manifest, meta)
    print(f"wrote {out_dir.relative_to(ROOT)}: {len(emb)} x {emb.shape[1]} "
          f"from {', '.join(video_tag(r) for r in ranges)}")


if __name__ == "__main__":
    main()
