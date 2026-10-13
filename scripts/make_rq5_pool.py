#!/usr/bin/env python
"""Draw the RQ5 failure-mode frame pool (pre-registration section 4.5). Run once.

    python scripts/make_rq5_pool.py

Reads frame identities only (video, frame index, path) of the Cholec80 test videos 41-80
from data/cholec80/manifest.parquet: the same 98,194 frames the test indexes hold.
Writes docs/rq5_frame_pool.csv (2,000 rows in tagging order; no label, score or
prediction) and prints its SHA-256, which goes into LOG.md before any tagging starts.

If the pool file exists, the script only re-derives the pool and checks it is identical:
it never rewrites a pool that tagging may already have started on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from srb.datasets.overlap import CHOLEC80_TEST_IDS
from srb.index import git_hash, manifest_hash
from srb.rq5 import ORDER_SEED, PER_VIDEO, POOL_SEED, draw_pool, pool_csv, pool_sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "rq5_frame_pool.csv"


def main() -> None:
    manifest = pd.read_parquet(ROOT / "data" / "cholec80" / "manifest.parquet",
                               columns=["dataset", "video_id", "frame_idx_25fps", "path"])
    test = (manifest[manifest["video_id"].isin(sorted(CHOLEC80_TEST_IDS))]
            .sort_values(["video_id", "frame_idx_25fps"]).reset_index(drop=True))
    pool = draw_pool(test)
    data, sha = pool_csv(pool), pool_sha256(pool)
    if OUT.exists():
        if OUT.read_bytes() != data:
            sys.exit(f"{OUT.relative_to(ROOT)} exists and differs from the re-derived pool")
        print(f"{OUT.relative_to(ROOT)} exists and is identical to the re-derived pool")
    else:
        OUT.write_bytes(data)
        print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"pool: {len(pool)} frames, {PER_VIDEO} per video x {pool.video_id.nunique()} videos")
    print(f"source: {len(test)} test frames, manifest_sha256 {manifest_hash(test)}")
    print(f"seeds: pool default_rng({POOL_SEED}), order default_rng({ORDER_SEED}); "
          f"numpy {np.__version__}; git {git_hash(ROOT)}")
    print(f"pool SHA-256: {sha}")


if __name__ == "__main__":
    main()
