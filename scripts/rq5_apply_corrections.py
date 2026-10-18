#!/usr/bin/env python
"""Build the corrected RQ5 tags from the raw tags and the out-of-body answers.

    python scripts/rq5_apply_corrections.py

Reads data/rq5/tags.csv (raw, never modified) and data/rq5/out_of_body.csv (answers
from tagging and from the image-only review of frames 0-435), writes
data/rq5/tags_corrected.csv: the raw columns plus ``out_of_body``. For out-of-body
frames smoke, glare, blood and occlusion are set to 0, blur is kept, and clean is 1
unless blur is set (tagging guide, change log 2026-10-18).

Refuses to run unless the raw tags' first 436 rows hash to the SHA-256 in LOG.md and the
review has answered every one of frames 0-435. Computes no per-tag metric: it prints
only tag counts, raw and corrected.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

from srb.rq5 import (ARTEFACTS, OUT_OF_BODY_COLUMNS, RAW_TAGS_SHA256, REVIEWED_FRAMES,
                     TAG_COLUMNS, TAGS, apply_corrections, raw_prefix_sha256,
                     reviewed_prefix, stopping_status)

ROOT = Path(__file__).resolve().parents[1]
RQ5 = ROOT / "data" / "rq5"


def build(tags_path: Path, oob_path: Path, out_path: Path) -> pd.DataFrame:
    raw = tags_path.read_bytes()
    try:
        prefix = raw_prefix_sha256(raw, REVIEWED_FRAMES)
    except ValueError as e:
        sys.exit(f"{tags_path}: {e}")
    if prefix != RAW_TAGS_SHA256:
        sys.exit(f"{tags_path}: the first {REVIEWED_FRAMES} rows differ from the frozen raw "
                 f"tags (SHA-256 in LOG.md)")
    tags, oob = pd.read_csv(tags_path), pd.read_csv(oob_path)
    if list(tags.columns) != TAG_COLUMNS or list(oob.columns) != OUT_OF_BODY_COLUMNS:
        sys.exit("unexpected columns")
    if list(tags["tag_order"]) != list(range(len(tags))):
        sys.exit("raw tags are not a prefix of the tagging order")
    try:
        k = reviewed_prefix(oob)
        if k < REVIEWED_FRAMES:
            sys.exit(f"the review has answered {k} of frames 0-{REVIEWED_FRAMES - 1}; "
                     f"finish it first (tag_rq5.py --review-out-of-body)")
        if set(oob.loc[oob["source"] == "review", "tag_order"]) != set(range(REVIEWED_FRAMES)):
            sys.exit(f"review answers outside frames 0-{REVIEWED_FRAMES - 1}")
        corrected = apply_corrections(tags, oob)
    except ValueError as e:
        sys.exit(f"{oob_path}: {e}")
    tmp = out_path.with_suffix(".tmp")
    corrected.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, out_path)
    return corrected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags", type=Path, default=RQ5 / "tags.csv")
    ap.add_argument("--out-of-body", type=Path, default=RQ5 / "out_of_body.csv")
    ap.add_argument("--out", type=Path, default=RQ5 / "tags_corrected.csv")
    args = ap.parse_args()
    if args.out.resolve() == args.tags.resolve():
        sys.exit("never overwrite the raw tags")
    before = hashlib.sha256(args.tags.read_bytes()).hexdigest()
    corrected = build(args.tags, args.out_of_body, args.out)
    assert hashlib.sha256(args.tags.read_bytes()).hexdigest() == before
    raw = pd.read_csv(args.tags)
    print(f"{len(corrected)} frames, {int(corrected['out_of_body'].sum())} out of body")
    print(f"{'tag':<10}{'raw':>6}{'corrected':>11}")
    for t in TAGS:
        print(f"{t:<10}{int(raw[t].sum()):>6}{int(corrected[t].sum()):>11}")
    s = stopping_status(corrected)
    counts = ", ".join(f"{a} {s['counts'][a]}" for a in ARTEFACTS)
    print(f"stopping rule on corrected tags: {s['reason'] or 'not met'} ({counts})")
    print(f"raw tags SHA-256 {before} (unchanged)\n"
          f"wrote {args.out} SHA-256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
