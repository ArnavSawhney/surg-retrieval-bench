#!/usr/bin/env python
"""Fetch the Endoscapes2023 CVS-annotated test frames by HTTP range request.

    python scripts/fetch_endoscapes.py labels   # label/split files only (no image)
    python scripts/fetch_endoscapes.py frames   # the 1,799 test keyframes

``frames`` (Deviations, 2026-10-20):
1. builds the manifest of the CVS keyframes of the official test split
   (``srb.datasets.endoscapes.build_manifest``; majority-vote relevance, asserted);
2. refuses if any of those videos is a Cholec80 video (``overlap.py``);
3. fetches each released JPEG into ``data/endoscapes/raw/test/``, checking its size and
   CRC-32 against the zip's central directory (requests of at most 1 MiB, trap 7);
4. reads the real pixel size of **every** JPEG and stops unless all are equal and
   854x480 (either orientation); the COCO width/height are not trusted;
5. resizes each to a 448 px short side (Pillow bicubic) at JPEG q90 into
   ``data/endoscapes/frames/<vid>/``, as for Cholec80 (section 1.1);
6. writes ``data/endoscapes/manifest.parquet`` and ``fetch_stats.json``.

Everything lands under ``data/`` (gitignored; CC BY-NC-SA). Already-fetched files whose
size and CRC match are skipped, so a re-run resumes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from srb.datasets.endoscapes import (LABEL_MEMBERS, MEMBER_PREFIX, RELEASED_SIZES,
                                     build_manifest, cholec80_overlap)
from srb.datasets.frames import resize_image
from srb.datasets.zipsource import (EXPECTED_BYTES, HTTPRangeFile, is_url, open_zip,
                                    read_small_member, resolve_zip, source_size)
from srb.index import git_hash

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "endoscapes"
LABELS = DATA / "labels"


def _ok(path: Path, info) -> bool:
    if not path.exists() or path.stat().st_size != info.file_size:
        return False
    return zlib.crc32(path.read_bytes()) == info.CRC


def fetch_members(src: str, zf, names: list[str], out_of, threads: int) -> int:
    """Fetch members to ``out_of(name)``, skipping verified files. Returns bytes fetched."""
    local = threading.local()
    todo = [n for n in names if not _ok(out_of(n), zf.getinfo(n))]
    print(f"{len(names) - len(todo)} of {len(names)} already on disk and verified; "
          f"fetching {len(todo)}", flush=True)
    done, t0 = [0, 0], time.time()
    lock = threading.Lock()

    def one(name: str) -> None:
        if is_url(src) and not hasattr(local, "f"):
            local.f = HTTPRangeFile(src)
        data = read_small_member(src, zf, name, getattr(local, "f", None))
        dest = out_of(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        tmp.write_bytes(data)
        tmp.rename(dest)
        with lock:
            done[0] += 1
            done[1] += len(data)
            if done[0] % 200 == 0 or done[0] == len(todo):
                print(f"  {done[0]}/{len(todo)} ({done[1] / 1e6:.1f} MB, "
                      f"{time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, todo))
    return done[1]


def check_source(src: str) -> None:
    size = source_size(src)
    if size != EXPECTED_BYTES["endoscapes"]:
        sys.exit(f"archive is {size:,} bytes, expected {EXPECTED_BYTES['endoscapes']:,}")


def cmd_labels(args) -> None:
    src = resolve_zip("endoscapes")
    check_source(src)
    zf = open_zip(src)
    names = [MEMBER_PREFIX + n for n in LABEL_MEMBERS]
    fetch_members(src, zf, names, lambda n: LABELS / n.removeprefix(MEMBER_PREFIX),
                  args.threads)


def cmd_frames(args) -> None:
    t0 = time.time()
    man = build_manifest(LABELS, "test")
    clash = cholec80_overlap(man.video_id.unique())
    if clash:
        sys.exit(f"Endoscapes test videos that are Cholec80 videos: {clash}; refusing")
    print(f"manifest: {len(man)} CVS keyframes, {man.video_id.nunique()} test videos; "
          "R (majority): " + ", ".join(f"{c} {int(man[c].sum())}" for c in ("c1", "c2", "c3")))

    src = resolve_zip("endoscapes")
    check_source(src)
    zf = open_zip(src)
    raw = DATA / "raw"
    raw_of = lambda n: raw / n.removeprefix(MEMBER_PREFIX)  # noqa: E731
    fetched = fetch_members(src, zf, list(man.member), raw_of, args.threads)

    sizes = Counter()
    for n in man.member:
        with Image.open(raw_of(n)) as im:
            sizes[im.size] += 1
    print(f"released JPEG sizes: {dict(sizes)}")
    if len(sizes) != 1 or next(iter(sizes)) not in RELEASED_SIZES:
        sys.exit(f"STOP: released sizes are {dict(sizes)}, expected one of "
                 f"{sorted(RELEASED_SIZES)} for every frame")

    for n, p in zip(man.member, man.path):
        dest = DATA / p
        if not dest.exists() or args.force:
            resize_image(raw_of(n), dest)
    out_sizes = Counter()
    for p in man.path:
        with Image.open(DATA / p) as im:
            out_sizes[im.size] += 1
    man.to_parquet(DATA / "manifest.parquet", index=False)

    raw_bytes = sum(raw_of(n).stat().st_size for n in man.member)
    frame_bytes = sum((DATA / p).stat().st_size for p in man.path)
    stats = {
        "n_frames": len(man), "n_videos": int(man.video_id.nunique()),
        "video_ids": sorted(int(v) for v in man.video_id.unique()),
        "released_size": [list(k) for k in sizes], "resized_size": [list(k) for k in out_sizes],
        "raw_bytes": raw_bytes, "frame_bytes": frame_bytes, "bytes_fetched_this_run": fetched,
        "R_majority": {c: int(man[c].sum()) for c in ("c1", "c2", "c3")},
        "R_unanimous": {c: int(man[f"{c}_unanimous"].sum()) for c in ("c1", "c2", "c3")},
        "git_hash": git_hash(ROOT), "seconds": round(time.time() - t0, 1),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (DATA / "fetch_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "video_ids"}, indent=2))
    if args.delete_raw:
        shutil.rmtree(raw)
        print("deleted data/endoscapes/raw")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("labels", "frames"):
        p = sub.add_parser(name)
        p.add_argument("--threads", type=int, default=8)
    sub.choices["frames"].add_argument("--force", action="store_true",
                                       help="re-resize frames that already exist")
    sub.choices["frames"].add_argument("--delete-raw", action="store_true")
    args = ap.parse_args()
    {"labels": cmd_labels, "frames": cmd_frames}[args.cmd](args)


if __name__ == "__main__":
    main()
