#!/usr/bin/env python
"""Integrity check of a dataset zip (local path or official URL): size and listing.

* size must equal the server's Content-Length recorded in zipsource.EXPECTED_BYTES;
* Cholec80: the listing must hold 80 videos and 80 phase + 80 tool annotation files;
* Endoscapes: listing counts are reported (nothing is extracted before Week 8).

Reads only the zip's central directory, never the members.

    python scripts/check_zip_integrity.py --dataset cholec80
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from srb.datasets.zipsource import EXPECTED_BYTES, open_zip, resolve_zip, source_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(EXPECTED_BYTES), required=True)
    args = ap.parse_args()

    src = resolve_zip(args.dataset)
    size = source_size(src)
    expected = EXPECTED_BYTES[args.dataset]
    kind = "URL" if src.startswith("http") else "local file"
    print(f"{args.dataset}: {kind}, {size:,} bytes (expected {expected:,})")
    if size != expected:
        sys.exit(f"FAIL: size mismatch ({size - expected:+,} bytes): truncated or different file")

    infos = [i for i in open_zip(src).infolist() if not i.is_dir()]
    print(f"members: {len(infos)} files, {sum(i.file_size for i in infos):,} bytes uncompressed")
    by_dir = Counter((i.filename.split("/")[-2] if "/" in i.filename else ".",
                      i.filename.rsplit(".", 1)[-1]) for i in infos)
    for (d, ext), n in sorted(by_dir.items()):
        print(f"  {d:>28}/*.{ext}: {n}")

    if args.dataset == "cholec80":
        names = [i.filename for i in infos]
        checks = {
            "videos": r"videos/video\d\d\.mp4",
            "phase annotations": r"phase_annotations/video\d\d-phase\.txt",
            "tool annotations": r"tool_annotations/video\d\d-tool\.txt",
        }
        ok = True
        for label, pat in checks.items():
            ids = sorted(int(re.search(r"video(\d\d)", n).group(1))
                         for n in names if re.fullmatch(pat, n))
            good = ids == list(range(1, 81))
            ok &= good
            print(f"  {label}: {len(ids)} (video01..video80 complete: {good})")
        if not ok:
            sys.exit("FAIL: Cholec80 listing incomplete")
    print("OK")


if __name__ == "__main__":
    main()
