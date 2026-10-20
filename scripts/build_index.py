#!/usr/bin/env python
"""Embed the frames of selected Cholec80 videos with one backbone and cache the index.

    python scripts/build_index.py --model clip-vit-l14 --videos 1-5
    python scripts/build_index.py --model siglip-so400m-384 --videos 1-5 --bench
    python scripts/build_index.py --model clip-vit-l14 --dataset endoscapes --bench-only

--dataset endoscapes indexes the CVS keyframes of the Endoscapes2023 test split
(``data/endoscapes/manifest.parquet``, from ``scripts/fetch_endoscapes.py``) into
``data/index/<model>/endoscapes_test``; --videos is not used.

--bench times a few batch sizes on --bench-frames frames (256) first and uses the
fastest (MPS memory and speed are found empirically, one model loaded at a time).
--bench-only stops after the bench and writes nothing.
--dtype float32 forces fp32 on cuda too (the registry default there is fp16), so a
GPU-built index is computed like the local MPS/CPU ones; meta.json records it.
Skips work if the cached index has the same model revision and manifest hash.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from srb.index import cache_is_valid, embed_frames, git_hash, manifest_hash, save_index
from srb.models.registry import _SPECS, SubprocessSpec, get_backbone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_cholec80_videos import parse_videos  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cholec80"
DATA_ROOTS = {"cholec80": DATA, "endoscapes": ROOT / "data" / "endoscapes"}


def video_tag(videos: list[int]) -> str:
    return f"videos_{videos[0]}-{videos[-1]}" if videos == list(
        range(videos[0], videos[-1] + 1)) else "videos_" + "_".join(map(str, videos))


def select_frames(dataset: str, videos_spec: str | None) -> tuple[pd.DataFrame, Path, str]:
    """(manifest rows to embed, in index order; their data root; the index tag)."""
    root = DATA_ROOTS[dataset]
    if not (root / "manifest.parquet").exists():
        sys.exit(f"no {root.relative_to(ROOT)}/manifest.parquet; extract or fetch frames first")
    manifest = pd.read_parquet(root / "manifest.parquet")
    if dataset == "endoscapes":
        if videos_spec:
            sys.exit("--videos does not apply to endoscapes (the whole test split is indexed)")
        if set(manifest.split) != {"test"}:
            sys.exit("the endoscapes manifest must hold the test split only")
        tag = "endoscapes_test"
    else:
        if not videos_spec:
            sys.exit("--videos is required for cholec80")
        videos = parse_videos(videos_spec)
        missing = sorted(set(videos) - set(manifest.video_id))
        if missing:
            sys.exit(f"videos {missing} are not in the manifest yet; run extract_frames.py")
        manifest = manifest[manifest.video_id.isin(videos)]
        tag = video_tag(videos)
    manifest = manifest.sort_values(["video_id", "frame_idx_25fps"]).reset_index(drop=True)
    return manifest, root, tag


def bench(backbone, paths, sizes=(8, 16, 32, 64), n: int = 256) -> tuple[int, dict]:
    sample = paths[:n]
    embed_frames(backbone, sample[:16], batch_size=8, log_every=0)  # warm-up
    rates = {}
    for bs in sizes:
        try:
            _, sec = embed_frames(backbone, sample, batch_size=bs, log_every=0)
            rates[bs] = round(len(sample) / sec, 2)
        except RuntimeError as e:  # MPS/CUDA out of memory
            print(f"  batch {bs}: failed ({str(e)[:80]})")
            break
        print(f"  batch {bs}: {rates[bs]} frames/s", flush=True)
    return max(rates, key=rates.get), rates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(_SPECS))
    ap.add_argument("--dataset", choices=sorted(DATA_ROOTS), default="cholec80")
    ap.add_argument("--videos")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--bench-frames", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto",
                    help="compute dtype; auto = registry default (fp16 iff cuda)")
    args = ap.parse_args()

    manifest, root, tag = select_frames(args.dataset, args.videos)
    videos = sorted(int(v) for v in manifest.video_id.unique())
    mhash = manifest_hash(manifest)
    spec = _SPECS[args.model]
    if isinstance(spec, SubprocessSpec):
        sys.exit(f"{args.model} runs in its own environment: {spec.python} {spec.script} "
                 f"index " + (f"--videos {args.videos}" if args.videos else
                              f"--dataset {args.dataset}"))
    out_dir = ROOT / "data" / "index" / args.model / tag
    if cache_is_valid(out_dir, spec.revision, mhash) and not args.force and not args.bench_only:
        print(f"cached: {out_dir.relative_to(ROOT)} (same revision and manifest); skipping")
        return

    fp16 = {"auto": None, "float32": False, "float16": True}[args.dtype]
    backbone = get_backbone(args.model, fp16=fp16)
    paths = [root / p for p in manifest.path]
    rates = {}
    bs = args.batch_size or spec.default_batch_size
    if args.bench or args.bench_only:
        bs, rates = bench(backbone, paths, n=args.bench_frames)
        print(f"using batch size {bs}")
        if args.bench_only:
            print(f"bench only ({args.bench_frames} frames, {backbone.device.type}, "
                  f"{backbone.dtype}): {rates}; nothing written")
            return

    print(f"embedding {len(paths)} frames with {args.model} on {backbone.device.type}")
    emb, seconds = embed_frames(backbone, paths, batch_size=bs)
    meta = {
        "model": args.model, "hf_id": spec.hf_id, "revision": spec.revision,
        "dataset": args.dataset, "split": sorted(set(manifest.split)), "videos": videos, "n_frames": len(emb), "dim": int(emb.shape[1]),
        "storage_dtype": "float16", "manifest_sha256": mhash,
        "git_hash": git_hash(ROOT), "device": backbone.device.type,
        "compute_dtype": str(backbone.dtype).removeprefix("torch."),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if backbone.device.type == "cuda" else None,
        "batch_size": bs, "bench_frames_per_sec": rates,
        "seconds": round(seconds, 1), "frames_per_sec": round(len(emb) / seconds, 2),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    save_index(out_dir, emb, manifest, meta)
    print(f"wrote {out_dir.relative_to(ROOT)}: {len(emb)} x {emb.shape[1]}, "
          f"{meta['frames_per_sec']} frames/s, git {meta['git_hash'][:12]}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f}s")
