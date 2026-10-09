#!/usr/bin/env python
"""Embed the frames of selected Cholec80 videos with one backbone and cache the index.

    python scripts/build_index.py --model clip-vit-l14 --videos 1-5
    python scripts/build_index.py --model siglip-so400m-384 --videos 1-5 --bench

--bench times a few batch sizes on 256 frames first and uses the fastest (MPS memory
and speed are found empirically, one model loaded at a time).
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
from fetch_cholec80_videos import parse_videos, refuse_test_videos  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cholec80"


def video_tag(videos: list[int]) -> str:
    return f"videos_{videos[0]}-{videos[-1]}" if videos == list(
        range(videos[0], videos[-1] + 1)) else "videos_" + "_".join(map(str, videos))


def bench(backbone, paths, sizes=(8, 16, 32, 64)) -> tuple[int, dict]:
    sample = paths[:256]
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
    ap.add_argument("--videos", required=True)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto",
                    help="compute dtype; auto = registry default (fp16 iff cuda)")
    args = ap.parse_args()

    videos = parse_videos(args.videos)
    refuse_test_videos(videos)
    manifest = pd.read_parquet(DATA / "manifest.parquet")
    missing = sorted(set(videos) - set(manifest.video_id))
    if missing:
        sys.exit(f"videos {missing} are not in the manifest yet; run extract_frames.py")
    manifest = (manifest[manifest.video_id.isin(videos)]
                .sort_values(["video_id", "frame_idx_25fps"]).reset_index(drop=True))
    mhash = manifest_hash(manifest)
    spec = _SPECS[args.model]
    if isinstance(spec, SubprocessSpec):
        sys.exit(f"{args.model} runs in its own environment: "
                 f"{spec.python} {spec.script} index --videos {args.videos}")
    out_dir = ROOT / "data" / "index" / args.model / video_tag(videos)
    if cache_is_valid(out_dir, spec.revision, mhash) and not args.force:
        print(f"cached: {out_dir.relative_to(ROOT)} (same revision and manifest); skipping")
        return

    fp16 = {"auto": None, "float32": False, "float16": True}[args.dtype]
    backbone = get_backbone(args.model, fp16=fp16)
    paths = [DATA / p for p in manifest.path]
    rates = {}
    bs = args.batch_size or spec.default_batch_size
    if args.bench:
        bs, rates = bench(backbone, paths)
        print(f"using batch size {bs}")

    print(f"embedding {len(paths)} frames with {args.model} on {backbone.device.type}")
    emb, seconds = embed_frames(backbone, paths, batch_size=bs)
    meta = {
        "model": args.model, "hf_id": spec.hf_id, "revision": spec.revision,
        "videos": videos, "n_frames": len(emb), "dim": int(emb.shape[1]),
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
