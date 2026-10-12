#!/usr/bin/env python
"""The registered RQ1/RQ2 evaluation on Cholec80 (docs/preregistration.md, section 4).

    python scripts/eval_registered.py --split val      # videos 33-40: dev check, not a result
    python scripts/eval_registered.py --split test     # videos 41-80: only once the scoring
                                                       # lock is lifted (Week 3, Task 5)

All 70 Cholec80 target strings + 7 negatives, for every model, on the same bootstrap
resamples (default_rng(20261011), B = 10,000; descriptive intervals from the first
2,000). Text embeddings are cached in data/text_emb/ (keyed by model revision and the
exact strings). The frame index is data/index/<model>/videos_1-40 (rows of 33-40) for
val and data/index/<model>/videos_41-80 for test.

val -> results/dev/registered_val_<timestamp>/, every file labelled "NOT A RESULT".
test -> results/results.csv (one row per model, setting, split and metric, with seed,
git hash and date) and the full tables in results/test/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from srb.evaluation import compare_models, evaluate_model, summarise
from srb.index import git_hash, load_index
from srb.models.registry import _SPECS, get_backbone
from srb.queries import cholec80_queries
from srb.resample import B_CONFIRMATORY, BOOT_SEED, resample_counts

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("clip-vit-l14", "siglip-so400m-384", "medsiglip-448", "peskavlp")
SPLITS = {"val": ("videos_1-40", list(range(33, 41))),
          "test": ("videos_41-80", list(range(41, 81)))}


def text_embeddings(model: str, texts: list[str]) -> dict[str, np.ndarray]:
    """Encode once per model revision; the cache must hold exactly these strings."""
    rev = _SPECS[model].revision
    path = ROOT / "data" / "text_emb" / f"{model}_{rev[:12]}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["texts"]) == texts and str(z["revision"]) == rev:
            return dict(zip(texts, z["emb"].astype(np.float32)))
    emb = get_backbone(model).encode_text(texts).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, texts=np.array(texts), emb=emb, revision=rev)
    return dict(zip(texts, emb))


def load_pool(model: str, split: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    tag, videos = SPLITS[split]
    emb, manifest, meta = load_index(ROOT / "data" / "index" / model / tag)
    if meta["revision"] != _SPECS[model].revision:
        sys.exit(f"{model}: index revision differs from the registry; rebuild it")
    keep = manifest.video_id.isin(videos).to_numpy()
    if sorted(manifest.video_id[keep].unique()) != videos:
        sys.exit(f"{model}: index {tag} does not hold every video of the {split} split")
    return emb[keep], manifest[keep].reset_index(drop=True), meta


def _evaluate(job) -> tuple:
    """One model in its own process: every model gets the same resamples (fixed seed)."""
    model, split, boot, text = job
    t0 = time.time()
    emb, manifest, meta = load_pool(model, split)
    counts = resample_counts(manifest.video_id.nunique(), boot, BOOT_SEED)
    ev = evaluate_model(model, emb, manifest, text, counts)
    return ev, meta.get("git_hash"), round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--boot", type=int, default=B_CONFIRMATORY,
                    help="resamples (the registered value is 10,000; lower only for a dry run)")
    ap.add_argument("--jobs", type=int, default=4, help="models evaluated in parallel")
    args = ap.parse_args()

    ghash = git_hash(ROOT)
    if args.split == "test" and (ghash.endswith("-dirty") or args.boot != B_CONFIRMATORY):
        sys.exit("a test run needs a clean tree and the registered B = 10,000")
    stamp = dt.datetime.now()
    texts = [q.text for q in cholec80_queries()]
    label = ("DEV, videos 33-40 (val split), NOT A RESULT" if args.split == "val"
             else "RESULT, Cholec80 test videos 41-80")
    out = (ROOT / "results" / "dev" / f"registered_val_{stamp:%Y%m%d-%H%M%S}"
           if args.split == "val" else ROOT / "results" / "test")
    out.mkdir(parents=True, exist_ok=True)

    text = {m: text_embeddings(m, texts) for m in args.models}  # one model at a time
    jobs = [(m, args.split, args.boot, text[m]) for m in args.models]
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        done = list(pool.map(_evaluate, jobs))
    evals, runtime, index_hash = {}, {}, {}
    for model, (ev, ihash, seconds) in zip(args.models, done):
        evals[model], index_hash[model], runtime[model] = ev, ihash, seconds
        for name in ("per_query", "per_target", "classification", "calibration"):
            df = getattr(ev, name).copy()
            df.insert(0, "label", label)
            df.insert(1, "model", model)
            df.to_csv(out / f"{model}_{name}.csv", index=False)
        print(f"{model}: {ev.summary['n_frames']} frames, {ev.summary['n_videos']} videos, "
              f"B = {args.boot}, {seconds} s", flush=True)

    summary = pd.concat([summarise(ev) for ev in evals.values()], ignore_index=True)
    pairs, flips = compare_models(evals)
    for name, df in (("summary", summary), ("pairs", pairs), ("rq2_order_changes", flips)):
        df = df.copy()
        df.insert(0, "label", label)
        df.to_csv(out / f"{name}.csv", index=False)
    run = {"label": label, "split": args.split, "models": args.models, "boot": args.boot,
           "seed": BOOT_SEED, "git_hash": ghash, "index_git_hash": index_hash,
           "date": stamp.isoformat(timespec="seconds"), "runtime_s": runtime}
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    if args.split == "test":
        rows = summary.assign(setting="zero-shot", split="test", seed=BOOT_SEED,
                              git_hash=ghash, date=stamp.date().isoformat())
        diffs = pairs.rename(columns={"pair": "model", "diff": "value"}).assign(
            metric="overall_map_difference", setting="zero-shot", split="test",
            seed=BOOT_SEED, git_hash=ghash, date=stamp.date().isoformat())
        results = ROOT / "results" / "results.csv"
        new = pd.concat([rows, diffs], ignore_index=True)
        if results.exists():
            new = pd.concat([pd.read_csv(results), new], ignore_index=True)
        new.to_csv(results, index=False)

    print(f"\n*** {label} ***")
    keys = ["overall_map", "overall_prevalence", "overall_lift", "phase_map", "tool_map",
            "zs_phase_accuracy", "zs_phase_macro_f1"]
    show = summary[summary.metric.isin(keys)]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(show.pivot(index="metric", columns="model", values="value").loc[keys].to_string())
        if len(pairs):
            print(pairs.to_string(index=False))
    print(f"*** {label} ***\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
