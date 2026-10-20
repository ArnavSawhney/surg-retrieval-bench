#!/usr/bin/env python
"""Endoscapes2023 CVS retrieval on the official test split (sections 1.2, 3.3, 4.1, 4.6;
Deviations, 2026-10-20). Descriptive only: section 5 makes no CVS prediction.

    python scripts/eval_endoscapes.py --check-only   # hashes, counts, R, query strings; no score
    python scripts/eval_endoscapes.py                # the single test run (after Arnav's go)
    python scripts/eval_endoscapes.py --root <scratch>   # dry run on a synthetic root

Pool: the 1,799 CVS keyframes of test videos 162-201 (``data/endoscapes/manifest.parquet``),
scored with the 15 CVS strings + 7 negatives of ``srb.queries`` through the same
``srb.evaluation.evaluate_model`` as Cholec80 (``queries=endoscapes_queries()``).
Relevance: majority vote (registered); a second pass with unanimity (3 of 3) is the
secondary analysis "unanimity, not registered". Intervals: the first 2,000 resamples of
``default_rng(20261011)`` over the 40 videos sorted by ID (the same index matrix as
Cholec80's, since n = 40), every model on the same resamples; pairwise CVS mAP
differences get intervals, no p-values.

Outputs (real root): ``results/test/endoscapes_*.csv`` and ``endoscapes_run.json``; rows
appended to ``results/results.csv`` with ``split = endoscapes-test`` (never re-written,
trap 16). The run refuses if ``endoscapes_run.json`` exists (it runs once), on a dirty
tree, or if any check fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_registered import MODELS, ROOT, text_embeddings  # noqa: E402

from srb.datasets.endoscapes import (RELEVANT_COLUMNS, UNANIMOUS_COLUMNS,  # noqa: E402
                                     build_manifest, cholec80_overlap)
from srb.evaluation import compare_models, evaluate_model, summarise  # noqa: E402
from srb.index import git_hash, load_index, manifest_hash  # noqa: E402
from srb.models.registry import _SPECS  # noqa: E402
from srb.queries import CVS_QUERIES, NEGATIVE_QUERIES, endoscapes_queries  # noqa: E402
from srb.resample import B_DESCRIPTIVE, BOOT_SEED, resample_counts  # noqa: E402

SPLIT = "endoscapes-test"
INDEX_TAG = "endoscapes_test"
TEXT_SUFFIX = "_endoscapes"
SETTING = "zero-shot"
UNANIMITY_SETTING = "zero-shot-unanimity-not-registered"
# Label counts fixed in the Deviations entry of 2026-10-20 (before any image was fetched).
EXPECTED = {"n_frames": 1799, "video_ids": list(range(162, 202)),
            "R": {"c1": 423, "c2": 302, "c3": 501},
            "R_unanimous": {"c1": 77, "c2": 138, "c3": 222}}
NORM_TOL = 2e-3  # float16 storage of unit vectors


def fail(msg: str) -> None:
    sys.exit(f"CHECK FAILED: {msg}")


def check(root: Path, models: list[str]) -> dict:
    """Every pre-score check; exits on the first failure. Returns what was verified."""
    data = root / "data" / "endoscapes"
    manifest = pd.read_parquet(data / "manifest.parquet")
    rebuilt = build_manifest(data / "labels", "test")  # re-applies the relevance asserts
    if not manifest.equals(rebuilt):
        fail("manifest.parquet differs from the manifest rebuilt from the label files")
    mhash = manifest_hash(manifest)
    ids = sorted(int(v) for v in manifest.video_id.unique())
    facts = {"manifest_sha256": mhash, "n_frames": len(manifest), "video_ids": ids,
             "R": {c: int(manifest[c].sum()) for c in RELEVANT_COLUMNS},
             "R_unanimous": {c.removesuffix("_unanimous"): int(manifest[c].sum())
                             for c in UNANIMOUS_COLUMNS}}
    for key in EXPECTED:
        if facts[key] != EXPECTED[key]:
            fail(f"{key} = {facts[key]}, expected {EXPECTED[key]} (Deviations, 2026-10-20)")
    missing = [p for p in manifest.path if not (data / p).exists()]
    if missing:
        fail(f"{len(missing)} frame files missing, e.g. {missing[0]}")
    if cholec80_overlap(ids):
        fail(f"test videos that are Cholec80 videos: {cholec80_overlap(ids)}")

    # every string comes from srb.queries, in the registered order
    registered = [t for crit in CVS_QUERIES.values() for t in crit] + list(NEGATIVE_QUERIES)
    for unanimous in (False, True):
        qs = endoscapes_queries(unanimous)
        if [q.text for q in qs] != registered:
            fail("query strings differ from srb.queries (CVS_QUERIES + NEGATIVE_QUERIES)")
        if len([q for q in qs if q.kind == "cvs"]) != 15:
            fail("expected 15 CVS strings (3 criteria x 5 rephrasings)")
    facts["n_query_strings"] = len(registered)

    facts["indexes"] = {}
    for m in models:
        emb, man, meta = load_index(root / "data" / "index" / m / INDEX_TAG)  # hash-checked
        if meta["revision"] != _SPECS[m].revision:
            fail(f"{m}: index revision {meta['revision'][:12]} != registry")
        if meta.get("dataset") != "endoscapes" or meta["manifest_sha256"] != mhash:
            fail(f"{m}: index is not of this Endoscapes manifest")
        norms = np.linalg.norm(emb, axis=1)
        if np.abs(norms - 1).max() > NORM_TOL:
            fail(f"{m}: embedding norms {norms.min():.4f}-{norms.max():.4f}")
        facts["indexes"][m] = {"n": len(emb), "dim": int(emb.shape[1]),
                               "device": meta.get("device"),
                               "compute_dtype": meta.get("compute_dtype"),
                               "norm_min": round(float(norms.min()), 5),
                               "norm_max": round(float(norms.max()), 5),
                               "git_hash": meta.get("git_hash")}
    return facts


def evaluate_all(root: Path, models: list[str], counts: np.ndarray, unanimous: bool):
    manifest = pd.read_parquet(root / "data" / "endoscapes" / "manifest.parquet")
    queries = endoscapes_queries(unanimous)
    texts = [q.text for q in queries]
    evals = {}
    for m in models:
        emb, _, _ = load_index(root / "data" / "index" / m / INDEX_TAG)
        text = text_embeddings(m, texts, root=root, suffix=TEXT_SUFFIX)
        evals[m] = evaluate_model(m, emb, manifest, text, counts, queries=queries)
    return evals


def write_tables(evals: dict, out: Path, prefix: str, label: str) -> pd.DataFrame:
    for m, ev in evals.items():
        for name in ("per_query", "per_target", "calibration"):
            df = getattr(ev, name).copy()
            df.insert(0, "label", label)
            df.insert(1, "model", m)
            df.to_csv(out / f"{prefix}_{m}_{name}.csv", index=False)
    summary = pd.concat([summarise(ev) for ev in evals.values()], ignore_index=True)
    pairs, flips = compare_models(evals, metric="cvs_map", confirmatory=False)
    flips["coarse_note"] = "3 targets: the fraction can only be 0, 1/3, 2/3 or 1"
    for name, df in (("summary", summary), ("pairs", pairs), ("rq2_order_changes", flips)):
        df = df.copy()
        df.insert(0, "label", label)
        df.to_csv(out / f"{prefix}_{name}.csv", index=False)
    return summary, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="a synthetic scratch root for a dry run (never the real index)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = ap.parse_args()
    root = args.root.resolve()
    real = root == ROOT
    ghash = git_hash(ROOT)

    facts = check(root, args.models)
    print(json.dumps({k: v for k, v in facts.items() if k != "video_ids"}, indent=2))
    print(f"video IDs (bootstrap order): {facts['video_ids'][0]}..{facts['video_ids'][-1]}, "
          f"{len(facts['video_ids'])} videos")
    out = root / "results" / "test"
    run_file = out / "endoscapes_run.json"
    if real and run_file.exists():
        sys.exit(f"{run_file.relative_to(ROOT)} exists: the Endoscapes test run happens once")
    if real and ghash.endswith("-dirty"):
        print("NOTE: the tree is dirty; a real run would refuse")
    if args.check_only:
        print("\nCHECK ONLY: all checks passed; stopped before any score.")
        return
    if real and ghash.endswith("-dirty"):
        sys.exit("the Endoscapes test run needs a clean tree")

    t0 = time.time()
    stamp = dt.datetime.now()
    label = ("RESULT, Endoscapes2023 test (CVS keyframes, descriptive)" if real
             else "DRY RUN on a synthetic root, NOT A RESULT")
    ulabel = label + "; UNANIMITY, NOT REGISTERED"
    out.mkdir(parents=True, exist_ok=True)
    counts = resample_counts(len(facts["video_ids"]), B_DESCRIPTIVE, BOOT_SEED)

    evals = evaluate_all(root, args.models, counts, unanimous=False)
    summary, pairs = write_tables(evals, out, "endoscapes", label)
    uevals = evaluate_all(root, args.models, counts, unanimous=True)
    usummary, upairs = write_tables(uevals, out, "endoscapes_unanimity", ulabel)

    common = {"split": SPLIT, "seed": BOOT_SEED, "git_hash": ghash,
              "date": stamp.date().isoformat()}
    diff = lambda p: p.rename(columns={"pair": "model", "diff": "value"}).assign(  # noqa: E731
        metric="cvs_map_difference")
    new = pd.concat([summary.assign(setting=SETTING, **common),
                     diff(pairs).assign(setting=SETTING, **common),
                     usummary.assign(setting=UNANIMITY_SETTING, **common),
                     diff(upairs).assign(setting=UNANIMITY_SETTING, **common)],
                    ignore_index=True)
    results = root / "results" / "results.csv"
    header = list(pd.read_csv(results, nrows=0).columns)
    extra = set(new.columns) - set(header)
    if extra:
        sys.exit(f"columns not in results.csv: {sorted(extra)}")
    new.reindex(columns=header).to_csv(results, mode="a", header=False, index=False)

    run = {"label": label, "split": SPLIT, "models": args.models, "boot": B_DESCRIPTIVE,
           "seed": BOOT_SEED, "bootstrap_video_order": facts["video_ids"],
           "bootstrap_note": "resample_counts(40, 2000, 20261011): the same index matrix "
                             "as Cholec80's first 2,000 resamples, mapped to these IDs",
           "git_hash": ghash, "manifest_sha256": facts["manifest_sha256"],
           "indexes": facts["indexes"], "rows_appended": len(new),
           "date": stamp.isoformat(timespec="seconds"),
           "runtime_s": round(time.time() - t0, 1)}
    run_file.write_text(json.dumps(run, indent=2) + "\n")

    print(f"\n*** {label} ***")
    keys = ["cvs_map", "cvs_prevalence", "cvs_lift", "cvs_r_prec", "cvs_per_video_ap",
            "cvs_ensemble_map", "rq2_mean_sd"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for name, s in (("majority (registered)", summary), ("unanimity, not registered",
                                                             usummary)):
            print(f"\n{name}")
            print(s[s.metric.isin(keys)][["model", "metric", "value", "ci_low", "ci_high"]]
                  .to_string(index=False))
    print(f"*** {label} ***\nwrote {out}; {len(new)} rows appended to {results}")


if __name__ == "__main__":
    main()
