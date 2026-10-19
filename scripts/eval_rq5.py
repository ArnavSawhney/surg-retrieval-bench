#!/usr/bin/env python
"""RQ5 failure modes: per-tag phase accuracy on the tagged pool frames (section 4.5).

    python scripts/eval_rq5.py --check-only   # inputs and reproduction checks, no per-tag number
    python scripts/eval_rq5.py                # the analysis, once, clean tree

Reads only the three frozen tag files in data/rq5 (tags.csv, out_of_body.csv,
tags_corrected.csv), which must hash to the SHA-256s in LOG.md (2026-10-19), and uses
the corrected tags (Deviations 2026-10-18). Per model (data/index/<model>/videos_41-80):

  * zero-shot r1-r5 phase predictions recomputed from the cached embeddings, and the
    linear-probe predictions from the saved test posteriors. Before any per-tag number,
    both must reproduce the registered overall accuracies on all frames of 41-80
    (results/test/<model>_classification.csv, results/test/probe_summary.csv) to 1e-12,
    or the script stops;
  * per tag (smoke, blur, glare, blood, occlusion, clean): n frames, zero-shot phase
    accuracy (headline: mean over r1-r5), linear-probe phase accuracy, and each
    artefact's drop = accuracy on clean frames - accuracy on the tag's frames (positive =
    worse; Deviations 2026-10-19). A frame with several tags counts in each. Fewer than
    20 frames: counts only.
  * the drop averaged over the four backbones per artefact (RQ5a, RQ5b), each model's
    mean drop over the five artefacts (RQ5c) and PeskaVLP's paired difference to CLIP
    and SigLIP; blur's drop minus each other artefact's.

Intervals: paired video-level bootstrap over the videos the tagged frames come from,
the first 2,000 resamples of default_rng(20261011), the same resamples for every model,
tag and setting; descriptive (section 4.6).

Settings: "rq5" (registered: out-of-body frames are clean, or blur only) and
"rq5-oob-excluded-not-registered" (Deviations 2026-10-18 (iii): every metric again with
the out-of-body frames dropped; part of no verdict). Writes results/test/rq5_*.csv and
appends rows to results/results.csv (append only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_probe import (MODELS, ROOT, check_zero_shot, load_pool,  # noqa: E402
                        text_embeddings)

from srb.index import git_hash  # noqa: E402
from srb.probe import phase_labels, zero_shot_phase_predictions  # noqa: E402
from srb.resample import B_DESCRIPTIVE, BOOT_SEED, percentile_ci, resample_counts  # noqa: E402
from srb.rq5 import (ARTEFACTS, FROZEN_SHA256, POOL_SHA256, TAG_COLUMNS, TAGS,  # noqa: E402
                     apply_corrections, check_frozen, drops, mean_drop, per_tag_accuracy,
                     stopping_status)

RQ5_DIR = ROOT / "data" / "rq5"
POOL = ROOT / "docs" / "rq5_frame_pool.csv"
PROBE_SUMMARY = ROOT / "results" / "test" / "probe_summary.csv"
SETTINGS = {"rq5": False, "rq5-oob-excluded-not-registered": True}
LABEL = {"rq5": "RESULT, RQ5 tagged pool frames (Cholec80 test 41-80)",
         "rq5-oob-excluded-not-registered":
             "RESULT, RQ5 tagged pool frames, OUT OF BODY EXCLUDED, NOT REGISTERED"}
METRICS = {"probe": "probe_accuracy", "zs": "zs_accuracy"}
MEAN = "mean(4 models)"
PESKA, VERSUS = "peskavlp", ("clip-vit-l14", "siglip-so400m-384")


def load_tags(rq5_dir: Path = RQ5_DIR, pool_path: Path = POOL,
              expected: dict = FROZEN_SHA256, pool_sha: str = POOL_SHA256) -> pd.DataFrame:
    """The corrected tags, after checking the frozen hashes, the pool, and that the
    corrected file is exactly raw tags + out-of-body answers."""
    try:
        check_frozen(rq5_dir, expected)
    except ValueError as e:
        raise SystemExit(f"{rq5_dir}: {e}") from None
    if hashlib.sha256(pool_path.read_bytes()).hexdigest() != pool_sha:
        raise SystemExit(f"{pool_path}: SHA-256 differs from the registered pool")
    raw = pd.read_csv(rq5_dir / "tags.csv")
    oob = pd.read_csv(rq5_dir / "out_of_body.csv")
    corr = pd.read_csv(rq5_dir / "tags_corrected.csv")
    want = apply_corrections(raw, oob)
    cols = TAG_COLUMNS + ["out_of_body"]
    if list(corr.columns) != cols or not all(
            np.array_equal(corr[c].to_numpy(), want[c].to_numpy()) for c in cols):
        raise SystemExit("tags_corrected.csv is not raw tags + out-of-body answers")
    pool = pd.read_csv(pool_path).set_index("tag_order")
    ref = pool.loc[corr["tag_order"]]
    if not (np.array_equal(ref["video_id"], corr["video_id"])
            and np.array_equal(ref["frame_idx_25fps"], corr["frame_idx_25fps"])):
        raise SystemExit("tagged frames do not match the pool")
    if not stopping_status(corr)["done"]:
        raise SystemExit("tagging is not complete: the stopping rule is not met")
    return corr


def model_correct(model: str, tagged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """(N, 5) zero-shot r1-r5 and (N,) probe correctness on the tagged frames, after the
    overall reproduction checks on all frames of 41-80."""
    emb, manifest, meta = load_pool(model, "videos_41-80")
    truth = phase_labels(manifest)
    zs = zero_shot_phase_predictions(emb, text_embeddings(model))
    check_zero_shot(model, "test", zs, manifest)  # stops unless equal to 1e-12
    z = np.load(ROOT / "data" / "probe" / model / "posteriors_videos_41-80.npz",
                allow_pickle=False)
    if not (np.array_equal(z["video_id"], manifest["video_id"].to_numpy())
            and np.array_equal(z["frame_idx_25fps"], manifest["frame_idx_25fps"].to_numpy())):
        raise SystemExit(f"{model}: posteriors are not row-aligned with the index")
    pred = z["posterior"].argmax(1)
    ref = pd.read_csv(PROBE_SUMMARY)
    ref = float(ref[(ref.model == model) & (ref.metric == "phase_accuracy")]["value"].iloc[0])
    acc = float((pred == truth).mean())
    if abs(acc - ref) > 1e-12:
        raise SystemExit(f"{model}: probe accuracy {acc} differs from the registered {ref}")
    key = pd.MultiIndex.from_arrays([manifest["video_id"], manifest["frame_idx_25fps"]])
    idx = key.get_indexer(pd.MultiIndex.from_arrays([tagged["video_id"],
                                                     tagged["frame_idx_25fps"]]))
    if (idx < 0).any():
        raise SystemExit(f"{model}: {(idx < 0).sum()} tagged frames are not in the index")
    zs_c = np.stack([zs[f"r{r}"][idx] == truth[idx] for r in range(1, 6)], 1)
    info = {"zs_accuracy_check": {k: float((v == truth).mean()) for k, v in sorted(zs.items())},
            "probe_accuracy_check": acc, "index_manifest_sha256": meta["manifest_sha256"],
            "revision": meta["revision"], "n_frames_41_80": len(manifest)}
    return zs_c, pred[idx] == truth[idx], info


def _ci(boot: np.ndarray) -> dict:
    if np.isnan(boot).all():
        return {"ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}
    lo, hi = percentile_ci(boot)
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot": int((~np.isnan(boot)).sum())}


def analyse(tagged: pd.DataFrame, correct: dict, counts: np.ndarray,
            videos: np.ndarray) -> dict[str, pd.DataFrame]:
    """All tables for one setting. ``correct[model] = {"probe": (N,), "zs": (N, 5)}``."""
    per_tag, mdrop, acc_all, drop_all = [], [], {}, {}
    for model, cm in correct.items():
        for key, metric in METRICS.items():
            acc = per_tag_accuracy(tagged, cm[key], videos, counts)
            d = drops(acc)
            acc_all[model, key], drop_all[model, key] = acc, d
            for t in TAGS:
                row = {"model": model, "tag": t, "n_frames": acc[t]["n"],
                       "n_videos": acc[t]["n_videos"], "metric": metric,
                       "value": acc[t]["value"], **_ci(acc[t]["boot"])}
                if t in ARTEFACTS:
                    dc = _ci(d[t]["boot"])
                    row |= {"drop": d[t]["value"], "drop_ci_low": dc["ci_low"],
                            "drop_ci_high": dc["ci_high"], "drop_n_boot": dc["n_boot"]}
                per_tag.append(row)
            md = mean_drop(d)
            mdrop.append({"model": model, "metric": metric, "mean_drop": md["value"],
                          **_ci(md["boot"])})
    models = list(correct)
    bmean, pairs = [], []
    for key, metric in METRICS.items():
        avg = {a: {"value": float(np.mean([drop_all[m, key][a]["value"] for m in models])),
                   "boot": np.mean([drop_all[m, key][a]["boot"] for m in models], 0)}
               for a in ARTEFACTS}
        for a in ARTEFACTS:
            bmean.append({"model": MEAN, "tag": a, "metric": metric,
                          "drop": avg[a]["value"], **_ci(avg[a]["boot"])})
        for a in ARTEFACTS:
            if a != "blur":
                pairs.append({"model": MEAN, "pair": f"blur - {a}", "metric": metric,
                              "quantity": "drop", "diff": avg["blur"]["value"] - avg[a]["value"],
                              **_ci(avg["blur"]["boot"] - avg[a]["boot"])})
        if PESKA in models:
            mp = mean_drop(drop_all[PESKA, key])
            for other in VERSUS:
                if other in models:
                    mo = mean_drop(drop_all[other, key])
                    pairs.append({"model": f"{PESKA} - {other}", "pair": f"{PESKA} - {other}",
                                  "metric": metric, "quantity": "mean_drop",
                                  "diff": mp["value"] - mo["value"],
                                  **_ci(mp["boot"] - mo["boot"])})
    return {"per_tag": pd.DataFrame(per_tag), "mean_drop": pd.DataFrame(mdrop),
            "backbone_mean": pd.DataFrame(bmean), "pairs": pd.DataFrame(pairs)}


def results_rows(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One results.csv row per (model, metric)."""
    rows = []
    for r in t["per_tag"].itertuples():
        if r.metric == METRICS["probe"]:
            rows.append({"model": r.model, "metric": f"rq5_{r.tag}_n_frames",
                         "value": r.n_frames})
        rows.append({"model": r.model, "metric": f"rq5_{r.tag}_{r.metric}", "value": r.value,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "n_boot": r.n_boot})
        if r.tag in ARTEFACTS:
            rows.append({"model": r.model, "metric": f"rq5_{r.tag}_{r.metric}_drop",
                         "value": r.drop, "ci_low": r.drop_ci_low,
                         "ci_high": r.drop_ci_high, "n_boot": r.drop_n_boot})
    for r in t["mean_drop"].itertuples():
        rows.append({"model": r.model, "metric": f"rq5_{r.metric}_mean_drop",
                     "value": r.mean_drop, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "n_boot": r.n_boot})
    for r in t["backbone_mean"].itertuples():
        rows.append({"model": r.model, "metric": f"rq5_{r.tag}_{r.metric}_drop",
                     "value": r.drop, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "n_boot": r.n_boot})
    for r in t["pairs"].itertuples():
        name = (f"rq5_{r.metric}_drop_difference_{r.pair.replace(' - ', '_minus_')}"
                if r.quantity == "drop" else f"rq5_{r.metric}_mean_drop_difference")
        rows.append({"model": r.model, "metric": name, "value": r.diff, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "n_boot": r.n_boot})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true",
                    help="inputs and reproduction checks only; no per-tag number")
    args = ap.parse_args()
    ghash = git_hash(ROOT)
    results = ROOT / "results" / "results.csv"
    if not args.check_only:
        if ghash.endswith("-dirty"):
            sys.exit("the RQ5 analysis needs a clean tree")
        if pd.read_csv(results, usecols=["setting"])["setting"].isin(list(SETTINGS)).any():
            sys.exit("results.csv already holds RQ5 rows: the analysis runs once")
    t0, stamp = time.time(), dt.datetime.now()
    tagged = load_tags()
    print(f"tags: {len(tagged)} frames, {int(tagged['out_of_body'].sum())} out of body, "
          f"{tagged['video_id'].nunique()} videos; frozen hashes OK", flush=True)

    correct, infos = {}, {}
    for m in MODELS:
        zs_c, pr_c, infos[m] = model_correct(m, tagged)
        correct[m] = {"zs": zs_c, "probe": pr_c}
        print(f"{m}: overall zero-shot r1-r5 and probe accuracies reproduce the registered "
              f"run (1e-12); {len(pr_c)} tagged frames found", flush=True)
    if args.check_only:
        print("check only: stopping before any per-tag number")
        return

    videos = np.sort(tagged["video_id"].unique())
    counts = resample_counts(len(videos), B_DESCRIPTIVE, BOOT_SEED)
    out = ROOT / "results" / "test"
    tables, new = {}, []
    for setting, exclude in SETTINGS.items():
        keep = (tagged["out_of_body"] == 0).to_numpy() if exclude else np.ones(len(tagged), bool)
        sub = {m: {k: v[keep] for k, v in c.items()} for m, c in correct.items()}
        t = analyse(tagged[keep].reset_index(drop=True), sub, counts, videos)
        for name, df in t.items():
            tables.setdefault(name, []).append(
                df.assign(setting=setting).assign(label=LABEL[setting]))
        new.append(results_rows(t).assign(setting=setting))
    for name, parts in tables.items():
        df = pd.concat(parts, ignore_index=True)
        df = df[["label", "setting"] + [c for c in df.columns if c not in ("label", "setting")]]
        df.to_csv(out / f"rq5_{name}.csv", index=False)
    rows = pd.concat(new, ignore_index=True).assign(
        split="test", seed=BOOT_SEED, git_hash=ghash, date=stamp.date().isoformat())
    header = list(pd.read_csv(results, nrows=0).columns)
    before = results.read_bytes()
    # append only: re-writing would re-format the registered rows' floats
    rows.reindex(columns=header).to_csv(results, mode="a", header=False, index=False)
    if not results.read_bytes().startswith(before):
        raise SystemExit("results.csv prefix changed")
    run = {"label": LABEL, "git_hash": ghash, "date": stamp.isoformat(timespec="seconds"),
           "boot": B_DESCRIPTIVE, "seed": BOOT_SEED, "videos": [int(v) for v in videos],
           "n_tagged": len(tagged), "n_out_of_body": int(tagged["out_of_body"].sum()),
           "frozen_sha256": FROZEN_SHA256, "rows_appended": len(rows),
           "wall_s": round(time.time() - t0, 1), "per_model": infos}
    (out / "rq5_run.json").write_text(json.dumps(run, indent=2) + "\n")

    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format,
                           "display.max_rows", 500):
        for setting in SETTINGS:
            print(f"\n*** {LABEL[setting]} ***")
            for name, parts in tables.items():
                df = pd.concat(parts, ignore_index=True)
                print(df[df.setting == setting].drop(columns=["label", "setting"])
                      .to_string(index=False))
    print(f"\nappended {len(rows)} rows to results.csv; tables results/test/rq5_*.csv")


if __name__ == "__main__":
    main()
