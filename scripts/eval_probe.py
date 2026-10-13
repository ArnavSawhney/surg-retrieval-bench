#!/usr/bin/env python
"""Linear probe on frozen embeddings, RQ3 (docs/preregistration.md, section 4.3).

    python scripts/eval_probe.py --split val    # fit on 1-32, select C on 33-40: dev, not a result
    python scripts/eval_probe.py --split test   # the selected probes, once, on 41-80

--split val, per model (data/index/<model>/videos_1-40):
  * phase head: multinomial logistic regression for every C in the grid, trained on
    1-32, selected by macro-F1 on 33-40; tool heads: one binary per tool, C by AP on 33-40;
  * saves the selected probes to data/probe/<model>/probe.npz and the phase posteriors
    of videos 1-40 to data/probe/<model>/posteriors_videos_1-40.npz (1-32 marked
    in-sample) for RQ4;
  * scores the selected probes on 33-40 -> results/dev/probe_val_<timestamp>/, every file
    labelled "NOT A RESULT".

--split test (clean tree only, once): loads the probes selected by the val run (no
refit, no reselection), scores them on 41-80 (data/index/<model>/videos_41-80), saves
posteriors_videos_41-80.npz, writes the tables to results/test/probe_*.csv and appends
rows with setting=linear-probe to results/results.csv.

Intervals: paired video-level bootstrap, the first 2,000 resamples of
default_rng(20261011), the same resamples for every model; descriptive (section 4.6).
The gain over zero-shot uses the r1-r5 zero-shot phase classifiers of section 4.2,
recomputed from the cached text embeddings for the paired resamples; their accuracies
must equal the registered run's classification table exactly, or the script stops.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from srb.datasets.cholec80 import PHASES
from srb.datasets.overlap import CHOLEC80_TRAIN_IDS
from srb.index import git_hash, load_index
from srb.models.registry import _SPECS
from srb.probe import (C_GRID, MAX_ITER, evaluate_probe, load_probes, phase_labels,
                       save_probes, select_phase_probe, select_tool_probes, softmax,
                       summarise_probes, zero_shot_phase_predictions)
from srb.queries import cholec80_queries
from srb.resample import B_DESCRIPTIVE, BOOT_SEED, resample_counts

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("clip-vit-l14", "siglip-so400m-384", "medsiglip-448", "peskavlp")
INDEX = {"val": "videos_1-40", "test": "videos_41-80"}
EVAL_VIDEOS = {"val": list(range(33, 41)), "test": list(range(41, 81))}
LABEL = {"val": "DEV, videos 33-40 (val split), probe trained on 1-32, NOT A RESULT",
         "test": "RESULT, Cholec80 test videos 41-80"}
REGISTERED_CLS = {"val": "results/dev/registered_val_20261012-220609",
                  "test": "results/test"}


def probe_dir(model: str) -> Path:
    return ROOT / "data" / "probe" / model


def text_embeddings(model: str) -> dict[str, np.ndarray]:
    """The cached text embeddings the registered run used (data/text_emb/)."""
    rev = _SPECS[model].revision
    path = ROOT / "data" / "text_emb" / f"{model}_{rev[:12]}.npz"
    texts = [q.text for q in cholec80_queries()]
    z = np.load(path, allow_pickle=False)
    if list(z["texts"]) != texts or str(z["revision"]) != rev:
        raise SystemExit(f"{path}: cached strings or revision differ from the registry")
    return dict(zip(texts, z["emb"].astype(np.float32)))


def check_zero_shot(model: str, split: str, zs_pred: dict, manifest: pd.DataFrame) -> None:
    """The recomputed r1-r5 accuracies must equal the registered classification table."""
    ref = pd.read_csv(ROOT / REGISTERED_CLS[split] / f"{model}_classification.csv")
    ref = ref.set_index("classifier")["accuracy"]
    truth = phase_labels(manifest)
    for name, pred in zs_pred.items():
        mine = float((pred == truth).mean())
        if abs(mine - ref[name]) > 1e-12:
            raise SystemExit(f"{model} {name}: zero-shot accuracy {mine} differs from the "
                             f"registered {ref[name]}")


def load_pool(model: str, tag: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    emb, manifest, meta = load_index(ROOT / "data" / "index" / model / tag)
    if meta["revision"] != _SPECS[model].revision:
        raise SystemExit(f"{model}: index revision differs from the registry")
    return emb, manifest, meta


def save_posteriors(model: str, tag: str, post: np.ndarray, manifest: pd.DataFrame,
                    phase_C: float, meta: dict) -> Path:
    path = probe_dir(model) / f"posteriors_{tag}.npz"
    np.savez(path, posterior=post.astype(np.float32),
             video_id=manifest["video_id"].to_numpy(),
             frame_idx_25fps=manifest["frame_idx_25fps"].to_numpy(),
             in_sample=manifest["video_id"].isin(sorted(CHOLEC80_TRAIN_IDS)).to_numpy(),
             phases=np.array(PHASES),
             phase_C=phase_C, meta=np.array(json.dumps(meta)))
    return path


def _run(job) -> dict:
    """One model in its own process."""
    model, split, ghash = job
    t0 = time.time()
    emb, manifest, imeta = load_pool(model, INDEX[split])
    info = {"index_git_hash": imeta.get("git_hash"), "index_manifest_sha256": imeta["manifest_sha256"]}
    if split == "val":
        phase, pinfos = select_phase_probe(emb, manifest)
        tools, tinfos = select_tool_probes(emb, manifest)
        fits = pd.DataFrame([vars(i) for i in pinfos + tinfos])
        pmeta = {"model": model, "revision": imeta["revision"], "index": INDEX[split],
                 "index_manifest_sha256": imeta["manifest_sha256"], "git_hash": ghash,
                 "created": dt.datetime.now().isoformat(timespec="seconds"),
                 "C_grid": list(C_GRID), "max_iter": MAX_ITER,
                 "train_videos": "1-32", "selection_videos": "33-40"}
        save_probes(probe_dir(model) / "probe.npz", phase, tools, pmeta)
    else:
        phase, tools, pmeta = load_probes(probe_dir(model) / "probe.npz")
        if pmeta["revision"] != imeta["revision"]:
            raise SystemExit(f"{model}: probe and index revisions differ")
        fits = None
    info["probe_git_hash"] = pmeta["git_hash"]

    post_all = softmax(phase.decision(emb))
    ppath = save_posteriors(model, INDEX[split], post_all, manifest, phase.C,
                            {"probe_git_hash": pmeta["git_hash"], "run_git_hash": ghash,
                             "index_manifest_sha256": imeta["manifest_sha256"]})
    info["posteriors"] = str(ppath.relative_to(ROOT))

    keep = manifest["video_id"].isin(EVAL_VIDEOS[split]).to_numpy()
    X, man = emb[keep], manifest[keep].reset_index(drop=True)
    if sorted(man["video_id"].unique()) != EVAL_VIDEOS[split]:
        raise SystemExit(f"{model}: index does not hold every {split} video")
    zs = zero_shot_phase_predictions(X, text_embeddings(model))
    check_zero_shot(model, split, zs, man)
    counts = resample_counts(man["video_id"].nunique(), B_DESCRIPTIVE, BOOT_SEED)
    ev = evaluate_probe(model, phase, tools, X, man, zs, counts)
    ev.posteriors = None  # saved above; not sent back to the parent
    info["seconds"] = round(time.time() - t0, 1)
    info["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
    return {"model": model, "eval": ev, "fits": fits, "info": info}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=sorted(INDEX))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--jobs", type=int, default=2, help="models in parallel (8 GB Mac: 2)")
    args = ap.parse_args()

    ghash = git_hash(ROOT)
    results = ROOT / "results" / "results.csv"
    if args.split == "test":
        if ghash.endswith("-dirty"):
            sys.exit("the probe test run needs a clean tree")
        if list(args.models) != list(MODELS):
            sys.exit("the probe test run covers all four models at once")
        prior = pd.read_csv(results)
        if (prior["setting"] == "linear-probe").any():
            sys.exit("results.csv already holds linear-probe rows: the test run happens once")
        for m in args.models:
            if not (probe_dir(m) / "probe.npz").exists():
                sys.exit(f"{m}: no selected probe; run --split val first")
    stamp = dt.datetime.now()
    label = LABEL[args.split]
    out = (ROOT / "results" / "dev" / f"probe_val_{stamp:%Y%m%d-%H%M%S}"
           if args.split == "val" else ROOT / "results" / "test")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    jobs = [(m, args.split, ghash) for m in args.models]
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        done = list(pool.map(_run, jobs))
    evals = {d["model"]: d["eval"] for d in done}
    infos = {d["model"]: d["info"] for d in done}

    summary, pairs = summarise_probes(evals)
    per_tool = pd.concat([ev.per_tool.assign(model=m) for m, ev in evals.items()],
                         ignore_index=True)
    tables = {"probe_summary": summary, "probe_pairs": pairs, "probe_per_tool": per_tool}
    if args.split == "val":
        tables["probe_selection"] = pd.concat(
            [d["fits"].assign(model=d["model"]) for d in done], ignore_index=True)
    for name, df in tables.items():
        df = df.copy()
        if "model" in df and df.columns[0] != "model":
            df.insert(0, "model", df.pop("model"))
        df.insert(0, "label", label)
        df.to_csv(out / f"{name}.csv", index=False)
    run = {"label": label, "split": args.split, "models": args.models,
           "boot": B_DESCRIPTIVE, "seed": BOOT_SEED, "git_hash": ghash,
           "date": stamp.isoformat(timespec="seconds"), "wall_s": round(time.time() - t0, 1),
           "jobs": args.jobs, "C_grid": list(C_GRID), "max_iter": MAX_ITER, "per_model": infos}
    (out / ("run.json" if args.split == "val" else "probe_run.json")).write_text(
        json.dumps(run, indent=2) + "\n")

    if args.split == "test":
        date = stamp.date().isoformat()
        common = dict(setting="linear-probe", split="test", seed=BOOT_SEED, git_hash=ghash,
                      date=date)
        rows = summary[summary["metric"] != "zs_phase_accuracy"].copy()
        rows["metric"] = "probe_" + rows["metric"]
        diffs = pairs.rename(columns={"pair": "model", "diff": "value"}).copy()
        diffs["metric"] = "probe_" + diffs["metric"] + "_difference"
        new = pd.concat([rows.assign(**common), diffs.assign(**common)], ignore_index=True)
        pd.concat([pd.read_csv(results), new], ignore_index=True).to_csv(results, index=False)

    print(f"\n*** {label} ***")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary.pivot(index="metric", columns="model", values="value").to_string())
        print(pairs.to_string(index=False))
        if args.split == "val":
            sel = tables["probe_selection"]
            print("\nfits not converged at max_iter:",
                  sel[~sel.converged][["model", "head", "C", "n_iter"]].to_string(index=False)
                  if (~sel.converged).any() else "none")
    for m, i in infos.items():
        print(f"{m}: {i['seconds']} s, peak RSS {i['peak_rss_mb']} MB, posteriors {i['posteriors']}")
    print(f"*** {label} ***\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
