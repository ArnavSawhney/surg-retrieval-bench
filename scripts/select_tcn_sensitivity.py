#!/usr/bin/env python
"""Sensitivity analysis of the causal TCN: SENSITIVITY, NOT REGISTERED.

    python scripts/select_tcn_sensitivity.py    # clean tree; after eval_temporal.py --split val

Deviations, 2026-10-13: on validation, 10 of the 12 (backbone, seed) TCN selections chose
depth 4, the largest registered depth (receptive field 31 s). Here each seed's TCN is
re-selected over depths 2-8 (receptive fields 7-511 s) and up to 100 epochs, everything
else as section 4.4 and its clarification: same seeds, same training videos 1-32, same
joint (depth, epoch) selection by macro-F1 on 33-40, same CPU threads. The registered
analysis is unchanged.

Per model (data/index/<model>/videos_1-40):
  * trains every (seed, depth) for 100 epochs. Training does not depend on the epoch cap,
    so for depths 2-4 the first 50 epochs must reproduce the registered validation
    curves exactly, and a seed whose choice stays inside the registered grid must give
    back the registered TCN bit for bit, or the script stops;
  * saves data/temporal/<model>/tcn_sensitivity_seed*.pt and selection_sensitivity.json,
    which eval_temporal.py --split test then scores on 41-80;
  * scores single frame, the registered TCN and the re-selected TCN on 33-40 ->
    results/dev/tcn_sensitivity_val_<timestamp>/, labelled NOT A RESULT.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_temporal import (MODELS, ROOT, SENS_SELECTION, TCN, TCN_SENS,  # noqa: E402
                           _setup_torch, by_video, check_single_frame, load_inputs,
                           summarise, temporal_dir)

import srb.temporal as tm  # noqa: E402
from srb.datasets.overlap import CHOLEC80_TRAIN_IDS, CHOLEC80_VAL_IDS  # noqa: E402
from srb.index import git_hash  # noqa: E402
from srb.resample import B_DESCRIPTIVE, BOOT_SEED, resample_counts  # noqa: E402

REGISTERED_VAL = ROOT / "results" / "dev" / "temporal_val_20261013-203339"
LABEL = ("DEV, videos 33-40 (val split), TCN SENSITIVITY, NOT REGISTERED "
         "(depths 2-8, up to 100 epochs), NOT A RESULT")


def _run(job) -> dict:
    model, ghash, threads = job
    _setup_torch(threads)
    import torch

    t0 = time.time()
    emb, y, post, vid, imeta = load_inputs(model, "val")
    tr, va = sorted(CHOLEC80_TRAIN_IDS), sorted(CHOLEC80_VAL_IDS)
    Xtr, ytr = by_video(emb, vid, tr), by_video(y, vid, tr)
    X, truth, P = by_video(emb, vid, va), by_video(y, vid, va), by_video(post, vid, va)
    # round_trip: the default parser can be off by an ulp, and the check is exact
    reg = pd.read_csv(REGISTERED_VAL / "temporal_tcn_curves.csv",
                      float_precision="round_trip")
    reg = reg[reg.model == model]
    tdir = temporal_dir(model)

    crow, sel_rows, chosen, reg_nets = [], [], {}, {}
    for seed in tm.SEEDS:
        runs = [tm.train_tcn(Xtr, ytr, X, truth, d, seed, epochs=tm.EPOCHS_SENSITIVITY)
                for d in tm.DEPTHS_SENSITIVITY]
        for r in runs:
            crow += [{"seed": seed, "depth": r.depth, "epoch": e + 1, "val_macro_f1": f,
                      "train_seconds": r.seconds} for e, f in enumerate(r.val_f1)]
            if r.depth in tm.DEPTHS:
                ref = reg[(reg.seed == seed) & (reg.depth == r.depth)].sort_values("epoch")
                if r.val_f1[:tm.EPOCHS] != ref["val_macro_f1"].tolist():
                    raise SystemExit(f"{model} seed {seed} depth {r.depth}: the first "
                                     f"{tm.EPOCHS} epochs differ from the registered run")
        best = tm.select_tcn(runs)
        ck = torch.load(tdir / f"tcn_seed{seed}.pt", weights_only=True)
        reg_nets[seed] = tm.load_tcn(ck["state"], ck["in_dim"], ck["depth"])
        if best.depth in tm.DEPTHS and best.best_epoch <= tm.EPOCHS:
            same = (best.depth == ck["depth"] and best.best_epoch == ck["epoch"]
                    and all(torch.equal(best.state[k], ck["state"][k]) for k in ck["state"]))
            if not same:
                raise SystemExit(f"{model} seed {seed}: choice inside the registered grid "
                                 f"but not the registered TCN")
        torch.save({"state": best.state, "depth": best.depth, "epoch": best.best_epoch,
                    "in_dim": emb.shape[1], "seed": seed},
                   tdir / f"tcn_sensitivity_seed{seed}.pt")
        chosen[seed] = best
        sel_rows.append({"method": TCN_SENS, "param": "depth,epoch",
                         "value": f"{best.depth},{best.best_epoch}", "seed": seed,
                         "val_macro_f1": best.best_f1,
                         "receptive_field_s": tm.receptive_field(best.depth),
                         "registered": f"{ck['depth']},{ck['epoch']}"})
    selection = {"model": model, "git_hash": ghash, "revision": imeta["revision"],
                 "index_manifest_sha256": imeta["manifest_sha256"],
                 "created": dt.datetime.now().isoformat(timespec="seconds"),
                 "label": "sensitivity, not registered (Deviations 2026-10-13)",
                 "depths": list(tm.DEPTHS_SENSITIVITY), "max_epochs": tm.EPOCHS_SENSITIVITY,
                 "tcn": {str(s): {"depth": r.depth, "epoch": r.best_epoch,
                                  "val_macro_f1": r.best_f1} for s, r in chosen.items()},
                 "train_videos": "1-32", "selection_videos": "33-40"}
    (tdir / SENS_SELECTION).write_text(json.dumps(selection, indent=2) + "\n")

    preds = {"single-frame": {v: p.argmax(1) for v, p in P.items()}}
    for seed in tm.SEEDS:
        preds[f"{TCN}-seed{seed}"] = tm.tcn_predict(reg_nets[seed], X)
        preds[f"{TCN_SENS}-seed{seed}"] = tm.tcn_predict(
            tm.load_tcn(chosen[seed].state, emb.shape[1], chosen[seed].depth), X)
    stats = {m: tm.video_stats(p, truth) for m, p in preds.items()}
    info = {"single_frame_accuracy_check": check_single_frame(model, "val", stats["single-frame"]),
            "threads": threads, "seconds": round(time.time() - t0, 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)}
    return {"model": model, "stats": stats, "curves": pd.DataFrame(crow),
            "sel_rows": sel_rows, "info": info}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2,
                    help="must equal the registered val run's (2) for the curve check")
    args = ap.parse_args()
    ghash = git_hash(ROOT)
    if ghash.endswith("-dirty"):
        sys.exit("the sensitivity selection needs a clean tree")
    reg_threads = json.loads((REGISTERED_VAL / "run.json").read_text())["threads"]
    if args.threads != reg_threads:
        sys.exit(f"use --threads {reg_threads}, as in the registered val run")
    stamp = dt.datetime.now()
    out = ROOT / "results" / "dev" / f"tcn_sensitivity_val_{stamp:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(m, ghash, args.threads) for m in args.models]
    if args.jobs == 1:
        done = [_run(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs)),
                                 max_tasks_per_child=1) as pool:
            done = list(pool.map(_run, jobs))
    counts = resample_counts(len(CHOLEC80_VAL_IDS), B_DESCRIPTIVE, BOOT_SEED)
    table, _, diff = summarise(done, counts)
    sel = pd.DataFrame([r | {"model": d["model"]} for d in done for r in d["sel_rows"]])
    curves = pd.concat([d["curves"].assign(model=d["model"]) for d in done], ignore_index=True)
    for name, df in (("tcn_sensitivity_summary", table), ("tcn_sensitivity_diff", diff),
                     ("tcn_sensitivity_selection", sel), ("tcn_sensitivity_curves", curves)):
        df = df.copy()
        if df.columns[0] != "model":
            df.insert(0, "model", df.pop("model"))
        df.insert(0, "label", LABEL)
        df.to_csv(out / f"{name}.csv", index=False)
    run = {"label": LABEL, "models": args.models, "git_hash": ghash, "boot": B_DESCRIPTIVE,
           "seed": BOOT_SEED, "date": stamp.isoformat(timespec="seconds"),
           "wall_s": round(time.time() - t0, 1), "jobs": args.jobs, "threads": args.threads,
           "depths": list(tm.DEPTHS_SENSITIVITY), "max_epochs": tm.EPOCHS_SENSITIVITY,
           "per_model": {d["model"]: d["info"] for d in done}}
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(f"\n*** {LABEL} ***")
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(sel.to_string(index=False))
        print(diff.to_string(index=False))
    for m, i in run["per_model"].items():
        print(f"{m}: {i['seconds']} s, peak RSS {i['peak_rss_mb']} MB")
    print(f"*** {LABEL} ***\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
