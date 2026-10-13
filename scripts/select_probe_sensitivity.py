#!/usr/bin/env python
"""Sensitivity analysis of the probe's tool heads: SENSITIVITY, NOT REGISTERED.

    python scripts/select_probe_sensitivity.py    # clean tree; after eval_probe.py --split val

Deviations, 2026-10-13: on validation, 13 of 28 tool heads chose the largest registered
C (100). Here each tool head is re-selected over C_GRID_SENSITIVITY (the registered grid
plus 1e3 and 1e4), trained on 1-32 and selected by AP on 33-40, everything else as
section 4.3. Phase heads are not touched. The primary analysis is unchanged.

Per model (data/index/<model>/videos_1-40):
  * refits all eight C values per tool. The six registered ones must reproduce the
    registered validation run's AP exactly, and a tool whose choice stays on the
    registered grid must give back the registered head bit for bit, or the script stops;
  * saves data/probe/<model>/probe_tools_sensitivity.npz (the registered phase head plus
    the re-selected tool heads), which eval_probe.py --split test then scores on 41-80;
  * scores primary and re-selected tool heads on 33-40 ->
    results/dev/probe_sensitivity_val_<timestamp>/, labelled NOT A RESULT.
"""

from __future__ import annotations

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
from eval_probe import EVAL_VIDEOS, MODELS, ROOT, load_pool, probe_dir  # noqa: E402

from srb.datasets.cholec80 import TOOL_COLUMNS  # noqa: E402
from srb.index import git_hash  # noqa: E402
from srb.probe import (C_GRID, C_GRID_SENSITIVITY, MAX_ITER,  # noqa: E402
                       compare_tool_grids, evaluate_tools, load_probes, save_probes,
                       select_tool_probes)
from srb.resample import B_DESCRIPTIVE, BOOT_SEED, resample_counts  # noqa: E402

REGISTERED_VAL = ROOT / "results" / "dev" / "probe_val_20261013-105258"
LABEL = ("DEV, videos 33-40 (val split), SENSITIVITY, NOT REGISTERED "
         "(extended tool C grid), NOT A RESULT")
SENS_FILE = "probe_tools_sensitivity.npz"


def _run(job) -> dict:
    model, ghash = job
    t0 = time.time()
    emb, manifest, imeta = load_pool(model, "videos_1-40")
    phase, tools, pmeta = load_probes(probe_dir(model) / "probe.npz")
    tools_s, infos = select_tool_probes(emb, manifest, grid=C_GRID_SENSITIVITY)
    fits = pd.DataFrame([vars(i) for i in infos])

    reg = pd.read_csv(REGISTERED_VAL / "probe_selection.csv")
    reg = reg[(reg.model == model) & (reg.head != "phase")].set_index(["head", "C"])
    for i in infos:
        if i.C in C_GRID and abs(i.val_score - reg.loc[(i.head, i.C), "val_score"]) > 1e-12:
            raise SystemExit(f"{model} {i.head} C={i.C}: AP differs from the registered run")
    for t in TOOL_COLUMNS:
        if tools_s[t].C in C_GRID and not (
                tools_s[t].C == tools[t].C
                and np.array_equal(tools_s[t].coef, tools[t].coef)
                and np.array_equal(tools_s[t].intercept, tools[t].intercept)):
            raise SystemExit(f"{model} {t}: re-selected head differs from the registered one")

    meta = {"model": model, "revision": imeta["revision"], "index": "videos_1-40",
            "index_manifest_sha256": imeta["manifest_sha256"], "git_hash": ghash,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "label": "sensitivity, not registered", "tool_C_grid": list(C_GRID_SENSITIVITY),
            "phase_head": "registered (probe.npz, git " + pmeta["git_hash"] + ")",
            "max_iter": MAX_ITER, "train_videos": "1-32", "selection_videos": "33-40"}
    save_probes(probe_dir(model) / SENS_FILE, phase, tools_s, meta)

    keep = manifest["video_id"].isin(EVAL_VIDEOS["val"]).to_numpy()
    X, man = emb[keep], manifest[keep].reset_index(drop=True)
    counts = resample_counts(man["video_id"].nunique(), B_DESCRIPTIVE, BOOT_SEED)
    pt_p, map_p, boot_p = evaluate_tools(tools, X, man, counts)
    pt_s, map_s, boot_s = evaluate_tools(tools_s, X, man, counts)
    return {"model": model, "fits": fits, "primary": (pt_p, map_p, boot_p),
            "sensitivity": (pt_s, map_s, boot_s),
            "info": {"seconds": round(time.time() - t0, 1),
                     "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                          / 2**20, 1)}}


def main() -> None:
    ghash = git_hash(ROOT)
    if ghash.endswith("-dirty"):
        sys.exit("the sensitivity selection needs a clean tree")
    for m in MODELS:
        if not (probe_dir(m) / "probe.npz").exists():
            sys.exit(f"{m}: no registered probe; run eval_probe.py --split val first")
    stamp = dt.datetime.now()
    out = ROOT / "results" / "dev" / f"probe_sensitivity_val_{stamp:%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as pool:
        done = list(pool.map(_run, [(m, ghash) for m in MODELS]))
    summary, per_tool = compare_tool_grids({d["model"]: d for d in done})
    fits = pd.concat([d["fits"].assign(model=d["model"]) for d in done], ignore_index=True)
    for name, df in (("summary", summary), ("per_tool", per_tool), ("selection", fits)):
        df = df.copy()
        df.insert(0, "model", df.pop("model"))
        df.insert(0, "label", LABEL)
        df.to_csv(out / f"{name}.csv", index=False)
    run = {"label": LABEL, "git_hash": ghash, "date": stamp.isoformat(timespec="seconds"),
           "wall_s": round(time.time() - t0, 1), "tool_C_grid": list(C_GRID_SENSITIVITY),
           "boot": B_DESCRIPTIVE, "seed": BOOT_SEED,
           "per_model": {d["model"]: d["info"] for d in done}}
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(f"\n*** {LABEL} ***")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
        print(per_tool.to_string(index=False))
        nc = fits[~fits.converged]
        print("\nfits not converged at max_iter:",
              nc[["model", "head", "C", "n_iter"]].to_string(index=False) if len(nc) else "none")
        print("max n_iter by C:", fits.groupby("C").n_iter.max().to_dict())
    for d in done:
        print(f"{d['model']}: {d['info']['seconds']} s, peak RSS {d['info']['peak_rss_mb']} MB")
    print(f"*** {LABEL} ***\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
