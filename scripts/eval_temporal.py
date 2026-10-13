#!/usr/bin/env python
"""Temporal models, RQ4 (docs/preregistration.md, section 4.4 and its 2026-10-13
clarification under Deviations).

    python scripts/eval_temporal.py --split val    # fit on 1-32, select on 33-40: dev, not a result
    python scripts/eval_temporal.py --split test   # the selected models, once, on 41-80

Inputs: the selected probe's phase posteriors exactly as saved by eval_probe.py
(data/probe/<model>/posteriors_videos_*.npz; the probes are not refit) and, for the
TCN, the frozen embeddings (data/index/<model>/videos_*).

--split val, per model: moving-average windows (causal and centred) chosen on 33-40;
the HMM estimated from the labels of 1-32; the TCN trained on 1-32 for every depth and
seed, (depth, epoch) chosen per seed on 33-40. The selection and the TCN weights go to
data/temporal/<model>/; every method is scored on 33-40 -> results/dev/temporal_val_<ts>/,
labelled "NOT A RESULT".

--split test (clean tree only, once): loads that selection (no refit, no reselection),
scores every method on 41-80, writes results/test/temporal_*.csv and appends rows to
results/results.csv (settings single-frame, ma-causal, ma-centred, hmm-forward,
hmm-viterbi, tcn-causal; TCN rows per seed plus the mean).

The single-frame prediction must reproduce the probe's registered phase accuracy on the
same split exactly, or the script stops. Intervals: paired video-level bootstrap, the
first 2,000 resamples of default_rng(20261011), the same resamples for every model and
method; descriptive (section 4.6).
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

import srb.temporal as tm
from srb.datasets.cholec80 import PHASES
from srb.datasets.overlap import CHOLEC80_TRAIN_IDS, CHOLEC80_VAL_IDS
from srb.index import git_hash, load_index
from srb.models.registry import _SPECS
from srb.probe import phase_labels
from srb.resample import B_DESCRIPTIVE, BOOT_SEED, percentile_ci, resample_counts

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("clip-vit-l14", "siglip-so400m-384", "medsiglip-448", "peskavlp")
INDEX = {"val": "videos_1-40", "test": "videos_41-80"}
EVAL_VIDEOS = {"val": list(range(33, 41)), "test": list(range(41, 81))}
LABEL = {"val": "DEV, videos 33-40 (val split), trained on 1-32, NOT A RESULT",
         "test": "RESULT, Cholec80 test videos 41-80"}
PROBE_SUMMARY = {"val": "results/dev/probe_val_20261013-105258/probe_summary.csv",
                 "test": "results/test/probe_summary.csv"}
SETTINGS = ("single-frame", "ma-causal", "ma-centred", "hmm-forward", "hmm-viterbi",
            "tcn-causal")
CAUSAL = ("ma-causal", "hmm-forward", "tcn-causal")


def temporal_dir(model: str) -> Path:
    return ROOT / "data" / "temporal" / model


def by_video(a: np.ndarray, vid: np.ndarray, keep) -> dict:
    return {int(v): a[vid == v] for v in keep}


def load_inputs(model: str, split: str):
    """Embeddings, labels and saved posteriors of the split's index, row-aligned."""
    emb, man, meta = load_index(ROOT / "data" / "index" / model / INDEX[split])
    if meta["revision"] != _SPECS[model].revision:
        raise SystemExit(f"{model}: index revision differs from the registry")
    z = np.load(ROOT / "data" / "probe" / model / f"posteriors_{INDEX[split]}.npz",
                allow_pickle=False)
    if not (np.array_equal(z["video_id"], man["video_id"].to_numpy())
            and np.array_equal(z["frame_idx_25fps"], man["frame_idx_25fps"].to_numpy())
            and list(z["phases"]) == list(PHASES)):
        raise SystemExit(f"{model}: posteriors are not row-aligned with the index")
    for v, g in man.groupby("video_id", sort=False):
        if not (np.diff(g["frame_idx_25fps"].to_numpy()) == 25).all():
            raise SystemExit(f"{model}: video {v} is not in 1 fps frame order")
    return emb, phase_labels(man), z["posterior"], man["video_id"].to_numpy(), meta


def check_single_frame(model: str, split: str, stats: tm.VideoStats) -> float:
    ref = pd.read_csv(ROOT / PROBE_SUMMARY[split])
    ref = float(ref[(ref.model == model) & (ref.metric == "phase_accuracy")]["value"].iloc[0])
    mine = tm.point_metrics(stats)["accuracy"]
    if abs(mine - ref) > 1e-12:
        raise SystemExit(f"{model}: single-frame accuracy {mine} differs from the "
                         f"registered probe accuracy {ref}")
    return mine


def _setup_torch(threads: int) -> None:
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def _run(job) -> dict:
    """One model in its own process: predictions per method on the evaluated videos,
    their per-video statistics and, on val, the selection."""
    model, split, ghash, threads = job
    _setup_torch(threads)
    import torch

    t0 = time.time()
    emb, y, post, vid, imeta = load_inputs(model, split)
    ev = EVAL_VIDEOS[split]
    truth = by_video(y, vid, ev)
    P = by_video(post, vid, ev)
    X = by_video(emb, vid, ev)
    tdir = temporal_dir(model)
    info = {"index_manifest_sha256": imeta["manifest_sha256"], "threads": threads}
    sel_rows, curves = [], None

    if split == "val":
        tr = sorted(CHOLEC80_TRAIN_IDS)
        hmm = tm.fit_hmm(by_video(y, vid, tr))
        w_c, sc_c = tm.select_window(P, truth, causal=True)
        w_o, sc_o = tm.select_window(P, truth, causal=False)
        sel_rows += [{"method": m, "param": "w", "value": w, "seed": "", "val_macro_f1": s[i]}
                     for m, w, s in (("ma-causal", w_c, sc_c), ("ma-centred", w_o, sc_o))
                     for i in [tm.WINDOWS.index(w)]]
        grid = [{"method": m, "w": w, "val_macro_f1": s}
                for m, sc in (("ma-causal", sc_c), ("ma-centred", sc_o))
                for w, s in zip(tm.WINDOWS, sc)]
        Xtr, ytr = by_video(emb, vid, tr), by_video(y, vid, tr)
        tcn, crow = {}, []
        for seed in tm.SEEDS:
            runs = [tm.train_tcn(Xtr, ytr, X, truth, d, seed) for d in tm.DEPTHS]
            best = tm.select_tcn(runs)
            crow += [{"seed": seed, "depth": r.depth, "epoch": e + 1, "val_macro_f1": f,
                      "train_seconds": r.seconds} for r in runs for e, f in enumerate(r.val_f1)]
            tcn[seed] = best
            sel_rows.append({"method": "tcn-causal", "param": "depth,epoch",
                             "value": f"{best.depth},{best.best_epoch}", "seed": seed,
                             "val_macro_f1": best.best_f1})
        curves = pd.DataFrame(crow)
        tdir.mkdir(parents=True, exist_ok=True)
        np.savez(tdir / "hmm.npz", pi=hmm.pi, A=hmm.A, prior=hmm.prior)
        for seed, r in tcn.items():
            torch.save({"state": r.state, "depth": r.depth, "epoch": r.best_epoch,
                        "in_dim": emb.shape[1], "seed": seed}, tdir / f"tcn_seed{seed}.pt")
        selection = {"model": model, "git_hash": ghash, "revision": imeta["revision"],
                     "index_manifest_sha256": imeta["manifest_sha256"],
                     "created": dt.datetime.now().isoformat(timespec="seconds"),
                     "w_causal": w_c, "w_centred": w_o,
                     "tcn": {str(s): {"depth": r.depth, "epoch": r.best_epoch,
                                      "val_macro_f1": r.best_f1} for s, r in tcn.items()},
                     "train_videos": "1-32", "selection_videos": "33-40"}
        (tdir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        nets = {s: tm.load_tcn(r.state, emb.shape[1], r.depth) for s, r in tcn.items()}
        info["ma_grid"] = grid
    else:
        selection = json.loads((tdir / "selection.json").read_text())
        if selection["revision"] != imeta["revision"]:
            raise SystemExit(f"{model}: selection and index revisions differ")
        w_c, w_o = selection["w_causal"], selection["w_centred"]
        h = np.load(tdir / "hmm.npz")
        hmm = tm.HMM(h["pi"], h["A"], h["prior"])
        nets = {}
        for seed in tm.SEEDS:
            ck = torch.load(tdir / f"tcn_seed{seed}.pt", weights_only=True)
            nets[seed] = tm.load_tcn(ck["state"], ck["in_dim"], ck["depth"])
    info["selection_git_hash"] = selection["git_hash"]

    preds = {"single-frame": {v: p.argmax(1) for v, p in P.items()},
             "ma-causal": tm.ma_predict(P, w_c, True),
             "ma-centred": tm.ma_predict(P, w_o, False)}
    preds["hmm-forward"], info["hmm_floored"] = tm.hmm_predict(P, hmm, "forward")
    preds["hmm-viterbi"], _ = tm.hmm_predict(P, hmm, "viterbi")
    for seed, net in nets.items():
        preds[f"tcn-causal-seed{seed}"] = tm.tcn_predict(net, X)
    stats = {m: tm.video_stats(p, truth) for m, p in preds.items()}
    info["single_frame_accuracy_check"] = check_single_frame(model, split, stats["single-frame"])
    info["tcn_receptive_field_s"] = {str(s): tm.receptive_field(int(selection["tcn"][str(s)]["depth"]))
                                     for s in tm.SEEDS}
    info["seconds"] = round(time.time() - t0, 1)
    info["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
    return {"model": model, "stats": stats, "selection": selection, "sel_rows": sel_rows,
            "curves": curves, "info": info}


def summarise(done: list[dict], counts: np.ndarray):
    """Per (model, method, metric): value and interval, and the paired gain over single
    frame; TCN seeds individually and as their mean (SD over seeds on the full set)."""
    rows, boots = [], {}
    for d in done:
        model, stats = d["model"], d["stats"]
        pt = {m: tm.point_metrics(s) for m, s in stats.items()}
        bt = {m: tm.weighted_metrics(s, counts) for m, s in stats.items()}
        seeds = [f"tcn-causal-seed{s}" for s in tm.SEEDS]
        pt["tcn-causal"] = {k: float(np.mean([pt[s][k] for s in seeds])) for k in tm.METRICS}
        bt["tcn-causal"] = {k: np.mean([bt[s][k] for s in seeds], 0) for k in tm.METRICS}
        sd = {k: float(np.std([pt[s][k] for s in seeds], ddof=1)) for k in tm.METRICS}
        for method in pt:
            for k in tm.METRICS:
                lo, hi = percentile_ci(bt[method][k])
                gain = bt[method][k] - bt["single-frame"][k]
                glo, ghi = percentile_ci(gain)
                rows.append({"model": model, "method": method, "metric": k,
                             "value": pt[method][k], "ci_low": lo, "ci_high": hi,
                             "sd_over_seeds": sd[k] if method == "tcn-causal" else np.nan,
                             "gain": pt[method][k] - pt["single-frame"][k],
                             "gain_ci_low": glo, "gain_ci_high": ghi,
                             "n_boot": len(counts)})
                boots[(model, method, k)] = bt[method][k]
    table = pd.DataFrame(rows)
    # gain over single frame averaged over the backbones (paired, same resamples)
    mrows = []
    models = [d["model"] for d in done]
    for method in table["method"].unique():
        for k in tm.METRICS:
            g = np.mean([boots[(m, method, k)] - boots[(m, "single-frame", k)] for m in models], 0)
            pts = table[(table.method == method) & (table.metric == k)]
            lo, hi = percentile_ci(g)
            mrows.append({"method": method, "metric": k, "mean_gain": float(pts["gain"].mean()),
                          "ci_low": lo, "ci_high": hi, "n_models": len(models),
                          "n_boot": len(counts)})
    return table, pd.DataFrame(mrows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=sorted(INDEX))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--jobs", type=int, default=2, help="models in parallel (8 GB Mac: 2)")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads per model")
    args = ap.parse_args()

    ghash = git_hash(ROOT)
    results = ROOT / "results" / "results.csv"
    if args.split == "test":
        if ghash.endswith("-dirty"):
            sys.exit("the temporal test run needs a clean tree")
        if list(args.models) != list(MODELS):
            sys.exit("the temporal test run covers all four models at once")
        prior = pd.read_csv(results)
        if prior["setting"].isin(SETTINGS).any():
            sys.exit("results.csv already holds temporal rows: the test run happens once")
        for m in args.models:
            if not (temporal_dir(m) / "selection.json").exists():
                sys.exit(f"{m}: no selection; run --split val first")
    stamp = dt.datetime.now()
    label = LABEL[args.split]
    out = (ROOT / "results" / "dev" / f"temporal_val_{stamp:%Y%m%d-%H%M%S}"
           if args.split == "val" else ROOT / "results" / "test")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    jobs = [(m, args.split, ghash, args.threads) for m in args.models]
    if args.jobs == 1:  # in-process (tests; peak RSS is then cumulative)
        done = [_run(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs)),
                                 max_tasks_per_child=1) as pool:
            done = list(pool.map(_run, jobs))
    infos = {d["model"]: d["info"] for d in done}
    counts = resample_counts(len(EVAL_VIDEOS[args.split]), B_DESCRIPTIVE, BOOT_SEED)
    table, backbone_mean = summarise(done, counts)
    sel = pd.DataFrame([r | {"model": d["model"]} for d in done for r in d["sel_rows"]]) \
        if args.split == "val" else pd.DataFrame(
            [{"model": d["model"], "method": "ma-causal", "param": "w",
              "value": d["selection"]["w_causal"]} for d in done]
            + [{"model": d["model"], "method": "ma-centred", "param": "w",
                "value": d["selection"]["w_centred"]} for d in done]
            + [{"model": d["model"], "method": "tcn-causal", "param": "depth,epoch",
                "seed": s, "value": f"{v['depth']},{v['epoch']}"}
               for d in done for s, v in d["selection"]["tcn"].items()])
    tables = {"temporal_summary": table, "temporal_backbone_mean_gain": backbone_mean,
              "temporal_selection": sel}
    if args.split == "val":
        tables["temporal_tcn_curves"] = pd.concat(
            [d["curves"].assign(model=d["model"]) for d in done], ignore_index=True)
        tables["temporal_ma_grid"] = pd.concat(
            [pd.DataFrame(d["info"].pop("ma_grid")).assign(model=d["model"]) for d in done],
            ignore_index=True)
    for name, df in tables.items():
        df = df.copy()
        if "model" in df and df.columns[0] != "model":
            df.insert(0, "model", df.pop("model"))
        df.insert(0, "label", label)
        df.to_csv(out / f"{name}.csv", index=False)
    run = {"label": label, "split": args.split, "models": args.models,
           "boot": B_DESCRIPTIVE, "seed": BOOT_SEED, "git_hash": ghash,
           "date": stamp.isoformat(timespec="seconds"), "wall_s": round(time.time() - t0, 1),
           "jobs": args.jobs, "threads": args.threads, "windows": list(tm.WINDOWS),
           "depths": list(tm.DEPTHS), "tcn_seeds": list(tm.SEEDS), "epochs": tm.EPOCHS,
           "per_model": infos}
    (out / ("run.json" if args.split == "val" else "temporal_run.json")).write_text(
        json.dumps(run, indent=2) + "\n")

    if args.split == "test":
        date = stamp.date().isoformat()
        common = dict(split="test", git_hash=ghash, date=date)
        rows = []
        for r in table.itertuples():
            if r.method.startswith("tcn-causal-seed"):
                setting, seed = "tcn-causal", r.method.removeprefix("tcn-causal-seed")
            else:
                setting = r.method
                seed = "mean(0,1,2)" if r.method == "tcn-causal" else BOOT_SEED
            rows.append({"model": r.model, "metric": f"temporal_{r.metric}", "value": r.value,
                         "ci_low": r.ci_low, "ci_high": r.ci_high, "n_boot": r.n_boot,
                         "setting": setting, "seed": seed})
            if r.method != "single-frame":
                rows.append({"model": r.model, "metric": f"temporal_{r.metric}_gain_over_single_frame",
                             "value": r.gain, "ci_low": r.gain_ci_low, "ci_high": r.gain_ci_high,
                             "n_boot": r.n_boot, "setting": setting, "seed": seed})
            if r.method == "tcn-causal":
                rows.append({"model": r.model, "metric": f"temporal_{r.metric}_sd_over_seeds",
                             "value": r.sd_over_seeds, "setting": setting, "seed": seed})
        for r in backbone_mean.itertuples():
            if r.method.startswith("tcn-causal-seed") or r.method == "single-frame":
                continue
            rows.append({"model": "mean over backbones",
                         "metric": f"temporal_{r.metric}_gain_over_single_frame",
                         "value": r.mean_gain, "ci_low": r.ci_low, "ci_high": r.ci_high,
                         "n_boot": r.n_boot, "setting": r.method,
                         "seed": "mean(0,1,2)" if r.method == "tcn-causal" else BOOT_SEED})
        new = pd.DataFrame(rows).assign(**common)
        # append only: re-writing would re-format the registered rows' floats (trap 16)
        header = list(pd.read_csv(results, nrows=0).columns)
        new.reindex(columns=header).to_csv(results, mode="a", header=False, index=False)

    print(f"\n*** {label} ***")
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format,
                           "display.max_rows", 500):
        print(sel.to_string(index=False))
        print(table.pivot_table(index=["model", "method"], columns="metric", values="value",
                                sort=False).reindex(columns=list(tm.METRICS)).to_string())
        print("\ngain over single frame (paired):")
        print(table[table.method != "single-frame"].pivot_table(
            index=["model", "method"], columns="metric", values="gain",
            sort=False).reindex(columns=list(tm.METRICS)).to_string())
        print("\ngain over single frame, mean over backbones:")
        print(backbone_mean.pivot_table(index="method", columns="metric", values="mean_gain",
                                        sort=False).reindex(columns=list(tm.METRICS)).to_string())
    for m, i in infos.items():
        print(f"{m}: {i['seconds']} s, peak RSS {i['peak_rss_mb']} MB, "
              f"HMM floored {i['hmm_floored']}, TCN RF {i['tcn_receptive_field_s']}")
    print(f"*** {label} ***\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
