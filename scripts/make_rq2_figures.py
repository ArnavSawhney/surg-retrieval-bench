#!/usr/bin/env python
"""RQ2 (prompt sensitivity) figures and table from the registered test run.

    python scripts/make_rq2_figures.py

Reads only the CSVs the registered test run already wrote (results/test/: the per-query
and per-target tables, summary.csv, rq2_order_changes.csv, run.json). Nothing is
re-scored: the RQ1/RQ2 test split is never rerun.

Writes to figures/:
  fig1a_rq2_phases.{png,pdf}, fig1b_rq2_tools.{png,pdf}
      AP of each of the 5 rephrasings per target and model, with the target's
      prevalence (random-ranker AP) as a reference line.
  fig2_rq2_roles.{png,pdf}
      mean AP by rephrasing role (r1-r5) per model, with 95% bootstrap intervals
      (descriptive, B = 2,000).
  rq2_table.csv, rq2_table.md
      per target and model, SD (ddof = 1) and range of AP over the 5 rephrasings; mean
      SD per model; fraction of targets on which each pair of models changes order.
  *.caption.txt
      caption and provenance ("RESULT, Cholec80 test 41-80", run hash) per figure; the
      same provenance is in each PNG's and PDF's metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from srb.plotting import (  # noqa: E402
    GRID, INK_MUTED, MODEL_STYLE, MODELS, ROLE_LABELS, TARGET_LABELS, apply_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "results" / "test"
OUT = ROOT / "figures"
PROVENANCE = "RESULT, Cholec80 test 41-80"
COLUMN_WIDTH_IN = 3.5  # one column of a two-column paper


def load_per_query() -> pd.DataFrame:
    """Phase and tool target strings (negatives excluded), all four models."""
    frames = []
    for m in MODELS:
        q = pd.read_csv(TEST / f"{m}_per_query.csv")
        frames.append(q[q["kind"].isin(["phase", "tool"])])
    q = pd.concat(frames, ignore_index=True)
    q["role"] = q["role"].astype(int)
    labels = set(q["label"])
    if labels != {"RESULT, Cholec80 test videos 41-80"}:
        raise SystemExit(f"unexpected provenance label(s) in per-query CSVs: {labels}")
    counts = q.groupby(["model", "target"]).size()
    if not (counts == 5).all() or len(counts) != len(MODELS) * 14:
        raise SystemExit("expected 4 models x 14 targets x 5 rephrasings")
    return q


def rq2_table(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Per target and model: headline AP, SD (ddof=1) and range over the 5 rephrasings.

    Recomputed from the per-query APs and checked against the per-target CSVs.
    """
    g = q.groupby(["model", "target"])["ap"]
    t = pd.DataFrame({
        "ap_mean": g.mean(),
        "ap_sd": g.std(ddof=1),
        "ap_range": g.max() - g.min(),
    }).reset_index()
    for m in MODELS:
        ref = pd.read_csv(TEST / f"{m}_per_target.csv").set_index("target")
        mine = t[t["model"] == m].set_index("target")
        for col in ("ap_sd", "ap_range"):
            if not np.allclose(mine[col], ref.loc[mine.index, col], rtol=0, atol=1e-12):
                raise SystemExit(f"{m} {col} disagrees with {m}_per_target.csv")
        if not np.allclose(mine["ap_mean"], ref.loc[mine.index, "ap"], rtol=0, atol=1e-12):
            raise SystemExit(f"{m} headline AP disagrees with {m}_per_target.csv")
    mean_sd = t.groupby("model")["ap_sd"].mean()
    summ = pd.read_csv(TEST / "summary.csv")
    ref_sd = summ[summ["metric"] == "rq2_mean_sd"].set_index("model")["value"]
    if not np.allclose(mean_sd[list(MODELS)], ref_sd[list(MODELS)], rtol=0, atol=1e-12):
        raise SystemExit("mean SD disagrees with summary.csv rq2_mean_sd")
    return t, mean_sd


def prevalence(q: pd.DataFrame) -> pd.Series:
    p = q.groupby("target")["prevalence"].agg(["min", "max"])
    if not np.allclose(p["min"], p["max"]):
        raise SystemExit("prevalence differs between models for the same target")
    return p["min"]


def save(fig, stem: str, caption: str, run: dict) -> None:
    desc = f"{PROVENANCE}. {caption} Source: results/test/ from the registered run at git {run['git_hash'][:7]} ({run['date'][:10]})."
    fig.savefig(OUT / f"{stem}.png", metadata={"Title": stem, "Description": desc, "Source": PROVENANCE})
    fig.savefig(OUT / f"{stem}.pdf", metadata={"Title": stem, "Subject": desc, "Keywords": PROVENANCE})
    (OUT / f"{stem}.caption.txt").write_text(desc + "\n")
    plt.close(fig)


def fig_strip(q: pd.DataFrame, prev: pd.Series, family: str, stem: str, run: dict) -> None:
    targets = [t for t in TARGET_LABELS if t.startswith(family + "_")]
    n_models = len(MODELS)
    band = 0.78  # vertical share of a target's row taken by its four model strips
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * band / n_models

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 0.42 * len(targets) + 0.75))
    for i, target in enumerate(targets):
        y0 = len(targets) - 1 - i  # first target at the top
        if i % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color="#f4f4f2", zorder=0, lw=0)
        ax.vlines(prev[target], y0 - band / 2 - 0.04, y0 + band / 2 + 0.04,
                  color=INK_MUTED, lw=0.9, ls=(0, (2, 1.5)), zorder=1)
        for j, m in enumerate(MODELS):
            st = MODEL_STYLE[m]
            y = y0 - offsets[j]  # model order top to bottom within the row
            ap = q[(q["model"] == m) & (q["target"] == target)]["ap"].to_numpy()
            ax.hlines(y, ap.min(), ap.max(), color=st["color"], lw=0.8, alpha=0.55, zorder=2)
            ax.scatter(ap, np.full_like(ap, y), s=9, marker=st["marker"], color=st["color"],
                       edgecolors="white", linewidths=0.35, zorder=3)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([TARGET_LABELS[t] for t in reversed(targets)])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, len(targets) - 0.5)
    ax.set_xlim(0, max(0.85, float(q[q["target"].isin(targets)]["ap"].max()) + 0.03))
    ax.set_xlabel("AP (one marker per rephrasing r1–r5)")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], ls="", marker=MODEL_STYLE[m]["marker"], color=MODEL_STYLE[m]["color"],
                      markersize=3.6, label=MODEL_STYLE[m]["label"]) for m in MODELS]
    handles.append(Line2D([], [], color=INK_MUTED, lw=0.9, ls=(0, (2, 1.5)), label="prevalence"))
    top = 1 - 0.36 / fig.get_figheight()  # room for a two-row legend above the axes
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, top))
    name = "phase" if family == "phase" else "tool"
    caption = (f"Prompt sensitivity, Cholec80 {name} queries: AP of each of the five rephrasings "
               f"(r1-r5) per {name} target and model; the line joins a model's lowest and highest "
               f"rephrasing; dashed tick = the target's prevalence (AP of a random ranking).")
    save(fig, stem, caption, run)


def fig_roles(run: dict) -> None:
    summ = pd.read_csv(TEST / "summary.csv")
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.3))
    roles = np.arange(1, 6)
    dodge = (np.arange(len(MODELS)) - 1.5) * 0.09
    for j, m in enumerate(MODELS):
        st = MODEL_STYLE[m]
        s = summ[(summ["model"] == m) & summ["metric"].str.match(r"role_r\d_map")].copy()
        s["role"] = s["metric"].str.extract(r"role_r(\d)_map")[0].astype(int)
        s = s.set_index("role").loc[roles]
        if set(s["n_boot"]) != {2000}:
            raise SystemExit("role intervals are expected to be descriptive (B = 2,000)")
        x = roles + dodge[j]
        ax.plot(x, s["value"], color=st["color"], lw=0.9, alpha=0.6, zorder=2)
        ax.errorbar(x, s["value"], yerr=[s["value"] - s["ci_low"], s["ci_high"] - s["value"]],
                    fmt="none", ecolor=st["color"], elinewidth=0.8, capsize=0, zorder=2)
        ax.scatter(x, s["value"], marker=st["marker"], s=16, color=st["color"],
                   edgecolors="white", linewidths=0.4, zorder=3, label=st["label"])
    ax.set_xticks(roles)
    ax.set_xticklabels([ROLE_LABELS[r] for r in roles])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("mean AP over 14 targets")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    caption = ("Prompt sensitivity by rephrasing role: mean AP over the 14 Cholec80 phase and tool "
               "targets for each role (r1 technical, r2 action, r3 visible, r4 lay, r5 label-like), per "
               "model; bars = 95% paired video-level bootstrap intervals (descriptive, B = 2,000).")
    save(fig, "fig2_rq2_roles", caption, run)


def write_table(t: pd.DataFrame, mean_sd: pd.Series, run: dict) -> None:
    wide = t.pivot(index="target", columns="model", values=["ap_sd", "ap_range"])
    wide = wide.loc[list(TARGET_LABELS)]
    rows = []
    for target in TARGET_LABELS:
        r = {"target": target}
        for m in MODELS:
            r[f"{m}_sd"] = wide.loc[target, ("ap_sd", m)]
            r[f"{m}_range"] = wide.loc[target, ("ap_range", m)]
        rows.append(r)
    mean_row = {"target": "mean over targets"}
    for m in MODELS:
        mean_row[f"{m}_sd"] = mean_sd[m]
        mean_row[f"{m}_range"] = t[t["model"] == m]["ap_range"].mean()
    rows.append(mean_row)
    table = pd.DataFrame(rows)
    table.insert(0, "label", PROVENANCE)
    table.to_csv(OUT / "rq2_table.csv", index=False)

    oc = pd.read_csv(TEST / "rq2_order_changes.csv")
    short = {m: MODEL_STYLE[m]["label"] for m in MODELS}
    lines = [
        f"# RQ2 prompt sensitivity: {PROVENANCE}",
        "",
        f"Registered run at git `{run['git_hash'][:7]}` ({run['date'][:10]}); built by "
        "`scripts/make_rq2_figures.py` from `results/test/`. Per target and model: SD (ddof = 1) "
        "and range (max − min) of AP over the five rephrasings. Point estimates (descriptive).",
        "",
        "| target | " + " | ".join(f"{short[m]} SD | {short[m]} range" for m in MODELS) + " |",
        "|---|" + "---:|" * (2 * len(MODELS)),
    ]
    for r in rows:
        name = TARGET_LABELS.get(r["target"], f"**{r['target']}**")
        cells = []
        for m in MODELS:
            sd, rg = f"{r[f'{m}_sd']:.3f}", f"{r[f'{m}_range']:.3f}"
            if r["target"] == "mean over targets":
                sd, rg = f"**{sd}**", f"**{rg}**"
            cells += [sd, rg]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Fraction of the 14 targets on which the two models' order changes between rephrasings "
        "(section 4.1 of the pre-registration; from `rq2_order_changes.csv`):",
        "",
        "| pair | targets | order changes |",
        "|---|---:|---:|",
    ]
    for _, r in oc.iterrows():
        a, b = r["pair"].split(" - ")
        k = round(r["order_changes_fraction"] * r["targets"])
        lines.append(f"| {short[a]} vs {short[b]} | {r['targets']} | {k}/{r['targets']} "
                     f"({r['order_changes_fraction']:.2f}) |")
    (OUT / "rq2_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    run = json.loads((TEST / "run.json").read_text())
    if run["split"] != "test":
        raise SystemExit("results/test/run.json is not the test run")
    apply_paper_style()
    q = load_per_query()
    t, mean_sd = rq2_table(q)
    prev = prevalence(q)
    fig_strip(q, prev, "phase", "fig1a_rq2_phases", run)
    fig_strip(q, prev, "tool", "fig1b_rq2_tools", run)
    fig_roles(run)
    write_table(t, mean_sd, run)
    print(f"{PROVENANCE}: wrote figures and rq2_table to {OUT.relative_to(ROOT)}/")
    for m in MODELS:
        print(f"  {MODEL_STYLE[m]['label']:<15} mean SD {mean_sd[m]:.4f}")


if __name__ == "__main__":
    main()
