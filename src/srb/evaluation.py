"""The registered Cholec80 evaluation (pre-registration sections 4.1, 4.2 and 4.6).

One call of ``evaluate_model`` scores one model on one pool of videos (validation
33-40, or test 41-80 once the scoring lock is lifted) and returns:

* ``per_query``: every query string (70 targets + 7 negatives): AP, prevalence, lift,
  R-Precision, P@10, P@50, distinct videos in the top 50, per-video AP, tie counts,
  score mean/max;
* ``per_target``: the 14 targets: headline AP (mean over the 5 rephrasings), SD
  (ddof = 1) and range across rephrasings, R-Precision and per-video AP likewise,
  prevalence, lift, and the prompt-ensemble AP;
* ``summary``: phase / tool / overall mAP and the same aggregation for R-Precision,
  per-video AP and the ensemble; mean prevalence and lift; RQ2 mean SD and the mean
  AP by rephrasing role; zero-shot phase classification; the calibration check;
* ``boot``: per-resample arrays from the paired video bootstrap (same resamples for
  every model), from which ``summarise`` and ``compare_models`` take intervals,
  p-values and Holm adjustment.

Lift for a family is its mAP over the mean prevalence of its targets (the
random-ranker level of that mAP). The CVS criteria need Endoscapes2023 frames and are
not part of a Cholec80 pool: an Endoscapes pool is scored with
``queries=endoscapes_queries()``, which yields the ``cvs`` family only (no overall
Cholec80 mAP, no confirmatory family, no phase classification; Deviations, 2026-10-20).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from srb.datasets.cholec80 import PHASES
from srb.metrics import (average_precision, count_ties, precision_at_k, r_precision,
                         random_baseline_ap)
from srb.queries import REPHRASING_ROLES, cholec80_queries
from srb.resample import (B_DESCRIPTIVE, SortedQuery, ap_from_groups, bootstrap_p, holm,
                          percentile_ci, r_precision_from_groups)
from srb.retrieval import Query, _cholec80_ids, refuse_test_scoring, scores_for

N_ROLES = len(REPHRASING_ROLES)
TOP_K_DISTINCT = 50


def target_of(q: Query) -> str:
    """'phase_preparation_r3' -> 'phase_preparation'."""
    return q.qid.rsplit("_r", 1)[0]


def role_of(q: Query) -> int:
    return int(q.qid.rsplit("_r", 1)[1])


def expected_distinct_videos(scores: np.ndarray, video: np.ndarray, k: int) -> float:
    """Expected number of distinct videos in the top k, tie-neutral.

    Videos with a frame in a group entirely inside the top k count 1. For the group
    that straddles the cutoff (g frames, t of its slots inside the top k), a video not
    already counted that has m frames in it is present with probability
    1 - C(g - m, t) / C(g, t) under a uniformly random order within the group.
    """
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    s, v = s[order], np.asarray(video)[order]
    k = min(k, len(s))
    cut = s[k - 1]
    above = s > cut
    inside = set(v[above].tolist())
    group = s == cut
    g, t = int(group.sum()), k - int(above.sum())
    out = float(len(inside))
    vids, counts = np.unique(v[group], return_counts=True)
    for vid, m in zip(vids.tolist(), counts.tolist()):
        if vid not in inside:
            out += 1 - math.comb(g - m, t) / math.comb(g, t)
    return out


def _family_groups(families: list[str]) -> list[tuple[str, np.ndarray]]:
    """(name, mask over targets) for each family in order, plus ``overall`` (all 14
    Cholec80 targets) when the families are exactly phase and tool."""
    fam = np.asarray(families)
    names = list(dict.fromkeys(families))
    groups = [(f, fam == f) for f in names]
    if names == ["phase", "tool"]:
        groups.append(("overall", np.ones(len(fam), bool)))
    return groups


def _f1_macro_from_confusion(c: np.ndarray) -> np.ndarray:
    """Macro-F1 over all classes from (..., K, K) confusion counts [true, pred];
    a class with no true and no predicted frame scores 0 (sklearn zero_division=0)."""
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    denom = c.sum(-1) + c.sum(-2)  # 2TP + FP + FN
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(-1)


@dataclass
class ModelEval:
    model: str
    per_query: pd.DataFrame
    per_target: pd.DataFrame
    summary: dict
    classification: pd.DataFrame
    calibration: pd.DataFrame
    boot: dict = field(repr=False)
    target_order: list = field(repr=False)


def evaluate_model(model: str, frame_emb: np.ndarray, manifest: pd.DataFrame,
                   text_emb: dict[str, np.ndarray], counts: np.ndarray | None,
                   queries: list[Query] | None = None) -> ModelEval:
    """Score one model on the frames of ``manifest`` (row i = ``frame_emb`` row i).

    ``text_emb`` maps each query string to its unit-norm text embedding. ``counts`` is
    the (B, V) bootstrap multiplicity matrix over the sorted video IDs of the pool
    (``srb.resample.resample_counts``), or None to skip the bootstrap. ``queries``
    defaults to the registered Cholec80 set.
    """
    refuse_test_scoring(_cholec80_ids(manifest))
    queries = cholec80_queries() if queries is None else queries
    pos = [q for q in queries if q.kind != "negative"]
    neg = [q for q in queries if q.kind == "negative"]
    videos = np.sort(manifest.video_id.unique())
    vpos = np.searchsorted(videos, manifest.video_id.to_numpy())
    S = scores_for(np.stack([text_emb[q.text] for q in queries]), frame_emb)
    score_of = dict(zip((q.qid for q in queries), S))

    # ---- per query string ------------------------------------------------------------
    rows, per_video_ap = [], {}
    for q in queries:
        s, rel = score_of[q.qid], q.relevant(manifest)
        tied_frames, tied_groups = count_ties(s)
        row = {"qid": q.qid, "kind": q.kind, "target": target_of(q) if q.kind != "negative"
               else None, "role": role_of(q) if q.kind != "negative" else None,
               "text": q.text, "n_frames": len(s), "n_relevant": int(rel.sum()),
               "score_mean": float(s.mean()), "score_max": float(s.max()),
               "tied_frames": tied_frames, "tied_groups": tied_groups,
               "p_at_10": precision_at_k(s, rel, 10), "p_at_50": precision_at_k(s, rel, 50),
               "distinct_videos_top50": expected_distinct_videos(s, vpos, TOP_K_DISTINCT)}
        if q.kind != "negative":
            pv = np.full(len(videos), np.nan)
            for i in range(len(videos)):
                m = vpos == i
                if rel[m].any():
                    pv[i] = average_precision(s[m], rel[m])
            per_video_ap[q.qid] = pv
            ap, prev = average_precision(s, rel), random_baseline_ap(rel)
            row.update(ap=ap, prevalence=prev, lift=ap / prev, r_prec=r_precision(s, rel),
                       per_video_ap=float(np.nanmean(pv)),
                       per_video_n=int(np.isfinite(pv).sum()))
        rows.append(row)
    per_query = pd.DataFrame(rows)

    # ---- prompt ensemble: one normalised mean embedding per target ----------------------
    target_order = list(dict.fromkeys(target_of(q) for q in pos))
    by_target = {t: [q for q in pos if target_of(q) == t] for t in target_order}
    ens_text = {}
    for t, qs in by_target.items():
        e = np.mean([text_emb[q.text] for q in qs], axis=0)
        ens_text[t] = (e / np.linalg.norm(e)).astype(np.float32)
    ens_scores = dict(zip(target_order, scores_for(np.stack(list(ens_text.values())),
                                                   frame_emb)))

    # ---- per target -----------------------------------------------------------------
    pq = per_query.set_index("qid")
    trows = []
    for t, qs in by_target.items():
        sub = pq.loc[[q.qid for q in qs]]
        rel = qs[0].relevant(manifest)
        ens = ens_scores[t]
        trows.append({
            "target": t, "family": qs[0].kind, "n_relevant": int(rel.sum()),
            "prevalence": random_baseline_ap(rel), "ap": sub.ap.mean(),
            "ap_sd": sub.ap.std(ddof=1), "ap_range": sub.ap.max() - sub.ap.min(),
            "r_prec": sub.r_prec.mean(), "per_video_ap": sub.per_video_ap.mean(),
            "per_video_n": int(sub.per_video_n.iloc[0]),
            "ensemble_ap": average_precision(ens, rel),
            "ensemble_r_prec": r_precision(ens, rel)})
    per_target = pd.DataFrame(trows)
    per_target["lift"] = per_target.ap / per_target.prevalence

    # ---- zero-shot phase classification (4.2; Cholec80 pools only) ----------------------
    has_phase = any(q.kind == "phase" for q in pos)
    preds = {}
    if has_phase:
        truth = pd.Categorical(manifest.phase, categories=PHASES).codes
        phase_q = {r: [next(q for q in pos if q.kind == "phase"
                            and target_of(q) == f"phase_{p.lower()}" and role_of(q) == r)
                       for p in PHASES] for r in range(1, N_ROLES + 1)}
        preds = {f"r{r}": np.argmax(np.stack([score_of[q.qid] for q in qs]), 0)
                 for r, qs in phase_q.items()}
        preds["ensemble"] = np.argmax(
            np.stack([ens_scores[f"phase_{p.lower()}"] for p in PHASES]), 0)
    K, V = len(PHASES), len(videos)
    conf = {}  # classifier -> (V, K, K) per-video confusion counts
    crows = []
    for name, pred in preds.items():
        c = np.zeros((V, K, K))
        np.add.at(c, (vpos, truth, pred), 1)
        conf[name] = c
        per_video_acc = np.trace(c, axis1=1, axis2=2) / c.sum((1, 2))
        crows.append({"classifier": name, "accuracy": float((pred == truth).mean()),
                      "macro_f1": float(f1_score(truth, pred, labels=range(K), average="macro",
                                                 zero_division=0)),
                      "per_video_accuracy": float(per_video_acc.mean())})
    classification = pd.DataFrame(crows, columns=["classifier", "accuracy", "macro_f1",
                                                  "per_video_accuracy"])

    # ---- calibration (negatives vs positive targets) -------------------------------------
    pos_stats = (per_query[per_query.kind != "negative"].groupby("target", sort=False)
                 .agg(score_mean=("score_mean", "mean"), score_max=("score_max", "max"))
                 .reset_index().rename(columns={"target": "query"}))
    negs = (per_query[per_query.kind == "negative"][["qid", "score_mean", "score_max"]]
            .rename(columns={"qid": "query"}))
    negs["below_every_positive_target_mean"] = negs.score_mean < pos_stats.score_mean.min()
    calibration = pd.concat([negs.assign(kind="negative"), pos_stats.assign(kind="positive target")],
                            ignore_index=True)

    # ---- summary ----------------------------------------------------------------------
    summary = {"model": model, "n_frames": len(manifest), "n_videos": int(V)}
    for fam, sel in _family_groups(list(per_target.family)):
        pt = per_target[sel]
        summary.update({
            f"{fam}_map": pt.ap.mean(), f"{fam}_prevalence": pt.prevalence.mean(),
            f"{fam}_lift": pt.ap.mean() / pt.prevalence.mean(),
            f"{fam}_r_prec": pt.r_prec.mean(), f"{fam}_per_video_ap": pt.per_video_ap.mean(),
            f"{fam}_ensemble_map": pt.ensemble_ap.mean(),
            f"{fam}_ensemble_r_prec": pt.ensemble_r_prec.mean()})
    summary["rq2_mean_sd"] = per_target.ap_sd.mean()
    for r in range(1, N_ROLES + 1):
        summary[f"role_r{r}_map"] = per_query[per_query.role == r].ap.mean()
    if has_phase:
        roles = classification[classification.classifier.str.match(r"r\d")]
        summary["zs_phase_accuracy"] = roles.accuracy.mean()
        summary["zs_phase_macro_f1"] = roles.macro_f1.mean()
        summary["zs_phase_per_video_accuracy"] = roles.per_video_accuracy.mean()
        ens = classification.set_index("classifier").loc["ensemble"]
        summary["zs_phase_accuracy_ensemble"] = ens.accuracy
        summary["zs_phase_macro_f1_ensemble"] = ens.macro_f1
    summary["calibration_all_negatives_below"] = bool(
        negs.below_every_positive_target_mean.all())

    boot = {} if counts is None else _bootstrap(
        counts, pos, score_of, manifest, vpos, per_video_ap, target_order, ens_scores, conf)
    return ModelEval(model, per_query, per_target, summary, classification, calibration,
                     boot, target_order)


def _bootstrap(counts, pos, score_of, manifest, vpos, per_video_ap, target_order,
               ens_scores, conf) -> dict:
    """Per-resample arrays. AP of every positive query on all B resamples (the
    confirmatory overall mAP needs B = 10,000); everything else on the first 2,000."""
    B, Bd = len(counts), min(B_DESCRIPTIVE, len(counts))
    n = len(pos)
    ap = np.full((B, n), np.nan)
    rp = np.full((Bd, n), np.nan)
    prev = np.full((Bd, n), np.nan)
    for j, q in enumerate(pos):
        sq = SortedQuery(score_of[q.qid], q.relevant(manifest), vpos)
        for b in range(B):
            g, r = sq.group_counts(counts[b])
            if r.sum() == 0:
                continue  # target absent from this resample: undefined, left nan
            ap[b, j] = ap_from_groups(g, r)
            if b < Bd:
                rp[b, j] = r_precision_from_groups(g, r)
                prev[b, j] = r.sum() / g.sum()
    m = counts[:Bd].astype(np.float64)
    pv = np.stack([per_video_ap[q.qid] for q in pos], 1)  # (V, n)
    ok = np.isfinite(pv)
    with np.errstate(invalid="ignore", divide="ignore"):
        pvap = (m @ np.where(ok, pv, 0)) / (m @ ok)
    ens = np.full((Bd, len(target_order)), np.nan)
    ens_rp = np.full((Bd, len(target_order)), np.nan)
    for j, t in enumerate(target_order):
        q0 = next(q for q in pos if target_of(q) == t)
        sq = SortedQuery(ens_scores[t], q0.relevant(manifest), vpos)
        for b in range(Bd):
            g, r = sq.group_counts(counts[b])
            if r.sum():
                ens[b, j] = ap_from_groups(g, r)
                ens_rp[b, j] = r_precision_from_groups(g, r)
    cls = {}
    for name, c in conf.items():
        cb = np.einsum("bv,vij->bij", m, c)
        acc_v = np.trace(c, axis1=1, axis2=2) / c.sum((1, 2))
        cls[name] = {"accuracy": np.trace(cb, axis1=1, axis2=2) / cb.sum((1, 2)),
                     "macro_f1": _f1_macro_from_confusion(cb),
                     "per_video_accuracy": (m @ acc_v) / m.sum(1)}
    return {"ap": ap, "r_prec": rp, "prevalence": prev, "per_video_ap": pvap,
            "ensemble_ap": ens, "ensemble_r_prec": ens_rp, "classification": cls,
            "qids": [q.qid for q in pos], "families": [q.kind for q in pos]}


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    """Mean over finite entries; nan (no warning) where a row has none."""
    ok = np.isfinite(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, a, 0).sum(axis) / ok.sum(axis)


def _nanstd1(a: np.ndarray) -> np.ndarray:
    """Row-wise SD (ddof = 1) over finite entries; nan where fewer than 2."""
    ok = np.isfinite(a)
    n = ok.sum(1)
    mu = _nanmean(a, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (np.where(ok, a - mu[:, None], 0) ** 2).sum(1) / (n - 1)
    return np.where(n > 1, np.sqrt(np.where(n > 1, var, 0)), np.nan)


def _target_means(a: np.ndarray, qids: list[str], target_order: list[str]) -> np.ndarray:
    """(B, n_queries) -> (B, n_targets): mean over each target's rephrasings."""
    t = np.array([q.rsplit("_r", 1)[0] for q in qids])
    return np.stack([_nanmean(a[:, t == name], 1) for name in target_order], 1)


def headline_resamples(ev: ModelEval, confirmatory: bool = False) -> dict[str, np.ndarray]:
    """Per-resample values of the summary metrics (B for overall_map if confirmatory,
    else the first 2,000 for everything)."""
    bt = ev.boot
    Bd = len(bt["r_prec"])
    groups = _family_groups([t.split("_")[0] for t in ev.target_order])
    out = {}
    ap_t_all = _target_means(bt["ap"], bt["qids"], ev.target_order)
    if confirmatory:
        if groups[-1][0] != "overall":
            raise ValueError("the confirmatory family (overall Cholec80 mAP) needs a Cholec80 pool")
        return {"overall_map": _nanmean(ap_t_all, 1)}
    ap_t = ap_t_all[:Bd]
    rp_t = _target_means(bt["r_prec"], bt["qids"], ev.target_order)
    pv_t = _target_means(bt["per_video_ap"], bt["qids"], ev.target_order)
    prev_t = _target_means(bt["prevalence"], bt["qids"], ev.target_order)
    for name, sel in groups:
        out[f"{name}_map"] = _nanmean(ap_t[:, sel], 1)
        out[f"{name}_prevalence"] = _nanmean(prev_t[:, sel], 1)
        out[f"{name}_lift"] = out[f"{name}_map"] / out[f"{name}_prevalence"]
        out[f"{name}_r_prec"] = _nanmean(rp_t[:, sel], 1)
        out[f"{name}_per_video_ap"] = _nanmean(pv_t[:, sel], 1)
        out[f"{name}_ensemble_map"] = _nanmean(bt["ensemble_ap"][:, sel], 1)
        out[f"{name}_ensemble_r_prec"] = _nanmean(bt["ensemble_r_prec"][:, sel], 1)
    roles = np.array([int(q.rsplit("_r", 1)[1]) for q in bt["qids"]])
    for r in range(1, N_ROLES + 1):
        out[f"role_r{r}_map"] = _nanmean(bt["ap"][:Bd, roles == r], 1)
    tq = np.array([q.rsplit("_r", 1)[0] for q in bt["qids"]])
    sd = np.stack([_nanstd1(bt["ap"][:Bd, tq == t]) for t in ev.target_order], 1)
    out["rq2_mean_sd"] = _nanmean(sd, 1)
    cls = bt["classification"]
    if cls:
        role_names = [k for k in cls if k.startswith("r")]
        for metric in ("accuracy", "macro_f1", "per_video_accuracy"):
            out[f"zs_phase_{metric}"] = np.mean([cls[k][metric] for k in role_names], 0)
        out["zs_phase_accuracy_ensemble"] = cls["ensemble"]["accuracy"]
        out["zs_phase_macro_f1_ensemble"] = cls["ensemble"]["macro_f1"]
    for j, t in enumerate(ev.target_order):
        out[f"target_ap:{t}"] = ap_t[:, j]
    return out


def summarise(ev: ModelEval) -> pd.DataFrame:
    """Long table: metric, value (full pool), 95% percentile interval, resamples used."""
    rows = []
    point = dict(ev.summary)
    point.update({f"target_ap:{t}": a for t, a in zip(ev.per_target.target, ev.per_target.ap)})
    res = headline_resamples(ev) if ev.boot else {}
    conf = (headline_resamples(ev, confirmatory=True)
            if ev.boot and "overall_map" in ev.summary else {})
    for metric, value in point.items():
        if metric in ("model", "n_frames", "n_videos", "calibration_all_negatives_below"):
            continue
        row = {"model": ev.model, "metric": metric, "value": float(value),
               "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}
        x = conf.get(metric) if metric in conf else res.get(metric)
        if x is not None:
            lo, hi = percentile_ci(x)
            row.update(ci_low=float(lo), ci_high=float(hi), n_boot=len(x))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(evals: dict[str, ModelEval], metric: str = "overall_map",
                   confirmatory: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confirmatory: pairwise differences in overall Cholec80 mAP (B = all resamples),
    two-sided bootstrap p, Holm-Bonferroni at 0.05. Plus the RQ2 order-flip fraction:
    per pair, the share of targets whose model order differs between rephrasings.

    With ``confirmatory=False`` (e.g. ``metric="cvs_map"`` on Endoscapes, section 4.6:
    descriptive) the difference gets a 95% interval from the first 2,000 resamples and
    no p-value or Holm adjustment."""
    names = list(evals)
    if confirmatory:
        res = {m: headline_resamples(evals[m], confirmatory=True)[metric]
               for m in names if evals[m].boot}
    else:
        res = {m: headline_resamples(evals[m])[metric] for m in names if evals[m].boot}
    rows, flips = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ea, eb = evals[a], evals[b]
            pair = f"{a} - {b}"
            row = {"pair": pair, "diff": ea.summary[metric] - eb.summary[metric]}
            if a in res and b in res:
                d = res[a] - res[b]
                lo, hi = percentile_ci(d)
                row.update(ci_low=float(lo), ci_high=float(hi), n_boot=len(d))
                if confirmatory:
                    row["p"] = bootstrap_p(d)
            rows.append(row)
            qa = ea.per_query.set_index("qid").ap
            qb = eb.per_query.set_index("qid").ap
            d = (qa - qb).dropna()
            tgt = d.index.str.rsplit("_r", n=1).str[0]
            changed = d.groupby(tgt).apply(lambda x: bool((x > 0).any() and (x < 0).any()))
            flips.append({"pair": pair, "targets": len(changed),
                          "order_changes_fraction": float(changed.mean())})
    pairs = pd.DataFrame(rows)
    if "p" in pairs:
        adj = holm(dict(zip(pairs.pair, pairs.p)))
        pairs["p_holm"] = [adj[p][0] for p in pairs.pair]
        pairs["significant_holm_0.05"] = [adj[p][1] for p in pairs.pair]
    return pairs, pd.DataFrame(flips)
