"""Linear probes on frozen embeddings (pre-registration section 4.3, RQ3).

* Input: frame embeddings exactly as stored in the index, upcast to float32
  (``srb.index.load_index``), with **no further normalisation**.
* **Phases:** multinomial logistic regression (scikit-learn ``lbfgs``, L2 penalty,
  ``max_iter = 5000``, no class weighting) for each C in ``C_GRID``, trained on videos
  1-32; the C with the best macro-F1 on videos 33-40 is selected.
* **Tools:** one binary logistic regression per tool, same grid and settings; C is
  selected per tool by AP on 33-40.
* Ties in the selection score go to the smaller C (the first in grid order).

Every fit goes through ``fit_head``, which refuses any row that is not a Cholec80
training video (1-32) and calls ``assert_no_test_leakage`` first. Validation and test
rows only ever reach ``LinearHead.decision``.

A model that stops at ``max_iter`` without converging is **recorded** (``FitInfo``),
never refitted with a larger ``max_iter``: that would be a protocol change.

The L2 penalty is scikit-learn 1.9's default (``l1_ratio = 0``); the ``penalty``
argument is deprecated there, so it is not passed. Scores are computed in float64 from
the fitted weights: phase posteriors are the softmax of the decision values (what
``predict_proba`` returns for a multinomial model), and a tool's ranking score is its
decision value, not its probability, which can saturate at exactly 1.0 and create
artificial ties.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from srb.datasets.cholec80 import PHASES, TOOL_COLUMNS
from srb.datasets.overlap import (CHOLEC80_TRAIN_IDS, CHOLEC80_VAL_IDS,
                                  assert_no_test_leakage)
from srb.metrics import average_precision

C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MAX_ITER = 5000
_CHUNK = 16384  # rows per float64 matmul, to bound memory on an 8 GB machine


@dataclass
class FitInfo:
    head: str  # "phase" or a tool column
    C: float
    n_iter: int
    converged: bool
    seconds: float
    val_score: float = float("nan")  # macro-F1 (phase) or AP (tool) on 33-40


@dataclass
class LinearHead:
    """Fitted weights of one logistic regression: decision = X @ coef.T + intercept."""

    coef: np.ndarray  # (n_out, dim); n_out = 7 for phases, 1 for a binary tool head
    intercept: np.ndarray  # (n_out,)
    C: float

    def decision(self, X: np.ndarray) -> np.ndarray:
        """float64 decision values, (N, n_out)."""
        W = self.coef.astype(np.float64).T
        b = self.intercept.astype(np.float64)
        out = np.empty((len(X), W.shape[1]))
        for s in range(0, len(X), _CHUNK):
            out[s:s + _CHUNK] = X[s:s + _CHUNK].astype(np.float64) @ W + b
        return out


def phase_labels(manifest: pd.DataFrame) -> np.ndarray:
    """Phase index 0..6 in the canonical order ``PHASES``."""
    y = pd.Categorical(manifest["phase"], categories=PHASES).codes
    if (y < 0).any():
        raise ValueError("unknown phase label in manifest")
    return np.asarray(y, dtype=np.int64)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _split_masks(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if (manifest["dataset"] != "cholec80").any():
        raise ValueError("the probe is trained and tuned on Cholec80 only")
    vid = manifest["video_id"].to_numpy()
    train = np.isin(vid, sorted(CHOLEC80_TRAIN_IDS))
    val = np.isin(vid, sorted(CHOLEC80_VAL_IDS))
    if not train.any() or not val.any():
        raise ValueError("need training (1-32) and validation (33-40) videos")
    return train, val


def fit_head(X: np.ndarray, y: np.ndarray, train_manifest: pd.DataFrame, C: float,
             head: str) -> tuple[LinearHead, FitInfo]:
    """Fit one logistic regression on training rows only. The single entry to ``fit``."""
    assert_no_test_leakage(train_manifest)
    vids = set(int(v) for v in train_manifest["video_id"])
    if not vids <= CHOLEC80_TRAIN_IDS:
        raise ValueError(f"probe training rows must come from videos 1-32, got "
                         f"{sorted(vids - CHOLEC80_TRAIN_IDS)}")
    if len(X) != len(train_manifest) or len(y) != len(X):
        raise ValueError("X, y and the training manifest must have the same rows")
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, y)
    for w in caught:  # pass on anything that is not the convergence warning
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    n_iter = int(np.max(clf.n_iter_))
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught) \
        and n_iter < MAX_ITER
    info = FitInfo(head=head, C=C, n_iter=n_iter, converged=converged,
                   seconds=round(time.time() - t0, 2))
    return LinearHead(clf.coef_.copy(), clf.intercept_.copy(), C), info


def _best(scores: list[float]) -> int:
    """Index of the best score; ties go to the first (smallest C)."""
    s = np.asarray(scores, dtype=np.float64)
    if np.isnan(s).all():
        raise ValueError("no finite validation score")
    return int(np.nanargmax(s))


def select_phase_probe(X: np.ndarray, manifest: pd.DataFrame
                       ) -> tuple[LinearHead, list[FitInfo]]:
    """Fit on 1-32 for every C, select by macro-F1 on 33-40."""
    train, val = _split_masks(manifest)
    assert_no_test_leakage(manifest[val])  # tuning data is checked too
    y = phase_labels(manifest)
    if set(np.unique(y[train])) != set(range(len(PHASES))):
        raise ValueError("every phase must occur in the training videos")
    heads, infos = [], []
    for C in C_GRID:
        head, info = fit_head(X[train], y[train], manifest[train], C, "phase")
        pred = head.decision(X[val]).argmax(1)
        info.val_score = float(f1_score(y[val], pred, labels=range(len(PHASES)),
                                        average="macro", zero_division=0))
        heads.append(head)
        infos.append(info)
    return heads[_best([i.val_score for i in infos])], infos


def select_tool_probes(X: np.ndarray, manifest: pd.DataFrame
                       ) -> tuple[dict[str, LinearHead], list[FitInfo]]:
    """One binary head per tool: fit on 1-32 for every C, select by AP on 33-40."""
    train, val = _split_masks(manifest)
    assert_no_test_leakage(manifest[val])
    selected, infos = {}, []
    for tool in TOOL_COLUMNS:
        y = manifest[tool].to_numpy(dtype=bool)
        if y[train].all() or not y[train].any():
            raise ValueError(f"{tool}: needs positive and negative training frames")
        heads, tool_infos = [], []
        for C in C_GRID:
            head, info = fit_head(X[train], y[train].astype(np.int64), manifest[train], C, tool)
            info.val_score = float(average_precision(head.decision(X[val])[:, 0], y[val]))
            heads.append(head)
            tool_infos.append(info)
        selected[tool] = heads[_best([i.val_score for i in tool_infos])]
        infos += tool_infos
    return selected, infos


# ---- saving and loading the selected probes (data/, gitignored) ----------------------

def save_probes(path, phase: LinearHead, tools: dict[str, LinearHead], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, phase_coef=phase.coef, phase_intercept=phase.intercept,
             phase_C=phase.C, tools=np.array(TOOL_COLUMNS),
             tool_coef=np.stack([tools[t].coef[0] for t in TOOL_COLUMNS]),
             tool_intercept=np.array([tools[t].intercept[0] for t in TOOL_COLUMNS]),
             tool_C=np.array([tools[t].C for t in TOOL_COLUMNS]),
             meta=np.array(pd.Series(meta).to_json()))


def load_probes(path) -> tuple[LinearHead, dict[str, LinearHead], dict]:
    import json

    z = np.load(path, allow_pickle=False)
    if list(z["tools"]) != list(TOOL_COLUMNS):
        raise ValueError(f"{path}: tool order differs from TOOL_COLUMNS")
    phase = LinearHead(z["phase_coef"], z["phase_intercept"], float(z["phase_C"]))
    tools = {t: LinearHead(z["tool_coef"][i:i + 1], z["tool_intercept"][i:i + 1],
                           float(z["tool_C"][i])) for i, t in enumerate(TOOL_COLUMNS)}
    return phase, tools, json.loads(str(z["meta"]))


# ---- evaluation (sections 4.2, 4.3 and 4.6) ------------------------------------------

def zero_shot_phase_predictions(frame_emb: np.ndarray, text_emb: dict[str, np.ndarray]
                                ) -> dict[str, np.ndarray]:
    """The r1-r5 zero-shot phase classifiers of section 4.2 (argmax cosine over the 7
    phase strings of each role), as in ``srb.evaluation.evaluate_model``. Needed here
    for the probe's paired gain over zero-shot; the caller checks the resulting
    accuracies against the registered run's classification table."""
    from srb.evaluation import N_ROLES, role_of, target_of
    from srb.queries import cholec80_queries
    from srb.retrieval import scores_for

    qs = [q for q in cholec80_queries() if q.kind == "phase"]
    out = {}
    for r in range(1, N_ROLES + 1):
        texts = [next(q.text for q in qs if target_of(q) == f"phase_{p.lower()}"
                      and role_of(q) == r) for p in PHASES]
        out[f"r{r}"] = np.argmax(scores_for(np.stack([text_emb[t] for t in texts]),
                                            frame_emb), 0)
    return out


def _confusion(truth: np.ndarray, pred: np.ndarray, vpos: np.ndarray, V: int) -> np.ndarray:
    """(V, K, K) per-video confusion counts [true, pred]."""
    K = len(PHASES)
    c = np.zeros((V, K, K))
    np.add.at(c, (vpos, truth, pred), 1)
    return c


@dataclass
class ProbeEval:
    model: str
    point: dict  # headline numbers on the evaluated videos
    per_tool: pd.DataFrame
    boot: dict  # metric -> (B,) per-resample values
    posteriors: np.ndarray  # (N, 7) float64 phase posteriors


def evaluate_probe(model: str, phase: LinearHead, tools: dict[str, LinearHead],
                   X: np.ndarray, manifest: pd.DataFrame,
                   zs_pred: dict[str, np.ndarray], counts: np.ndarray) -> ProbeEval:
    """Probe phase accuracy, macro-F1 (pooled), per-video accuracy, tool mAP, and the
    gain over zero-shot phase accuracy (mean over r1-r5), on the frames of ``manifest``.

    ``counts`` is the (B, V) multiplicity matrix over the sorted video IDs of the pool
    (``srb.resample.resample_counts``); every model gets the same one.
    """
    from srb.evaluation import _f1_macro_from_confusion
    from srb.resample import SortedQuery, ap_from_groups

    truth = phase_labels(manifest)
    videos = np.sort(manifest["video_id"].unique())
    vpos = np.searchsorted(videos, manifest["video_id"].to_numpy())
    V, K = len(videos), len(PHASES)
    if counts.shape[1] != V:
        raise ValueError("bootstrap counts do not match the number of videos")
    m = counts.astype(np.float64)

    post = softmax(phase.decision(X))
    pred = post.argmax(1)
    c = _confusion(truth, pred, vpos, V)
    acc_v = np.trace(c, axis1=1, axis2=2) / c.sum((1, 2))
    point = {
        "phase_accuracy": float((pred == truth).mean()),
        "phase_macro_f1": float(f1_score(truth, pred, labels=range(K), average="macro",
                                         zero_division=0)),
        "phase_per_video_accuracy": float(acc_v.mean()),
        "phase_C": phase.C,
    }
    cb = np.einsum("bv,vij->bij", m, c)
    boot = {"phase_accuracy": np.trace(cb, axis1=1, axis2=2) / cb.sum((1, 2)),
            "phase_macro_f1": _f1_macro_from_confusion(cb),
            "phase_per_video_accuracy": (m @ acc_v) / m.sum(1)}

    zs_acc, zs_boot = [], []
    for name in sorted(zs_pred):
        cz = np.einsum("bv,vij->bij", m, _confusion(truth, zs_pred[name], vpos, V))
        zs_acc.append(float((zs_pred[name] == truth).mean()))
        zs_boot.append(np.trace(cz, axis1=1, axis2=2) / cz.sum((1, 2)))
    point["zs_phase_accuracy"] = float(np.mean(zs_acc))
    point["gain_over_zs_accuracy"] = point["phase_accuracy"] - point["zs_phase_accuracy"]
    boot["zs_phase_accuracy"] = np.mean(zs_boot, 0)
    boot["gain_over_zs_accuracy"] = boot["phase_accuracy"] - boot["zs_phase_accuracy"]

    rows, tool_boot = [], []
    for tool in TOOL_COLUMNS:
        rel = manifest[tool].to_numpy(dtype=bool)
        s = tools[tool].decision(X)[:, 0]
        rows.append({"tool": tool, "C": tools[tool].C, "n_relevant": int(rel.sum()),
                     "prevalence": float(rel.mean()), "ap": average_precision(s, rel)})
        sq = SortedQuery(s, rel, vpos)
        b_ap = np.full(len(m), np.nan)
        for b in range(len(m)):
            g, r = sq.group_counts(counts[b])
            if r.sum():
                b_ap[b] = ap_from_groups(g, r)
        tool_boot.append(b_ap)
    per_tool = pd.DataFrame(rows)
    point["tool_map"] = float(per_tool["ap"].mean())
    tb = np.stack(tool_boot, 1)
    ok = np.isfinite(tb)
    boot["tool_map"] = np.where(ok, tb, 0).sum(1) / ok.sum(1)
    return ProbeEval(model, point, per_tool, boot, post)


HEADLINE = ("phase_accuracy", "phase_macro_f1", "phase_per_video_accuracy", "tool_map",
            "zs_phase_accuracy", "gain_over_zs_accuracy")
COMPARED = ("phase_accuracy", "phase_macro_f1", "tool_map")


def summarise_probes(evals: dict[str, ProbeEval]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model headline numbers with 95% percentile intervals, and every pairwise
    difference (plus the best-to-worst spread across models) with intervals.
    Descriptive (pre-registration 4.6): no p-values, no multiplicity correction."""
    from srb.resample import percentile_ci

    rows = []
    for model, ev in evals.items():
        for metric in HEADLINE:
            lo, hi = percentile_ci(ev.boot[metric])
            rows.append({"model": model, "metric": metric, "value": ev.point[metric],
                         "ci_low": float(lo), "ci_high": float(hi),
                         "n_boot": len(ev.boot[metric])})
    summary = pd.DataFrame(rows)

    models = list(evals)
    prow = []
    for metric in COMPARED:
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                d = evals[a].boot[metric] - evals[b].boot[metric]
                lo, hi = percentile_ci(d)
                prow.append({"pair": f"{a} - {b}", "metric": metric,
                             "diff": evals[a].point[metric] - evals[b].point[metric],
                             "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(d)})
    for metric in ("phase_accuracy", "zs_phase_accuracy"):
        pts = np.array([evals[mm].point[metric] for mm in models])
        bs = np.stack([evals[mm].boot[metric] for mm in models], 1)
        lo, hi = percentile_ci(bs.max(1) - bs.min(1))
        prow.append({"pair": "best - worst (spread)", "metric": metric,
                     "diff": float(pts.max() - pts.min()), "ci_low": float(lo),
                     "ci_high": float(hi), "n_boot": len(bs)})
    return summary, pd.DataFrame(prow)
