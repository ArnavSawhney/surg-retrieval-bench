"""Temporal models over per-frame phase evidence (pre-registration section 4.4, RQ4),
with the details fixed by the clarification of 2026-10-13 (Deviations).

Every sequence is one video at 1 fps, in frame order; collections of sequences are
dicts keyed by Cholec80 video ID.

* **Single frame:** argmax of the probe posterior (the RQ3 prediction).
* **Moving average:** mean of the posteriors over a window of w frames, then argmax.
  Causal (primary): frames t-w+1 .. t. Centred (secondary, offline): frames
  t-floor(w/2) .. t+ceil(w/2)-1. Both are shortened to the frames that exist at the
  ends of a video. w is chosen on 33-40 by macro-F1; ties go to the smaller w.
* **HMM:** 7 states; pi and A from the phase sequences of videos 1-32 with add-one
  smoothing; emissions = posterior (floored at 1e-12) / training class prior. Forward
  filtering (primary, causal) and Viterbi (secondary, offline). Nothing is tuned.
* **Causal TCN:** dilated causal convolutions over the frozen embeddings (``CausalTCN``);
  depth and epoch chosen jointly on 33-40 per seed; trained on the CPU with
  deterministic algorithms. A secondary sensitivity analysis, **not registered**
  (Deviations, 2026-10-13), re-selects over depths 2-8 and up to 100 epochs, all else
  identical.
* **Metrics:** accuracy and macro-F1 pooled over frames (as in 4.2); Jaccard per phase
  per video, averaged over the videos in which the phase occurs, then over phases;
  segmental edit score (Lea et al., 2016) per video, averaged over videos. Bootstrap
  values on the paired video resamples of section 4.6 come from per-video statistics
  (``VideoStats``), weighted by each video's multiplicity.

Leakage guards: ``fit_hmm`` and ``train_tcn`` accept only training videos (1-32);
``select_window`` and the TCN's epoch/depth selection accept only validation videos
(33-40). Test videos (41-80) can only be passed to the prediction functions.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score

from srb.datasets.cholec80 import PHASES
from srb.datasets.overlap import CHOLEC80_TRAIN_IDS, CHOLEC80_VAL_IDS

K = len(PHASES)
WINDOWS = (5, 10, 20, 30, 60, 120)  # seconds = frames at 1 fps
DEPTHS = (2, 3, 4)
SEEDS = (0, 1, 2)
EPOCHS = 50
# secondary TCN sensitivity analysis, not registered (Deviations, 2026-10-13)
DEPTHS_SENSITIVITY = tuple(range(2, 9))
EPOCHS_SENSITIVITY = 100
LR = 1e-3
CHANNELS = 64
KERNEL = 3
EMISSION_FLOOR = 1e-12

Seqs = dict  # video ID -> np.ndarray, one video per entry


def _require(videos, allowed: frozenset, what: str) -> None:
    bad = sorted(int(v) for v in videos if int(v) not in allowed)
    if bad:
        raise ValueError(f"{what} only accepts videos {min(allowed)}-{max(allowed)}, "
                         f"got {bad}")
    if not len(list(videos)):
        raise ValueError(f"{what}: no videos")


def macro_f1(truth: Seqs, pred: Seqs) -> float:
    """Pooled macro-F1 over the 7 phases, as in 4.2 (sklearn, zero_division=0)."""
    vids = sorted(truth)
    return float(f1_score(np.concatenate([truth[v] for v in vids]),
                          np.concatenate([pred[v] for v in vids]),
                          labels=range(K), average="macro", zero_division=0))


# ---- moving average -----------------------------------------------------------------

def moving_average(post: np.ndarray, w: int, causal: bool = True) -> np.ndarray:
    """(T, K) mean of the posteriors over the window ending at t (causal) or centred
    on t, shortened at the ends of the video."""
    if w < 1:
        raise ValueError("w must be >= 1")
    T = len(post)
    c = np.zeros((T + 1, post.shape[1]))
    c[1:] = np.cumsum(post.astype(np.float64), 0)
    t = np.arange(T)
    if causal:
        lo, hi = np.maximum(t - w + 1, 0), t + 1
    else:
        lo, hi = np.maximum(t - w // 2, 0), np.minimum(t + (w + 1) // 2, T)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def ma_predict(post: Seqs, w: int, causal: bool = True) -> Seqs:
    return {v: moving_average(p, w, causal).argmax(1) for v, p in post.items()}


def select_window(post: Seqs, truth: Seqs, causal: bool = True,
                  grid: tuple = WINDOWS) -> tuple[int, list[float]]:
    """w with the best macro-F1 on validation videos 33-40 (ties: smaller w)."""
    _require(post, CHOLEC80_VAL_IDS, "select_window")
    scores = [macro_f1(truth, ma_predict(post, w, causal)) for w in grid]
    return grid[int(np.argmax(scores))], scores


# ---- HMM ----------------------------------------------------------------------------

@dataclass
class HMM:
    pi: np.ndarray  # (K,)
    A: np.ndarray  # (K, K), A[i, j] = P(next = j | current = i)
    prior: np.ndarray  # (K,) class shares of the training frames


def fit_hmm(labels: Seqs) -> HMM:
    """pi, A (add-one smoothing) and the class prior from training videos 1-32."""
    _require(labels, CHOLEC80_TRAIN_IDS, "fit_hmm")
    start = np.ones(K)
    trans = np.ones((K, K))
    frames = np.zeros(K)
    for y in labels.values():
        y = np.asarray(y)
        start[y[0]] += 1
        np.add.at(trans, (y[:-1], y[1:]), 1)
        frames += np.bincount(y, minlength=K)
    if (frames == 0).any():
        raise ValueError("every phase must occur in the training videos")
    return HMM(start / start.sum(), trans / trans.sum(1, keepdims=True),
               frames / frames.sum())


def emissions(post: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, int]:
    """Scaled likelihoods posterior / prior, and the number of floored posteriors."""
    p = post.astype(np.float64)
    n_floored = int((p < EMISSION_FLOOR).sum())
    return np.maximum(p, EMISSION_FLOOR) / prior, n_floored


def forward_filter(e: np.ndarray, hmm: HMM) -> np.ndarray:
    """(T, K) filtered state probabilities P(state_t | e_0..e_t)."""
    alpha = np.empty_like(e, dtype=np.float64)
    a = hmm.pi * e[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(e)):
        a = (alpha[t - 1] @ hmm.A) * e[t]
        alpha[t] = a / a.sum()
    return alpha


def viterbi(e: np.ndarray, hmm: HMM) -> np.ndarray:
    """Most likely state path (log space); ties go to the lower state index."""
    T, n = e.shape
    logA, le = np.log(hmm.A), np.log(e)
    delta = np.log(hmm.pi) + le[0]
    back = np.empty((T, n), dtype=np.int64)
    for t in range(1, T):
        cand = delta[:, None] + logA  # [from, to]
        back[t] = cand.argmax(0)
        delta = cand.max(0) + le[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def hmm_predict(post: Seqs, hmm: HMM, method: str = "forward") -> tuple[Seqs, int]:
    out, floored = {}, 0
    for v, p in post.items():
        e, n = emissions(p, hmm.prior)
        floored += n
        out[v] = forward_filter(e, hmm).argmax(1) if method == "forward" else viterbi(e, hmm)
    return out, floored


# ---- causal TCN ---------------------------------------------------------------------

def _torch():
    import torch
    return torch


def make_tcn(in_dim: int, depth: int):
    """Layer 1: causal conv in_dim -> 64 (kernel 3, dilation 1); layers 2..depth: 64 ->
    64 with dilations 2, 4, 8, ...; ReLU after each; 1x1 conv 64 -> 7. Causality by
    left-only zero padding of (k-1)*d frames. No normalisation, dropout or residuals."""
    torch = _torch()
    nn = torch.nn

    class CausalTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.dilations = [2 ** i for i in range(depth)]
            self.convs = nn.ModuleList(
                nn.Conv1d(in_dim if i == 0 else CHANNELS, CHANNELS, KERNEL, dilation=d)
                for i, d in enumerate(self.dilations))
            self.out = nn.Conv1d(CHANNELS, K, 1)

        def forward(self, x):  # (1, in_dim, T) -> (1, K, T)
            for conv, d in zip(self.convs, self.dilations):
                x = torch.relu(conv(nn.functional.pad(x, ((KERNEL - 1) * d, 0))))
            return self.out(x)

    return CausalTCN()


def receptive_field(depth: int) -> int:
    return 1 + (KERNEL - 1) * sum(2 ** i for i in range(depth))


def _as_tensor(x: np.ndarray):
    torch = _torch()
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32).T[None])


def tcn_logits(net, X: np.ndarray) -> np.ndarray:
    """(T, K) logits for one video's (T, dim) embeddings."""
    torch = _torch()
    net.eval()
    with torch.no_grad():
        return net(_as_tensor(X))[0].T.numpy()


def tcn_predict(net, X: Seqs) -> Seqs:
    return {v: tcn_logits(net, x).argmax(1) for v, x in X.items()}


@dataclass
class TCNRun:
    depth: int
    seed: int
    val_f1: list  # macro-F1 on 33-40 after each epoch (1-based epoch = index + 1)
    best_epoch: int
    state: dict = field(repr=False)  # weights at best_epoch
    seconds: float = 0.0

    @property
    def best_f1(self) -> float:
        return self.val_f1[self.best_epoch - 1]


def train_tcn(X_train: Seqs, y_train: Seqs, X_val: Seqs, y_val: Seqs, depth: int,
              seed: int, epochs: int | None = None) -> TCNRun:
    """Adam (lr 1e-3), mean cross-entropy over one video's frames per step, the video
    order reshuffled each epoch by a generator seeded with ``seed``. After every epoch
    the validation macro-F1 is computed; the weights of the best epoch are kept (ties:
    the earlier epoch). CPU, deterministic algorithms."""
    _require(X_train, CHOLEC80_TRAIN_IDS, "train_tcn")
    _require(X_val, CHOLEC80_VAL_IDS, "TCN epoch selection")
    if set(X_train) != set(y_train) or set(X_val) != set(y_val):
        raise ValueError("embeddings and labels must cover the same videos")
    epochs = EPOCHS if epochs is None else epochs
    torch = _torch()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    dim = next(iter(X_train.values())).shape[1]
    net = make_tcn(dim, depth)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    vids = sorted(X_train)
    xs = [_as_tensor(X_train[v]) for v in vids]
    ys = [torch.from_numpy(np.asarray(y_train[v], dtype=np.int64)) for v in vids]
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    curve, best, best_state = [], -np.inf, None
    for _ in range(epochs):
        net.train()
        for i in torch.randperm(len(vids), generator=gen).tolist():
            loss = torch.nn.functional.cross_entropy(net(xs[i])[0].T, ys[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = macro_f1(y_val, tcn_predict(net, X_val))
        curve.append(f1)
        if f1 > best:
            best, best_state = f1, copy.deepcopy(net.state_dict())
    return TCNRun(depth, seed, curve, int(np.argmax(curve)) + 1, best_state,
                  round(time.time() - t0, 1))


def select_tcn(runs: list[TCNRun]) -> TCNRun:
    """The (depth, epoch) with the best validation macro-F1 for one seed; ties go to the
    smaller depth, then the earlier epoch."""
    if len({r.seed for r in runs}) != 1:
        raise ValueError("select within one seed")
    runs = sorted(runs, key=lambda r: r.depth)
    return runs[int(np.argmax([r.best_f1 for r in runs]))]


def load_tcn(state: dict, in_dim: int, depth: int):
    net = make_tcn(in_dim, depth)
    net.load_state_dict(state)
    return net


# ---- metrics ------------------------------------------------------------------------

def segments(y: np.ndarray) -> np.ndarray:
    """Label of each run of equal consecutive labels."""
    y = np.asarray(y)
    if not len(y):
        return y
    return y[np.r_[True, y[1:] != y[:-1]]]


def levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, z in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != z)))
        prev = cur
    return prev[-1]


def edit_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """100 * (1 - Levenshtein(segments) / max(#segments)) (Lea et al., 2016)."""
    p, g = segments(pred), segments(truth)
    return 100.0 * (1 - levenshtein(p.tolist(), g.tolist()) / max(len(p), len(g)))


def jaccard_per_phase(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(K,) |P & G| / |P | G| per phase; nan for a phase absent from the ground truth."""
    out = np.full(K, np.nan)
    for k in np.unique(truth):
        p, g = pred == k, truth == k
        out[k] = (p & g).sum() / (p | g).sum()
    return out


@dataclass
class VideoStats:
    videos: np.ndarray  # sorted video IDs
    conf: np.ndarray  # (V, K, K) [true, pred]
    jac: np.ndarray  # (V, K), nan where the phase is absent from the video
    edit: np.ndarray  # (V,)


def video_stats(pred: Seqs, truth: Seqs) -> VideoStats:
    if set(pred) != set(truth):
        raise ValueError("predictions and labels must cover the same videos")
    vids = np.array(sorted(truth))
    conf = np.zeros((len(vids), K, K))
    jac = np.empty((len(vids), K))
    edit = np.empty(len(vids))
    for i, v in enumerate(vids):
        p, g = np.asarray(pred[v]), np.asarray(truth[v])
        if len(p) != len(g):
            raise ValueError(f"video {v}: {len(p)} predictions for {len(g)} frames")
        np.add.at(conf[i], (g, p), 1)
        jac[i] = jaccard_per_phase(p, g)
        edit[i] = edit_score(p, g)
    return VideoStats(vids, conf, jac, edit)


METRICS = ("accuracy", "macro_f1", "jaccard", "edit")


def weighted_metrics(s: VideoStats, m: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics on video multiplicities ``m`` (B, V): each video counted m times. With
    m = ones this is the point estimate."""
    from srb.evaluation import _f1_macro_from_confusion

    m = np.asarray(m, dtype=np.float64)
    cb = np.einsum("bv,vij->bij", m, s.conf)
    present = np.isfinite(s.jac)
    wsum = m @ present  # (B, K)
    jsum = m @ np.where(present, s.jac, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_phase = np.where(wsum > 0, jsum / np.where(wsum > 0, wsum, 1), np.nan)
    ok = np.isfinite(per_phase)
    return {"accuracy": np.trace(cb, axis1=1, axis2=2) / cb.sum((1, 2)),
            "macro_f1": _f1_macro_from_confusion(cb),
            "jaccard": np.where(ok, per_phase, 0).sum(1) / ok.sum(1),
            "edit": (m @ s.edit) / m.sum(1)}


def point_metrics(s: VideoStats) -> dict[str, float]:
    return {k: float(v[0]) for k, v in weighted_metrics(s, np.ones((1, len(s.videos)))).items()}
