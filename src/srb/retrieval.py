"""Score an index against text queries and compute per-query retrieval metrics.

Similarities are computed in **float32** from upcast embeddings (``load_index``
returns float32) and handed to the metrics as float64. Every row reports the tie count
of its score vector (``count_ties``), since the metrics' tie handling only matters if
ties exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from srb.metrics import (
    average_precision,
    count_ties,
    precision_at_k,
    r_precision,
    random_baseline_ap,
)


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    kind: str  # "phase" | "tool" | "negative"
    target: str | None  # phase name, tool column, or None for a negative control

    def relevant(self, manifest: pd.DataFrame) -> np.ndarray:
        if self.kind == "phase":
            return (manifest["phase"] == self.target).to_numpy()
        if self.kind == "tool":
            return manifest[self.target].to_numpy(dtype=bool)
        if self.kind == "negative":
            return np.zeros(len(manifest), dtype=bool)
        raise ValueError(f"unknown query kind {self.kind!r}")


def scores_for(text_emb: np.ndarray, frame_emb: np.ndarray) -> np.ndarray:
    """(n_queries, n_frames) cosine similarities in float32 (inputs are unit-norm)."""
    if frame_emb.dtype == np.float16 or text_emb.dtype == np.float16:
        raise TypeError("upcast fp16 embeddings to float32 before computing similarities")
    return text_emb.astype(np.float32) @ frame_emb.astype(np.float32).T


def evaluate(queries: list[Query], scores: np.ndarray, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, s in zip(queries, scores):
        rel = q.relevant(manifest)
        tied_frames, tied_groups = count_ties(s)
        row = {"qid": q.qid, "kind": q.kind, "target": q.target, "text": q.text,
               "n_frames": len(s), "n_relevant": int(rel.sum()),
               "score_mean": float(s.mean()), "score_max": float(s.max()),
               "tied_frames": tied_frames, "tied_groups": tied_groups}
        if q.kind == "negative" or rel.sum() == 0:
            # AP is undefined with R = 0; negatives are a calibration check, not mAP.
            row.update(ap=np.nan, prevalence=np.nan, lift=np.nan, r_prec=np.nan,
                       p_at_10=precision_at_k(s, rel, 10))
        else:
            ap, prev = average_precision(s, rel), random_baseline_ap(rel)
            row.update(ap=ap, prevalence=prev, lift=ap / prev, r_prec=r_precision(s, rel),
                       p_at_10=precision_at_k(s, rel, 10))
        rows.append(row)
    return pd.DataFrame(rows)
