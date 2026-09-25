"""Why retrieval uses AP / R-Precision instead of Recall@k (Cholec80-scale simulation).

Class sizes are approximate: EndoNet (Twinanda et al., IEEE TMI 2017) mean phase
durations in seconds x 40 test videos at 1 fps. Replace with real label counts once
the annotations are parsed. Simulated scores, NOT model results.

Run: python docs/metric_rationale.py
"""
import numpy as np
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(0)
MEAN_DUR_S = {  # EndoNet table, mean seconds per video
    "Preparation": 125, "CalotTriangleDissection": 954, "ClippingCutting": 168,
    "GallbladderDissection": 857, "GallbladderPackaging": 98,
    "CleaningCoagulation": 178, "GallbladderRetraction": 83,
}
counts = {k: v * 40 for k, v in MEAN_DUR_S.items()}
N = sum(counts.values())


def metrics(scores, rel, ks=(1, 10, 50)):
    order = np.argsort(-scores, kind="stable")
    r = rel[order]
    R = int(rel.sum())
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = r[:k].sum() / R
        out[f"Hit@{k}"] = float(r[:k].any())
        out[f"P@{k}"] = r[:k].mean()
    disc = 1 / np.log2(np.arange(2, 12))
    out["nDCG@10"] = (r[:10] * disc).sum() / disc[: min(10, R)].sum()
    out["R-Prec"] = r[:R].mean()
    hits = np.cumsum(r)
    out["AP"] = (hits[r == 1] / (np.flatnonzero(r) + 1)).mean()
    assert abs(out["AP"] - average_precision_score(rel, scores)) < 1e-6
    out["prior"] = R / N
    return out


def model_scores(rel, kind):
    n = len(rel)
    if kind == "random":
        return rng.random(n)
    if kind == "perfect":
        return rel + 0.01 * rng.random(n)
    if kind.startswith("gauss"):  # relevant ~ N(d,1), irrelevant ~ N(0,1)
        return rng.normal(0, 1, n) + float(kind.split("_")[1]) * rel
    if kind == "shortcut":  # one 50-frame clip ranked on top, rest random
        s = rng.random(n)
        s[rng.choice(np.flatnonzero(rel), 50, replace=False)] = 2.0
        return s
    raise ValueError(kind)


cols = ["prior", "Recall@10", "Recall@50", "Hit@1", "Hit@10", "P@10", "nDCG@10", "R-Prec", "AP"]
print(f"N test frames ~ {N}")
for cls in ["CalotTriangleDissection", "ClippingCutting", "GallbladderRetraction"]:
    R = counts[cls]
    rel = np.zeros(N, dtype=int)
    rel[:R] = 1
    print(f"\n== query class {cls}: R={R} relevant frames ==")
    print(f"{'model':<10}" + "".join(f"{c:>11}" for c in cols))
    for kind in ["random", "gauss_0.5", "gauss_1.0", "gauss_2.0", "shortcut", "perfect"]:
        m = metrics(model_scores(rel, kind), rel)
        print(f"{kind:<10}" + "".join(f"{m[c]:>11.4f}" for c in cols))

print("\nRandom ranker Hit@10 = 1-(1-p)^10, and the ceiling on Recall@10 = 10/R:")
for cls, R in counts.items():
    p = R / N
    print(f"  {cls:<26} p={p:.3f}  Hit@10={1-(1-p)**10:.3f}  max Recall@10={10/R:.5f}")
