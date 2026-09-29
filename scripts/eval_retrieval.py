#!/usr/bin/env python
"""DEV-MODE toy retrieval on a cached index (Week 1: videos 1-5 only).

    python scripts/eval_retrieval.py --model clip-vit-l14 --videos 1-5

Writes results/dev/<model>_<videos>_<timestamp>.csv. These are **dev sanity checks on
training videos, not results**: they check that the pipeline runs and that nothing is
obviously broken (e.g. AP ~ prevalence for every query, or a negative control that
scores like a real query). They never go to results/results.csv.

The queries below are a dev smoke set, not the pre-registered query list.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from srb.index import load_index
from srb.models.registry import _SPECS, get_backbone
from srb.retrieval import Query, evaluate, scores_for

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import video_tag  # noqa: E402
from fetch_cholec80_videos import parse_videos, refuse_test_videos  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dev_label(manifest: pd.DataFrame) -> str:
    """Banner naming the videos and split actually evaluated (never a result)."""
    ids = sorted(manifest.video_id.unique())
    if len(ids) == 1:
        vids = f"video {ids[0]}"
    elif ids == list(range(ids[0], ids[-1] + 1)):
        vids = f"videos {ids[0]}-{ids[-1]}"
    else:
        vids = "videos " + ",".join(map(str, ids))
    split = "/".join(sorted(manifest.split.unique()))
    return f"DEV, {vids} ({split} split), NOT A RESULT"

DEV_QUERIES = [
    Query("phase_calot", "dissection of Calot's triangle in a laparoscopic cholecystectomy",
          "phase", "CalotTriangleDissection"),
    Query("phase_clipping", "clipping and cutting the cystic duct and artery",
          "phase", "ClippingCutting"),
    Query("tool_hook", "a hook electrode in the image", "tool", "tool_hook"),
    Query("neg_car", "a car on a road", "negative", None),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(_SPECS))
    ap.add_argument("--videos", required=True)
    args = ap.parse_args()
    videos = parse_videos(args.videos)
    refuse_test_videos(videos)

    emb, manifest, meta = load_index(ROOT / "data" / "index" / args.model / video_tag(videos))
    if meta["revision"] != _SPECS[args.model].revision:
        sys.exit("index was built with a different model revision; rebuild it")
    backbone = get_backbone(args.model)
    text = backbone.encode_text([q.text for q in DEV_QUERIES])
    scores = scores_for(text, emb)  # float32
    table = evaluate(DEV_QUERIES, scores, manifest)
    label = dev_label(manifest)
    table.insert(0, "label", label)
    table.insert(1, "model", args.model)
    table["index_git_hash"] = meta["git_hash"]

    out = ROOT / "results" / "dev"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model}_{video_tag(videos)}_{dt.datetime.now():%Y%m%d-%H%M%S}.csv"
    table.to_csv(path, index=False)

    print(f"\n*** {label} ***")
    cols = ["qid", "n_relevant", "ap", "prevalence", "lift", "r_prec", "p_at_10",
            "tied_frames", "tied_groups", "score_mean", "score_max"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table[cols].to_string(index=False))
    print(f"*** {label} ***\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
