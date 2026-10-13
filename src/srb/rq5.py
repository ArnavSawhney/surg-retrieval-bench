"""The RQ5 failure-mode frame pool (pre-registration section 4.5).

* 50 frames per Cholec80 test video (41-80), drawn uniformly without replacement within
  each video with ``default_rng(20261008)``; videos in ascending order, each video's
  frames in ascending ``frame_idx_25fps`` before the draw.
* The 2,000 pooled frames are put in a random tagging order with ``default_rng(20261009)``.
* The pool is drawn from frame identities only (video, frame index, path). It never
  sees a label, a model score or a prediction.

The pool list is written once, hashed (SHA-256 of the CSV bytes) and the hash recorded
before tagging starts.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from srb.datasets.overlap import CHOLEC80_TEST_IDS

POOL_SEED = 20261008
ORDER_SEED = 20261009
PER_VIDEO = 50
POOL_COLUMNS = ["tag_order", "draw_idx", "video_id", "frame_idx_25fps", "path"]


def draw_pool(manifest: pd.DataFrame) -> pd.DataFrame:
    """The pool in tagging order: ``tag_order`` 0..1999, ``draw_idx`` the row's position
    in the per-video draw (video 41's 50 frames first, in draw order)."""
    frames = manifest[["dataset", "video_id", "frame_idx_25fps", "path"]]
    if (frames["dataset"] != "cholec80").any():
        raise ValueError("the RQ5 pool is drawn from Cholec80 only")
    videos = sorted(CHOLEC80_TEST_IDS)
    if sorted(frames["video_id"].unique()) != videos:
        raise ValueError("the manifest must hold exactly the test videos 41-80")
    if frames.duplicated(["video_id", "frame_idx_25fps"]).any():
        raise ValueError("duplicate frames in the manifest")
    rng = np.random.default_rng(POOL_SEED)
    parts = []
    for v in videos:
        fv = frames[frames["video_id"] == v].sort_values("frame_idx_25fps")
        pick = rng.choice(len(fv), size=PER_VIDEO, replace=False)
        parts.append(fv.iloc[pick])
    pool = pd.concat(parts, ignore_index=True).drop(columns="dataset")
    pool.insert(0, "draw_idx", np.arange(len(pool)))
    order = np.random.default_rng(ORDER_SEED).permutation(len(pool))
    pool = pool.iloc[order].reset_index(drop=True)
    pool.insert(0, "tag_order", np.arange(len(pool)))
    return pool[POOL_COLUMNS]


def pool_csv(pool: pd.DataFrame) -> bytes:
    """The canonical bytes that are written and hashed."""
    return pool[POOL_COLUMNS].to_csv(index=False, lineterminator="\n").encode()


def pool_sha256(pool: pd.DataFrame) -> str:
    return hashlib.sha256(pool_csv(pool)).hexdigest()


# ---- tagging (section 4.5) ----------------------------------------------------------

POOL_SHA256 = "0655cf3ab2ff7bc4d4149a87e976e63fd8705e03c2f8f5c52dd7ca32ec73c291"  # LOG.md
ARTEFACTS = ("smoke", "blur", "glare", "blood", "occlusion")
TAGS = ARTEFACTS + ("clean",)
STOP_PER_ARTEFACT = 50  # every artefact type ("clean" does not count) has >= 50 frames
MIN_TAGGED = 300
MAX_TAGGED = 1000
TAG_COLUMNS = ["tag_order", "video_id", "frame_idx_25fps", *TAGS, "tagged_at"]


def check_tags(tags: set[str]) -> None:
    """A frame gets every artefact that applies, or "clean" alone."""
    unknown = set(tags) - set(TAGS)
    if unknown:
        raise ValueError(f"unknown tags {sorted(unknown)}")
    if not tags:
        raise ValueError("choose at least one tag (or clean)")
    if "clean" in tags and len(tags) > 1:
        raise ValueError('"clean" cannot be combined with an artefact')


def stopping_status(tagged: pd.DataFrame) -> dict:
    """Progress against the stopping rule: stop once every artefact type has at least 50
    frames and at least 300 frames are tagged, or at 1,000 frames."""
    n = len(tagged)
    counts = {t: int(tagged[t].astype(bool).sum()) if n else 0 for t in TAGS}
    all_artefacts = all(counts[a] >= STOP_PER_ARTEFACT for a in ARTEFACTS)
    if n >= MAX_TAGGED:
        reason = f"cap of {MAX_TAGGED} frames reached"
    elif all_artefacts and n >= MIN_TAGGED:
        reason = f"every artefact type has >= {STOP_PER_ARTEFACT} frames"
    else:
        reason = ""
    return {"n_tagged": n, "counts": counts, "done": bool(reason), "reason": reason}


def next_frame(pool: pd.DataFrame, tagged: pd.DataFrame) -> int | None:
    """tag_order of the next frame to tag: strictly the first untagged one in pool order
    (no skipping), or None once the stopping rule is met or the pool is exhausted."""
    done = list(tagged["tag_order"]) if len(tagged) else []
    if done != list(range(len(done))):
        raise ValueError("tags are not a prefix of the tagging order")
    if stopping_status(tagged)["done"] or len(done) >= len(pool):
        return None
    return len(done)
