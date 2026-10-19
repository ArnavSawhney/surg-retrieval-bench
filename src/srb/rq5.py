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


# Out of body (tagging guide, change log 2026-10-18): the camera is outside the patient.
# Not an artefact: the frame is "clean" (section 4.5's "none") plus a flag kept in a
# separate file, so the tags file keeps its schema. Blur may be added; smoke, glare,
# blood and occlusion are only seen inside the abdomen.
OUT_OF_BODY_COLUMNS = ["tag_order", "out_of_body", "source", "timestamp"]
OUT_OF_BODY_SOURCES = ("tagging", "review")
NOT_WITH_OUT_OF_BODY = ("smoke", "glare", "blood", "occlusion")
# Frames 0-435 were tagged before the out-of-body key existed; an image-only review
# pass records out of body yes/no for each. Their raw rows are frozen (LOG.md):
REVIEWED_FRAMES = 436
RAW_TAGS_SHA256 = "750483857807963d270812ab1696242186d6e105004772d83ec5070f50d732a6"


def check_tags(tags: set[str], out_of_body: bool = False) -> None:
    """A frame gets every artefact that applies, or "clean" alone; an out-of-body frame
    gets nothing or blur."""
    unknown = set(tags) - set(TAGS)
    if unknown:
        raise ValueError(f"unknown tags {sorted(unknown)}")
    if out_of_body:
        if set(tags) - {"blur"}:
            raise ValueError("out of body can only be combined with blur")
        return
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


def tag_values(tags: set[str], out_of_body: bool = False) -> dict[str, int]:
    """The 0/1 values of the tags-file columns: out of body is clean unless blurred."""
    check_tags(tags, out_of_body)
    row = {t: int(t in tags) for t in TAGS}
    if out_of_body:
        row["clean"] = int(not row["blur"])
    return row


def raw_prefix_sha256(raw: bytes, n: int = REVIEWED_FRAMES) -> str:
    """SHA-256 of the header and first n rows of a tags file."""
    lines = raw.splitlines(keepends=True)
    if len(lines) < n + 1:
        raise ValueError(f"the tags file has fewer than {n} rows")
    return hashlib.sha256(b"".join(lines[:n + 1])).hexdigest()


def out_of_body_flags(oob: pd.DataFrame) -> pd.Series:
    """out_of_body (0/1) indexed by tag_order; at most one answer per frame."""
    if list(oob.columns) != OUT_OF_BODY_COLUMNS:
        raise ValueError("unexpected out-of-body columns")
    if not oob["source"].isin(OUT_OF_BODY_SOURCES).all():
        raise ValueError("unknown out-of-body source")
    if not oob["out_of_body"].isin([0, 1]).all():
        raise ValueError("out_of_body must be 0 or 1")
    if oob["tag_order"].duplicated().any():
        raise ValueError("more than one out-of-body answer for a frame")
    return oob.set_index("tag_order")["out_of_body"].astype(int)


def reviewed_prefix(oob: pd.DataFrame) -> int:
    """How many of frames 0, 1, ... have a review answer (answers are given in order)."""
    rev = list(oob.loc[oob["source"] == "review", "tag_order"])
    if rev != list(range(len(rev))):
        raise ValueError("review answers are not a prefix of the tagging order")
    return len(rev)


def apply_corrections(tags: pd.DataFrame, oob: pd.DataFrame) -> pd.DataFrame:
    """Corrected tags: for out-of-body frames smoke, glare, blood and occlusion are 0,
    blur is kept, and clean is 1 unless blur is set. Adds an out_of_body column."""
    flags = out_of_body_flags(oob)
    unknown = set(flags.index) - set(tags["tag_order"])
    if unknown:
        raise ValueError(f"out-of-body answers for untagged frames {sorted(unknown)[:5]}")
    out = tags.copy()
    m = out["tag_order"].map(flags).fillna(0).astype(int).to_numpy() == 1
    out.loc[m, list(NOT_WITH_OUT_OF_BODY)] = 0
    out.loc[m, "clean"] = (out.loc[m, "blur"] == 0).astype(int)
    out["out_of_body"] = m.astype(int)
    return out


# ---- per-tag analysis (section 4.5; Deviations 2026-10-18 and 2026-10-19) -------------

# The frozen tag files (LOG.md, 2026-10-19); the analysis reads nothing else from data/rq5.
FROZEN_SHA256 = {
    "tags.csv": "c741a3f2e4da25b2ac0e79f6c13d65c2da46151cae9c60279c837b19e6881582",
    "out_of_body.csv": "3d4516d749b0d4099a1bd78901c0dcff57519e20b965558f5cc9bdc04228aaea",
    "tags_corrected.csv": "8be085f776c218080141409e6a935fd325a4bd9b7baf06ec5a4a4f93b1d34ef1",
}
MIN_FRAMES_FOR_ACCURACY = 20  # fewer tagged frames: counts only (section 4.5)


def check_frozen(rq5_dir, expected: dict[str, str] = FROZEN_SHA256) -> None:
    """Every frozen file must hash to its logged SHA-256."""
    from pathlib import Path

    for name, sha in expected.items():
        got = hashlib.sha256((Path(rq5_dir) / name).read_bytes()).hexdigest()
        if got != sha:
            raise ValueError(f"{name}: SHA-256 {got} differs from the frozen {sha}")


def per_tag_accuracy(tags: pd.DataFrame, correct: np.ndarray, videos: np.ndarray,
                     counts: np.ndarray) -> dict[str, dict]:
    """Accuracy on the frames of each tag (a frame counts in every one of its tags).

    ``correct`` is (N,) or (N, R) 0/1, row-aligned with ``tags``; with R classifiers
    the accuracy is the mean over them of each one's pooled accuracy (the zero-shot
    headline over r1-r5). ``counts`` is the (B, V) multiplicity matrix over ``videos``
    (sorted). Returns, per tag: n frames, n videos, the point accuracy and its (B,)
    resampled values; NaN when fewer than MIN_FRAMES_FOR_ACCURACY frames are tagged,
    and NaN in a resample that draws none of the tag's frames.
    """
    c = np.asarray(correct, dtype=np.float64)
    c = c[:, None] if c.ndim == 1 else c
    if len(c) != len(tags):
        raise ValueError("correct is not row-aligned with the tags")
    vid = tags["video_id"].to_numpy()
    if not np.isin(vid, videos).all():
        raise ValueError("tagged frames from videos outside the resampled set")
    vpos = np.searchsorted(videos, vid)
    V, m = len(videos), counts.astype(np.float64)
    out = {}
    for t in TAGS:
        sel = tags[t].to_numpy() == 1
        n = int(sel.sum())
        n_v = np.bincount(vpos[sel], minlength=V).astype(np.float64)
        c_v = np.stack([np.bincount(vpos[sel], weights=c[sel, r], minlength=V)
                        for r in range(c.shape[1])])  # (R, V)
        if n < MIN_FRAMES_FOR_ACCURACY:
            value, boot = np.nan, np.full(len(m), np.nan)
        else:
            value = float((c_v.sum(1) / n).mean())
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = ((m @ c_v.T) / (m @ n_v)[:, None]).mean(1)
        out[t] = {"n": n, "n_videos": int((n_v > 0).sum()), "value": value, "boot": boot}
    return out


def drops(acc: dict[str, dict]) -> dict[str, dict]:
    """Drop per artefact: accuracy on clean frames minus accuracy on the tag's frames
    (positive = worse on the tag), point and per resample (Deviations 2026-10-19)."""
    cl = acc["clean"]
    return {a: {"value": cl["value"] - acc[a]["value"], "boot": cl["boot"] - acc[a]["boot"]}
            for a in ARTEFACTS}


def mean_drop(d: dict[str, dict]) -> dict:
    """Unweighted mean of the five artefacts' drops (RQ5c)."""
    return {"value": float(np.mean([d[a]["value"] for a in ARTEFACTS])),
            "boot": np.mean([d[a]["boot"] for a in ARTEFACTS], 0)}
