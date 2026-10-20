"""Endoscapes2023 label reader, CVS relevance rule and test-frame manifest.

Source files (inside the official archive, under ``endoscapes/``; README.md there):

* ``all_metadata.csv``: one row per released 1 fps frame (58,813). Columns used here:
  ``vid``, ``frame`` (the frame's 25 fps index; the JPEG is ``{vid}_{frame}.jpg``),
  ``C1``-``C3`` (mean of the three annotators), ``is_ds_keyframe`` (the 11,090 frames
  annotated for CVS by 3 experts, 5 s apart) and ``cvs_annotator_1..3`` (each a list
  ``[C1, C2, C3]`` of 0/1, present on keyframes only).
* ``train_vids.txt`` / ``val_vids.txt`` / ``test_vids.txt``: the official splits, one
  video ID per line, written as floats (``1.620000000000000000e+02``).

Protocol (pre-registration 1.2 and the Deviations entry of 2026-10-20):

* **Evaluated frames:** the CVS keyframes (``is_ds_keyframe``) of the test videos. The
  ``C1``-``C3`` values of non-keyframes are derived, not annotated, and are never used.
* **Relevance:** frame relevant to C*k* iff at least 2 of the 3 annotators marked C*k*
  achieved. Computed from the per-annotator votes, and asserted equal to the released
  mean ``Ck >= 0.5`` on every frame.
* **Unanimity** (secondary, not registered): relevant iff all 3 marked it.

Nothing here is trained or tuned on; Endoscapes is a test set only (section 1.2).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from srb.datasets.overlap import ENDOSCAPES_TO_CHOLEC80

DATASET = "endoscapes"
FPS = 25
CRITERIA = ("C1", "C2", "C3")
ANNOTATOR_COLUMNS = ("cvs_annotator_1", "cvs_annotator_2", "cvs_annotator_3")
SPLITS = ("train", "val", "test")
MEMBER_PREFIX = "endoscapes/"
# Label, split and metadata files, fetched before any image (Task 1, 2026-10-20).
LABEL_MEMBERS = ("README.md", "LICENSE", "all_metadata.csv", "train_vids.txt",
                 "val_vids.txt", "test_vids.txt", "test/annotation_coco.json",
                 "test/annotation_coco_vid.json", "test/annotation_ds_coco.json")
# Released frame size (pixels, either orientation); anything else stops the fetch.
RELEASED_SIZES = frozenset({(854, 480), (480, 854)})

# Manifest columns: the Cholec80 key columns first, then CVS labels.
VOTE_COLUMNS = tuple(f"{c.lower()}_votes" for c in CRITERIA)        # 0..3
RELEVANT_COLUMNS = tuple(c.lower() for c in CRITERIA)               # majority (registered)
UNANIMOUS_COLUMNS = tuple(f"{c.lower()}_unanimous" for c in CRITERIA)  # 3 of 3 (secondary)
MANIFEST_COLUMNS = ("dataset", "video_id", "split", "frame_idx_25fps", "t_sec", "path",
                    "member", *VOTE_COLUMNS, *RELEVANT_COLUMNS, *UNANIMOUS_COLUMNS)


def read_split_ids(path) -> list[int]:
    """Video IDs of one split file, as ints (the file writes them as floats)."""
    ids = []
    for line in Path(path).read_text().split():
        x = float(line)
        if x != int(x) or x <= 0:
            raise ValueError(f"{path}: {line!r} is not a positive integer video ID")
        ids.append(int(x))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate video IDs")
    return sorted(ids)


def read_splits(label_dir) -> dict[str, list[int]]:
    """{'train', 'val', 'test'} -> sorted video IDs; the three must be disjoint."""
    splits = {s: read_split_ids(Path(label_dir) / f"{s}_vids.txt") for s in SPLITS}
    seen: set[int] = set()
    for s, ids in splits.items():
        if seen & set(ids):
            raise ValueError(f"split {s} shares videos {sorted(seen & set(ids))} with another")
        seen |= set(ids)
    return splits


def parse_votes(cell) -> tuple[int, int, int]:
    """One annotator's ``"[c1, c2, c3]"`` cell -> three 0/1 ints. Anything else raises."""
    if not isinstance(cell, str):
        raise ValueError(f"missing annotator vote: {cell!r}")
    v = ast.literal_eval(cell)
    if not (isinstance(v, list) and len(v) == 3 and all(x in (0, 1) for x in v)):
        raise ValueError(f"annotator vote must be a list of three 0/1 values, got {cell!r}")
    return tuple(int(x) for x in v)


def cvs_labels(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-keyframe votes and relevance, from the per-annotator columns.

    Takes ``all_metadata.csv`` rows (any subset) and returns its **keyframes** only, with
    ``vid``, ``frame``, ``c*_votes`` (0-3), ``c*`` (majority, >= 2 of 3) and
    ``c*_unanimous`` (3 of 3). Raises if a keyframe lacks an annotator or a vote is not
    0/1, if the votes' mean differs from the released ``Ck``, or if the majority rule
    differs from ``Ck >= 0.5`` on any frame.
    """
    k = meta[meta["is_ds_keyframe"].astype(bool)]
    votes = np.zeros((len(k), 3), dtype=np.int64)
    for col in ANNOTATOR_COLUMNS:
        votes += np.array([parse_votes(c) for c in k[col]], dtype=np.int64).reshape(-1, 3)
    released = k[list(CRITERIA)].to_numpy(dtype=np.float64)
    if not np.allclose(votes / 3.0, released, rtol=0, atol=1e-9):
        bad = np.flatnonzero(~np.isclose(votes / 3.0, released, rtol=0, atol=1e-9).all(1))
        raise ValueError(f"annotator mean != released C1-C3 on {len(bad)} keyframes, "
                         f"e.g. vid {k['vid'].iloc[bad[0]]} frame {k['frame'].iloc[bad[0]]}")
    majority = votes >= 2
    if not (majority == (released >= 0.5)).all():
        raise ValueError("majority vote (>= 2 of 3) differs from the released mean >= 0.5")
    out = pd.DataFrame({"vid": k["vid"].astype("int64").to_numpy(),
                        "frame": k["frame"].astype("int64").to_numpy()})
    for j in range(3):
        out[VOTE_COLUMNS[j]] = votes[:, j]
    for j in range(3):
        out[RELEVANT_COLUMNS[j]] = majority[:, j]
    for j in range(3):
        out[UNANIMOUS_COLUMNS[j]] = votes[:, j] == 3
    return out


def frame_name(video_id: int, frame: int) -> str:
    return f"{video_id}_{frame}.jpg"


def build_manifest(label_dir, split: str = "test") -> pd.DataFrame:
    """Manifest of the CVS keyframes of one official split, sorted by video then frame.

    ``path`` is relative to ``data/endoscapes/`` (the resized 448 px frame);
    ``member`` is the released JPEG's name inside the archive.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    label_dir = Path(label_dir)
    ids = read_splits(label_dir)[split]
    meta = pd.read_csv(label_dir / "all_metadata.csv")
    lab = cvs_labels(meta[meta["vid"].isin(ids)])
    if (lab["frame"] % FPS).any():
        raise ValueError("a keyframe's frame number is not a multiple of 25")
    missing = sorted(set(ids) - set(lab["vid"]))
    if missing:
        raise ValueError(f"split {split}: videos without CVS keyframes {missing}")
    lab = lab.sort_values(["vid", "frame"]).reset_index(drop=True)
    vid, frame = lab["vid"].to_numpy(), lab["frame"].to_numpy()
    man = pd.DataFrame({
        "dataset": DATASET,
        "video_id": vid,
        "split": split,
        "frame_idx_25fps": frame,
        "t_sec": frame / FPS,
        "path": [f"frames/{v}/{frame_name(v, f)}" for v, f in zip(vid, frame)],
        "member": [f"{MEMBER_PREFIX}{split}/{frame_name(v, f)}" for v, f in zip(vid, frame)],
    })
    for col in (*VOTE_COLUMNS, *RELEVANT_COLUMNS, *UNANIMOUS_COLUMNS):
        man[col] = lab[col].to_numpy()
    if man.duplicated(["video_id", "frame_idx_25fps"]).any():
        raise ValueError("duplicate (video, frame) in the manifest")
    return man[list(MANIFEST_COLUMNS)]


def cholec80_overlap(video_ids) -> dict[int, int]:
    """Endoscapes video ID -> Cholec80 video ID, for the given IDs that are Cholec80
    videos (CAMMA ``camma_dataset_overlaps`` @ 8347b9f). Empty for the official test split."""
    return {int(v): ENDOSCAPES_TO_CHOLEC80[int(v)] for v in video_ids
            if int(v) in ENDOSCAPES_TO_CHOLEC80}
