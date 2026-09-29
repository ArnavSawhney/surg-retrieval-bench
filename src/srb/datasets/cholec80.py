"""Cholec80 label reader and 1 fps label alignment.

File formats (from the dataset's README.txt, and checked on videos 1-5):

* ``phase_annotations/videoXX-phase.txt``: header ``Frame<TAB>Phase``, then one row
  per frame of the 25 fps video, 0-based frame index, contiguous from 0.
* ``tool_annotations/videoXX-tool.txt``: header ``Frame<TAB>Grasper<TAB>...``, then
  one row per second: frame indices 0, 25, 50, ... (multiples of 25), 7 binary columns.

Alignment rule (closes open decision #3)
    A 1 fps sample is a 25 fps frame index that is a multiple of 25, so phase and tool
    labels are joined on the **frame index**, never on a timestamp. Every tool index
    must exist in the phase table.

    End-of-video off-by-one: on every video checked so far (1-5) the last phase frame
    is itself a multiple of 25 (e.g. video01: 43325 = 1733 x 25) but the tool file
    stops one sample earlier (43300). That trailing sample has a phase and no tool
    vector. It is **dropped explicitly and counted** (``AlignedLabels.dropped_tail``);
    any other gap raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FPS = 25

PHASES = (
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
)
TOOLS = ("Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag")
# Manifest column names for the tool booleans.
TOOL_COLUMNS = tuple(f"tool_{t.lower()}" for t in TOOLS)


def phase_path(root: Path, video_id: int) -> Path:
    return Path(root) / "phase_annotations" / f"video{video_id:02d}-phase.txt"


def tool_path(root: Path, video_id: int) -> Path:
    return Path(root) / "tool_annotations" / f"video{video_id:02d}-tool.txt"


def read_phases(path) -> pd.DataFrame:
    """Phase table -> DataFrame ``frame_idx`` (int64), ``phase`` (str). Validated."""
    df = pd.read_csv(path, sep="\t", dtype={"Frame": "int64", "Phase": "string"})
    if list(df.columns) != ["Frame", "Phase"]:
        raise ValueError(f"{path}: unexpected phase header {list(df.columns)}")
    df = df.rename(columns={"Frame": "frame_idx", "Phase": "phase"})
    if not (df["frame_idx"].to_numpy() == range(len(df))).all():
        raise ValueError(f"{path}: phase frame indices are not contiguous from 0")
    unknown = set(df["phase"]) - set(PHASES)
    if unknown:
        raise ValueError(f"{path}: non-canonical phase names {sorted(unknown)}")
    df["phase"] = df["phase"].astype(str)
    return df


def read_tools(path) -> pd.DataFrame:
    """Tool table -> DataFrame ``frame_idx`` + one bool column per tool. Validated."""
    df = pd.read_csv(path, sep="\t")
    if list(df.columns) != ["Frame", *TOOLS]:
        raise ValueError(f"{path}: unexpected tool header {list(df.columns)}")
    idx = df["Frame"].to_numpy()
    if not (idx == range(0, FPS * len(df), FPS)).all():
        raise ValueError(f"{path}: tool frame indices are not 0, 25, 50, ...")
    vals = df[list(TOOLS)]
    if not vals.isin([0, 1]).all().all():
        raise ValueError(f"{path}: tool labels must be 0/1")
    out = pd.DataFrame({"frame_idx": idx.astype("int64")})
    for t, col in zip(TOOLS, TOOL_COLUMNS):
        out[col] = vals[t].astype(bool).to_numpy()
    return out


@dataclass
class AlignedLabels:
    labels: pd.DataFrame  # frame_idx, t_sec, phase, tool_* ; one row per 1 fps sample
    n_phase_frames: int  # rows in the 25 fps phase file
    n_samples_1fps: int  # multiples of 25 inside the phase file
    dropped_tail: list[int]  # 1 fps samples with a phase but no tool vector (end only)


def align_1fps(phases: pd.DataFrame, tools: pd.DataFrame) -> AlignedLabels:
    """Join phase and tool labels on 25 fps frame index, at every 25th frame."""
    phase_idx = set(phases["frame_idx"])
    missing = [i for i in tools["frame_idx"] if i not in phase_idx]
    if missing:
        raise ValueError(f"tool frame indices absent from the phase labels: {missing[:5]}")

    samples = phases[phases["frame_idx"] % FPS == 0]
    merged = samples.merge(tools, on="frame_idx", how="left", indicator=True)
    no_tool = merged.loc[merged["_merge"] == "left_only", "frame_idx"].tolist()
    last = int(samples["frame_idx"].max())
    if no_tool and (len(no_tool) > 1 or no_tool[0] != last):
        raise ValueError(f"1 fps samples without tool labels that are not the single "
                         f"final sample: {no_tool[:5]}")
    labels = merged[merged["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    labels[list(TOOL_COLUMNS)] = labels[list(TOOL_COLUMNS)].astype(bool)
    labels.insert(1, "t_sec", labels["frame_idx"] / FPS)
    return AlignedLabels(labels, len(phases), len(samples), no_tool)


def load_video_labels(root: Path, video_id: int) -> AlignedLabels:
    return align_1fps(read_phases(phase_path(root, video_id)),
                      read_tools(tool_path(root, video_id)))
