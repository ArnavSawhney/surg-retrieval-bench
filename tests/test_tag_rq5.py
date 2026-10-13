"""scripts/tag_rq5.py: order, image-only, save-every-frame/resume, and the pool guards."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from srb.rq5 import POOL_COLUMNS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tag_rq5.py"
spec = importlib.util.spec_from_file_location("tag_rq5", SCRIPT)
tag_rq5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tag_rq5)


def _pool(tmp_path, videos):
    rows = []
    for i, v in enumerate(videos):
        rel = f"frames/video{v:02d}/{25 * i:06d}.jpg"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"\xff\xd8jpeg" + bytes([i]))
        rows.append({"tag_order": i, "draw_idx": i, "video_id": v, "frame_idx_25fps": 25 * i,
                     "path": rel})
    p = tmp_path / "pool.csv"
    pd.DataFrame(rows)[POOL_COLUMNS].to_csv(p, index=False, lineterminator="\n")
    return p


def test_demo_tags_in_order_and_resumes(tmp_path):
    pool = _pool(tmp_path, [1, 2, 3, 4, 5])
    tags = tmp_path / "tags.csv"
    t = tag_rq5.Tagger(pool, tags, tmp_path, demo=True)
    s = t.state()
    assert s["next"] == 0 and s["status"]["n_tagged"] == 0
    assert set(s) == {"next", "pool_size", "status", "artefacts", "tags", "per_artefact",
                      "min", "max"}  # nothing about the frame beyond its position
    assert t.image(0).endswith(b"\x00")
    with pytest.raises(PermissionError):
        t.image(1)  # only the next frame
    with pytest.raises(ValueError):
        t.save(1, ["blur"])  # no skipping
    with pytest.raises(ValueError):
        t.save(0, ["clean", "blur"])
    t.save(0, ["smoke", "blur"])
    t.save(1, ["clean"])
    # a new session resumes from the file
    t2 = tag_rq5.Tagger(pool, tags, tmp_path, demo=True)
    assert t2.state()["next"] == 2
    df = pd.read_csv(tags)
    assert df[["smoke", "blur", "clean"]].values.tolist() == [[1, 1, 0], [0, 0, 1]]
    assert df.video_id.tolist() == [1, 2]
    t2.undo()
    assert t2.state()["next"] == 1
    for i in range(1, 5):
        t2.save(i, ["glare"])
    assert t2.state()["next"] is None


def test_demo_refuses_non_training_frames(tmp_path):
    for v in (33, 41):
        with pytest.raises(SystemExit, match="training videos"):
            tag_rq5.Tagger(_pool(tmp_path, [1, v]), tmp_path / "t.csv", tmp_path, demo=True)


def test_registered_mode_checks_the_pool_hash(tmp_path):
    with pytest.raises(SystemExit, match="SHA-256"):
        tag_rq5.Tagger(_pool(tmp_path, [41, 42]), tmp_path / "t.csv", tmp_path, demo=False)


def test_registered_pool_file_matches_the_logged_hash():
    root = Path(__file__).resolve().parents[1]
    import hashlib
    from srb.rq5 import POOL_SHA256
    assert hashlib.sha256((root / "docs/rq5_frame_pool.csv").read_bytes()).hexdigest() == POOL_SHA256


def test_tags_file_must_match_the_pool(tmp_path):
    pool = _pool(tmp_path, [1, 2])
    tags = tmp_path / "tags.csv"
    tag_rq5.Tagger(pool, tags, tmp_path, demo=True).save(0, ["blood"])
    tags.write_text(tags.read_text().replace(",1,0,", ",2,0,", 1))  # wrong video ID
    with pytest.raises(SystemExit, match="do not match"):
        tag_rq5.Tagger(pool, tags, tmp_path, demo=True)
