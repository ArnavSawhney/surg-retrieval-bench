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
    assert set(s) == {"next", "pool_size", "status", "n_out_of_body", "review_pending",
                      "artefacts", "tags", "per_artefact", "min",
                      "max"}  # nothing about the frame beyond its position
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


# ---- out of body (tagging guide, change log 2026-10-18) -------------------------------

def _tagger(tmp_path, n=6, **kw):
    pool = _pool(tmp_path, list(range(1, n + 1)))
    return pool, tag_rq5.Tagger(pool, tmp_path / "tags.csv", tmp_path, demo=True, **kw)


def test_out_of_body_key_writes_clean_plus_a_separate_flag(tmp_path):
    pool, t = _tagger(tmp_path)
    assert t.oob_path == tmp_path / "out_of_body.csv"
    with pytest.raises(ValueError, match="only be combined with blur"):
        t.save(0, ["occlusion"], out_of_body=True)
    t.save(0, [], out_of_body=True)
    t.save(1, ["blur"], out_of_body=True)
    t.save(2, ["occlusion"])
    tags = pd.read_csv(tmp_path / "tags.csv")
    assert list(tags.columns) == list(tag_rq5.TAG_COLUMNS)  # schema unchanged
    assert tags[["blur", "occlusion", "clean"]].values.tolist() == [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    oob = pd.read_csv(tmp_path / "out_of_body.csv")
    assert oob[["tag_order", "out_of_body", "source"]].values.tolist() == [
        [0, 1, "tagging"], [1, 1, "tagging"]]
    s = t.state()
    assert s["n_out_of_body"] == 2 and s["status"]["counts"]["occlusion"] == 1
    t.undo()  # frame 2: no out-of-body row to remove
    t.undo()  # frame 1: its out-of-body row goes too
    assert pd.read_csv(tmp_path / "out_of_body.csv")["tag_order"].tolist() == [0]
    assert t.state()["next"] == 1


def test_crash_between_the_two_writes_is_reconciled(tmp_path):
    pool, t = _tagger(tmp_path)
    t.save(0, ["smoke"])
    tag_rq5._append(t.oob_path, [1, 1, "tagging", "x"])  # out-of-body row, no tags row
    t2 = tag_rq5.Tagger(pool, tmp_path / "tags.csv", tmp_path, demo=True)
    assert len(t2.out_of_body()) == 0 and t2.state()["next"] == 1


def test_review_is_image_only_in_order_resumable_with_undo(tmp_path):
    pool, t = _tagger(tmp_path)
    for i in range(4):
        t.save(i, ["occlusion"])
    pool, t = _tagger(tmp_path, review_n=3)
    r = t.review_state()
    assert r == {"next": 0, "n_reviewed": 0, "n_total": 3}  # no tags, labels or scores
    assert t.state()["next"] is None and t.state()["review_pending"]  # tagging waits
    assert t.review_image(0).endswith(b"\x00")
    with pytest.raises(PermissionError):
        t.review_image(1)
    with pytest.raises(ValueError):
        t.review_answer(1, True)
    t.review_answer(0, True)
    t.review_answer(1, True)
    t.review_undo()
    t2 = tag_rq5.Tagger(pool, tmp_path / "tags.csv", tmp_path, demo=True, review_n=3)
    assert t2.review_state()["next"] == 1  # resumes from the file
    t2.review_answer(1, False)
    t2.review_answer(2, True)
    assert t2.review_state()["next"] is None
    oob = pd.read_csv(t2.oob_path)
    assert oob[["tag_order", "out_of_body", "source"]].values.tolist() == [
        [0, 1, "review"], [1, 0, "review"], [2, 1, "review"]]
    # the stopping counter uses the corrected tags: frames 0 and 2 lose occlusion
    s = t2.state()
    assert s["status"]["counts"]["occlusion"] == 2 and s["status"]["counts"]["clean"] == 2
    assert s["next"] == 4 and s["n_out_of_body"] == 2
    assert pd.read_csv(tmp_path / "tags.csv")["occlusion"].tolist() == [1, 1, 1, 1]  # raw kept
    t2.undo()  # frame 3 is after the reviewed frames
    with pytest.raises(ValueError, match="frozen"):
        t2.undo()


def test_review_needs_the_frames_tagged(tmp_path):
    pool, t = _tagger(tmp_path)
    t.save(0, ["blur"])
    with pytest.raises(SystemExit, match="not all tagged"):
        tag_rq5.Tagger(pool, tmp_path / "tags.csv", tmp_path, demo=True, review_n=2).review_state()


def test_registered_mode_refuses_changed_raw_tags(tmp_path, monkeypatch):
    pool, t = _tagger(tmp_path)
    monkeypatch.setattr(tag_rq5, "POOL_SHA256",
                        __import__("hashlib").sha256(pool.read_bytes()).hexdigest())
    monkeypatch.setattr(tag_rq5, "CHOLEC80_TEST_IDS", set(range(1, 81)))
    with pytest.raises(SystemExit, match="frozen raw tags"):
        tag_rq5.Tagger(pool, tmp_path / "tags.csv", tmp_path, demo=False)


# ---- scripts/rq5_apply_corrections.py ----------------------------------------------

spec2 = importlib.util.spec_from_file_location(
    "rq5_apply_corrections", SCRIPT.parent / "rq5_apply_corrections.py")
corr = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(corr)


def test_apply_corrections_script(tmp_path, monkeypatch):
    import hashlib
    pool, t = _tagger(tmp_path)
    t.save(0, ["occlusion", "smoke"])
    t.save(1, ["occlusion", "blur", "glare"])
    t.save(2, ["blood"])
    raw = (tmp_path / "tags.csv").read_bytes()
    monkeypatch.setattr(corr, "REVIEWED_FRAMES", 2)
    out = tmp_path / "tags_corrected.csv"
    monkeypatch.setattr(corr, "RAW_TAGS_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="frozen raw"):
        corr.build(tmp_path / "tags.csv", t.oob_path, out)
    from srb.rq5 import raw_prefix_sha256
    monkeypatch.setattr(corr, "RAW_TAGS_SHA256", raw_prefix_sha256(raw, 2))
    with pytest.raises(SystemExit, match="finish it first"):
        corr.build(tmp_path / "tags.csv", t.oob_path, out)
    t2 = tag_rq5.Tagger(pool, tmp_path / "tags.csv", tmp_path, demo=True, review_n=2)
    t2.review_answer(0, True)
    t2.review_answer(1, True)
    t2.save(3, [], out_of_body=True)
    df = corr.build(tmp_path / "tags.csv", t2.oob_path, out)
    assert (tmp_path / "tags.csv").read_bytes()[:len(raw)] == raw  # raw only appended to
    cols = ["smoke", "blur", "glare", "blood", "occlusion", "clean", "out_of_body"]
    assert df[cols].values.tolist() == [[0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0, 1],
                                        [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1]]
    assert pd.read_csv(out)[cols].values.tolist() == df[cols].values.tolist()
    assert hashlib.sha256(raw).hexdigest() != hashlib.sha256(out.read_bytes()).hexdigest()


def test_real_raw_tags_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    p = root / "data/rq5/tags.csv"
    if not p.exists():
        pytest.skip("data/rq5/tags.csv is not tracked")
    from srb.rq5 import RAW_TAGS_SHA256, raw_prefix_sha256
    assert raw_prefix_sha256(p.read_bytes()) == RAW_TAGS_SHA256
