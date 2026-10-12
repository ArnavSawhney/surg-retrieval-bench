"""Index build/cache/load and the retrieval evaluation, with a fake backbone."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from srb.index import (cache_is_valid, embed_frames, git_hash, load_index, manifest_hash,
                       merge_indexes, save_index)
from srb.retrieval import Query, evaluate, scores_for


class FakeBackbone:
    """Embeds an image as its unit-normalised mean RGB (deterministic, no weights)."""

    def encode_image(self, images, batch_size=None):
        v = np.array([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) + 1 for im in images])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def frames(tmp_path):
    rows = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (250, 5, 0), (0, 0, 255)]):
        p = tmp_path / f"{i:06d}.jpg"
        Image.new("RGB", (8, 8), color).save(p)
        rows.append({"dataset": "cholec80", "video_id": 1, "frame_idx_25fps": 25 * i,
                     "path": str(p), "phase": "Preparation" if i in (0, 2) else "ClippingCutting",
                     "tool_hook": i == 1})
    return pd.DataFrame(rows)


def test_manifest_hash_depends_on_rows_and_order(frames):
    h = manifest_hash(frames)
    assert h == manifest_hash(frames.copy())
    assert h != manifest_hash(frames.iloc[::-1])
    assert h != manifest_hash(frames.iloc[:3])


def test_save_load_roundtrip_upcasts_and_checks(tmp_path, frames):
    emb, _ = embed_frames(FakeBackbone(), [p for p in frames.path], batch_size=2, log_every=0)
    meta = {"revision": "abc", "manifest_sha256": manifest_hash(frames)}
    save_index(tmp_path / "idx", emb, frames, meta)
    assert np.load(tmp_path / "idx" / "embeddings.npy").dtype == np.float16
    e32, m, _ = load_index(tmp_path / "idx")
    assert e32.dtype == np.float32 and np.allclose(e32, emb, atol=1e-3)
    assert cache_is_valid(tmp_path / "idx", "abc", manifest_hash(frames))
    assert not cache_is_valid(tmp_path / "idx", "other-rev", manifest_hash(frames))
    assert not cache_is_valid(tmp_path / "idx", "abc", manifest_hash(frames.iloc[:2]))


def test_load_rejects_tampered_manifest(tmp_path, frames):
    emb = np.eye(4, 3, dtype=np.float32)
    save_index(tmp_path / "idx", emb, frames, {"manifest_sha256": manifest_hash(frames)})
    frames.iloc[::-1].to_parquet(tmp_path / "idx" / "manifest.parquet", index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_index(tmp_path / "idx")


def test_git_hash_falls_back_to_bundle_file(tmp_path):
    (tmp_path / "GIT_HASH").write_text("abc123-dirty\n")
    assert git_hash(tmp_path) == "abc123-dirty"


def _save_part(path, emb, manifest, **meta):
    meta = {"model": "m", "revision": "abc", "dim": emb.shape[1], "storage_dtype": "float16",
            "videos": sorted(set(manifest.video_id)),
            "manifest_sha256": manifest_hash(manifest), **meta}
    save_index(path, emb, manifest.reset_index(drop=True), meta)


def test_merge_indexes_equals_direct_build(tmp_path, frames):
    """Two parts that tile the expected rows merge to the hash of a direct build."""
    full = pd.concat([frames, frames.assign(video_id=2)], ignore_index=True)
    emb = np.random.default_rng(0).standard_normal((8, 3)).astype(np.float32)
    _save_part(tmp_path / "a", emb[:4], full.iloc[:4], device="mps")
    _save_part(tmp_path / "b", emb[4:], full.iloc[4:], device="cuda")
    e, m, meta = merge_indexes([tmp_path / "a", tmp_path / "b"], full)
    assert meta["manifest_sha256"] == manifest_hash(full) and meta["videos"] == [1, 2]
    assert [p["device"] for p in meta["merged_from"]] == ["mps", "cuda"]
    assert np.array_equal(e.astype(np.float16), emb.astype(np.float16))
    save_index(tmp_path / "ab", e, m, meta)  # round-trips through load_index
    assert np.array_equal(load_index(tmp_path / "ab")[0], e)


def test_merge_indexes_refuses_bad_tiling_or_mixed_revisions(tmp_path, frames):
    full = pd.concat([frames, frames.assign(video_id=2)], ignore_index=True)
    emb = np.ones((8, 3), np.float32)
    _save_part(tmp_path / "a", emb[:4], full.iloc[:4])
    _save_part(tmp_path / "b", emb[4:], full.iloc[4:])
    with pytest.raises(ValueError, match="tile"):  # wrong order
        merge_indexes([tmp_path / "b", tmp_path / "a"], full)
    with pytest.raises(ValueError, match="tile"):  # gap
        merge_indexes([tmp_path / "a"], full)
    _save_part(tmp_path / "c", emb[4:], full.iloc[4:], revision="other")
    with pytest.raises(ValueError, match="revision"):
        merge_indexes([tmp_path / "a", tmp_path / "c"], full)


def test_scores_refuse_fp16():
    with pytest.raises(TypeError, match="upcast"):
        scores_for(np.ones((1, 3), np.float32), np.ones((2, 3), np.float16))


def test_evaluate_on_fake_embeddings(frames):
    """A 'red' query retrieves the two red Preparation frames first -> AP 1.0.

    Prevalence 2/4 = 0.5, so lift 2.0. The negative control has AP = nan (R = 0),
    never 0.0, and is reported with its score stats only.
    """
    emb, _ = embed_frames(FakeBackbone(), list(frames.path), batch_size=4, log_every=0)
    q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    queries = [Query("red", "red", "phase", "Preparation"),
               Query("hook", "green", "tool", "tool_hook"),
               Query("neg", "blue", "negative", None)]
    with pytest.warns(RuntimeWarning, match="exceeds the number of frames"):  # P@10, N=4
        t = evaluate(queries, scores_for(q, emb), frames).set_index("qid")
    assert t.loc["red", "ap"] == 1.0 and t.loc["red", "prevalence"] == 0.5
    assert t.loc["red", "lift"] == 2.0
    assert t.loc["hook", "ap"] == 1.0
    assert np.isnan(t.loc["neg", "ap"])
    assert {"tied_frames", "tied_groups"} <= set(t.columns)


# --- test-split scoring lock: unlocked after prereg-v1; the guard must still work ---

@pytest.fixture
def locked(monkeypatch):
    from srb import retrieval
    monkeypatch.setattr(retrieval, "TEST_SCORING_UNLOCKED", False)


def test_test_scoring_is_unlocked_after_prereg():
    from srb import retrieval
    assert retrieval.TEST_SCORING_UNLOCKED is True


@pytest.mark.parametrize("ids", [[41], [80], [1, 2, 60], range(1, 81)])
def test_refuse_test_scoring_raises_on_any_test_video(ids, locked):
    from srb.retrieval import refuse_test_scoring
    with pytest.raises(PermissionError, match="test videos"):
        refuse_test_scoring(ids)


def test_refuse_test_scoring_allows_train_and_val():
    from srb.retrieval import refuse_test_scoring
    refuse_test_scoring(range(1, 41))


def test_evaluate_refuses_manifest_with_test_video(frames, locked):
    m = frames.copy()
    m.loc[3, "video_id"] = 41
    q = [Query("p", "x", "phase", "Preparation")]
    with pytest.raises(PermissionError):
        evaluate(q, np.zeros((1, len(m)), dtype=np.float32), m)


@pytest.mark.filterwarnings("ignore:k=10 exceeds the number of frames")
def test_evaluate_ignores_other_datasets_ids(frames):
    """Endoscapes video 41 is not Cholec80 video 41."""
    m = frames.copy()
    m.loc[3, ["dataset", "video_id"]] = ["endoscapes", 41]
    q = [Query("p", "x", "phase", "Preparation")]
    evaluate(q, np.arange(len(m), dtype=np.float32)[None], m)


def test_eval_script_refuses_test_videos_before_loading_anything():
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_retrieval.py"
    r = subprocess.run([sys.executable, str(script), "--model", "clip-vit-l14",
                        "--videos", "40-41"], capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "refusing to score Cholec80 test videos [41] with the dev queries" in r.stderr
