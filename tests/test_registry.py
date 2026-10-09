"""Tests for srb.models.registry that do NOT download a model.

The end-to-end check (real weights, real device, real similarity matrix) is
``scripts/smoke_clip.py`` -- it needs ~1.7 GB of weights, so it stays out of the test
suite. What is tested here is the logic that surrounds the model: device selection,
the fp16 guard, input validation, normalisation, and the transformers-version-
tolerant feature extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from srb.models.registry import (
    Backbone,
    _projected_features,
    get_backbone,
    list_backbones,
    select_device,
)

# --------------------------------------------------------------------------- #
# registry contents
# --------------------------------------------------------------------------- #


def test_registry_ships_the_four_preregistered_models():
    assert list_backbones() == ["clip-vit-l14", "medsiglip-448", "peskavlp", "siglip-so400m-384"]


def test_unknown_backbone_raises_with_the_available_names():
    with pytest.raises(KeyError, match="clip-vit-l14"):
        get_backbone("surgvlp")


def test_medsiglip_is_gated_and_needs_hf_token(monkeypatch):
    """Without HF_TOKEN the gated model fails with a clear message, before any download."""
    from srb.models.registry import _SPECS, _hf_token

    assert _SPECS["medsiglip-448"].gated is True
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="export HF_TOKEN"):
        _hf_token(required=True, model="google/medsiglip-448")


@pytest.mark.parametrize("name", ["clip-vit-l14", "siglip-so400m-384", "medsiglip-448",
                                  "peskavlp"])
def test_revision_is_pinned_to_a_commit_hash(name):
    """A floating 'main' revision would let an upstream re-upload change our numbers."""
    from srb.models.registry import _SPECS

    rev = _SPECS[name].revision
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


@pytest.mark.parametrize("name", ["siglip-so400m-384", "medsiglip-448"])
def test_siglip_text_is_padded_to_max_length_64(name):
    """SigLIP-family models must call the processor with padding="max_length", max_length=64.

    Checked on the real call path with a fake processor/model, so no weights needed.
    """
    from types import SimpleNamespace

    from srb.models.registry import _SPECS, HFDualEncoder

    calls = []

    def fake_processor(**kwargs):
        calls.append(kwargs)
        return {"input_ids": torch.zeros(len(kwargs["text"]), 64, dtype=torch.long)}

    fake_model = SimpleNamespace(get_text_features=lambda **kw: torch.ones(
        kw["input_ids"].shape[0], 4))
    enc = HFDualEncoder.__new__(HFDualEncoder)
    enc.spec, enc.processor, enc.model = _SPECS[name], fake_processor, fake_model
    enc.device, enc.dtype = torch.device("cpu"), torch.float32

    enc._encode_text_batch(["a", "b"])
    assert calls[0]["padding"] == "max_length"
    assert calls[0]["max_length"] == 64
    assert calls[0]["truncation"] is True


def test_embed_dim_falls_back_to_vision_hidden_size():
    """SigLIP's config has no projection_dim; its embedding is vision hidden_size."""
    from types import SimpleNamespace

    from srb.models.registry import _embed_dim

    assert _embed_dim(SimpleNamespace(projection_dim=768)) == 768
    assert _embed_dim(SimpleNamespace(vision_config=SimpleNamespace(hidden_size=1152))) == 1152


# --------------------------------------------------------------------------- #
# device selection
# --------------------------------------------------------------------------- #


def test_select_device_auto_returns_an_available_device():
    dev = select_device()
    assert dev.type in {"cuda", "mps", "cpu"}
    if dev.type == "cpu":
        assert not torch.cuda.is_available() and not torch.backends.mps.is_available()


def test_select_device_prefers_cuda_then_mps_then_cpu():
    """The documented order. Only the branch this machine can reach is asserted."""
    dev = select_device()
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    elif torch.backends.mps.is_available():
        assert dev.type == "mps"


def test_select_device_cpu_is_always_available():
    assert select_device("cpu").type == "cpu"


def test_requesting_an_unavailable_device_raises_instead_of_falling_back():
    """A silent fallback to cpu would make a run 50x slower without saying so."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cuda"):
            select_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="mps"):
            select_device("mps")


# --------------------------------------------------------------------------- #
# fp16 policy: cuda only
# --------------------------------------------------------------------------- #


def test_fp16_is_rejected_off_cuda():
    """MPS/CPU fp16 changes numerics; Mac runs validate code paths, not numbers."""
    from srb.models.registry import _resolve_dtype

    with pytest.raises(ValueError, match="only supported on cuda"):
        _resolve_dtype(torch.device("cpu"), fp16=True)
    with pytest.raises(ValueError, match="only supported on cuda"):
        _resolve_dtype(torch.device("mps"), fp16=True)


def test_default_dtype_is_fp32_off_cuda_and_fp16_on_cuda():
    from srb.models.registry import _resolve_dtype

    assert _resolve_dtype(torch.device("cpu"), None) is torch.float32
    assert _resolve_dtype(torch.device("mps"), None) is torch.float32
    assert _resolve_dtype(torch.device("cuda"), None) is torch.float16
    assert _resolve_dtype(torch.device("cuda"), False) is torch.float32


# --------------------------------------------------------------------------- #
# feature extraction across transformers versions
# --------------------------------------------------------------------------- #


def test_projected_features_accepts_a_bare_tensor():
    """transformers 4.x behaviour."""
    t = torch.ones(2, 8)
    assert _projected_features(t) is t


def test_projected_features_accepts_pooler_output():
    """transformers 5.x: the projected features are written into pooler_output."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    expected = torch.arange(6.0).reshape(2, 3)
    out = BaseModelOutputWithPooling(last_hidden_state=torch.zeros(2, 4, 3), pooler_output=expected)
    assert torch.equal(_projected_features(out), expected)


def test_projected_features_accepts_a_tuple():
    t = torch.ones(2, 8)
    assert _projected_features((t, torch.zeros(2, 4))) is t


def test_projected_features_raises_on_an_unknown_shape():
    """Fail loudly rather than embed the wrong tensor after a dependency bump."""
    with pytest.raises(TypeError, match="transformers API changed"):
        _projected_features({"not": "a tensor"})


# --------------------------------------------------------------------------- #
# the Backbone contract, exercised with a fake encoder (no weights)
# --------------------------------------------------------------------------- #


class FakeBackbone(Backbone):
    """Deterministic stand-in: embeddings are NOT unit norm until the base class fixes them."""

    def __init__(self, dim: int = 4, batch_size: int = 3, zero_vector: bool = False):
        self.name = "fake"
        self.embed_dim = dim
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.batch_size = batch_size
        self.zero_vector = zero_vector
        self.batch_sizes_seen: list[int] = []

    def _fake(self, items: list) -> torch.Tensor:
        self.batch_sizes_seen.append(len(items))
        if self.zero_vector:
            return torch.zeros(len(items), self.embed_dim)
        # magnitudes deliberately != 1 so normalisation is observable
        return torch.arange(1.0, len(items) * self.embed_dim + 1).reshape(len(items), self.embed_dim)

    def _encode_image_batch(self, images):
        return self._fake(images)

    def _encode_text_batch(self, texts):
        return self._fake(texts)


def _imgs(n: int) -> list[Image.Image]:
    return [Image.new("RGB", (8, 8)) for _ in range(n)]


def test_embeddings_are_float32_and_unit_norm():
    out = FakeBackbone().encode_text(["a", "b", "c", "d", "e"])
    assert out.dtype == np.float32
    assert out.shape == (5, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_batching_covers_every_item_exactly_once():
    """7 items at batch_size 3 -> batches of 3, 3, 1."""
    m = FakeBackbone(batch_size=3)
    out = m.encode_text([str(i) for i in range(7)])
    assert m.batch_sizes_seen == [3, 3, 1]
    assert out.shape == (7, 4)


def test_batch_size_override_is_honoured():
    m = FakeBackbone(batch_size=3)
    m.encode_text([str(i) for i in range(5)], batch_size=2)
    assert m.batch_sizes_seen == [2, 2, 1]


def test_zero_vector_raises_rather_than_dividing_by_zero():
    with pytest.raises(RuntimeError, match="zero vector"):
        FakeBackbone(zero_vector=True).encode_text(["a"])


def test_encode_image_rejects_a_single_image():
    with pytest.raises(TypeError, match="list of PIL.Image"):
        FakeBackbone().encode_image(Image.new("RGB", (8, 8)))


def test_encode_text_rejects_a_bare_string():
    """encode_text("query") would otherwise silently encode 5 single characters."""
    with pytest.raises(TypeError, match="not a single str"):
        FakeBackbone().encode_text("query")


def test_encode_rejects_empty_input():
    with pytest.raises(ValueError, match="empty list"):
        FakeBackbone().encode_text([])
    with pytest.raises(ValueError, match="empty list"):
        FakeBackbone().encode_image([])


def test_encode_image_rejects_non_images():
    with pytest.raises(TypeError, match=r"images\[1\] is str"):
        FakeBackbone().encode_image([Image.new("RGB", (8, 8)), "not an image"])


def test_encode_text_rejects_non_strings():
    with pytest.raises(TypeError, match=r"texts\[1\] is int"):
        FakeBackbone().encode_text(["ok", 3])


def test_cosine_similarity_of_unit_embeddings_is_a_dot_product():
    """The whole point of normalising in the base class: text @ image.T is cosine sim."""
    m = FakeBackbone()
    txt, img = m.encode_text(["a", "b"]), m.encode_image(_imgs(3))
    sim = txt @ img.T
    assert sim.shape == (2, 3)
    assert np.all(sim <= 1.0 + 1e-6) and np.all(sim >= -1.0 - 1e-6)


# --------------------------------------------------------------------------- #
# PeskaVLP: separate environment behind a script boundary
# --------------------------------------------------------------------------- #


def test_peskavlp_pins_are_complete():
    """Code commit, checkpoint hash and BERT revision are all pinned (no floating ids)."""
    from srb.models.peskavlp_pins import PESKAVLP

    hexdigits = set("0123456789abcdef")
    for key, n in [("surgvlp_commit", 40), ("bert_revision", 40),
                   ("checkpoint_sha256", 64), ("checkpoint_zip_sha256", 64)]:
        assert len(PESKAVLP[key]) == n and set(PESKAVLP[key]) <= hexdigits, key
    assert PESKAVLP["revision"] == PESKAVLP["surgvlp_commit"]


def _fake_peskavlp(monkeypatch, tmp_path, script_body):
    """A SubprocessBackbone pointed at a stub script run by this interpreter."""
    import sys

    from srb.models.registry import _SPECS, SubprocessBackbone

    script = tmp_path / "stub.py"
    script.write_text(script_body)
    enc = SubprocessBackbone.__new__(SubprocessBackbone)
    enc.name, enc.spec = "peskavlp", _SPECS["peskavlp"]
    enc.root, enc.python = tmp_path, Path(sys.executable)
    enc.device_arg, enc.device, enc.dtype = None, torch.device("cpu"), torch.float32
    enc.batch_size, enc.embed_dim = 32, 4
    monkeypatch.setattr(enc, "spec", type(enc.spec)(**{**enc.spec.__dict__, "script": "stub.py"}))
    return enc


STUB = """
import json, sys, numpy as np
a = sys.argv[1:]
texts = json.loads(open(a[a.index("--texts") + 1]).read())
emb = np.arange(1, 4 * len(texts) + 1, dtype=np.float32).reshape(len(texts), 4)
np.save(a[a.index("--out") + 1], emb)
"""


def test_peskavlp_text_goes_through_the_subprocess_and_is_normalised(monkeypatch, tmp_path):
    enc = _fake_peskavlp(monkeypatch, tmp_path, STUB)
    out = enc.encode_text(["a grasper", "a hook"])
    assert out.shape == (2, 4) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_peskavlp_subprocess_does_not_see_hf_token(monkeypatch, tmp_path):
    body = "import os, sys\nif 'HF_TOKEN' in os.environ: sys.exit('HF_TOKEN leaked')\n" + STUB
    enc = _fake_peskavlp(monkeypatch, tmp_path, body)
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    enc.encode_text(["a grasper"])


def test_peskavlp_subprocess_failure_raises(monkeypatch, tmp_path):
    enc = _fake_peskavlp(monkeypatch, tmp_path, "import sys; sys.exit('boom')")
    with pytest.raises(RuntimeError, match="boom"):
        enc.encode_text(["a grasper"])


def test_peskavlp_refuses_image_encoding_through_the_registry(monkeypatch, tmp_path):
    enc = _fake_peskavlp(monkeypatch, tmp_path, STUB)
    with pytest.raises(NotImplementedError, match="index frames with"):
        enc.encode_image([Image.new("RGB", (8, 8))])
