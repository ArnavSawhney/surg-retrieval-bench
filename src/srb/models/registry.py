"""Uniform encode_image / encode_text interface for every backbone in the benchmark.

Every backbone returns **L2-normalised float32** embeddings of shape ``(n, dim)``, so
cosine similarity is a plain dot product and callers never need to know which model
produced the vectors::

    from srb.models.registry import get_backbone, list_backbones

    model = get_backbone("clip-vit-l14")
    img = model.encode_image([pil_image, ...])   # (n_images, dim) float32, unit norm
    txt = model.encode_text(["clipping the cystic duct", ...])  # (n_texts, dim)
    sim = txt @ img.T                            # (n_texts, n_images) cosine sim

Adding a backbone
-----------------
SigLIP and MedSigLIP are ``transformers`` dual encoders like CLIP, so they only need a
new ``HFDualEncoder`` spec in ``_SPECS`` (SigLIP needs ``padding="max_length"`` for
text, which is what ``text_padding`` is for). PeskaVLP / SurgVLP is not a
``transformers`` model and needs its own environment (``.venv-surgvlp``), so
``get_backbone("peskavlp")`` returns a ``SubprocessBackbone`` whose ``encode_text``
runs ``scripts/peskavlp_encode.py`` in that environment. Callers keep using
``get_backbone(name)`` and the encode methods, so nothing downstream changes. Its
frame index is built by the same script (``index``), not by ``build_index.py``.

Model revisions are pinned by commit hash (see ``_SPECS``) so that a silent upstream
re-upload cannot change published numbers.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from PIL import Image

from srb.models.peskavlp_pins import PESKAVLP

__all__ = ["Backbone", "HFDualEncoder", "SubprocessBackbone", "get_backbone",
           "list_backbones", "select_device"]


# --------------------------------------------------------------------------- #
# device
# --------------------------------------------------------------------------- #
def select_device(requested: str | None = None) -> torch.device:
    """Pick a device: explicit request, else cuda -> mps -> cpu.

    Raises if an explicitly requested device is not available, rather than silently
    falling back to cpu and making a run 50x slower without saying so.
    """
    if requested is not None:
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' requested but torch.backends.mps.is_available() is False")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(device: torch.device, fp16: bool | None) -> torch.dtype:
    """fp16 on cuda only.

    MPS fp16 silently changes numerics on some ops, and this repo's Mac runs exist to
    validate code paths, not to produce published numbers. Default: fp16 iff cuda.
    """
    if fp16 is None:
        fp16 = device.type == "cuda"
    if fp16 and device.type != "cuda":
        raise ValueError(
            f"fp16=True is only supported on cuda, not {device.type!r}. "
            "Run the full extraction on a GPU, or leave fp16 unset."
        )
    return torch.float16 if fp16 else torch.float32


# --------------------------------------------------------------------------- #
# base interface
# --------------------------------------------------------------------------- #
class Backbone(ABC):
    """A frozen image-text encoder.

    Subclasses implement ``_encode_image_batch`` / ``_encode_text_batch`` and return
    un-normalised torch tensors; this base class handles batching, normalisation,
    the float32 cast and the numpy conversion so that every backbone agrees.
    """

    name: str
    embed_dim: int
    device: torch.device
    dtype: torch.dtype

    # -- subclass hooks ----------------------------------------------------- #
    @abstractmethod
    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor: ...

    @abstractmethod
    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor: ...

    # -- public API --------------------------------------------------------- #
    def encode_image(self, images: list[Image.Image], batch_size: int | None = None) -> np.ndarray:
        """Encode PIL images -> ``(len(images), embed_dim)`` float32, L2-normalised."""
        if not isinstance(images, (list, tuple)):
            raise TypeError("encode_image expects a list of PIL.Image, not a single image")
        if len(images) == 0:
            raise ValueError("encode_image got an empty list")
        for i, im in enumerate(images):
            if not isinstance(im, Image.Image):
                raise TypeError(f"images[{i}] is {type(im).__name__}, expected PIL.Image")
        return self._run(list(images), self._encode_image_batch, batch_size)

    def encode_text(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode query strings -> ``(len(texts), embed_dim)`` float32, L2-normalised."""
        if isinstance(texts, str):
            raise TypeError("encode_text expects a list of str, not a single str")
        if not isinstance(texts, (list, tuple)):
            raise TypeError("encode_text expects a list of str")
        if len(texts) == 0:
            raise ValueError("encode_text got an empty list")
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"texts[{i}] is {type(t).__name__}, expected str")
        return self._run(list(texts), self._encode_text_batch, batch_size)

    # -- shared machinery --------------------------------------------------- #
    def _run(self, items: list, fn: Callable[[list], torch.Tensor], batch_size: int | None):
        bs = batch_size or self.batch_size
        if bs < 1:
            raise ValueError(f"batch_size must be >= 1, got {bs}")
        out = []
        with torch.inference_mode():
            for start in range(0, len(items), bs):
                feats = fn(items[start : start + bs])
                out.append(self._postprocess(feats))
        return np.concatenate(out, axis=0)

    @staticmethod
    def _postprocess(feats: torch.Tensor) -> np.ndarray:
        """float32 -> L2-normalise -> numpy. Cast before normalising to avoid fp16 overflow."""
        feats = feats.detach().to(torch.float32)
        norms = feats.norm(dim=-1, keepdim=True)
        if torch.any(norms == 0):
            raise RuntimeError("encoder produced a zero vector; cannot L2-normalise")
        return (feats / norms).cpu().numpy().astype(np.float32, copy=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"{type(self).__name__}(name={self.name!r}, dim={self.embed_dim}, "
            f"device={self.device.type!r}, dtype={str(self.dtype).removeprefix('torch.')!r})"
        )


# --------------------------------------------------------------------------- #
# transformers dual encoders (CLIP now; SigLIP / MedSigLIP later)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HFDualEncoderSpec:
    """Everything that differs between CLIP-shaped ``transformers`` dual encoders."""

    hf_id: str
    revision: str  # pinned commit hash: an upstream re-upload must not move our numbers
    default_batch_size: int = 32
    text_padding: str | bool = True  # SigLIP needs "max_length"
    text_max_length: int | None = None  # SigLIP: 64, the length it was trained with
    gated: bool = False  # needs HF_TOKEN (e.g. MedSigLIP HAI-DEF terms)
    notes: str = ""
    processor_kwargs: dict = field(default_factory=dict)


class HFDualEncoder(Backbone):
    """CLIP-shaped ``transformers`` model: ``get_image_features`` / ``get_text_features``."""

    def __init__(self, name: str, spec: HFDualEncoderSpec, *, device: str | None = None,
                 fp16: bool | None = None, batch_size: int | None = None):
        from transformers import AutoModel, AutoProcessor

        self.name = name
        self.spec = spec
        self.device = select_device(device)
        self.dtype = _resolve_dtype(self.device, fp16)
        self.batch_size = batch_size or spec.default_batch_size

        token = _hf_token(required=spec.gated, model=spec.hf_id)
        auth = {"token": token} if token else {}

        self.processor = AutoProcessor.from_pretrained(
            spec.hf_id, revision=spec.revision, **auth, **spec.processor_kwargs
        )
        self.model = (
            AutoModel.from_pretrained(spec.hf_id, revision=spec.revision, dtype=self.dtype, **auth)
            .to(self.device)
            .eval()
        )
        self.embed_dim = _embed_dim(self.model.config)

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        # Always convert to RGB: some extracted frames may carry an alpha channel.
        batch = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        batch = {k: v.to(self.device, self.dtype if v.is_floating_point() else None)
                 for k, v in batch.items()}
        return _projected_features(self.model.get_image_features(**batch))

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        length = {"max_length": self.spec.text_max_length} if self.spec.text_max_length else {}
        batch = self.processor(
            text=texts, padding=self.spec.text_padding, truncation=True, return_tensors="pt",
            **length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return _projected_features(self.model.get_text_features(**batch))


# --------------------------------------------------------------------------- #
# models that live in their own environment (PeskaVLP)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubprocessSpec:
    """A backbone whose code runs in a separate venv, behind a script boundary."""

    hf_id: str  # not necessarily on HF; recorded in index meta
    revision: str  # code commit, used as the index cache key
    python: str  # interpreter, relative to the repo root
    script: str  # encoder script, relative to the repo root
    embed_dim: int
    default_batch_size: int = 32
    notes: str = ""


class SubprocessBackbone(Backbone):
    """Text encoding through ``<python> <script> text``; frames are indexed by the
    script's ``index`` command, so ``encode_image`` refuses rather than spawning a
    process (and reloading the model) for every chunk of frames."""

    def __init__(self, name: str, spec: SubprocessSpec, *, device: str | None = None,
                 fp16: bool | None = None, batch_size: int | None = None):
        if fp16:
            raise ValueError(f"{name} runs in float32 only")
        self.name, self.spec = name, spec
        self.root = Path(__file__).resolve().parents[3]
        self.python = self.root / spec.python
        if not self.python.exists():
            raise RuntimeError(f"{name} needs its own environment at {spec.python}; "
                               "see PROGRESS.md (PeskaVLP setup)")
        self.device_arg = device
        self.device = torch.device(device or "cpu")  # where the subprocess is asked to run
        self.dtype = torch.float32
        self.batch_size = batch_size or spec.default_batch_size
        self.embed_dim = spec.embed_dim

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        raise NotImplementedError(f"index frames with `{self.spec.python} {self.spec.script} "
                                  "index --videos ...`, not through the registry")

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "texts.json", Path(tmp) / "emb.npy"
            src.write_text(json.dumps(texts))
            cmd = [str(self.python), str(self.root / self.spec.script)]
            if self.device_arg:
                cmd += ["--device", self.device_arg]
            cmd += ["text", "--texts", str(src), "--out", str(out)]
            env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}  # not needed there
            r = subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{self.name} text encoder failed:\n{r.stderr[-2000:]}")
            emb = np.load(out)
        if emb.shape != (len(texts), self.embed_dim):
            raise RuntimeError(f"{self.name} returned {emb.shape}, expected "
                               f"({len(texts)}, {self.embed_dim})")
        return torch.from_numpy(emb)


def _embed_dim(config) -> int:
    """Output dimension: CLIP has ``projection_dim``; SigLIP has no projection layer
    config, and its pooled embedding has the vision tower's ``hidden_size``."""
    dim = getattr(config, "projection_dim", None)
    if dim is None:
        dim = config.vision_config.hidden_size
    return int(dim)


def _projected_features(out) -> torch.Tensor:
    """Pull the projected embedding out of a ``get_*_features`` return value.

    transformers 4.x returned a bare tensor. transformers 5.x returns a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` has been overwritten with
    the projected features. We support both so that a dependency bump cannot silently
    change what gets embedded -- and fail loudly if a future version returns something
    else again, rather than embedding the wrong tensor.
    """
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(
        f"cannot extract projected features from a {type(out).__name__}; "
        "the transformers API changed -- check get_image_features/get_text_features "
        "before trusting any embedding produced by this build."
    )


def _hf_token(*, required: bool, model: str) -> str | None:
    """Read HF_TOKEN from the environment. Never logged, never written to disk."""
    token = os.environ.get("HF_TOKEN")
    if required and not token:
        raise RuntimeError(
            f"{model} is gated on Hugging Face. Accept its terms on the model page, "
            "then export HF_TOKEN in your shell. Never pass the token as a CLI argument."
        )
    return token


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
# CLIP and SigLIP (Week 1), MedSigLIP and PeskaVLP (Week 2). PeskaVLP runs in its own
# environment behind scripts/peskavlp_encode.py (pins in srb.models.peskavlp_pins).
_SPECS: dict[str, HFDualEncoderSpec | SubprocessSpec] = {
    "clip-vit-l14": HFDualEncoderSpec(
        hf_id="openai/clip-vit-large-patch14",
        revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        default_batch_size=32,
        notes="General-purpose baseline (Radford et al., 2021). 224 px, projection_dim 768.",
    ),
    # SigLIP was trained with text padded to exactly 64 tokens. With default
    # (longest) padding the pooled text embedding shifts and quality silently drops,
    # so padding="max_length" + max_length=64 is required, not cosmetic.
    #
    # Ranking by cosine similarity is correct for SigLIP: its sigmoid logit is
    # ``t * cos + b`` with a learned scale t > 0 and bias b that are per-model
    # constants, so the logit (and the sigmoid of it) is a monotone function of the
    # cosine and gives the identical ranking for a query.
    "siglip-so400m-384": HFDualEncoderSpec(
        hf_id="google/siglip-so400m-patch14-384",
        revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        default_batch_size=16,
        text_padding="max_length",
        text_max_length=64,
        notes="Strong general baseline (Zhai et al., 2023). 384 px, embedding dim 1152.",
    ),
    # MedSigLIP is a SigLIP-architecture model (SiglipModel), so the same padding rule
    # and the same cosine-ranking argument apply. Checked against its own model card and
    # configs, not copied from SigLIP: 448x448 input (preprocessor_config.json), 64 text
    # tokens (text_config.max_position_embeddings, tokenizer model_max_length), embedding
    # dim 1152. The card's example resizes with tf.image.resize (bilinear) to match Big
    # Vision, but states that the Transformers processor's own resize may be used instead;
    # we use the processor (bicubic, as configured), as for the other models.
    "medsiglip-448": HFDualEncoderSpec(
        hf_id="google/medsiglip-448",
        revision="9cea28a1a1195f665105faa6e8544c112fd960a4",
        default_batch_size=8,
        text_padding="max_length",
        text_max_length=64,
        gated=True,
        notes="Medical SigLIP (Google HAI-DEF). Gated: accept the HAI-DEF terms on HF and "
              "export HF_TOKEN. 448 px, embedding dim 1152.",
    ),
    "peskavlp": SubprocessSpec(
        hf_id=PESKAVLP["hf_id"],
        revision=PESKAVLP["revision"],
        python=PESKAVLP["python"],
        script="scripts/peskavlp_encode.py",
        embed_dim=PESKAVLP["embed_dim"],
        default_batch_size=PESKAVLP["default_batch_size"],
        notes="Surgical VLP (Yuan et al., NeurIPS 2024): ResNet-50 + Bio_ClinicalBERT, "
              "224 px, 77 tokens, embedding dim 768. Separate env .venv-surgvlp.",
    ),
}


def list_backbones() -> list[str]:
    """Names accepted by :func:`get_backbone`."""
    return sorted(_SPECS)


def get_backbone(name: str, *, device: str | None = None, fp16: bool | None = None,
                 batch_size: int | None = None) -> Backbone:
    """Build a backbone by registry name.

    ``device``: None auto-selects cuda -> mps -> cpu. ``fp16``: None means fp16 iff cuda.
    """
    if name not in _SPECS:
        raise KeyError(f"unknown backbone {name!r}; available: {list_backbones()}")
    spec = _SPECS[name]
    cls = SubprocessBackbone if isinstance(spec, SubprocessSpec) else HFDualEncoder
    return cls(name, spec, device=device, fp16=fp16, batch_size=batch_size)
