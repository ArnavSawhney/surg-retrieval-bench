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
``transformers`` model and will need its own ``Backbone`` subclass plus a separate
environment -- but callers keep using ``get_backbone(name)`` and the two encode
methods, so nothing downstream changes.

Model revisions are pinned by commit hash (see ``_SPECS``) so that a silent upstream
re-upload cannot change published numbers.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from PIL import Image

__all__ = ["Backbone", "HFDualEncoder", "get_backbone", "list_backbones", "select_device"]


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
        self.embed_dim = int(self.model.config.projection_dim)

    def _encode_image_batch(self, images: list[Image.Image]) -> torch.Tensor:
        # Always convert to RGB: some extracted frames may carry an alpha channel.
        batch = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        batch = {k: v.to(self.device, self.dtype if v.is_floating_point() else None)
                 for k, v in batch.items()}
        return _projected_features(self.model.get_image_features(**batch))

    def _encode_text_batch(self, texts: list[str]) -> torch.Tensor:
        batch = self.processor(
            text=texts, padding=self.spec.text_padding, truncation=True, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return _projected_features(self.model.get_text_features(**batch))


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
# Session 1 deliberately ships CLIP only. The commented rows are the Week 2 targets,
# left here so the shape of a new entry is obvious; they are NOT available yet.
_SPECS: dict[str, HFDualEncoderSpec] = {
    "clip-vit-l14": HFDualEncoderSpec(
        hf_id="openai/clip-vit-large-patch14",
        revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        default_batch_size=32,
        notes="General-purpose baseline (Radford et al., 2021). 224 px, projection_dim 768.",
    ),
    # "siglip-so400m-384": HFDualEncoderSpec(
    #     hf_id="google/siglip-so400m-patch14-384", revision="...",
    #     text_padding="max_length", notes="Strong general baseline (Zhai et al., 2023)."),
    # "medsiglip-448": HFDualEncoderSpec(
    #     hf_id="google/medsiglip-448", revision="...", text_padding="max_length",
    #     gated=True, notes="Accept the HAI-DEF terms on HF first. projection_dim 1152."),
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
    return HFDualEncoder(name, _SPECS[name], device=device, fp16=fp16, batch_size=batch_size)
