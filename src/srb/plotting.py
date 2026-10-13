"""Shared figure style: one fixed colour and marker per model, used by every figure.

Colours are slots 1-4 of a categorical palette checked for colour-vision deficiency
with every pair in play (worst pair, deuteranopia, OKLab dE 9.2; normal vision 16.3).
The aqua slot is below 3:1 contrast on white, so every model also has its own marker
shape and every figure has a legend: identity never rests on colour alone.
"""

from __future__ import annotations

MODELS = ("clip-vit-l14", "siglip-so400m-384", "medsiglip-448", "peskavlp")

MODEL_STYLE = {
    "clip-vit-l14": {"label": "CLIP ViT-L/14", "color": "#2a78d6", "marker": "o"},
    "siglip-so400m-384": {"label": "SigLIP so400m", "color": "#eb6834", "marker": "s"},
    "medsiglip-448": {"label": "MedSigLIP-448", "color": "#1baf7a", "marker": "^"},
    "peskavlp": {"label": "PeskaVLP", "color": "#4a3aa7", "marker": "D"},
}

# Recessive ink for reference lines, axes and grid.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

TARGET_LABELS = {
    "phase_preparation": "Preparation",
    "phase_calottriangledissection": "Calot triangle dissection",
    "phase_clippingcutting": "Clipping and cutting",
    "phase_gallbladderdissection": "Gallbladder dissection",
    "phase_gallbladderpackaging": "Gallbladder packaging",
    "phase_cleaningcoagulation": "Cleaning and coagulation",
    "phase_gallbladderretraction": "Gallbladder retraction",
    "tool_grasper": "Grasper",
    "tool_bipolar": "Bipolar",
    "tool_hook": "Hook",
    "tool_scissors": "Scissors",
    "tool_clipper": "Clipper",
    "tool_irrigator": "Irrigator",
    "tool_specimenbag": "Specimen bag",
}

ROLE_LABELS = {
    1: "r1 technical",
    2: "r2 action",
    3: "r3 visible",
    4: "r4 lay",
    5: "r5 label",
}


def apply_paper_style() -> None:
    """Small, plain matplotlib defaults for a figure printed at paper column width."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed TrueType, so text stays text in the PDF
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
