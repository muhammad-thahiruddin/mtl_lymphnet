"""Backbone registry for the encoder ablation study (R1.4).

All backbone definitions live here. To switch the proposed backbone,
change PROPOSED_KEY below (or run ``python configs/switch_backbone.py <key>``).

The framework is intentionally backbone-agnostic: any key below (or any
``timm``-compatible encoder name accepted by ``segmentation_models_pytorch``)
can be selected via config, so the manuscript's final decision does not
require further code changes beyond updating PROPOSED_KEY.
"""
from __future__ import annotations


# --- Central backbone registry -------------------------------------------- #
# Human-readable name -> timm model string
BACKBONES = {
    "TinyViT-5M": "tu-tiny_vit_5m_224.dist_in22k_ft_in1k",
    "MobileNetV3-Large": "tu-mobilenetv3_large_100",
    "EfficientNet-B0": "tu-efficientnet_b0",
}

# --- THE SINGLE SOURCE OF TRUTH ------------------------------------------ #
# Change this ONE value to switch the proposed backbone everywhere.
PROPOSED_KEY = "TinyViT-5M"
# -------------------------------------------------------------------------- #


def proposed_backbone() -> str:
    """Return the timm model string for the proposed backbone."""
    return BACKBONES[PROPOSED_KEY]


def proposed_name() -> str:
    """Return the human-readable name of the proposed backbone."""
    return PROPOSED_KEY


def variant_backbones() -> dict[str, str]:
    """Return {name: timm_key} for all non-proposed backbones."""
    return {k: v for k, v in BACKBONES.items() if k != PROPOSED_KEY}


def variant_names() -> list[str]:
    """Return the list of non-proposed backbone names (for stats/reporting)."""
    return [k for k in BACKBONES if k != PROPOSED_KEY]
