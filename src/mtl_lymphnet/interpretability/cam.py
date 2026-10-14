"""CAM-based interpretability for the classification head.

Interpretability caveat (R1.11): the classification head consumes pooled
ENCODER features, not the predicted segmentation mask. CAM overlays show what
the encoder attends to for the classification decision; they are
complementary spatial context, not a causal explanation of why the
segmentation mask looks the way it does (or vice versa).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class MTLUNetCAMWrapper(nn.Module):
    """Wraps `MTLUNet` so `pytorch_grad_cam` can attach hooks to the encoder
    bottleneck (works across TinyViT / MobileNetV3 / EfficientNet backbones).

    ``bottleneck_tap`` (nn.Identity) is the CAM target layer: it sits between
    the encoder bottleneck output and the classification head, guaranteeing a
    4-D [B, C, Hb, Wb] spatial signal regardless of backbone family. The
    wrapper returns 2-column logits [B, 2] as expected by
    `ClassifierOutputTarget` (col 0 = benign score, col 1 = malignant score).
    """

    def __init__(self, mtlunet: nn.Module):
        super().__init__()
        self.mtlunet = mtlunet
        self.bottleneck_tap = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mtlunet.unet.encoder(x)
        bottleneck = self.bottleneck_tap(features[-1])
        cls_logit = self.mtlunet.classification_head(bottleneck)
        return torch.cat([-cls_logit, cls_logit], dim=-1)


def build_cam_methods(cam_wrapper: MTLUNetCAMWrapper):
    """Instantiate the three CAM methods used for TinyViT-family backbones:
    EigenCAM (gradient-free, most stable for ViT), GradCAM++ (better boundary
    localisation), and LayerCAM (sharpest fine-grained detail)."""
    from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus, LayerCAM

    target_layers = [cam_wrapper.bottleneck_tap]
    return [
        ("EigenCAM", EigenCAM(model=cam_wrapper, target_layers=target_layers)),
        ("GradCAM++", GradCAMPlusPlus(model=cam_wrapper, target_layers=target_layers)),
        ("LayerCAM", LayerCAM(model=cam_wrapper, target_layers=target_layers)),
    ]


def compute_cam_mask_iou(cam_map: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between a thresholded CAM heatmap and the ground-truth segmentation
    mask -- a rough proxy for whether the classifier attends to the lesion."""
    cam_bin = (cam_map > threshold).astype(np.uint8)
    inter = (cam_bin * gt_mask).sum()
    union = cam_bin.sum() + gt_mask.sum() - inter
    return float(inter / (union + 1e-7))


def release_cam_hooks(cam_objects: list[tuple[str, object]]) -> None:
    for _, cam_fn in cam_objects:
        cam_fn.activations_and_grads.release()
