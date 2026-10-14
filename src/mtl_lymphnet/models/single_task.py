"""Single-task baselines used in the MTL-vs-single-task ablation (Experiment 3)."""
from __future__ import annotations

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SegOnlyUNet(nn.Module):
    """Segmentation-only UNet (no classification head)."""

    def __init__(self, backbone: str = "tu-tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained: Optional[str] = "imagenet"):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=backbone, encoder_weights=pretrained, in_channels=3, classes=1, activation=None,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_out = self.unet(x)
        cls_dummy = torch.zeros(x.size(0), 1, device=x.device)
        return seg_out, cls_dummy


class ClsOnlyNet(nn.Module):
    """Classification-only network (encoder + global pool + MLP).

    Head is identical to `MTLUNet.classification_head` for a fair ablation.
    """

    def __init__(self, backbone: str = "tu-tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained: Optional[str] = "imagenet"):
        super().__init__()
        self.encoder = smp.Unet(
            encoder_name=backbone, encoder_weights=pretrained, in_channels=3, classes=1, activation=None,
        ).encoder
        enc_ch = self.encoder.out_channels[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(enc_ch, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        cls_out = self.classifier(features[-1])
        seg_dummy = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return seg_dummy, cls_out
