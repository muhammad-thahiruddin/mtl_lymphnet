"""Multi-task UNet: shared encoder with segmentation + classification heads.

Interpretability note (R1.11): the classification head reads pooled encoder
(bottleneck) features -- NOT the predicted segmentation mask. Any figure that
places a segmentation overlay next to a malignancy probability is providing
complementary spatial context, not a causal explanation of the classification
decision. Keep this distinction explicit in manuscript text/captions.
"""
from __future__ import annotations

from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class MTLUNet(nn.Module):
    """Shared encoder with a UNet segmentation decoder and a classification head."""

    def __init__(self, backbone: str, pretrained: Optional[str] = "imagenet"):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights=pretrained,
            in_channels=3,
            classes=1,
            activation=None,  # raw logits for BCEWithLogitsLoss
        )

        encoder_out_channels = self.unet.encoder.out_channels[-1]

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(encoder_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.unet.encoder(x)

        decoder_out = self.unet.decoder(features)
        seg_logits = self.unet.segmentation_head(decoder_out)  # [B, 1, H, W]

        bottleneck = features[-1]
        cls_logits = self.classification_head(bottleneck)  # [B, 1]

        return seg_logits, cls_logits


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {"total": total, "trainable": trainable, "size_mb": size_mb}
