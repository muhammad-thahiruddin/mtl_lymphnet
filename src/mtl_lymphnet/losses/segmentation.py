"""Segmentation losses: soft Dice + compound Dice/BCE."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Binary Soft Dice Loss."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CompoundSegLoss(nn.Module):
    """`dice_weight * DiceLoss + (1 - dice_weight) * BCEWithLogitsLoss`.

    Rationale (R2.F): Dice loss alone handles class imbalance well but can
    produce unstable gradients early in training when predictions are far
    from the target; BCE provides stable per-pixel supervision throughout.
    The compound loss is a well-established combination (e.g. nnU-Net).
    """

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=1.0)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice_loss(logits, targets) + (
            1 - self.dice_weight
        ) * self.bce_loss(logits, targets)
