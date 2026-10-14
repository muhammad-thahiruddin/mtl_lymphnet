"""Multi-task loss weighting strategies.

Verified against implementation_plan_1.md R1.8:
    - Label smoothing here maps {0,1} -> {eps/2, 1-eps/2}, which is
      SYMMETRIC (e.g. eps=0.05 -> {0.025, 0.975}). The manuscript text
      describing an asymmetric {eps/2, 1-eps} mapping does not match this
      implementation; the manuscript equation is what needs correcting, not
      this code.
    - `DynamicWeightAverage` already separates the task-weight update
      (`update_weights`, a softmax over loss ratios) from the total loss
      (`w_k(t) * L_k` summed over tasks) -- matching the corrected two-line
      formula in the plan, not the single (incorrect) equation in the
      manuscript.

R1.5 (uncertainty parameters / weight trajectories): every strategy below
subclasses `BaseMTLLoss`, which accumulates a per-epoch trajectory via
`record_epoch()` and exposes it through `export_history()`. Callers (see
`mtl_lymphnet.engine.trainer.fit`) call `record_epoch(epoch)` once per epoch
so the effective weights (and, for uncertainty weighting, the learned
log-variance parameters) can be exported and plotted without re-running
training.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .segmentation import CompoundSegLoss


class BaseMTLLoss(nn.Module):
    """Common interface for all MTL weighting strategies."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict] = []

    def get_weights(self) -> dict[str, float]:
        raise NotImplementedError

    def record_epoch(self, epoch: int) -> None:
        """Append the current effective weights to the trajectory history."""
        entry = dict(self.get_weights())
        entry["epoch"] = epoch
        self.history.append(entry)

    def export_history(self) -> list[dict]:
        """Return the full per-epoch weight trajectory (R1.5)."""
        return list(self.history)


class FixedWeightMTL(BaseMTLLoss):
    """Fixed-weight multi-task loss: `L = w_seg * L_seg + w_cls * L_cls`."""

    def __init__(self, w_seg: float = 0.5, pos_weight: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.w_seg = w_seg
        self.w_cls = 1.0 - w_seg
        self.label_smoothing = label_smoothing
        self.seg_loss_fn = CompoundSegLoss()
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, seg_logits, seg_targets, cls_logits, cls_targets):
        l_seg = self.seg_loss_fn(seg_logits, seg_targets)
        cls_t = cls_targets.float().view(-1)
        if self.label_smoothing > 0:
            cls_t = cls_t * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        l_cls = self.cls_loss_fn(cls_logits.float().view(-1), cls_t)
        loss = self.w_seg * l_seg + self.w_cls * l_cls
        return loss, l_seg.detach(), l_cls.detach()

    def get_weights(self) -> dict[str, float]:
        return {"w_seg": self.w_seg, "w_cls": self.w_cls}


class UncertaintyWeightMTL(BaseMTLLoss):
    """Kendall et al., CVPR 2018 -- homoscedastic uncertainty weighting.

    `L = (1 / 2*sigma_seg^2) * L_seg + (1 / 2*sigma_cls^2) * L_cls
         + log(sigma_seg) + log(sigma_cls)`

    Parameterized as log(sigma^2) for numerical stability.
    """

    def __init__(self, pos_weight: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.label_smoothing = label_smoothing
        self.seg_loss_fn = CompoundSegLoss()
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, seg_logits, seg_targets, cls_logits, cls_targets):
        l_seg = self.seg_loss_fn(seg_logits, seg_targets)
        cls_t = cls_targets.float().view(-1)
        if self.label_smoothing > 0:
            cls_t = cls_t * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        l_cls = self.cls_loss_fn(cls_logits.float().view(-1), cls_t)

        precision_seg = torch.exp(-self.log_var_seg)
        precision_cls = torch.exp(-self.log_var_cls)

        loss = (
            precision_seg * l_seg + 0.5 * self.log_var_seg
            + precision_cls * l_cls + 0.5 * self.log_var_cls
        )
        return loss, l_seg.detach(), l_cls.detach()

    def get_weights(self) -> dict[str, float]:
        return {
            "w_seg": torch.exp(-self.log_var_seg).item(),
            "w_cls": torch.exp(-self.log_var_cls).item(),
            "log_var_seg": self.log_var_seg.item(),
            "log_var_cls": self.log_var_cls.item(),
        }


class DynamicWeightAverage(BaseMTLLoss):
    """Liu et al., CVPR 2019 -- Dynamic Weight Average.

    Two-step formulation (see module docstring for the R1.8 correction):
        w_k(t) = K * exp(r_k(t-1) / T) / sum_j exp(r_j(t-1) / T)
        L_total = sum_k w_k(t) * L_k
    """

    def __init__(
        self,
        n_tasks: int = 2,
        temperature: float = 2.0,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.temperature = temperature
        self.seg_loss_fn = CompoundSegLoss()
        self.label_smoothing = label_smoothing
        self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.loss_history: list[list[float]] = [[], []]  # seg, cls
        self.current_weights = [1.0, 1.0]

    def update_weights(self) -> None:
        """Call once per epoch (after `record_epoch_loss`) to update `w_k(t)`."""
        if len(self.loss_history[0]) < 2:
            self.current_weights = [1.0, 1.0]
            return

        ratios = []
        for k in range(self.n_tasks):
            r = self.loss_history[k][-1] / (self.loss_history[k][-2] + 1e-8)
            ratios.append(r / self.temperature)

        exp_ratios = np.exp(ratios)
        weights = self.n_tasks * exp_ratios / exp_ratios.sum()
        self.current_weights = weights.tolist()

    def forward(self, seg_logits, seg_targets, cls_logits, cls_targets):
        l_seg = self.seg_loss_fn(seg_logits, seg_targets)
        cls_t = cls_targets.float().view(-1)
        if self.label_smoothing > 0:
            cls_t = cls_t * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        l_cls = self.cls_loss_fn(cls_logits.float().view(-1), cls_t)

        w_seg, w_cls = self.current_weights
        loss = w_seg * l_seg + w_cls * l_cls
        return loss, l_seg.detach(), l_cls.detach()

    def record_epoch_loss(self, seg_loss: float, cls_loss: float) -> None:
        """Feed the epoch's mean task losses so weights can update (call
        this once per epoch instead of / in addition to `record_epoch`)."""
        self.loss_history[0].append(seg_loss)
        self.loss_history[1].append(cls_loss)
        self.update_weights()

    def get_weights(self) -> dict[str, float]:
        return {"w_seg": self.current_weights[0], "w_cls": self.current_weights[1]}


def build_criterion(
    strategy: str = "uncertainty",
    pos_weight: float = 1.0,
    w_seg: float = 0.5,
    label_smoothing: float = 0.0,
) -> BaseMTLLoss:
    """Factory for MTL loss criteria."""
    if strategy == "fixed":
        return FixedWeightMTL(w_seg=w_seg, pos_weight=pos_weight, label_smoothing=label_smoothing)
    if strategy == "uncertainty":
        return UncertaintyWeightMTL(pos_weight=pos_weight, label_smoothing=label_smoothing)
    if strategy == "dwa":
        return DynamicWeightAverage(pos_weight=pos_weight, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown strategy: {strategy}")
