"""Custom LR scheduler: Cosine Annealing with Warm Restarts + peak-LR decay."""
from __future__ import annotations

import math

import torch


class CosineAnnealingWarmRestartsDecay(torch.optim.lr_scheduler._LRScheduler):
    """CosineAnnealingWarmRestarts with linear warmup and multiplicative decay.

    - Linear warmup for the first `warmup_epochs` epochs.
    - After warmup, behaves like `CosineAnnealingWarmRestarts(T_0, T_mult)`,
      except the peak LR decays by `gamma` at each restart.
    """

    def __init__(
        self,
        optimizer,
        T_0: int = 15,
        T_mult: int = 2,
        warmup_epochs: int = 5,
        gamma: float = 0.8,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            alpha = epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]

        t = epoch - self.warmup_epochs

        cycle_epoch = t
        current_T = self.T_0
        restart_count = 0

        while cycle_epoch >= current_T:
            cycle_epoch -= current_T
            restart_count += 1
            current_T = int(current_T * self.T_mult)

        decay = self.gamma ** restart_count
        cos_value = math.cos(math.pi * cycle_epoch / current_T)

        return [
            self.eta_min + (base_lr * decay - self.eta_min) * (1 + cos_value) / 2
            for base_lr in self.base_lrs
        ]
