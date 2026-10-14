"""Early stopping on a monitored (validation) metric."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn


class EarlyStopping:
    """Early stopping with patience on a monitored metric.

    IMPORTANT (R1.1): the `score` passed to `__call__` must always come from
    the INNER validation split, never from the held-out test fold or an
    external test center. See `mtl_lymphnet.engine.trainer.fit`.
    """

    def __init__(self, patience: int = 20, mode: str = "min", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
        self.best_state: Optional[dict] = None
        self.best_epoch: int = 0

    def __call__(self, score: float, model: nn.Module, epoch: int = 0) -> None:
        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < (self.best_score - self.delta)
        return score > (self.best_score + self.delta)

    def load_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
