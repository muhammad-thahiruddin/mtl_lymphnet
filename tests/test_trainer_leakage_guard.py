"""Tests for the trainer's leakage guard (R1.1).

`fit()` must refuse to run if `test_loader` is literally the same object as
`val_loader` -- the exact mistake present in the original notebook, where the
held-out fold was passed in as `val_loader` and used for early stopping.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from mtl_lymphnet.config import Config
from mtl_lymphnet.engine.trainer import fit
from mtl_lymphnet.losses.weighting import build_criterion


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)


def test_fit_raises_if_test_loader_is_val_loader(tmp_path):
    cfg = Config(device="cpu", num_epochs=1, output_dir=str(tmp_path))
    model = _DummyModel()
    criterion = build_criterion("fixed", pos_weight=1.0)
    shared_loader = object()  # identity check happens before it is ever used

    with pytest.raises(ValueError, match="test_loader must not be the same object as val_loader"):
        fit(
            model,
            train_loader=object(),
            val_loader=shared_loader,
            config=cfg,
            criterion=criterion,
            test_loader=shared_loader,
        )


def test_fit_allows_distinct_val_and_test_loaders_past_the_guard(tmp_path):
    """A distinct (even if otherwise invalid) test_loader must NOT trigger the
    leakage guard -- only literal identity with val_loader should."""
    cfg = Config(device="cpu", num_epochs=1, output_dir=str(tmp_path))
    model = _DummyModel()
    criterion = build_criterion("fixed", pos_weight=1.0)

    with pytest.raises(Exception) as exc_info:
        # Fails later (dummy loaders have no real .dataset/batches), proving
        # the ValueError leakage guard itself was not what triggered this.
        fit(
            model,
            train_loader=object(),
            val_loader=object(),
            config=cfg,
            criterion=criterion,
            test_loader=object(),
        )

    assert "test_loader must not be the same object as val_loader" not in str(exc_info.value)

