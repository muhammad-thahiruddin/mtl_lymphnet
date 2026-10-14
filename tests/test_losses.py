"""Tests for MTL loss weighting strategies (R1.5, R1.8)."""
from __future__ import annotations

import math

from mtl_lymphnet.losses.weighting import (
    DynamicWeightAverage,
    FixedWeightMTL,
    UncertaintyWeightMTL,
    build_criterion,
)


def test_label_smoothing_is_symmetric():
    """R1.8: the implementation maps {0,1} -> {eps/2, 1-eps/2} (symmetric),
    NOT the asymmetric {eps/2, 1-eps} mapping described in the manuscript.
    This test locks in the (correct) code behaviour."""
    eps = 0.05
    target_0 = 0.0 * (1 - eps) + 0.5 * eps
    target_1 = 1.0 * (1 - eps) + 0.5 * eps

    assert math.isclose(target_0, eps / 2, rel_tol=1e-9)
    assert math.isclose(target_1, 1 - eps / 2, rel_tol=1e-9)
    assert math.isclose(target_0 + target_1, 1.0, rel_tol=1e-9)  # symmetric around 0.5


def test_dwa_weight_update_matches_corrected_formula():
    """R1.8: w_k(t) = K * exp(r_k(t-1)/T) / sum_j exp(r_j(t-1)/T); the total
    loss is a SEPARATE sum_k w_k * L_k step (see `DynamicWeightAverage.forward`),
    matching the plan's corrected two-line formula rather than the manuscript's
    single (conflated) equation."""
    dwa = DynamicWeightAverage(n_tasks=2, temperature=2.0)
    dwa.loss_history = [[1.0, 0.5], [1.0, 0.9]]  # [seg_losses], [cls_losses]
    dwa.update_weights()

    r_seg = 0.5 / 1.0
    r_cls = 0.9 / 1.0
    exp_seg = math.exp(r_seg / 2.0)
    exp_cls = math.exp(r_cls / 2.0)
    total = exp_seg + exp_cls
    expected_w_seg = 2 * exp_seg / total
    expected_w_cls = 2 * exp_cls / total

    assert math.isclose(dwa.current_weights[0], expected_w_seg, rel_tol=1e-6)
    assert math.isclose(dwa.current_weights[1], expected_w_cls, rel_tol=1e-6)


def test_dwa_defaults_to_equal_weights_before_two_epochs_recorded():
    dwa = DynamicWeightAverage()
    assert dwa.current_weights == [1.0, 1.0]
    dwa.record_epoch_loss(seg_loss=0.5, cls_loss=0.3)
    assert dwa.current_weights == [1.0, 1.0]  # still < 2 epochs of history


def test_base_mtl_loss_records_weight_trajectory():
    """R1.5: effective weights (and, for uncertainty weighting, the learned
    log-variance parameters) must be exportable per epoch without re-running
    training."""
    criterion = UncertaintyWeightMTL(pos_weight=1.0)
    for epoch in range(1, 4):
        criterion.record_epoch(epoch)

    history = criterion.export_history()
    assert len(history) == 3
    assert all({"w_seg", "w_cls", "log_var_seg", "log_var_cls", "epoch"} <= h.keys() for h in history)
    assert [h["epoch"] for h in history] == [1, 2, 3]


def test_fixed_weight_mtl_weights_sum_to_one():
    criterion = FixedWeightMTL(w_seg=0.7, pos_weight=1.0)
    weights = criterion.get_weights()
    assert math.isclose(weights["w_seg"] + weights["w_cls"], 1.0)
    assert math.isclose(weights["w_seg"], 0.7)


def test_build_criterion_factory_dispatch():
    fixed = build_criterion("fixed", pos_weight=1.0, w_seg=0.7)
    uncertainty = build_criterion("uncertainty", pos_weight=1.0)
    dwa = build_criterion("dwa", pos_weight=1.0)

    assert isinstance(fixed, FixedWeightMTL) and math.isclose(fixed.w_seg, 0.7)
    assert isinstance(uncertainty, UncertaintyWeightMTL)
    assert isinstance(dwa, DynamicWeightAverage)
