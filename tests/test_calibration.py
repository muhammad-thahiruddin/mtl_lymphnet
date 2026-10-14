"""Tests for calibration analysis (R1.7)."""
from __future__ import annotations

import numpy as np

from mtl_lymphnet.calibration.reliability import compute_ece
from mtl_lymphnet.calibration.threshold import DEFAULT_THRESHOLD, apply_threshold, select_threshold


def test_perfect_calibration_has_near_zero_ece():
    rng = np.random.RandomState(0)
    probs = rng.uniform(0, 1, size=3000)
    # Construct targets so that P(target=1 | prob) == prob exactly (in expectation)
    targets = (rng.uniform(0, 1, size=3000) < probs).astype(int)

    ece = compute_ece(probs, targets, n_bins=10)
    assert ece < 0.05


def test_systematic_overconfidence_has_high_ece():
    # 100 samples all predicted with 0.9 confidence, but only 50% are
    # actually positive -> all fall in one bin with a large accuracy gap.
    probs = np.full(100, 0.9)
    targets = np.array([1] * 50 + [0] * 50)

    ece = compute_ece(probs, targets, n_bins=10)
    assert ece > 0.3


def test_apply_threshold_default_is_one_half():
    assert DEFAULT_THRESHOLD == 0.5
    preds = apply_threshold(np.array([0.3, 0.5, 0.51, 0.9]))
    assert preds.tolist() == [0, 1, 1, 1]


def test_select_threshold_youden_on_separable_data():
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    targets = np.array([0, 0, 0, 1, 1, 1])

    t = select_threshold(probs, targets, metric="youden")
    assert 0.3 < t <= 0.7  # any threshold in this range perfectly separates classes
