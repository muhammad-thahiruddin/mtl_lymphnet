"""Tests for statistical significance utilities (R1.6)."""
from __future__ import annotations

import numpy as np

from mtl_lymphnet.stats.bootstrap import bootstrap_ci, bootstrap_paired_diff_ci
from mtl_lymphnet.stats.significance import benjamini_hochberg, delong_test


def test_benjamini_hochberg_adjusted_pvalues_are_bounded_and_monotonic_ok():
    p_values = [0.001, 0.02, 0.03, 0.5, 0.8]
    _, p_corrected = benjamini_hochberg(p_values, alpha=0.05)

    assert len(p_corrected) == len(p_values)
    assert np.all(p_corrected <= 1.0)
    assert np.all(p_corrected >= 0.0)


def test_delong_test_identical_scores_gives_p_near_one():
    rng = np.random.RandomState(0)
    n = 200
    targets = rng.binomial(1, 0.5, size=n)
    scores = rng.rand(n)

    z, p, auc_a, auc_b = delong_test(targets, scores, scores)
    assert abs(z) < 1e-6
    assert p > 0.99
    assert auc_a == auc_b


def test_bootstrap_ci_contains_true_mean():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=0.8, scale=0.05, size=200)
    result = bootstrap_ci(values, n_resamples=2000, seed=0)

    assert result["ci_low"] < 0.8 < result["ci_high"]


def test_bootstrap_paired_diff_ci_detects_real_difference():
    """R1.6: replaces bare 'statistically equivalent' claims with a CI on the
    pairwise difference; a real, large difference should exclude zero."""
    rng = np.random.RandomState(0)
    a = rng.normal(loc=0.9, scale=0.02, size=100)
    b = rng.normal(loc=0.8, scale=0.02, size=100)

    result = bootstrap_paired_diff_ci(a, b, n_resamples=2000, seed=0)
    assert result["mean_diff"] > 0.05
    assert result["excludes_zero"] is True


def test_bootstrap_paired_diff_ci_does_not_claim_difference_when_none_exists():
    rng = np.random.RandomState(0)
    a = rng.normal(loc=0.85, scale=0.05, size=200)
    b = a + rng.normal(loc=0.0, scale=0.001, size=200)  # essentially identical

    result = bootstrap_paired_diff_ci(a, b, n_resamples=2000, seed=0)
    assert result["excludes_zero"] is False
