"""Bootstrap confidence intervals and multi-seed aggregation.

R1.6 requires: (1) replacing bare "statistically equivalent" claims with
confidence intervals on pairwise differences, and (2) accounting for training
variability via multiple random seeds. This module provides both.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Generic percentile-bootstrap CI for a 1D array of per-item values
    (e.g. per-image Dice, or per-fold metric values)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    n = len(values)

    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if n == 1:
        v = float(values[0])
        return {"mean": v, "ci_low": v, "ci_high": v}

    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    alpha = (1 - confidence_level) / 2
    lo, hi = np.percentile(boot_stats, [100 * alpha, 100 * (1 - alpha)])
    return {"mean": float(statistic(values)), "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_auc_ci(
    probs: np.ndarray,
    targets: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Stratified-resample bootstrap CI for AUC on pooled probabilities/targets."""
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(targets)

    boot_aucs = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        t, p = targets[idx], probs[idx]
        if len(np.unique(t)) >= 2:
            boot_aucs.append(roc_auc_score(t, p))
    boot_aucs = np.array(boot_aucs)

    alpha = (1 - confidence_level) / 2
    lo, hi = np.percentile(boot_aucs, [100 * alpha, 100 * (1 - alpha)])
    return {"mean": float(roc_auc_score(targets, probs)), "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_paired_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for a PAIRED difference (values_a - values_b), e.g. Delta-Dice
    or Delta-AUC between two models evaluated on the same items/folds (R1.6).

    `values_a` and `values_b` must be aligned (same length, same
    image/fold order).
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length (paired samples).")

    rng = np.random.default_rng(seed)
    n = len(values_a)
    diffs = values_a - values_b

    boot_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[i] = diffs[idx].mean()

    alpha = (1 - confidence_level) / 2
    lo, hi = np.percentile(boot_diffs, [100 * alpha, 100 * (1 - alpha)])
    mean_diff = float(diffs.mean())
    return {
        "mean_diff": mean_diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def aggregate_multi_seed(results_by_seed: dict[int, list[dict]], skip_prefix: str = "_") -> dict:
    """Aggregate fold-level result dicts across multiple random seeds (R1.6).

    Parameters
    ----------
    results_by_seed: {seed: [fold_result_dict, ...]} -- typically the output
        of running the same experiment once per seed.

    Returns
    -------
    dict with per-metric {mean, std, values, n_seeds, n_folds_per_seed} pooled
    across all (seed x fold) runs, so variance reflects both cross-validation
    spread and training-stochasticity (seed) spread.
    """
    all_results = [r for fold_list in results_by_seed.values() for r in fold_list]
    if not all_results:
        return {}

    keys = [k for k in all_results[0] if not k.startswith(skip_prefix)]
    summary = {}
    for k in keys:
        vals = [r[k] for r in all_results if k in r and isinstance(r[k], (int, float, np.floating))]
        if not vals:
            continue
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals],
        }
    summary["_n_seeds"] = len(results_by_seed)
    summary["_n_folds_per_seed"] = len(next(iter(results_by_seed.values())))
    return summary
