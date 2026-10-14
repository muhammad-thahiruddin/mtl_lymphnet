"""Calibration analysis: Expected/Maximum Calibration Error and reliability
diagrams (R1.7).

The manuscript previously reported no calibration analysis despite a marked
sensitivity/specificity imbalance in external validation. These utilities let
the framework quantify over/under-confidence on any probability/label pair
(internal CV pooled predictions, or external validation predictions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBins:
    bin_edges: np.ndarray
    bin_confidence: np.ndarray  # mean predicted probability per bin
    bin_accuracy: np.ndarray  # observed frequency of positive class per bin
    bin_count: np.ndarray  # number of samples per bin


def compute_reliability_bins(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> ReliabilityBins:
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)

    bin_idx = np.clip(np.digitize(probs, bin_edges[1:-1], right=True), 0, n_bins - 1)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bin_count[b] = count
        if count > 0:
            bin_confidence[b] = probs[mask].mean()
            bin_accuracy[b] = targets[mask].mean()

    return ReliabilityBins(bin_edges, bin_confidence, bin_accuracy, bin_count)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: sum_b (n_b/N) * |acc_b - conf_b|."""
    bins = compute_reliability_bins(probs, targets, n_bins)
    n = bins.bin_count.sum()
    if n == 0:
        return float("nan")
    valid = bins.bin_count > 0
    weights = bins.bin_count[valid] / n
    gaps = np.abs(bins.bin_accuracy[valid] - bins.bin_confidence[valid])
    return float((weights * gaps).sum())


def compute_mce(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error: max_b |acc_b - conf_b|."""
    bins = compute_reliability_bins(probs, targets, n_bins)
    valid = bins.bin_count > 0
    if not valid.any():
        return float("nan")
    gaps = np.abs(bins.bin_accuracy[valid] - bins.bin_confidence[valid])
    return float(gaps.max())


def calibration_summary(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> dict:
    """Convenience wrapper returning ECE, MCE, and an over/under-confidence verdict."""
    ece = compute_ece(probs, targets, n_bins)
    mce = compute_mce(probs, targets, n_bins)
    bins = compute_reliability_bins(probs, targets, n_bins)
    valid = bins.bin_count > 0
    mean_signed_gap = float((bins.bin_confidence[valid] - bins.bin_accuracy[valid]).mean()) if valid.any() else float("nan")
    verdict = "overconfident" if mean_signed_gap > 0.02 else ("underconfident" if mean_signed_gap < -0.02 else "well-calibrated")
    return {"ece": ece, "mce": mce, "mean_signed_gap": mean_signed_gap, "verdict": verdict, "n_bins": n_bins}


def plot_reliability_diagram(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10, ax=None, title: str | None = None):
    """Plot a reliability diagram (confidence vs. observed accuracy) with an ECE annotation."""
    import matplotlib.pyplot as plt

    bins = compute_reliability_bins(probs, targets, n_bins)
    ece = compute_ece(probs, targets, n_bins)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    bin_centers = (bins.bin_edges[:-1] + bins.bin_edges[1:]) / 2
    valid = bins.bin_count > 0

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(
        bin_centers[valid], bins.bin_accuracy[valid], width=1.0 / n_bins * 0.9,
        edgecolor="black", alpha=0.7, label="Observed frequency",
    )
    ax.scatter(bins.bin_confidence[valid], bins.bin_accuracy[valid], color="red", zorder=5, s=20)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of positive class")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Reliability Diagram (ECE={ece:.4f})")
    ax.legend(fontsize=8)
    return ax
