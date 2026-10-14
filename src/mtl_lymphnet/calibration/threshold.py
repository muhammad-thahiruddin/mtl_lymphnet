"""Decision-threshold policy and selection utilities (R1.7).

The manuscript did not previously state how probabilities/logits were
converted into binary predictions. This module makes the policy explicit and
provides a leakage-safe way to tune a threshold when needed.

Policy
------
Unless a validation-selected threshold is explicitly requested, both
classification probabilities and segmentation sigmoid maps are thresholded at
the fixed value `DEFAULT_THRESHOLD = 0.5`, with no connected-component
filtering or morphological post-processing.

Leakage rule
------------
`select_threshold` must only ever be called with validation-split data (never
the held-out test fold / external test center). The chosen threshold is then
applied, unmodified, to test-set probabilities via `apply_threshold`.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_curve

DEFAULT_THRESHOLD = 0.5


def apply_threshold(probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Binarize probabilities/sigmoid outputs at `threshold`."""
    return (np.asarray(probs) >= threshold).astype(int)


def select_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    metric: str = "youden",
) -> float:
    """Select an operating threshold from VALIDATION data only.

    Parameters
    ----------
    metric: 'youden' (maximize sensitivity + specificity - 1) or 'f1'
            (maximize F1 score).

    Warning
    -------
    Never call this with test-set / external-center probabilities or targets;
    doing so re-introduces the exact threshold-selection leakage flagged in
    R1.7. Select on validation data, then reuse the same threshold value on
    the test set via `apply_threshold`.
    """
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(targets, probs)
        youden_j = tpr - fpr
        return float(thresholds[int(np.argmax(youden_j))])
    if metric == "f1":
        candidate_thresholds = np.unique(probs)
        best_t, best_f1 = DEFAULT_THRESHOLD, -1.0
        for t in candidate_thresholds:
            f1 = f1_score(targets, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t
    raise ValueError(f"Unknown threshold-selection metric: {metric}")
