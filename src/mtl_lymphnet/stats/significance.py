"""Statistical significance testing: DeLong's test, Benjamini-Hochberg FDR
correction, and Friedman/Wilcoxon pairwise comparisons.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, wilcoxon


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Returns (reject, p_corrected)."""
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    ranks = np.arange(1, n + 1)
    adjusted = np.minimum(sorted_p * n / ranks, 1.0)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    p_corrected = np.empty(n)
    p_corrected[sorted_idx] = adjusted
    reject = p_corrected <= alpha
    return reject, p_corrected


def _placement_values(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Structural components (placement values) for DeLong's test."""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    v10 = np.array(
        [(np.sum(neg_scores < p) + 0.5 * np.sum(neg_scores == p)) / len(neg_scores) for p in pos_scores]
    )
    v01 = np.array(
        [(np.sum(pos_scores > n) + 0.5 * np.sum(pos_scores == n)) / len(pos_scores) for n in neg_scores]
    )
    return v10, v01


def delong_test(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray) -> tuple[float, float, float, float]:
    """DeLong's test for comparing two correlated AUC curves.

    References: DeLong et al. (1988) Biometrics 44:837-845;
    Sun & Xu (2014) IEEE SPL 21(11):1389-1393.

    Returns (z_stat, p_value, auc_a, auc_b).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)

    m = np.sum(y_true == 1)
    n = np.sum(y_true == 0)
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan, np.nan

    v10_a, v01_a = _placement_values(y_true, y_score_a)
    v10_b, v01_b = _placement_values(y_true, y_score_b)

    auc_a = np.mean(v10_a)
    auc_b = np.mean(v10_b)

    s10_a, s01_a = np.var(v10_a, ddof=1), np.var(v01_a, ddof=1)
    s10_b, s01_b = np.var(v10_b, ddof=1), np.var(v01_b, ddof=1)

    s10_ab = np.cov(v10_a, v10_b, ddof=1)[0, 1] if len(v10_a) > 1 else 0.0
    s01_ab = np.cov(v01_a, v01_b, ddof=1)[0, 1] if len(v01_a) > 1 else 0.0

    var_auc_a = s10_a / m + s01_a / n
    var_auc_b = s10_b / m + s01_b / n
    cov_ab = s10_ab / m + s01_ab / n

    var_diff = var_auc_a + var_auc_b - 2 * cov_ab
    if var_diff <= 1e-15:
        return 0.0, 1.0, auc_a, auc_b

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2.0 * norm.sf(abs(z))
    return z, p, auc_a, auc_b


def pairwise_delong(probs_by_model: dict[str, np.ndarray], targets_by_model: dict[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise DeLong tests across all model pairs, with BH correction."""
    rows = []
    raw_p = []
    for (name_a, probs_a), (name_b, probs_b) in combinations(probs_by_model.items(), 2):
        targets = targets_by_model[name_a]
        z, p, auc_a, auc_b = delong_test(targets, probs_a, probs_b)
        raw_p.append(p if not np.isnan(p) else 1.0)
        rows.append({"comparison": f"{name_a} vs {name_b}", "auc_a": auc_a, "auc_b": auc_b, "delta_auc": auc_a - auc_b, "z_stat": z, "p_raw": p})

    if raw_p:
        _, p_corrected = benjamini_hochberg(raw_p, alpha)
        for row, p_bh in zip(rows, p_corrected):
            row["p_bh"] = p_bh
            row["significant"] = bool(p_bh < alpha)
    return pd.DataFrame(rows)


def pairwise_wilcoxon(arrays_by_model: dict[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests across all model pairs, with BH correction.

    Also runs a Friedman test first (recorded in the returned DataFrame attrs)
    when >= 3 groups are supplied, matching the manuscript's stated protocol
    (Friedman -> pairwise Wilcoxon -> Benjamini-Hochberg).
    """
    names = list(arrays_by_model.keys())
    arrays = [arrays_by_model[n] for n in names]

    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All arrays must have equal length (paired samples); got lengths {lengths}")

    friedman_stat, friedman_p = (np.nan, np.nan)
    if len(arrays) >= 3:
        friedman_stat, friedman_p = friedmanchisquare(*arrays)

    rows = []
    raw_p = []
    for (name_a, arr_a), (name_b, arr_b) in combinations(arrays_by_model.items(), 2):
        diff = arr_a - arr_b
        if np.all(diff == 0):
            stat, p = np.nan, np.nan
        else:
            stat, p = wilcoxon(arr_a, arr_b, alternative="two-sided")

        n_nonzero = np.sum(diff != 0)
        r_effect = 1.0 - (2.0 * stat) / (n_nonzero * (n_nonzero + 1) / 2) if n_nonzero > 0 and not np.isnan(stat) else 0.0

        raw_p.append(p if not np.isnan(p) else 1.0)
        rows.append({
            "comparison": f"{name_a} vs {name_b}",
            "delta_mean": arr_a.mean() - arr_b.mean(),
            "w_stat": stat,
            "p_raw": p,
            "r_effect": r_effect,
        })

    if raw_p:
        _, p_corrected = benjamini_hochberg(raw_p, alpha)
        for row, p_bh in zip(rows, p_corrected):
            row["p_bh"] = p_bh
            row["significant"] = bool(p_bh < alpha)

    df = pd.DataFrame(rows)
    df.attrs["friedman_stat"] = friedman_stat
    df.attrs["friedman_p"] = friedman_p
    return df
