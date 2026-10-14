"""LaTeX table generation for manuscript results.

R1.3: the "Combined" score (average of Dice and AUC) is REMOVED from every
table. It lacked clinical/statistical justification because Dice and AUC
measure different tasks on different scales; the manuscript's related
"99.5% of separate-model performance" claim was misleading and should not be
reproduced. Tables report task metrics separately.

Multi-seed awareness (R1.6):
- Internal CV table reports per-seed breakdown + sub-means + pooled total.
- Ablation tables report proposed variant with both "all seeds" and "primary"
  rows so that ablation comparison is fair (n=5 vs n=5).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "seg_dice": "Dice", "seg_iou": "IoU", "seg_hd95": "HD95",
    "cls_auc": "AUC", "cls_accuracy": "Acc", "cls_sensitivity": "Sens",
    "cls_specificity": "Spec", "cls_f1": "F1", "cls_mcc": "MCC",
}

DEFAULT_METRICS = ["seg_dice", "seg_iou", "cls_auc", "cls_accuracy",
                   "cls_sensitivity", "cls_specificity", "cls_f1", "cls_mcc"]


def _fmt(vals: list[float]) -> str:
    """Format mean +/- std from a list of values."""
    return f"{np.nanmean(vals):.4f} +/- {np.nanstd(vals):.4f}"


def _ensure_hd95(metric_order: list[str], sample: dict) -> list[str]:
    """Insert seg_hd95 after seg_iou if present in data."""
    if sample and "seg_hd95" in sample and "seg_hd95" not in metric_order:
        metric_order = list(metric_order)
        metric_order.insert(2, "seg_hd95")
    return metric_order


# ═══════════════════════════════════════════════════════════════════════════ #
#  LaTeX export helper
# ═══════════════════════════════════════════════════════════════════════════ #


def to_latex_booktabs(df: pd.DataFrame, caption: str, label: str,
                      filename: str, tables_dir: str,
                      note: str | None = None,
                      midrule_after: list[int] | None = None) -> str:
    """Convert a DataFrame into a booktabs-style LaTeX table and save it.

    Parameters
    ----------
    midrule_after : list[int] | None
        Zero-based row indices after which a ``\\cmidrule`` is inserted
        (useful for separating seed groups or all/primary rows).
    """
    n_cols = len(df.columns)
    col_spec = "l" + "c" * n_cols
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{" + col_spec + "}",
        r"    \toprule",
        "    " + " & ".join([df.index.name or ""] + list(df.columns)) + r" \\",
        r"    \midrule",
    ]
    midrule_set = set(midrule_after or [])
    for row_idx, (idx, row) in enumerate(df.iterrows()):
        vals = [str(idx)] + [str(v) for v in row.values]
        lines.append("    " + " & ".join(vals) + r" \\")
        if row_idx in midrule_set:
            lines.append(r"    \cmidrule(lr){1-" + str(n_cols + 1) + "}")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    if note:
        lines.append(r"  \vspace{2pt}")
        lines.append(r"  \footnotesize{" + note + "}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    os.makedirs(tables_dir, exist_ok=True)
    filepath = os.path.join(tables_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_str)
    return latex_str


# ═══════════════════════════════════════════════════════════════════════════ #
#  Table 2: Internal CV -- multi-seed (R1.6)
# ═══════════════════════════════════════════════════════════════════════════ #


def _get_primary_seed_folds(exp1_data: dict) -> list[dict]:
    """Extract fold results from the primary (first) seed only."""
    results_by_seed = exp1_data.get("results_by_seed", {})
    seeds = exp1_data.get("seeds", [])
    if results_by_seed and seeds:
        return results_by_seed[seeds[0]]
    return exp1_data["fold_results"]


def build_internal_cv_table(fold_results: list[dict],
                            metric_order: list[str] | None = None) -> pd.DataFrame:
    """Legacy: flat per-fold table (backward compatible)."""
    metric_order = list(metric_order or DEFAULT_METRICS)
    metric_order = _ensure_hd95(metric_order, fold_results[0] if fold_results else {})

    rows = []
    for i, fr in enumerate(fold_results):
        row = {"Fold": f"Fold {i}"}
        for m in metric_order:
            row[DISPLAY_NAMES.get(m, m)] = fr.get(m, np.nan)
        rows.append(row)

    mean_row = {"Fold": "Mean +/- Std"}
    for m in metric_order:
        vals = [fr.get(m, np.nan) for fr in fold_results]
        mean_row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
    rows.append(mean_row)

    return pd.DataFrame(rows).set_index("Fold")


def build_internal_cv_table_detailed(exp1_data: dict,
                                     metric_order: list[str] | None = None) -> pd.DataFrame:
    """Detailed table: per-seed, per-fold + sub-means + pooled.

    Layout:
        Seed 42, Fold 1-5, sub mean +/- std
        Seed 43, Fold 1-5, sub mean +/- std
        Seed 44, Fold 1-5, sub mean +/- std
        Pooled, mean +/- std (all 15 runs)
    """
    metric_order = list(metric_order or DEFAULT_METRICS)
    sample = exp1_data["fold_results"][0] if exp1_data["fold_results"] else {}
    metric_order = _ensure_hd95(metric_order, sample)

    results_by_seed = exp1_data.get("results_by_seed", {})
    seeds = exp1_data.get("seeds", [])
    all_folds = exp1_data["fold_results"]
    display_cols = [DISPLAY_NAMES.get(m, m) for m in metric_order]

    rows = []
    midrule_after = []

    if results_by_seed and len(seeds) > 1:
        for seed in seeds:
            folds = results_by_seed[seed]
            for i, fr in enumerate(folds):
                row = {"Seed": str(seed) if i == 0 else "",
                       "Fold": str(i + 1)}
                for m in metric_order:
                    row[DISPLAY_NAMES.get(m, m)] = f"{fr.get(m, np.nan):.4f}"
                rows.append(row)
            # Sub-mean row
            sub_row = {"Seed": "", "Fold": "Mean"}
            for m in metric_order:
                vals = [fr.get(m, np.nan) for fr in folds]
                sub_row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
            rows.append(sub_row)
            midrule_after.append(len(rows) - 1)

        # Pooled row
        pooled_row = {"Seed": "Pooled", "Fold": "All"}
        for m in metric_order:
            vals = [fr.get(m, np.nan) for fr in all_folds]
            pooled_row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
        rows.append(pooled_row)
    else:
        # Single seed fallback
        for i, fr in enumerate(all_folds):
            row = {"Seed": "", "Fold": str(i + 1)}
            for m in metric_order:
                row[DISPLAY_NAMES.get(m, m)] = f"{fr.get(m, np.nan):.4f}"
            rows.append(row)
        mean_row = {"Seed": "", "Fold": "Mean"}
        for m in metric_order:
            vals = [fr.get(m, np.nan) for fr in all_folds]
            mean_row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
        rows.append(mean_row)

    df = pd.DataFrame(rows)
    df.index.name = ""
    return df, midrule_after


def build_internal_cv_table_compact(exp1_data: dict,
                                    metric_order: list[str] | None = None) -> pd.DataFrame:
    """Compact table: per-seed sub-means + pooled only (4 rows for 3 seeds).

    Layout:
        Seed 42, mean +/- std
        Seed 43, mean +/- std
        Seed 44, mean +/- std
        Pooled,  mean +/- std (all 15 runs)
    """
    metric_order = list(metric_order or DEFAULT_METRICS)
    sample = exp1_data["fold_results"][0] if exp1_data["fold_results"] else {}
    metric_order = _ensure_hd95(metric_order, sample)

    results_by_seed = exp1_data.get("results_by_seed", {})
    seeds = exp1_data.get("seeds", [])
    all_folds = exp1_data["fold_results"]

    rows = []
    midrule_after = []

    if results_by_seed and len(seeds) > 1:
        for seed in seeds:
            folds = results_by_seed[seed]
            row = {"Seed": str(seed), "N": str(len(folds))}
            for m in metric_order:
                vals = [fr.get(m, np.nan) for fr in folds]
                row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
            rows.append(row)
        midrule_after.append(len(rows) - 1)

        pooled_row = {"Seed": "Pooled", "N": str(len(all_folds))}
        for m in metric_order:
            vals = [fr.get(m, np.nan) for fr in all_folds]
            pooled_row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
        rows.append(pooled_row)
    else:
        row = {"Seed": "All", "N": str(len(all_folds))}
        for m in metric_order:
            vals = [fr.get(m, np.nan) for fr in all_folds]
            row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index.name = ""
    return df, midrule_after


# ═══════════════════════════════════════════════════════════════════════════ #
#  Table 3: External Validation
# ═══════════════════════════════════════════════════════════════════════════ #


def build_external_validation_table(exp1_summary: dict, ext_a: dict,
                                    ext_b: dict,
                                    extra_variants: Optional[dict[str, dict]] = None) -> pd.DataFrame:
    """Bidirectional external validation table.

    `extra_variants`, if given, is ``{row_label: ext_result_dict}`` for
    additional backbone / loss-strategy sweep configurations produced by the
    Bug B1 fix (see ``experiments/external_validation.py`` -- results stored
    under ``results["exp2_variants"]`` when ``--backbone`` / ``--loss-strategy``
    / ``--w-seg`` overrides are used). Appended as extra rows after the two
    primary directions, so the manuscript's backbone/strategy trade-off table
    (R1.4, R1.5) can be built from the same function without special-casing.
    """
    metric_keys = ["seg_dice", "seg_iou", "cls_auc", "cls_accuracy",
                   "cls_sensitivity", "cls_specificity", "cls_f1"]
    if "seg_hd95" in ext_a["results"]:
        metric_keys.insert(2, "seg_hd95")

    rows = []
    row_int = {"Scenario": "Internal CV"}
    for m in metric_keys:
        row_int[DISPLAY_NAMES.get(m, m)] = exp1_summary.get(m, {}).get("mean", np.nan)
    rows.append(row_int)

    for label, ext in [("C1 -> C2", ext_a), ("C2 -> C1", ext_b)]:
        row = {"Scenario": label}
        for m in metric_keys:
            row[DISPLAY_NAMES.get(m, m)] = ext["results"].get(m, np.nan)
        rows.append(row)

    for label, ext in (extra_variants or {}).items():
        row = {"Scenario": label}
        for m in metric_keys:
            row[DISPLAY_NAMES.get(m, m)] = ext.get("results", {}).get(m, np.nan)
        rows.append(row)

    return pd.DataFrame(rows).set_index("Scenario").round(4)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Ablation helper: dual-row for proposed (all seeds + primary)
# ═══════════════════════════════════════════════════════════════════════════ #


def _ablation_row(label: str, folds: list[dict], metrics: list[str],
                  extra_cols: Optional[dict] = None) -> dict:
    """Build a single ablation row from fold results."""
    row = extra_cols.copy() if extra_cols else {}
    for m in metrics:
        vals = [f.get(m, np.nan) for f in folds]
        row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
    return row


def _dual_proposed_rows(index_name: str,
                        proposed_label: str,
                        all_folds: list[dict],
                        primary_folds: list[dict],
                        metrics: list[str],
                        extra_cols_fn=None) -> list[tuple[str, dict]]:
    """Generate two rows for the proposed variant: all seeds + primary.

    Returns list of (index_value, row_dict) tuples.
    """
    n_all = len(all_folds)
    n_pri = len(primary_folds)

    extra_all = extra_cols_fn(all_folds) if extra_cols_fn else {}
    extra_pri = extra_cols_fn(primary_folds) if extra_cols_fn else {}

    row_all = _ablation_row(
        f"{proposed_label} (all seeds, n={n_all})", all_folds, metrics, extra_all)
    row_pri = _ablation_row(
        f"{proposed_label} (primary, n={n_pri})", primary_folds, metrics, extra_pri)

    return [
        (f"{proposed_label} (all seeds, n={n_all})", row_all),
        (f"{proposed_label} (primary, n={n_pri})", row_pri),
    ]


# ═══════════════════════════════════════════════════════════════════════════ #
#  Table 4: MTL ablation
# ═══════════════════════════════════════════════════════════════════════════ #


def build_mtl_ablation_table(exp3_results: dict,
                             exp1_data: Optional[dict] = None) -> pd.DataFrame:
    """MTL vs Seg-Only vs Cls-Only. No 'Combined' column (R1.3).

    If exp1_data is provided, the MTL row is split into all-seeds and primary.
    """
    metrics = ["seg_dice", "seg_iou", "cls_auc", "cls_f1"]

    rows = []
    midrule_after = []

    # MTL proposed
    mtl_folds = exp3_results["mtl"]
    if exp1_data and exp1_data.get("results_by_seed"):
        primary_folds = _get_primary_seed_folds(exp1_data)
        for idx_label, row in _dual_proposed_rows("Model", "MTL", mtl_folds, primary_folds, metrics):
            rows.append({"Model": idx_label, **row})
        midrule_after.append(len(rows) - 1)
    else:
        row = {"Model": f"MTL (n={len(mtl_folds)})"}
        for m in metrics:
            vals = [f.get(m, np.nan) for f in mtl_folds]
            row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
        rows.append(row)

    # Seg-Only, Cls-Only
    for variant, label in [("seg_only", "Seg-Only"), ("cls_only", "Cls-Only")]:
        folds = exp3_results[variant]
        row = {"Model": f"{label} (n={len(folds)})"}
        for m in metrics:
            is_seg = m.startswith("seg_")
            is_cls = m.startswith("cls_")
            if (variant == "seg_only" and is_cls) or (variant == "cls_only" and is_seg):
                row[DISPLAY_NAMES.get(m, m)] = "--"
            else:
                vals = [f.get(m, np.nan) for f in folds]
                row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df, midrule_after


# ═══════════════════════════════════════════════════════════════════════════ #
#  Table 5: Backbone ablation
# ═══════════════════════════════════════════════════════════════════════════ #


def build_backbone_ablation_table(exp4_results: dict,
                                  exp1_data: Optional[dict] = None,
                                  backbone_order: list[str] | None = None) -> pd.DataFrame:
    if backbone_order is None:
        from ..models.backbones import BACKBONES, PROPOSED_KEY
        backbone_order = [PROPOSED_KEY] + [k for k in BACKBONES if k != PROPOSED_KEY]
        proposed_key = PROPOSED_KEY
    else:
        proposed_key = backbone_order[0]

    metrics = ["seg_dice", "cls_auc"]

    def _extra(folds):
        extra = {}
        extra["Params (M)"] = f"{folds[0].get('params', 0) / 1e6:.2f}"
        if "infer_gpu_ms" in folds[0]:
            extra["GPU (ms)"] = f"{np.mean([f.get('infer_gpu_ms', np.nan) for f in folds]):.1f}"
        extra["CPU (ms)"] = f"{np.mean([f.get('infer_cpu_ms', np.nan) for f in folds]):.1f}"
        return extra

    rows = []
    midrule_after = []

    for bb_name in backbone_order:
        if bb_name not in exp4_results:
            continue
        folds = exp4_results[bb_name]

        if bb_name == proposed_key and exp1_data and exp1_data.get("results_by_seed"):
            primary_folds = _get_primary_seed_folds(exp1_data)
            for idx_label, row in _dual_proposed_rows(
                    "Backbone", bb_name, folds, primary_folds, metrics, _extra):
                rows.append({"Backbone": idx_label, **row})
            midrule_after.append(len(rows) - 1)
        else:
            n = len(folds)
            row = {"Backbone": f"{bb_name} (n={n})"}
            row.update(_extra(folds))
            for m in metrics:
                vals = [f.get(m, np.nan) for f in folds]
                row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("Backbone")
    return df, midrule_after


# ═══════════════════════════════════════════════════════════════════════════ #
#  Table 6: Loss weighting ablation
# ═══════════════════════════════════════════════════════════════════════════ #


def build_loss_ablation_table(exp5_results: dict,
                              exp1_data: Optional[dict] = None,
                              strategy_order: list[str] | None = None) -> pd.DataFrame:
    """Loss weighting strategy comparison. No 'Combined' column (R1.3)."""
    strategy_order = strategy_order or ["Uncertainty", "Fixed (0.5/0.5)",
                                        "Fixed (0.7/0.3)", "DWA"]
    # The proposed strategy is the one that matches exp1 (Uncertainty)
    proposed_strategy = "Uncertainty"
    metrics = ["seg_dice", "seg_iou", "cls_auc", "cls_f1"]

    rows = []
    midrule_after = []

    for strat_name in strategy_order:
        if strat_name not in exp5_results:
            continue
        folds = exp5_results[strat_name]

        if strat_name == proposed_strategy and exp1_data and exp1_data.get("results_by_seed"):
            primary_folds = _get_primary_seed_folds(exp1_data)
            for idx_label, row in _dual_proposed_rows(
                    "Strategy", strat_name, folds, primary_folds, metrics):
                rows.append({"Strategy": idx_label, **row})
            midrule_after.append(len(rows) - 1)
        else:
            n = len(folds)
            row = {"Strategy": f"{strat_name} (n={n})"}
            for m in metrics:
                vals = [f.get(m, np.nan) for f in folds]
                row[DISPLAY_NAMES.get(m, m)] = _fmt(vals)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")
    return df, midrule_after


# ═══════════════════════════════════════════════════════════════════════════ #
#  Weight trajectory table (unchanged)
# ═══════════════════════════════════════════════════════════════════════════ #


def build_weight_trajectory_table(fold_results: list[dict]) -> pd.DataFrame:
    """Flatten `_weight_history` across folds into a single DataFrame for
    plotting/export (R1.5: learned uncertainty parameters and effective loss
    weights across epochs)."""
    rows = []
    for fold_idx, fr in enumerate(fold_results):
        for entry in fr.get("_weight_history", []):
            rows.append({"fold": fold_idx, **entry})
    return pd.DataFrame(rows)
