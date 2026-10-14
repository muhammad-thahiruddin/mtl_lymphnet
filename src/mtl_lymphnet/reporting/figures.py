"""Reusable figure-generation functions for manuscript results.

Ports the original notebook's plotting cells into parametrized functions
(data + output path in, saved figure out) so they can be called from the CLI,
tests, or ad-hoc scripts without re-running the whole notebook.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..calibration.reliability import compute_ece, plot_reliability_diagram

PUB_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
}


def _save(fig, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_kfold_boxplot(fold_results: list[dict], output_path: str):
    """Box plots of segmentation and classification metrics across folds."""
    seg_keys = [k for k in fold_results[0] if k.startswith("seg_") and not k.startswith("_")]
    cls_keys = [k for k in fold_results[0] if k.startswith("cls_") and not k.startswith("_")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    seg_data = [[fr[k] for fr in fold_results] for k in seg_keys]
    if seg_data:
        axes[0].boxplot(seg_data, tick_labels=seg_keys, patch_artist=True,
                         boxprops=dict(facecolor="#2196F3", alpha=0.6))
        for i, vals in enumerate(seg_data):
            axes[0].scatter([i + 1] * len(vals), vals, color="#1565C0", s=30, zorder=5)
        axes[0].set_title("Segmentation Metrics", fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)

    cls_data = [[fr[k] for fr in fold_results] for k in cls_keys]
    if cls_data:
        axes[1].boxplot(cls_data, tick_labels=cls_keys, patch_artist=True,
                         boxprops=dict(facecolor="#FF9800", alpha=0.6))
        for i, vals in enumerate(cls_data):
            axes[1].scatter([i + 1] * len(vals), vals, color="#E65100", s=30, zorder=5)
        axes[1].set_title("Classification Metrics", fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("K-Fold Cross-Validation Metric Distribution", fontweight="bold")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_training_curves(fold_histories: list[dict], output_path: str):
    """Training/validation curves (loss, Dice, AUC) overlaid across folds."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(fold_histories), 3)))

    curve_specs = [
        ("loss", "Total Loss"), ("seg_loss", "Segmentation Loss"), ("cls_loss", "Classification Loss"),
        ("seg_dice", "Dice Score"), ("seg_iou", "IoU Score"), ("cls_auc", "AUC-ROC"),
    ]

    for ax, (key, title) in zip(axes.flat, curve_specs):
        for fold_i, hist in enumerate(fold_histories):
            train_vals = [ep.get(key, np.nan) for ep in hist["train"]]
            val_vals = [ep.get(key, np.nan) for ep in hist["val"]]
            ax.plot(range(1, len(train_vals) + 1), train_vals, "-", color=colors[fold_i], alpha=0.4, linewidth=1)
            ax.plot(range(1, len(val_vals) + 1), val_vals, "-", color=colors[fold_i], alpha=0.9, linewidth=2, label=f"Fold {fold_i} (val)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Training & Validation Curves Across Folds", fontweight="bold")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_fold_auc_bar(fold_results: list[dict], output_path: str):
    fold_auc = [fr.get("cls_auc", 0) for fr in fold_results]
    mean_auc, std_auc = np.mean(fold_auc), np.std(fold_auc)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(fold_results), 3)))
    ax.barh(range(len(fold_results)), fold_auc, color=colors[: len(fold_results)], edgecolor="black")
    ax.axvline(mean_auc, color="red", linestyle="--", label=f"Mean = {mean_auc:.4f} +/- {std_auc:.4f}")
    ax.set_yticks(range(len(fold_results)))
    ax.set_yticklabels([f"Fold {i}" for i in range(len(fold_results))])
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Per-Fold AUC-ROC (Internal CV)", fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_external_validation_bars(exp1_summary: dict, ext_a: dict, ext_b: dict, output_path: str):
    metric_keys = ["seg_dice", "seg_iou", "cls_auc", "cls_accuracy", "cls_sensitivity", "cls_specificity", "cls_f1"]
    scenario_values = [
        [exp1_summary.get(m, {}).get("mean", 0) for m in metric_keys],
        [ext_a["results"].get(m, 0) for m in metric_keys],
        [ext_b["results"].get(m, 0) for m in metric_keys],
    ]
    scenario_labels = ["Internal CV", ext_a["direction"], ext_b["direction"]]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    seg_idx = [i for i, m in enumerate(metric_keys) if m.startswith("seg_")]
    cls_idx = [i for i, m in enumerate(metric_keys) if m.startswith("cls_")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, idx_list, title in [(axes[0], seg_idx, "Segmentation"), (axes[1], cls_idx, "Classification")]:
        x = np.arange(len(idx_list))
        width = 0.25
        for s_i, (label, color, vals) in enumerate(zip(scenario_labels, colors, scenario_values)):
            ax.bar(x + s_i * width, [vals[j] for j in idx_list], width, label=label, color=color, alpha=0.85, edgecolor="black")
        ax.set_xticks(x + width)
        ax.set_xticklabels([metric_keys[j] for j in idx_list])
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Internal CV vs Bidirectional External Validation", fontweight="bold")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_backbone_pareto(exp4_results: dict, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, metric, label in [(axes[0], "seg_dice", "Dice"), (axes[1], "cls_auc", "AUC")]:
        for i, (bb_name, folds) in enumerate(exp4_results.items()):
            params_m = folds[0].get("params", 0) / 1e6
            vals = [f.get(metric, np.nan) for f in folds]
            ax.errorbar(params_m, np.nanmean(vals), yerr=np.nanstd(vals), fmt="o", markersize=12,
                        capsize=6, color=colors[i % len(colors)], label=bb_name, markeredgecolor="black")
            ax.annotate(bb_name, (params_m, np.nanmean(vals)), textcoords="offset points", xytext=(8, 8), fontsize=9)
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Model Size", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Backbone Comparison -- Performance vs Model Size", fontweight="bold")
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_loss_ablation_bars(exp5_results: dict, output_path: str):
    chart_metrics = ["seg_dice", "seg_iou", "cls_auc", "cls_f1"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(chart_metrics))
    n_strats = len(exp5_results)
    width = 0.8 / max(n_strats, 1)

    for i, (strat_name, folds) in enumerate(exp5_results.items()):
        means = [np.nanmean([f.get(m, np.nan) for f in folds]) for m in chart_metrics]
        stds = [np.nanstd([f.get(m, np.nan) for f in folds]) for m in chart_metrics]
        ax.bar(x + i * width, means, width, yerr=stds, label=strat_name, color=colors[i % len(colors)],
               alpha=0.85, edgecolor="black", capsize=4)

    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels(chart_metrics)
    ax.set_title("Loss Weighting Strategy Comparison", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_weight_trajectory(weight_df: pd.DataFrame, output_path: str, strategy_name: str = ""):
    """Plot per-epoch effective task weights (and, for uncertainty weighting,
    learned log-variance parameters) across folds. Directly addresses R1.5's
    request for weight trajectory evidence."""
    if weight_df.empty:
        raise ValueError("weight_df is empty -- no weight history was recorded for this run.")

    has_logvar = "log_var_seg" in weight_df.columns
    fig, axes = plt.subplots(1, 2 if has_logvar else 1, figsize=(14 if has_logvar else 7, 5), squeeze=False)
    axes = axes[0]

    for fold_id, sub in weight_df.groupby("fold"):
        axes[0].plot(sub["epoch"], sub["w_seg"], label=f"Fold {fold_id} w_seg", alpha=0.8)
        axes[0].plot(sub["epoch"], sub["w_cls"], "--", label=f"Fold {fold_id} w_cls", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Effective task weight")
    axes[0].set_title(f"Effective Weights{f' -- {strategy_name}' if strategy_name else ''}", fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    if has_logvar:
        for fold_id, sub in weight_df.groupby("fold"):
            axes[1].plot(sub["epoch"], sub["log_var_seg"], label=f"Fold {fold_id} log-var seg", alpha=0.8)
            axes[1].plot(sub["epoch"], sub["log_var_cls"], "--", label=f"Fold {fold_id} log-var cls", alpha=0.8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("log(sigma^2)")
        axes[1].set_title("Learned Uncertainty Parameters", fontweight="bold")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


def plot_reliability_grid(probs_targets_by_model: dict[str, tuple[np.ndarray, np.ndarray]], output_path: str, n_bins: int = 10):
    """Grid of reliability diagrams, one per model/scenario (R1.7)."""
    n = len(probs_targets_by_model)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for i, (name, (probs, targets)) in enumerate(probs_targets_by_model.items()):
        ax = axes[i // n_cols, i % n_cols]
        ece = compute_ece(probs, targets, n_bins)
        plot_reliability_diagram(probs, targets, n_bins=n_bins, ax=ax, title=f"{name}\n(ECE={ece:.4f})")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle("Reliability Diagrams (Calibration Analysis)", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, output_path)
    return fig
