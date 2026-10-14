"""Experiment 1: Internal cross-validation (leakage-safe).

Fixes R1.1: previously, the K-fold "validation" split was actually the held-
out test fold, used directly for early stopping and best-checkpoint
selection, then reported as the model's performance. Now each outer fold's
training partition is further split into an inner train/val (early stopping
only); the outer fold itself is evaluated exactly once and is what gets
reported.

R1.6: pass `seeds=[42, 43, 44, ...]` to additionally account for training
variability across random seeds; results are pooled across (fold x seed).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch.nn as nn

from ..config import Config
from ..models.mtl_unet import MTLUNet
from .base import build_criterion_factory, run_variant_kfold, select_representative_fold, summarize_folds


def run_internal_cv(
    config: Config,
    df: pd.DataFrame,
    experiment_name: str = "exp1_mtl_main",
    backbone: Optional[str] = None,
    loss_strategy: str = "uncertainty",
    w_seg: float = 0.5,
    group_col: Optional[str] = None,
    seeds: Optional[list[int]] = None,
) -> dict:
    """Run leakage-safe stratified K-fold CV and return aggregated results."""
    backbone = backbone or config.backbone
    seeds = seeds or [config.seed]

    results_by_seed: dict[int, list[dict]] = {}
    val_by_seed: dict[int, list[dict]] = {}
    hist_by_seed: dict[int, list[dict]] = {}

    for seed in seeds:
        seed_cfg = config.replace(seed=seed)
        exp_name_seed = experiment_name if len(seeds) == 1 else f"{experiment_name}_seed{seed}"

        def model_factory(_backbone=backbone, _cfg=seed_cfg) -> nn.Module:
            return MTLUNet(backbone=_backbone, pretrained=_cfg.pretrained)

        criterion_factory = build_criterion_factory(loss_strategy, w_seg, seed_cfg.label_smoothing)

        fold_results, fold_val, fold_hist = run_variant_kfold(
            seed_cfg, df, model_factory, criterion_factory, exp_name_seed, group_col=group_col,
        )
        results_by_seed[seed] = fold_results
        val_by_seed[seed] = fold_val
        hist_by_seed[seed] = fold_hist

    all_fold_results = [r for seed_results in results_by_seed.values() for r in seed_results]
    summary = summarize_folds(all_fold_results)
    representative_fold_idx = select_representative_fold(all_fold_results, metric="seg_dice")

    print(f"\n{'=' * 60}\n  {experiment_name} -- Aggregated Results "
          f"({config.n_folds} folds x {len(seeds)} seed(s) = {len(all_fold_results)} runs)\n{'=' * 60}")
    for k, v in summary.items():
        print(f"  {k:20s}: {v['mean']:.4f} +/- {v['std']:.4f}")

    return {
        "experiment": experiment_name,
        "fold_results": all_fold_results,
        "fold_val_metrics": [v for vs in val_by_seed.values() for v in vs],
        "fold_histories": [h for hs in hist_by_seed.values() for h in hs],
        "results_by_seed": results_by_seed,
        "summary": summary,
        "representative_fold_idx": representative_fold_idx,
        "seeds": seeds,
    }
