"""Experiment 2: Bidirectional external (cross-center) validation (leakage-safe).

Fixes R1.1: previously the external test center was passed directly as
`val_loader`, so it was used for early stopping / checkpoint selection on the
very data meant to test cross-center generalization. Now the training
center is split into an inner train/val (early stopping only); the external
test center is evaluated exactly once, after training, and never seen before.

Fixes Bug B1: previously every call to `run_external_validation()` wrote to
a FIXED checkpoint path (`checkpoints/ext_{direction_name}.pt`) and a FIXED
TensorBoard run name (`ext_{direction_name}`), with no distinction between
backbone, loss-weighting strategy, or seed. Re-running with a different
configuration (needed for R1.4's per-backbone external evidence and R1.5's
per-strategy cross-center results) silently overwrote the previous run's
checkpoint and logs. `run_external_validation()` now:
  - accepts `variant_key` (auto-derived from backbone/loss_strategy/w_seg
    if not given) and folds it into every checkpoint path / run name, so
    different configurations never collide;
  - accepts `seeds` (a list) and trains one model per seed internally
    (mirroring the multi-seed loop already used by `run_internal_cv`,
    R1.6), returning both the primary seed's results (for backward
    compatibility with single-seed callers) and the full per-seed detail;
  - sanitizes all path components (fixes Bug B3's root cause for this file,
    and a related issue where "->" in direction names such as "C1->C2"
    silently became "-_" when results were copied from a Linux training
    environment to Windows, since ">" is not a valid Windows filename
    character).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import Config
from ..data.dataset import compute_pos_weight, get_dataloaders, get_eval_loader
from ..data.splits import apply_external_split, make_external_validation_split
from ..engine.trainer import fit
from ..models.backbones import BACKBONES
from ..models.mtl_unet import MTLUNet
from .base import build_criterion_factory, sanitize_name, save_checkpoint


def _short_backbone_name(timm_key: str) -> str:
    """Reverse-lookup a human-readable backbone name for a timm model
    string, falling back to the raw string if it's not in the registry."""
    for name, key in BACKBONES.items():
        if key == timm_key:
            return name
    return timm_key


def run_external_validation(
    config: Config,
    df_train_center: pd.DataFrame,
    df_test_center: pd.DataFrame,
    direction_name: str = "C1->C2",
    backbone: Optional[str] = None,
    loss_strategy: str = "uncertainty",
    w_seg: float = 0.5,
    group_col: Optional[str] = None,
    seeds: Optional[list[int]] = None,
    variant_key: Optional[str] = None,
) -> dict:
    """Train on one center's inner train/val split; evaluate once on the
    other (fully external) center. Repeats for every seed in `seeds`.

    Returns a dict whose top-level `results`/`val_results`/`history`/
    `weight_history`/`n_train`/`n_val`/`n_test` keys always hold the
    PRIMARY seed's data (so existing single-seed callers keep working
    unchanged), plus `results_by_seed` / `val_results_by_seed` /
    `checkpoint_paths_by_seed` / `summary` (mean+std across seeds, when
    more than one) for full multi-seed detail.
    """
    backbone = backbone or config.backbone
    seeds = seeds or [config.seed]

    if variant_key is None:
        variant_key = f"{_short_backbone_name(backbone)}_{loss_strategy}_w{w_seg}"
    variant_key = sanitize_name(variant_key)
    direction_slug = sanitize_name(direction_name)

    if config.subset_fraction < 1.0:
        df_train_center = df_train_center.sample(frac=config.subset_fraction, random_state=config.seed).reset_index(drop=True)
        df_test_center = df_test_center.sample(frac=config.subset_fraction, random_state=config.seed).reset_index(drop=True)

    results_by_seed: dict[int, dict] = {}
    val_by_seed: dict[int, dict] = {}
    history_by_seed: dict[int, dict] = {}
    weight_history_by_seed: dict[int, list] = {}
    ckpt_by_seed: dict[int, str] = {}
    n_train = n_val = n_test = 0

    for seed in seeds:
        seed_cfg = config.replace(seed=seed)
        run_tag = f"{direction_slug}__{variant_key}_seed{seed}"

        print(f"\n{'-' * 60}\n  External Validation: {direction_name}  [{variant_key}, seed={seed}]\n"
              f"  Train center: {len(df_train_center)}, External test center: {len(df_test_center)}\n{'-' * 60}")

        split = make_external_validation_split(df_train_center, seed_cfg.val_fraction, "label", group_col, seed_cfg.seed)
        df_train, df_val, df_test = apply_external_split(df_train_center, split, df_test_center)

        train_loader, val_loader = get_dataloaders(df_train, df_val, seed_cfg)
        test_loader = get_eval_loader(df_test, seed_cfg)

        model = MTLUNet(backbone=backbone, pretrained=seed_cfg.pretrained).to(seed_cfg.device)
        if seed_cfg.multi_gpu:
            model = nn.DataParallel(model)

        pos_weight = compute_pos_weight(df_train)
        criterion = build_criterion_factory(loss_strategy, w_seg, seed_cfg.label_smoothing)(pos_weight).to(seed_cfg.device)

        result = fit(
            model, train_loader, val_loader, seed_cfg, criterion,
            test_loader=test_loader, fold=0, experiment_name=f"ext_{run_tag}",
            test_label=(
                f"External Validation Results: {direction_name} "
                f"[{variant_key}, seed={seed}] (external center, never used for model selection)"
            ),
        )

        ckpt_path = f"{seed_cfg.output_dir}/checkpoints/ext_{run_tag}.pt"
        save_checkpoint(model, ckpt_path)

        results_by_seed[seed] = result.test_metrics
        val_by_seed[seed] = result.best_val_metrics
        history_by_seed[seed] = result.history
        weight_history_by_seed[seed] = result.weight_history
        ckpt_by_seed[seed] = ckpt_path
        n_train, n_val, n_test = split.n_train, split.n_val, len(df_test)

        del model, criterion, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    primary_seed = seeds[0]
    scalar_keys = [k for k in results_by_seed[primary_seed] if not k.startswith("_")]
    summary = {
        k: {
            "mean": float(np.mean([results_by_seed[s][k] for s in seeds])),
            "std": float(np.std([results_by_seed[s][k] for s in seeds])),
        }
        for k in scalar_keys
    } if len(seeds) > 1 else {}

    return {
        "direction": direction_name,
        "backbone": backbone,
        "loss_strategy": loss_strategy,
        "w_seg": w_seg,
        "variant_key": variant_key,
        "seeds": seeds,
        # Backward-compatible: primary seed's raw metrics under the same
        # keys single-seed callers have always used.
        "results": results_by_seed[primary_seed],
        "val_results": val_by_seed[primary_seed],
        "history": history_by_seed[primary_seed],
        "weight_history": weight_history_by_seed[primary_seed],
        # Full multi-seed detail (new).
        "results_by_seed": results_by_seed,
        "val_results_by_seed": val_by_seed,
        "checkpoint_paths_by_seed": ckpt_by_seed,
        "summary": summary,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
