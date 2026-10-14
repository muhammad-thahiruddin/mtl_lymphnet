"""Experiment 4: Ablation -- encoder backbone comparison (R1.4).

The proposed backbone (determined by ``backbones.PROPOSED_KEY``) is reused from
Experiment 1 (identical loss strategy, folds, seed). All other registered
backbones are trained fresh with the SAME leakage-safe nested-split protocol.

To switch the proposed backbone, change ``PROPOSED_KEY`` in
``src/mtl_lymphnet/models/backbones.py`` (or run
``python configs/switch_backbone.py <key>``).  No other code changes needed.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import Config
from ..models.backbones import BACKBONES, PROPOSED_KEY, proposed_backbone, variant_backbones
from ..models.mtl_unet import MTLUNet
from .base import benchmark_variant, build_criterion_factory, run_variant_kfold


def run_ablation_backbone(
    config: Config,
    df: pd.DataFrame,
    exp1_results: dict,
    loss_strategy: str = "uncertainty",
    w_seg: float = 0.5,
    group_col: Optional[str] = None,
) -> dict:
    all_results: dict[str, list[dict]] = {}

    # Reuse proposed backbone results from Experiment 1
    proposed_name = PROPOSED_KEY
    proposed_folds = [dict(fr) for fr in exp1_results["fold_results"]]
    print(f"Reusing {proposed_name} (proposed) from Experiment 1 ({len(proposed_folds)} runs)")

    bench = benchmark_variant(
        lambda: MTLUNet(backbone=proposed_backbone(), pretrained=None), config,
    )
    for fr in proposed_folds:
        fr.update(bench)
    all_results[proposed_name] = proposed_folds

    # Train remaining (variant) backbones fresh
    for bb_name, bb_key in variant_backbones().items():
        def model_factory(_bb_key=bb_key) -> MTLUNet:
            return MTLUNet(backbone=_bb_key, pretrained=config.pretrained)

        criterion_factory = build_criterion_factory(loss_strategy, w_seg, config.label_smoothing)

        fold_results, _, _ = run_variant_kfold(
            config, df, model_factory, criterion_factory, f"exp4_{bb_name}", group_col=group_col,
        )
        bench = benchmark_variant(model_factory, config)
        for fr in fold_results:
            fr.update(bench)
        all_results[bb_name] = fold_results

    return all_results
