"""Experiment 3: Ablation -- MTL (proposed) vs Seg-Only vs Cls-Only.

The MTL variant is reused from Experiment 1's results (identical backbone,
loss strategy, folds, and seed) rather than retrained -- matching the
original design intent and avoiding redundant compute. Seg-Only and Cls-Only
are trained fresh using the SAME leakage-safe nested-split protocol
(`run_variant_kfold`), on the exact same folds (deterministic given the same
`df` + `config.seed`), for a fair comparison.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch.nn as nn

from ..config import Config
from ..losses.weighting import build_criterion
from ..models.single_task import ClsOnlyNet, SegOnlyUNet
from .base import benchmark_variant, run_variant_kfold


def run_ablation_mtl(
    config: Config,
    df: pd.DataFrame,
    exp1_results: dict,
    group_col: Optional[str] = None,
    benchmark: bool = True,
) -> dict:
    mtl_fold_results = [dict(fr) for fr in exp1_results["fold_results"]]
    print(f"Reusing MTL results from Experiment 1 ({len(mtl_fold_results)} runs)")

    if benchmark:
        from ..models.mtl_unet import MTLUNet

        bench = benchmark_variant(lambda: MTLUNet(backbone=config.backbone, pretrained=None), config)
        for fr in mtl_fold_results:
            fr.update(bench)

    def seg_model_factory() -> nn.Module:
        return SegOnlyUNet(backbone=config.backbone, pretrained=config.pretrained)

    def seg_criterion_factory(pos_weight: float):
        return build_criterion(strategy="fixed", pos_weight=pos_weight, w_seg=1.0, label_smoothing=config.label_smoothing)

    seg_fold_results, _, _ = run_variant_kfold(
        config, df, seg_model_factory, seg_criterion_factory, "exp3_seg_only", group_col=group_col,
    )
    if benchmark:
        bench = benchmark_variant(seg_model_factory, config)
        for fr in seg_fold_results:
            fr.update(bench)

    def cls_model_factory() -> nn.Module:
        return ClsOnlyNet(backbone=config.backbone, pretrained=config.pretrained)

    def cls_criterion_factory(pos_weight: float):
        return build_criterion(strategy="fixed", pos_weight=pos_weight, w_seg=0.0, label_smoothing=config.label_smoothing)

    cls_fold_results, _, _ = run_variant_kfold(
        config, df, cls_model_factory, cls_criterion_factory, "exp3_cls_only", group_col=group_col,
    )
    if benchmark:
        bench = benchmark_variant(cls_model_factory, config)
        for fr in cls_fold_results:
            fr.update(bench)

    return {"mtl": mtl_fold_results, "seg_only": seg_fold_results, "cls_only": cls_fold_results}
