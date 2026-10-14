"""Experiment 5: Ablation -- MTL loss weighting strategy comparison.

"Uncertainty" is reused from Experiment 1 (identical backbone, folds, seed).
Fixed (0.5/0.5), Fixed (0.7/0.3), and DWA are trained fresh with the SAME
leakage-safe nested-split protocol.

R1.5: every strategy's per-epoch weight trajectory (including the learned
log-variance parameters for uncertainty weighting) is available per fold via
`fold_result["_weight_history"]`, exported by `BaseMTLLoss.export_history()`.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import Config
from ..models.mtl_unet import MTLUNet
from .base import build_criterion_factory, run_variant_kfold, sanitize_name

LOSS_ABLATION = {
    "Fixed (0.5/0.5)": {"strategy": "fixed", "w_seg": 0.5},
    "Fixed (0.7/0.3)": {"strategy": "fixed", "w_seg": 0.7},
    "Uncertainty": {"strategy": "uncertainty", "w_seg": 0.5},
    "DWA": {"strategy": "dwa", "w_seg": 0.5},
}

# Strategies that need fresh training (Uncertainty is reused from Experiment 1)
LOSS_TRAIN_ONLY = {
    "Fixed (0.5/0.5)": {"strategy": "fixed", "w_seg": 0.5},
    "Fixed (0.7/0.3)": {"strategy": "fixed", "w_seg": 0.7},
    "DWA": {"strategy": "dwa", "w_seg": 0.5},
}


def run_ablation_loss(
    config: Config,
    df: pd.DataFrame,
    exp1_results: dict,
    group_col: Optional[str] = None,
) -> dict:
    all_results: dict[str, list[dict]] = {
        "Uncertainty": [dict(fr) for fr in exp1_results["fold_results"]],
    }
    print(f"Reusing 'Uncertainty' results from Experiment 1 ({len(all_results['Uncertainty'])} runs)")

    for strat_name, strat_cfg in LOSS_TRAIN_ONLY.items():
        def model_factory() -> MTLUNet:
            return MTLUNet(backbone=config.backbone, pretrained=config.pretrained)

        criterion_factory = build_criterion_factory(
            strat_cfg["strategy"], strat_cfg["w_seg"], config.label_smoothing
        )

        fold_results, _, _ = run_variant_kfold(
            config, df, model_factory, criterion_factory, f"exp5_{sanitize_name(strat_name)}", group_col=group_col,
        )
        all_results[strat_name] = fold_results

    return all_results
