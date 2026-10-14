"""Regression tests for R1.3: the misleading 'Combined' (avg Dice+AUC) score
must never appear in any generated table."""
from __future__ import annotations

from mtl_lymphnet.reporting.tables import build_loss_ablation_table, build_mtl_ablation_table


def _fake_fold_results(n_folds: int = 3) -> list[dict]:
    return [
        {
            "seg_dice": 0.90 + i * 0.01,
            "seg_iou": 0.82,
            "cls_auc": 0.95,
            "cls_f1": 0.90,
            "params": 5_000_000,
            "infer_cpu_ms": 50.0,
        }
        for i in range(n_folds)
    ]


def test_mtl_ablation_table_has_no_combined_column():
    exp3 = {
        "mtl": _fake_fold_results(),
        "seg_only": _fake_fold_results(),
        "cls_only": _fake_fold_results(),
    }
    df, _ = build_mtl_ablation_table(exp3)
    assert "Combined" not in df.columns


def test_loss_ablation_table_has_no_combined_column():
    exp5 = {
        "Fixed (0.5/0.5)": _fake_fold_results(),
        "Uncertainty": _fake_fold_results(),
    }
    df, _ = build_loss_ablation_table(exp5)
    assert "Combined" not in df.columns
