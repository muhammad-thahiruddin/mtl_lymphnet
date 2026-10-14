"""Shared utilities for experiment runners: checkpoint saving, JSON
serialization, fold-result aggregation, and the generic nested K-fold loop
used by every experiment (internal CV and all ablations).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import Config
from ..data.dataset import compute_pos_weight, get_dataloaders, get_eval_loader
from ..data.splits import apply_split, make_nested_kfold_splits
from ..engine.trainer import fit
from ..losses.weighting import BaseMTLLoss


def serialize_for_json(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=serialize_for_json)


def save_pickle(obj, path: str) -> None:
    """Persist full experiment results (including per-image arrays and weight
    trajectories) so downstream experiments/CLI invocations can reuse a
    "proposed" variant without retraining. Only ever load pickle files that
    this framework produced -- never unpickle untrusted data."""
    import pickle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, path)


def sanitize_name(text: str) -> str:
    """Sanitize a label for safe, cross-platform use as a path component.

    Bug B3: labels like "Fixed (0.5/0.5)" contain a literal "/" (from the
    w_seg display convention), which -- when used directly as part of an
    experiment_name/checkpoint path -- silently splits into a nested
    directory (observed as `exp5_Fixed (0.5` / `0.5)_fold0`). This also
    strips other characters invalid on Windows (`< > : " \\ | ? *`) and
    converts "->" to "_to_" (direction names like "C1->C2" contain a
    Windows-invalid ">").

    Shared by every experiment runner that builds a path/experiment_name
    from a variant label (`external_validation.py`, `ablation_loss.py`) so
    the fix lives in one place.
    """
    import re

    text = text.replace("->", "_to_")
    text = re.sub(r'[<>:"/\\|?*,()]', "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def summarize_folds(fold_results: list[dict], skip_prefix: str = "_") -> dict:
    """Aggregate a list of per-fold metric dicts into {metric: {mean, std, values}}."""
    if not fold_results:
        return {}
    keys = [k for k in fold_results[0] if not k.startswith(skip_prefix)]
    summary = {}
    for k in keys:
        vals = [r[k] for r in fold_results if k in r and isinstance(r[k], (int, float, np.floating))]
        if not vals:
            continue
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": [float(v) for v in vals]}
    return summary


def select_representative_fold(fold_results: list[dict], metric: str = "seg_dice") -> int:
    """Pick a fold for qualitative visualization (segmentation overlays,
    Grad-CAM, failure-case analysis).

    R1.3: the original notebook picked the "best fold" using an unjustified
    average of Dice and AUC ("combined score"). That combined score has been
    removed from all quantitative reporting. For the unrelated purpose of
    choosing ONE fold to visualize, we use the fold whose `metric` value is
    closest to the median across folds -- i.e. a "typical", not "best", fold.
    This is presentation-only and never affects any reported metric.
    """
    vals = [fr.get(metric, np.nan) for fr in fold_results]
    median_val = np.nanmedian(vals)
    return int(np.nanargmin([abs(v - median_val) for v in vals]))


def build_criterion_factory(
    loss_strategy: str, w_seg: float, label_smoothing: float
) -> Callable[[float], BaseMTLLoss]:
    from ..losses.weighting import build_criterion

    def factory(pos_weight: float) -> BaseMTLLoss:
        return build_criterion(strategy=loss_strategy, pos_weight=pos_weight, w_seg=w_seg, label_smoothing=label_smoothing)

    return factory


def benchmark_variant(model_factory: Callable[[], nn.Module], config: Config, n_reps: int = 50) -> dict:
    """Benchmark parameter count, model size, and GPU/CPU inference latency
    for a freshly-constructed (untrained) instance of a model variant.

    Untrained weights do not affect parameter count or latency, so this can
    run independently of any training loop -- used by the MTL-vs-single-task
    and backbone ablations to populate their "Params" / "GPU (ms)" / "CPU
    (ms)" table columns with a fully specified benchmark (R1.11).
    """
    from ..benchmarking.inference_benchmark import benchmark_inference
    from ..models.mtl_unet import count_parameters

    model = model_factory()
    info = count_parameters(model)
    result = {"params": info["total"], "size_mb": info["size_mb"]}

    if torch.cuda.is_available():
        gpu_bench = benchmark_inference(
            model, (3, config.img_size, config.img_size), device="cuda",
            batch_size=1, precision="fp32", n_warmup=10, n_reps=n_reps,
        )
        result["infer_gpu_ms"] = gpu_bench.mean_ms
        model = model.cpu()

    cpu_bench = benchmark_inference(
        model, (3, config.img_size, config.img_size), device="cpu",
        batch_size=1, precision="fp32", n_warmup=5, n_reps=max(20, n_reps // 2),
    )
    result["infer_cpu_ms"] = cpu_bench.mean_ms
    return result


def run_variant_kfold(
    config: Config,
    df: pd.DataFrame,
    model_factory: Callable[[], nn.Module],
    criterion_factory: Callable[[float], BaseMTLLoss],
    experiment_name: str,
    group_col: Optional[str] = None,
    save_checkpoints: bool = True,
) -> tuple[list[dict], list[dict], list[list[dict]]]:
    """Generic leakage-safe nested K-fold loop, reused by internal CV and every
    ablation study.

    For each outer fold: train on the inner train split, select the best
    checkpoint using the inner val split, then evaluate exactly once on the
    held-out test fold. The returned per-fold metrics are always the TEST
    fold's metrics (never the validation split's), which is the concrete fix
    for R1.1.

    Returns
    -------
    (fold_results, fold_val_metrics, fold_histories)
        fold_results: final TEST metrics per fold (what should be reported).
        fold_val_metrics: final inner-VAL metrics per fold (diagnostic only).
        fold_histories: per-epoch train/val history per fold.
    """
    if config.subset_fraction < 1.0:
        df = df.sample(frac=config.subset_fraction, random_state=config.seed).reset_index(drop=True)
        print(f"Subset mode: using {len(df)} samples ({config.subset_fraction * 100:.0f}%)")

    splits = make_nested_kfold_splits(df, config.n_folds, config.val_fraction, "label", group_col, config.seed)

    fold_results, fold_val_metrics, fold_histories = [], [], []

    for fold, split in enumerate(splits):
        print(f"\n{'-' * 60}\n  Fold {fold + 1}/{config.n_folds} -- {experiment_name}\n{'-' * 60}")
        df_train, df_val, df_test = apply_split(df, split)

        train_loader, val_loader = get_dataloaders(df_train, df_val, config)
        test_loader = get_eval_loader(df_test, config)

        model = model_factory().to(config.device)
        if config.multi_gpu:
            model = nn.DataParallel(model)

        pos_weight = compute_pos_weight(df_train)
        criterion = criterion_factory(pos_weight).to(config.device)

        result = fit(
            model, train_loader, val_loader, config, criterion,
            test_loader=test_loader, fold=fold, experiment_name=experiment_name,
        )

        test_metrics = dict(result.test_metrics)
        test_metrics["_weight_history"] = result.weight_history
        test_metrics["_n_train"] = split.n_train
        test_metrics["_n_val"] = split.n_val
        test_metrics["_n_test"] = split.n_test
        fold_results.append(test_metrics)
        fold_val_metrics.append(result.best_val_metrics)
        fold_histories.append(result.history)

        if save_checkpoints:
            ckpt_path = os.path.join(config.output_dir, "checkpoints", f"{experiment_name}_fold{fold}.pt")
            save_checkpoint(model, ckpt_path)
            print(f"  Checkpoint saved -> {ckpt_path}")

        del model, criterion, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fold_results, fold_val_metrics, fold_histories
