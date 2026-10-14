"""Training engine implementing the leakage-safe fit/evaluate protocol.

Fixes reviewer comment R1.1. Contract:

- `val_loader` (the INNER validation split, ~20% of the training partition)
  is used for early stopping and best-checkpoint selection.
- `test_loader`, if provided, is the held-out fold or external test center. It
  is evaluated exactly ONCE, via `evaluate()`, after the best checkpoint
  (selected using `val_loader` only) has been restored. It never influences
  any training-time decision.
- `fit()` raises `ValueError` if `test_loader is val_loader`, guarding against
  the exact mistake present in the original notebook (test fold passed in as
  `val_loader`, then used for early stopping).

Bug B2 fix: the generic "Held-out Test Results" print below fires for every
caller that passes a `test_loader` (both k-fold internal CV and external
cross-center validation). Previously this produced a confusing duplicate
print when `external_validation.py` re-printed the same metrics under its
own "External Validation Results" header. `test_label` lets a caller supply
a context-specific header so the two no longer look like two different
evaluations with coincidentally identical numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ..config import Config
from ..losses.weighting import BaseMTLLoss, DynamicWeightAverage
from .early_stopping import EarlyStopping
from .metrics import MetricTracker, compute_hd95
from .scheduler import CosineAnnealingWarmRestartsDecay


@dataclass
class FitResult:
    history: dict = field(default_factory=lambda: {"train": [], "val": []})
    best_val_metrics: dict = field(default_factory=dict)
    test_metrics: Optional[dict] = None
    best_epoch: int = 0
    weight_history: list = field(default_factory=list)


def train_one_epoch(
    model: nn.Module, loader, optimizer, criterion, scaler, config: Config, epoch: int,
    quiet: bool = False,
) -> dict:
    model.train()
    tracker = MetricTracker()
    total_loss = total_seg_loss = total_cls_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train Ep{epoch}", leave=False, disable=quiet)
    for batch in pbar:
        images = batch["image"].to(config.device, non_blocking=True)
        masks = batch["mask"].to(config.device, non_blocking=True)
        labels = batch["label"].to(config.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if config.use_amp:
            with torch.amp.autocast(config.device):
                seg_logits, cls_logits = model(images)
                loss, l_seg, l_cls = criterion(seg_logits, masks, cls_logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_logits, cls_logits = model(images)
            loss, l_seg, l_cls = criterion(seg_logits, masks, cls_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        tracker.update_seg(seg_logits, masks)
        tracker.update_cls(cls_logits, labels)

        total_loss += loss.item()
        total_seg_loss += l_seg.item()
        total_cls_loss += l_cls.item()
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "seg": f"{l_seg.item():.4f}", "cls": f"{l_cls.item():.4f}"})

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
        **{f"seg_{k}": v for k, v in tracker.compute_seg_metrics().items()},
        **{f"cls_{k}": v for k, v in tracker.compute_cls_metrics().items()},
    }


@torch.no_grad()
def validate(
    model: nn.Module, loader, criterion, config: Config, compute_hd: bool = False, quiet: bool = False,
) -> dict:
    """Evaluate `model` on `loader` without updating any parameters.

    Used both for the inner validation split (every epoch) and for the final,
    single evaluation on the held-out test fold / external test center.
    """
    model.eval()
    tracker = MetricTracker()
    total_loss = total_seg_loss = total_cls_loss = 0.0
    n_batches = 0
    hd95_list: list[float] = []

    pbar = tqdm(loader, desc="Eval", leave=False, disable=quiet)
    for batch in pbar:
        images = batch["image"].to(config.device, non_blocking=True)
        masks = batch["mask"].to(config.device, non_blocking=True)
        labels = batch["label"].to(config.device, non_blocking=True)

        if config.use_amp:
            with torch.amp.autocast(config.device):
                seg_logits, cls_logits = model(images)
                loss, l_seg, l_cls = criterion(seg_logits, masks, cls_logits, labels)
        else:
            seg_logits, cls_logits = model(images)
            loss, l_seg, l_cls = criterion(seg_logits, masks, cls_logits, labels)

        tracker.update_seg(seg_logits, masks)
        tracker.update_cls(cls_logits, labels)

        total_loss += loss.item()
        total_seg_loss += l_seg.item()
        total_cls_loss += l_cls.item()
        n_batches += 1

        if compute_hd:
            preds_np = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)
            for i in range(preds_np.shape[0]):
                hd = compute_hd95(preds_np[i, 0], masks_np[i, 0])
                if hd != np.inf:
                    hd95_list.append(hd)

    n = max(n_batches, 1)
    result = {
        "loss": total_loss / n,
        "seg_loss": total_seg_loss / n,
        "cls_loss": total_cls_loss / n,
        **{f"seg_{k}": v for k, v in tracker.compute_seg_metrics().items()},
        **{f"cls_{k}": v for k, v in tracker.compute_cls_metrics().items()},
    }
    if compute_hd and hd95_list:
        result["seg_hd95"] = float(np.mean(hd95_list))

    # Per-image data for statistical tests (prefixed with _ to skip aggregation)
    result["_per_image_dice"] = list(tracker.seg_dice_per_image)
    result["_cls_probs"] = list(tracker.cls_probs)
    result["_cls_targets"] = list(tracker.cls_targets)
    return result


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Config,
    criterion: BaseMTLLoss,
    *,
    test_loader=None,
    fold: int = 0,
    experiment_name: str = "default",
    log_dir: Optional[str] = None,
    test_label: Optional[str] = None,
) -> FitResult:
    """Full training loop with scheduling, early stopping, and (optionally) a
    single final evaluation on a held-out test set.

    `val_loader` drives early stopping / checkpoint selection. `test_loader`,
    if given, is evaluated exactly once at the very end and never seen by the
    optimizer, the scheduler, or `EarlyStopping`.

    `test_label`, if given, overrides the default "Held-out Test Results
    (Fold N, never used for model selection)" print header -- used by
    external-validation callers to make clear the results shown are a
    cross-center evaluation, not a k-fold held-out test (bug B2 fix).
    """
    if test_loader is not None and test_loader is val_loader:
        raise ValueError(
            "test_loader must not be the same object as val_loader -- this "
            "would leak the held-out test set into early stopping / "
            "checkpoint selection (see implementation_plan_1.md R1.1)."
        )

    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestartsDecay(
        optimizer, T_0=15, T_mult=2, warmup_epochs=5, gamma=0.8, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(config.device, enabled=config.use_amp and config.device == "cuda")
    early_stop = EarlyStopping(patience=config.patience, mode="min")

    log_dir = log_dir or os.path.join(config.output_dir, "runs", f"{experiment_name}_fold{fold}")
    writer = SummaryWriter(log_dir=log_dir)

    history = {"train": [], "val": []}
    log_interval = max(1, config.num_epochs // 5) if config.mode == "full" else 1
    quiet = config.mode == "full"

    print(f"\n{'=' * 60}")
    print(f"  Training: {experiment_name} -- Fold {fold + 1}")
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}"
          + (f", Test (held out): {len(test_loader.dataset)}" if test_loader is not None else ""))
    print(f"{'=' * 60}")

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, config, epoch, quiet=quiet)

        compute_hd = (epoch == config.num_epochs) or (epoch % 10 == 0)
        val_metrics = validate(model, val_loader, criterion, config, compute_hd=compute_hd, quiet=quiet)

        scheduler.step()

        history["train"].append({k: v for k, v in train_metrics.items() if not k.startswith("_")})
        history["val"].append({k: v for k, v in val_metrics.items() if not k.startswith("_")})

        for k, v in train_metrics.items():
            if not k.startswith("_"):
                writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            if not k.startswith("_"):
                writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if hasattr(criterion, "get_weights"):
            for wk, wv in criterion.get_weights().items():
                writer.add_scalar(f"weights/{wk}", wv, epoch)
        if hasattr(criterion, "record_epoch"):
            criterion.record_epoch(epoch)
        if isinstance(criterion, DynamicWeightAverage):
            criterion.record_epoch_loss(train_metrics["seg_loss"], train_metrics["cls_loss"])

        if epoch % log_interval == 0 or epoch == 1 or epoch == config.num_epochs:
            print(
                f"Ep {epoch:03d}/{config.num_epochs} | TrL {train_metrics['loss']:.4f} | "
                f"VaL {val_metrics['loss']:.4f} | Dice {val_metrics['seg_dice']:.4f} | "
                f"IoU {val_metrics['seg_iou']:.4f} | AUC {val_metrics['cls_auc']:.4f} | "
                f"F1 {val_metrics['cls_f1']:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Early stopping decision uses ONLY the inner validation split.
        early_stop(val_metrics["loss"], model, epoch=epoch)
        if early_stop.should_stop:
            print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
            break

    early_stop.load_best(model)
    writer.close()

    final_val_metrics = validate(model, val_loader, criterion, config, compute_hd=True)
    print(f"\n--- Best Model Validation Results (Fold {fold + 1}, inner val split) ---")
    for k, v in final_val_metrics.items():
        if not k.startswith("_"):
            print(f"  {k:20s}: {v:.4f}")

    test_metrics = None
    if test_loader is not None:
        test_metrics = validate(model, test_loader, criterion, config, compute_hd=True)
        label = test_label or f"Held-out Test Results (Fold {fold + 1}, never used for model selection)"
        print(f"\n--- {label} ---")
        for k, v in test_metrics.items():
            if not k.startswith("_"):
                print(f"  {k:20s}: {v:.4f}")

    return FitResult(
        history=history,
        best_val_metrics=final_val_metrics,
        test_metrics=test_metrics,
        best_epoch=early_stop.best_epoch,
        weight_history=criterion.export_history() if hasattr(criterion, "export_history") else [],
    )
