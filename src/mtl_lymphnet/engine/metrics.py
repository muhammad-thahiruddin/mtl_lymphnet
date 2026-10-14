"""Segmentation & classification metric tracking, including HD95."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree as cKDTree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


class MetricTracker:
    """Track segmentation and classification metrics across batches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seg_dice_sum = 0.0
        self.seg_iou_sum = 0.0
        self.seg_count = 0
        self.seg_dice_per_image: list[float] = []  # per-image Dice for stats tests
        self.cls_targets: list[float] = []
        self.cls_probs: list[float] = []

    def update_seg(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits).detach()
        preds = (probs > 0.5).float()

        for i in range(preds.size(0)):
            p = preds[i].view(-1)
            t = targets[i].view(-1)
            intersection = (p * t).sum()
            union = p.sum() + t.sum()

            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            iou = (intersection + 1e-7) / (union - intersection + 1e-7)

            self.seg_dice_per_image.append(dice.item())
            self.seg_dice_sum += dice.item()
            self.seg_iou_sum += iou.item()
            self.seg_count += 1

    def update_cls(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
        targets_np = targets.view(-1).detach().cpu().numpy()
        self.cls_probs.extend(probs.tolist())
        self.cls_targets.extend(targets_np.tolist())

    def compute_seg_metrics(self) -> dict:
        n = max(self.seg_count, 1)
        return {"dice": self.seg_dice_sum / n, "iou": self.seg_iou_sum / n}

    def compute_cls_metrics(self, threshold: float = 0.5) -> dict:
        y_true = np.array(self.cls_targets)
        y_prob = np.array(self.cls_probs)
        y_pred = (y_prob >= threshold).astype(int)

        auc = 0.0 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "auc": auc,
            "accuracy": accuracy_score(y_true, y_pred),
            "sensitivity": tp / (tp + fn + 1e-8),
            "specificity": tn / (tn + fp + 1e-8),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }


def compute_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th percentile Hausdorff Distance between two binary masks."""
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float(np.inf)

    pred_dt = np.asarray(distance_transform_edt(pred), dtype=np.float32)
    target_dt = np.asarray(distance_transform_edt(target), dtype=np.float32)

    pred_border = pred.astype(bool) & (pred_dt <= 1.0)
    target_border = target.astype(bool) & (target_dt <= 1.0)

    pred_pts = np.argwhere(pred_border)
    target_pts = np.argwhere(target_border)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float(np.inf)

    tree_t = cKDTree(target_pts)
    tree_p = cKDTree(pred_pts)

    dist_p2t, _ = tree_t.query(pred_pts)
    dist_t2p, _ = tree_p.query(target_pts)

    all_dist = np.concatenate([dist_p2t, dist_t2p])
    return float(np.percentile(all_dist, 95))
