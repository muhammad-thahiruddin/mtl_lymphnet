"""PyTorch Dataset and DataLoader factories for lymph node ultrasound images."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..config import Config
from .transforms import get_train_transforms, get_val_transforms


class LymphNodeDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for lymph node ultrasound images with masks."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img = np.array(Image.open(row["image_path"]).convert("RGB"), dtype=np.uint8)

        mask = np.array(Image.open(row["mask_path"]), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Binarize with a threshold relative to the mask's own max, robust to
        # both 0/1 and 0/255 mask encodings.
        mask = (mask > mask.max() / 2).astype(np.float32) if mask.max() > 0 else mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        label = torch.tensor(row["label"], dtype=torch.float32)

        return {
            "image": img,
            "mask": mask,
            "label": label,
            "meta": {
                "sample_id": row["sample_id"],
                "center": row["center"],
                "label_name": row["label_name"],
            },
        }


def _build_loader(
    df: pd.DataFrame,
    config: Config,
    transform,
    shuffle: bool,
    sampler=None,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    ds = LymphNodeDataset(df, transform=transform)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    config: Config,
    use_weighted_sampler: bool = False,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation DataLoaders.

    `df_val` here is the INNER validation split (used for early stopping /
    checkpoint selection), not the held-out test fold. See
    `mtl_lymphnet.data.splits` for the nested-split protocol.
    """
    sampler = None
    if use_weighted_sampler:
        labels = df_train["label"].to_numpy(dtype=np.int64, copy=False)
        class_counts = np.bincount(labels)
        weights = (1.0 / class_counts[labels]).astype(np.float64).tolist()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

    train_loader = _build_loader(
        df_train, config, get_train_transforms(config.img_size), shuffle=True,
        sampler=sampler, drop_last=True,
    )
    val_loader = _build_loader(
        df_val, config, get_val_transforms(config.img_size), shuffle=False, drop_last=False,
    )
    return train_loader, val_loader


def get_eval_loader(df: pd.DataFrame, config: Config) -> torch.utils.data.DataLoader:
    """Create a no-shuffle, val-transform DataLoader for final (test) evaluation.

    Intentionally separate from `get_dataloaders` so that call sites cannot
    accidentally reuse the same loader object for both validation and test
    (the exact mistake that caused the R1.1 leakage).
    """
    return _build_loader(
        df, config, get_val_transforms(config.img_size), shuffle=False, drop_last=False,
    )


def compute_pos_weight(df: pd.DataFrame, label_col: str = "label") -> float:
    """Positive-class weight for BCEWithLogitsLoss, computed from TRAIN data only."""
    n_pos = int((df[label_col] == 1).sum())
    n_neg = int((df[label_col] == 0).sum())
    return n_neg / max(n_pos, 1)
