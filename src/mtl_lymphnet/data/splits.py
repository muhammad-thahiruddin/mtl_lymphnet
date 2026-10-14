"""Nested (train / validation / test) split utilities.

This module directly fixes reviewer comment R1.1: the original notebook
passed the held-out fold (or the external test center) into the training loop
as `val_loader`, which was then used for early stopping AND best-checkpoint
selection -- i.e. the "test" set was used for model selection. That is data
leakage and invalidates any metric reported on it.

Fixed protocol
--------------
1. **Outer split** -- `StratifiedKFold` (or `StratifiedGroupKFold` if a
   group/patient column is supplied) produces K `(train_val_idx, test_idx)`
   pairs.
2. **Inner split** -- `train_val_idx` is further split (stratified,
   `val_fraction`, default 80/20) into `(train_idx, val_idx)`.
3. `val_idx` is used exclusively for early stopping / checkpoint selection
   during training (see `mtl_lymphnet.engine.trainer.fit`).
4. `test_idx` is evaluated exactly once, after training is complete, and
   never influences any training-time decision (early stopping, checkpoint
   choice, backbone choice, loss-weight choice, or threshold choice).

For bidirectional external (cross-center) validation, the same inner split is
applied to the full training-center dataframe; the external test center is
never touched until final evaluation (see `make_external_validation_split`).

Patient-level splitting (R1.10)
--------------------------------
All split functions accept an optional `group_col`. If the dataset's metadata
identifies which images belong to the same patient/lymph node, pass that
column name so that `StratifiedGroupKFold` / `GroupShuffleSplit` are used
instead of plain stratified splitting, preventing same-patient images from
appearing in both train and val/test.

The LymphUs inventory has one row per image and no explicit patient-ID
column -- but the dataset's original publication confirms each image
corresponds to a distinct patient/lymph node (1 image = 1 patient = 1 lymph
node; confirmed by the study authors against the source dataset paper).
Under that one-to-one mapping, `group_col=None` (image-level splitting, the
current default) is ALREADY equivalent to patient-level splitting: every row
is already a unique patient, so no patient can be duplicated across splits.
`group_col` remains available and should be used if a future dataset
revision (or a different dataset) has a genuine one-to-many patient/image
relationship.

This resolves the "no patient overlap across splits" part of reviewer
comment R1.10. It does NOT resolve who created the reference-standard masks
or whether inter-observer agreement was measured for them -- that
information has not yet been located in the source dataset publication (see
docs/status_report_and_action_plan.md, R1.10, for the remaining open item).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def assert_disjoint(*index_arrays: Sequence[int]) -> None:
    """Raise ValueError if any two index arrays overlap.

    This is the core leakage guard used throughout the framework -- call it
    whenever train/val/test indices are constructed.
    """
    seen: set[int] = set()
    for arr in index_arrays:
        s = {int(i) for i in arr}
        overlap = seen & s
        if overlap:
            raise ValueError(
                f"Data leakage detected: {len(overlap)} indices appear in more "
                f"than one split (e.g. {sorted(overlap)[:5]})."
            )
        seen |= s


@dataclass(frozen=True)
class DataSplit:
    """Disjoint train/val/test index arrays into the SAME dataframe."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        assert_disjoint(self.train_idx, self.val_idx, self.test_idx)

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


@dataclass(frozen=True)
class ExternalSplit:
    """Disjoint train/val index arrays into the training-center dataframe.

    The external test center is used in full (no slicing needed -- by
    definition every row of it is held out) so it is not represented here.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray

    def __post_init__(self) -> None:
        assert_disjoint(self.train_idx, self.val_idx)

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)


def make_inner_split(
    df: pd.DataFrame,
    indices: np.ndarray,
    val_fraction: float = 0.2,
    label_col: str = "label",
    group_col: Optional[str] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split `indices` into `(train_idx, val_idx)`, stratified by label.

    Used to create the validation set for early stopping / checkpoint
    selection, strictly separate from any held-out test fold or external
    test center.
    """
    indices = np.asarray(indices)
    sub = df.iloc[indices]

    if group_col is not None and sub[group_col].duplicated().any():
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr_rel, va_rel = next(splitter.split(sub, sub[label_col], groups=sub[group_col]))
    else:
        tr_rel, va_rel = train_test_split(
            np.arange(len(indices)),
            test_size=val_fraction,
            stratify=sub[label_col].to_numpy(),
            random_state=seed,
        )

    train_idx = indices[tr_rel]
    val_idx = indices[va_rel]
    assert_disjoint(train_idx, val_idx)
    return train_idx, val_idx


def make_outer_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_col: str = "label",
    group_col: Optional[str] = None,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield `(train_val_idx, test_idx)` pairs for K-fold CV (outer loop)."""
    labels = df[label_col].to_numpy()

    if group_col is not None and df[group_col].duplicated().any():
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(df, labels, groups=df[group_col])
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(df, labels)


def make_nested_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    val_fraction: float = 0.2,
    label_col: str = "label",
    group_col: Optional[str] = None,
    seed: int = 42,
) -> list[DataSplit]:
    """Full nested K-fold protocol.

    For each outer fold, produce a `DataSplit` with disjoint train/val/test
    index arrays. `test_idx` must only be consumed once, at final evaluation.
    """
    splits = []
    for train_val_idx, test_idx in make_outer_folds(df, n_splits, label_col, group_col, seed):
        train_idx, val_idx = make_inner_split(
            df, train_val_idx, val_fraction, label_col, group_col, seed
        )
        splits.append(DataSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
    return splits


def make_external_validation_split(
    df_train_center: pd.DataFrame,
    val_fraction: float = 0.2,
    label_col: str = "label",
    group_col: Optional[str] = None,
    seed: int = 42,
) -> ExternalSplit:
    """Nested split for bidirectional external validation.

    The training center is split into train/val (for early stopping); the
    caller's external test-center dataframe is used in full and must never be
    passed to anything that influences training (early stopping, checkpoint
    selection, or threshold selection) -- only to a final `evaluate()` call.
    """
    all_idx = np.arange(len(df_train_center))
    train_idx, val_idx = make_inner_split(
        df_train_center, all_idx, val_fraction, label_col, group_col, seed
    )
    return ExternalSplit(train_idx=train_idx, val_idx=val_idx)


def apply_split(df: pd.DataFrame, split: DataSplit) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_train, df_val, df_test) sub-frames for a `DataSplit`."""
    df_train = df.iloc[split.train_idx].reset_index(drop=True)
    df_val = df.iloc[split.val_idx].reset_index(drop=True)
    df_test = df.iloc[split.test_idx].reset_index(drop=True)
    return df_train, df_val, df_test


def apply_external_split(
    df_train_center: pd.DataFrame,
    split: ExternalSplit,
    df_test_center: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_train, df_val, df_test) for external validation.

    `df_test` is the entire external-center dataframe, untouched.
    """
    df_train = df_train_center.iloc[split.train_idx].reset_index(drop=True)
    df_val = df_train_center.iloc[split.val_idx].reset_index(drop=True)
    df_test = df_test_center.reset_index(drop=True)
    return df_train, df_val, df_test
