"""Tests for the leakage-safe nested split protocol (R1.1).

These are the most important tests in the framework: they prove that the
train/val/test index sets are always disjoint, and that the API makes it
structurally difficult to repeat the original notebook's mistake (passing the
held-out fold to early stopping).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtl_lymphnet.data.splits import (
    ExternalSplit,
    DataSplit,
    apply_external_split,
    apply_split,
    assert_disjoint,
    make_external_validation_split,
    make_inner_split,
    make_nested_kfold_splits,
    make_outer_folds,
)


def _make_df(n_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = n_per_class * 2
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    return pd.DataFrame(
        {
            "sample_id": np.arange(n),
            "label": labels,
            "value": rng.rand(n),
        }
    )


def test_assert_disjoint_passes_for_disjoint_arrays():
    assert_disjoint(np.array([1, 2, 3]), np.array([4, 5]), np.array([6]))


def test_assert_disjoint_raises_for_overlap():
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(np.array([1, 2, 3]), np.array([3, 4]))


def test_data_split_rejects_overlapping_indices():
    with pytest.raises(ValueError):
        DataSplit(train_idx=np.array([0, 1]), val_idx=np.array([1, 2]), test_idx=np.array([3]))


def test_external_split_rejects_overlapping_indices():
    with pytest.raises(ValueError):
        ExternalSplit(train_idx=np.array([0, 1]), val_idx=np.array([1, 2]))


def test_make_inner_split_is_disjoint_and_covers_all_indices():
    df = _make_df()
    indices = np.arange(len(df))
    train_idx, val_idx = make_inner_split(df, indices, val_fraction=0.2, seed=42)

    assert_disjoint(train_idx, val_idx)
    assert set(train_idx) | set(val_idx) == set(indices)
    # Roughly 80/20 (allow rounding)
    assert abs(len(val_idx) / len(indices) - 0.2) < 0.05


def test_make_outer_folds_cover_all_rows_exactly_once_as_test():
    df = _make_df()
    seen_test_indices: list[int] = []
    for train_val_idx, test_idx in make_outer_folds(df, n_splits=5, seed=42):
        assert_disjoint(train_val_idx, test_idx)
        seen_test_indices.extend(test_idx.tolist())

    assert sorted(seen_test_indices) == list(range(len(df)))


def test_make_nested_kfold_splits_are_fully_disjoint_per_fold():
    df = _make_df()
    splits = make_nested_kfold_splits(df, n_splits=5, val_fraction=0.2, seed=42)

    assert len(splits) == 5
    for split in splits:
        assert_disjoint(split.train_idx, split.val_idx, split.test_idx)
        # every row accounted for exactly once within this fold
        all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
        assert sorted(all_idx.tolist()) == list(range(len(df)))


def test_nested_kfold_splits_deterministic_given_same_seed():
    df = _make_df()
    splits_a = make_nested_kfold_splits(df, n_splits=5, val_fraction=0.2, seed=42)
    splits_b = make_nested_kfold_splits(df, n_splits=5, val_fraction=0.2, seed=42)

    for a, b in zip(splits_a, splits_b):
        assert np.array_equal(a.train_idx, b.train_idx)
        assert np.array_equal(a.val_idx, b.val_idx)
        assert np.array_equal(a.test_idx, b.test_idx)


def test_apply_split_returns_correct_row_counts():
    df = _make_df()
    split = make_nested_kfold_splits(df, n_splits=5, val_fraction=0.2, seed=42)[0]
    df_train, df_val, df_test = apply_split(df, split)

    assert len(df_train) == split.n_train
    assert len(df_val) == split.n_val
    assert len(df_test) == split.n_test
    assert len(df_train) + len(df_val) + len(df_test) == len(df)


def test_external_validation_split_never_touches_test_center():
    df_train_center = _make_df(n_per_class=30, seed=1)
    df_test_center = _make_df(n_per_class=20, seed=2)

    split = make_external_validation_split(df_train_center, val_fraction=0.2, seed=42)
    assert_disjoint(split.train_idx, split.val_idx)

    df_train, df_val, df_test = apply_external_split(df_train_center, split, df_test_center)
    assert len(df_train) + len(df_val) == len(df_train_center)
    # The external test set is used in full and is untouched by the split.
    assert len(df_test) == len(df_test_center)
    pd.testing.assert_frame_equal(df_test.reset_index(drop=True), df_test_center.reset_index(drop=True))
