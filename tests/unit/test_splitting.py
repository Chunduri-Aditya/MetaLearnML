from __future__ import annotations

import numpy as np
import pytest

from engine.splitting import (
    create_early_stopping_split,
    create_inner_splits,
    create_outer_split,
    encode_targets,
)


def test_stratified_outer_split_is_disjoint_and_complete():
    target = np.repeat(["a", "b", "c"], 20)
    split = create_outer_split(target, "classification", random_state=9)
    assert not np.intersect1d(split.development_indices, split.test_indices).size
    assert sorted(np.r_[split.development_indices, split.test_indices]) == list(range(60))
    assert set(target[split.test_indices]) == {"a", "b", "c"}


def test_regression_outer_split_is_reproducible():
    target = np.arange(50, dtype=float)
    first = create_outer_split(target, "regression", random_state=3)
    second = create_outer_split(target, "regression", random_state=3)
    assert np.array_equal(first.test_indices, second.test_indices)


def test_invalid_class_counts_fail_usefully():
    with pytest.raises(ValueError, match="at least two rows"):
        create_outer_split([0, 0, 1], "classification")
    with pytest.raises(ValueError, match="smallest class"):
        create_inner_splits([0, 0, 0, 1, 1], "classification", cv_folds=3)


def test_inner_repeated_folds_and_early_stop_are_isolated():
    target = np.repeat([0, 1], 20)
    folds = create_inner_splits(target, "classification", cv_folds=4, cv_repeats=2)
    assert len(folds) == 8
    for fold in folds:
        assert not np.intersect1d(fold.train_indices, fold.validation_indices).size
    fit, early = create_early_stopping_split(target, "classification", random_state=2)
    assert not np.intersect1d(fit, early).size
    assert set(fit) | set(early) == set(range(len(target)))


def test_label_encoder_fits_development_only():
    encoded = encode_targets(["a", "b", "a"], ["b"], "classification")
    assert encoded.label_encoder.classes_.tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="absent from development"):
        encode_targets(["a", "b"], ["c"], "classification")


def test_regression_targets_are_float_and_unencoded():
    encoded = encode_targets([1, 2], [3], "regression")
    assert encoded.development.dtype == np.float32
    assert encoded.label_encoder is None
