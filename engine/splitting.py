"""Deterministic, leakage-resistant split definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder


def is_classification(task_type: str) -> bool:
    return task_type in {"binary_classification", "multiclass_classification", "classification"}


@dataclass(frozen=True)
class OuterSplit:
    development_indices: np.ndarray
    test_indices: np.ndarray
    random_state: int


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    repeat_id: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class EncodedTargets:
    development: np.ndarray
    test: np.ndarray
    label_encoder: LabelEncoder | None


def create_outer_split(
    y: Any,
    task_type: str,
    *,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> OuterSplit:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be within (0, 1), got {test_ratio}")
    y_array = np.asarray(y)
    if len(y_array) < 2:
        raise ValueError("At least two rows are required for a development/test split")
    indices = np.arange(len(y_array))
    stratify = None
    if is_classification(task_type):
        classes, counts = np.unique(y_array, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two target classes")
        if counts.min() < 2:
            raise ValueError(
                "Every classification class needs at least two rows for a stratified test split"
            )
        test_size = max(len(classes), math.ceil(test_ratio * len(y_array)))
        if test_size >= len(y_array):
            raise ValueError("The requested test split leaves no development rows")
        stratify = y_array
    else:
        test_size = test_ratio
    development, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    development = np.asarray(sorted(development), dtype=np.int64)
    test = np.asarray(sorted(test), dtype=np.int64)
    if np.intersect1d(development, test).size:
        raise RuntimeError("Development/test split indices overlap")
    if len(development) + len(test) != len(indices):
        raise RuntimeError("Development/test split does not cover every row")
    return OuterSplit(development, test, random_state)


def encode_targets(
    y_development: Any,
    y_test: Any,
    task_type: str,
) -> EncodedTargets:
    if not is_classification(task_type):
        return EncodedTargets(
            np.asarray(y_development, dtype=np.float32),
            np.asarray(y_test, dtype=np.float32),
            None,
        )
    encoder = LabelEncoder()
    development = encoder.fit_transform(np.asarray(y_development))
    known = set(encoder.classes_.tolist())
    unseen = sorted(set(np.asarray(y_test).tolist()) - known, key=str)
    if unseen:
        raise ValueError(
            f"The untouched test partition contains target classes absent from development: {unseen}"
        )
    test = encoder.transform(np.asarray(y_test))
    dtype = np.int64
    return EncodedTargets(development.astype(dtype), test.astype(dtype), encoder)


def create_inner_splits(
    y_development: Any,
    task_type: str,
    *,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    random_state: int = 42,
) -> list[FoldSplit]:
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if cv_repeats < 1:
        raise ValueError("cv_repeats must be at least 1")
    y_array = np.asarray(y_development)
    if len(y_array) < cv_folds:
        raise ValueError(f"cv_folds={cv_folds} exceeds development rows={len(y_array)}")
    if is_classification(task_type):
        _, counts = np.unique(y_array, return_counts=True)
        if counts.min() < cv_folds:
            raise ValueError(
                f"cv_folds={cv_folds} requires at least {cv_folds} development rows in "
                f"every class; smallest class has {int(counts.min())}"
            )
        splitter = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            if cv_repeats == 1
            else RepeatedStratifiedKFold(
                n_splits=cv_folds,
                n_repeats=cv_repeats,
                random_state=random_state,
            )
        )
        iterator = splitter.split(np.zeros((len(y_array), 1)), y_array)
    else:
        splitter = (
            KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            if cv_repeats == 1
            else RepeatedKFold(
                n_splits=cv_folds,
                n_repeats=cv_repeats,
                random_state=random_state,
            )
        )
        iterator = splitter.split(np.zeros((len(y_array), 1)))
    result = []
    for split_number, (train, validation) in enumerate(iterator):
        if np.intersect1d(train, validation).size:
            raise RuntimeError(f"Inner fold {split_number} has overlapping indices")
        result.append(
            FoldSplit(
                fold_id=split_number,
                repeat_id=split_number // cv_folds,
                train_indices=np.asarray(train, dtype=np.int64),
                validation_indices=np.asarray(validation, dtype=np.int64),
            )
        )
    return result


def create_early_stopping_split(
    y_fold_train: Any,
    task_type: str,
    *,
    random_state: int,
    ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    y_array = np.asarray(y_fold_train)
    indices = np.arange(len(y_array))
    if len(indices) < 4:
        raise ValueError("At least four fold-training rows are required for early stopping")
    if is_classification(task_type):
        classes, counts = np.unique(y_array, return_counts=True)
        if counts.min() < 2:
            raise ValueError("Every class needs at least two fold-training rows for early stopping")
        validation_size = max(len(classes), math.ceil(ratio * len(indices)))
        if validation_size >= len(indices):
            raise ValueError("Early-stopping split leaves no fitting rows")
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=random_state,
        )
        fit, early = next(splitter.split(np.zeros((len(indices), 1)), y_array))
    else:
        fit, early = train_test_split(
            indices,
            test_size=ratio,
            random_state=random_state,
            shuffle=True,
        )
    return np.asarray(fit, dtype=np.int64), np.asarray(early, dtype=np.int64)
