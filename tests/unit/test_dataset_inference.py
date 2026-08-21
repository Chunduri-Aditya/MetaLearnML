from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.datasets import (
    _auto_detect_label_column,
    analyze_target,
    compute_dataset_metadata,
    infer_feature_types,
    infer_task_type,
    load_csv_classification,
    load_raw_tabular_dataset,
    load_tabular_dataset,
    validate_dataset_size,
)


def test_missing_fraction_exact_and_empty_edge_case():
    frame = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4))
    frame.iloc[0, :] = np.nan
    meta = compute_dataset_metadata(frame, np.arange(5), "regression")
    assert meta["missing_value_fraction"] == 0.2
    empty = compute_dataset_metadata(pd.DataFrame(index=range(3)), [1, 2, 3], "regression")
    assert empty["missing_value_fraction"] == 0.0


def test_text_is_inferred_and_rejected():
    frame = pd.DataFrame(
        {"notes": [f"long natural language sentence number {index}" for index in range(20)]}
    )
    assert infer_feature_types(frame)["notes"] == "text"
    with pytest.raises(ValueError, match="free-text"):
        compute_dataset_metadata(frame, np.arange(20), "regression", reject_text=True)


def test_codes_are_categorical_not_text():
    frame = pd.DataFrame({"zip": [f"{90000 + index}" for index in range(30)]})
    assert infer_feature_types(frame)["zip"] == "categorical"


@pytest.mark.parametrize(
    ("values", "expected"),
    [([0.1, 0.2, 0.3], "regression"), ([0, 1, 0, 1], "binary_classification")],
)
def test_task_inference(values, expected):
    assert infer_task_type(pd.Series(values)) == expected


def test_raw_and_compatibility_loaders(tiny_csv):
    raw = load_raw_tabular_dataset(str(tiny_csv), label_col="label")
    loaded = load_tabular_dataset(str(tiny_csv), label_col="label")
    assert raw["X_df"].shape == (30, 2)
    assert loaded["meta"]["missing_value_fraction"] == 0.0
    assert set(loaded["y"]) == {0, 1}


def test_legacy_classification_loader_and_target_helpers(tiny_csv):
    descriptor = load_csv_classification(str(tiny_csv), label_col="label")
    assert descriptor.n_samples == 30
    assert descriptor.n_classes == 2
    assert len(descriptor.to_tensor_dataset()) == 30
    assert _auto_detect_label_column(pd.read_csv(tiny_csv)) == "label"
    assert analyze_target(pd.Series(["a", "a", "b"]))["target_type"] == "binary"
    with pytest.raises(ValueError, match="too small"):
        validate_dataset_size(4, 2)
