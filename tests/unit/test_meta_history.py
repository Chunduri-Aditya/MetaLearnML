from __future__ import annotations

import pandas as pd
import pytest

from engine.meta_history import (
    MetaHistoryValidationError,
    append_history_records,
    load_history,
    validate_history_dataframe,
)
from scripts.migrate_meta_history import migrate_dataframe, migrate_preprocessing_name


def test_history_round_trip_preserves_missingness_and_text(tmp_path, history_frame):
    path = tmp_path / "history.csv"
    history_frame.to_csv(path, index=False)
    loaded = load_history(path, min_valid_rows=20)
    assert loaded.accepted_rows == 24
    assert loaded.dataframe.loc[0, "missing_value_fraction"] == 0.2
    assert loaded.dataframe.loc[0, "n_text"] == 0


def test_history_missing_columns_always_fail(history_frame):
    with pytest.raises(MetaHistoryValidationError, match="missing required"):
        validate_history_dataframe(history_frame.drop(columns=["n_text"]))


def test_history_rejection_threshold_and_reason(history_frame):
    corrupt = history_frame.copy()
    corrupt.loc[:2, "missing_value_fraction"] = 2.0
    with pytest.raises(MetaHistoryValidationError, match="range:missing_value_fraction"):
        validate_history_dataframe(corrupt)


def test_append_deduplicates_idempotently(tmp_path, history_frame):
    path = tmp_path / "history.csv"
    records = []
    for row in history_frame.to_dict("records"):
        from engine.meta_history import MetaHistoryRecord

        records.append(MetaHistoryRecord(**row))
    first = append_history_records(path, records)
    second = append_history_records(path, records)
    assert len(first.dataframe) == len(second.dataframe) == 24
    assert second.duplicate_rows == 24


def test_legacy_migration_known_unknown_and_idempotent():
    legacy = pd.DataFrame(
        [
            {
                "dataset_name": "old",
                "task_type": "classification",
                "model_name": "rf",
                "preprocessing_name": "numeric_plus_cat_onehot",
                "metric_name": "f1_macro",
                "metric": 0.8,
                "n_samples": 100,
                "n_features": 4,
                "n_numeric": 3,
                "n_categorical": 1,
            }
        ]
    )
    migrated, summary = migrate_dataframe(legacy)
    assert summary["output_rows"] == 1
    assert migrated.loc[0, "numeric_scaler"] == "standard"
    repeated, repeated_summary = migrate_dataframe(migrated)
    pd.testing.assert_frame_equal(migrated, repeated)
    assert repeated_summary["output_rows"] == 1
    with pytest.warns(UserWarning, match="Could not"):
        assert migrate_preprocessing_name("mystery") == ("unknown",) * 4
