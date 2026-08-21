from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

import benchmarks.manifest as manifest_module
from benchmarks.manifest import (
    LockedDataset,
    ManifestValidationError,
    sha256_file,
    validate_locked_datasets,
    validate_manifest_definition,
)


def _locked_fixture(tmp_path):
    cache_dir = tmp_path / "cache" / "datasets"
    cache_dir.mkdir(parents=True)
    snapshot = cache_dir / "fixture.csv"
    pd.DataFrame({"feature": [1, 2], "target": ["a", "b"]}).to_csv(snapshot, index=False)
    record = LockedDataset(
        benchmark_id="fixture",
        openml_task_id=123,
        task_type="binary_classification",
        openml_dataset_id=456,
        openml_dataset_version=1,
        dataset_name="fixture",
        target_column="target",
        license="CC0",
        license_source="fixture",
        provider_checksum="provider-checksum",
        local_sha256=sha256_file(snapshot),
        cache_file="cache/datasets/fixture.csv",
        n_rows=2,
        n_columns=2,
        retrieved_at="fixture",
    )
    definition = {
        "schema_version": 1,
        "provider": "openml",
        "datasets": [
            {"id": "fixture", "openml_task_id": 123, "task_type": "binary_classification"}
        ],
    }
    return cache_dir, definition, record


def test_manifest_definition_requires_pinned_unique_openml_tasks():
    valid = {
        "schema_version": 1,
        "provider": "openml",
        "datasets": [{"id": "one", "openml_task_id": 1, "task_type": "binary_classification"}],
    }
    validate_manifest_definition(valid)
    duplicate = {**valid, "datasets": valid["datasets"] * 2}
    with pytest.raises(ManifestValidationError, match="Duplicate"):
        validate_manifest_definition(duplicate)
    missing_task = {**valid, "datasets": [{"id": "one", "task_type": "regression"}]}
    with pytest.raises(ManifestValidationError, match="missing fields"):
        validate_manifest_definition(missing_task)


def test_locked_manifest_blocks_checksum_target_license_and_task_changes(tmp_path, monkeypatch):
    cache_dir, definition, record = _locked_fixture(tmp_path)
    monkeypatch.setattr(manifest_module, "BENCHMARK_ROOT", tmp_path)
    assert validate_locked_datasets(definition, [record], cache_dir=cache_dir) == [record]
    with pytest.raises(ManifestValidationError, match="Checksum mismatch"):
        validate_locked_datasets(
            definition,
            [replace(record, local_sha256="0" * 64)],
            cache_dir=cache_dir,
        )
    with pytest.raises(ManifestValidationError, match="missing or ambiguous"):
        validate_locked_datasets(
            definition,
            [replace(record, license="unknown")],
            cache_dir=cache_dir,
        )
    with pytest.raises(ManifestValidationError, match="has no target"):
        validate_locked_datasets(
            definition,
            [replace(record, target_column="absent")],
            cache_dir=cache_dir,
        )
    with pytest.raises(ManifestValidationError, match="Task type mismatch"):
        validate_locked_datasets(
            definition,
            [replace(record, task_type="regression")],
            cache_dir=cache_dir,
        )
