from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd
import pytest

from benchmarks.manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestValidationError,
    load_manifest,
    validate_manifest_definition,
)
from benchmarks.protocol import leave_one_dataset_out
from benchmarks.run import run_smoke
from engine.meta_history import load_history

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_offline_two_dataset_benchmark_smoke(tmp_path):
    output = tmp_path / "smoke"
    run_smoke(output)
    summary = json.loads((output / "smoke_summary.json").read_text())
    raw = pd.read_csv(output / "candidate_results.csv.gz")
    methods = pd.read_csv(output / "method_results.csv")
    assert summary["status"] == "passed" and summary["dataset_count"] == 2
    assert raw["dataset_id"].nunique() == methods["dataset_id"].nunique() == 2
    assert raw["candidate_fold_seed"].notna().all()
    assert set(methods["method"]) == {"exhaustive", "proxy", "meta", "proxy_meta", "random"}
    history = load_history(output / "meta_history_v2.csv", min_valid_rows=20)
    assert history.rejected_rows == 0


def test_committed_benchmark_schemas_and_lodo():
    root = Path(__file__).resolve().parents[2] / "benchmarks"
    raw = pd.read_csv(root / "results" / "candidate_results.csv.gz")
    summary = json.loads((root / "results" / "aggregate_summary.json").read_text())
    candidate_schema = json.loads((root / "schemas" / "candidate_results.schema.json").read_text())
    summary_schema = json.loads((root / "schemas" / "summary.schema.json").read_text())
    jsonschema.validate(raw.iloc[0].dropna().to_dict(), candidate_schema)
    jsonschema.validate(summary, summary_schema)
    history = pd.read_csv(root / "results" / "meta_history_v2.csv")
    target = history.iloc[0]
    lodo = leave_one_dataset_out(
        history,
        target_checksum=target["dataset_checksum"],
        task_type=target["task_type"],
        metric_name=target["metric_name"],
    )
    assert target["dataset_checksum"] not in set(lodo["dataset_checksum"])


def test_manifest_duplicate_detection():
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    duplicate = {**manifest, "datasets": manifest["datasets"] + [manifest["datasets"][0]]}
    with pytest.raises(ManifestValidationError, match="Duplicate"):
        validate_manifest_definition(duplicate)
