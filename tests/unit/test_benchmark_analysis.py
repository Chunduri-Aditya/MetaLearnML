from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from benchmarks.analyze import evidence_hashes, generate_evidence
from benchmarks.protocol import budgets, candidate_universe_hash
from engine.search import generate_candidate_universe


def test_budget_deduplication_and_universe_hash(numeric_strategy):
    meta = {"n_numeric": 2, "n_categorical": 0, "n_samples": 50}
    candidates = generate_candidate_universe(
        meta, "regression", strategies=[numeric_strategy], max_epochs=1
    )
    assert len(candidate_universe_hash(candidates)) == 64
    assert budgets(9, [1, 3, 5, 10], [0.25, 0.5, 1]) == [1, 3, 5, 9]


def test_committed_evidence_hashes_are_complete():
    root = Path(__file__).resolve().parents[2] / "benchmarks"
    hashes = evidence_hashes(root)
    assert len(hashes) == 8
    assert all(len(value) == 64 for value in hashes.values())
    summary = json.loads((root / "results" / "aggregate_summary.json").read_text())
    assert summary["dataset_count"] == 15
    assert summary["speedup_claim_eligible"] is False


def test_evidence_reads_method_results_floats_exactly(tmp_path):
    """The median of three per-seed rows is the middle source value, bit for bit.

    pandas' default C float parser is not correctly rounded, and its last bits differ
    between arm64 and x86_64 builds. That made report-check fail on Linux only.
    """
    root = Path(__file__).resolve().parents[2] / "benchmarks"
    generate_evidence(results_dir=root / "results", output_root=tmp_path)
    with (root / "results" / "method_results.csv").open(newline="", encoding="utf-8") as handle:
        source = [row for row in csv.DictReader(handle) if row["method"] == "exhaustive"]
    with (tmp_path / "results" / "per_dataset_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = [row for row in csv.DictReader(handle) if row["method"] == "exhaustive"]
    assert len(summary) == 15
    for row in summary:
        per_seed = [
            float(item["cached_total_seconds"])
            for item in source
            if item["dataset_id"] == row["dataset_id"]
        ]
        assert len(per_seed) == 3
        assert float(row["median_cached_total_seconds"]) == statistics.median(per_seed), row[
            "dataset_id"
        ]
