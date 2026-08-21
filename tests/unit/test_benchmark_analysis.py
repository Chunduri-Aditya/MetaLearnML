from __future__ import annotations

import json
from pathlib import Path

from benchmarks.analyze import evidence_hashes
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
    assert len(hashes) == 9
    assert all(len(value) == 64 for value in hashes.values())
    summary = json.loads((root / "results" / "aggregate_summary.json").read_text())
    assert summary["dataset_count"] == 15
    assert summary["impact_resume_bullet_eligible"] is False
