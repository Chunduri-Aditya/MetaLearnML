from __future__ import annotations

import json
from datetime import UTC, datetime

from engine.report_generator import (
    generate_json_report,
    generate_markdown_report,
    generate_report,
    generate_text_report,
)


def _payload():
    return generate_json_report(
        "fixture",
        "classification",
        {
            "candidate_id": "abc",
            "model_name": "rf",
            "preprocessing_name": "standard",
            "metric_name": "f1_macro",
            "selection_cv_mean": 0.8,
            "selection_cv_std": 0.02,
            "test_score": 0.75,
            "generalization_gap": 0.05,
        },
        [
            {
                "model_name": "rf",
                "preprocessing_name": "standard",
                "selection_cv_mean": 0.8,
                "selection_cv_std": 0.02,
                "status": "success",
            }
        ],
        {"n_samples": 80, "n_features": 4, "missing_value_fraction": 0.1},
        1.2,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_reports_distinguish_cv_and_untouched_test(tmp_path):
    payload = _payload()
    markdown = generate_markdown_report(payload)
    text = generate_text_report(payload)
    assert "Selection CV mean" in markdown
    assert "Untouched test score" in markdown
    assert "Untouched test" in text
    paths = generate_report(
        "fixture",
        "classification",
        {
            "candidate_id": "abc",
            "model_name": "rf",
            "preprocessing_name": "standard",
            "selection_cv_mean": 0.8,
            "test_score": 0.75,
        },
        [],
        {"n_samples": 80},
        1.0,
        str(tmp_path),
    )
    saved = json.loads(open(paths["json"]).read())
    assert saved["report_schema_version"] == 2
    assert saved["final_test"]["test_score"] == 0.75
