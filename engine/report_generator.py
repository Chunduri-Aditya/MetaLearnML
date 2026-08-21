"""Human- and machine-readable reports with separate selection/test results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA_VERSION = 2


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _successful_results(all_results: list[dict]) -> list[dict]:
    return sorted(
        (result for result in all_results if result.get("selection_cv_mean") is not None),
        key=lambda result: float(result["selection_cv_mean"]),
        reverse=True,
    )


def generate_report(
    dataset_name: str,
    task_type: str,
    best_info: dict,
    all_results: list[dict],
    dataset_meta: dict,
    time_taken: float,
    output_dir: str = "experiments/reports",
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC)
    base_name = f"{dataset_name}_{generated_at.strftime('%Y%m%d_%H%M%S')}"
    payload = generate_json_report(
        dataset_name,
        task_type,
        best_info,
        all_results,
        dataset_meta,
        time_taken,
        generated_at=generated_at,
    )
    paths = {
        "markdown": output / f"{base_name}.md",
        "json": output / f"{base_name}.json",
        "text": output / f"{base_name}.txt",
    }
    paths["markdown"].write_text(generate_markdown_report(payload))
    paths["json"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    paths["text"].write_text(generate_text_report(payload))
    return {name: str(path) for name, path in paths.items()}


def generate_json_report(
    dataset_name: str,
    task_type: str,
    best_info: dict,
    all_results: list[dict],
    dataset_meta: dict,
    time_taken: float,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC)
    return _native(
        {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": generated_at.isoformat(),
            "dataset_name": dataset_name,
            "task_type": task_type,
            "dataset_meta": dataset_meta,
            "selection": {
                "candidate_id": best_info.get("candidate_id"),
                "model_identity": best_info.get("model_name"),
                "preprocessing_name": best_info.get("preprocessing_name"),
                "metric_name": best_info.get("metric_name"),
                "selection_cv_mean": best_info.get("selection_cv_mean"),
                "selection_cv_std": best_info.get("selection_cv_std"),
            },
            "final_test": {
                "test_score": best_info.get("test_score"),
                "generalization_gap": best_info.get("generalization_gap"),
            },
            "bundle_path": best_info.get("bundle_path"),
            "candidate_results": all_results,
            "performance": {
                "time_taken_seconds": time_taken,
                "candidates_attempted": len(all_results),
                "candidate_failures": sum(
                    result.get("status") != "success" for result in all_results
                ),
            },
        }
    )


def generate_markdown_report(payload: dict[str, Any], *legacy_args: Any) -> str:
    if legacy_args:
        raise TypeError("generate_markdown_report now accepts a schema-v2 report payload")
    meta = payload["dataset_meta"]
    selection = payload["selection"]
    final = payload["final_test"]
    lines = [
        f"# AutoML Experiment Report: {payload['dataset_name']}",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Dataset",
        "",
        f"- Task: {payload['task_type']}",
        f"- Development rows: {meta.get('n_samples', 'N/A')}",
        f"- Features: {meta.get('n_features', 'N/A')}",
        f"- Missing-value fraction: {meta.get('missing_value_fraction', 'N/A')}",
        "",
        "## Selection and final evaluation",
        "",
        f"- Candidate: {selection.get('candidate_id') or 'N/A'}",
        f"- Model: {selection.get('model_identity') or 'N/A'}",
        f"- Preprocessing: {selection.get('preprocessing_name') or 'N/A'}",
        f"- Selection CV mean: {selection.get('selection_cv_mean')}",
        f"- Selection CV std: {selection.get('selection_cv_std')}",
        f"- Untouched test score: {final.get('test_score')}",
        f"- Direction-adjusted generalization gap: {final.get('generalization_gap')}",
        "",
        "## Candidate results",
        "",
        "| Model | Preprocessing | CV mean | CV std | Status |",
        "|---|---|---:|---:|---|",
    ]
    for result in _successful_results(payload["candidate_results"]):
        lines.append(
            f"| {result.get('model_name')} | {result.get('preprocessing_name')} | "
            f"{result.get('selection_cv_mean')} | {result.get('selection_cv_std')} | "
            f"{result.get('status')} |"
        )
    return "\n".join(lines) + "\n"


def generate_text_report(payload: dict[str, Any], *legacy_args: Any) -> str:
    if legacy_args:
        raise TypeError("generate_text_report now accepts a schema-v2 report payload")
    selection = payload["selection"]
    final = payload["final_test"]
    return (
        f"MetaLearnML report: {payload['dataset_name']}\n"
        f"Task: {payload['task_type']}\n"
        f"Selected: {selection.get('model_identity')} + "
        f"{selection.get('preprocessing_name')}\n"
        f"CV: {selection.get('selection_cv_mean')} ± {selection.get('selection_cv_std')}\n"
        f"Untouched test: {final.get('test_score')}\n"
        f"Generalization gap: {final.get('generalization_gap')}\n"
    )
