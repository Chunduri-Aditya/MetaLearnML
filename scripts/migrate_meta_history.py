#!/usr/bin/env python3
"""Migrate legacy MetaLearnML experiment history into the strict v2 schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.meta_history import (  # noqa: E402
    META_HISTORY_SCHEMA_VERSION,
    REQUIRED_HISTORY_COLUMNS,
    validate_history_dataframe,
)
from models.model_zoo import MODEL_REGISTRY  # noqa: E402

LEGACY_PREPROCESSING = {
    "basic_numeric_only": ("standard", "median", "none", "none"),
    "numeric_plus_cat_onehot": ("standard", "median", "onehot", "most_frequent"),
    "robust_numeric": ("robust", "median", "onehot", "most_frequent"),
    "text_tfidf_basic": ("unknown", "unknown", "unknown", "unknown"),
}


def _generated_name_lookup() -> dict[str, tuple[str, str, str, str]]:
    lookup = dict(LEGACY_PREPROCESSING)
    scalers = ("standard", "minmax", "robust", "power")
    numeric_imputers = ("mean", "median", "knn")
    encoders = ("onehot", "ordinal")
    categorical_imputers = ("most_frequent", "constant")
    for scaler in scalers:
        for numeric_imputer in numeric_imputers:
            lookup[f"{numeric_imputer}_{scaler}"] = (
                scaler,
                numeric_imputer,
                "none",
                "none",
            )
            for encoder in encoders:
                for categorical_imputer in categorical_imputers:
                    lookup[f"{numeric_imputer}_{scaler}_{encoder}_{categorical_imputer}"] = (
                        scaler,
                        numeric_imputer,
                        encoder,
                        categorical_imputer,
                    )
    for encoder in encoders:
        for categorical_imputer in categorical_imputers:
            lookup[f"{encoder}_{categorical_imputer}"] = (
                "none",
                "none",
                encoder,
                categorical_imputer,
            )
    return lookup


GENERATED_NAME_LOOKUP = _generated_name_lookup()


def migrate_preprocessing_name(name: str) -> tuple[str, str, str, str]:
    if name in GENERATED_NAME_LOOKUP:
        components = GENERATED_NAME_LOOKUP[name]
        if "unknown" in components:
            warnings.warn(
                f"Legacy preprocessing name {name!r} is unsupported in v2; "
                "mapping its components to 'unknown'",
                stacklevel=2,
            )
        return components
    warnings.warn(
        f"Could not deterministically migrate preprocessing name {name!r}; "
        "mapping every component to 'unknown'",
        stacklevel=2,
    )
    return ("unknown", "unknown", "unknown", "unknown")


def _candidate_id(
    model_identity: str,
    family: str,
    components: tuple[str, str, str, str],
) -> str:
    scaler, numeric_imputer, encoder, categorical_imputer = components
    payload = {
        "candidate_schema_version": 2,
        "model_identity": model_identity,
        "model_family": family,
        "preprocessing": {
            "numeric_scaler": scaler,
            "numeric_imputer": numeric_imputer,
            "categorical_encoder": encoder,
            "categorical_imputer": categorical_imputer,
        },
        "training": {"legacy_configuration": True},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def migrate_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "schema_version" in dataframe.columns:
        versions = set(pd.to_numeric(dataframe["schema_version"], errors="coerce").dropna())
        if versions == {META_HISTORY_SCHEMA_VERSION} and set(REQUIRED_HISTORY_COLUMNS).issubset(
            dataframe.columns
        ):
            validated = validate_history_dataframe(dataframe, max_invalid_fraction=0.0)
            return validated.dataframe, {
                "input_rows": len(dataframe),
                "output_rows": len(validated.dataframe),
                "duplicates_removed": validated.duplicate_rows,
                "unknown_strategies": int(
                    validated.dataframe[
                        [
                            "numeric_scaler",
                            "numeric_imputer",
                            "categorical_encoder",
                            "categorical_imputer",
                        ]
                    ]
                    .eq("unknown")
                    .any(axis=1)
                    .sum()
                ),
                "unknown_models": int(validated.dataframe["model_identity"].eq("unknown").sum()),
            }
        if META_HISTORY_SCHEMA_VERSION in versions:
            raise ValueError("Cannot migrate a file that mixes v2 and legacy history rows")
    rows = []
    unknown_strategies = 0
    unknown_models = 0
    for index, row in dataframe.iterrows():
        preprocessing_name = str(row.get("preprocessing_name", "unknown"))
        components = migrate_preprocessing_name(preprocessing_name)
        if "unknown" in components:
            unknown_strategies += 1
        model_identity = str(row.get("model_identity", row.get("model_name", "unknown")))
        entry = MODEL_REGISTRY.get(model_identity)
        family = str(
            row.get(
                "model_family",
                row.get("family", entry.get("family") if entry else "unknown"),
            )
        )
        if entry is None:
            unknown_models += 1
            model_identity = "unknown"
            family = "unknown"
        scaler, numeric_imputer, encoder, categorical_imputer = components
        dataset_name = str(row.get("dataset_name", "legacy_dataset"))
        metric_value = float(row.get("metric_value", row.get("metric", 0.0)))
        legacy_task = str(row.get("task_type", "unknown"))
        if "classification" in legacy_task:
            task_type = "classification"
        elif legacy_task == "regression":
            task_type = "regression"
        else:
            task_type = legacy_task
        rows.append(
            {
                "schema_version": META_HISTORY_SCHEMA_VERSION,
                "run_id": str(row.get("run_id", f"legacy-{index}")),
                "dataset_id": str(row.get("dataset_id", f"legacy:{dataset_name}")),
                "dataset_version": str(row.get("dataset_version", "legacy")),
                "dataset_checksum": str(
                    row.get(
                        "dataset_checksum",
                        hashlib.sha256(dataset_name.encode("utf-8")).hexdigest(),
                    )
                ),
                "dataset_name": dataset_name,
                "task_type": task_type,
                "n_samples": int(row.get("n_samples", 0)),
                "n_features": int(row.get("n_features", 0)),
                "n_numeric": int(row.get("n_numeric", 0)),
                "n_categorical": int(row.get("n_categorical", 0)),
                "n_text": int(row.get("n_text", 0)),
                "n_classes": int(row.get("n_classes", 0) or 0),
                "class_imbalance": float(row.get("class_imbalance", 0) or 0),
                "missing_value_fraction": float(row.get("missing_value_fraction", 0)),
                "candidate_id": _candidate_id(model_identity, family, components),
                "model_identity": model_identity,
                "model_family": family,
                "preprocessing_name": preprocessing_name,
                "numeric_scaler": scaler,
                "numeric_imputer": numeric_imputer,
                "categorical_encoder": encoder,
                "categorical_imputer": categorical_imputer,
                "metric_name": str(row.get("metric_name", "unknown")),
                "metric_objective_value": metric_value,
                "metric_display_value": metric_value,
                "outer_split_seed": int(row.get("split_seed", row.get("seed", 0)) or 0),
                "inner_fold_id": int(row.get("fold_id", 0) or 0),
                "repository_commit": str(row.get("repository_commit", "legacy")),
                "environment_hash": str(row.get("environment_hash", "legacy")),
            }
        )
    migrated = pd.DataFrame(rows)
    summary = validate_history_dataframe(migrated, max_invalid_fraction=0.0)
    return summary.dataframe, {
        "input_rows": len(dataframe),
        "output_rows": len(summary.dataframe),
        "duplicates_removed": summary.duplicate_rows,
        "unknown_strategies": unknown_strategies,
        "unknown_models": unknown_models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input; migration never overwrites its source")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    if summary_path.exists():
        parser.error(f"Summary output already exists: {summary_path}")
    migrated, summary = migrate_dataframe(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    migrated.to_csv(args.output, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
