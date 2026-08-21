"""Versioned experiment-history schema and validation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from engine.meta_features import META_FEATURE_SCHEMA_VERSION, PREPROCESSING_VOCABULARIES

META_HISTORY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MetaHistoryRecord:
    schema_version: int
    run_id: str
    dataset_id: str
    dataset_version: str
    dataset_checksum: str
    dataset_name: str
    task_type: str
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_text: int
    n_classes: int
    class_imbalance: float
    missing_value_fraction: float
    candidate_id: str
    model_identity: str
    model_family: str
    preprocessing_name: str
    numeric_scaler: str
    numeric_imputer: str
    categorical_encoder: str
    categorical_imputer: str
    metric_name: str
    metric_objective_value: float
    metric_display_value: float
    outer_split_seed: int
    inner_fold_id: int
    repository_commit: str
    environment_hash: str

    @classmethod
    def create(cls, **values: Any) -> MetaHistoryRecord:
        return cls(
            schema_version=META_HISTORY_SCHEMA_VERSION,
            run_id=str(values.pop("run_id", uuid.uuid4())),
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_HISTORY_COLUMNS = tuple(MetaHistoryRecord.__dataclass_fields__)
HISTORY_DEDUPLICATION_COLUMNS = (
    "dataset_checksum",
    "outer_split_seed",
    "inner_fold_id",
    "candidate_id",
    "metric_name",
    "repository_commit",
    "environment_hash",
)


@dataclass(frozen=True)
class HistoryValidationSummary:
    dataframe: pd.DataFrame
    accepted_rows: int
    rejected_rows: int
    rejection_reasons: dict[str, int]
    duplicate_rows: int


class MetaHistoryValidationError(ValueError):
    pass


def repository_commit(workdir: str | Path = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_fingerprint() -> tuple[str, dict[str, str]]:
    cpu = platform.processor().strip()
    if platform.system() == "Darwin":
        try:
            mac_cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            cpu = mac_cpu or cpu
        except (OSError, subprocess.SubprocessError):
            pass
    if not cpu:
        cpu = platform.machine() or "unknown"
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "accelerator": "cpu",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "meta_feature_schema": str(META_FEATURE_SCHEMA_VERSION),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, metadata


def _validate_row(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row.get("schema_version") != META_HISTORY_SCHEMA_VERSION:
        reasons.append("schema_version")
    if row.get("task_type") not in {"classification", "regression"}:
        reasons.append("vocabulary:task_type")
    for column in (
        "run_id",
        "dataset_id",
        "dataset_version",
        "dataset_checksum",
        "dataset_name",
        "task_type",
        "candidate_id",
        "model_identity",
        "model_family",
        "preprocessing_name",
        "metric_name",
        "repository_commit",
        "environment_hash",
    ):
        value = row.get(column)
        if pd.isna(value) or not str(value).strip():
            reasons.append(f"missing:{column}")

    for column in (
        "n_samples",
        "n_features",
        "n_numeric",
        "n_categorical",
        "n_text",
        "n_classes",
        "outer_split_seed",
        "inner_fold_id",
    ):
        value = row.get(column)
        try:
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0:
                reasons.append(f"invalid:{column}")
        except (TypeError, ValueError):
            reasons.append(f"invalid:{column}")

    for column in (
        "class_imbalance",
        "missing_value_fraction",
        "metric_objective_value",
        "metric_display_value",
    ):
        value = row.get(column)
        try:
            if not np.isfinite(float(value)):
                reasons.append(f"invalid:{column}")
        except (TypeError, ValueError):
            reasons.append(f"invalid:{column}")

    try:
        missing = float(row.get("missing_value_fraction"))
        if not 0.0 <= missing <= 1.0:
            reasons.append("range:missing_value_fraction")
    except (TypeError, ValueError):
        pass

    for group, vocabulary in PREPROCESSING_VOCABULARIES.items():
        if row.get(group) not in vocabulary:
            reasons.append(f"vocabulary:{group}")
    return sorted(set(reasons))


def validate_history_dataframe(
    dataframe: pd.DataFrame,
    *,
    max_invalid_fraction: float = 0.05,
    min_valid_rows: int = 0,
) -> HistoryValidationSummary:
    missing_columns = sorted(set(REQUIRED_HISTORY_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise MetaHistoryValidationError(
            f"History schema v2 is missing required columns: {missing_columns}. "
            "Run scripts/migrate_meta_history.py for legacy history."
        )

    accepted_indices: list[Any] = []
    reason_counts: dict[str, int] = {}
    for index, row in dataframe.iterrows():
        reasons = _validate_row(row)
        if reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            accepted_indices.append(index)

    rejected = len(dataframe) - len(accepted_indices)
    invalid_fraction = rejected / len(dataframe) if len(dataframe) else 0.0
    accepted = dataframe.loc[accepted_indices].copy()
    before_dedup = len(accepted)
    accepted = accepted.drop_duplicates(
        subset=list(HISTORY_DEDUPLICATION_COLUMNS), keep="last"
    ).reset_index(drop=True)
    duplicates = before_dedup - len(accepted)

    if invalid_fraction > max_invalid_fraction:
        raise MetaHistoryValidationError(
            f"Rejected {rejected}/{len(dataframe)} history rows "
            f"({invalid_fraction:.1%}), above the {max_invalid_fraction:.1%} limit. "
            f"Reasons: {reason_counts}"
        )
    if len(accepted) < min_valid_rows:
        raise MetaHistoryValidationError(
            f"History has {len(accepted)} valid rows after validation/deduplication; "
            f"at least {min_valid_rows} are required. Rejections: {reason_counts}"
        )
    return HistoryValidationSummary(
        dataframe=accepted,
        accepted_rows=len(accepted),
        rejected_rows=rejected,
        rejection_reasons=reason_counts,
        duplicate_rows=duplicates,
    )


def load_history(path: str | Path, **validation_kwargs: Any) -> HistoryValidationSummary:
    path = Path(path)
    if not path.exists():
        empty = pd.DataFrame(columns=REQUIRED_HISTORY_COLUMNS)
        return validate_history_dataframe(empty, **validation_kwargs)
    return validate_history_dataframe(pd.read_csv(path), **validation_kwargs)


def append_history_records(
    path: str | Path,
    records: Iterable[MetaHistoryRecord],
) -> HistoryValidationSummary:
    path = Path(path)
    existing = load_history(path).dataframe
    new_rows = pd.DataFrame([record.to_dict() for record in records])
    if existing.empty:
        combined = new_rows.copy()
    elif new_rows.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    summary = validate_history_dataframe(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.dataframe.to_csv(path, index=False)
    return summary
