"""Pinned OpenML benchmark manifest fetching and validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from utils.datasets import compute_dataset_metadata

BENCHMARK_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = BENCHMARK_ROOT / "datasets.yaml"
DEFAULT_LOCK_PATH = BENCHMARK_ROOT / "datasets.lock.json"
DEFAULT_CACHE_DIR = BENCHMARK_ROOT / "cache" / "datasets"


class ManifestValidationError(ValueError):
    """Raised when benchmark provenance is incomplete or has changed."""


@dataclass(frozen=True)
class LockedDataset:
    benchmark_id: str
    openml_task_id: int
    task_type: str
    openml_dataset_id: int
    openml_dataset_version: int
    dataset_name: str
    target_column: str
    license: str
    license_source: str
    provider_checksum: str
    local_sha256: str
    cache_file: str
    n_rows: int
    n_columns: int
    retrieved_at: str

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> LockedDataset:
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ManifestValidationError(f"Benchmark manifest does not exist: {path}")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    validate_manifest_definition(manifest)
    return manifest


def validate_manifest_definition(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ManifestValidationError("datasets.yaml must declare schema_version: 1")
    if manifest.get("provider") != "openml":
        raise ManifestValidationError("The balanced-15 profile requires provider: openml")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ManifestValidationError("datasets.yaml must contain a non-empty datasets list")
    required = {"id", "openml_task_id", "task_type"}
    supported = {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }
    ids: set[str] = set()
    task_ids: set[int] = set()
    for index, entry in enumerate(datasets):
        missing = required - set(entry)
        if missing:
            raise ManifestValidationError(
                f"Dataset entry {index} is missing fields: {sorted(missing)}"
            )
        benchmark_id = str(entry["id"]).strip()
        task_id = int(entry["openml_task_id"])
        if not benchmark_id:
            raise ManifestValidationError(f"Dataset entry {index} has an empty id")
        if benchmark_id in ids:
            raise ManifestValidationError(f"Duplicate dataset id: {benchmark_id}")
        if task_id in task_ids:
            raise ManifestValidationError(f"Duplicate OpenML task id: {task_id}")
        if entry["task_type"] not in supported:
            raise ManifestValidationError(
                f"Dataset {benchmark_id!r} has unsupported task_type {entry['task_type']!r}"
            )
        ids.add(benchmark_id)
        task_ids.add(task_id)


def load_lock(path: str | Path = DEFAULT_LOCK_PATH) -> list[LockedDataset]:
    path = Path(path)
    if not path.exists():
        raise ManifestValidationError(
            f"Pinned dataset lock does not exist: {path}. "
            "A maintainer must run benchmark fetch with --update-lock and review it."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ManifestValidationError("datasets.lock.json has an unsupported schema version")
    return [LockedDataset.from_mapping(item) for item in payload.get("datasets", [])]


def _normalize_license(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value if str(item).strip())
    license_name = str(value or "").strip()
    ambiguous = {"", "none", "unknown", "n/a", "na", "?"}
    if license_name.lower() in ambiguous:
        raise ManifestValidationError(
            "OpenML returned a missing or ambiguous dataset license; "
            "the dataset cannot enter published evidence"
        )
    return license_name


def _task_kind(task: Any) -> str:
    raw_task_type = getattr(task, "task_type_id", 0)
    task_type_id = int(getattr(raw_task_type, "value", raw_task_type))
    if task_type_id == 1:
        return "classification"
    if task_type_id == 2:
        return "regression"
    raise ManifestValidationError(
        f"OpenML task {task.task_id} has unsupported task type id {task_type_id}"
    )


def fetch_and_lock(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    update_lock: bool = False,
) -> list[LockedDataset]:
    """Fetch tasks, cache exact CSV snapshots, and create or verify the lock."""
    try:
        import openml
    except ImportError as exc:  # pragma: no cover - exercised by command-line setup
        raise RuntimeError(
            "OpenML support is not installed. Install requirements-benchmark.txt first."
        ) from exc

    manifest = load_manifest(manifest_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[LockedDataset] = []
    expected_by_id: dict[str, LockedDataset] = {}
    lock_path = Path(lock_path)
    if lock_path.exists() and not update_lock:
        expected_by_id = {item.benchmark_id: item for item in load_lock(lock_path)}

    for entry in manifest["datasets"]:
        task = openml.tasks.get_task(int(entry["openml_task_id"]), download_data=False)
        expected_kind = "regression" if entry["task_type"] == "regression" else "classification"
        actual_kind = _task_kind(task)
        if actual_kind != expected_kind:
            raise ManifestValidationError(
                f"Task {task.task_id} is {actual_kind}, expected {expected_kind}"
            )
        dataset = openml.datasets.get_dataset(
            int(task.dataset_id),
            download_data=True,
            download_qualities=False,
            download_features_meta_data=True,
        )
        target = str(task.target_name)
        X, y, _, _ = dataset.get_data(
            target=target,
            dataset_format="dataframe",
        )
        if y is None or target in X.columns:
            raise ManifestValidationError(
                f"Task {task.task_id} did not provide one unambiguous target column"
            )
        task_type = "regression" if expected_kind == "regression" else "classification"
        compute_dataset_metadata(
            X,
            y,
            task_type,
            target_name=target,
            reject_text=True,
        )
        snapshot = X.copy()
        snapshot[target] = y
        filename = f"task_{task.task_id}_dataset_{dataset.dataset_id}_v{dataset.version}.csv"
        cache_path = cache_dir / filename
        snapshot.to_csv(cache_path, index=False, lineterminator="\n")
        provider_checksum = str(
            getattr(dataset, "md5_checksum", None) or getattr(dataset, "md5", None) or ""
        ).strip()
        if not provider_checksum:
            raise ManifestValidationError(
                f"OpenML dataset {dataset.dataset_id} has no provider checksum"
            )
        record = LockedDataset(
            benchmark_id=str(entry["id"]),
            openml_task_id=int(task.task_id),
            task_type=str(entry["task_type"]),
            openml_dataset_id=int(dataset.dataset_id),
            openml_dataset_version=int(dataset.version),
            dataset_name=str(dataset.name),
            target_column=target,
            license=_normalize_license(
                entry.get("license_override", getattr(dataset, "licence", None))
            ),
            license_source=str(entry.get("license_source", "openml")),
            provider_checksum=provider_checksum,
            local_sha256=sha256_file(cache_path),
            cache_file=str(cache_path.relative_to(BENCHMARK_ROOT)),
            n_rows=int(len(snapshot)),
            n_columns=int(snapshot.shape[1]),
            retrieved_at=str(getattr(dataset, "upload_date", "unknown")),
        )
        if expected_by_id and expected_by_id.get(record.benchmark_id) != record:
            raise ManifestValidationError(
                f"Pinned provenance changed for {record.benchmark_id!r}. "
                "Use --update-lock only after reviewing the OpenML change."
            )
        records.append(record)

    validate_locked_datasets(manifest, records, cache_dir=cache_dir)
    if update_lock or not lock_path.exists():
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "provider": "openml",
            "datasets": [record.to_dict() for record in records],
        }
        lock_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return records


def validate_locked_datasets(
    manifest: dict[str, Any] | None = None,
    locked: Iterable[LockedDataset] | None = None,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> list[LockedDataset]:
    manifest = manifest or load_manifest()
    locked_list = list(locked or load_lock())
    manifest_entries = {str(item["id"]): item for item in manifest["datasets"]}
    locked_by_id = {item.benchmark_id: item for item in locked_list}
    if len(locked_by_id) != len(locked_list):
        raise ManifestValidationError("datasets.lock.json contains duplicate benchmark ids")
    if set(locked_by_id) != set(manifest_entries):
        missing = sorted(set(manifest_entries) - set(locked_by_id))
        extra = sorted(set(locked_by_id) - set(manifest_entries))
        raise ManifestValidationError(
            f"Manifest/lock dataset mismatch; missing={missing}, extra={extra}"
        )
    cache_dir = Path(cache_dir)
    for benchmark_id, record in locked_by_id.items():
        definition = manifest_entries[benchmark_id]
        if int(definition["openml_task_id"]) != record.openml_task_id:
            raise ManifestValidationError(f"Task id mismatch for {benchmark_id}")
        if str(definition["task_type"]) != record.task_type:
            raise ManifestValidationError(f"Task type mismatch for {benchmark_id}")
        _normalize_license(record.license)
        if not record.license_source.strip():
            raise ManifestValidationError(f"Dataset {benchmark_id} has no license source")
        if not record.target_column.strip() or not record.provider_checksum.strip():
            raise ManifestValidationError(
                f"Dataset {benchmark_id} has incomplete target/checksum provenance"
            )
        cache_path = BENCHMARK_ROOT / record.cache_file
        if not cache_path.exists():
            raise ManifestValidationError(
                f"Cached snapshot is missing for {benchmark_id}: {cache_path}"
            )
        if (
            cache_path.parent != cache_dir.resolve()
            and cache_path.parent.resolve() != cache_dir.resolve()
        ):
            raise ManifestValidationError(
                f"Locked cache file for {benchmark_id} is outside {cache_dir}"
            )
        actual_hash = sha256_file(cache_path)
        if actual_hash != record.local_sha256:
            raise ManifestValidationError(
                f"Checksum mismatch for {benchmark_id}: {actual_hash} != {record.local_sha256}"
            )
        frame = pd.read_csv(cache_path)
        if record.target_column not in frame.columns:
            raise ManifestValidationError(
                f"Cached snapshot {benchmark_id} has no target {record.target_column!r}"
            )
    return locked_list


def load_locked_frame(record: LockedDataset) -> tuple[pd.DataFrame, pd.Series]:
    cache_path = BENCHMARK_ROOT / record.cache_file
    frame = pd.read_csv(cache_path)
    if record.target_column not in frame.columns:
        raise ManifestValidationError(
            f"Cached dataset {record.benchmark_id} is missing target {record.target_column!r}"
        )
    y = frame.pop(record.target_column)
    return frame, y
