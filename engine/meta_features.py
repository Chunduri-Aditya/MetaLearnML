"""Canonical, versioned meta-feature encoding for datasets and candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

META_FEATURE_SCHEMA_VERSION = 2

DATASET_FEATURE_NAMES = (
    "n_samples",
    "n_features",
    "n_numeric",
    "n_categorical",
    "n_text",
    "n_classes",
    "class_imbalance",
    "missing_value_fraction",
    "log_n_samples",
    "log_n_features",
    "feature_to_sample_ratio",
    "text_fraction",
    "categorical_fraction",
)

PREPROCESSING_VOCABULARIES = {
    "numeric_scaler": ("standard", "minmax", "robust", "power", "none", "unknown"),
    "numeric_imputer": ("mean", "median", "knn", "none", "unknown"),
    "categorical_encoder": ("onehot", "ordinal", "none", "unknown"),
    "categorical_imputer": ("most_frequent", "constant", "none", "unknown"),
}


@dataclass(frozen=True)
class FeatureSchema:
    """Complete ordered feature contract used by a fitted meta-model."""

    version: int
    feature_names: tuple[str, ...]
    model_families: tuple[str, ...]
    model_identities: tuple[str, ...]
    registry_snapshot: tuple[tuple[str, str, str], ...]
    schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "model_families": list(self.model_families),
            "model_identities": list(self.model_identities),
            "registry_snapshot": [list(item) for item in self.registry_snapshot],
            "schema_hash": self.schema_hash,
        }


@dataclass(frozen=True)
class EncodedMetaFeatures:
    values: np.ndarray
    names: tuple[str, ...]
    schema_hash: str


def _live_registry() -> Mapping[str, Mapping[str, Any]]:
    from models.model_zoo import MODEL_REGISTRY

    return MODEL_REGISTRY


def build_feature_schema(
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> FeatureSchema:
    """Build a deterministic schema from the current model registry."""
    registry = registry or _live_registry()
    snapshot_items: list[tuple[str, str, str]] = []
    for key, entry in registry.items():
        identity = str(entry.get("identity", ""))
        family = str(entry.get("family", ""))
        backend = str(entry.get("backend", ""))
        if not identity or not family or not backend:
            raise ValueError(
                f"MODEL_REGISTRY entry {key!r} must declare identity, family, and backend"
            )
        if identity != key:
            raise ValueError(
                f"MODEL_REGISTRY key {key!r} must equal its stable identity {identity!r}"
            )
        snapshot_items.append((identity, family, backend))

    snapshot = tuple(sorted(snapshot_items))
    identities = tuple(sorted(identity for identity, _, _ in snapshot)) + ("unknown",)
    families = tuple(sorted({family for _, family, _ in snapshot})) + ("unknown",)

    names = list(DATASET_FEATURE_NAMES)
    names.extend(f"model_family_{family}" for family in families)
    names.extend(f"model_identity_{identity}" for identity in identities)
    for group, vocabulary in PREPROCESSING_VOCABULARIES.items():
        names.extend(f"preprocessing_{group}_{value}" for value in vocabulary)

    payload = {
        "version": META_FEATURE_SCHEMA_VERSION,
        "feature_names": names,
        "registry_snapshot": snapshot,
        "preprocessing_vocabularies": PREPROCESSING_VOCABULARIES,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FeatureSchema(
        version=META_FEATURE_SCHEMA_VERSION,
        feature_names=tuple(names),
        model_families=families,
        model_identities=identities,
        registry_snapshot=snapshot,
        schema_hash=digest,
    )


def _finite_number(meta: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    raw = meta.get(key, default)
    if raw is None:
        raw = default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Dataset meta-feature {key!r} must be numeric, got {raw!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Dataset meta-feature {key!r} must be finite, got {raw!r}")
    return value


def preprocessing_components(strategy: Any) -> dict[str, str]:
    """Extract and normalize structured preprocessing components."""
    if isinstance(strategy, Mapping):
        source = strategy
        get_value = source.get
    else:

        def get_value(name: str, default: Any = None) -> Any:
            return getattr(strategy, name, default)

    components: dict[str, str] = {}
    for group, vocabulary in PREPROCESSING_VOCABULARIES.items():
        raw = get_value(group, "unknown")
        value = str(raw) if raw is not None else "none"
        components[group] = value if value in vocabulary else "unknown"
    return components


def encode_meta_features(
    dataset_meta: Mapping[str, Any],
    model_identity: str,
    strategy: Any,
    *,
    model_family: str | None = None,
    schema: FeatureSchema | None = None,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> EncodedMetaFeatures:
    """Encode a dataset and structured candidate using the canonical schema."""
    registry = registry or _live_registry()
    schema = schema or build_feature_schema(registry)

    n_samples = _finite_number(dataset_meta, "n_samples")
    n_features = _finite_number(dataset_meta, "n_features")
    n_numeric = _finite_number(dataset_meta, "n_numeric")
    n_categorical = _finite_number(dataset_meta, "n_categorical")
    n_text = _finite_number(dataset_meta, "n_text")
    n_classes = _finite_number(dataset_meta, "n_classes")
    class_imbalance = _finite_number(dataset_meta, "class_imbalance")
    missing_fraction = _finite_number(dataset_meta, "missing_value_fraction")
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError(
            "Dataset meta-feature 'missing_value_fraction' must be within [0, 1], "
            f"got {missing_fraction}"
        )

    values: list[float] = [
        n_samples,
        n_features,
        n_numeric,
        n_categorical,
        n_text,
        n_classes,
        class_imbalance,
        missing_fraction,
        float(np.log10(n_samples + 1.0)),
        float(np.log10(n_features + 1.0)),
        n_features / n_samples if n_samples > 0 else 0.0,
        n_text / n_features if n_features > 0 else 0.0,
        n_categorical / n_features if n_features > 0 else 0.0,
    ]

    entry = registry.get(model_identity, {})
    resolved_identity = model_identity if model_identity in schema.model_identities else "unknown"
    resolved_family = model_family or entry.get("family", "unknown")
    if resolved_family not in schema.model_families:
        resolved_family = "unknown"

    values.extend(float(family == resolved_family) for family in schema.model_families)
    values.extend(float(identity == resolved_identity) for identity in schema.model_identities)

    components = preprocessing_components(strategy)
    for group, vocabulary in PREPROCESSING_VOCABULARIES.items():
        selected = components[group]
        values.extend(float(value == selected) for value in vocabulary)

    array = np.asarray(values, dtype=np.float32)
    if array.shape != (len(schema.feature_names),):
        raise RuntimeError(
            f"Meta-feature encoder produced {array.shape[0]} values for "
            f"{len(schema.feature_names)} feature names"
        )
    return EncodedMetaFeatures(array, schema.feature_names, schema.schema_hash)
