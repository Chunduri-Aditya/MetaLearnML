"""Schema-checked meta-learner for candidate performance prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from engine.candidates import CandidateSpec
from engine.meta_features import (
    META_FEATURE_SCHEMA_VERSION,
    PREPROCESSING_VOCABULARIES,
    FeatureSchema,
    build_feature_schema,
    encode_meta_features,
)
from engine.meta_history import (
    MetaHistoryValidationError,
    validate_history_dataframe,
)
from engine.meta_history import (
    load_history as load_validated_history,
)


class IncompatibleMetaModelError(RuntimeError):
    """Raised when a persisted meta-model does not match the live feature contract."""


class MetaLearner:
    """Predict candidate objective values from prior, validated experiments."""

    def __init__(
        self,
        meta_data_path: str = "experiments/meta_train.csv",
        task_type: str | None = None,
        metric_name: str | None = None,
    ):
        self.meta_data_path = Path(meta_data_path)
        self.task_type = task_type
        self.metric_name = metric_name
        self.model: RandomForestRegressor | None = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.schema: FeatureSchema = build_feature_schema()
        self.feature_names = list(self.schema.feature_names)
        self.min_history_size = 20
        self.last_history_summary: dict[str, Any] = {}

    def load_history(self) -> pd.DataFrame:
        summary = load_validated_history(self.meta_data_path)
        self.last_history_summary = {
            "accepted_rows": summary.accepted_rows,
            "rejected_rows": summary.rejected_rows,
            "rejection_reasons": summary.rejection_reasons,
            "duplicate_rows": summary.duplicate_rows,
        }
        if summary.rejected_rows or summary.duplicate_rows:
            print(f"[WARN] Meta-history validation summary: {self.last_history_summary}")
        return summary.dataframe

    def _get_model_path(self) -> Path:
        base = Path("experiments")
        if self.task_type and self.metric_name:
            return base / f"meta_model_{self.task_type}_{self.metric_name}.pkl"
        return base / "meta_model.pkl"

    def extract_features(
        self,
        dataset_meta: Mapping[str, Any],
        model_name: str,
        preprocessing: Any,
        *,
        model_family: str | None = None,
    ) -> np.ndarray:
        if isinstance(preprocessing, str):
            raise TypeError(
                "MetaLearner requires a PreprocessingStrategy or component mapping; "
                "display-name parsing is supported only by the history migration command"
            )
        encoded = encode_meta_features(
            dataset_meta,
            model_name,
            preprocessing,
            model_family=model_family,
            schema=self.schema,
        )
        self.feature_names = list(encoded.names)
        return encoded.values

    def train(self, history_df: pd.DataFrame | None = None) -> None:
        if history_df is None:
            history_df = self.load_history()
        else:
            summary = validate_history_dataframe(
                history_df,
                min_valid_rows=0,
            )
            history_df = summary.dataframe
            self.last_history_summary = {
                "accepted_rows": summary.accepted_rows,
                "rejected_rows": summary.rejected_rows,
                "rejection_reasons": summary.rejection_reasons,
                "duplicate_rows": summary.duplicate_rows,
            }

        if self.task_type:
            history_df = history_df[history_df["task_type"] == self.task_type]
        if self.metric_name:
            history_df = history_df[history_df["metric_name"] == self.metric_name]
        if len(history_df) < self.min_history_size:
            self.is_trained = False
            raise MetaHistoryValidationError(
                "Not enough compatible validated history for meta-learning: "
                f"need at least {self.min_history_size} rows after task/metric filtering, "
                f"got {len(history_df)}"
            )

        features: list[np.ndarray] = []
        targets: list[float] = []
        for _, row in history_df.iterrows():
            dataset_meta = {
                "n_samples": row["n_samples"],
                "n_features": row["n_features"],
                "n_numeric": row["n_numeric"],
                "n_categorical": row["n_categorical"],
                "n_text": row["n_text"],
                "n_classes": row["n_classes"],
                "class_imbalance": row["class_imbalance"],
                "missing_value_fraction": row["missing_value_fraction"],
            }
            preprocessing = {
                "numeric_scaler": row["numeric_scaler"],
                "numeric_imputer": row["numeric_imputer"],
                "categorical_encoder": row["categorical_encoder"],
                "categorical_imputer": row["categorical_imputer"],
            }
            features.append(
                self.extract_features(
                    dataset_meta,
                    str(row["model_identity"]),
                    preprocessing,
                    model_family=str(row["model_family"]),
                )
            )
            targets.append(float(row["metric_objective_value"]))

        matrix = np.asarray(features, dtype=np.float32)
        target = np.asarray(targets, dtype=np.float64)
        scaled = self.scaler.fit_transform(matrix)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(scaled, target)
        self.is_trained = True
        print(f"[INFO] Meta-model trained on {len(matrix)} validated experiments")

    def predict(
        self,
        dataset_meta: Mapping[str, Any],
        model_name: str,
        preprocessing: Any,
        *,
        model_family: str | None = None,
    ) -> float:
        if not self.is_trained or self.model is None:
            return 0.0
        features = self.extract_features(
            dataset_meta,
            model_name,
            preprocessing,
            model_family=model_family,
        )
        scaled = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(scaled)[0])

    def rank_specs(
        self,
        dataset_meta: Mapping[str, Any],
        candidates: Sequence[CandidateSpec],
    ) -> list[tuple[CandidateSpec, float]]:
        if not self.is_trained:
            return [(candidate, 0.0) for candidate in candidates]
        ranked = [
            (
                candidate,
                self.predict(
                    dataset_meta,
                    candidate.model_identity,
                    candidate.preprocessing,
                    model_family=candidate.model_family,
                ),
            )
            for candidate in candidates
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].candidate_id))
        return ranked

    def rank_candidates(
        self,
        dataset_meta: Mapping[str, Any],
        candidates: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, Any, float]]:
        scored = [
            (model, preprocessing, self.predict(dataset_meta, model, preprocessing))
            for model, preprocessing in candidates
        ]
        scored.sort(key=lambda item: (-item[2], item[0], item[1].name))
        return scored

    def save(self, path: str | None = None) -> None:
        if not self.is_trained or self.model is None:
            return
        artifact_path = Path(path) if path is not None else self._get_model_path()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artifact_type": "MetaLearnMLMetaModel",
                "meta_feature_schema_version": META_FEATURE_SCHEMA_VERSION,
                "feature_schema": self.schema.to_dict(),
                "feature_names": list(self.schema.feature_names),
                "registry_snapshot": [list(item) for item in self.schema.registry_snapshot],
                "preprocessing_vocabularies": {
                    name: list(values) for name, values in PREPROCESSING_VOCABULARIES.items()
                },
                "model": self.model,
                "scaler": self.scaler,
                "task_type": self.task_type,
                "metric_name": self.metric_name,
            },
            artifact_path,
        )
        print(f"[INFO] Meta-model saved to {artifact_path}")

    def load(self, path: str | None = None) -> None:
        artifact_path = Path(path) if path is not None else self._get_model_path()
        if not artifact_path.exists():
            self.is_trained = False
            return
        data = joblib.load(artifact_path)
        saved_schema = data.get("feature_schema", {})
        version = data.get("meta_feature_schema_version")
        incompatibilities = []
        if version != META_FEATURE_SCHEMA_VERSION:
            incompatibilities.append(f"schema version {version!r} != {META_FEATURE_SCHEMA_VERSION}")
        if saved_schema.get("schema_hash") != self.schema.schema_hash:
            incompatibilities.append("feature schema hash differs from the live registry/schema")
        if tuple(data.get("feature_names", ())) != self.schema.feature_names:
            incompatibilities.append("ordered feature names differ")
        saved_vocabularies = data.get("preprocessing_vocabularies", {})
        live_vocabularies = {
            name: list(values) for name, values in PREPROCESSING_VOCABULARIES.items()
        }
        if saved_vocabularies != live_vocabularies:
            incompatibilities.append("preprocessing vocabularies differ")
        if self.task_type and data.get("task_type") != self.task_type:
            incompatibilities.append(
                f"task {data.get('task_type')!r} != requested {self.task_type!r}"
            )
        if self.metric_name and data.get("metric_name") != self.metric_name:
            incompatibilities.append(
                f"metric {data.get('metric_name')!r} != requested {self.metric_name!r}"
            )
        if incompatibilities:
            self.is_trained = False
            raise IncompatibleMetaModelError(
                f"Cannot load meta-model {artifact_path}: "
                + "; ".join(incompatibilities)
                + ". Retrain it from validated schema-v2 history."
            )
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.task_type = data.get("task_type", self.task_type)
        self.metric_name = data.get("metric_name", self.metric_name)
        self.feature_names = list(self.schema.feature_names)
        self.is_trained = True
        print(f"[INFO] Meta-model loaded from {artifact_path}")

    def rank_candidates_with_fallback(
        self,
        dataset_meta: Mapping[str, Any],
        candidates: Sequence[tuple[str, Any]],
        min_history_size: int = 20,
    ) -> list[tuple[str, Any, float]]:
        try:
            history_size = len(self.load_history())
        except MetaHistoryValidationError:
            history_size = 0
        if not self.is_trained or history_size < min_history_size:
            return self._rank_with_diversity_fallback(candidates)
        return self.rank_candidates(dataset_meta, candidates)

    @staticmethod
    def _rank_with_diversity_fallback(
        candidates: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, Any, float]]:
        from models.model_zoo import MODEL_REGISTRY

        families_seen: set[str] = set()
        first: list[tuple[str, Any, float]] = []
        remaining: list[tuple[str, Any, float]] = []
        for model_name, preprocessing in candidates:
            family = MODEL_REGISTRY.get(model_name, {}).get("family", "unknown")
            target = first if family not in families_seen else remaining
            target.append((model_name, preprocessing, 0.0))
            families_seen.add(family)
        return first + remaining
