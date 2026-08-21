"""Typed candidate and search-result contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from engine.preprocessing_strategies import PreprocessingStrategy
from models.model_zoo import MODEL_REGISTRY


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    max_epochs: int = 20
    early_stopping_patience: int | None = 5
    batch_size: int = 32
    random_state: int = 42

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> TrainingConfig:
        values = values or {}
        return cls(
            learning_rate=float(values.get("learning_rate", values.get("lr", 1e-3))),
            max_epochs=int(values.get("max_epochs", values.get("epochs", 20))),
            early_stopping_patience=values.get(
                "early_stopping_patience", values.get("patience", 5)
            ),
            batch_size=int(values.get("batch_size", 32)),
            random_state=int(values.get("random_state", 42)),
        )


@dataclass(frozen=True)
class CandidateSpec:
    model_identity: str
    model_family: str
    preprocessing: PreprocessingStrategy
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        entry = MODEL_REGISTRY.get(self.model_identity)
        if entry is None:
            raise ValueError(f"Unknown model identity: {self.model_identity}")
        if entry["family"] != self.model_family:
            raise ValueError(
                f"Model {self.model_identity!r} belongs to {entry['family']!r}, "
                f"not {self.model_family!r}"
            )

    @classmethod
    def create(
        cls,
        model_identity: str,
        preprocessing: PreprocessingStrategy,
        training: TrainingConfig | None = None,
    ) -> CandidateSpec:
        entry = MODEL_REGISTRY.get(model_identity)
        if entry is None:
            raise ValueError(f"Unknown model identity: {model_identity}")
        return cls(
            model_identity=model_identity,
            model_family=entry["family"],
            preprocessing=preprocessing,
            training=training or TrainingConfig(),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "candidate_schema_version": 2,
            "model_identity": self.model_identity,
            "model_family": self.model_family,
            "preprocessing": self.preprocessing.to_component_dict(),
            "training": asdict(self.training),
        }

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    @property
    def display_id(self) -> str:
        return self.candidate_id[:12]


class ResultStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    status: ResultStatus
    objective_value: float | None = None
    display_value: float | None = None
    best_epoch: int | None = None
    fit_time_seconds: float = 0.0
    score_time_seconds: float = 0.0
    failure_type: str | None = None
    failure_reason: str | None = None


@dataclass
class CandidateResult:
    candidate: CandidateSpec
    fold_results: list[FoldResult]
    selection_objective_mean: float | None = None
    selection_objective_std: float | None = None
    selection_cv_mean: float | None = None
    selection_cv_std: float | None = None
    predicted_score: float | None = None
    proposed_rank: int | None = None

    @property
    def status(self) -> ResultStatus:
        if self.fold_results and all(
            fold.status == ResultStatus.SUCCESS for fold in self.fold_results
        ):
            return ResultStatus.SUCCESS
        return ResultStatus.FAILED


@dataclass
class SearchResult:
    best_candidate: CandidateSpec | None
    candidate_results: list[CandidateResult]
    proposed_ranking: list[CandidateSpec]
    metric_name: str
    predicted_scores: dict[str, float] = field(default_factory=dict)
    selection_cv_mean: float | None = None
    selection_cv_std: float | None = None
    selection_objective_mean: float | None = None
    test_score: float | None = None
    test_objective: float | None = None
    generalization_gap: float | None = None
    bundle_path: str | None = None
    reached_target: bool = False
    meta_ranking_reason: str | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        best_info = None
        if self.best_candidate is not None:
            best_info = {
                "candidate_id": self.best_candidate.candidate_id,
                "model_name": self.best_candidate.model_identity,
                "preprocessing_name": self.best_candidate.preprocessing.name,
                "metric": self.selection_cv_mean,
                "metric_name": self.metric_name,
                "selection_cv_mean": self.selection_cv_mean,
                "selection_cv_std": self.selection_cv_std,
                "test_score": self.test_score,
                "generalization_gap": self.generalization_gap,
                "bundle_path": self.bundle_path,
            }
        tried = []
        for result in self.candidate_results:
            if result.selection_cv_mean is not None:
                tried.append(
                    (
                        result.candidate.model_identity,
                        result.candidate.preprocessing.name,
                        result.selection_cv_mean,
                    )
                )
        return {
            "best_info": best_info,
            "tried": tried,
            "reached_target": self.reached_target,
            "search_result": self,
        }


def derived_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
