from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.candidates import CandidateSpec, TrainingConfig
from engine.meta_history import MetaHistoryRecord
from engine.preprocessing_strategies import PreprocessingStrategy


@pytest.fixture
def numeric_regression() -> tuple[pd.DataFrame, np.ndarray]:
    generator = np.random.default_rng(7)
    size = 60
    first = generator.normal(size=size)
    second = generator.normal(size=size)
    frame = pd.DataFrame({"first": first, "second": second})
    frame.loc[[2, 9, 21], "second"] = np.nan
    target = 2.0 * first - np.nan_to_num(second) + generator.normal(scale=0.1, size=size)
    return frame, target.astype(np.float32)


@pytest.fixture
def mixed_binary() -> tuple[pd.DataFrame, np.ndarray]:
    generator = np.random.default_rng(11)
    size = 100
    number = generator.normal(size=size)
    group = np.where(number > 0, "positive", "negative").astype(object)
    group[::17] = None
    frame = pd.DataFrame({"number": number, "group": group})
    target = np.asarray([0] * 70 + [1] * 30, dtype=np.int64)
    generator.shuffle(target)
    return frame, target


@pytest.fixture
def multiclass_data() -> tuple[pd.DataFrame, np.ndarray]:
    generator = np.random.default_rng(19)
    frame = pd.DataFrame(generator.normal(size=(90, 4)), columns=list("abcd"))
    target = np.repeat(np.arange(3), 30).astype(np.int64)
    return frame, target


@pytest.fixture
def standard_strategy() -> PreprocessingStrategy:
    return PreprocessingStrategy.from_components(
        numeric_scaler="standard",
        numeric_imputer="median",
        categorical_encoder="onehot",
        categorical_imputer="most_frequent",
    )


@pytest.fixture
def numeric_strategy() -> PreprocessingStrategy:
    return PreprocessingStrategy.from_components(
        numeric_scaler="standard",
        numeric_imputer="median",
        categorical_encoder="none",
        categorical_imputer="none",
    )


@pytest.fixture
def classification_candidate(numeric_strategy) -> CandidateSpec:
    return CandidateSpec.create(
        "logreg",
        numeric_strategy,
        TrainingConfig(max_epochs=1, early_stopping_patience=1, random_state=42),
    )


def make_history_record(index: int, *, missing_fraction: float = 0.2) -> MetaHistoryRecord:
    strategy = PreprocessingStrategy.from_components(
        numeric_scaler="standard",
        numeric_imputer="median",
        categorical_encoder="none",
        categorical_imputer="none",
    )
    candidate = CandidateSpec.create(
        "logreg",
        strategy,
        TrainingConfig(random_state=index),
    )
    return MetaHistoryRecord.create(
        run_id=f"run-{index}",
        dataset_id=f"dataset-{index % 3}",
        dataset_version="1",
        dataset_checksum=f"checksum-{index % 3}",
        dataset_name=f"Dataset {index % 3}",
        task_type="classification",
        n_samples=100 + index,
        n_features=4,
        n_numeric=4,
        n_categorical=0,
        n_text=0,
        n_classes=2,
        class_imbalance=0.1,
        missing_value_fraction=missing_fraction,
        candidate_id=candidate.candidate_id,
        model_identity=candidate.model_identity,
        model_family=candidate.model_family,
        preprocessing_name=strategy.name,
        **strategy.to_component_dict(),
        metric_name="f1_macro",
        metric_objective_value=0.5 + index / 100,
        metric_display_value=0.5 + index / 100,
        outer_split_seed=42,
        inner_fold_id=index,
        repository_commit="test-commit",
        environment_hash="test-environment",
    )


@pytest.fixture
def history_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(make_history_record(index)) for index in range(24)])


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "number": np.arange(30, dtype=float),
            "category": ["a", "b", "c"] * 10,
            "label": [0, 1] * 15,
        }
    ).to_csv(path, index=False)
    return path
