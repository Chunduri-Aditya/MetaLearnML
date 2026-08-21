"""Shared fold evaluation and final refitting for every search implementation."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from engine.candidates import (
    CandidateResult,
    CandidateSpec,
    FoldResult,
    ResultStatus,
    derived_seed,
)
from engine.evaluator import MetricSpec
from engine.preprocessing_strategies import build_preprocessing_pipeline
from engine.splitting import FoldSplit, create_early_stopping_split
from engine.trainer import evaluate as evaluate_loader
from engine.trainer import train_model
from models.model_zoo import MODEL_REGISTRY, build_model
from models.sk_models import SkLearnWrapper
from utils.seed_utils import set_seed


@dataclass
class FittedCandidate:
    candidate: CandidateSpec
    preprocessor: Any
    model: Any
    numeric_columns: list[str]
    categorical_columns: list[str]
    input_columns: list[str]
    input_dtypes: dict[str, str]
    fitted_epochs: int | None

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        normalized = frame.copy()
        normalized.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
        missing = sorted(set(self.input_columns) - set(normalized.columns))
        if missing:
            raise ValueError(f"Prediction input is missing required columns: {missing}")
        ordered = normalized.loc[:, self.input_columns]
        if self.preprocessor is None:
            values = ordered.to_numpy()
        else:
            values = self.preprocessor.transform(ordered)
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class FinalEvaluation:
    fitted: FittedCandidate
    test_objective: float
    test_score: float
    fit_time_seconds: float
    score_time_seconds: float


def _columns(dataset_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
    types = dataset_meta["column_types"]
    numeric = [column for column, kind in types.items() if kind == "numeric"]
    categorical = [column for column, kind in types.items() if kind == "categorical"]
    return numeric, categorical


def _processed(preprocessor: Any, fit: pd.DataFrame, transform: pd.DataFrame):
    if preprocessor is None:
        return (
            fit.to_numpy(dtype=np.float32),
            transform.to_numpy(dtype=np.float32),
        )
    return (
        np.asarray(preprocessor.fit_transform(fit), dtype=np.float32),
        np.asarray(preprocessor.transform(transform), dtype=np.float32),
    )


def _target_tensor(y: np.ndarray, task_type: str) -> torch.Tensor:
    if task_type == "regression":
        return torch.from_numpy(np.asarray(y, dtype=np.float32))
    return torch.from_numpy(np.asarray(y, dtype=np.int64))


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    drop_singleton_tail = bool(shuffle and len(y) > 1 and len(y) % batch_size == 1)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), _target_tensor(y, task_type)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_singleton_tail,
    )


def _training_dict(candidate: CandidateSpec, metric: MetricSpec, task_type: str) -> dict[str, Any]:
    return {
        "lr": candidate.training.learning_rate,
        "epochs": candidate.training.max_epochs,
        "patience": candidate.training.early_stopping_patience,
        "loss_fn": torch.nn.MSELoss() if task_type == "regression" else torch.nn.CrossEntropyLoss(),
        "metric_fn": metric.callable,
        "metric_name": metric.name,
        "task_type": task_type,
    }


def _score_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    metric: MetricSpec,
    device: str,
) -> tuple[float, float]:
    loader = _loader(X, y, task_type, batch_size=256, shuffle=False)
    objective = float(evaluate_loader(model, loader, device, metric.callable, task_type=task_type))
    return objective, metric.display_from_objective(objective)


def evaluate_candidate(
    candidate: CandidateSpec,
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    dataset_meta: dict[str, Any],
    task_type: str,
    metric: MetricSpec,
    folds: Sequence[FoldSplit],
    *,
    device: str = "cpu",
    outer_seed: int = 42,
) -> CandidateResult:
    """Evaluate one candidate using only the supplied development folds."""
    numeric_columns, categorical_columns = _columns(dataset_meta)
    fold_results: list[FoldResult] = []
    out_dim = 1 if task_type == "regression" else int(dataset_meta["n_classes"])
    backend = MODEL_REGISTRY[candidate.model_identity]["backend"]

    for fold in folds:
        seed = derived_seed(outer_seed, candidate.candidate_id, fold.fold_id)
        set_seed(seed)
        fit_started = time.perf_counter()
        try:
            preprocessor = build_preprocessing_pipeline(
                candidate.preprocessing,
                numeric_columns,
                categorical_columns,
            )
            X_fold_train = X_development.iloc[fold.train_indices]
            X_fold_validation = X_development.iloc[fold.validation_indices]
            y_fold_train = y_development[fold.train_indices]
            y_fold_validation = y_development[fold.validation_indices]
            X_train, X_validation = _processed(
                preprocessor,
                X_fold_train,
                X_fold_validation,
            )
            model = build_model(candidate.model_identity, X_train.shape[1], out_dim)
            best_epoch = None
            if backend == "torch":
                fit_indices, early_indices = create_early_stopping_split(
                    y_fold_train,
                    task_type,
                    random_state=seed,
                )
                train_loader = _loader(
                    X_train[fit_indices],
                    y_fold_train[fit_indices],
                    task_type,
                    batch_size=candidate.training.batch_size,
                    shuffle=True,
                )
                early_loader = _loader(
                    X_train[early_indices],
                    y_fold_train[early_indices],
                    task_type,
                    batch_size=256,
                    shuffle=False,
                )
                outcome = train_model(
                    model,
                    train_loader,
                    early_loader,
                    device,
                    _training_dict(candidate, metric, task_type),
                )
                if outcome.best_state_dict is None:
                    raise RuntimeError("Torch training did not produce a best state")
                model.load_state_dict(outcome.best_state_dict)
                model.to(device)
                best_epoch = outcome.best_epoch
            else:
                if not isinstance(model, SkLearnWrapper):
                    raise TypeError("Sklearn registry entry did not produce a SkLearnWrapper")
                parameters = model.sklearn_model.get_params(deep=False)
                updates = {}
                if "random_state" in parameters:
                    updates["random_state"] = seed
                if "n_jobs" in parameters:
                    updates["n_jobs"] = 1
                if updates:
                    model.sklearn_model.set_params(**updates)
                model.fit(torch.from_numpy(X_train), _target_tensor(y_fold_train, task_type))
            fit_time = time.perf_counter() - fit_started
            score_started = time.perf_counter()
            objective, display = _score_model(
                model,
                X_validation,
                y_fold_validation,
                task_type,
                metric,
                device,
            )
            score_time = time.perf_counter() - score_started
            fold_results.append(
                FoldResult(
                    fold_id=fold.fold_id,
                    status=ResultStatus.SUCCESS,
                    objective_value=objective,
                    display_value=display,
                    best_epoch=best_epoch,
                    fit_time_seconds=fit_time,
                    score_time_seconds=score_time,
                )
            )
        except Exception as exc:
            fold_results.append(
                FoldResult(
                    fold_id=fold.fold_id,
                    status=ResultStatus.FAILED,
                    fit_time_seconds=time.perf_counter() - fit_started,
                    failure_type=type(exc).__name__,
                    failure_reason=str(exc)[:1000],
                )
            )

    result = CandidateResult(candidate=candidate, fold_results=fold_results)
    if result.status == ResultStatus.SUCCESS:
        objectives = [float(fold.objective_value) for fold in fold_results]
        displays = [float(fold.display_value) for fold in fold_results]
        result.selection_objective_mean = statistics.fmean(objectives)
        result.selection_objective_std = float(np.std(objectives, ddof=0))
        result.selection_cv_mean = statistics.fmean(displays)
        result.selection_cv_std = float(np.std(displays, ddof=0))
    return result


def _fit_torch_fixed_epochs(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    candidate: CandidateSpec,
    epochs: int,
    device: str,
) -> None:
    loader = _loader(
        X,
        y,
        task_type,
        batch_size=candidate.training.batch_size,
        shuffle=True,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate.training.learning_rate)
    loss_function = torch.nn.MSELoss() if task_type == "regression" else torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(features)
            if task_type == "regression":
                targets = targets.float().reshape(-1, 1)
                output = output.reshape(-1, 1)
            else:
                targets = targets.long()
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()


def fit_and_evaluate_final(
    candidate: CandidateSpec,
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    dataset_meta: dict[str, Any],
    task_type: str,
    metric: MetricSpec,
    *,
    selected_epochs: int | None,
    device: str = "cpu",
    random_state: int = 42,
) -> FinalEvaluation:
    """Refit a frozen selection on all development rows, then score test once."""
    set_seed(random_state)
    numeric_columns, categorical_columns = _columns(dataset_meta)
    fit_started = time.perf_counter()
    preprocessor = build_preprocessing_pipeline(
        candidate.preprocessing,
        numeric_columns,
        categorical_columns,
    )
    X_fit, X_final_test = _processed(preprocessor, X_development, X_test)
    out_dim = 1 if task_type == "regression" else int(dataset_meta["n_classes"])
    model = build_model(candidate.model_identity, X_fit.shape[1], out_dim)
    backend = MODEL_REGISTRY[candidate.model_identity]["backend"]
    fitted_epochs = None
    if backend == "torch":
        fitted_epochs = selected_epochs or candidate.training.max_epochs
        _fit_torch_fixed_epochs(
            model,
            X_fit,
            y_development,
            task_type,
            candidate,
            fitted_epochs,
            device,
        )
    else:
        parameters = model.sklearn_model.get_params(deep=False)
        updates = {}
        if "random_state" in parameters:
            updates["random_state"] = random_state
        if "n_jobs" in parameters:
            updates["n_jobs"] = 1
        if updates:
            model.sklearn_model.set_params(**updates)
        model.fit(torch.from_numpy(X_fit), _target_tensor(y_development, task_type))
    fit_time = time.perf_counter() - fit_started
    fitted = FittedCandidate(
        candidate=candidate,
        preprocessor=preprocessor,
        model=model,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        input_columns=list(X_development.columns),
        input_dtypes={column: str(dtype) for column, dtype in X_development.dtypes.items()},
        fitted_epochs=fitted_epochs,
    )
    score_started = time.perf_counter()
    objective, display = _score_model(
        model,
        X_final_test,
        y_test,
        task_type,
        metric,
        device,
    )
    return FinalEvaluation(
        fitted=fitted,
        test_objective=objective,
        test_score=display,
        fit_time_seconds=fit_time,
        score_time_seconds=time.perf_counter() - score_started,
    )


def selected_epoch(candidate_result: CandidateResult) -> int | None:
    epochs = [
        fold.best_epoch for fold in candidate_result.fold_results if fold.best_epoch is not None
    ]
    return int(round(statistics.median(epochs))) if epochs else None
