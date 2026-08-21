"""Unified candidate generation, ranking, evaluation, and final selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from engine.bundle import save_pipeline_bundle
from engine.candidate_evaluator import (
    evaluate_candidate,
    fit_and_evaluate_final,
    selected_epoch,
)
from engine.candidates import CandidateResult, CandidateSpec, SearchResult, TrainingConfig
from engine.evaluator import MetricSpec, get_metric_spec
from engine.meta_learner import IncompatibleMetaModelError, MetaLearner
from engine.preprocessing_strategies import (
    PreprocessingStrategy,
    generate_preprocessing_strategies,
    rank_preprocessors_by_proxy,
)
from engine.splitting import (
    EncodedTargets,
    FoldSplit,
    OuterSplit,
    create_inner_splits,
    create_outer_split,
    encode_targets,
)
from models.model_zoo import MODEL_REGISTRY
from utils.datasets import compute_dataset_metadata

OrderingMode = Literal["registry", "random", "proxy", "meta", "proxy_meta"]


@dataclass(frozen=True)
class PreparedSearchData:
    X_development: pd.DataFrame
    X_test: pd.DataFrame
    y_development: np.ndarray
    y_test: np.ndarray
    dataset_meta: dict[str, Any]
    label_encoder: Any
    outer_split: OuterSplit
    folds: tuple[FoldSplit, ...]


@dataclass(frozen=True)
class SearchOptions:
    ordering_mode: OrderingMode = "proxy_meta"
    max_candidates: int = 20
    top_k_preprocessing: int = 3
    exhaustive: bool = False
    use_proxy_ranking: bool = True
    use_meta_learning: bool = True
    random_state: int = 42
    n_jobs: int = 1
    device: str = "cpu"
    target_metric: float | None = None
    refit_on_development: bool = True
    bundle_dir: str = "saved_models/best_pipeline"
    meta_data_path: str = "experiments/meta_train.csv"
    meta_model_path: str | None = None

    def __post_init__(self):
        if self.ordering_mode not in {"registry", "random", "proxy", "meta", "proxy_meta"}:
            raise ValueError(f"Unsupported ordering_mode: {self.ordering_mode}")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.top_k_preprocessing < 1:
            raise ValueError("top_k_preprocessing must be at least 1")
        if self.ordering_mode in {"proxy", "proxy_meta"} and not self.use_proxy_ranking:
            raise ValueError(
                f"ordering_mode={self.ordering_mode!r} requires use_proxy_ranking=True"
            )
        if self.ordering_mode in {"meta", "proxy_meta"} and not self.use_meta_learning:
            raise ValueError(
                f"ordering_mode={self.ordering_mode!r} requires use_meta_learning=True"
            )


def prepare_search_data(
    X_df: pd.DataFrame,
    y: Any,
    task_type: str,
    *,
    base_meta: dict[str, Any] | None = None,
    test_ratio: float = 0.2,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    random_state: int = 42,
    target_name: str | None = None,
) -> PreparedSearchData:
    outer = create_outer_split(
        y,
        task_type,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    y_array = np.asarray(y)
    X_development = X_df.iloc[outer.development_indices].copy()
    X_test = X_df.iloc[outer.test_indices].copy()
    encoded: EncodedTargets = encode_targets(
        y_array[outer.development_indices],
        y_array[outer.test_indices],
        task_type,
    )
    metadata = compute_dataset_metadata(
        X_development,
        encoded.development,
        task_type,
        target_name=target_name,
        label_encoder=encoded.label_encoder,
        reject_text=True,
    )
    for key, value in (base_meta or {}).items():
        if key in {"dataset_id", "dataset_version", "dataset_checksum", "dataset_name"}:
            metadata[key] = value
    folds = create_inner_splits(
        encoded.development,
        task_type,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        random_state=random_state,
    )
    return PreparedSearchData(
        X_development=X_development,
        X_test=X_test,
        y_development=encoded.development,
        y_test=encoded.test,
        dataset_meta=metadata,
        label_encoder=encoded.label_encoder,
        outer_split=outer,
        folds=tuple(folds),
    )


def compatible_models(task_type: str) -> list[str]:
    models = []
    for identity, entry in sorted(MODEL_REGISTRY.items()):
        family = entry["family"]
        if task_type == "regression":
            if "regression" in family:
                models.append(identity)
        elif "supervised" in family or "classification" in family:
            models.append(identity)
    if not models:
        raise ValueError(f"No registered models support task type {task_type!r}")
    return models


def _model_training_config(
    model_identity: str,
    *,
    max_epochs: int,
    early_stopping_patience: int,
    random_state: int,
) -> TrainingConfig:
    model_config_path = Path("configs/models.yaml")
    configured: dict[str, Any] = {}
    if model_config_path.exists():
        configured = yaml.safe_load(model_config_path.read_text()) or {}
    values = configured.get(model_identity, {})
    return TrainingConfig(
        learning_rate=float(values.get("lr", 1e-3)),
        max_epochs=min(int(values.get("epochs", max_epochs)), max_epochs),
        early_stopping_patience=min(
            int(values.get("patience", early_stopping_patience)),
            early_stopping_patience,
        ),
        batch_size=int(values.get("batch_size", 32)),
        random_state=random_state,
    )


def generate_candidate_universe(
    dataset_meta: dict[str, Any],
    task_type: str,
    *,
    max_epochs: int = 20,
    early_stopping_patience: int = 5,
    random_state: int = 42,
    strategies: Sequence[PreprocessingStrategy] | None = None,
) -> list[CandidateSpec]:
    if strategies is None:
        strategies = generate_preprocessing_strategies(
            dataset_meta.get("n_numeric", 0) > 0,
            dataset_meta.get("n_categorical", 0) > 0,
            int(dataset_meta.get("n_samples", 0)),
            dataset_meta=dataset_meta,
        )
    candidates = []
    for strategy in strategies:
        strategy.validate_executable(
            has_numeric=dataset_meta.get("n_numeric", 0) > 0,
            has_categorical=dataset_meta.get("n_categorical", 0) > 0,
        )
        for model_identity in compatible_models(task_type):
            candidates.append(
                CandidateSpec.create(
                    model_identity,
                    strategy,
                    _model_training_config(
                        model_identity,
                        max_epochs=max_epochs,
                        early_stopping_patience=early_stopping_patience,
                        random_state=random_state,
                    ),
                )
            )
    return candidates


def _proxy_scores(
    candidates: Sequence[CandidateSpec],
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    task_type: str,
    metric: MetricSpec,
    random_state: int,
) -> tuple[dict[str, float], list[str]]:
    strategies = list(dict.fromkeys(candidate.preprocessing for candidate in candidates))
    try:
        ranked = rank_preprocessors_by_proxy(
            X_development,
            y_development,
            strategies,
            task_type,
            metric.callable,
            sample_size=min(512, len(X_development)),
            random_state=random_state,
        )
    except Exception:
        ranked = []
    scores = {strategy.name: float(score) for strategy, score in ranked}
    ordered = [strategy.name for strategy, _ in ranked]
    ordered.extend(sorted(strategy.name for strategy in strategies if strategy.name not in scores))
    for name in ordered:
        scores.setdefault(name, float("-inf"))
    return scores, ordered


def rank_candidate_universe(
    candidates: Sequence[CandidateSpec],
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    dataset_meta: dict[str, Any],
    task_type: str,
    metric: MetricSpec,
    options: SearchOptions,
) -> tuple[list[CandidateSpec], dict[str, float], str | None]:
    candidates = list(candidates)
    predicted: dict[str, float] = {}
    meta_reason = None
    if options.ordering_mode == "random":
        generator = np.random.default_rng(options.random_state)
        generator.shuffle(candidates)
        return candidates, predicted, meta_reason

    proxy_scores: dict[str, float] = {}
    proxy_order: list[str] = []
    if options.ordering_mode in {"proxy", "proxy_meta"}:
        proxy_scores, proxy_order = _proxy_scores(
            candidates,
            X_development,
            y_development,
            task_type,
            metric,
            options.random_state,
        )
    proxy_rank = {name: rank for rank, name in enumerate(proxy_order)}
    priority_names = set(proxy_order[: options.top_k_preprocessing])

    if options.ordering_mode in {"meta", "proxy_meta"}:
        learner = MetaLearner(
            meta_data_path=options.meta_data_path,
            task_type=task_type,
            metric_name=metric.name,
        )
        try:
            learner.load(options.meta_model_path)
        except IncompatibleMetaModelError as exc:
            meta_reason = str(exc)
        if learner.is_trained:
            for candidate, score in learner.rank_specs(dataset_meta, candidates):
                predicted[candidate.candidate_id] = score
        else:
            meta_reason = meta_reason or "No compatible trained meta-model; stable fallback used"

    if options.ordering_mode == "registry":
        return candidates, predicted, meta_reason
    if options.ordering_mode == "proxy":
        candidates.sort(
            key=lambda candidate: (
                0 if candidate.preprocessing.name in priority_names else 1,
                proxy_rank.get(candidate.preprocessing.name, len(proxy_rank)),
                candidate.model_identity,
                candidate.candidate_id,
            )
        )
    elif options.ordering_mode == "meta":
        candidates.sort(
            key=lambda candidate: (
                -predicted.get(candidate.candidate_id, float("-inf")),
                candidate.candidate_id,
            )
        )
    else:
        candidates.sort(
            key=lambda candidate: (
                0 if candidate.preprocessing.name in priority_names else 1,
                -predicted.get(candidate.candidate_id, float("-inf")),
                proxy_rank.get(candidate.preprocessing.name, len(proxy_rank)),
                candidate.candidate_id,
            )
        )
    return candidates, predicted, meta_reason


def _evaluate_ranked(
    ranked: Sequence[CandidateSpec],
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    dataset_meta: dict[str, Any],
    task_type: str,
    metric: MetricSpec,
    folds: Sequence[FoldSplit],
    options: SearchOptions,
) -> list[CandidateResult]:
    stopping_target = None if options.exhaustive else options.target_metric
    if options.n_jobs == 1 or stopping_target is not None:
        results = []
        for candidate in ranked:
            result = evaluate_candidate(
                candidate,
                X_development,
                y_development,
                dataset_meta,
                task_type,
                metric,
                folds,
                device=options.device,
                outer_seed=options.random_state,
            )
            results.append(result)
            if (
                stopping_target is not None
                and result.selection_cv_mean is not None
                and metric.target_reached(result.selection_cv_mean, stopping_target)
            ):
                break
        return results

    sklearn_candidates = [
        candidate
        for candidate in ranked
        if MODEL_REGISTRY[candidate.model_identity]["backend"] == "sklearn"
    ]
    torch_candidates = [
        candidate
        for candidate in ranked
        if MODEL_REGISTRY[candidate.model_identity]["backend"] == "torch"
    ]
    evaluated = Parallel(n_jobs=options.n_jobs, backend="loky")(
        delayed(evaluate_candidate)(
            candidate,
            X_development,
            y_development,
            dataset_meta,
            task_type,
            metric,
            folds,
            device="cpu",
            outer_seed=options.random_state,
        )
        for candidate in sklearn_candidates
    )
    evaluated.extend(
        evaluate_candidate(
            candidate,
            X_development,
            y_development,
            dataset_meta,
            task_type,
            metric,
            folds,
            device=options.device,
            outer_seed=options.random_state,
        )
        for candidate in torch_candidates
    )
    result_by_id = {result.candidate.candidate_id: result for result in evaluated}
    return [result_by_id[candidate.candidate_id] for candidate in ranked]


def run_ranked_search(
    X_development: pd.DataFrame,
    y_development: np.ndarray,
    dataset_meta: dict[str, Any],
    task_type: str,
    metric_name: str,
    folds: Sequence[FoldSplit],
    *,
    options: SearchOptions,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
    label_encoder: Any = None,
    max_epochs: int = 20,
    early_stopping_patience: int = 5,
    strategies: Sequence[PreprocessingStrategy] | None = None,
) -> SearchResult:
    metric = get_metric_spec(metric_name)
    universe = generate_candidate_universe(
        dataset_meta,
        task_type,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        random_state=options.random_state,
        strategies=strategies,
    )
    ranked, predicted, meta_reason = rank_candidate_universe(
        universe,
        X_development,
        y_development,
        dataset_meta,
        task_type,
        metric,
        options,
    )
    proposed_ranking = list(ranked)
    if not options.exhaustive:
        ranked = ranked[: options.max_candidates]
    results = _evaluate_ranked(
        ranked,
        X_development,
        y_development,
        dataset_meta,
        task_type,
        metric,
        folds,
        options,
    )
    for rank, result in enumerate(results, start=1):
        result.proposed_rank = rank
        result.predicted_score = predicted.get(result.candidate.candidate_id)
    successful = [result for result in results if result.selection_objective_mean is not None]
    best = (
        max(successful, key=lambda result: result.selection_objective_mean) if successful else None
    )
    search_result = SearchResult(
        best_candidate=best.candidate if best else None,
        candidate_results=results,
        proposed_ranking=proposed_ranking,
        metric_name=metric.name,
        predicted_scores=predicted,
        selection_cv_mean=best.selection_cv_mean if best else None,
        selection_cv_std=best.selection_cv_std if best else None,
        selection_objective_mean=best.selection_objective_mean if best else None,
        meta_ranking_reason=meta_reason,
    )
    if best and options.target_metric is not None:
        search_result.reached_target = metric.target_reached(
            float(best.selection_cv_mean), options.target_metric
        )
    if best and options.refit_on_development and X_test is not None and y_test is not None:
        final = fit_and_evaluate_final(
            best.candidate,
            X_development,
            y_development,
            X_test,
            y_test,
            dataset_meta,
            task_type,
            metric,
            selected_epochs=selected_epoch(best),
            device=options.device,
            random_state=options.random_state,
        )
        search_result.test_score = final.test_score
        search_result.test_objective = final.test_objective
        search_result.generalization_gap = metric.generalization_gap(
            float(best.selection_cv_mean), final.test_score
        )
        bundle_path = save_pipeline_bundle(
            final.fitted,
            options.bundle_dir,
            task_type=task_type,
            metric_name=metric.name,
            label_encoder=label_encoder,
            selection_cv_mean=best.selection_cv_mean,
            selection_cv_std=best.selection_cv_std,
            test_score=final.test_score,
        )
        search_result.bundle_path = str(bundle_path)
    return search_result
