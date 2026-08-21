"""Balanced-15 candidate measurement and leakage-resistant ranking replay."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from benchmarks.manifest import LockedDataset
from engine.candidate_evaluator import (
    evaluate_candidate,
    fit_and_evaluate_final,
    selected_epoch,
)
from engine.candidates import CandidateResult, CandidateSpec, ResultStatus, derived_seed
from engine.evaluator import MetricSpec, get_metric_spec
from engine.meta_history import (
    META_HISTORY_SCHEMA_VERSION,
    MetaHistoryRecord,
    environment_fingerprint,
    repository_commit,
)
from engine.meta_learner import MetaLearner
from engine.preprocessing_strategies import PreprocessingStrategy
from engine.search import (
    PreparedSearchData,
    _proxy_scores,
    generate_candidate_universe,
    prepare_search_data,
)
from models.model_zoo import MODEL_REGISTRY

RAW_RESULT_SCHEMA_VERSION = 1
METHOD_RESULT_SCHEMA_VERSION = 1


@dataclass
class MeasuredRun:
    dataset: LockedDataset
    prepared: PreparedSearchData
    metric: MetricSpec
    candidates: list[CandidateSpec]
    results: list[CandidateResult]
    outer_seed: int

    @property
    def result_by_id(self) -> dict[str, CandidateResult]:
        return {result.candidate.candidate_id: result for result in self.results}


def benchmark_task(record: LockedDataset) -> str:
    return "regression" if record.task_type == "regression" else "classification"


def benchmark_metric(record: LockedDataset) -> str:
    return "r2" if record.task_type == "regression" else "f1_macro"


def fixed_strategies(dataset_meta: dict[str, Any]) -> list[PreprocessingStrategy]:
    """Return the four published strategies, explicitly marking absent groups."""
    has_numeric = int(dataset_meta.get("n_numeric", 0)) > 0
    has_categorical = int(dataset_meta.get("n_categorical", 0)) > 0
    definitions = (
        ("standard", "median", "onehot", "most_frequent"),
        ("minmax", "mean", "ordinal", "constant"),
        ("robust", "median", "onehot", "constant"),
        ("power", "knn", "ordinal", "most_frequent"),
    )
    return [
        PreprocessingStrategy.from_components(
            numeric_scaler=scaler if has_numeric else "none",
            numeric_imputer=imputer if has_numeric else "none",
            categorical_encoder=encoder if has_categorical else "none",
            categorical_imputer=cat_imputer if has_categorical else "none",
        )
        for scaler, imputer, encoder, cat_imputer in definitions
    ]


def expected_candidate_count(task_type: str) -> int:
    return 36 if task_type == "regression" else 28


def candidate_universe_hash(candidates: Sequence[CandidateSpec]) -> str:
    import hashlib

    payload = "\n".join(candidate.candidate_id for candidate in candidates).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def measure_dataset_split(
    record: LockedDataset,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_seed: int,
    test_ratio: float,
    cv_folds: int,
    cv_repeats: int,
    max_epochs: int,
    patience: int,
    device: str = "cpu",
    n_jobs: int = 1,
) -> MeasuredRun:
    """Measure every candidate once on development-only folds."""
    task_type = benchmark_task(record)
    metric = get_metric_spec(benchmark_metric(record))
    base_meta = {
        "dataset_id": record.benchmark_id,
        "dataset_name": record.dataset_name,
        "dataset_version": str(record.openml_dataset_version),
        "dataset_checksum": record.local_sha256,
    }
    prepared = prepare_search_data(
        X,
        y,
        task_type,
        base_meta=base_meta,
        test_ratio=test_ratio,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        random_state=outer_seed,
        target_name=record.target_column,
    )
    strategies = fixed_strategies(prepared.dataset_meta)
    candidates = generate_candidate_universe(
        prepared.dataset_meta,
        task_type,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        random_state=outer_seed,
        strategies=strategies,
    )
    expected = expected_candidate_count(task_type)
    if len(candidates) != expected:
        raise RuntimeError(
            f"{record.benchmark_id} produced {len(candidates)} candidates; expected {expected}"
        )
    if n_jobs == 1:
        results = [
            evaluate_candidate(
                candidate,
                prepared.X_development,
                prepared.y_development,
                prepared.dataset_meta,
                task_type,
                metric,
                prepared.folds,
                device=device,
                outer_seed=outer_seed,
            )
            for candidate in candidates
        ]
    else:
        sklearn_candidates = [
            candidate
            for candidate in candidates
            if MODEL_REGISTRY[candidate.model_identity]["backend"] == "sklearn"
        ]
        torch_candidates = [
            candidate
            for candidate in candidates
            if MODEL_REGISTRY[candidate.model_identity]["backend"] == "torch"
        ]
        evaluated = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_candidate)(
                candidate,
                prepared.X_development,
                prepared.y_development,
                prepared.dataset_meta,
                task_type,
                metric,
                prepared.folds,
                device="cpu",
                outer_seed=outer_seed,
            )
            for candidate in sklearn_candidates
        )
        evaluated.extend(
            evaluate_candidate(
                candidate,
                prepared.X_development,
                prepared.y_development,
                prepared.dataset_meta,
                task_type,
                metric,
                prepared.folds,
                device=device,
                outer_seed=outer_seed,
            )
            for candidate in torch_candidates
        )
        by_id = {result.candidate.candidate_id: result for result in evaluated}
        results = [by_id[candidate.candidate_id] for candidate in candidates]
    if not any(result.status == ResultStatus.SUCCESS for result in results):
        raise RuntimeError(
            f"All candidates failed for {record.benchmark_id}, outer seed {outer_seed}"
        )
    return MeasuredRun(record, prepared, metric, candidates, results, outer_seed)


def history_records_for_run(run: MeasuredRun) -> list[MetaHistoryRecord]:
    environment_hash, _ = environment_fingerprint()
    commit = repository_commit()
    records: list[MetaHistoryRecord] = []
    meta = run.prepared.dataset_meta
    for result in run.results:
        candidate = result.candidate
        components = candidate.preprocessing.to_component_dict()
        for fold in result.fold_results:
            if fold.status != ResultStatus.SUCCESS:
                continue
            records.append(
                MetaHistoryRecord.create(
                    run_id=(
                        f"balanced-15:{run.dataset.benchmark_id}:"
                        f"{run.outer_seed}:{fold.fold_id}:{candidate.display_id}"
                    ),
                    dataset_id=run.dataset.benchmark_id,
                    dataset_version=str(run.dataset.openml_dataset_version),
                    dataset_checksum=run.dataset.local_sha256,
                    dataset_name=run.dataset.dataset_name,
                    task_type=benchmark_task(run.dataset),
                    n_samples=int(meta["n_samples"]),
                    n_features=int(meta["n_features"]),
                    n_numeric=int(meta["n_numeric"]),
                    n_categorical=int(meta["n_categorical"]),
                    n_text=int(meta["n_text"]),
                    n_classes=int(meta.get("n_classes") or 0),
                    class_imbalance=float(meta.get("class_imbalance") or 0.0),
                    missing_value_fraction=float(meta["missing_value_fraction"]),
                    candidate_id=candidate.candidate_id,
                    model_identity=candidate.model_identity,
                    model_family=candidate.model_family,
                    preprocessing_name=candidate.preprocessing.name,
                    **components,
                    metric_name=run.metric.name,
                    metric_objective_value=float(fold.objective_value),
                    metric_display_value=float(fold.display_value),
                    outer_split_seed=run.outer_seed,
                    inner_fold_id=fold.fold_id,
                    repository_commit=commit,
                    environment_hash=environment_hash,
                )
            )
    return records


def candidate_rows(run: MeasuredRun) -> list[dict[str, Any]]:
    environment_hash, environment = environment_fingerprint()
    commit = repository_commit()
    universe_hash = candidate_universe_hash(run.candidates)
    rows: list[dict[str, Any]] = []
    for result in run.results:
        candidate = result.candidate
        components = candidate.preprocessing.to_component_dict()
        for fold in result.fold_results:
            rows.append(
                {
                    "schema_version": RAW_RESULT_SCHEMA_VERSION,
                    "profile": "balanced-15-v1",
                    "dataset_id": run.dataset.benchmark_id,
                    "openml_task_id": run.dataset.openml_task_id,
                    "dataset_version": run.dataset.openml_dataset_version,
                    "dataset_checksum": run.dataset.local_sha256,
                    "task_type": benchmark_task(run.dataset),
                    "outer_seed": run.outer_seed,
                    "inner_fold_id": fold.fold_id,
                    "candidate_fold_seed": derived_seed(
                        run.outer_seed,
                        candidate.candidate_id,
                        fold.fold_id,
                    ),
                    "method": "exhaustive_cache",
                    "candidate_budget": len(run.candidates),
                    "candidate_universe_hash": universe_hash,
                    "candidate_id": candidate.candidate_id,
                    "model_identity": candidate.model_identity,
                    "model_family": candidate.model_family,
                    "preprocessing_name": candidate.preprocessing.name,
                    **components,
                    "learning_rate": candidate.training.learning_rate,
                    "max_epochs": candidate.training.max_epochs,
                    "patience": candidate.training.early_stopping_patience,
                    "batch_size": candidate.training.batch_size,
                    "metric_name": run.metric.name,
                    "metric_objective_value": fold.objective_value,
                    "metric_display_value": fold.display_value,
                    "selection_objective_mean": result.selection_objective_mean,
                    "selection_objective_std": result.selection_objective_std,
                    "selection_cv_mean": result.selection_cv_mean,
                    "selection_cv_std": result.selection_cv_std,
                    "test_objective": np.nan,
                    "test_score": np.nan,
                    "fit_time_seconds": fold.fit_time_seconds,
                    "score_time_seconds": fold.score_time_seconds,
                    "ranking_overhead_seconds": 0.0,
                    "peak_memory_mb": np.nan,
                    "status": fold.status.value,
                    "failure_type": fold.failure_type,
                    "failure_reason": fold.failure_reason,
                    "repository_commit": commit,
                    "environment_hash": environment_hash,
                    "python_version": environment["python"],
                    "platform": environment["platform"],
                }
            )
    return rows


def _rank_metrics(
    ordered_ids: Sequence[str],
    result_by_id: dict[str, CandidateResult],
) -> tuple[float | None, float, float]:
    successful = [
        candidate_id
        for candidate_id in ordered_ids
        if result_by_id[candidate_id].selection_objective_mean is not None
    ]
    if len(successful) < 2:
        return None, 0.0, 0.0
    predicted_positions = pd.Series(
        np.arange(1, len(successful) + 1, dtype=float), index=successful
    )
    actual_values = pd.Series(
        {
            candidate_id: float(result_by_id[candidate_id].selection_objective_mean)
            for candidate_id in successful
        }
    )
    actual_positions = actual_values.rank(method="average", ascending=False)
    spearman = float(predicted_positions.corr(actual_positions, method="spearman"))
    actual_top = set(actual_positions.nsmallest(min(5, len(successful))).index)
    predicted_top = set(successful[:5])
    recall = len(actual_top & predicted_top) / len(actual_top)
    relevance = actual_values.rank(method="average", ascending=True).to_dict()
    gains = [float(relevance[candidate_id]) for candidate_id in successful[:5]]
    ideal = sorted(relevance.values(), reverse=True)[:5]

    def dcg(values: Sequence[float]) -> float:
        return sum((2.0**value - 1.0) / math.log2(index + 2) for index, value in enumerate(values))

    ideal_dcg = dcg(ideal)
    ndcg = dcg(gains) / ideal_dcg if ideal_dcg else 0.0
    return spearman, float(recall), float(ndcg)


def budgets(
    candidate_count: int, absolute: Iterable[int], fractional: Iterable[float]
) -> list[int]:
    values = {min(candidate_count, max(1, int(value))) for value in absolute}
    values.update(
        min(candidate_count, max(1, math.ceil(float(fraction) * candidate_count)))
        for fraction in fractional
    )
    values.add(candidate_count)
    return sorted(values)


def _selected_at_budget(
    ordering: Sequence[CandidateSpec],
    result_by_id: dict[str, CandidateResult],
    budget: int,
) -> CandidateResult | None:
    successful = [
        result_by_id[candidate.candidate_id]
        for candidate in ordering[:budget]
        if result_by_id[candidate.candidate_id].selection_objective_mean is not None
    ]
    return (
        max(successful, key=lambda result: float(result.selection_objective_mean))
        if successful
        else None
    )


def _method_orderings(
    run: MeasuredRun,
    history: pd.DataFrame,
    random_seeds: Sequence[int],
    *,
    meta_artifact_dir: str | Path,
) -> list[tuple[str, int | None, list[CandidateSpec], float]]:
    result: list[tuple[str, int | None, list[CandidateSpec], float]] = []
    result.append(("exhaustive", None, list(run.candidates), 0.0))
    started = time.perf_counter()
    _, proxy_order = _proxy_scores(
        run.candidates,
        run.prepared.X_development,
        run.prepared.y_development,
        benchmark_task(run.dataset),
        run.metric,
        run.outer_seed,
    )
    proxy_rank = {name: index for index, name in enumerate(proxy_order)}
    proxy = sorted(
        run.candidates,
        key=lambda candidate: (
            proxy_rank.get(candidate.preprocessing.name, len(proxy_rank)),
            candidate.model_identity,
            candidate.candidate_id,
        ),
    )
    proxy_time = time.perf_counter() - started
    result.append(("proxy", None, proxy, proxy_time))

    target_checksum = run.dataset.local_sha256
    target_rows = history[history["dataset_checksum"] == target_checksum]
    if not target_rows.empty:
        raise RuntimeError(
            "Target-dataset rows reached leave-one-dataset-out meta-training history"
        )
    learner = MetaLearner(
        task_type=benchmark_task(run.dataset),
        metric_name=run.metric.name,
    )
    started = time.perf_counter()
    learner.train(history)
    if not learner.is_trained:
        raise RuntimeError(
            f"Leave-one-dataset-out history could not train for {run.dataset.benchmark_id}"
        )
    artifact = Path(meta_artifact_dir) / (
        f"{run.dataset.benchmark_id}_{run.outer_seed}_{run.metric.name}.pkl"
    )
    learner.save(str(artifact))
    scored = learner.rank_specs(run.prepared.dataset_meta, run.candidates)
    meta_scores = {candidate.candidate_id: score for candidate, score in scored}
    meta = [candidate for candidate, _ in scored]
    meta_time = time.perf_counter() - started
    result.append(("meta", None, meta, meta_time))
    top_proxy = set(proxy_order[: min(3, len(proxy_order))])
    proxy_meta = sorted(
        run.candidates,
        key=lambda candidate: (
            0 if candidate.preprocessing.name in top_proxy else 1,
            -meta_scores[candidate.candidate_id],
            proxy_rank.get(candidate.preprocessing.name, len(proxy_rank)),
            candidate.candidate_id,
        ),
    )
    result.append(("proxy_meta", None, proxy_meta, proxy_time + meta_time))
    for random_seed in random_seeds:
        random_order = list(run.candidates)
        np.random.default_rng(random_seed).shuffle(random_order)
        result.append(("random", int(random_seed), random_order, 0.0))
    return result


def replay_and_test(
    run: MeasuredRun,
    leave_one_out_history: pd.DataFrame,
    *,
    absolute_budgets: Sequence[int],
    fractional_budgets: Sequence[float],
    random_seeds: Sequence[int],
    meta_artifact_dir: str | Path,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    """Freeze every method selection from CV, then open the untouched test gate."""
    orderings = _method_orderings(
        run,
        leave_one_out_history,
        random_seeds,
        meta_artifact_dir=meta_artifact_dir,
    )
    result_by_id = run.result_by_id
    successful = [result for result in run.results if result.selection_objective_mean is not None]
    if not successful:
        raise RuntimeError("Exhaustive cache contains no successful candidate")
    oracle_best = max(float(result.selection_objective_mean) for result in successful)
    oracle_worst = min(float(result.selection_objective_mean) for result in successful)
    denominator = oracle_best - oracle_worst + np.finfo(float).eps
    candidate_budgets = budgets(len(run.candidates), absolute_budgets, fractional_budgets)

    # This map is frozen before X_test/y_test are passed to any fitting routine.
    frozen: list[
        tuple[str, int | None, int, list[CandidateSpec], CandidateResult | None, float]
    ] = []
    for method, ordering_seed, ordering, overhead in orderings:
        method_budgets = [len(run.candidates)] if method == "exhaustive" else candidate_budgets
        for budget in method_budgets:
            frozen.append(
                (
                    method,
                    ordering_seed,
                    budget,
                    ordering,
                    _selected_at_budget(ordering, result_by_id, budget),
                    overhead,
                )
            )

    selected_results = {
        item[4].candidate.candidate_id: item[4] for item in frozen if item[4] is not None
    }
    final_scores: dict[str, tuple[float, float]] = {}
    final_failures: dict[str, tuple[str, str]] = {}
    for candidate_id, selection in selected_results.items():
        try:
            final = fit_and_evaluate_final(
                selection.candidate,
                run.prepared.X_development,
                run.prepared.y_development,
                run.prepared.X_test,
                run.prepared.y_test,
                run.prepared.dataset_meta,
                benchmark_task(run.dataset),
                run.metric,
                selected_epochs=selected_epoch(selection),
                device=device,
                random_state=run.outer_seed,
            )
            final_scores[candidate_id] = (final.test_objective, final.test_score)
        except Exception as exc:
            final_scores[candidate_id] = (float("nan"), float("nan"))
            final_failures[candidate_id] = (type(exc).__name__, str(exc)[:1000])

    exhaustive_selection = next(
        selection
        for method, _, _, _, selection, _ in frozen
        if method == "exhaustive" and selection is not None
    )
    exhaustive_test_objective = final_scores[exhaustive_selection.candidate.candidate_id][0]

    rows: list[dict[str, Any]] = []
    for method, ordering_seed, budget, ordering, selection, overhead in frozen:
        ordered_ids = [candidate.candidate_id for candidate in ordering]
        spearman, recall5, ndcg5 = _rank_metrics(ordered_ids, result_by_id)
        selected_objective = (
            float(selection.selection_objective_mean) if selection is not None else np.nan
        )
        regret = (oracle_best - selected_objective) / denominator if selection is not None else 1.0
        test_objective, test_score = (
            final_scores[selection.candidate.candidate_id]
            if selection is not None
            else (np.nan, np.nan)
        )
        cached_evaluation_seconds = sum(
            fold.fit_time_seconds + fold.score_time_seconds
            for candidate in ordering[:budget]
            for fold in result_by_id[candidate.candidate_id].fold_results
        )
        rows.append(
            {
                "schema_version": METHOD_RESULT_SCHEMA_VERSION,
                "profile": "balanced-15-v1",
                "dataset_id": run.dataset.benchmark_id,
                "openml_task_id": run.dataset.openml_task_id,
                "dataset_checksum": run.dataset.local_sha256,
                "task_type": benchmark_task(run.dataset),
                "outer_seed": run.outer_seed,
                "method": method,
                "ordering_seed": ordering_seed,
                "candidate_budget": budget,
                "candidate_count": len(run.candidates),
                "candidate_universe_hash": candidate_universe_hash(run.candidates),
                "selected_candidate_id": (
                    selection.candidate.candidate_id if selection is not None else None
                ),
                "selection_objective": selected_objective,
                "selection_cv_score": (
                    selection.selection_cv_mean if selection is not None else np.nan
                ),
                "oracle_best_objective": oracle_best,
                "oracle_worst_objective": oracle_worst,
                "normalized_regret": float(np.clip(regret, 0.0, 1.0)),
                "quality_retention": float(np.clip(1.0 - regret, 0.0, 1.0)),
                "regret_le_005": bool(regret <= 0.05),
                "test_objective": test_objective,
                "test_score": test_score,
                "final_test_normalized_regret": (
                    float(
                        np.clip(
                            (exhaustive_test_objective - test_objective) / denominator,
                            0.0,
                            1.0,
                        )
                    )
                    if np.isfinite(test_objective)
                    else np.nan
                ),
                "spearman": spearman,
                "recall_at_5": recall5,
                "ndcg_at_5": ndcg5,
                "ranking_overhead_seconds": overhead,
                "cached_evaluation_seconds": cached_evaluation_seconds,
                "cached_total_seconds": overhead + cached_evaluation_seconds,
                "candidate_failure_count": sum(
                    result.status != ResultStatus.SUCCESS for result in run.results
                ),
                "final_failure_type": (
                    final_failures.get(selection.candidate.candidate_id, (None, None))[0]
                    if selection is not None
                    else None
                ),
                "final_failure_reason": (
                    final_failures.get(selection.candidate.candidate_id, (None, None))[1]
                    if selection is not None
                    else None
                ),
            }
        )
    return rows, final_scores


def history_dataframe(records: Iterable[MetaHistoryRecord]) -> pd.DataFrame:
    dataframe = pd.DataFrame([record.to_dict() for record in records])
    if dataframe.empty:
        return pd.DataFrame(columns=MetaHistoryRecord.__dataclass_fields__)
    if not (dataframe["schema_version"] == META_HISTORY_SCHEMA_VERSION).all():
        raise RuntimeError("Benchmark history contains a non-v2 row")
    return dataframe


def leave_one_dataset_out(
    history: pd.DataFrame,
    *,
    target_checksum: str,
    task_type: str,
    metric_name: str,
    min_training_datasets: int = 2,
) -> pd.DataFrame:
    result = history[
        (history["dataset_checksum"] != target_checksum)
        & (history["task_type"] == task_type)
        & (history["metric_name"] == metric_name)
    ].copy()
    if (result["dataset_checksum"] == target_checksum).any():
        raise RuntimeError("Leave-one-dataset-out filtering failed")
    if result["dataset_id"].nunique() < min_training_datasets:
        raise RuntimeError(
            f"Meta-training requires at least {min_training_datasets} non-target datasets"
        )
    return result
