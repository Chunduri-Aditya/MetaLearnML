"""Top-level orchestration for leakage-safe MetaLearnML v2 runs."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from engine.config import AutoMLConfig
from engine.meta_history import (
    MetaHistoryRecord,
    MetaHistoryValidationError,
    append_history_records,
    environment_fingerprint,
    repository_commit,
)
from engine.meta_learner import MetaLearner
from engine.report_generator import generate_report
from engine.search import SearchOptions, prepare_search_data, run_ranked_search
from models.model_zoo import MODEL_REGISTRY
from utils.datasets import load_raw_tabular_dataset


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def _result_rows(search_result, dataset_name: str, task_type: str) -> list[dict[str, Any]]:
    rows = []
    evaluated_ids = set()
    for candidate_result in search_result.candidate_results:
        candidate = candidate_result.candidate
        evaluated_ids.add(candidate.candidate_id)
        for fold in candidate_result.fold_results:
            rows.append(
                {
                    "result_schema_version": 2,
                    "dataset_name": dataset_name,
                    "task_type": task_type,
                    "candidate_id": candidate.candidate_id,
                    "model_identity": candidate.model_identity,
                    "model_family": candidate.model_family,
                    "preprocessing_name": candidate.preprocessing.name,
                    **candidate.preprocessing.to_component_dict(),
                    "training_config": json.dumps(asdict(candidate.training), sort_keys=True),
                    "proposed_rank": candidate_result.proposed_rank,
                    "predicted_score": candidate_result.predicted_score,
                    "fold_id": fold.fold_id,
                    "status": fold.status.value,
                    "objective_value": fold.objective_value,
                    "display_value": fold.display_value,
                    "best_epoch": fold.best_epoch,
                    "fit_time_seconds": fold.fit_time_seconds,
                    "score_time_seconds": fold.score_time_seconds,
                    "failure_type": fold.failure_type,
                    "failure_reason": fold.failure_reason,
                }
            )
    for proposed_rank, candidate in enumerate(search_result.proposed_ranking, start=1):
        if candidate.candidate_id in evaluated_ids:
            continue
        rows.append(
            {
                "result_schema_version": 2,
                "dataset_name": dataset_name,
                "task_type": task_type,
                "candidate_id": candidate.candidate_id,
                "model_identity": candidate.model_identity,
                "model_family": candidate.model_family,
                "preprocessing_name": candidate.preprocessing.name,
                **candidate.preprocessing.to_component_dict(),
                "training_config": json.dumps(asdict(candidate.training), sort_keys=True),
                "proposed_rank": proposed_rank,
                "predicted_score": search_result.predicted_scores.get(candidate.candidate_id),
                "fold_id": -1,
                "status": "skipped",
                "objective_value": None,
                "display_value": None,
                "best_epoch": None,
                "fit_time_seconds": 0.0,
                "score_time_seconds": 0.0,
                "failure_type": None,
                "failure_reason": (
                    "Not evaluated because the candidate budget or stopping condition was reached"
                ),
            }
        )
    return rows


def _history_records(
    search_result,
    prepared,
    raw: dict[str, Any],
    task_type: str,
    metric_name: str,
    random_state: int,
    run_id: str,
) -> list[MetaHistoryRecord]:
    environment_hash, _ = environment_fingerprint()
    commit = repository_commit()
    meta = prepared.dataset_meta
    records = []
    for result in search_result.candidate_results:
        candidate = result.candidate
        for fold in result.fold_results:
            if fold.objective_value is None or fold.display_value is None:
                continue
            records.append(
                MetaHistoryRecord.create(
                    run_id=run_id,
                    dataset_id=str(raw["dataset_id"]),
                    dataset_version=str(raw["dataset_version"]),
                    dataset_checksum=str(raw["dataset_checksum"]),
                    dataset_name=str(raw["dataset_name"]),
                    task_type=task_type,
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
                    **candidate.preprocessing.to_component_dict(),
                    metric_name=metric_name,
                    metric_objective_value=float(fold.objective_value),
                    metric_display_value=float(fold.display_value),
                    outer_split_seed=random_state,
                    inner_fold_id=fold.fold_id,
                    repository_commit=commit,
                    environment_hash=environment_hash,
                )
            )
    return records


def _log_optional_neo4j(
    dataset_name: str,
    task_type: str,
    search_result,
    meta: dict[str, Any],
    run_id: str,
    outer_seed: int,
) -> None:
    if search_result.best_candidate is None:
        return
    try:
        from graph.neo4j_client import Neo4jGraph

        graph = Neo4jGraph()
        if not graph.is_connected():
            return
        candidate = search_result.best_candidate
        graph.upsert_dataset(dataset_name, {**meta, "task_type": task_type})
        entry = MODEL_REGISTRY[candidate.model_identity]
        graph.upsert_model(candidate.model_identity, entry["family"], entry["backend"])
        graph.upsert_preprocessor(
            candidate.preprocessing.name,
            components=candidate.preprocessing.to_component_dict(),
        )
        environment_hash, _ = environment_fingerprint()
        graph.create_run(
            run_id,
            search_result.metric_name,
            search_result.selection_objective_mean,
            display_value=search_result.selection_cv_mean,
            candidate_id=candidate.candidate_id,
            repository_commit=repository_commit(),
            environment_hash=environment_hash,
            outer_seed=outer_seed,
        )
        graph.link_run(
            dataset_name,
            candidate.model_identity,
            candidate.preprocessing.name,
            run_id,
        )
        graph.close()
    except Exception as exc:
        print(f"[WARN] Could not log to Neo4j: {exc}")


def run_automl(config: AutoMLConfig) -> dict[str, Any]:
    """Run candidate selection with inner CV and final untouched-test evaluation."""
    started = time.perf_counter()
    raw = load_raw_tabular_dataset(
        config.csv_path,
        label_col=config.label_col,
        dataset_name=config.dataset_name,
        task_type=config.task_type,
    )
    task_type = "regression" if raw["task_type"] == "regression" else "classification"
    metric_name = config.metric_name or ("r2" if task_type == "regression" else "f1_macro")
    base_meta = {
        "dataset_id": raw["dataset_id"],
        "dataset_version": raw["dataset_version"],
        "dataset_checksum": raw["dataset_checksum"],
        "dataset_name": raw["dataset_name"],
    }
    prepared = prepare_search_data(
        raw["X_df"],
        raw["y_series"].to_numpy(),
        task_type,
        base_meta=base_meta,
        test_ratio=config.test_ratio,
        cv_folds=config.cv_folds,
        cv_repeats=config.cv_repeats,
        random_state=config.random_state,
        target_name=raw["target_name"],
    )
    dataset_name = raw["dataset_name"]
    options = SearchOptions(
        ordering_mode=config.ordering_mode,
        max_candidates=config.max_candidates,
        top_k_preprocessing=config.top_k_preprocessing,
        exhaustive=config.exhaustive,
        use_proxy_ranking=config.use_proxy_ranking,
        use_meta_learning=config.use_meta_learning,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        device=config.device,
        target_metric=config.target_metric,
        refit_on_development=config.refit_on_development,
        bundle_dir=str(Path(config.ckpt_dir) / f"{_safe_name(dataset_name)}_best_pipeline"),
    )
    search_result = run_ranked_search(
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        task_type,
        metric_name,
        prepared.folds,
        options=options,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        label_encoder=prepared.label_encoder,
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
    )

    result_rows = _result_rows(search_result, dataset_name, task_type)
    results_path = Path(config.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(results_path, index=False)

    run_id = str(uuid.uuid4())
    history_summary = None
    if config.use_meta_learning:
        records = _history_records(
            search_result,
            prepared,
            raw,
            task_type,
            search_result.metric_name,
            config.random_state,
            run_id,
        )
        if records:
            history_summary = append_history_records("experiments/meta_train.csv", records)
            learner = MetaLearner(
                task_type=task_type,
                metric_name=search_result.metric_name,
            )
            try:
                learner.train(history_summary.dataframe)
                learner.save()
            except MetaHistoryValidationError as exc:
                reason = f"Meta-model retraining disabled: {exc}"
                search_result.meta_ranking_reason = (
                    f"{search_result.meta_ranking_reason}; {reason}"
                    if search_result.meta_ranking_reason
                    else reason
                )

    _log_optional_neo4j(
        dataset_name,
        task_type,
        search_result,
        prepared.dataset_meta,
        run_id,
        config.random_state,
    )
    elapsed = time.perf_counter() - started
    legacy = search_result.to_legacy_dict()
    all_results = [
        {
            "candidate_id": result.candidate.candidate_id,
            "model_name": result.candidate.model_identity,
            "preprocessing_name": result.candidate.preprocessing.name,
            "selection_cv_mean": result.selection_cv_mean,
            "selection_cv_std": result.selection_cv_std,
            "status": result.status.value,
        }
        for result in search_result.candidate_results
    ]
    report_paths = generate_report(
        dataset_name=dataset_name,
        task_type=task_type,
        best_info=legacy["best_info"] or {},
        all_results=all_results,
        dataset_meta=prepared.dataset_meta,
        time_taken=elapsed,
        output_dir=config.report_dir,
    )
    return {
        "best_info": legacy["best_info"],
        "all_results": all_results,
        "report_paths": report_paths,
        "time_taken": elapsed,
        "search_result": search_result,
        "selection_cv_mean": search_result.selection_cv_mean,
        "selection_cv_std": search_result.selection_cv_std,
        "test_score": search_result.test_score,
        "generalization_gap": search_result.generalization_gap,
        "bundle_path": search_result.bundle_path,
        "history_summary": history_summary,
    }
