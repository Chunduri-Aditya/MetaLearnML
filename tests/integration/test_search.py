from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.bundle import load_pipeline_bundle, save_pipeline_bundle
from engine.candidate_evaluator import evaluate_candidate, fit_and_evaluate_final
from engine.candidates import CandidateSpec, ResultStatus, TrainingConfig
from engine.evaluator import get_metric_spec
from engine.predictor import BestModelPredictor
from engine.preprocessing_strategies import PreprocessingStrategy
from engine.search import (
    SearchOptions,
    generate_candidate_universe,
    prepare_search_data,
    rank_candidate_universe,
)

pytestmark = pytest.mark.integration


def test_shared_evaluator_returns_fold_results_and_final_bundle(
    tmp_path, numeric_regression, numeric_strategy
):
    frame, target = numeric_regression
    prepared = prepare_search_data(frame, target, "regression", cv_folds=3, random_state=42)
    candidate = CandidateSpec.create(
        "ridge", numeric_strategy, TrainingConfig(max_epochs=1, random_state=42)
    )
    metric = get_metric_spec("r2")
    result = evaluate_candidate(
        candidate,
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        "regression",
        metric,
        prepared.folds,
    )
    assert result.status == ResultStatus.SUCCESS
    assert len(result.fold_results) == 3
    final = fit_and_evaluate_final(
        candidate,
        prepared.X_development,
        prepared.y_development,
        prepared.X_test,
        prepared.y_test,
        prepared.dataset_meta,
        "regression",
        metric,
        selected_epochs=None,
    )
    path = save_pipeline_bundle(
        final.fitted,
        tmp_path / "bundle",
        task_type="regression",
        metric_name="r2",
        test_score=final.test_score,
    )
    loaded = load_pipeline_bundle(path)
    predictions = loaded.predict(prepared.X_test.iloc[:3])
    assert predictions.shape == (3,)
    predictor = BestModelPredictor(bundle_path=str(path))
    assert predictor.predict_raw(prepared.X_test.iloc[:2]).shape == (2,)
    with pytest.raises(ValueError, match="missing required"):
        loaded.predict(pd.DataFrame({"wrong": [1.0]}))


def test_failed_candidate_is_structured(numeric_regression):
    frame, target = numeric_regression
    prepared = prepare_search_data(frame, target, "regression", cv_folds=3)
    unknown = PreprocessingStrategy.from_components(
        numeric_scaler="unknown",
        numeric_imputer="unknown",
        categorical_encoder="none",
        categorical_imputer="none",
    )
    candidate = CandidateSpec.create("ridge", unknown)
    result = evaluate_candidate(
        candidate,
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        "regression",
        get_metric_spec("r2"),
        prepared.folds,
    )
    assert result.status == ResultStatus.FAILED
    assert all(fold.failure_type == "ValueError" for fold in result.fold_results)


def test_preprocessor_fit_never_sees_validation_or_test_rows(
    monkeypatch, numeric_regression, numeric_strategy
):
    frame, target = numeric_regression
    prepared = prepare_search_data(frame, target, "regression", cv_folds=3)
    events = []

    class SpyPreprocessor:
        def fit_transform(self, fit_frame):
            events.append(("fit", frozenset(fit_frame.index)))
            return np.nan_to_num(fit_frame.to_numpy(dtype="float32"))

        def transform(self, score_frame):
            events.append(("transform", frozenset(score_frame.index)))
            return np.nan_to_num(score_frame.to_numpy(dtype="float32"))

    monkeypatch.setattr(
        "engine.candidate_evaluator.build_preprocessing_pipeline",
        lambda *args, **kwargs: SpyPreprocessor(),
    )
    candidate = CandidateSpec.create("ridge", numeric_strategy)
    metric = get_metric_spec("r2")
    evaluate_candidate(
        candidate,
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        "regression",
        metric,
        prepared.folds,
    )
    development_index = set(prepared.X_development.index)
    test_index = set(prepared.X_test.index)
    fit_events = [set(indices) for kind, indices in events if kind == "fit"]
    transform_events = [set(indices) for kind, indices in events if kind == "transform"]
    assert len(fit_events) == len(transform_events) == 3
    for fit_indices, validation_indices in zip(fit_events, transform_events, strict=True):
        assert fit_indices.isdisjoint(validation_indices)
        assert fit_indices | validation_indices == development_index
        assert fit_indices.isdisjoint(test_index)
        assert validation_indices.isdisjoint(test_index)

    events.clear()
    fit_and_evaluate_final(
        candidate,
        prepared.X_development,
        prepared.y_development,
        prepared.X_test,
        prepared.y_test,
        prepared.dataset_meta,
        "regression",
        metric,
        selected_epochs=None,
    )
    assert events == [
        ("fit", frozenset(prepared.X_development.index)),
        ("transform", frozenset(prepared.X_test.index)),
    ]


def test_all_ordering_modes_and_random_determinism(
    mixed_binary, standard_strategy, tmp_path, history_frame
):
    frame, target = mixed_binary
    prepared = prepare_search_data(frame, target, "classification", cv_folds=3)
    candidates = generate_candidate_universe(
        prepared.dataset_meta,
        "classification",
        strategies=[standard_strategy],
        max_epochs=1,
        early_stopping_patience=1,
    )
    metric = get_metric_spec("f1_macro")
    from engine.meta_learner import MetaLearner

    learner = MetaLearner(task_type="classification", metric_name="f1_macro")
    learner.train(history_frame)
    artifact = tmp_path / "meta.pkl"
    learner.save(str(artifact))

    rankings = {}
    for mode in ("registry", "random", "proxy", "meta", "proxy_meta"):
        options = SearchOptions(
            ordering_mode=mode,
            use_proxy_ranking=mode in {"proxy", "proxy_meta"},
            use_meta_learning=mode in {"meta", "proxy_meta"},
            random_state=8,
            meta_model_path=str(artifact),
        )
        ranking, _, _ = rank_candidate_universe(
            candidates,
            prepared.X_development,
            prepared.y_development,
            prepared.dataset_meta,
            "classification",
            metric,
            options,
        )
        rankings[mode] = [candidate.candidate_id for candidate in ranking]
        assert set(rankings[mode]) == {candidate.candidate_id for candidate in candidates}
    repeated, _, _ = rank_candidate_universe(
        candidates,
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        "classification",
        metric,
        SearchOptions(
            ordering_mode="random",
            use_proxy_ranking=False,
            use_meta_learning=False,
            random_state=8,
        ),
    )
    assert rankings["random"] == [candidate.candidate_id for candidate in repeated]


def test_flag_requirements_are_independent():
    with pytest.raises(ValueError, match="requires use_proxy"):
        SearchOptions(
            ordering_mode="proxy",
            use_proxy_ranking=False,
            use_meta_learning=False,
        )
    options = SearchOptions(ordering_mode="meta", use_proxy_ranking=False, use_meta_learning=True)
    assert not options.use_proxy_ranking and options.use_meta_learning
