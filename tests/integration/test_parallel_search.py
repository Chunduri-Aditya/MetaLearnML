from __future__ import annotations

import pytest

from engine.search import SearchOptions, prepare_search_data, run_ranked_search

pytestmark = pytest.mark.integration


def test_sequential_parallel_parity_and_exhaustive_behavior(numeric_regression, numeric_strategy):
    frame, target = numeric_regression
    prepared = prepare_search_data(frame, target, "regression", cv_folds=3)
    common = dict(
        ordering_mode="registry",
        max_candidates=1,
        use_proxy_ranking=False,
        use_meta_learning=False,
        refit_on_development=False,
        random_state=42,
    )
    sequential = run_ranked_search(
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        "regression",
        "r2",
        prepared.folds,
        options=SearchOptions(**common, n_jobs=1),
        strategies=[numeric_strategy],
        max_epochs=1,
        early_stopping_patience=1,
    )
    parallel = run_ranked_search(
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        "regression",
        "r2",
        prepared.folds,
        options=SearchOptions(**common, n_jobs=2),
        strategies=[numeric_strategy],
        max_epochs=1,
        early_stopping_patience=1,
    )
    assert sequential.best_candidate == parallel.best_candidate
    assert sequential.selection_cv_mean == pytest.approx(parallel.selection_cv_mean)
    assert len(sequential.proposed_ranking) == 9
    assert len(sequential.candidate_results) == 1

    exhaustive = run_ranked_search(
        prepared.X_development,
        prepared.y_development,
        prepared.dataset_meta,
        "regression",
        "r2",
        prepared.folds,
        options=SearchOptions(
            **common,
            exhaustive=True,
            n_jobs=2,
            target_metric=-1_000_000.0,
        ),
        strategies=[numeric_strategy],
        max_epochs=1,
        early_stopping_patience=1,
    )
    assert len(exhaustive.candidate_results) == 9
