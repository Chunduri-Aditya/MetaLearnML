"""Backward-compatible sequential adapter for the unified v2 search."""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.search import SearchOptions, prepare_search_data, run_ranked_search


def _ordering_mode(use_proxy: bool, use_meta_learning: bool) -> str:
    if use_proxy and use_meta_learning:
        return "proxy_meta"
    if use_proxy:
        return "proxy"
    if use_meta_learning:
        return "meta"
    return "registry"


def run_intelligent_search(
    X_df,
    y: np.ndarray,
    task_type: str,
    metric_name: str,
    dataset_meta: dict,
    device: str = "cpu",
    target_metric: float | None = None,
    max_models: int = 20,
    top_k_preprocessing: int = 3,
    use_proxy: bool = True,
    use_meta_learning: bool = True,
) -> dict[str, Any]:
    """Run v2 CV/test search while preserving the historical call signature."""
    prepared = prepare_search_data(
        X_df,
        y,
        task_type,
        base_meta=dataset_meta,
        test_ratio=0.2,
        cv_folds=5,
        random_state=42,
        target_name=dataset_meta.get("target_name"),
    )
    options = SearchOptions(
        ordering_mode=_ordering_mode(use_proxy, use_meta_learning),
        max_candidates=max_models,
        top_k_preprocessing=top_k_preprocessing,
        use_proxy_ranking=use_proxy,
        use_meta_learning=use_meta_learning,
        target_metric=target_metric,
        n_jobs=1,
        device=device,
    )
    result = run_ranked_search(
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
    )
    return result.to_legacy_dict()
