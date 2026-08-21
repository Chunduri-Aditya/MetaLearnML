from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.evaluator import get_metric_spec
from engine.preprocessing_strategies import (
    PreprocessingStrategy,
    build_preprocessing_pipeline,
    rank_preprocessors_by_proxy,
)


def test_mixed_pipeline_handles_missing_and_unseen_categories(standard_strategy):
    train = pd.DataFrame({"number": [1.0, np.nan, 3.0], "group": ["a", None, "b"]})
    test = pd.DataFrame({"number": [np.nan], "group": ["unseen"]})
    pipeline = build_preprocessing_pipeline(standard_strategy, ["number"], ["group"])
    fit = pipeline.fit_transform(train)
    transformed = pipeline.transform(test)
    assert fit.shape[0] == 3
    assert transformed.shape == (1, fit.shape[1])
    assert np.isfinite(transformed).all()


def test_unknown_strategy_cannot_execute():
    strategy = PreprocessingStrategy.from_components(
        numeric_scaler="unknown",
        numeric_imputer="unknown",
        categorical_encoder="unknown",
        categorical_imputer="unknown",
    )
    with pytest.raises(ValueError, match="cannot run"):
        strategy.validate_executable(has_numeric=True, has_categorical=False)


def test_none_is_required_for_absent_groups():
    strategy = PreprocessingStrategy.from_components(
        numeric_scaler="standard",
        numeric_imputer="median",
        categorical_encoder="onehot",
        categorical_imputer="constant",
    )
    with pytest.raises(ValueError, match="no categorical"):
        strategy.validate_executable(has_numeric=True, has_categorical=False)


def test_proxy_ranking_is_repeatable(mixed_binary, standard_strategy):
    frame, target = mixed_binary
    ordinal = PreprocessingStrategy.from_components(
        numeric_scaler="robust",
        numeric_imputer="mean",
        categorical_encoder="ordinal",
        categorical_imputer="constant",
    )
    metric = get_metric_spec("f1_macro")
    first = rank_preprocessors_by_proxy(
        frame, target, [standard_strategy, ordinal], "classification", metric.callable
    )
    second = rank_preprocessors_by_proxy(
        frame, target, [standard_strategy, ordinal], "classification", metric.callable
    )
    assert [(item.name, score) for item, score in first] == [
        (item.name, score) for item, score in second
    ]
