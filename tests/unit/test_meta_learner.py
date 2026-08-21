from __future__ import annotations

import pytest

from engine.meta_history import MetaHistoryValidationError
from engine.meta_learner import MetaLearner


def test_meta_learner_trains_predicts_and_ranks(history_frame, numeric_strategy):
    learner = MetaLearner(task_type="classification", metric_name="f1_macro")
    learner.train(history_frame)
    assert learner.is_trained
    meta = {
        "n_samples": 100,
        "n_features": 4,
        "n_numeric": 4,
        "n_categorical": 0,
        "n_text": 0,
        "n_classes": 2,
        "class_imbalance": 0.1,
        "missing_value_fraction": 0.2,
    }
    score = learner.predict(meta, "logreg", numeric_strategy)
    assert isinstance(score, float)
    with pytest.raises(TypeError, match="requires a PreprocessingStrategy"):
        learner.extract_features(meta, "logreg", "old_name")


def test_insufficient_history_fails_training(history_frame):
    learner = MetaLearner(task_type="classification", metric_name="f1_macro")
    with pytest.raises(MetaHistoryValidationError, match="at least 20"):
        learner.train(history_frame.iloc[:5])
    assert not learner.is_trained
