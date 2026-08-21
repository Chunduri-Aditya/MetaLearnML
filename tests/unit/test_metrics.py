from __future__ import annotations

import pytest
import torch

from engine.evaluator import (
    accuracy_metric,
    f1_macro_metric,
    get_metric_spec,
    neg_mae_metric,
    neg_rmse_metric,
    r2_metric,
)


def test_classification_metrics():
    logits = torch.tensor([[3.0, 1.0], [0.0, 2.0], [2.0, 0.0]])
    target = torch.tensor([0, 1, 1])
    assert accuracy_metric(logits, target) == pytest.approx(2 / 3)
    assert 0 <= f1_macro_metric(logits, target) <= 1


def test_regression_objective_display_and_gap_direction():
    target = torch.tensor([1.0, 2.0, 3.0])
    prediction = torch.tensor([[1.0], [2.0], [4.0]])
    assert neg_rmse_metric(target, prediction) < 0
    assert neg_mae_metric(target, prediction) < 0
    assert r2_metric(target, prediction) < 1
    rmse = get_metric_spec("rmse")
    objective, display = rmse.score_tensors(target, prediction)
    assert objective == -display
    assert rmse.target_reached(display, display + 0.1)
    assert rmse.generalization_gap(2.0, 3.0) == 1.0


def test_higher_is_better_threshold_and_unknown_metric():
    f1 = get_metric_spec("f1_macro")
    assert f1.target_reached(0.8, 0.75)
    assert f1.generalization_gap(0.8, 0.7) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric_spec("made-up")
