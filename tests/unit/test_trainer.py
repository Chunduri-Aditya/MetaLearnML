from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from engine.evaluator import accuracy_metric
from engine.trainer import evaluate, train_model, train_one_model
from models.model_zoo import build_model


def _classification_loaders():
    features = torch.randn(24, 3)
    target = (features[:, 0] > 0).long()
    dataset = TensorDataset(features, target)
    return DataLoader(dataset, batch_size=8), DataLoader(dataset, batch_size=12)


def test_torch_and_legacy_training_paths():
    train, validation = _classification_loaders()
    model = build_model("mlp_small", 3, 2)
    config = {
        "lr": 1e-3,
        "epochs": 2,
        "patience": 1,
        "metric_fn": accuracy_metric,
        "task_type": "classification",
        "loss_fn": torch.nn.CrossEntropyLoss(),
    }
    outcome = train_model(model, train, validation, "cpu", config)
    assert outcome.best_state_dict and outcome.best_epoch
    metric, state = train_one_model(
        build_model("mlp_small", 3, 2), train, validation, "cpu", config
    )
    assert isinstance(metric, float) and state


def test_sklearn_training_path():
    train, validation = _classification_loaders()
    model = build_model("logreg", 3, 2)
    outcome = train_model(
        model,
        train,
        validation,
        "cpu",
        {"lr": 1e-3, "epochs": 1, "metric_fn": accuracy_metric, "task_type": "classification"},
    )
    assert outcome.best_state_dict == {"sklearn_fitted": True}
    assert evaluate(model, validation, "cpu", accuracy_metric) >= 0
