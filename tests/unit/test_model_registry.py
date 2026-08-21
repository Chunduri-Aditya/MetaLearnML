from __future__ import annotations

import pytest
import torch

from models.model_zoo import MODEL_REGISTRY, build_model, validate_model_registry


def test_registry_has_stable_required_fields():
    snapshot = validate_model_registry()
    assert len(snapshot) == len(MODEL_REGISTRY)
    assert {identity for identity, _, _ in snapshot} == set(MODEL_REGISTRY)


@pytest.mark.parametrize("identity", ["rf", "gb", "logreg", "svm_clf"])
def test_classification_registry_models_build_and_forward(identity):
    model = build_model(identity, 3, 2)
    if MODEL_REGISTRY[identity]["backend"] == "sklearn":
        features = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0]])
        target = torch.tensor([0, 1, 1])
        model.fit(features, target)
        assert model(features).shape == (3, 2)


@pytest.mark.parametrize("identity", ["linreg", "ridge", "lasso", "rf_reg", "gb_reg", "svm_reg"])
def test_regression_registry_models_build_and_forward(identity):
    model = build_model(identity, 2, 1)
    features = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    target = torch.tensor([0.0, 1.0, 2.0, 3.0])
    model.fit(features, target)
    assert model(features).shape == (4, 1)


def test_unknown_registry_model_fails():
    with pytest.raises(ValueError, match="Unknown"):
        build_model("does-not-exist", 2, 2)
