from __future__ import annotations

import joblib
import pytest

from engine.candidates import CandidateSpec, TrainingConfig
from engine.meta_features import (
    META_FEATURE_SCHEMA_VERSION,
    PREPROCESSING_VOCABULARIES,
    build_feature_schema,
    encode_meta_features,
)
from engine.meta_learner import IncompatibleMetaModelError, MetaLearner
from engine.preprocessing_strategies import (
    PreprocessingStrategy,
    generate_preprocessing_strategies,
)
from models.model_zoo import MODEL_REGISTRY

DATASET_META = {
    "n_samples": 100,
    "n_features": 4,
    "n_numeric": 4,
    "n_categorical": 0,
    "n_text": 0,
    "n_classes": 2,
    "class_imbalance": 0.2,
    "missing_value_fraction": 0.1,
}


@pytest.mark.parametrize(
    ("has_numeric", "has_categorical"), [(True, False), (False, True), (True, True)]
)
def test_every_generated_strategy_has_one_bit_per_component(has_numeric, has_categorical):
    strategies = generate_preprocessing_strategies(
        has_numeric,
        has_categorical,
        2_000,
        {"n_categorical": int(has_categorical)},
    )
    assert strategies
    for strategy in strategies:
        encoded = strategy.to_feature_dict()
        for group, vocabulary in PREPROCESSING_VOCABULARIES.items():
            assert sum(encoded[f"preprocessing_{group}_{item}"] for item in vocabulary) == 1


def test_names_are_labels_not_feature_sources(standard_strategy):
    renamed = PreprocessingStrategy(name="display-only", **standard_strategy.to_component_dict())
    assert renamed.to_feature_dict() == standard_strategy.to_feature_dict()


def test_component_difference_changes_only_expected_block(standard_strategy):
    changed = PreprocessingStrategy.from_components(
        **{**standard_strategy.to_component_dict(), "numeric_scaler": "robust"}
    )
    first = standard_strategy.to_feature_dict()
    second = changed.to_feature_dict()
    differing = {key for key in first if first[key] != second[key]}
    assert differing == {
        "preprocessing_numeric_scaler_standard",
        "preprocessing_numeric_scaler_robust",
    }


def test_schema_order_and_hash_are_deterministic():
    first = build_feature_schema()
    second = build_feature_schema()
    assert first == second
    assert first.version == META_FEATURE_SCHEMA_VERSION
    assert len(first.schema_hash) == 64


@pytest.mark.parametrize("identity", sorted(MODEL_REGISTRY))
def test_every_registry_model_has_exact_identity_and_family_bits(identity, numeric_strategy):
    encoded = encode_meta_features(DATASET_META, identity, numeric_strategy)
    values = dict(zip(encoded.names, encoded.values, strict=True))
    identity_bits = [values[name] for name in encoded.names if name.startswith("model_identity_")]
    family_bits = [values[name] for name in encoded.names if name.startswith("model_family_")]
    assert sum(identity_bits) == 1
    assert sum(family_bits) == 1
    assert values[f"model_identity_{identity}"] == 1
    assert values[f"model_family_{MODEL_REGISTRY[identity]['family']}"] == 1


def test_unknown_values_are_explicit(numeric_strategy):
    encoded = encode_meta_features(
        DATASET_META,
        "new_model",
        {key: "future_value" for key in PREPROCESSING_VOCABULARIES},
    )
    values = dict(zip(encoded.names, encoded.values, strict=True))
    assert values["model_identity_unknown"] == 1
    assert values["model_family_unknown"] == 1
    for group in PREPROCESSING_VOCABULARIES:
        assert values[f"preprocessing_{group}_unknown"] == 1


def test_candidate_hash_covers_training_config(numeric_strategy):
    first = CandidateSpec.create("rf", numeric_strategy, TrainingConfig(random_state=1))
    second = CandidateSpec.create("rf", numeric_strategy, TrainingConfig(random_state=2))
    renamed = CandidateSpec.create(
        "rf",
        PreprocessingStrategy(name="display-only", **numeric_strategy.to_component_dict()),
        TrainingConfig(random_state=1),
    )
    assert len(first.candidate_id) == 64
    assert first.display_id == first.candidate_id[:12]
    assert first.candidate_id != second.candidate_id
    assert first.candidate_id == renamed.candidate_id


def test_meta_model_artifact_rejects_schema_mismatch(tmp_path, history_frame):
    learner = MetaLearner(task_type="classification", metric_name="f1_macro")
    learner.train(history_frame)
    artifact = tmp_path / "model.pkl"
    learner.save(str(artifact))
    reloaded = MetaLearner(task_type="classification", metric_name="f1_macro")
    reloaded.load(str(artifact))
    assert reloaded.is_trained
    wrong_metric = MetaLearner(task_type="classification", metric_name="accuracy")
    with pytest.raises(IncompatibleMetaModelError, match="metric"):
        wrong_metric.load(str(artifact))

    payload = joblib.load(artifact)
    payload["meta_feature_schema_version"] = 1
    stale = tmp_path / "stale.pkl"
    joblib.dump(payload, stale)
    with pytest.raises(IncompatibleMetaModelError, match="Retrain"):
        reloaded.load(str(stale))


def test_missing_fraction_must_be_in_range(numeric_strategy):
    with pytest.raises(ValueError, match="within"):
        encode_meta_features(
            {**DATASET_META, "missing_value_fraction": 1.2}, "rf", numeric_strategy
        )
