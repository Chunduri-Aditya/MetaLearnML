from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.config import AutoMLConfig
from engine.orchestrator import run_automl
from engine.predictor import BestModelPredictor

pytestmark = pytest.mark.integration


def test_end_to_end_untouched_test_and_raw_bundle(tmp_path, tiny_csv):
    config = AutoMLConfig(
        csv_path=str(tiny_csv),
        label_col="label",
        use_proxy_ranking=False,
        use_meta_learning=False,
        ordering_mode="registry",
        max_candidates=1,
        cv_folds=3,
        max_epochs=1,
        early_stopping_patience=1,
        n_jobs=1,
        results_csv=str(tmp_path / "results.csv"),
        ckpt_dir=str(tmp_path / "models"),
        report_dir=str(tmp_path / "reports"),
    )
    result = run_automl(config)
    assert result["selection_cv_mean"] is not None
    assert result["test_score"] is not None
    bundle = result["best_info"]["bundle_path"]
    assert Path(bundle, "manifest.json").exists()
    assert Path(bundle, "label_encoder.joblib").exists()
    predictor = BestModelPredictor(bundle_path=bundle)
    raw = pd.read_csv(tiny_csv).drop(columns=["label"]).iloc[:3]
    assert len(predictor.predict_raw(raw)) == 3
    assert len(predictor.predict_dataframe(raw)) == 3
    assert Path(result["report_paths"]["json"]).exists()
    persisted = pd.read_csv(tmp_path / "results.csv")
    assert persisted["candidate_id"].nunique() == len(result["search_result"].proposed_ranking)
    assert "skipped" in set(persisted["status"])


def test_config_defaults_and_compatibility_alias(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(20), "label": [0, 1] * 10}).to_csv(csv_path, index=False)
    config = AutoMLConfig(
        str(csv_path),
        use_proxy_ranking=False,
        use_meta_learning=True,
    )
    assert config.ordering_mode == "meta"
    assert config.test_ratio == 0.2 and config.cv_folds == 5 and config.random_state == 42
    with pytest.warns(DeprecationWarning):
        translated = AutoMLConfig(str(csv_path), train_ratio=0.75)
    assert translated.test_ratio == 0.25
    with pytest.raises(ValueError, match="test_ratio"):
        AutoMLConfig(str(csv_path), test_ratio=1.0)
