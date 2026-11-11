import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from engine.predictor import BestModelPredictor, _load_best_from_csv
from models.model_zoo import build_model


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> str:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def test_load_best_from_csv_picks_highest_metric_and_skips_blank(tmp_path):
    results = _write_csv(
        tmp_path / "results.csv",
        ["model_name", "metric", "checkpoint_path"],
        [["mlp_small", "0.5", "a.pth"], ["mlp_medium", "0.9", "b.pth"], ["mlp_deep", "", "c.pth"]],
    )
    assert _load_best_from_csv(results) == ("mlp_medium", 0.9, "b.pth")


def test_load_best_from_csv_accepts_alternate_column_names(tmp_path):
    results = _write_csv(
        tmp_path / "results.csv",
        ["model_name", "metric_value", "ckpt_path"],
        [["mlp_small", "0.25", "x.pth"], ["mlp_deep", "0.75", "y.pth"]],
    )
    assert _load_best_from_csv(results) == ("mlp_deep", 0.75, "y.pth")


def test_load_best_from_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Results file not found"):
        _load_best_from_csv(str(tmp_path / "absent.csv"))


def test_load_best_from_csv_header_only_raises(tmp_path):
    results = _write_csv(tmp_path / "results.csv", ["model_name", "metric", "checkpoint_path"], [])
    with pytest.raises(RuntimeError, match="No rows found"):
        _load_best_from_csv(results)


def test_load_best_from_csv_best_row_without_checkpoint_raises(tmp_path):
    results = _write_csv(
        tmp_path / "results.csv",
        ["model_name", "metric", "checkpoint_path"],
        [["mlp_small", "0.5", "a.pth"], ["mlp_medium", "0.9", ""]],
    )
    with pytest.raises(RuntimeError, match="Checkpoint path missing"):
        _load_best_from_csv(results)


def test_predictor_without_arguments_requires_bundle_path():
    with pytest.raises(TypeError, match="bundle_path"):
        BestModelPredictor()


@pytest.fixture
def legacy_predictor(tmp_path):
    torch.manual_seed(0)
    source = build_model("mlp_small", 4, 3)
    ckpt = tmp_path / "mlp_small.pth"
    torch.save(source.state_dict(), ckpt)
    results = _write_csv(
        tmp_path / "results.csv",
        ["model_name", "metric", "checkpoint_path"],
        [["mlp_small", "0.8", str(ckpt)]],
    )
    predictor = BestModelPredictor(results_csv=results, in_dim=4, out_dim=3, device="cpu")
    return source.eval(), predictor


def test_legacy_predictor_reads_best_row(legacy_predictor):
    _, predictor = legacy_predictor
    assert predictor.model_name == "mlp_small"
    assert predictor.best_metric == 0.8
    assert predictor.device == "cpu"


def test_legacy_predict_logits_matches_source_model(legacy_predictor):
    source, predictor = legacy_predictor
    x = np.random.default_rng(1).normal(size=(5, 4)).astype(np.float32)
    expected = source(torch.from_numpy(x)).detach()
    logits = predictor.predict_logits(x)
    assert logits.shape == (5, 3)
    assert torch.allclose(logits, expected)


def test_legacy_predict_proba_rows_sum_to_one(legacy_predictor):
    _, predictor = legacy_predictor
    x = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)
    probs = predictor.predict_proba(x)
    assert probs.shape == (6, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(6), atol=1e-5)


def test_legacy_predict_classes_is_argmax_of_logits(legacy_predictor):
    _, predictor = legacy_predictor
    x = np.random.default_rng(3).normal(size=(7, 4)).astype(np.float32)
    classes = predictor.predict_classes(x)
    assert classes.shape == (7,)
    assert torch.equal(classes, torch.argmax(predictor.predict_logits(x), dim=1))


def test_to_tensor_converts_float64_to_float32(legacy_predictor):
    _, predictor = legacy_predictor
    tensor = predictor._to_tensor(np.ones((2, 4), dtype=np.float64))
    assert tensor.dtype == torch.float32
    assert tensor.shape == (2, 4)


def test_to_tensor_rejects_list(legacy_predictor):
    _, predictor = legacy_predictor
    with pytest.raises(TypeError, match="Unsupported type"):
        predictor._to_tensor([[1.0, 2.0, 3.0, 4.0]])


def test_predict_raw_requires_bundle(legacy_predictor):
    _, predictor = legacy_predictor
    with pytest.raises(RuntimeError, match="bundle_path"):
        predictor.predict_raw(pd.DataFrame({"a": [1.0]}))
    with pytest.raises(RuntimeError, match="bundle_path"):
        predictor.predict_dataframe(pd.DataFrame({"a": [1.0]}))
