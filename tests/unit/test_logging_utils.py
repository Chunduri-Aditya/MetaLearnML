import csv
from types import SimpleNamespace

from utils.logging_utils import (
    init_results_file,
    init_results_file_with_preprocessing,
    log_result,
    log_result_with_preprocessing,
)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_init_results_file_simple_header(tmp_path):
    path = tmp_path / "sub" / "r.csv"
    init_results_file(str(path), include_meta_features=False)
    with open(path, newline="") as f:
        assert next(csv.reader(f)) == ["model_name", "metric", "checkpoint_path"]


def test_init_results_file_meta_header(tmp_path):
    path = tmp_path / "r.csv"
    init_results_file(str(path))
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    assert header[:5] == ["dataset_name", "model_name", "family", "metric_name", "metric"]
    assert len(header) == 13


def test_init_results_file_keeps_existing_content(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("keep,me\n")
    init_results_file(str(path))
    assert path.read_text() == "keep,me\n"


def test_log_result_simple_row_roundtrip(tmp_path):
    path = tmp_path / "r.csv"
    init_results_file(str(path), include_meta_features=False)
    log_result(str(path), "mlp_small", 0.75, "ckpt.pth")
    assert _read(path) == [
        {"model_name": "mlp_small", "metric": "0.75", "checkpoint_path": "ckpt.pth"}
    ]


def test_log_result_meta_row_roundtrip(tmp_path):
    path = tmp_path / "r.csv"
    init_results_file(str(path))
    meta = {"n_samples": 150, "n_features": 4, "n_classes": 3}
    log_result(
        str(path),
        "logreg",
        0.9,
        "c.joblib",
        dataset_name="iris",
        meta_features=meta,
        model_family="classical",
        metric_name="accuracy",
    )
    row = _read(path)[0]
    assert row["dataset_name"] == "iris"
    assert row["model_name"] == "logreg"
    assert row["family"] == "classical"
    assert row["metric_name"] == "accuracy"
    assert row["metric"] == "0.9"
    assert row["checkpoint_path"] == "c.joblib"
    assert row["n_samples"] == "150"
    assert row["n_features"] == "4"
    assert row["n_classes"] == "3"
    assert row["n_numeric"] == ""
    assert row["avg_feature_variance"] == ""


def test_init_results_file_with_preprocessing_header_and_idempotent(tmp_path):
    path = tmp_path / "p.csv"
    init_results_file_with_preprocessing(str(path))
    first = path.read_text()
    init_results_file_with_preprocessing(str(path))
    assert path.read_text() == first
    header = first.strip().split(",")
    assert header[0] == "run_id"
    assert header[-1] == "seed"
    assert len(header) == 19


def test_log_result_with_preprocessing_row_roundtrip(tmp_path):
    path = tmp_path / "p.csv"
    init_results_file_with_preprocessing(str(path))
    candidate = SimpleNamespace(family="neural", model_name="mlp_small")
    descriptor = SimpleNamespace(
        name="iris",
        n_samples=150,
        n_features=4,
        target_type="classification",
        n_classes=3,
        meta={"class_imbalance": 1.0},
    )
    prep = SimpleNamespace(
        name="standard",
        numeric_imputer="median",
        scaler="standard",
        categorical_imputer="most_frequent",
        encoder="onehot",
        text_mode="none",
    )
    log_result_with_preprocessing(
        str(path), candidate, 0.8, "k.pth", descriptor, prep, {"metric_name": "accuracy"}, seed=5
    )
    row = _read(path)[0]
    assert len(row["run_id"]) == 8
    assert row["dataset_name"] == "iris"
    assert row["n_samples"] == "150"
    assert row["target_type"] == "classification"
    assert row["class_imbalance"] == "1.0"
    assert row["preprocessing_name"] == "standard"
    assert row["scaler"] == "standard"
    assert row["encoder"] == "onehot"
    assert row["model_family"] == "neural"
    assert row["model_name"] == "mlp_small"
    assert row["metric_name"] == "accuracy"
    assert row["metric"] == "0.8"
    assert row["checkpoint_path"] == "k.pth"
    assert row["seed"] == "5"


def test_log_result_with_preprocessing_defaults_to_blank_fields(tmp_path):
    path = tmp_path / "p.csv"
    init_results_file_with_preprocessing(str(path))
    candidate = SimpleNamespace(preprocessing_name="fallback_prep", model_name="m")
    log_result_with_preprocessing(
        str(path), candidate, 0.1, "k.pth", SimpleNamespace(), SimpleNamespace(), {}
    )
    row = _read(path)[0]
    assert row["preprocessing_name"] == "fallback_prep"
    assert row["dataset_name"] == ""
    assert row["model_family"] == ""
    assert row["model_name"] == "m"
    assert row["metric_name"] == ""
    assert row["seed"] == ""
