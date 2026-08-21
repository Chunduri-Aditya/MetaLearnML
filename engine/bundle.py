"""Persistence for complete raw-row-to-prediction pipeline bundles."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from engine.candidate_evaluator import FittedCandidate
from engine.meta_features import META_FEATURE_SCHEMA_VERSION
from models.model_zoo import MODEL_REGISTRY, build_model

PIPELINE_BUNDLE_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@dataclass
class LoadedPipelineBundle:
    path: Path
    manifest: dict[str, Any]
    preprocessor: Any
    model: Any
    label_encoder: Any = None
    device: str = "cpu"

    def _transform(self, frame: pd.DataFrame) -> torch.Tensor:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Bundle prediction requires a pandas DataFrame with raw columns")
        expected = self.manifest["input_columns"]
        normalized = frame.copy()
        normalized.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
        missing = sorted(set(expected) - set(normalized.columns))
        if missing:
            raise ValueError(f"Prediction input is missing required columns: {missing}")
        ordered = normalized.loc[:, expected]
        transformed = (
            ordered.to_numpy()
            if self.preprocessor is None
            else self.preprocessor.transform(ordered)
        )
        return torch.from_numpy(np.asarray(transformed, dtype=np.float32))

    @torch.no_grad()
    def predict_outputs(self, frame: pd.DataFrame) -> torch.Tensor:
        features = self._transform(frame)
        backend = self.manifest["backend"]
        if backend == "torch":
            features = features.to(self.device)
        output = self.model(features)
        return output.detach().cpu()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.manifest["task_type"] == "regression":
            raise ValueError("Regression bundles do not provide class probabilities")
        return torch.softmax(self.predict_outputs(frame), dim=1).numpy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        output = self.predict_outputs(frame)
        if self.manifest["task_type"] == "regression":
            return output.numpy().reshape(-1)
        indices = torch.argmax(output, dim=1).numpy()
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(indices)
        classes = self.manifest.get("label_classes")
        if classes is None:
            return indices
        return np.asarray(classes, dtype=object)[indices]


def save_pipeline_bundle(
    fitted: FittedCandidate,
    path: str | Path,
    *,
    task_type: str,
    metric_name: str,
    label_encoder: Any = None,
    selection_cv_mean: float | None = None,
    selection_cv_std: float | None = None,
    test_score: float | None = None,
) -> Path:
    bundle_path = Path(path)
    bundle_path.mkdir(parents=True, exist_ok=True)
    backend = MODEL_REGISTRY[fitted.candidate.model_identity]["backend"]
    preprocessor_name = "preprocessor.joblib"
    if fitted.preprocessor is None:
        preprocessor_name = None
    else:
        joblib.dump(fitted.preprocessor, bundle_path / preprocessor_name)
    transformed_dim = None
    if fitted.preprocessor is not None and hasattr(fitted.preprocessor, "get_feature_names_out"):
        transformed_dim = len(fitted.preprocessor.get_feature_names_out())
    elif hasattr(fitted.model, "in_dim"):
        transformed_dim = int(fitted.model.in_dim)

    if backend == "sklearn":
        model_file = "model.joblib"
        joblib.dump(fitted.model, bundle_path / model_file)
    else:
        model_file = "model_state.pt"
        torch.save(fitted.model.state_dict(), bundle_path / model_file)

    label_classes = None
    label_encoder_file = None
    if label_encoder is not None:
        label_classes = [_json_value(value) for value in label_encoder.classes_]
        label_encoder_file = "label_encoder.joblib"
        joblib.dump(label_encoder, bundle_path / label_encoder_file)
    out_dim = 1 if task_type == "regression" else len(label_classes or [])
    if not out_dim and hasattr(fitted.model, "out_dim"):
        out_dim = int(fitted.model.out_dim)
    manifest = {
        "bundle_schema_version": PIPELINE_BUNDLE_SCHEMA_VERSION,
        "meta_feature_schema_version": META_FEATURE_SCHEMA_VERSION,
        "candidate_schema_version": 2,
        "candidate_id": fitted.candidate.candidate_id,
        "model_identity": fitted.candidate.model_identity,
        "model_family": fitted.candidate.model_family,
        "backend": backend,
        "preprocessing_name": fitted.candidate.preprocessing.name,
        "preprocessing": fitted.candidate.preprocessing.to_component_dict(),
        "training": asdict(fitted.candidate.training),
        "fitted_epochs": fitted.fitted_epochs,
        "task_type": task_type,
        "metric_name": metric_name,
        "selection_cv_mean": selection_cv_mean,
        "selection_cv_std": selection_cv_std,
        "test_score": test_score,
        "input_columns": fitted.input_columns,
        "input_dtypes": fitted.input_dtypes,
        "numeric_columns": fitted.numeric_columns,
        "categorical_columns": fitted.categorical_columns,
        "transformed_input_dim": transformed_dim,
        "output_dim": out_dim,
        "label_classes": label_classes,
        "label_encoder_file": label_encoder_file,
        "preprocessor_file": preprocessor_name,
        "model_file": model_file,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    (bundle_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return bundle_path


def load_pipeline_bundle(path: str | Path, *, device: str = "cpu") -> LoadedPipelineBundle:
    bundle_path = Path(path)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pipeline bundle manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bundle_schema_version") != PIPELINE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported pipeline bundle schema {manifest.get('bundle_schema_version')!r}; "
            f"expected {PIPELINE_BUNDLE_SCHEMA_VERSION}"
        )
    preprocessor_file = manifest.get("preprocessor_file")
    preprocessor = joblib.load(bundle_path / preprocessor_file) if preprocessor_file else None
    label_encoder_file = manifest.get("label_encoder_file")
    label_encoder = joblib.load(bundle_path / label_encoder_file) if label_encoder_file else None
    if manifest["backend"] == "sklearn":
        model = joblib.load(bundle_path / manifest["model_file"])
        model.eval()
    else:
        model = build_model(
            manifest["model_identity"],
            int(manifest["transformed_input_dim"]),
            int(manifest["output_dim"]),
        )
        state = torch.load(bundle_path / manifest["model_file"], map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    return LoadedPipelineBundle(
        bundle_path,
        manifest,
        preprocessor,
        model,
        label_encoder=label_encoder,
        device=device,
    )
