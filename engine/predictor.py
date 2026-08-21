# engine/predictor.py

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models.model_zoo import build_model


def _load_best_from_csv(results_csv: str) -> tuple[str, float, str]:
    """
    Reads the results.csv and returns (model_name, metric, ckpt_path)
    for the row with the best metric.
    """
    best_row: dict | None = None
    path = Path(results_csv)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_str = row.get("metric", row.get("metric_value", ""))
            if metric_str in ("", None):
                continue
            metric = float(metric_str)
            if best_row is None:
                best_row = row
                best_row["_metric"] = metric
            elif metric > float(best_row["_metric"]):
                best_row = row
                best_row["_metric"] = metric

    if best_row is None:
        raise RuntimeError(f"No rows found in {results_csv}")

    checkpoint_path = best_row.get("checkpoint_path", best_row.get("ckpt_path", ""))
    if not checkpoint_path:
        raise RuntimeError(f"Checkpoint path missing in best row for {results_csv}")
    return best_row["model_name"], float(best_row["_metric"]), checkpoint_path


class BestModelPredictor:
    def __init__(
        self,
        results_csv: str | None = None,
        in_dim: int | None = None,
        out_dim: int | None = None,
        device: str | None = None,
        *,
        bundle_path: str | None = None,
    ):
        self.bundle = None
        if bundle_path is not None:
            from engine.bundle import load_pipeline_bundle

            self.bundle = load_pipeline_bundle(bundle_path, device=device or "cpu")
            self.results_csv = results_csv
            self.in_dim = self.bundle.manifest["transformed_input_dim"]
            self.out_dim = self.bundle.manifest["output_dim"]
            self.device = device or "cpu"
            self.model_name = self.bundle.manifest["model_identity"]
            self.best_metric = self.bundle.manifest.get("test_score")
            self.ckpt_path = bundle_path
            self.model = self.bundle.model
            return
        if results_csv is None or in_dim is None or out_dim is None:
            raise TypeError(
                "Provide bundle_path for v2 raw-row prediction, or results_csv/in_dim/out_dim "
                "for the deprecated checkpoint adapter"
            )
        self.results_csv = results_csv
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_name, metric, ckpt_path = _load_best_from_csv(results_csv)
        self.model_name = model_name
        self.best_metric = metric
        self.ckpt_path = ckpt_path

        # Build and load the model based on backend
        from models.model_zoo import MODEL_REGISTRY

        backend = MODEL_REGISTRY.get(model_name, {}).get("backend", "torch")

        if backend == "sklearn":
            # Sklearn model: load using joblib
            # Check for .joblib extension first, fallback to .pkl for backward compatibility
            if ckpt_path.endswith(".joblib"):
                sk_path = ckpt_path
            elif ckpt_path.endswith(".pkl"):
                sk_path = ckpt_path
            else:
                # Replace .pth with .joblib
                sk_path = ckpt_path.replace(".pth", ".joblib")

            # Load the entire wrapper (includes fitted sklearn model + scaler)
            self.model = joblib.load(sk_path)
            # Sklearn models don't need .to(device) but we can still call eval()
            self.model.eval()
        else:
            # PyTorch model: load state dict
            self.model = build_model(model_name, in_dim, out_dim)
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

    def _to_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input to tensor efficiently."""
        if isinstance(x, np.ndarray):
            # Use float32 for memory efficiency
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported type for x: {type(x)}")
        return x

    @torch.no_grad()
    def predict_logits(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits from the best model.
        Shape: [N, out_dim]
        """
        if self.bundle is not None:
            if not isinstance(x, pd.DataFrame):
                raise TypeError("A v2 pipeline bundle requires raw input as a pandas DataFrame")
            return self.bundle.predict_outputs(x)
        x = self._to_tensor(x).to(self.device)
        logits = self.model(x)
        return logits.detach().cpu()

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities over classes.
        """
        logits = self.predict_logits(x)
        probs = torch.softmax(logits, dim=1)
        return probs

    @torch.no_grad()
    def predict_classes(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Returns predicted class indices [N].
        """
        if self.bundle is not None and isinstance(x, pd.DataFrame):
            return self.bundle.predict(x)
        probs = self.predict_proba(x)
        preds = torch.argmax(probs, dim=1)
        return preds

    def predict_raw(self, frame: pd.DataFrame):
        """Predict labels or regression values from raw rows using a v2 bundle."""
        if self.bundle is None:
            raise RuntimeError("predict_raw requires a v2 bundle_path")
        return self.bundle.predict(frame)

    def predict_dataframe(self, frame: pd.DataFrame):
        """Explicit DataFrame-named alias for v2 raw-row prediction."""
        return self.predict_raw(frame)
