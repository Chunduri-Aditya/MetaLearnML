# engine/predictor.py

import csv
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import numpy as np
import joblib

from models.model_zoo import build_model


def _load_best_from_csv(results_csv: str) -> Tuple[str, float, str]:
    """
    Reads the results.csv and returns (model_name, metric, ckpt_path)
    for the row with the best metric.
    """
    best_row: Optional[dict] = None
    path = Path(results_csv)

    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = float(row["metric"])
            if best_row is None or metric > float(best_row["metric"]):
                best_row = row

    if best_row is None:
        raise RuntimeError(f"No rows found in {results_csv}")

    return best_row["model_name"], float(best_row["metric"]), best_row["checkpoint_path"]


class BestModelPredictor:
    def __init__(
        self,
        results_csv: str,
        in_dim: int,
        out_dim: int,
        device: Optional[str] = None,
    ):
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
            if ckpt_path.endswith('.joblib'):
                sk_path = ckpt_path
            elif ckpt_path.endswith('.pkl'):
                sk_path = ckpt_path
            else:
                # Replace .pth with .joblib
                sk_path = ckpt_path.replace('.pth', '.joblib')
            
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

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
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
    def predict_logits(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Returns raw logits from the best model.
        Shape: [N, out_dim]
        """
        x = self._to_tensor(x).to(self.device)
        logits = self.model(x)
        return logits.detach().cpu()

    @torch.no_grad()
    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Returns softmax probabilities over classes.
        """
        logits = self.predict_logits(x)
        probs = torch.softmax(logits, dim=1)
        return probs

    @torch.no_grad()
    def predict_classes(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Returns predicted class indices [N].
        """
        probs = self.predict_proba(x)
        preds = torch.argmax(probs, dim=1)
        return preds

