"""
Online/adaptive controller for continual learning.
Allows the system to fine-tune on new data and adapt over time.
"""

import random
from collections import deque
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

from engine.evaluator import accuracy_metric
from engine.predictor import BestModelPredictor
from engine.trainer import train_one_model
from models.model_zoo import MODEL_ZOO, build_model


class OnlineController:
    """
    Controller for online/adaptive model selection and fine-tuning.

    This allows the system to:
    - Fine-tune the current best model on new incoming data
    - Periodically challenge with alternative models
    - Switch to better models if they outperform the current best
    """

    def __init__(
        self,
        results_csv: str,
        in_dim: int,
        out_dim: int,
        device: str | None = None,
        buffer_size: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = BestModelPredictor(results_csv, in_dim, out_dim, self.device)
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Recent data buffer for evaluation & challengers
        self.buffer_size = buffer_size
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)

    def fine_tune_on_stream(
        self,
        data_stream: Iterable[tuple[torch.Tensor, torch.Tensor]],
        base_config: dict,
        challenge_every: int = 5,
        challenge_epochs: int = 3,
        metric_fn=accuracy_metric,
        promote_margin: float = 0.01,
    ):
        """
        Fine-tune the current best model on a stream of new data.

        Args:
            data_stream: Iterable of (X_batch, y_batch) tuples
            base_config: Training configuration dict
            challenge_every: After how many batches to consider testing another model
            challenge_epochs: Number of epochs to train challenger models
            metric_fn: Metric function to compare models
            promote_margin: Minimum improvement required to promote challenger
        """
        batch_count = 0

        for X_batch, y_batch in data_stream:
            batch_count += 1

            # Add to buffer
            self._add_to_buffer(X_batch, y_batch)

            # Fine-tune current best on this batch (1 epoch)
            self._fine_tune_batch(X_batch, y_batch, base_config)

            # Occasionally run a challenger
            if batch_count % challenge_every == 0 and len(self.buffer_x) > 0:
                self._challenge_with_new_model(
                    base_config=base_config,
                    epochs=challenge_epochs,
                    metric_fn=metric_fn,
                    promote_margin=promote_margin,
                )

    def _add_to_buffer(self, X_batch, y_batch):
        """Add batch to the buffer (automatically manages size via deque maxlen)."""
        X_batch = X_batch.detach().cpu()
        y_batch = y_batch.detach().cpu()
        for x, y in zip(X_batch, y_batch, strict=True):
            self.buffer_x.append(x)
            self.buffer_y.append(y)

    def _fine_tune_batch(self, X_batch, y_batch, config):
        """
        Fine-tune the current best model on a single batch.
        Uses minimal epochs (1) to avoid overfitting to a single batch.
        """
        # Create a tiny DataLoader for this batch
        ds = TensorDataset(X_batch, y_batch)
        dl = DataLoader(ds, batch_size=min(len(ds), 32), shuffle=True)

        # Fine-tune with minimal epochs
        fine_tune_config = {**config, "epochs": 1, "patience": None}

        train_one_model(
            model=self.predictor.model,
            train_loader=dl,
            val_loader=dl,  # Use same data for validation (quick check)
            device=self.device,
            config=fine_tune_config,
        )

    def _challenge_with_new_model(
        self, base_config: dict, epochs: int, metric_fn, promote_margin: float
    ):
        """
        Challenge the current best model with an alternative from the zoo.

        Strategy:
        - Pick a challenger model different from current best
        - Train it briefly on buffered data
        - Compare both models on the buffer
        - If challenger is better by margin, promote it
        """
        # Pick a challenger different from current model
        current_name = self.predictor.model_name
        candidates = [m for m in MODEL_ZOO if m != current_name]
        if not candidates:
            return

        challenger_name = random.choice(candidates)

        print(f"\n[ONLINE] Challenging current model ({current_name}) with {challenger_name}")

        # Build challenger
        challenger = build_model(challenger_name, self.in_dim, self.out_dim).to(self.device)

        # Build a small dataset from buffer
        X_buf = torch.stack(list(self.buffer_x)).to(self.device)
        y_buf = torch.stack(list(self.buffer_y)).to(self.device)

        ds = TensorDataset(X_buf, y_buf)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        # Local config (short training)
        cfg = base_config.copy()
        cfg["epochs"] = epochs
        cfg["patience"] = None  # No early stopping for challengers

        # Train challenger on buffered data
        metric_challenger, state_challenger = train_one_model(
            challenger,
            dl,
            dl,  # Using buffer as both train/val here (simple baseline)
            self.device,
            cfg,
        )

        # Evaluate current model on the same buffer
        self.predictor.model.eval()
        with torch.no_grad():
            logits_current = self.predictor.model(X_buf)
        metric_current = metric_fn(logits_current.cpu(), y_buf.cpu())

        print(f"[ONLINE] Current metric:   {metric_current:.4f}")
        print(f"[ONLINE] Challenger metric: {metric_challenger:.4f}")

        # Promote challenger if clearly better
        if metric_challenger > metric_current + promote_margin:
            print(f"[ONLINE] ✓ Promoting {challenger_name} to new best model!")
            self.predictor.model_name = challenger_name
            self.predictor.model.load_state_dict(state_challenger)
            self.predictor.model.to(self.device)
            self.predictor.model.eval()
        else:
            print(f"[ONLINE] ✗ Challenger {challenger_name} rejected (not better enough)")

    def predict(self, x: torch.Tensor):
        """Convenience method to get predictions from current best model."""
        return self.predictor.predict_classes(x)

    def predict_proba(self, x: torch.Tensor):
        """Convenience method to get probabilities from current best model."""
        return self.predictor.predict_proba(x)
