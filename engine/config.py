"""
Configuration schema for AutoML runs.
Centralizes all configuration to avoid duplication.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class AutoMLConfig:
    """
    Complete configuration for an AutoML run.
    All menu options and CLI arguments should construct this.
    """

    # Dataset
    csv_path: str
    label_col: str | None = None
    dataset_name: str | None = None

    # Task & Metric
    task_type: str | None = None  # Auto-detected if None
    metric_name: str | None = None  # Auto-selected if None
    selection_metric: str | None = None
    target_metric: float | None = None  # Stop when reached

    # Search Configuration
    mode: Literal["basic", "intelligent"] = "intelligent"
    max_candidates: int = 20
    top_k_preprocessing: int = 3
    use_meta_learning: bool = True
    use_proxy_ranking: bool = True
    ordering_mode: Literal["registry", "random", "proxy", "meta", "proxy_meta"] | None = None
    exhaustive: bool = False

    # Training Configuration
    test_ratio: float = 0.2
    cv_folds: int = 5
    cv_repeats: int = 1
    random_state: int = 42
    refit_on_development: bool = True
    train_ratio: float | None = None  # Deprecated compatibility alias
    device: str = "cpu"  # "auto", "cpu", "cuda", "mps"

    # Output Configuration
    results_csv: str = "experiments/results.csv"
    ckpt_dir: str = "saved_models"
    report_dir: str = "experiments/reports"

    # Advanced
    early_stopping_patience: int = 5
    max_epochs: int = 20
    learning_rate: float = 1e-3
    n_jobs: int = -1  # Parallel execution (-1 = use all cores)

    def __post_init__(self):
        """Validate and normalize config."""
        # Validate paths
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        if self.train_ratio is not None:
            warnings.warn(
                "AutoMLConfig.train_ratio is deprecated; use test_ratio instead",
                DeprecationWarning,
                stacklevel=2,
            )
            translated = 1.0 - self.train_ratio
            if self.test_ratio != 0.2 and abs(self.test_ratio - translated) > 1e-12:
                raise ValueError("train_ratio and test_ratio specify conflicting splits")
            self.test_ratio = translated
        if not 0.0 < self.test_ratio < 1.0:
            raise ValueError("test_ratio must be within (0, 1)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.cv_repeats < 1:
            raise ValueError("cv_repeats must be at least 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.selection_metric and self.metric_name and self.selection_metric != self.metric_name:
            raise ValueError("selection_metric and metric_name must match when both are supplied")
        if self.selection_metric and not self.metric_name:
            self.metric_name = self.selection_metric
        if self.ordering_mode is None:
            if self.use_proxy_ranking and self.use_meta_learning:
                self.ordering_mode = "proxy_meta"
            elif self.use_proxy_ranking:
                self.ordering_mode = "proxy"
            elif self.use_meta_learning:
                self.ordering_mode = "meta"
            else:
                self.ordering_mode = "registry"

        # Normalize device
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set random state if provided
        from utils.seed_utils import set_seed

        set_seed(self.random_state)
