"""
Configuration schema for AutoML runs.
Centralizes all configuration to avoid duplication.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class AutoMLConfig:
    """
    Complete configuration for an AutoML run.
    All menu options and CLI arguments should construct this.
    """
    # Dataset
    csv_path: str
    label_col: Optional[str] = None
    dataset_name: Optional[str] = None
    
    # Task & Metric
    task_type: Optional[str] = None  # Auto-detected if None
    metric_name: Optional[str] = None  # Auto-selected if None
    target_metric: Optional[float] = None  # Stop when reached
    
    # Search Configuration
    mode: Literal["basic", "intelligent"] = "intelligent"
    max_candidates: int = 20
    top_k_preprocessing: int = 3
    use_meta_learning: bool = True
    use_proxy_ranking: bool = True
    
    # Training Configuration
    train_ratio: float = 0.8
    random_state: Optional[int] = None
    device: str = "auto"  # "auto", "cpu", "cuda"
    
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
        
        # Normalize device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set random state if provided
        if self.random_state is not None:
            from utils.seed_utils import set_seed
            set_seed(self.random_state)

