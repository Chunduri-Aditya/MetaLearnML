import csv
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
from datetime import datetime


def init_results_file(path: str, include_meta_features: bool = True):
    """Initialize results CSV with headers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            if include_meta_features:
                writer.writerow([
                    "dataset_name", "model_name", "family", "metric_name", "metric", "checkpoint_path",
                    "n_samples", "n_features", "n_numeric", "n_categorical", 
                    "n_classes", "class_imbalance", "avg_feature_variance"
                ])
            else:
                writer.writerow(["model_name", "metric", "checkpoint_path"])


def log_result(
    path: str, 
    model_name: str, 
    metric: float, 
    ckpt_path: str,
    dataset_name: Optional[str] = None,
    meta_features: Optional[Dict] = None,
    model_family: Optional[str] = None,
    metric_name: Optional[str] = None,
):
    """
    Log experiment result with optional meta-features.
    
    Args:
        path: Path to results CSV
        model_name: Name of the model
        metric: Performance metric value
        ckpt_path: Path to checkpoint
        dataset_name: Name of dataset (for meta-learning)
        meta_features: Dictionary of meta-features (for meta-learning)
        model_family: Family of the model (e.g., "neural_supervised")
        metric_name: Name of the metric (e.g., "accuracy", "f1")
    """
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if meta_features is not None and dataset_name is not None:
            writer.writerow([
                dataset_name,
                model_name,
                model_family or "",
                metric_name or "",
                metric,
                ckpt_path,
                meta_features.get("n_samples", ""),
                meta_features.get("n_features", ""),
                meta_features.get("n_numeric", ""),
                meta_features.get("n_categorical", ""),
                meta_features.get("n_classes", ""),
                meta_features.get("class_imbalance", ""),
                meta_features.get("avg_feature_variance", ""),
            ])
        else:
            # Backward compatibility: simple logging
            writer.writerow([model_name, metric, ckpt_path])


def init_results_file_with_preprocessing(path: str):
    """Initialize results CSV with preprocessing columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id",
                "dataset_name",
                "n_samples",
                "n_features",
                "target_type",
                "n_classes",
                "class_imbalance",
                "preprocessing_name",
                "numeric_imputer",
                "scaler",
                "categorical_imputer",
                "encoder",
                "text_mode",
                "model_family",
                "model_name",
                "metric_name",
                "metric_value",
                "checkpoint_path",
                "seed",
            ])


def log_result_with_preprocessing(
    results_csv: str,
    candidate,  # SearchCandidate - imported locally to avoid circular import
    metric: float,
    ckpt_path: str,
    descriptor,  # DatasetDescriptor - imported locally
    preprocessing_config,  # PreprocessingConfig - imported locally
    base_config: Dict[str, Any],
    seed: Optional[int] = None,
):
    """
    Log experiment result with preprocessing information.
    
    Args:
        results_csv: Path to results CSV
        candidate: Search candidate (preprocessing + model)
        metric: Performance metric value
        ckpt_path: Path to checkpoint
        descriptor: Dataset descriptor
        preprocessing_config: Preprocessing configuration
        base_config: Base training configuration
        seed: Random seed (optional)
    """
    run_id = str(uuid.uuid4())[:8]  # Short UUID
    metric_name = base_config.get("metric_name", "")
    
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id,
            descriptor.name,
            descriptor.n_samples,
            descriptor.n_features,
            descriptor.target_type,
            descriptor.n_classes,
            descriptor.meta.get("class_imbalance", ""),
            preprocessing_config.name,
            preprocessing_config.numeric_imputer or "",
            preprocessing_config.scaler or "",
            preprocessing_config.categorical_imputer or "",
            preprocessing_config.encoder or "",
            preprocessing_config.text_mode or "",
            candidate.family,
            candidate.model_name,
            metric_name,
            metric,
            ckpt_path,
            seed or "",
        ])

