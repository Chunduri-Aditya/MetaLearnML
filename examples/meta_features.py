"""
Extract meta-features from datasets to characterize them.
These features are used by the meta-model to predict which configs will work well.
"""

from collections import Counter

import numpy as np
import pandas as pd


def compute_meta_features_from_csv(path: str, label_col: str) -> dict:
    """
    Compute meta-features from a CSV dataset.

    Args:
        path: Path to CSV file
        label_col: Name of label column

    Returns:
        Dictionary of meta-features
    """
    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")

    y = df[label_col]
    X = df.drop(columns=[label_col])

    n_samples = len(df)
    n_features = X.shape[1]

    # Numeric vs categorical features
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    n_numeric = len(num_cols)
    n_categorical = len(cat_cols)

    # Class balance
    counts = Counter(y)
    n_classes = len(counts)
    if n_classes > 0:
        freq = np.array(list(counts.values())) / n_samples
        imbalance = float(freq.max() - freq.min()) if n_classes > 1 else 0.0
    else:
        imbalance = 0.0

    # Additional features (can be extended)
    avg_feature_variance = (
        float(X.select_dtypes(include=[np.number]).var().mean()) if n_numeric > 0 else 0.0
    )

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_classes": n_classes,
        "class_imbalance": imbalance,
        "avg_feature_variance": avg_feature_variance,
    }


def compute_meta_features_from_dataset(dataset, in_dim: int, out_dim: int, n_samples: int) -> dict:
    """
    Compute meta-features from an already-loaded dataset.
    Used when dataset is already in memory.

    Args:
        dataset: PyTorch dataset
        in_dim: Input dimension
        out_dim: Number of classes
        n_samples: Number of samples

    Returns:
        Dictionary of meta-features
    """
    # Estimate class imbalance from a sample
    # For large datasets, we sample; for small ones, use all
    sample_size = min(1000, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)

    labels = []
    for idx in sample_indices:
        _, label = dataset[idx]
        labels.append(label.item() if hasattr(label, "item") else int(label))

    counts = Counter(labels)
    n_classes = len(counts)
    if n_classes > 0:
        freq = np.array(list(counts.values())) / len(labels)
        imbalance = float(freq.max() - freq.min()) if n_classes > 1 else 0.0
    else:
        imbalance = 0.0

    return {
        "n_samples": n_samples,
        "n_features": in_dim,
        "n_numeric": in_dim,  # Assume numeric for loaded datasets
        "n_categorical": 0,
        "n_classes": out_dim,
        "class_imbalance": imbalance,
        "avg_feature_variance": 0.0,  # Would need to compute from data
    }
