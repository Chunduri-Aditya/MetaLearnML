# Dataset loading utilities
# This can be extended to load real datasets from CSV, etc.

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split


def infer_task_type(y: pd.Series) -> str:
    """
    Infer task type from target column.

    Args:
        y: Target series

    Returns:
        "binary_classification", "multiclass_classification", or "regression"
    """
    n_samples = len(y)
    n_unique = y.nunique(dropna=True)
    is_float = pd.api.types.is_float_dtype(y)
    is_int = pd.api.types.is_integer_dtype(y)

    # Float dtype → regression
    if is_float:
        task_type = "regression"
    # Integer with many unique values → regression
    elif is_int:
        unique_ratio = n_unique / max(1, n_samples)
        if n_unique >= 50 and unique_ratio >= 0.05:
            task_type = "regression"
        elif n_unique == 2:
            task_type = "binary_classification"
        else:
            task_type = "multiclass_classification"
    # String or other → classification
    else:
        if n_unique == 2:
            task_type = "binary_classification"
        else:
            task_type = "multiclass_classification"

    print(f"[INFO] Inferred task type: {task_type} (n_samples={n_samples}, n_unique={n_unique})")
    return task_type


@dataclass
class DatasetDescriptor:
    """
    Unified representation of a dataset with metadata.
    This abstraction allows the system to work with any dataset in a consistent way.
    """

    name: str
    X: torch.Tensor  # shape [n_samples, n_features]
    y: torch.Tensor  # shape [n_samples]
    feature_names: list[str]
    target_name: str
    column_types: dict[str, str]  # e.g. {"age": "numeric", "gender": "categorical", "text": "text"}
    target_type: str  # "binary", "multiclass", (later: "regression")
    n_samples: int
    n_features: int
    n_classes: int | None  # None if regression
    meta: dict[str, Any]  # extra stuff (class_imbalance, variances, etc.)

    def to_tensor_dataset(self) -> TensorDataset:
        """Convert to PyTorch TensorDataset for compatibility."""
        return TensorDataset(self.X, self.y)


def create_dummy_dataset(n_samples=1000, n_features=20, n_classes=3):
    """Create a dummy dataset for testing."""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(X, y)


def split_dataset(dataset, train_ratio=0.8):
    """Split dataset into train and validation sets."""
    n_train = int(train_ratio * len(dataset))
    n_val = len(dataset) - n_train
    return random_split(dataset, [n_train, n_val])


def create_data_loaders(train_dataset, val_dataset, train_batch_size=32, val_batch_size=64):
    """Create DataLoaders for train and validation sets."""
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
    return train_loader, val_loader


def load_csv_with_fallback(path: str) -> pd.DataFrame:
    """
    Load CSV with robust handling of missing/ugly headers.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with clean column names
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # First attempt: read with default settings
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {path}: {e}") from e

    # Drop obvious junk index columns
    junk_cols = [c for c in df.columns if str(c).lower() in ["unnamed: 0", "index", "id"]]
    if junk_cols:
        df = df.drop(columns=junk_cols)
        print(f"[INFO] Dropped index columns: {junk_cols}")

    # Check if column names are all unnamed or ints
    has_real_header = not all(
        str(col).startswith("Unnamed") or isinstance(col, int) or not str(col).strip()
        for col in df.columns
    )

    if not has_real_header:
        # Re-read with header=None and create placeholder names
        df = pd.read_csv(path, header=None, low_memory=False)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        print("[INFO] No meaningful headers detected. Using placeholder names: col_0, col_1, ...")
    else:
        # Normalize existing headers (remove spaces, handle empty strings)
        new_cols = []
        for i, c in enumerate(df.columns):
            col_str = str(c).strip()
            if not col_str or col_str.lower().startswith("unnamed:"):
                new_cols.append(f"col_{i}")
            else:
                # Replace spaces with underscores, normalize
                new_cols.append(col_str.replace(" ", "_"))
        df.columns = new_cols

    return df


def load_tabular_dataset(
    path: str,
    label_col: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Unified dataset loader that infers task type and returns structured data.

    Args:
        path: Path to CSV file
        label_col: Column name for labels. If None, auto-detects (defaults to last column)
        dataset_name: Name for the dataset. If None, uses filename.

    Returns:
        Dictionary with keys:
        - "X_df": DataFrame with features
        - "y": Series or array with target
        - "task_type": "regression" | "binary_classification" | "multiclass_classification"
        - "meta": Dictionary with metadata
    """
    raw = load_raw_tabular_dataset(path, label_col=label_col, dataset_name=dataset_name)
    X_df = raw["X_df"]
    y_series = raw["y_series"]
    task_type = raw["task_type"]
    dataset_name = raw["dataset_name"]

    # Process target based on task type
    label_encoder = None
    if task_type == "regression":
        # Keep as float
        y = y_series.values.astype(np.float32)
    else:
        # Classification: encode to 0-indexed integers
        if pd.api.types.is_numeric_dtype(y_series):
            # Check if needs remapping
            unique_vals = y_series.unique()
            min_val, max_val = unique_vals.min(), unique_vals.max()
            n_unique = len(unique_vals)

            if min_val != 0 or max_val >= n_unique:
                # Remap to 0-indexed
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y_series.values).astype(np.int64)
            else:
                y = y_series.values.astype(np.int64)
        else:
            # String labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_series.values).astype(np.int64)

    meta = compute_dataset_metadata(
        X_df,
        y,
        task_type,
        target_name=raw["target_name"],
        label_encoder=label_encoder,
    )
    meta["dataset_id"] = raw["dataset_id"]
    meta["dataset_version"] = raw["dataset_version"]
    meta["dataset_checksum"] = raw["dataset_checksum"]

    return {
        "X_df": X_df,
        "y": y,
        "task_type": task_type,
        "meta": meta,
        "dataset_name": dataset_name,
    }


def load_raw_tabular_dataset(
    path: str,
    label_col: str | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Load raw feature/target frames without fitting target or feature transforms."""
    import hashlib

    source = Path(path)
    dataset_name = dataset_name or source.stem
    df = load_csv_with_fallback(source)
    if label_col is None:
        label_col = _auto_detect_label_column(df)
        if label_col is None:
            label_col = df.columns[-1]
            print(f"[INFO] Auto-detected label column (last column): {label_col}")
        else:
            print(f"[INFO] Auto-detected label column: {label_col}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")
    X_df = df.drop(columns=[label_col])
    y_series = df[label_col]
    resolved_task = task_type or infer_task_type(y_series)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "X_df": X_df,
        "y_series": y_series,
        "task_type": resolved_task,
        "dataset_name": dataset_name,
        "target_name": label_col,
        "dataset_id": f"local:{source.resolve()}",
        "dataset_version": "1",
        "dataset_checksum": checksum,
    }


def infer_feature_types(X_df: pd.DataFrame) -> dict[str, str]:
    """Infer feature types from a feature-only development frame."""
    types: dict[str, str] = {}
    for column in X_df.columns:
        series = X_df[column]
        if pd.api.types.is_numeric_dtype(series):
            types[column] = "numeric"
            continue
        non_missing = series.dropna().astype(str)
        if non_missing.empty:
            types[column] = "categorical"
            continue
        average_length = float(non_missing.str.len().mean())
        unique_count = int(non_missing.nunique())
        unique_ratio = unique_count / max(len(non_missing), 1)
        average_words = float(non_missing.str.split().str.len().mean())
        if unique_ratio > 0.2 and average_length > 12 and average_words >= 2:
            types[column] = "text"
        else:
            types[column] = "categorical"
    return types


def compute_dataset_metadata(
    X_df: pd.DataFrame,
    y: Any,
    task_type: str,
    *,
    target_name: str | None = None,
    label_encoder: Any = None,
    column_types: dict[str, str] | None = None,
    reject_text: bool = False,
) -> dict[str, Any]:
    """Compute search metadata from development rows only."""
    column_types = column_types or infer_feature_types(X_df)
    numeric_cols = [c for c in X_df.columns if column_types.get(c) == "numeric"]
    categorical_cols = [c for c in X_df.columns if column_types.get(c) == "categorical"]
    text_cols = [c for c in X_df.columns if column_types.get(c) == "text"]
    if reject_text and text_cols:
        raise ValueError(
            "MetaLearnML v2 does not support inferred free-text features. "
            f"Remove or transform these columns before running: {text_cols}"
        )
    missing_fraction = float(X_df.isna().to_numpy().mean()) if X_df.size else 0.0
    if not np.isfinite(missing_fraction) or not 0.0 <= missing_fraction <= 1.0:
        raise ValueError(f"Invalid missing_value_fraction computed: {missing_fraction!r}")
    y_array = np.asarray(y)
    if task_type == "regression":
        n_classes = None
        class_imbalance = None
    else:
        values, counts = np.unique(y_array, return_counts=True)
        n_classes = int(len(values))
        proportions = counts / counts.sum() if counts.sum() else np.zeros_like(counts)
        class_imbalance = float(proportions.max() - proportions.min()) if n_classes > 1 else 0.0
    return {
        "n_samples": len(X_df),
        "n_features": len(X_df.columns),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "n_text": len(text_cols),
        "n_classes": n_classes,
        "class_imbalance": class_imbalance,
        "column_types": column_types,
        "feature_names": list(X_df.columns),
        "target_name": target_name,
        "label_encoder": label_encoder,
        "has_missing": bool(X_df.isna().any().any()),
        "missing_counts": X_df.isna().sum().to_dict(),
        "missing_value_fraction": missing_fraction,
    }


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: replace missing/ugly names with placeholders.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with normalized column names
    """
    new_cols = []
    for idx, col in enumerate(df.columns):
        col_str = str(col).strip()
        if not col_str or col_str.lower().startswith("unnamed:"):
            new_cols.append(f"col_{idx}")
        else:
            new_cols.append(col_str)
    df.columns = new_cols
    return df


def infer_column_types(df: pd.DataFrame, target_col: str) -> dict[str, str]:
    """
    Infer column types: numeric, categorical, or text.

    Args:
        df: DataFrame
        target_col: Name of target column (excluded from inference)

    Returns:
        Dictionary mapping column names to types
    """
    types = {}
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]

        # Heuristic: if dtype is number & n_unique > 20% of samples -> numeric
        if pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"
        else:
            # Try to detect "short text" vs "long text"
            avg_len = series.dropna().astype(str).str.len().mean()
            n_unique = series.nunique(dropna=True)
            if avg_len < 30 and n_unique < 0.5 * len(series):
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types


def analyze_target(y: pd.Series) -> dict[str, Any]:
    """
    Analyze target column to determine type and properties.

    Args:
        y: Target series

    Returns:
        Dictionary with target analysis
    """
    n_classes = y.nunique()
    value_counts = y.value_counts(normalize=True)
    imbalance = float(value_counts.max() - value_counts.min()) if n_classes > 1 else 0.0

    if n_classes == 2:
        target_type = "binary"
    else:
        target_type = "multiclass"

    return {
        "n_classes": n_classes,
        "target_type": target_type,
        "class_imbalance": imbalance,
        "class_distribution": value_counts.to_dict(),
    }


def _auto_detect_label_column(df: pd.DataFrame) -> str | None:
    """
    Try to auto-detect label column from common names and patterns.

    Uses a sophisticated scoring system to avoid false positives.

    Args:
        df: DataFrame

    Returns:
        Column name if found, None otherwise
    """
    import re

    # Patterns that strongly indicate a target column
    LIKELY_TARGET_PATTERNS = [
        r"^target$",
        r"^label$",
        r"^y$",
        r"^class$",
        r"^income$",
        r"^survived$",
        r"^price$",
        r"^score$",
        r"^output$",
        r".*_target$",
        r".*_label$",
        r"^yield$",
        r"^salary$",
        r"^cost$",
        r"^value$",
        r"^outcome$",
        r"^response$",
    ]

    # Patterns that indicate a feature (not a target)
    UNLIKELY_FEATURE_PATTERNS = [
        r"^id$",
        r".*_id$",
        r"^index$",
        r"^row_.*$",
        r"^name$",
        r"^date$",
        r"^time$",
        r"^timestamp$",
        r"^age$",
        r"^count$",
        r"^number$",
        r"^num$",
    ]

    cols = list(df.columns)
    n = len(df)

    # Step 1: Filter out obvious ID/feature columns
    candidate_cols = []
    for c in cols:
        name = c.lower().strip()
        # Skip if matches unlikely feature patterns
        if any(re.match(p, name) for p in UNLIKELY_FEATURE_PATTERNS):
            continue
        candidate_cols.append(c)

    if not candidate_cols:
        # All columns look like features, use all
        candidate_cols = cols

    # Step 2: Pattern-based priority (exact matches first)
    for pattern in LIKELY_TARGET_PATTERNS:
        for c in candidate_cols:
            if re.match(pattern, c.lower()):
                return c

    # Step 3: Cardinality-based scoring
    # We prefer columns with:
    # - Low cardinality (binary/multi-class classification)
    # - Moderate cardinality (regression, but not == n for pure IDs)
    best_col = None
    best_score = -1

    for c in candidate_cols:
        nunique = df[c].nunique(dropna=True)
        ratio = nunique / max(n, 1)

        # Skip degenerate cases
        if nunique == 1:  # Constant column
            continue
        if nunique == n:  # Unique ID column
            continue

        # Score function: lower cardinality gets higher score
        # But we also consider the ratio
        if nunique == 2:
            # Binary classification - very likely target
            score = 10.0
        elif nunique < 10:
            # Multi-class classification - likely target
            score = 5.0 / (1.0 + nunique)
        elif ratio < 0.3:
            # Regression with moderate cardinality - possible target
            score = 2.0 / (1.0 + nunique)
        else:
            # High cardinality - less likely to be target
            score = 1.0 / (1.0 + nunique)

        if score > best_score:
            best_score = score
            best_col = c

    if best_col is not None:
        return best_col

    # Step 4: Fallback - last column if not obviously a feature
    last_col = cols[-1]
    if last_col in candidate_cols:
        return last_col

    return None


def validate_dataset_size(n_samples: int, n_classes: int, val_ratio: float = 0.2):
    """
    Validate that dataset is large enough for train/val split.

    Args:
        n_samples: Total number of samples
        n_classes: Number of classes
        val_ratio: Validation set ratio

    Raises:
        ValueError: If dataset is too small
    """
    min_samples = max(10, n_classes * 5)  # At least 5 samples per class
    if n_samples < min_samples:
        raise ValueError(
            f"Dataset too small: {n_samples} samples. "
            f"Need at least {min_samples} samples (5 per class for {n_classes} classes)."
        )

    n_val = int(n_samples * val_ratio)
    if n_val < n_classes:
        raise ValueError(
            f"Validation set too small: {n_val} samples with {val_ratio * 100}% split. "
            f"Need at least {n_classes} samples in validation set (1 per class). "
            f"Try reducing validation ratio or getting more data."
        )


def load_csv_classification(
    path: str,
    label_col: str | None = None,
    feature_cols: list[str] | None = None,
    auto_encode_labels: bool = True,
    validate_size: bool = True,
    dataset_name: str | None = None,
) -> DatasetDescriptor:
    """
    Load a CSV for supervised classification and return DatasetDescriptor.

    Args:
        path: Path to CSV file
        label_col: Column name for labels. If None, tries to auto-detect.
        feature_cols: List of column names to use as features. If None, auto-detects all numeric columns except label_col
        auto_encode_labels: If True, automatically encode string labels to integers
        validate_size: If True, validate dataset size requirements
        dataset_name: Name for the dataset. If None, uses filename.

    Returns:
        DatasetDescriptor
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Use dataset name from path if not provided
    if dataset_name is None:
        dataset_name = path.stem

    # Use unified loader
    df = load_csv_with_fallback(path)

    # Auto-detect label column if not provided
    if label_col is None:
        label_col = _auto_detect_label_column(df)
        if label_col is None:
            raise ValueError(
                f"Could not auto-detect label column. Available columns: {list(df.columns)}\n"
                f"Please specify label_col parameter."
            )
        print(f"[INFO] Auto-detected label column: {label_col}")

    # Check label column exists
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in CSV. Available columns: {list(df.columns)}"
        )

    # Infer column types
    column_types = infer_column_types(df, label_col)

    # Analyze target
    target_analysis = analyze_target(df[label_col])

    # Auto-detect features if not specified
    if feature_cols is None:
        # Prefer numeric columns, but can handle mixed types
        numeric_cols = [
            c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        feature_cols = numeric_cols
        if len(feature_cols) == 0:
            # No numeric columns - will need preprocessing
            feature_cols = [c for c in df.columns if c != label_col]
            print(
                f"[INFO] No numeric columns found. Will use all non-target columns: {feature_cols}"
            )
        else:
            print(
                f"[INFO] Auto-detected {len(feature_cols)} numeric feature columns: {feature_cols}"
            )

    # Extract features - use float32 for memory efficiency
    # For now, only numeric features are extracted here
    # Mixed types will be handled by preprocessing
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_feature_cols) > 0:
        X = df[numeric_feature_cols].values.astype(np.float32, copy=False)
    else:
        # Fallback: create dummy numeric features (will be replaced by preprocessing)
        X = np.zeros((len(df), 1), dtype=np.float32)

    # Extract and encode labels if needed
    y_raw = df[label_col]

    # Always remap labels to 0-indexed classes, even if they're already integers
    # This handles cases where labels are large integers (e.g., 35028, 35412)
    if pd.api.types.is_numeric_dtype(y_raw):
        # Numeric labels - check if they need remapping
        unique_vals = y_raw.unique()
        min_val = unique_vals.min()
        max_val = unique_vals.max()
        n_unique = len(unique_vals)

        # If labels are not 0-indexed or have gaps, remap them
        if (
            min_val != 0
            or max_val >= n_unique
            or len(set(range(int(min_val), int(max_val) + 1))) != n_unique
        ):
            # Need to remap to 0-indexed
            y_cat = pd.Categorical(y_raw)
            y = y_cat.codes.astype(np.int64)
            print(
                f"[INFO] Remapped numeric labels to 0-indexed classes (was {int(min_val)}-{int(max_val)}, now 0-{n_unique - 1})"
            )
        else:
            # Already 0-indexed, use as-is
            y = y_raw.values.astype(np.int64)
    elif auto_encode_labels:
        # String labels - encode to integers
        y_cat = pd.Categorical(y_raw)
        y = y_cat.codes.astype(np.int64)
        if (y < 0).any():
            raise ValueError("Found unlabeled samples (NaN) in label column")
        print("[INFO] Encoded string labels to integer classes.")
    else:
        # Use as-is (shouldn't happen often)
        y = y_raw.values.astype(np.int64)

    # Final validation: ensure labels are 0-indexed
    y_min, y_max = int(y.min()), int(y.max())
    if y_min < 0:
        raise ValueError(f"Labels must be non-negative, found minimum: {y_min}")
    if y_max >= len(np.unique(y)):
        # This shouldn't happen after remapping, but double-check
        print("[WARN] Label range issue detected. Remapping again...")
        y_unique = np.unique(y)
        y_map = {val: idx for idx, val in enumerate(y_unique)}
        y = np.array([y_map[v] for v in y], dtype=np.int64)

    # Create tensors directly from arrays (more memory efficient)
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))

    n_samples = len(df)
    n_features = X.shape[1]
    n_classes = target_analysis["n_classes"]

    # Validate dataset size
    if validate_size:
        validate_dataset_size(n_samples, n_classes)

    # Build meta-features
    meta = {
        "class_imbalance": target_analysis["class_imbalance"],
        "class_distribution": target_analysis["class_distribution"],
        "has_missing": df[feature_cols].isnull().any().any() if feature_cols else False,
        "missing_counts": df[feature_cols].isnull().sum().to_dict() if feature_cols else {},
    }

    # Create descriptor
    descriptor = DatasetDescriptor(
        name=dataset_name,
        X=X_tensor,
        y=y_tensor,
        feature_names=feature_cols,
        target_name=label_col,
        column_types=column_types,
        target_type=target_analysis["target_type"],
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        meta=meta,
    )

    return descriptor


def print_dataset_insights(descriptor: DatasetDescriptor):
    """
    Print human-readable insights about a dataset.
    Used in interactive CLI to show what the system detected.
    """
    print("\n" + "=" * 60)
    print("Dataset Insights")
    print("=" * 60)
    print(f"Dataset: {descriptor.name}")
    print(f"Total samples: {descriptor.n_samples}")
    print(f"Total features: {descriptor.n_features}")
    print(f"Target: {descriptor.target_name} ({descriptor.target_type})")
    print(f"Number of classes: {descriptor.n_classes}")

    # Column type breakdown
    type_counts = {}
    for col_type in descriptor.column_types.values():
        type_counts[col_type] = type_counts.get(col_type, 0) + 1

    type_str = ", ".join([f"{count} {typ}" for typ, count in type_counts.items()])
    print(f"Column types: {type_str}")

    # Class imbalance warning
    imbalance = descriptor.meta.get("class_imbalance", 0.0)
    if imbalance > 0.3:
        print(f"\n⚠️  Warning: High class imbalance detected ({imbalance:.2f})")
        print("   Recommendation: Use F1 score instead of Accuracy")

    # Missing values
    missing_counts = descriptor.meta.get("missing_counts", {})
    if missing_counts:
        high_missing = {k: v for k, v in missing_counts.items() if v > 0}
        if high_missing:
            print("\nMissing values detected:")
            for col, count in high_missing.items():
                pct = (count / descriptor.n_samples) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")

    print("=" * 60 + "\n")


def load_csv_classification_legacy(
    path: str,
    label_col: str,
    feature_cols: list[str] | None = None,
    auto_encode_labels: bool = True,
    validate_size: bool = True,
) -> tuple[TensorDataset, int, int]:
    """
    Legacy function for backward compatibility.
    Returns (dataset, in_dim, num_classes) instead of DatasetDescriptor.
    """
    descriptor = load_csv_classification(
        path=path,
        label_col=label_col,
        feature_cols=feature_cols,
        auto_encode_labels=auto_encode_labels,
        validate_size=validate_size,
    )
    dataset = descriptor.to_tensor_dataset()
    return dataset, descriptor.n_features, descriptor.n_classes


def load_text_csv_classification(
    path: str,
    text_cols: list[str],
    label_col: str,
    max_features: int = 5000,
    validate_size: bool = True,
) -> tuple[TensorDataset, int, int]:
    """
    Load a CSV where features are text columns and label is a categorical column.
    - Concatenates the given text columns into a single string per row.
    - Builds TF-IDF vectors as numeric features.
    - Encodes labels to integer classes.

    Args:
        path: Path to CSV file
        text_cols: List of column names containing text features
        label_col: Column name for labels
        max_features: Maximum number of TF-IDF features to extract
        validate_size: If True, validate dataset size requirements

    Returns:
        (dataset, in_dim, num_classes)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Read CSV efficiently
    try:
        df = pd.read_csv(path, low_memory=False)
    except MemoryError:
        chunk_list = []
        for chunk in pd.read_csv(path, chunksize=10000, low_memory=False):
            chunk_list.append(chunk)
        df = pd.concat(chunk_list, ignore_index=True)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")

    for col in text_cols:
        if col not in df.columns:
            raise ValueError(f"Text column '{col}' not found in CSV.")

    # Fill missing text with empty strings (in-place for memory efficiency)
    df[text_cols] = df[text_cols].fillna("")

    # Concatenate selected text columns into one big string per row
    text_series = df[text_cols].agg(" ".join, axis=1)

    print(f"[INFO] Building TF-IDF features from columns: {text_cols}")
    print(f"[INFO] Using max_features={max_features}")

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_sparse = vectorizer.fit_transform(text_series)

    # Convert to dense array with float32 for memory efficiency
    X = X_sparse.astype(np.float32).toarray()

    # Encode label column to integer classes
    y_raw = df[label_col]
    y_cat = y_raw.astype("category")
    y = y_cat.cat.codes.values.astype(np.int64)
    print("[INFO] Encoded labels to integer classes.")
    print(f"[INFO] Label classes: {list(y_cat.cat.categories)}")

    # Create tensors directly from arrays
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    in_dim = X.shape[1]
    num_classes = len(np.unique(y))

    print(f"[INFO] Final feature shape: {X_tensor.shape}, classes: {num_classes}")

    # Validate dataset size
    if validate_size:
        validate_dataset_size(len(dataset), num_classes)

    return dataset, in_dim, num_classes
