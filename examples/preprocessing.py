"""
Preprocessing strategies for dataset-agnostic AutoML.
Each strategy is a bundle of preprocessing choices that can be searched over.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import torch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from utils.datasets import DatasetDescriptor


@dataclass
class PreprocessingConfig:
    """Configuration for a preprocessing strategy."""
    name: str
    numeric_imputer: Optional[str] = None  # "mean", "median", "most_frequent", "knn", "constant"
    scaler: Optional[str] = None           # "standard", "minmax", "robust"
    categorical_imputer: Optional[str] = None
    encoder: Optional[str] = None          # "onehot", "ordinal"
    text_mode: Optional[str] = None        # "tfidf"
    max_features: Optional[int] = None
    ngram_range: Optional[Tuple[int, int]] = None


def apply_preprocessing(
    descriptor: DatasetDescriptor,
    config: PreprocessingConfig,
    train_indices: List[int],
    val_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """
    Apply preprocessing strategy to dataset.
    
    Args:
        descriptor: Dataset descriptor
        config: Preprocessing configuration
        train_indices: List of training indices
        val_indices: List of validation indices
    
    Returns:
        (X_train, y_train, X_val, y_val, fitted_transformer)
    """
    # Split data
    X_train_raw = descriptor.X[train_indices].numpy()
    y_train = descriptor.y[train_indices]
    X_val_raw = descriptor.X[val_indices].numpy()
    y_val = descriptor.y[val_indices]
    
    # Get column types
    column_types = descriptor.column_types
    feature_names = descriptor.feature_names
    
    # Separate columns by type
    numeric_cols = [i for i, name in enumerate(feature_names) if column_types.get(name) == "numeric"]
    categorical_cols = [i for i, name in enumerate(feature_names) if column_types.get(name) == "categorical"]
    text_cols = [i for i, name in enumerate(feature_names) if column_types.get(name) == "text"]
    
    # Handle text-only mode
    if config.text_mode == "tfidf" and len(text_cols) > 0:
        # Text preprocessing
        # For now, assume text columns are already processed or we need to handle differently
        # This is a simplified version - full implementation would handle raw text
        X_train_processed, X_val_processed, transformer = _apply_text_preprocessing(
            descriptor, config, train_indices, val_indices, text_cols
        )
        return (
            torch.from_numpy(X_train_processed).float(),
            y_train,
            torch.from_numpy(X_val_processed).float(),
            y_val,
            transformer,
        )
    
    # Build preprocessing pipeline
    transformers = []
    
    # Numeric preprocessing
    if len(numeric_cols) > 0:
        numeric_steps = []
        
        # Imputation
        if config.numeric_imputer:
            if config.numeric_imputer == "knn":
                numeric_steps.append(("imputer", KNNImputer(n_neighbors=5)))
            elif config.numeric_imputer in ["mean", "median", "most_frequent"]:
                numeric_steps.append(("imputer", SimpleImputer(strategy=config.numeric_imputer)))
            elif config.numeric_imputer == "constant":
                numeric_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))
        
        # Scaling
        if config.scaler:
            if config.scaler == "standard":
                numeric_steps.append(("scaler", StandardScaler()))
            elif config.scaler == "minmax":
                numeric_steps.append(("scaler", MinMaxScaler()))
            elif config.scaler == "robust":
                numeric_steps.append(("scaler", RobustScaler()))
        
        if numeric_steps:
            transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
        else:
            # No preprocessing for numeric - just pass through
            transformers.append(("numeric", "passthrough", numeric_cols))
    
    # Categorical preprocessing
    if len(categorical_cols) > 0:
        cat_steps = []
        
        # Imputation
        if config.categorical_imputer:
            if config.categorical_imputer == "most_frequent":
                cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            elif config.categorical_imputer == "constant":
                cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))
        
        # Encoding
        if config.encoder:
            if config.encoder == "onehot":
                cat_steps.append(("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")))
            elif config.encoder == "ordinal":
                cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        
        if cat_steps:
            transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))
        else:
            # No preprocessing - pass through (will need to be handled)
            transformers.append(("categorical", "passthrough", categorical_cols))
    
    # Create column transformer
    if transformers:
        preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    else:
        # No preprocessing needed - return as is
        X_train_processed = X_train_raw
        X_val_processed = X_val_raw
        transformer = None
        return (
            torch.from_numpy(X_train_processed).float(),
            y_train,
            torch.from_numpy(X_val_processed).float(),
            y_val,
            transformer,
        )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_val_processed = preprocessor.transform(X_val_raw)
    
    # Convert to float32
    if isinstance(X_train_processed, np.ndarray):
        X_train_processed = X_train_processed.astype(np.float32)
        X_val_processed = X_val_processed.astype(np.float32)
    
    return (
        torch.from_numpy(X_train_processed).float(),
        y_train,
        torch.from_numpy(X_val_processed).float(),
        y_val,
        preprocessor,
    )


def _apply_text_preprocessing(
    descriptor: DatasetDescriptor,
    config: PreprocessingConfig,
    train_indices: List[int],
    val_indices: List[int],
    text_cols: List[int],
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Apply text preprocessing (TF-IDF).
    
    This is a placeholder - full implementation would need access to raw text data.
    For now, we assume text has already been converted to numeric features.
    """
    # Simplified: return numeric features as-is
    # Full implementation would load raw text and apply TF-IDF
    X_train = descriptor.X[train_indices].numpy()
    X_val = descriptor.X[val_indices].numpy()
    
    return X_train, X_val, None

