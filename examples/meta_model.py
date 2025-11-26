"""
Meta-model (optimization proxy) that predicts model performance
based on dataset meta-features and model configuration.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from .meta_features import compute_meta_features_from_csv


META_MODEL_PATH = "experiments/meta_model.pkl"


def load_experiment_log(path: str = "experiments/results.csv") -> pd.DataFrame:
    """Load experiment results CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results.csv found at {path}")
    df = pd.read_csv(path)
    return df


def train_meta_model(results_csv: str = "experiments/results.csv"):
    """
    Train the meta-model (proxy) on historical experiment results.
    
    Requires results.csv to have columns:
    - dataset_name, model_name, metric
    - n_samples, n_features, n_classes, class_imbalance (meta-features)
    """
    df = load_experiment_log(results_csv)
    
    if df.empty:
        raise RuntimeError("results.csv is empty, nothing to train on.")
    
    # Check required columns
    required_cols = ["dataset_name", "model_name", "metric", 
                     "n_samples", "n_features", "n_classes", "class_imbalance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"results.csv missing required columns: {missing}")
    
    # Prepare features
    feature_cols = ["dataset_name", "model_name", "n_samples", "n_features", 
                    "n_classes", "class_imbalance"]
    X = df[feature_cols]
    y = df["metric"].values
    
    # Preprocessing: one-hot encode categoricals, passthrough numerics
    cat_cols = ["dataset_name", "model_name"]
    num_cols = ["n_samples", "n_features", "n_classes", "class_imbalance"]
    
    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    
    # Meta-model: Random Forest
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    
    pipe = Pipeline(steps=[
        ("preprocess", preproc),
        ("model", model),
    ])
    
    pipe.fit(X, y)
    
    # Save
    Path(META_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, META_MODEL_PATH)
    
    # Evaluate on training data
    train_pred = pipe.predict(X)
    train_mae = np.mean(np.abs(train_pred - y))
    train_rmse = np.sqrt(np.mean((train_pred - y) ** 2))
    
    print(f"[INFO] Meta-model trained on {len(df)} examples")
    print(f"[INFO] Training MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
    print(f"[INFO] Saved meta-model to {META_MODEL_PATH}")
    
    return pipe


def predict_config_scores_for_dataset(
    dataset_name: str,
    csv_path: Optional[str],
    candidate_models: List[str],
    meta_features: Optional[Dict] = None,
    label_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Predict performance scores for candidate models on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        csv_path: Path to CSV (if available, used to compute meta-features)
        candidate_models: List of model names to score
        meta_features: Pre-computed meta-features (if None, computed from csv_path)
        label_col: Label column name (required if computing from csv_path)
    
    Returns:
        Dictionary mapping model_name -> predicted metric score
    """
    if not os.path.exists(META_MODEL_PATH):
        print("[WARN] No meta-model found, using uniform scores (0.5).")
        return {m: 0.5 for m in candidate_models}
    
    pipe = joblib.load(META_MODEL_PATH)
    
    # Get meta-features
    if meta_features is None:
        if csv_path is None or label_col is None:
            raise ValueError("Either meta_features or (csv_path + label_col) must be provided")
        from .meta_features import compute_meta_features_from_csv
        meta_features = compute_meta_features_from_csv(csv_path, label_col)
    
    # Build feature rows for each candidate model
    rows = []
    for model_name in candidate_models:
        row = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "n_samples": meta_features["n_samples"],
            "n_features": meta_features["n_features"],
            "n_classes": meta_features["n_classes"],
            "class_imbalance": meta_features["class_imbalance"],
        }
        rows.append(row)
    
    X = pd.DataFrame(rows)
    preds = pipe.predict(X)
    
    return {m: float(p) for m, p in zip(candidate_models, preds)}


def predict_config_scores_from_meta_features(
    dataset_name: str,
    candidate_models: List[str],
    meta_features: Dict,
) -> Dict[str, float]:
    """
    Predict scores when meta-features are already computed.
    """
    return predict_config_scores_for_dataset(
        dataset_name=dataset_name,
        csv_path=None,
        candidate_models=candidate_models,
        meta_features=meta_features,
    )

