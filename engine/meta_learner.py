"""
Meta-learning brain for the AutoML engine.
Learns from past experiments to predict model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class MetaLearner:
    """
    Meta-model that learns to predict model performance based on dataset characteristics.
    This is the "brain" of the AutoML engine.
    
    Version 2: Supports per-task+metric models for better specialization.
    """
    
    def __init__(
        self, 
        meta_data_path: str = "experiments/meta_train.csv",
        task_type: Optional[str] = None,
        metric_name: Optional[str] = None,
    ):
        self.meta_data_path = Path(meta_data_path)
        self.task_type = task_type
        self.metric_name = metric_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.min_history_size = 20  # Minimum experiments needed for reliable meta-learning
    
    def load_history(self) -> pd.DataFrame:
        """Load historical experiment data."""
        if not self.meta_data_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.meta_data_path)
            return df
        except Exception as e:
            print(f"[WARN] Could not load meta-learning history: {e}")
            return pd.DataFrame()
    
    def _get_model_path(self) -> Path:
        """Get model path based on task_type and metric_name."""
        base = Path("experiments")
        if self.task_type and self.metric_name:
            return base / f"meta_model_{self.task_type}_{self.metric_name}.pkl"
        else:
            return base / "meta_model.pkl"  # Legacy fallback
    
    def extract_features(self, dataset_meta: Dict, model_name: str, preprocessing_name: str) -> np.ndarray:
        """
        Extract features for meta-model prediction.
        
        Args:
            dataset_meta: Dataset metadata
            model_name: Model name
            preprocessing_name: Preprocessing strategy name
        
        Returns:
            Feature vector
        """
        features = []
        feature_names = []
        
        # Dataset features (raw)
        n_samples = dataset_meta.get("n_samples", 0)
        n_features = dataset_meta.get("n_features", 0)
        n_numeric = dataset_meta.get("n_numeric", 0)
        n_categorical = dataset_meta.get("n_categorical", 0)
        n_text = dataset_meta.get("n_text", 0)
        n_classes = dataset_meta.get("n_classes", 0) or 0
        class_imbalance = dataset_meta.get("class_imbalance", 0) or 0
        missing_frac = dataset_meta.get("missing_value_fraction", 0)
        
        # Raw features
        features.extend([n_samples, n_features, n_numeric, n_categorical, n_classes, class_imbalance, missing_frac])
        feature_names.extend(["n_samples", "n_features", "n_numeric", "n_categorical", "n_classes", "class_imbalance", "missing_value_fraction"])
        
        # Derived features (richer representation)
        features.append(np.log10(n_samples + 1))
        feature_names.append("log_n_samples")
        
        features.append(np.log10(n_features + 1))
        feature_names.append("log_n_features")
        
        # Feature density (for sparse datasets like TF-IDF)
        if n_samples > 0 and n_features > 0:
            features.append(n_features / n_samples)
            feature_names.append("feature_density")
        else:
            features.append(0.0)
            feature_names.append("feature_density")
        
        # Text fraction
        if n_features > 0:
            features.append(n_text / n_features)
            feature_names.append("text_fraction")
        else:
            features.append(0.0)
            feature_names.append("text_fraction")
        
        # Categorical fraction
        if n_features > 0:
            features.append(n_categorical / n_features)
            feature_names.append("categorical_fraction")
        else:
            features.append(0.0)
            feature_names.append("categorical_fraction")
        
        # Model family encoding (one-hot like)
        model_families = ["neural_supervised", "classical_supervised", "linear_regression", 
                         "tree_regression", "neural_regression"]
        model_family = self._get_model_family(model_name)
        for fam in model_families:
            features.append(1.0 if fam == model_family else 0.0)
            feature_names.append(f"model_family_{fam}")
        
        # Preprocessing encoding
        preprocessing_types = ["basic_numeric_only", "numeric_plus_cat_onehot", 
                             "robust_numeric", "text_tfidf_basic"]
        for prep in preprocessing_types:
            features.append(1.0 if prep == preprocessing_name else 0.0)
            feature_names.append(f"preprocessing_{prep}")
        
        self.feature_names = feature_names
        return np.array(features, dtype=np.float32)
    
    def _get_model_family(self, model_name: str) -> str:
        """Get model family from model name."""
        from models.model_zoo import MODEL_REGISTRY
        return MODEL_REGISTRY.get(model_name, {}).get("family", "unknown")
    
    def train(self, history_df: Optional[pd.DataFrame] = None):
        """
        Train meta-model on historical data.
        
        Args:
            history_df: Optional DataFrame with history. If None, loads from file.
        """
        if history_df is None:
            history_df = self.load_history()
        
        # Filter by task_type and metric_name if specified
        if self.task_type:
            history_df = history_df[history_df.get("task_type", "") == self.task_type]
        if self.metric_name:
            history_df = history_df[history_df.get("metric_name", "") == self.metric_name]
        
        if len(history_df) < self.min_history_size:
            print(f"[INFO] Not enough history for meta-learning (need at least {self.min_history_size} experiments, got {len(history_df)})")
            self.is_trained = False
            return
        
        # Prepare features and targets
        X_list = []
        y_list = []
        
        for _, row in history_df.iterrows():
            try:
                # Extract dataset metadata
                dataset_meta = {
                    "n_samples": row.get("n_samples", 0),
                    "n_features": row.get("n_features", 0),
                    "n_numeric": row.get("n_numeric", 0),
                    "n_categorical": row.get("n_categorical", 0),
                    "n_classes": row.get("n_classes", 0),
                    "class_imbalance": row.get("class_imbalance", 0),
                    "missing_value_fraction": 0,  # Would need to be logged
                }
                
                model_name = row.get("model_name", "")
                preprocessing_name = row.get("preprocessing_name", "basic_numeric_only")
                metric_value = row.get("metric", 0)
                
                features = self.extract_features(dataset_meta, model_name, preprocessing_name)
                X_list.append(features)
                y_list.append(metric_value)
            except Exception as e:
                continue
        
        if len(X_list) < 10:
            print("[INFO] Not enough valid history for meta-learning")
            self.is_trained = False
            return
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"[INFO] Meta-model trained on {len(X_list)} experiments")
    
    def predict(self, dataset_meta: Dict, model_name: str, preprocessing_name: str) -> float:
        """
        Predict model performance.
        
        Args:
            dataset_meta: Dataset metadata
            model_name: Model name
            preprocessing_name: Preprocessing strategy name
        
        Returns:
            Predicted metric value
        """
        if not self.is_trained:
            return 0.0  # Default prediction
        
        features = self.extract_features(dataset_meta, model_name, preprocessing_name)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        return float(prediction)
    
    def rank_candidates(self, dataset_meta: Dict, candidates: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """
        Rank candidates by predicted performance.
        
        Args:
            dataset_meta: Dataset metadata
            candidates: List of (model_name, preprocessing_name) tuples
        
        Returns:
            List of (model_name, preprocessing_name, predicted_score) sorted by score
        """
        if not self.is_trained:
            # Return in original order with default scores
            return [(m, p, 0.0) for m, p in candidates]
        
        scored = []
        for model_name, preprocessing_name in candidates:
            score = self.predict(dataset_meta, model_name, preprocessing_name)
            scored.append((model_name, preprocessing_name, score))
        
        # Sort by predicted score (descending)
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored
    
    def save(self, path: Optional[str] = None):
        """Save trained meta-model."""
        if not self.is_trained:
            return
        
        if path is None:
            path = self._get_model_path()
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
        }, path)
        print(f"[INFO] Meta-model saved to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load trained meta-model."""
        if path is None:
            path = self._get_model_path()
        else:
            path = Path(path)
        
        if path.exists():
            data = joblib.load(path)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data.get("feature_names", [])
            self.task_type = data.get("task_type", self.task_type)
            self.metric_name = data.get("metric_name", self.metric_name)
            self.is_trained = True
            print(f"[INFO] Meta-model loaded from {path}")
        else:
            self.is_trained = False
    
    def rank_candidates_with_fallback(
        self, 
        dataset_meta: Dict, 
        candidates: List[Tuple[str, str]],
        min_history_size: int = 20,
    ) -> List[Tuple[str, str, float]]:
        """
        Rank candidates with fallback guardrails.
        
        If history is too small, ensures diversity in top candidates.
        """
        if not self.is_trained or len(self.load_history()) < min_history_size:
            # Fallback: ensure diversity
            print("[INFO] Meta-learner not confident. Using diversity-based ranking with fallback.")
            return self._rank_with_diversity_fallback(dataset_meta, candidates)
        
        # Use meta-learning predictions
        return self.rank_candidates(dataset_meta, candidates)
    
    def _rank_with_diversity_fallback(
        self, 
        dataset_meta: Dict, 
        candidates: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, float]]:
        """
        Fallback ranking that ensures model family diversity.
        """
        from models.model_zoo import MODEL_REGISTRY
        
        # Group by model family
        families_seen = set()
        ranked = []
        remaining = candidates.copy()
        
        # First pass: one from each family
        for model_name, preprocessing_name in candidates:
            family = MODEL_REGISTRY.get(model_name, {}).get("family", "unknown")
            if family not in families_seen:
                families_seen.add(family)
                ranked.append((model_name, preprocessing_name, 0.0))
                remaining.remove((model_name, preprocessing_name))
        
        # Second pass: add remaining
        for model_name, preprocessing_name in remaining:
            ranked.append((model_name, preprocessing_name, 0.0))
        
        return ranked

