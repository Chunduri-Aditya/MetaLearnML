"""
Comprehensive preprocessing strategy generation.
Creates multiple preprocessing pipelines for search.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from engine.meta_features import PREPROCESSING_VOCABULARIES


def _normalize_categorical_missing(values):
    """Represent Python ``None`` and pandas missing values uniformly for sklearn."""
    if isinstance(values, pd.DataFrame):
        return values.astype(object).where(pd.notna(values), np.nan)
    array = np.asarray(values, dtype=object)
    array[pd.isna(array)] = np.nan
    return array


@dataclass(frozen=True)
class PreprocessingStrategy:
    """A complete preprocessing pipeline configuration."""

    name: str
    numeric_scaler: str  # "standard", "minmax", "robust", "power", "none"
    categorical_encoder: str  # "onehot", "ordinal", "none"
    numeric_imputer: str  # "mean", "median", "knn", "iterative", "none"
    categorical_imputer: str  # "most_frequent", "constant", "none"

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Preprocessing strategy name must be non-empty")
        for field_name, vocabulary in PREPROCESSING_VOCABULARIES.items():
            value = getattr(self, field_name)
            if value not in vocabulary:
                raise ValueError(f"Invalid {field_name}={value!r}; expected one of {vocabulary}")

    @classmethod
    def from_components(
        cls,
        *,
        numeric_scaler: str,
        numeric_imputer: str,
        categorical_encoder: str,
        categorical_imputer: str,
        name: str | None = None,
    ) -> "PreprocessingStrategy":
        components = {
            "numeric_scaler": numeric_scaler,
            "numeric_imputer": numeric_imputer,
            "categorical_encoder": categorical_encoder,
            "categorical_imputer": categorical_imputer,
        }
        if name is None:
            name = "__".join(f"{key}={value}" for key, value in components.items())
        return cls(name=name, **components)

    def to_component_dict(self) -> dict[str, str]:
        return {
            "numeric_scaler": self.numeric_scaler,
            "numeric_imputer": self.numeric_imputer,
            "categorical_encoder": self.categorical_encoder,
            "categorical_imputer": self.categorical_imputer,
        }

    def to_feature_dict(self) -> dict[str, float]:
        """Return the canonical one-hot component representation."""
        result: dict[str, float] = {}
        for group, vocabulary in PREPROCESSING_VOCABULARIES.items():
            selected = getattr(self, group)
            for value in vocabulary:
                result[f"preprocessing_{group}_{value}"] = float(value == selected)
        return result

    def validate_executable(
        self,
        *,
        has_numeric: bool,
        has_categorical: bool,
    ) -> None:
        if "unknown" in self.to_component_dict().values():
            raise ValueError(f"Strategy {self.name!r} contains unknown components and cannot run")
        if has_numeric:
            if self.numeric_imputer == "none":
                raise ValueError("Numeric columns require an explicit imputer")
        elif self.numeric_scaler != "none" or self.numeric_imputer != "none":
            raise ValueError("Numeric components must be 'none' when no numeric columns exist")
        if has_categorical:
            if self.categorical_encoder == "none" or self.categorical_imputer == "none":
                raise ValueError("Categorical columns require an encoder and imputer")
        elif self.categorical_encoder != "none" or self.categorical_imputer != "none":
            raise ValueError(
                "Categorical components must be 'none' when no categorical columns exist"
            )


def generate_preprocessing_strategies(
    has_numeric: bool,
    has_categorical: bool,
    n_samples: int,
    dataset_meta: dict | None = None,
) -> list[PreprocessingStrategy]:
    """
    Generate preprocessing strategies based on dataset characteristics.

    Uses tier-based generation:
    - Tier 1 (safe defaults): Always included
    - Tier 2 (robust): For datasets with outliers
    - Tier 3 (fancy): Only for large datasets or when meta suggests

    Args:
        has_numeric: Whether dataset has numeric features
        has_categorical: Whether dataset has categorical features
        n_samples: Number of samples (affects which imputers to use)
        dataset_meta: Optional dataset metadata for smarter decisions

    Returns:
        List of preprocessing strategies
    """
    strategies = []
    dataset_meta = dataset_meta or {}

    # Detect outliers (if variance info available)
    has_outliers = dataset_meta.get("has_outliers", False)
    high_cardinality_cats = dataset_meta.get("high_cardinality_cats", 0)

    # Tier 1: Safe defaults (always include)
    tier1_scalers = ["standard"]
    tier1_imputers = ["median"]  # Median is robust to outliers

    # Tier 2: Robust options (include if outliers detected or dataset is medium-sized)
    tier2_scalers = ["robust"] if has_outliers or n_samples > 500 else []
    tier2_imputers = []

    # Tier 3: Fancy options (only for large datasets)
    tier3_scalers = []
    if n_samples > 1000:
        tier3_scalers = ["minmax"]
        if n_samples > 5000:
            tier3_scalers.append("power")  # PowerTransformer is expensive

    # Numeric imputers
    numeric_imputers = tier1_imputers + tier2_imputers
    if n_samples < 10000:  # KNN is expensive for large datasets
        numeric_imputers.append("knn")

    # Categorical encoders
    # If high cardinality, prefer ordinal over onehot (onehot can explode)
    if high_cardinality_cats > 0:
        categorical_encoders = ["ordinal", "onehot"]  # Prefer ordinal first
    else:
        categorical_encoders = ["onehot", "ordinal"]

    # Categorical imputers
    categorical_imputers = ["most_frequent", "constant"]

    # Combine all scalers (tier1 + tier2 + tier3)
    all_scalers = tier1_scalers + tier2_scalers + tier3_scalers

    # Generate combinations with tier awareness
    if has_numeric and has_categorical:
        # Mixed: prioritize tier1, then tier2, then tier3
        for scaler in all_scalers:
            for cat_enc in categorical_encoders:
                # Limit imputers based on tier
                imputers_to_use = (
                    numeric_imputers if scaler in tier1_scalers else numeric_imputers[:2]
                )
                for num_imp in imputers_to_use:
                    for cat_imp in categorical_imputers:
                        name = f"{num_imp}_{scaler}_{cat_enc}_{cat_imp}"
                        strategies.append(
                            PreprocessingStrategy(
                                name=name,
                                numeric_scaler=scaler,
                                categorical_encoder=cat_enc,
                                numeric_imputer=num_imp,
                                categorical_imputer=cat_imp,
                            )
                        )
    elif has_numeric:
        # Numeric only
        for scaler in all_scalers:
            imputers_to_use = numeric_imputers if scaler in tier1_scalers else numeric_imputers[:2]
            for num_imp in imputers_to_use:
                name = f"{num_imp}_{scaler}"
                strategies.append(
                    PreprocessingStrategy(
                        name=name,
                        numeric_scaler=scaler,
                        categorical_encoder="none",
                        numeric_imputer=num_imp,
                        categorical_imputer="none",
                    )
                )
    elif has_categorical:
        # Categorical only
        for cat_enc in categorical_encoders:
            for cat_imp in categorical_imputers:
                name = f"{cat_enc}_{cat_imp}"
                strategies.append(
                    PreprocessingStrategy(
                        name=name,
                        numeric_scaler="none",
                        categorical_encoder=cat_enc,
                        numeric_imputer="none",
                        categorical_imputer=cat_imp,
                    )
                )

    # Limit to top 12 strategies to avoid combinatorial explosion
    # Prioritize tier1 strategies
    if len(strategies) > 12:
        # Sort: tier1 first, then tier2, then tier3
        tier1_strategies = [s for s in strategies if s.numeric_scaler in tier1_scalers]
        tier2_strategies = [s for s in strategies if s.numeric_scaler in tier2_scalers]
        tier3_strategies = [s for s in strategies if s.numeric_scaler in tier3_scalers]

        # Keep all tier1, some tier2, few tier3
        strategies = tier1_strategies[:6] + tier2_strategies[:4] + tier3_strategies[:2]

    return strategies


def build_preprocessing_pipeline(
    strategy: PreprocessingStrategy,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """
    Build a sklearn Pipeline from a strategy.

    Args:
        strategy: Preprocessing strategy
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names

    Returns:
        sklearn Pipeline or ColumnTransformer
    """
    strategy.validate_executable(
        has_numeric=bool(numeric_cols),
        has_categorical=bool(categorical_cols),
    )
    transformers = []

    # Numeric preprocessing
    if len(numeric_cols) > 0:
        numeric_steps = []

        # Imputation
        if strategy.numeric_imputer == "mean":
            numeric_steps.append(("imputer", SimpleImputer(strategy="mean")))
        elif strategy.numeric_imputer == "median":
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        elif strategy.numeric_imputer == "knn":
            numeric_steps.append(("imputer", KNNImputer(n_neighbors=5)))

        # Scaling
        if strategy.numeric_scaler == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif strategy.numeric_scaler == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        elif strategy.numeric_scaler == "robust":
            numeric_steps.append(("scaler", RobustScaler()))
        elif strategy.numeric_scaler == "power":
            numeric_steps.append(("scaler", PowerTransformer(method="yeo-johnson")))

        if numeric_steps:
            transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
        else:
            transformers.append(("numeric", "passthrough", numeric_cols))

    # Categorical preprocessing
    if len(categorical_cols) > 0:
        cat_steps = [
            (
                "normalize_missing",
                FunctionTransformer(
                    _normalize_categorical_missing,
                    feature_names_out="one-to-one",
                ),
            )
        ]

        # Imputation
        if strategy.categorical_imputer == "most_frequent":
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        elif strategy.categorical_imputer == "constant":
            cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))

        # Encoding
        if strategy.categorical_encoder == "onehot":
            cat_steps.append(
                ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
            )
        elif strategy.categorical_encoder == "ordinal":
            cat_steps.append(
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            )

        if cat_steps:
            transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))
        else:
            transformers.append(("categorical", "passthrough", categorical_cols))

    if transformers:
        return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    else:
        # No preprocessing needed
        return None


def rank_preprocessors_by_proxy(
    X_df: pd.DataFrame,
    y: np.ndarray,
    strategies: list[PreprocessingStrategy],
    task_type: str,
    metric_fn,
    sample_size: int = 512,
    random_state: int = 42,
) -> list[tuple[PreprocessingStrategy, float]]:
    """
    Rank preprocessing strategies using a fast proxy evaluation.

    Args:
        X_df: Feature DataFrame
        y: Target array
        strategies: List of preprocessing strategies
        task_type: "regression" or "classification"
        metric_fn: Metric function
        sample_size: Number of samples to use for proxy evaluation

    Returns:
        List of (strategy, proxy_score) tuples sorted by score
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Subsample for speed
    if len(X_df) > sample_size:
        generator = np.random.default_rng(random_state)
        indices = generator.choice(len(X_df), sample_size, replace=False)
        X_sample = X_df.iloc[indices].copy()
        y_sample = y[indices]
    else:
        X_sample = X_df.copy()
        y_sample = y

    # Split
    stratify = y_sample if task_type != "regression" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_sample,
        y_sample,
        test_size=0.3,
        random_state=random_state,
        stratify=stratify,
    )

    # Get column types
    numeric_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_sample.select_dtypes(exclude=[np.number]).columns.tolist()

    scored_strategies = []

    for strategy in strategies:
        try:
            # Build pipeline
            pipeline = build_preprocessing_pipeline(strategy, numeric_cols, categorical_cols)

            if pipeline is None:
                # No preprocessing needed
                X_train_proc = X_train.values
                X_val_proc = X_val.values
            else:
                # Fit and transform
                X_train_proc = pipeline.fit_transform(X_train)
                X_val_proc = pipeline.transform(X_val)

            # Train cheap baseline model
            if task_type == "regression":
                baseline = RandomForestRegressor(
                    n_estimators=50, random_state=random_state, n_jobs=1
                )
            else:
                baseline = LogisticRegression(max_iter=200, random_state=random_state, n_jobs=1)

            baseline.fit(X_train_proc, y_train)

            # Evaluate
            if task_type == "regression":
                y_pred = baseline.predict(X_val_proc)
                # Convert to tensor for metric function
                import torch

                y_pred_t = torch.from_numpy(y_pred.reshape(-1, 1)).float()
                y_val_t = torch.from_numpy(y_val.astype(np.float32))
                score = metric_fn(y_val_t, y_pred_t)
            else:
                y_pred = baseline.predict_proba(X_val_proc)
                import torch

                y_pred_t = torch.from_numpy(y_pred).float()
                y_val_t = torch.from_numpy(y_val.astype(np.int64))
                score = metric_fn(y_pred_t, y_val_t)

            scored_strategies.append((strategy, score))
        except Exception:
            # Skip failed strategies
            continue

    # Sort by score (descending)
    scored_strategies.sort(key=lambda x: x[1], reverse=True)
    return scored_strategies
