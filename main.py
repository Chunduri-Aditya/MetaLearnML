import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Legacy import - kept for backward compatibility
# New code should use engine.orchestrator.run_automl()
try:
    from engine.legacy.search_controller_v0 import run_model_search
except ImportError:
    # Fallback if legacy module not found
    def run_model_search(*args, **kwargs):
        raise NotImplementedError(
            "Legacy search_controller moved to engine.legacy. Use engine.orchestrator.run_automl() instead."
        )


from engine.evaluator import accuracy_metric, confusion_matrix_for_model, f1_macro_metric
from engine.predictor import BestModelPredictor
from utils.datasets import load_csv_classification, load_text_csv_classification
from utils.seed_utils import get_seed_from_env, set_seed


def get_metric_function(metric_name: str) -> Callable:
    """Get metric function by name."""
    metric_map = {
        "accuracy": accuracy_metric,
        "f1": f1_macro_metric,
    }
    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}. Choose from {list(metric_map.keys())}")
    return metric_map[metric_name]


def create_base_config(metric_fn: Callable, metric_name: str) -> dict:
    """Create base configuration dictionary."""
    return {
        "lr": 1e-3,
        "epochs": 20,
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "metric_fn": metric_fn,
        "metric_name": metric_name,
        "patience": 5,
    }


def split_and_create_loaders(
    dataset: TensorDataset,
    train_ratio: float = 0.8,
    train_batch_size: int = 32,
    val_batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset and create data loaders."""
    n_train = int(train_ratio * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size)
    return train_loader, val_loader


def compute_meta_features_safe(
    dataset: TensorDataset, in_dim: int, out_dim: int, n_samples: int
) -> dict | None:
    """Safely compute meta-features with error handling."""
    try:
        from examples.meta_features import compute_meta_features_from_dataset

        return compute_meta_features_from_dataset(dataset, in_dim, out_dim, n_samples)
    except Exception as e:
        print(f"[WARN] Could not compute meta-features: {e}")
        return None


def print_confusion_matrix_results(
    predictor: BestModelPredictor, val_loader: DataLoader, device: str, out_dim: int
):
    """Print confusion matrix and metrics."""
    print("\n" + "=" * 60)
    print("Confusion Matrix on Validation Set")
    print("=" * 60)
    cm, metrics = confusion_matrix_for_model(predictor.model, val_loader, device)
    print("\nConfusion Matrix:")
    print(cm)
    print("\n(rows = true labels, columns = predicted labels)")

    # For binary classification, show some interpretability
    if out_dim == 2:
        tn, fp, fn, tp = cm.ravel()
        print("\nBinary Classification Metrics:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        if metrics:
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")


def load_dataset_by_name(dataset: str) -> tuple[TensorDataset, int, int, str]:
    """
    Load dataset by name.

    Returns:
        (dataset_obj, in_dim, out_dim, dataset_name)
    """
    dataset_configs = {
        "iris": {
            "path": "data/iris.csv",
            "label_col": "label",
            "feature_cols": [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
            "error_msg": "➡ Run: python examples/prepare_datasets.py first",
        },
        "titanic": {
            "path": "data/titanic_clean.csv",
            "label_col": "Survived",
            "feature_cols": [
                "Pclass",
                "Sex_male",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked_C",
                "Embarked_Q",
                "Embarked_S",
            ],
            "error_msg": "➡ Run: python examples/prepare_titanic.py (after placing Kaggle train.csv in data/)",
        },
    }

    if dataset not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_configs.keys())}")

    config = dataset_configs[dataset]
    path = config["path"]

    if not Path(path).exists():
        print(f"[ERROR] {path} not found.")
        print(config["error_msg"])
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataset_obj, in_dim, out_dim = load_csv_classification(
        path=path,
        label_col=config["label_col"],
        feature_cols=config["feature_cols"],
    )

    print(
        f"[INFO] Loaded {dataset.upper()} dataset: {len(dataset_obj)} samples, {in_dim} features, {out_dim} classes"
    )

    return dataset_obj, in_dim, out_dim, dataset


def run_experiment(dataset: str, metric_name: str):
    """
    Run a complete AutoML experiment.

    Args:
        dataset: "iris" or "titanic"
        metric_name: "accuracy" or "f1"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print(f"Running Experiment: {dataset.upper()} with {metric_name.upper()} metric")
    print(f"{'=' * 60}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Select metric function
    metric_fn = get_metric_function(metric_name)

    # 2) Load dataset
    try:
        dataset_obj, in_dim, out_dim, dataset_name = load_dataset_by_name(dataset)
    except FileNotFoundError:
        return

    # 3) Train/val split
    train_loader, val_loader = split_and_create_loaders(dataset_obj)

    # 4) Base config
    base_config = create_base_config(metric_fn, metric_name)

    print(f"[INFO] Using metric: {metric_name.upper()}")
    print()

    # Compute meta-features
    meta_features = compute_meta_features_safe(dataset_obj, in_dim, out_dim, len(dataset_obj))

    # 5) Run model search
    best_model_info = run_model_search(
        train_loader=train_loader,
        val_loader=val_loader,
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
        base_config=base_config,
        results_csv="experiments/results.csv",
        dataset_name=dataset_name,
        meta_features=meta_features,
    )

    print("\n[INFO] Best model from search:")
    print(best_model_info)

    # 6) Load best model via predictor
    predictor = BestModelPredictor(
        results_csv="experiments/results.csv",
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
    )

    print(f"\n[INFO] Loaded best model: {predictor.model_name}")
    print(f"[INFO] Best recorded metric: {predictor.best_metric:.4f}")

    # 7) Sample predictions
    val_batch = next(iter(DataLoader(val_loader.dataset, batch_size=5)))
    x_val_small, y_val_small = val_batch

    preds = predictor.predict_classes(x_val_small)
    print("\nSample predictions:")
    print("True labels:   ", y_val_small.tolist())
    print("Predicted:     ", preds.tolist())

    # 8) Confusion matrix
    print_confusion_matrix_results(predictor, val_loader, device, out_dim)

    # 9) Next steps
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("  ✓ Check experiments/results.csv for all model scores")
    print("  ✓ Document this run in EXPERIMENTS.md")
    if dataset == "titanic":
        print("  ✓ Run `python online_titanic_demo.py` to see online adaptation")
    print()


def run_custom_csv_experiment():
    print("============================================================")
    print("Custom CSV AutoML Mode")
    print("============================================================\n")

    dataset_name = (
        input("Enter a name for this dataset (for logging): ").strip() or "custom_dataset"
    )
    csv_name = input("Enter CSV file name in data/ (e.g. my_data.csv): ").strip()
    csv_path = os.path.join("data", csv_name)

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        print("➡ Place your CSV in the data/ directory and run again.")
        return

    label_col = input("Enter the label/target column name: ").strip()

    print("\nSelect metric:")
    print("  [1] Accuracy")
    print("  [2] F1 macro")
    metric_choice = input("Enter choice [1-2]: ").strip()

    metric_map = {"1": "accuracy", "2": "f1"}
    if metric_choice not in metric_map:
        print("[ERROR] Invalid metric choice.")
        return

    metric_name = metric_map[metric_choice]
    metric_fn = get_metric_function(metric_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 1) Try numeric auto-detection first ===
    try:
        dataset, in_dim, out_dim = load_csv_classification(
            path=csv_path,
            feature_cols=None,  # auto-detect numeric features
            label_col=label_col,
        )
        mode = "numeric"
    except Exception as e:
        print(f"\n[INFO] Numeric feature auto-detection failed: {e}")
        print("[INFO] Switching to TEXT MODE (TF-IDF).\n")

        # Show columns to help pick text features
        df_preview = pd.read_csv(csv_path, nrows=3)
        print("Available columns:", list(df_preview.columns))
        print("\nExample row (first 3):")
        for idx, row in df_preview.head(3).iterrows():
            print(f"\nRow {idx}:")
            for col in df_preview.columns:
                val = row[col]
                if isinstance(val, str) and len(val) > 120:
                    val = val[:117] + "..."
                print(f"  {col}: {val}")

        text_cols_input = input(
            "\nEnter comma-separated text columns to use as features\n"
            "(e.g. content,skills,what_you_learn): "
        ).strip()
        text_cols = [c.strip() for c in text_cols_input.split(",") if c.strip()]

        if not text_cols:
            print("[ERROR] No text columns provided. Aborting.")
            return

        dataset, in_dim, out_dim = load_text_csv_classification(
            path=csv_path,
            text_cols=text_cols,
            label_col=label_col,
            max_features=5000,
        )
        mode = "text"

    print(f"\n[INFO] Loaded '{dataset_name}' from {csv_path} ({mode} mode)")
    print(f"[INFO] Samples: {len(dataset)}, Features: {in_dim}, Classes: {out_dim}")
    print(f"[INFO] Using metric: {metric_name}\n")

    # === 2) Train/val split ===
    train_loader, val_loader = split_and_create_loaders(dataset)

    # === 3) Base config ===
    base_config = create_base_config(metric_fn, metric_name)

    results_csv = "experiments/results.csv"

    # === Compute meta-features ===
    meta_features = compute_meta_features_safe(dataset, in_dim, out_dim, len(dataset))

    # === 4) Run model search ===
    best_model_info = run_model_search(
        train_loader=train_loader,
        val_loader=val_loader,
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
        base_config=base_config,
        results_csv=results_csv,
        dataset_name=dataset_name,
        meta_features=meta_features,
    )

    print("\n=== Best model decided by AutoML ===")
    print(f"  Model:  {best_model_info['model_name']}")
    print(f"  Metric: {best_model_info['metric']:.4f} ({metric_name})")
    print(f"  Checkpoint: {best_model_info['ckpt_path']}\n")

    # === 5) Load best model with predictor ===
    predictor = BestModelPredictor(
        results_csv=results_csv,
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
    )

    cm, metrics = confusion_matrix_for_model(
        model=predictor.model,
        data_loader=val_loader,
        device=device,
    )

    print("============================================================")
    print("Confusion Matrix on Validation Set")
    print("============================================================\n")
    print("Confusion Matrix:")
    print(cm)
    print("\n(rows = true labels, columns = predicted labels)\n")

    if metrics is not None:
        print("Binary classification metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()

    # === 6) Show all model scores ===
    print("============================================================")
    print("All Models Tried (with metrics)")
    print("============================================================\n")
    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        print(df_results[["model_name", "metric"]])
    else:
        print("[WARN] results.csv not found; something went wrong in logging.")

    print("\n============================================================")
    print("Next Steps")
    print("============================================================")
    print(f"  ✓ Check {results_csv} for detailed per-model results")
    print(f"  ✓ Document this run in EXPERIMENTS.md with dataset name: {dataset_name}")
    print()


def run_custom_csv_threshold_experiment():
    """Run AutoML with threshold-based search on a custom CSV file."""
    print("============================================================")
    print("Custom CSV Threshold Search Mode")
    print("============================================================\n")

    dataset_name = (
        input("Enter a name for this dataset (for logging): ").strip() or "custom_dataset"
    )
    csv_name = input("Enter CSV file name in data/: ").strip()
    csv_path = os.path.join("data", csv_name)

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    label_col = input("Enter the label/target column name: ").strip()

    print("\nSelect metric:")
    print("  [1] Accuracy")
    print("  [2] F1 macro")
    metric_choice = input("Enter choice [1-2]: ").strip()

    metric_map = {"1": "accuracy", "2": "f1"}
    if metric_choice not in metric_map:
        print("[ERROR] Invalid metric choice.")
        return

    metric_name = metric_map[metric_choice]
    metric_fn = get_metric_function(metric_name)

    target_str = input(f"Enter target {metric_name} (e.g. 0.80): ").strip()
    try:
        target_metric = float(target_str)
    except ValueError:
        print("[ERROR] Invalid target value.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load dataset (numeric or text) ===
    mode = None
    try:
        dataset, in_dim, out_dim = load_csv_classification(
            path=csv_path,
            feature_cols=None,  # auto-detect numeric features
            label_col=label_col,
        )
        mode = "numeric"
    except Exception as e:
        if "No numeric feature columns found" in str(e):
            print(f"\n[INFO] Numeric feature auto-detection failed: {e}")
            print("[INFO] Switching to TEXT MODE (TF-IDF).\n")

            df_preview = pd.read_csv(csv_path, nrows=3)
            print("Available columns:", list(df_preview.columns))
            print("\nExample row (first 3):")
            for idx, row in df_preview.head(3).iterrows():
                print(f"\nRow {idx}:")
                for col in df_preview.columns:
                    val = row[col]
                    if isinstance(val, str) and len(val) > 120:
                        val = val[:117] + "..."
                    print(f"  {col}: {val}")

            text_cols_input = input(
                "\nEnter comma-separated text columns to use as features\n"
                "(e.g. content,skills,what_you_learn): "
            ).strip()
            text_cols = [c.strip() for c in text_cols_input.split(",") if c.strip()]

            if not text_cols:
                print("[ERROR] No text columns provided. Aborting.")
                return

            dataset, in_dim, out_dim = load_text_csv_classification(
                path=csv_path,
                text_cols=text_cols,
                label_col=label_col,
                max_features=5000,
            )
            mode = "text"
        else:
            print(f"\n[ERROR] Failed to load dataset: {e}\n")
            return

    print(f"\n[INFO] Loaded '{dataset_name}' from {csv_path} ({mode} mode)")
    print(f"[INFO] Samples: {len(dataset)}, Features: {in_dim}, Classes: {out_dim}")
    print(f"[INFO] Using metric: {metric_name}")
    print(f"[INFO] Target metric: {target_metric:.4f}\n")

    # === Compute meta-features for proxy ===
    from examples.meta_features import compute_meta_features_from_csv

    try:
        if mode == "numeric":
            meta_features = compute_meta_features_from_csv(csv_path, label_col)
        else:
            # For text mode, compute from loaded dataset
            meta_features = compute_meta_features_safe(dataset, in_dim, out_dim, len(dataset))
        if meta_features:
            print(f"[INFO] Meta-features computed: {meta_features}")
    except Exception as e:
        print(f"[WARN] Could not compute meta-features: {e}. Proceeding without proxy.")
        meta_features = None

    # === Train/val split ===
    train_loader, val_loader = split_and_create_loaders(dataset)

    # === Base config ===
    base_config = create_base_config(metric_fn, metric_name)

    results_csv = "experiments/results.csv"

    from engine.legacy.search_controller_v0 import run_model_search_until_threshold

    result = run_model_search_until_threshold(
        train_loader=train_loader,
        val_loader=val_loader,
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
        base_config=base_config,
        results_csv=results_csv,
        target_metric=target_metric,
        family_order=[
            "classical_supervised",
            "neural_supervised",
        ],  # Try classical first, then neural
        dataset_name=dataset_name,
        meta_features=meta_features,
        use_proxy=True,
    )

    best_info = result["best_info"]
    reached = result["reached_target"]

    print("\n============================================================")
    print("Threshold Search Summary")
    print("============================================================")
    print(f"Tried models (in order): {result['tried_models']}")
    print(f"Best model: {best_info['model_name']}")
    print(f"Best metric: {best_info['metric']:.4f} ({metric_name})")
    print(f"Target metric: {target_metric:.4f}")
    print(f"Reached target: {'✓ YES' if reached else '✗ NO'}")

    # Show proxy gaps if available
    if result.get("proxy_gaps"):
        print("\nProxy Performance:")
        for model, pred, actual, gap in result["proxy_gaps"]:
            print(f"  {model}: predicted={pred:.4f}, actual={actual:.4f}, gap={gap:.4f}")

    # Load best model and show confusion matrix
    predictor = BestModelPredictor(
        results_csv=results_csv,
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
    )

    cm, metrics = confusion_matrix_for_model(
        model=predictor.model,
        data_loader=val_loader,
        device=device,
    )

    print("\n============================================================")
    print("Confusion Matrix on Validation Set")
    print("============================================================\n")
    print("Confusion Matrix:")
    print(cm)
    print("\n(rows = true labels, columns = predicted labels)")

    if metrics is not None:
        print("\nBinary classification metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n============================================================")
    print("Next Steps")
    print("============================================================")
    print(f"  ✓ Check {results_csv} for detailed per-model results")
    print(f"  ✓ Document this run in EXPERIMENTS.md with dataset name: {dataset_name}")
    print()


def run_dataset_agnostic_experiment(
    csv_path: str, label_col: str | None = None, metric_name: str | None = None
):
    """
    Run experiment using the new dataset-agnostic system.
    Supports both classification and regression automatically.
    """
    import numpy as np
    import yaml
    from torch.utils.data import DataLoader, TensorDataset

    from engine.evaluator import METRIC_REGISTRY
    from engine.trainer import train_one_model
    from models.model_zoo import MODEL_REGISTRY, build_model
    from utils.datasets import load_tabular_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print("🔥 Dataset-Agnostic AutoML Engine")
    print("=" * 60)
    print(f"Loading dataset: {csv_path}\n")

    # Load dataset with unified loader
    try:
        dataset_info = load_tabular_dataset(
            path=csv_path,
            label_col=label_col,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback

        traceback.print_exc()
        return

    task_type = dataset_info["task_type"]
    X_df = dataset_info["X_df"]
    y = dataset_info["y"]
    meta = dataset_info["meta"]
    dataset_name = dataset_info["dataset_name"]

    # Print insights
    print("\n" + "=" * 60)
    print("Dataset Insights")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Task type: {task_type}")
    print(f"Total samples: {meta['n_samples']}")
    print(f"Total features: {meta['n_features']}")
    print(f"Numeric features: {meta['n_numeric']}")
    print(f"Categorical features: {meta['n_categorical']}")
    if task_type != "regression":
        print(f"Number of classes: {meta['n_classes']}")
        if meta.get("class_imbalance"):
            print(f"Class imbalance: {meta['class_imbalance']:.3f}")
    print("=" * 60 + "\n")

    # Select metric if not provided
    if metric_name is None:
        if task_type == "regression":
            print("Select metric for regression:")
            print("  [1] R² (coefficient of determination)")
            print("  [2] Negative RMSE (lower is better)")
            print("  [3] Negative MAE (lower is better)")
            choice = input("Enter choice [1-3]: ").strip()
            metric_map = {"1": "r2", "2": "neg_rmse", "3": "neg_mae"}
            metric_name = metric_map.get(choice, "r2")
        else:
            print("Select metric for classification:")
            print("  [1] Accuracy")
            print("  [2] F1 macro")
            choice = input("Enter choice [1-2]: ").strip()
            metric_map = {"1": "accuracy", "2": "f1_macro"}
            metric_name = metric_map.get(choice, "accuracy")

    # Get metric function
    if metric_name not in METRIC_REGISTRY:
        print(f"[WARN] Unknown metric {metric_name}, using default")
        metric_name = "r2" if task_type == "regression" else "accuracy"
    metric_fn = METRIC_REGISTRY[metric_name]

    # Filter models by task type
    available_models = []
    for model_name, model_info in MODEL_REGISTRY.items():
        family = model_info.get("family", "")
        if task_type == "regression":
            if "regression" in family:
                available_models.append(model_name)
        else:
            if (
                "supervised" in family
                or "classification" in family
                or family == "classical_supervised"
            ):
                available_models.append(model_name)

    if not available_models:
        print(f"[ERROR] No models available for task type: {task_type}")
        return

    print(f"[INFO] Selected {len(available_models)} models for {task_type}: {available_models}\n")

    # Convert to tensors (use numeric features only for now)
    numeric_cols = [c for c in X_df.columns if meta["column_types"].get(c) == "numeric"]
    if len(numeric_cols) > 0:
        X_numeric = X_df[numeric_cols].values.astype(np.float32)
    else:
        X_numeric = X_df.values.astype(np.float32)

    X_tensor = torch.from_numpy(X_numeric)
    if task_type == "regression":
        y_tensor = torch.from_numpy(y.astype(np.float32))
    else:
        y_tensor = torch.from_numpy(y.astype(np.int64))

    # Create train/val split
    n_samples = len(X_tensor)
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Base config
    if task_type == "regression":
        loss_fn = torch.nn.MSELoss()
        out_dim = 1
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        out_dim = meta["n_classes"]

    base_config = {
        "lr": 1e-3,
        "epochs": 20,
        "loss_fn": loss_fn,
        "metric_fn": metric_fn,
        "metric_name": metric_name,
        "patience": 5,
        "task_type": task_type,
    }

    in_dim = X_tensor.shape[1]

    # Load per-model configs
    models_config_path = Path("configs/models.yaml")
    model_cfgs = {}
    if models_config_path.exists():
        with open(models_config_path) as f:
            model_cfgs = yaml.safe_load(f) or {}

    # Search over models
    best_metric = float("-inf")
    best_model_info = None
    results_csv = "experiments/results.csv"

    # Initialize results file
    from utils.logging_utils import init_results_file

    init_results_file(results_csv, include_meta_features=True)

    print(f"[INFO] Training {len(available_models)} models...\n")

    for model_name in available_models:
        print(f"\n=== Training: {model_name} ===")

        try:
            # Build model
            model = build_model(model_name, in_dim, out_dim)

            # Merge config
            config = base_config.copy()
            if model_name in model_cfgs:
                config.update(model_cfgs[model_name])

            # Train
            metric, best_state = train_one_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=config,
            )

            print(f"[INFO] {model_name} achieved {metric_name}={metric:.4f}")

            # Log result
            from utils.logging_utils import log_result

            log_result(
                results_csv,
                model_name,
                metric,
                f"saved_models/{model_name}_best.pth",
                dataset_name=dataset_name,
                meta_features={
                    "n_samples": meta["n_samples"],
                    "n_features": meta["n_features"],
                    "n_classes": meta.get("n_classes"),
                    "class_imbalance": meta.get("class_imbalance"),
                },
                model_family=MODEL_REGISTRY[model_name].get("family", ""),
                metric_name=metric_name,
            )

            if metric > best_metric:
                best_metric = metric
                best_model_info = {
                    "model_name": model_name,
                    "metric": metric,
                    "metric_name": metric_name,
                }

        except Exception as e:
            print(f"[ERROR] Failed to train {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if best_model_info:
        print("\n" + "=" * 60)
        print("🏆 Best Model")
        print("=" * 60)
        print(f"Model: {best_model_info['model_name']}")
        print(f"Metric ({best_model_info['metric_name']}): {best_model_info['metric']:.4f}")
        if task_type == "regression":
            if "neg" in best_model_info["metric_name"]:
                print(f"  (Actual value: {abs(best_model_info['metric']):.4f})")
        print("=" * 60 + "\n")
    else:
        print("\n[WARN] No successful models found.\n")


def print_banner():
    print("=" * 60)
    print("🔥 AutoML Thinking Machine – Interactive Mode")
    print("=" * 60)
    print("This tool can:")
    print("  - Train multiple models and pick the best one")
    print("  - Evaluate with Accuracy or F1 (classification)")
    print("  - Evaluate with R², RMSE, MAE (regression)")
    print("  - Work with Iris (demo), Titanic (Kaggle), or any custom CSV")
    print("  - Search until a target metric is reached (threshold mode)")
    print("  - NEW: Intelligent mode with meta-learning proxy")
    print("  - NEW: Automatic preprocessing strategy search")
    print("  - NEW: Automatic report generation")
    print()
    print("Usage: just pick an option below. No code changes needed.")
    print("=" * 60)
    print()


def main_cli():
    # Set seed for reproducibility (can be overridden by SEED env var)
    seed = get_seed_from_env()
    set_seed(seed)
    if seed != 42:  # Only print if non-default
        print(f"[INFO] Using random seed: {seed}")

    print_banner()

    print("Select an option:")
    print("  [1] Iris demo (Accuracy)")
    print("  [2] Iris demo (F1)")
    print("  [3] Titanic (Accuracy)")
    print("  [4] Titanic (F1)")
    print("  [5] Custom CSV (Auto-detect features)")
    print("  [6] Threshold search (Custom CSV)")
    print("  [7] Stress test suite")
    print("  [8] Dataset-agnostic mode (NEW - with preprocessing)")
    print("  [9] Exit")

    choice = input("\nEnter choice [1-8]: ").strip()

    if choice == "1":
        dataset = "iris"
        metric = "accuracy"
    elif choice == "2":
        dataset = "iris"
        metric = "f1"
    elif choice == "3":
        dataset = "titanic"
        metric = "accuracy"
    elif choice == "4":
        dataset = "titanic"
        metric = "f1"
    elif choice == "5":
        run_custom_csv_experiment()
        return
    elif choice == "6":
        run_custom_csv_threshold_experiment()
        return
    elif choice == "7":
        from examples.stress_test import run_all_stress_tests

        run_all_stress_tests()
        return
    elif choice == "8":
        print("\n" + "=" * 60)
        print("Dataset-Agnostic Mode")
        print("=" * 60)
        print("Choose mode:")
        print("  [1] Basic mode (quick search)")
        print("  [2] Intelligent mode (with meta-learning proxy)")
        mode_choice = input("Enter choice [1-2] (default: 2): ").strip() or "2"

        csv_name = input("Enter CSV file name in data/ (e.g. paddydataset.csv): ").strip()
        csv_path = os.path.join("data", csv_name)

        if not os.path.exists(csv_path):
            print(f"[ERROR] File not found: {csv_path}")
            return

        label_col_input = input("Enter label column name (or press Enter to auto-detect): ").strip()
        label_col = label_col_input if label_col_input else None

        if mode_choice == "2":
            # Use intelligent orchestrator with new config system

            from engine.config import AutoMLConfig
            from engine.orchestrator import run_automl

            target_input = input("Enter target metric (or press Enter to try all models): ").strip()
            target_metric = float(target_input) if target_input else None

            # Create config
            config = AutoMLConfig(
                csv_path=csv_path,
                label_col=label_col,
                mode="intelligent",
                target_metric=target_metric,
                use_meta_learning=True,
                use_proxy_ranking=True,
                max_candidates=20,
                device="cpu",
            )

            run_automl(config)
        else:
            # Use basic mode
            run_dataset_agnostic_experiment(csv_path, label_col, metric_name=None)
        return
    elif choice == "9":
        print("\nExiting. Bye 👋\n")
        return
    else:
        print("\n❌ Invalid choice. Please run again.\n")
        return

    print(f"\n[INFO] You selected: dataset={dataset}, metric={metric}")
    print("This run will:")
    print("  - Load the dataset")
    print("  - Train all models in the zoo with per-model configs")
    print("  - Use early stopping to save compute")
    print("  - Select the best model")
    print("  - Print confusion matrix and metrics")
    print()

    run_experiment(dataset, metric)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoML Thinking Machine")
    parser.add_argument("--autopilot", type=str, help="Path to CSV file (autopilot mode)")
    parser.add_argument("--label", type=str, default=None, help="Label column name")
    parser.add_argument(
        "--metric", type=str, default=None, help="Metric to optimize (accuracy, f1_macro, r2, etc.)"
    )
    parser.add_argument(
        "--target", type=float, default=None, help="Target metric value (stop when reached)"
    )
    parser.add_argument(
        "--max-candidates", type=int, default=20, help="Maximum number of candidates to try"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="intelligent",
        choices=["basic", "intelligent"],
        help="Search mode",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Untouched test fraction")
    parser.add_argument("--cv-folds", type=int, default=5, help="Inner cross-validation folds")
    parser.add_argument("--cv-repeats", type=int, default=1, help="Inner CV repeats")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split/model seed")
    parser.add_argument(
        "--ordering",
        choices=["registry", "random", "proxy", "meta", "proxy_meta"],
        default=None,
        help="Candidate ordering method",
    )
    parser.add_argument(
        "--use-proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable preprocessing proxy scoring",
    )
    parser.add_argument(
        "--use-meta-learning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learned candidate ranking",
    )
    parser.add_argument("--exhaustive", action="store_true", help="Evaluate the full universe")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel candidate workers")

    args, unknown = parser.parse_known_args()

    if args.autopilot:
        # Autopilot mode: run AutoML directly from command line
        import torch

        from engine.config import AutoMLConfig
        from engine.orchestrator import run_automl

        print("\n" + "=" * 60)
        print("🤖 AutoML Autopilot Mode")
        print("=" * 60)

        config = AutoMLConfig(
            csv_path=args.autopilot,
            label_col=args.label,
            metric_name=args.metric,
            target_metric=args.target,
            mode=args.mode,
            max_candidates=args.max_candidates,
            use_meta_learning=args.use_meta_learning,
            use_proxy_ranking=args.use_proxy,
            ordering_mode=args.ordering,
            exhaustive=args.exhaustive,
            test_ratio=args.test_ratio,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            device="cpu",
        )

        run_automl(config)
    else:
        # Interactive mode
        main_cli()
