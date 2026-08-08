"""
Parallel version of intelligent search for faster candidate evaluation.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from pathlib import Path
import yaml
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed

from models.model_zoo import MODEL_REGISTRY, build_model
from engine.trainer import train_one_model
from engine.evaluator import METRIC_REGISTRY
from engine.meta_learner import MetaLearner
from engine.preprocessing_strategies import (
    generate_preprocessing_strategies,
    build_preprocessing_pipeline,
    rank_preprocessors_by_proxy,
    PreprocessingStrategy,
)
from utils.logging_utils import init_results_file, log_result


def _evaluate_candidate_wrapper(
    candidate: Tuple[str, PreprocessingStrategy],
    X_df,
    y: np.ndarray,
    train_indices: List[int],
    val_indices: List[int],
    task_type: str,
    metric_name: str,
    dataset_meta: Dict,
    device: str,
    model_cfgs: Dict,
    ckpt_dir: str,
) -> Tuple[str, str, float, Optional[Dict], Optional[str]]:
    """
    Wrapper function for parallel candidate evaluation.
    
    Returns:
        (model_name, preprocessing_name, metric, best_info_dict, ckpt_path)
    """
    model_name, strategy = candidate
    
    try:
        # Apply preprocessing
        numeric_cols = [c for c in X_df.columns if dataset_meta['column_types'].get(c) == "numeric"]
        categorical_cols = [c for c in X_df.columns if dataset_meta['column_types'].get(c) == "categorical"]
        
        pipeline = build_preprocessing_pipeline(strategy, numeric_cols, categorical_cols)
        
        X_train_df = X_df.iloc[train_indices]
        X_val_df = X_df.iloc[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        if pipeline is None:
            X_train_proc = X_train_df.values.astype(np.float32)
            X_val_proc = X_val_df.values.astype(np.float32)
        else:
            X_train_proc = pipeline.fit_transform(X_train_df).astype(np.float32)
            X_val_proc = pipeline.transform(X_val_df).astype(np.float32)
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train_proc)
        X_val_t = torch.from_numpy(X_val_proc)
        
        if task_type == "regression":
            y_train_t = torch.from_numpy(y_train.astype(np.float32))
            y_val_t = torch.from_numpy(y_val.astype(np.float32))
            out_dim = 1
        else:
            y_train_t = torch.from_numpy(y_train.astype(np.int64))
            y_val_t = torch.from_numpy(y_val.astype(np.int64))
            out_dim = dataset_meta.get("n_classes", 2)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Build model
        in_dim = X_train_proc.shape[1]
        model = build_model(model_name, in_dim, out_dim)
        
        # Config
        if task_type == "regression":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        metric_fn = METRIC_REGISTRY[metric_name]
        
        config = {
            "lr": 1e-3,
            "epochs": 20,
            "loss_fn": loss_fn,
            "metric_fn": metric_fn,
            "metric_name": metric_name,
            "patience": 5,
            "task_type": task_type,
        }
        
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

        # Save checkpoint
        ckpt_dir_path = Path(ckpt_dir)
        ckpt_dir_path.mkdir(parents=True, exist_ok=True)
        backend = MODEL_REGISTRY.get(model_name, {}).get("backend", "torch")
        ckpt_path = ckpt_dir_path / f"{strategy.name}_{model_name}_best.pth"
        if backend == "sklearn":
            from models.sk_models import SkLearnWrapper
            if isinstance(model, SkLearnWrapper):
                import joblib
                sk_path = ckpt_dir_path / f"{strategy.name}_{model_name}_best.joblib"
                joblib.dump(model, sk_path, compress=3)
                ckpt_path = sk_path
        else:
            torch.save(best_state, ckpt_path, _use_new_zipfile_serialization=True)
        
        best_info = {
            "model_name": model_name,
            "preprocessing_name": strategy.name,
            "metric": metric,
            "metric_name": metric_name,
            "ckpt_path": str(ckpt_path),
        }
        
        return (model_name, strategy.name, metric, best_info, str(ckpt_path))
    
    except Exception as e:
        # Return failure indicator
        return (model_name, strategy.name, float("-inf"), None, None)


def run_intelligent_search_parallel(
    X_df,
    y: np.ndarray,
    task_type: str,
    metric_name: str,
    dataset_meta: Dict,
    device: str = "cpu",
    target_metric: Optional[float] = None,
    max_models: int = 20,
    top_k_preprocessing: int = 3,
    use_proxy: bool = True,
    use_meta_learning: bool = True,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Parallel version of intelligent search.
    
    Args:
        n_jobs: Number of parallel jobs (-1 = use all cores)
    """
    # Get metric function
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_name}")
    metric_fn = METRIC_REGISTRY[metric_name]
    
    # Filter models by task type
    available_models = []
    for model_name, model_info in MODEL_REGISTRY.items():
        family = model_info.get("family", "")
        if task_type == "regression":
            if "regression" in family:
                available_models.append(model_name)
        else:
            if "supervised" in family or "classification" in family or family == "classical_supervised" or "svm" in family:
                available_models.append(model_name)
    
    if not available_models:
        raise ValueError(f"No models available for task type: {task_type}")
    
    print(f"[INFO] Found {len(available_models)} models for {task_type}")
    
    # Generate preprocessing strategies
    has_numeric = dataset_meta.get("n_numeric", 0) > 0
    has_categorical = dataset_meta.get("n_categorical", 0) > 0
    n_samples = dataset_meta.get("n_samples", 0)
    
    all_strategies = generate_preprocessing_strategies(
        has_numeric, 
        has_categorical, 
        n_samples,
        dataset_meta=dataset_meta,
    )
    print(f"[INFO] Generated {len(all_strategies)} preprocessing strategies")
    
    # Rank preprocessing strategies using proxy
    if len(all_strategies) > top_k_preprocessing and use_proxy:
        print(f"[INFO] Ranking preprocessing strategies using proxy evaluation...")
        ranked_strategies = rank_preprocessors_by_proxy(
            X_df, y, all_strategies, task_type, metric_fn, sample_size=min(512, n_samples)
        )
        top_strategies = [s for s, _ in ranked_strategies[:top_k_preprocessing]]
        print(f"[INFO] Selected top {len(top_strategies)} preprocessing strategies")
    else:
        top_strategies = all_strategies[:top_k_preprocessing]
    
    # Generate candidates
    candidates = []
    for strategy in top_strategies:
        for model_name in available_models:
            candidates.append((model_name, strategy))
    
    print(f"[INFO] Total candidates: {len(candidates)} (preprocessing × models)")
    
    # Use meta-learner to rank candidates if available
    meta_learner = None
    if use_meta_learning:
        meta_learner = MetaLearner(
            task_type=task_type,
            metric_name=metric_name,
        )
        meta_learner.load()
    
    if meta_learner and meta_learner.is_trained and use_proxy:
        print("[INFO] Using meta-learning proxy to rank candidates...")
        ranked_candidates = meta_learner.rank_candidates_with_fallback(dataset_meta, candidates)
        candidates = [(m, p) for m, p, _ in ranked_candidates]
        print(f"[INFO] Top 3 predicted: {[(m, p.name) for m, p, s in ranked_candidates[:3]]}")
    
    # Limit candidates
    if len(candidates) > max_models:
        candidates = candidates[:max_models]
        print(f"[INFO] Limited to top {max_models} candidates")
    
    # Create train/val split
    n_samples = len(X_df)
    n_train = int(0.8 * n_samples)
    n_val = n_samples - n_train
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()

    # Initialize results
    results_csv = "experiments/results.csv"
    results_path = Path(results_csv)
    if results_path.exists():
        results_path.unlink()
    init_results_file(results_csv, include_meta_features=True)
    ckpt_dir = "saved_models"
    
    # Load model configs
    models_config_path = Path("configs/models.yaml")
    model_cfgs = {}
    if models_config_path.exists():
        with open(models_config_path) as f:
            model_cfgs = yaml.safe_load(f) or {}
    
    # Parallel evaluation
    print(f"[INFO] Training {len(candidates)} candidates in parallel (n_jobs={n_jobs})...")
    
    # Separate sklearn and torch models for safer parallelization
    sklearn_candidates = [(m, s) for m, s in candidates if MODEL_REGISTRY[m].get("backend") == "sklearn"]
    torch_candidates = [(m, s) for m, s in candidates if MODEL_REGISTRY[m].get("backend") == "torch"]
    
    results = []
    
    # Parallelize sklearn models (safer, no GPU conflicts)
    if sklearn_candidates and n_jobs != 1:
        print(f"[INFO] Training {len(sklearn_candidates)} sklearn models in parallel...")
        sklearn_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_evaluate_candidate_wrapper)(
                candidate, X_df, y, train_indices, val_indices, task_type,
                metric_name, dataset_meta, device, model_cfgs, ckpt_dir
            )
            for candidate in sklearn_candidates
        )
        results.extend(sklearn_results)
    
    # Sequential for torch models (to avoid GPU conflicts)
    if torch_candidates:
        print(f"[INFO] Training {len(torch_candidates)} PyTorch models sequentially...")
        for candidate in torch_candidates:
            result = _evaluate_candidate_wrapper(
                candidate, X_df, y, train_indices, val_indices, task_type,
                metric_name, dataset_meta, device, model_cfgs, ckpt_dir
            )
            results.append(result)
    
    # Process results
    best_metric = float("-inf")
    best_info = None
    tried = []
    
    for model_name, prep_name, metric, info, ckpt_path in results:
        if metric > float("-inf"):
            tried.append((model_name, prep_name, metric))
            print(f"[INFO] {prep_name} + {model_name}: {metric_name}={metric:.4f}")

            log_result(
                path=results_csv,
                model_name=model_name,
                metric=metric,
                ckpt_path=ckpt_path or "",
                dataset_name=dataset_meta.get("dataset_name", "unknown_dataset"),
                meta_features={
                    "n_samples": dataset_meta.get("n_samples", ""),
                    "n_features": dataset_meta.get("n_features", ""),
                    "n_numeric": dataset_meta.get("n_numeric", ""),
                    "n_categorical": dataset_meta.get("n_categorical", ""),
                    "n_classes": dataset_meta.get("n_classes", ""),
                    "class_imbalance": dataset_meta.get("class_imbalance", ""),
                },
                model_family=MODEL_REGISTRY[model_name].get("family", ""),
                metric_name=metric_name,
            )
            
            if metric > best_metric:
                best_metric = metric
                best_info = info
            
            # Check if target reached
            if target_metric is not None and metric >= target_metric:
                print(f"\n[SUCCESS] Target metric {target_metric:.4f} reached!")
                break
    
    return {
        "best_info": best_info,
        "tried": tried,
        "reached_target": target_metric is not None and best_metric >= target_metric if best_info else False,
    }
