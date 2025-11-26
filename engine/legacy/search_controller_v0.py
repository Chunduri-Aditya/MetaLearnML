"""
LEGACY SEARCH CONTROLLER - DEPRECATED

This module is kept for backward compatibility only.
New code should use engine.intelligent_search.run_intelligent_search() instead.

This was the v0 search system before the refactoring.
It will be removed in a future version.
"""
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from models.model_zoo import MODEL_ZOO, MODEL_REGISTRY, build_model, get_model_entry
from engine.trainer import train_one_model
from utils.logging_utils import init_results_file, log_result, init_results_file_with_preprocessing, log_result_with_preprocessing
from utils.datasets import DatasetDescriptor
from engine.preprocessing import PreprocessingConfig, apply_preprocessing


@dataclass
class SearchCandidate:
    """Represents a (preprocessing, model) combination to search over."""
    family: str              # "neural_supervised" / "classical_supervised"
    model_name: str          # "mlp_small", "rf", ...
    preprocessing_name: str  # "basic_numeric_only", "numeric_plus_cat_onehot", ...


def load_preprocessing_strategies() -> Dict[str, PreprocessingConfig]:
    """Load preprocessing strategies from YAML."""
    config_path = Path("configs/preprocessing.yaml")
    if not config_path.exists():
        print("[WARN] preprocessing.yaml not found, using default strategies")
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    
    strategies = {}
    for name, params in config.get("preprocessing_strategies", {}).items():
        strategies[name] = PreprocessingConfig(name=name, **params)
    
    return strategies


def select_strategies_for_dataset(
    descriptor: DatasetDescriptor,
    all_strategies: Dict[str, PreprocessingConfig]
) -> List[PreprocessingConfig]:
    """
    Select appropriate preprocessing strategies based on dataset meta-features.
    Never hardcode dataset names - use meta-features only.
    """
    types = descriptor.column_types
    
    has_text = any(t == "text" for t in types.values())
    has_cat = any(t == "categorical" for t in types.values())
    has_num = any(t == "numeric" for t in types.values())
    
    selected = []
    
    # Pure numeric datasets
    if has_num and not has_cat and not has_text:
        if "basic_numeric_only" in all_strategies:
            selected.append(all_strategies["basic_numeric_only"])
        if "robust_numeric" in all_strategies:
            selected.append(all_strategies["robust_numeric"])
    
    # Mixed numeric + categorical
    if has_cat:
        if "numeric_plus_cat_onehot" in all_strategies:
            selected.append(all_strategies["numeric_plus_cat_onehot"])
        if "robust_numeric" in all_strategies:
            selected.append(all_strategies["robust_numeric"])
    
    # Text-only datasets
    if has_text and not has_num:
        if "text_tfidf_basic" in all_strategies:
            selected.append(all_strategies["text_tfidf_basic"])
    
    # Fallback: if nothing selected, use basic_numeric_only
    if not selected and "basic_numeric_only" in all_strategies:
        selected.append(all_strategies["basic_numeric_only"])
    
    return selected


def run_model_search_with_preprocessing(
    descriptor: DatasetDescriptor,
    device: str,
    base_config: Dict,
    results_csv: str = "experiments/results.csv",
    ckpt_dir: str = "saved_models",
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Run model search with preprocessing strategies.
    This is the new dataset-agnostic search function.
    
    Args:
        descriptor: Dataset descriptor
        device: Device to use
        base_config: Base training configuration
        results_csv: Path to results CSV
        ckpt_dir: Directory for checkpoints
        train_ratio: Train/val split ratio
    
    Returns:
        Dictionary with best model info including preprocessing
    """
    from torch.utils.data import DataLoader, random_split
    
    # Start with a fresh results file
    results_path = Path(results_csv)
    if results_path.exists():
        results_path.unlink()
    
    # Initialize results file with preprocessing columns
    init_results_file_with_preprocessing(results_csv)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Load preprocessing strategies
    all_strategies = load_preprocessing_strategies()
    selected_strategies = select_strategies_for_dataset(descriptor, all_strategies)
    
    print(f"\n[INFO] Selected preprocessing strategies: {[s.name for s in selected_strategies]}")
    
    # Create train/val split indices
    n_samples = descriptor.n_samples
    n_train = int(train_ratio * n_samples)
    n_val = n_samples - n_train
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()
    
    # Load per-model configs
    models_config_path = Path("configs/models.yaml")
    model_cfgs = {}
    if models_config_path.exists():
        with open(models_config_path) as f:
            model_cfgs = yaml.safe_load(f) or {}
    
    # Generate candidates: (preprocessing, model) pairs
    candidates = []
    for strategy in selected_strategies:
        for model_name in MODEL_ZOO:
            model_family = MODEL_REGISTRY.get(model_name, {}).get("family", "")
            candidates.append(SearchCandidate(
                family=model_family,
                model_name=model_name,
                preprocessing_name=strategy.name,
            ))
    
    print(f"[INFO] Total candidates: {len(candidates)} (preprocessing × models)")
    
    best_overall_metric = float("-inf")
    best_candidate_info = None
    
    # Search over candidates
    for candidate in candidates:
        print(f"\n=== Training: {candidate.preprocessing_name} + {candidate.model_name} ===")
        
        # Get preprocessing config
        preprocessing_config = all_strategies[candidate.preprocessing_name]
        
        # Apply preprocessing
        try:
            X_train, y_train, X_val, y_val, transformer = apply_preprocessing(
                descriptor=descriptor,
                config=preprocessing_config,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            continue
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Update in_dim based on preprocessed features
        in_dim = X_train.shape[1]
        out_dim = descriptor.n_classes
        
        # Build model
        model = build_model(candidate.model_name, in_dim, out_dim)
        
        # Merge base_config with model-specific config
        config = base_config.copy()
        if candidate.model_name in model_cfgs:
            config.update(model_cfgs[candidate.model_name])
        
        # Train model
        metric, best_state = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
        )
        
        # Save checkpoint
        backend = MODEL_REGISTRY.get(candidate.model_name, {}).get("backend", "torch")
        ckpt_path = ckpt_dir / f"{candidate.preprocessing_name}_{candidate.model_name}_best.pth"
        
        if backend == "sklearn":
            from models.sk_models import SkLearnWrapper
            if isinstance(model, SkLearnWrapper):
                sk_path = ckpt_dir / f"{candidate.preprocessing_name}_{candidate.model_name}_best.joblib"
                import joblib
                joblib.dump(model, sk_path, compress=3)
                ckpt_path = sk_path
        else:
            torch.save(best_state, ckpt_path, _use_new_zipfile_serialization=True)
        
        # Log result with preprocessing info
        log_result_with_preprocessing(
            results_csv=results_csv,
            candidate=candidate,
            metric=metric,
            ckpt_path=str(ckpt_path),
            descriptor=descriptor,
            preprocessing_config=preprocessing_config,
            base_config=base_config,
        )
        
        if metric > best_overall_metric:
            best_overall_metric = metric
            best_candidate_info = {
                "model_name": candidate.model_name,
                "preprocessing_name": candidate.preprocessing_name,
                "metric": metric,
                "ckpt_path": str(ckpt_path),
            }
    
    print("\n=== Best pipeline ===")
    if best_candidate_info:
        print(f"Preprocessing: {best_candidate_info['preprocessing_name']}")
        print(f"Model: {best_candidate_info['model_name']}")
        print(f"Metric: {best_candidate_info['metric']:.4f}")
    
    return best_candidate_info or {}


def run_model_search(
    train_loader,
    val_loader,
    in_dim: int,
    out_dim: int,
    device: str,
    base_config: Dict,
    results_csv: str = "experiments/results.csv",
    ckpt_dir: str = "saved_models",
    dataset_name: Optional[str] = None,
    meta_features: Optional[Dict] = None,
) -> Dict[str, any]:
    """
    Run model search across all models in the zoo.
    
    Returns:
        Dictionary with keys: model_name, metric, ckpt_path
    """
    # Start with a fresh results file for each run (self-contained behavior)
    # Later: can add run_id column to track history across runs
    results_path = Path(results_csv)
    if results_path.exists():
        results_path.unlink()
    
    init_results_file(results_csv, include_meta_features=(meta_features is not None))
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load per-model configs from YAML
    models_config_path = Path("configs/models.yaml")
    model_cfgs = {}
    if models_config_path.exists():
        with open(models_config_path) as f:
            model_cfgs = yaml.safe_load(f) or {}
    else:
        print("[WARNING] configs/models.yaml not found, using base_config only")

    best_overall_metric = float("-inf")
    best_model_info = None

    for model_name in MODEL_ZOO:
        print(f"\n=== Training model: {model_name} ===")
        model = build_model(model_name, in_dim, out_dim)

        # Merge base_config with model-specific config
        config = base_config.copy()
        if model_name in model_cfgs:
            config.update(model_cfgs[model_name])
            print(f"[INFO] Using model-specific config: {model_cfgs[model_name]}")

        metric, best_state = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
        )

        # Save checkpoint based on backend
        backend = MODEL_REGISTRY.get(model_name, {}).get("backend", "torch")
        ckpt_path = ckpt_dir / f"{model_name}_best.pth"  # Default for logging
        
        if backend == "sklearn":
            # For sklearn models, save using joblib
            from models.sk_models import SkLearnWrapper
            if isinstance(model, SkLearnWrapper):
                sk_path = ckpt_dir / f"{model_name}_best.joblib"
                import joblib
                joblib.dump(model, sk_path, compress=3)  # Use compression to save space
                ckpt_path = sk_path  # Update path for logging
            else:
                raise ValueError(f"Model {model_name} marked as sklearn but is not SkLearnWrapper")
        else:
            # For PyTorch models, save state dict with compression
            torch.save(best_state, ckpt_path, _use_new_zipfile_serialization=True)

        # Get model family and metric name
        model_family = MODEL_REGISTRY.get(model_name, {}).get("family", "")
        metric_name = base_config.get("metric_name", "")
        
        log_result(
            results_csv, 
            model_name, 
            metric, 
            str(ckpt_path),
            dataset_name=dataset_name,
            meta_features=meta_features,
            model_family=model_family,
            metric_name=metric_name,
        )

        if metric > best_overall_metric:
            best_overall_metric = metric
            best_model_info = {
                "model_name": model_name,
                "metric": metric,
                "ckpt_path": str(ckpt_path),
            }

    print("\n=== Best model ===")
    print(best_model_info)
    return best_model_info


def train_and_eval_single_model(
    model_name: str,
    train_loader,
    val_loader,
    in_dim: int,
    out_dim: int,
    device: str,
    base_config: Dict,
    ckpt_dir: str = "saved_models",
):
    """
    Train a single model and return its metric + checkpoint path.
    Respects per-model overrides from models.yaml.
    
    Returns:
        dict with keys: model_name, metric, ckpt_path
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Load per-model configs from YAML
    models_config_path = Path("configs/models.yaml")
    model_cfgs = {}
    if models_config_path.exists():
        with open(models_config_path) as f:
            model_cfgs = yaml.safe_load(f) or {}
    
    model = build_model(model_name, in_dim, out_dim)
    
    # Merge base_config with model-specific config
    config = base_config.copy()
    if model_name in model_cfgs:
        config.update(model_cfgs[model_name])
    
    metric, best_state = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
    )
    
    # Save checkpoint based on backend
    backend = MODEL_REGISTRY.get(model_name, {}).get("backend", "torch")
    ckpt_path = ckpt_dir / f"{model_name}_best.pth"  # Default path for logging
    
    if backend == "sklearn":
        # For sklearn models, save using joblib
        from models.sk_models import SkLearnWrapper
        if isinstance(model, SkLearnWrapper):
            sk_path = ckpt_dir / f"{model_name}_best.joblib"
            import joblib
            # Save the entire wrapper (includes fitted sklearn model + scaler)
            joblib.dump(model, sk_path, compress=3)  # Use compression
            ckpt_path = sk_path  # Update path for logging
        else:
            raise ValueError(f"Model {model_name} marked as sklearn but is not SkLearnWrapper")
    else:
        # For PyTorch models, save state dict with compression
        torch.save(best_state, ckpt_path, _use_new_zipfile_serialization=True)
    
    return {
        "model_name": model_name,
        "metric": metric,
        "ckpt_path": str(ckpt_path),
    }


def run_model_search_until_threshold(
    train_loader,
    val_loader,
    in_dim: int,
    out_dim: int,
    device: str,
    base_config: Dict,
    results_csv: str,
    target_metric: float,
    family_order: Optional[List[str]] = None,
    ckpt_dir: str = "saved_models",
    dataset_name: Optional[str] = None,
    meta_features: Optional[Dict] = None,
    use_proxy: bool = True,
):
    """
    Try models in a specified family order until:
      - a model reaches target_metric or
      - we run out of models.
    
    Uses meta-model proxy to rank models within each family if available.
    
    Args:
        family_order: List of family names to try in order.
                     Default: ["neural_supervised"]
        dataset_name: Name of dataset (for meta-learning)
        meta_features: Meta-features dict (for proxy ranking)
        use_proxy: Whether to use meta-model proxy for ranking
    
    Returns:
        {
            "best_info": {...},
            "reached_target": bool,
            "tried_models": [list of model names in order],
            "proxy_gaps": [list of (model_name, predicted, actual, gap)]
        }
    """
    if family_order is None:
        family_order = ["neural_supervised"]  # default
    
    # Start with a fresh results file
    results_path = Path(results_csv)
    if results_path.exists():
        results_path.unlink()
    init_results_file(results_csv, include_meta_features=(meta_features is not None))
    
    # Get proxy scores if available
    proxy_scores = {}
    if use_proxy and meta_features is not None and dataset_name is not None:
        try:
            from meta.meta_model import predict_config_scores_from_meta_features
            all_models = list(MODEL_REGISTRY.keys())
            proxy_scores = predict_config_scores_from_meta_features(
                dataset_name=dataset_name,
                candidate_models=all_models,
                meta_features=meta_features,
            )
            print(f"[INFO] Meta-model proxy loaded. Predicted scores:")
            for name, score in sorted(proxy_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {score:.4f}")
        except Exception as e:
            print(f"[WARN] Could not load proxy: {e}. Proceeding without proxy ranking.")
            proxy_scores = {}
    
    # Order models by family, then by proxy score (descending)
    ordered_models = []
    for fam in family_order:
        fam_models = [
            (name, MODEL_REGISTRY[name])
            for name, entry in MODEL_REGISTRY.items()
            if entry["family"] == fam
        ]
        
        # Sort by proxy score if available
        if proxy_scores:
            fam_models.sort(key=lambda x: proxy_scores.get(x[0], 0.0), reverse=True)
            print(f"[INFO] Family '{fam}' models ordered by proxy: {[m[0] for m in fam_models]}")
        
        ordered_models.extend([name for name, _ in fam_models])
    
    if not ordered_models:
        raise ValueError(f"No models found for families: {family_order}")
    
    best_info = None
    best_metric = float("-inf")
    tried = []
    proxy_gaps = []  # Track predicted vs actual for outlier detection
    big_gap_count = 0
    BIG_GAP_THRESHOLD = 0.2  # Consider gap "big" if abs(gap) > this
    MAX_BIG_GAPS = 2  # After this many big gaps, fallback to brute-force
    proxy_unreliable = False
    
    for model_name in ordered_models:
        print(f"\n=== Threshold search – training model: {model_name} ===")
        
        # Show proxy prediction if available
        predicted_metric = proxy_scores.get(model_name)
        if predicted_metric is not None and not proxy_unreliable:
            print(f"[PROXY] Predicted metric: {predicted_metric:.4f}")
        
        result = train_and_eval_single_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            in_dim=in_dim,
            out_dim=out_dim,
            device=device,
            base_config=base_config,
            ckpt_dir=ckpt_dir,
        )
        
        tried.append(model_name)
        
        metric = result["metric"]
        print(f"[INFO] {model_name} achieved metric={metric:.4f}")
        
        # Track proxy gap and check reliability
        if predicted_metric is not None and not proxy_unreliable:
            gap = predicted_metric - metric
            proxy_gaps.append((model_name, predicted_metric, metric, gap))
            print(f"[PROXY] Gap (predicted - actual): {gap:.4f}")
            
            # Check if gap is too large (proxy unreliable)
            if abs(gap) > BIG_GAP_THRESHOLD:
                big_gap_count += 1
                if big_gap_count >= MAX_BIG_GAPS:
                    print(f"\n[WARN] Proxy seems unreliable on this dataset ({big_gap_count} large gaps detected).")
                    print("[INFO] Falling back to brute-force search order (ignoring proxy rankings).")
                    proxy_unreliable = True
                    # Rebuild ordered_models without proxy ranking
                    ordered_models_remaining = []
                    for fam in family_order:
                        for name, entry in MODEL_REGISTRY.items():
                            if entry["family"] == fam and name not in tried:
                                ordered_models_remaining.append(name)
                    ordered_models = ordered_models_remaining
                    print(f"[INFO] Remaining models to try: {ordered_models}")
        
        # Log to results.csv with meta-features
        model_family = MODEL_REGISTRY.get(model_name, {}).get("family", "")
        metric_name = base_config.get("metric_name", "")
        
        log_result(
            results_csv, 
            model_name, 
            metric, 
            result["ckpt_path"],
            dataset_name=dataset_name,
            meta_features=meta_features,
            model_family=model_family,
            metric_name=metric_name,
        )
        
        if metric > best_metric:
            best_metric = metric
            best_info = result
        
        if metric >= target_metric:
            print(f"\n[SUCCESS] Target metric {target_metric:.4f} reached by {model_name}!")
            return {
                "best_info": result,
                "reached_target": True,
                "tried_models": tried,
                "proxy_gaps": proxy_gaps,
                "proxy_unreliable": proxy_unreliable,
            }
    
    print(f"\n[WARN] Target metric {target_metric:.4f} NOT reached.")
    print(f"[WARN] Best we got: {best_metric:.4f} from {best_info['model_name']}")
    
    # Soft fail recommendations
    if best_info is not None:
        ratio = best_metric / target_metric if target_metric > 0 else 0
        print("\n[INFO] Performance Analysis:")
        if ratio >= 0.95:
            print("  ✓ This is very close to target (≥95%). Consider accepting this result.")
        elif ratio >= 0.85:
            print("  ⚠ This is reasonably close (≥85%). May be worth trying more models or tuning hyperparameters.")
        elif ratio >= 0.70:
            print("  ⚠ This is below target (70-85%). Consider:")
            print("    - Expanding model zoo (add more architectures)")
            print("    - Feature engineering")
            print("    - Different metrics or loss functions")
        else:
            print("  ✗ This is far from target (<70%). Consider:")
            print("    - Dataset may be too difficult for current model zoo")
            print("    - Check data quality and preprocessing")
            print("    - Try different model families (classical, semi-supervised)")
    
    # Outlier detection summary
    if proxy_gaps:
        large_gaps = [(m, g) for m, _, _, g in proxy_gaps if abs(g) > BIG_GAP_THRESHOLD]
        if large_gaps:
            print(f"\n[INFO] Found {len(large_gaps)} models with large proxy gaps:")
            for model, gap in large_gaps:
                print(f"  {model}: gap = {gap:.4f}")
    
    return {
        "best_info": best_info,
        "reached_target": False,
        "tried_models": tried,
        "proxy_gaps": proxy_gaps,
        "proxy_unreliable": proxy_unreliable,
    }

