"""
Full pipeline orchestrator - the "thinking machine" entry point.
"""

import time
import uuid
from typing import Dict, Optional, Any
from pathlib import Path

import pandas as pd
from utils.datasets import load_tabular_dataset
from engine.intelligent_search import run_intelligent_search
from engine.intelligent_search_parallel import run_intelligent_search_parallel
from engine.report_generator import generate_report
from engine.meta_learner import MetaLearner
from engine.config import AutoMLConfig
from models.model_zoo import MODEL_REGISTRY


def run_automl(config: AutoMLConfig) -> Dict[str, Any]:
    """
    Full AutoML pipeline orchestrator.
    
    This is the main entry point that implements the complete "thinking machine" behavior.
    All logic is centralized here - main.py is just a UI wrapper.
    
    Args:
        config: AutoMLConfig with all parameters
    
    Returns:
        Dictionary with results and report paths
    """
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("🧠 AutoML Thinking Machine")
    print("=" * 60)
    print(f"Loading dataset: {config.csv_path}\n")
    
    # Phase 1: Load and understand dataset
    dataset_info = load_tabular_dataset(config.csv_path, label_col=config.label_col)
    task_type = dataset_info["task_type"] or config.task_type
    X_df = dataset_info["X_df"]
    y = dataset_info["y"]
    meta = dataset_info["meta"]
    dataset_name = dataset_info["dataset_name"] or config.dataset_name or Path(config.csv_path).stem
    search_meta = dict(meta)
    search_meta["dataset_name"] = dataset_name
    
    print(f"[INFO] Task type: {task_type}")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Samples: {meta['n_samples']}, Features: {meta['n_features']}")
    
    # Phase 2: Select metric if not provided
    metric_name = config.metric_name
    if metric_name is None:
        # Auto-select based on task type
        if task_type == "regression":
            metric_name = "r2"  # Default for regression
        else:
            metric_name = "accuracy"  # Default for classification
    
    print(f"[INFO] Using metric: {metric_name}")
    
    # Phase 3: Run intelligent search
    print("\n" + "=" * 60)
    print("Starting Intelligent Search")
    print("=" * 60)
    
    # Use parallel search if n_jobs > 1, otherwise sequential
    if config.n_jobs != 1 and config.n_jobs != 0:
        search_results = run_intelligent_search_parallel(
            X_df=X_df,
            y=y,
            task_type=task_type,
            metric_name=metric_name,
            dataset_meta=search_meta,
            device=config.device,
            target_metric=config.target_metric,
            max_models=config.max_candidates,
            top_k_preprocessing=config.top_k_preprocessing,
            use_proxy=config.use_proxy_ranking,
            use_meta_learning=config.use_meta_learning,
            n_jobs=config.n_jobs,
        )
    else:
        search_results = run_intelligent_search(
            X_df=X_df,
            y=y,
            task_type=task_type,
            metric_name=metric_name,
            dataset_meta=search_meta,
            device=config.device,
            target_metric=config.target_metric,
            max_models=config.max_candidates,
            top_k_preprocessing=config.top_k_preprocessing,
            use_proxy=config.use_proxy_ranking,
            use_meta_learning=config.use_meta_learning,
        )
    
    best_info = search_results.get("best_info")
    tried = search_results.get("tried", [])
    
    # Phase 3.5: Log to Neo4j (if available)
    if best_info:
        try:
            from graph.neo4j_client import Neo4jGraph
            
            graph = Neo4jGraph()
            if graph.is_connected():
                run_id = str(uuid.uuid4())
                
                # Upsert dataset
                graph.upsert_dataset(dataset_name, {
                    "n_samples": meta.get("n_samples", 0),
                    "n_features": meta.get("n_features", 0),
                    "task_type": task_type,
                    "n_numeric": meta.get("n_numeric", 0),
                    "n_categorical": meta.get("n_categorical", 0),
                    "n_classes": meta.get("n_classes"),
                })
                
                # Upsert model
                model_name = best_info["model_name"]
                model_entry = MODEL_REGISTRY.get(model_name, {})
                graph.upsert_model(
                    model_name,
                    model_entry.get("family", "unknown"),
                    model_entry.get("backend", "unknown"),
                )
                
                # Upsert preprocessor
                preproc_name = best_info.get("preprocessing_name", "unknown")
                graph.upsert_preprocessor(preproc_name)
                
                # Create run
                graph.create_run(
                    run_id,
                    best_info["metric_name"],
                    best_info["metric"],
                )
                
                # Link relationships
                graph.link_run(dataset_name, model_name, preproc_name, run_id)
                
                graph.close()
                print(f"[INFO] Logged run to Neo4j: {run_id}")
        except Exception as e:
            print(f"[WARN] Could not log to Neo4j: {e}")
    
    # Phase 4: Update meta-learner (if enabled)
    if config.use_meta_learning:
        print("\n[INFO] Updating meta-learning brain...")
        meta_learner = MetaLearner(
            task_type=task_type,
            metric_name=metric_name,
        )
        history = meta_learner.load_history()
        
        # Add new results to history
        new_rows = []
        for model_name, prep_name, metric_val in tried:
            new_rows.append({
                "dataset_name": dataset_name,
                "task_type": task_type,
                "n_samples": meta.get("n_samples", 0),
                "n_features": meta.get("n_features", 0),
                "n_numeric": meta.get("n_numeric", 0),
                "n_categorical": meta.get("n_categorical", 0),
                "n_text": meta.get("n_text", 0),
                "n_classes": meta.get("n_classes", 0),
                "class_imbalance": meta.get("class_imbalance", 0),
                "missing_value_fraction": meta.get("missing_value_fraction", 0),
                "model_name": model_name,
                "preprocessing_name": prep_name,
                "metric": metric_val,
                "metric_name": metric_name,
            })
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if len(history) > 0:
                history = pd.concat([history, new_df], ignore_index=True)
            else:
                history = new_df
            
            # Save updated history
            meta_learner.meta_data_path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(meta_learner.meta_data_path, index=False)
            
            # Retrain meta-model (per-task+metric)
            meta_learner.train(history)
            meta_learner.save()
    
    # Phase 5: Generate report
    time_taken = time.time() - start_time
    
    all_results = [
        {"model_name": m, "preprocessing_name": p, "metric": met}
        for m, p, met in tried
    ]
    
    report_paths = generate_report(
        dataset_name=dataset_name,
        task_type=task_type,
        best_info=best_info or {},
        all_results=all_results,
        dataset_meta=meta,
        time_taken=time_taken,
        output_dir=config.report_dir,
    )
    
    # Phase 6: Print summary
    print("\n" + "=" * 60)
    print("🏆 Final Results")
    print("=" * 60)
    if best_info:
        print(f"Best Model: {best_info['model_name']}")
        print(f"Preprocessing: {best_info.get('preprocessing_name', 'N/A')}")
        print(f"Metric ({best_info['metric_name']}): {best_info['metric']:.4f}")
    print(f"Models Tried: {len(tried)}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    print(f"Reports: {list(report_paths.values())}")
    print("=" * 60)
    
    return {
        "best_info": best_info,
        "all_results": all_results,
        "report_paths": report_paths,
        "time_taken": time_taken,
    }
