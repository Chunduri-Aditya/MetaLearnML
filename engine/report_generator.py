"""
Automatic report generation for AutoML experiments.
"""

from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


def generate_report(
    dataset_name: str,
    task_type: str,
    best_info: Dict,
    all_results: List[Dict],
    dataset_meta: Dict,
    time_taken: float,
    output_dir: str = "experiments/reports",
) -> Dict[str, str]:
    """
    Generate comprehensive report for AutoML run.
    
    Returns:
        Dictionary with paths to generated reports (markdown, json, text)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset_name}_{timestamp}"
    
    # Generate Markdown report
    md_path = output_dir / f"{base_name}.md"
    md_content = generate_markdown_report(
        dataset_name, task_type, best_info, all_results, dataset_meta, time_taken
    )
    md_path.write_text(md_content)
    
    # Generate JSON report
    json_path = output_dir / f"{base_name}.json"
    json_content = generate_json_report(
        dataset_name, task_type, best_info, all_results, dataset_meta, time_taken
    )
    
    # Remove non-serializable objects from dataset_meta before JSON serialization
    # This includes LabelEncoder and other sklearn objects
    if "dataset_meta" in json_content:
        # Remove label_encoder (not JSON serializable)
        json_content["dataset_meta"].pop("label_encoder", None)
        # Remove any other sklearn/preprocessing objects
        keys_to_remove = []
        for key, value in json_content["dataset_meta"].items():
            # Check if it's a non-serializable object
            if hasattr(value, '__class__') and not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                # Check if it's a numpy/scikit-learn object
                import numpy as np
                if isinstance(value, (np.ndarray, np.generic)):
                    continue  # Will be handled by convert_to_native
                # It's likely a custom object (LabelEncoder, etc.)
                keys_to_remove.append(key)
        for key in keys_to_remove:
            json_content["dataset_meta"].pop(key, None)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle non-serializable objects (LabelEncoder, etc.)
        if hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            if class_name in ['LabelEncoder', 'StandardScaler', 'OneHotEncoder', 'SimpleImputer']:
                # Return string representation instead of object
                return f"{class_name}()"
        
        # Handle collections
        if isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        
        return obj
    
    json_content = convert_to_native(json_content)
    json_path.write_text(json.dumps(json_content, indent=2))
    
    # Generate text report
    txt_path = output_dir / f"{base_name}.txt"
    txt_content = generate_text_report(
        dataset_name, task_type, best_info, all_results, dataset_meta, time_taken
    )
    txt_path.write_text(txt_content)
    
    print(f"\n[INFO] Reports generated:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Text: {txt_path}")
    
    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "text": str(txt_path),
    }


def generate_markdown_report(
    dataset_name: str,
    task_type: str,
    best_info: Dict,
    all_results: List[Dict],
    dataset_meta: Dict,
    time_taken: float,
) -> str:
    """Generate Markdown report."""
    lines = []
    lines.append(f"# AutoML Experiment Report: {dataset_name}")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n## Dataset Overview\n")
    lines.append(f"- **Task Type:** {task_type}")
    lines.append(f"- **Samples:** {dataset_meta.get('n_samples', 'N/A')}")
    lines.append(f"- **Features:** {dataset_meta.get('n_features', 'N/A')}")
    lines.append(f"- **Numeric Features:** {dataset_meta.get('n_numeric', 'N/A')}")
    lines.append(f"- **Categorical Features:** {dataset_meta.get('n_categorical', 'N/A')}")
    if task_type != "regression":
        lines.append(f"- **Classes:** {dataset_meta.get('n_classes', 'N/A')}")
        lines.append(f"- **Class Imbalance:** {dataset_meta.get('class_imbalance', 0):.3f}")
    
    lines.append(f"\n## Best Model\n")
    if best_info:
        lines.append(f"- **Model:** {best_info['model_name']}")
        lines.append(f"- **Preprocessing:** {best_info.get('preprocessing_name', 'N/A')}")
        lines.append(f"- **Metric ({best_info['metric_name']}):** {best_info['metric']:.4f}")
    
    lines.append(f"\n## All Models Tried\n")
    lines.append("| Model | Preprocessing | Metric |")
    lines.append("|-------|---------------|--------|")
    for result in sorted(all_results, key=lambda x: x.get('metric', 0), reverse=True):
        lines.append(f"| {result.get('model_name', 'N/A')} | {result.get('preprocessing_name', 'N/A')} | {result.get('metric', 0):.4f} |")
    
    lines.append(f"\n## Performance Summary\n")
    lines.append(f"- **Total Time:** {time_taken:.2f} seconds")
    lines.append(f"- **Models Tried:** {len(all_results)}")
    
    lines.append(f"\n## Next Steps\n")
    if best_info:
        lines.append(f"1. The best model is **{best_info['model_name']}** with preprocessing **{best_info.get('preprocessing_name', 'N/A')}**")
        lines.append(f"2. Consider fine-tuning hyperparameters for further improvement")
        lines.append(f"3. Evaluate on a held-out test set for final validation")
    
    return "\n".join(lines)


def generate_json_report(
    dataset_name: str,
    task_type: str,
    best_info: Dict,
    all_results: List[Dict],
    dataset_meta: Dict,
    time_taken: float,
) -> Dict:
    """Generate JSON report."""
    return {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "timestamp": datetime.now().isoformat(),
        "dataset_meta": dataset_meta,
        "best_model": best_info,
        "all_results": all_results,
        "performance": {
            "time_taken": time_taken,
            "models_tried": len(all_results),
        },
    }


def generate_text_report(
    dataset_name: str,
    task_type: str,
    best_info: Dict,
    all_results: List[Dict],
    dataset_meta: Dict,
    time_taken: float,
) -> str:
    """Generate plain text report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"AutoML Experiment Report: {dataset_name}")
    lines.append("=" * 60)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nDataset Overview:")
    lines.append(f"  Task Type: {task_type}")
    lines.append(f"  Samples: {dataset_meta.get('n_samples', 'N/A')}")
    lines.append(f"  Features: {dataset_meta.get('n_features', 'N/A')}")
    
    lines.append(f"\nBest Model:")
    if best_info:
        lines.append(f"  Model: {best_info['model_name']}")
        lines.append(f"  Preprocessing: {best_info.get('preprocessing_name', 'N/A')}")
        lines.append(f"  Metric ({best_info['metric_name']}): {best_info['metric']:.4f}")
    
    lines.append(f"\nAll Models (sorted by performance):")
    for i, result in enumerate(sorted(all_results, key=lambda x: x.get('metric', 0), reverse=True), 1):
        lines.append(f"  {i}. {result.get('model_name', 'N/A')} + {result.get('preprocessing_name', 'N/A')}: {result.get('metric', 0):.4f}")
    
    lines.append(f"\nPerformance:")
    lines.append(f"  Time Taken: {time_taken:.2f} seconds")
    lines.append(f"  Models Tried: {len(all_results)}")
    lines.append("=" * 60)
    
    return "\n".join(lines)

