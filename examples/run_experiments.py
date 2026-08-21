"""
Run experiments comparing accuracy vs F1 on a dataset.
This script runs the AutoML search twice with different metrics.
"""

import sys
from pathlib import Path

# Add current directory to path so we can import main
sys.path.append(str(Path(__file__).parent))

from main import run_experiment


def run_experiment_safe(dataset: str, metric: str):
    """Run experiment and catch errors."""
    print("=" * 60)
    print(f"Experiment: {dataset.upper()} with {metric.upper()} metric")
    print("=" * 60)
    print()

    try:
        run_experiment(dataset, metric)
        return True
    except Exception as e:
        print(f"ERROR: Experiment failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run AutoML experiments with different metrics")
    parser.add_argument(
        "--dataset",
        choices=["iris", "titanic"],
        default="iris",
        help="Dataset to use (default: iris)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["accuracy", "f1", "both"],
        default=["both"],
        help="Metrics to test (default: both)",
    )

    args = parser.parse_args()

    metrics_to_run = []
    if "both" in args.metrics:
        metrics_to_run = ["accuracy", "f1"]
    else:
        metrics_to_run = args.metrics

    print("=" * 60)
    print("AutoML Experiment Runner")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Metrics: {', '.join(metrics_to_run)}")
    print()

    results = {}

    for metric in metrics_to_run:
        success = run_experiment_safe(args.dataset, metric)
        results[metric] = success
        print()

    print("=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    for metric, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{metric.upper()}: {status}")

    print()
    print("Check experiments/results.csv for detailed results")
    print("Compare which model wins under different metrics!")


if __name__ == "__main__":
    main()
