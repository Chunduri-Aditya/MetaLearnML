"""
Train the meta-model (optimization proxy) on historical experiment results.
Run this periodically to update the proxy as you accumulate more experiments.
"""

from pathlib import Path

from meta.meta_model import train_meta_model


def main():
    print("=" * 60)
    print("Training Meta-Model (Optimization Proxy)")
    print("=" * 60)
    print()

    results_csv = "experiments/results.csv"

    if not Path(results_csv).exists():
        print(f"[ERROR] {results_csv} not found.")
        print("Run some experiments first to generate training data.")
        return

    try:
        train_meta_model(results_csv)
        print()
        print("=" * 60)
        print("✓ Meta-model training complete!")
        print("=" * 60)
        print()
        print("The proxy is now ready to predict model performance")
        print("based on dataset meta-features.")
        print()
        print("Next time you run threshold search, it will:")
        print("  - Rank models by predicted performance")
        print("  - Try promising models first")
        print("  - Stop early when target is reached")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to train meta-model: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
