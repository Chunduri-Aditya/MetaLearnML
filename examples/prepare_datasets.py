"""
Generate redistributable sklearn example datasets for local use.

The generated CSV files are intentionally ignored by Git. Published benchmark data is
instead fetched and checksum-validated through ``make benchmark-fetch``.
"""

from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_iris, load_wine


def prepare_iris(output_dir: str = "data"):
    """Prepare Iris dataset (3-class classification, 4 features, 150 samples)."""
    print("Preparing Iris dataset...")
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Rename target to label for consistency
    df.rename(columns={"target": "label"}, inplace=True)

    # Save to CSV
    output_path = Path(output_dir) / "iris.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Saved Iris dataset to {output_path}")
    print(f"  Features: {list(df.columns[:-1])}")
    print("  Label column: label")
    print(f"  Samples: {len(df)}, Classes: {df['label'].nunique()}")
    return output_path


def prepare_wine(output_dir: str = "data"):
    """Prepare Wine dataset (3-class classification, 13 features, 178 samples)."""
    print("Preparing Wine dataset...")
    wine = load_wine(as_frame=True)
    df = wine.frame

    df.rename(columns={"target": "label"}, inplace=True)

    output_path = Path(output_dir) / "wine.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Saved Wine dataset to {output_path}")
    print(f"  Features: {list(df.columns[:-1])}")
    print("  Label column: label")
    print(f"  Samples: {len(df)}, Classes: {df['label'].nunique()}")
    return output_path


def prepare_breast_cancer(output_dir: str = "data"):
    """Prepare Breast Cancer dataset (binary classification, 30 features, 569 samples)."""
    print("Preparing Breast Cancer dataset...")
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame

    df.rename(columns={"target": "label"}, inplace=True)

    output_path = Path(output_dir) / "breast_cancer.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Saved Breast Cancer dataset to {output_path}")
    print(f"  Features: {list(df.columns[:-1])}")
    print("  Label column: label")
    print(f"  Samples: {len(df)}, Classes: {df['label'].nunique()}")
    return output_path


def main():
    """Prepare all available datasets."""
    print("=" * 60)
    print("Preparing datasets for AutoML Project")
    print("=" * 60)
    print()

    prepare_iris()
    print()
    prepare_wine()
    print()
    prepare_breast_cancer()
    print()

    print("=" * 60)
    print("✓ All datasets prepared! You can now use them in main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
