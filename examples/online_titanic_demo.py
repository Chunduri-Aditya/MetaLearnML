"""
Online learning demo for Titanic dataset.
Simulates a data stream and shows adaptive model selection.
"""

import torch
from engine.online_controller import OnlineController
from torch.utils.data import DataLoader, random_split

from engine.evaluator import accuracy_metric
from utils.datasets import load_csv_classification


def main():
    print("=" * 60)
    print("Titanic Online Learning Demo")
    print("=" * 60)
    print()

    # Load Titanic dataset
    dataset, in_dim, out_dim = load_csv_classification(
        path="data/titanic_clean.csv",
        feature_cols=[
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
        label_col="Survived",
    )

    print(f"[INFO] Loaded Titanic dataset: {len(dataset)} samples")
    print()

    # Split into base vs stream
    # Use 70% for base (initial search), 30% for streaming
    n_base = int(0.7 * len(dataset))
    n_stream = len(dataset) - n_base
    base_ds, stream_ds = random_split(dataset, [n_base, n_stream])

    print("[INFO] Split dataset:")
    print(f"  Base (for initial search): {len(base_ds)} samples")
    print(f"  Stream (for online learning): {len(stream_ds)} samples")
    print()
    print("[NOTE] Make sure you've run main.py first to get initial best model!")
    print()

    # Build stream loader
    stream_loader = DataLoader(stream_ds, batch_size=16, shuffle=True)

    controller = OnlineController(
        results_csv="experiments/results.csv",
        in_dim=in_dim,
        out_dim=out_dim,
        buffer_size=128,  # Larger buffer for bigger dataset
    )

    print(f"[INFO] Starting online controller with model: {controller.predictor.model_name}")
    print(f"[INFO] Buffer size: {controller.buffer_size}")
    print()

    base_config = {
        "lr": 1e-3,
        "epochs": 5,
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "metric_fn": accuracy_metric,
        "patience": 3,
    }

    print("[INFO] Processing streaming data...")
    print("[INFO] Fine-tuning current model and occasionally challenging with alternatives")
    print()

    controller.fine_tune_on_stream(
        data_stream=stream_loader,
        base_config=base_config,
        challenge_every=4,  # Challenge every 4 batches (Titanic has more data)
        challenge_epochs=5,  # More epochs for challengers on bigger dataset
        metric_fn=accuracy_metric,
        promote_margin=0.02,  # Require 2% improvement to promote
    )

    print()
    print("=" * 60)
    print("Final State")
    print("=" * 60)
    print(f"Current best model: {controller.predictor.model_name}")
    print(f"Buffer size: {len(controller.buffer_x)} samples")
    print()
    print("✓ Online learning demo complete!")
    print()
    print("The system has:")
    print("  - Fine-tuned the initial best model on streaming data")
    print("  - Tested challenger models periodically")
    print("  - Promoted better models when found")
    print()


if __name__ == "__main__":
    main()
