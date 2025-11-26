# online_demo.py

import torch
from torch.utils.data import DataLoader, random_split

from utils.datasets import load_csv_classification
from engine.online_controller import OnlineController
from engine.evaluator import accuracy_metric


def main():
    # 1. Load Iris dataset
    dataset, in_dim, out_dim = load_csv_classification(
        path="data/iris.csv",
        feature_cols=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
        label_col="label",
    )

    # 2. Split into base vs stream
    n_base = int(0.6 * len(dataset))
    n_stream = len(dataset) - n_base
    base_ds, stream_ds = random_split(dataset, [n_base, n_stream])

    # NOTE: base_ds is what you *would* use for initial search.
    # For this demo we just assume main.py already trained & wrote results.csv.

    stream_loader = DataLoader(stream_ds, batch_size=8, shuffle=True)

    controller = OnlineController(
        results_csv="experiments/results.csv",
        in_dim=in_dim,
        out_dim=out_dim,
    )

    base_config = {
        "lr": 1e-3,
        "epochs": 5,
        "loss_fn": torch.nn.CrossEntropyLoss(),
        "metric_fn": accuracy_metric,
        "patience": 3,
    }

    controller.fine_tune_on_stream(
        data_stream=stream_loader,
        base_config=base_config,
        challenge_every=3,     # every 3 batches, try a challenger
        challenge_epochs=3,    # challengers get a few short epochs
        metric_fn=accuracy_metric,
        promote_margin=0.02,   # require challenger to be clearly better
    )


if __name__ == "__main__":
    main()
