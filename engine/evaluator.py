from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def accuracy_metric(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Standard accuracy metric.
    Assumes logits shape [N, num_classes] and targets shape [N].

    Args:
        logits: Model output logits [N, num_classes]
        targets: True labels [N]

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(targets) == 0:
        return 0.0
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float().sum().item()
    return correct / len(targets)


def f1_macro_metric(logits, targets):
    """
    F1 score with macro averaging (good for imbalanced classes).
    """
    preds = torch.argmax(logits, dim=1).numpy()
    y_true = targets.numpy()
    return f1_score(y_true, preds, average="macro")


def f1_weighted_metric(logits, targets):
    """
    F1 score with weighted averaging (accounts for class imbalance).
    """
    preds = torch.argmax(logits, dim=1).numpy()
    y_true = targets.numpy()
    return f1_score(y_true, preds, average="weighted")


def roc_auc_metric(logits, targets):
    """
    ROC-AUC score (for binary classification only).
    For multiclass, use one-vs-rest strategy.
    """
    preds = torch.softmax(logits, dim=1).numpy()
    y_true = targets.numpy()

    # Handle binary vs multiclass
    num_classes = preds.shape[1]
    if num_classes == 2:
        return roc_auc_score(y_true, preds[:, 1])
    else:
        # Multiclass: use one-vs-rest
        return roc_auc_score(y_true, preds, multi_class="ovr", average="macro")


def confusion_matrix_for_model(
    model: torch.nn.Module, data_loader, device: str
) -> tuple[torch.Tensor, dict[str, float] | None]:
    """
    Compute confusion matrix for a model on a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        confusion_matrix: numpy array of shape (n_classes, n_classes)
        metrics: dict with recall/precision for binary classification, None otherwise
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    cm = confusion_matrix(y_true, y_pred)

    # Compute metrics for binary classification
    metrics = None
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics = {}
        if tp + fn > 0:
            metrics["Recall"] = tp / (tp + fn)
        if tp + fp > 0:
            metrics["Precision"] = tp / (tp + fp)

    return cm, metrics


# Regression metrics (higher is better convention)
def neg_rmse_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Negative RMSE metric (higher is better for search).

    Args:
        y_true: True values
        y_pred: Predicted values (logits for regression models)

    Returns:
        Negative RMSE
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # For regression models, predictions are direct (no argmax needed)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    elif len(y_pred.shape) > 1:
        # If shape is [N, 1], flatten
        y_pred = y_pred.flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return -rmse  # Negative so higher is better


def neg_mae_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Negative MAE metric (higher is better for search).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Negative MAE
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    elif len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    return -mae  # Negative so higher is better


def r2_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    R² metric (already higher is better).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    elif len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()

    return r2_score(y_true, y_pred)


# Metric registry for easy lookup
METRIC_REGISTRY: dict[str, Callable] = {
    "accuracy": accuracy_metric,
    "f1_macro": f1_macro_metric,
    "f1_weighted": f1_weighted_metric,
    "roc_auc": roc_auc_metric,
    "neg_rmse": neg_rmse_metric,
    "neg_mae": neg_mae_metric,
    "rmse": neg_rmse_metric,
    "mae": neg_mae_metric,
    "r2": r2_metric,
}


@dataclass(frozen=True)
class MetricSpec:
    """Metric behavior with separate optimization and human display semantics."""

    name: str
    callable: Callable
    greater_is_better: bool
    task_kind: str
    display_name: str
    objective_is_negated_display: bool = False

    def score_tensors(self, targets: torch.Tensor, outputs: torch.Tensor) -> tuple[float, float]:
        if self.task_kind == "regression":
            objective = float(self.callable(targets, outputs))
        else:
            objective = float(self.callable(outputs, targets))
        display = -objective if self.objective_is_negated_display else objective
        return objective, display

    def objective_from_display(self, value: float) -> float:
        return -value if self.objective_is_negated_display else value

    def display_from_objective(self, value: float) -> float:
        return -value if self.objective_is_negated_display else value

    def target_reached(self, display_value: float, target: float) -> bool:
        return display_value >= target if self.greater_is_better else display_value <= target

    def generalization_gap(self, cv_display: float, test_display: float) -> float:
        """Positive means test performance degraded relative to CV."""
        direction = 1.0 if self.greater_is_better else -1.0
        return direction * (cv_display - test_display)


METRIC_SPECS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", accuracy_metric, True, "classification", "Accuracy"),
    "f1_macro": MetricSpec("f1_macro", f1_macro_metric, True, "classification", "Macro F1"),
    "f1_weighted": MetricSpec(
        "f1_weighted", f1_weighted_metric, True, "classification", "Weighted F1"
    ),
    "roc_auc": MetricSpec("roc_auc", roc_auc_metric, True, "classification", "ROC-AUC"),
    "r2": MetricSpec("r2", r2_metric, True, "regression", "R²"),
    "rmse": MetricSpec(
        "rmse", neg_rmse_metric, False, "regression", "RMSE", objective_is_negated_display=True
    ),
    "mae": MetricSpec(
        "mae", neg_mae_metric, False, "regression", "MAE", objective_is_negated_display=True
    ),
}
METRIC_SPECS["neg_rmse"] = METRIC_SPECS["rmse"]
METRIC_SPECS["neg_mae"] = METRIC_SPECS["mae"]


def get_metric_spec(metric_name: str) -> MetricSpec:
    try:
        return METRIC_SPECS[metric_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown metric {metric_name!r}; expected one of {sorted(METRIC_SPECS)}"
        ) from exc
