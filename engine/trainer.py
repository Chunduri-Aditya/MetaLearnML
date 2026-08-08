import torch
from torch import nn
from typing import Dict, Tuple, Optional


def train_one_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    config: Dict,
) -> Tuple[float, Optional[Dict]]:
    """
    Train a model and return best metric and state dict.
    
    Returns:
        (best_metric, best_state_dict)
    """
    # Check if this is a sklearn wrapper (needs different training)
    try:
        from models.sk_models import SkLearnWrapper
        if isinstance(model, SkLearnWrapper):
            # Sklearn models don't use GPU, always use CPU
            return train_sklearn_model(model, train_loader, val_loader, "cpu", config)
    except ImportError:
        pass  # Sklearn models not available, use standard training
    
    # Standard PyTorch training
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = config.get("loss_fn")  # May be None for regression
    metric_fn = config["metric_fn"]
    epochs = config["epochs"]
    
    # Auto-detect task type from loss function or model output shape
    task_type = config.get("task_type", "classification")
    if loss_fn is None:
        # Default loss based on task type
        if task_type == "regression":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

    best_metric = float("-inf")
    best_state_dict = None
    patience = config.get("patience", None)  # None = no early stopping
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            
            # Handle regression vs classification
            if task_type == "regression":
                # For regression, yb might need reshaping
                if len(yb.shape) == 1:
                    yb = yb.unsqueeze(1).float()
                if len(output.shape) == 1:
                    output = output.unsqueeze(1)
                loss = loss_fn(output, yb)
            else:
                # Classification: yb should be long integers
                if yb.dtype != torch.long:
                    yb = yb.long()
                loss = loss_fn(output, yb)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # validation
        metric = evaluate(model, val_loader, device, metric_fn, task_type=task_type)

        if metric > best_metric:
            best_metric = metric
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Early stopping
        if patience is not None and epochs_without_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    return best_metric, best_state_dict


def train_sklearn_model(
    model,
    train_loader,
    val_loader,
    device: str,
    config: Dict,
) -> Tuple[float, Dict]:
    """
    Train a sklearn model wrapper.
    Sklearn models don't use epochs/optimizers - they fit once on all data.
    
    Returns:
        (metric, dummy_state_dict)
    """
    """
    Train a sklearn model wrapper.
    Sklearn models don't use epochs/optimizers - they fit once on all data.
    """
    metric_fn = config["metric_fn"]
    task_type = config.get("task_type", "classification")
    
    # Collect all training data
    X_train = []
    y_train = []
    for xb, yb in train_loader:
        X_train.append(xb.cpu())
        y_train.append(yb.cpu())
    
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    
    # Fit the sklearn model
    print("[INFO] Fitting sklearn model on training data...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    metric = evaluate(model, val_loader, device, metric_fn, task_type=task_type)
    
    # For sklearn models, we return a dummy state dict (they don't use PyTorch state)
    # The model itself holds the sklearn model state
    best_state_dict = {"sklearn_fitted": True}
    
    return metric, best_state_dict


def evaluate(model: nn.Module, data_loader, device: str, metric_fn, task_type: str = "classification") -> float:
    """
    Evaluate a model on a data loader.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader
        device: Device
        metric_fn: Metric function
        task_type: "classification" or "regression"
    
    Returns:
        Metric value (float)
    """
    # Check if this is a sklearn wrapper
    try:
        from models.sk_models import SkLearnWrapper
        is_sklearn = isinstance(model, SkLearnWrapper)
    except ImportError:
        is_sklearn = False
    
    if is_sklearn:
        # Sklearn models run on CPU, but we still call .eval() for consistency
        model.eval()
        eval_device = "cpu"  # Force CPU for sklearn
    else:
        model.eval()
        eval_device = device
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in data_loader:
            # For sklearn models, always use CPU
            if is_sklearn:
                xb = xb.cpu()
            else:
                xb = xb.to(device)
            output = model(xb)
            # Detach and move to CPU immediately to save GPU memory
            all_preds.append(output.detach().cpu())
            all_targets.append(yb.cpu() if not is_sklearn else yb)

    # Concatenate efficiently
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if task_type == "regression":
        return metric_fn(targets, preds)
    return metric_fn(preds, targets)
