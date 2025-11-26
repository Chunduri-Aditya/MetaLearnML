"""
Scikit-learn model wrappers for classical supervised learning.
These provide an alternative to neural networks in the model zoo.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler


class SkLearnWrapper(nn.Module):
    """
    Base wrapper for sklearn models to work with PyTorch training pipeline.
    """
    def __init__(self, sklearn_model, in_dim: int, out_dim: int):
        super().__init__()
        self.clf = sklearn_model  # Expose as 'clf' for consistency with predictor
        self.sklearn_model = sklearn_model  # Keep for backward compatibility
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def to(self, device):
        """
        Override .to() to be a no-op for sklearn models.
        Sklearn models don't use GPU, but we need this for compatibility.
        """
        return self  # Return self to allow chaining
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert to numpy, predict, return logits/predictions.
        Note: This is called during evaluation. Training happens in fit().
        
        Args:
            x: Input tensor [N, in_dim]
        
        Returns:
            Logits tensor [N, out_dim] for classification
            Predictions tensor [N, 1] for regression
        """
        if not self.is_fitted:
            # Return random output if not fitted yet
            return torch.randn(x.shape[0], self.out_dim, device=x.device, dtype=torch.float32)
        
        # Convert to numpy (detach and move to CPU)
        x_np = x.detach().cpu().numpy()
        
        # Scale
        x_scaled = self.scaler.transform(x_np)
        
        # Check if this is a regressor (has predict method but no predict_proba)
        if hasattr(self.sklearn_model, 'predict_proba'):
            # Classification: predict probabilities
            probs = self.sklearn_model.predict_proba(x_scaled)
            # Convert to logits (inverse softmax approximation)
            epsilon = 1e-8
            logits = np.log(probs + epsilon).astype(np.float32)
            output = logits
        else:
            # Regression: predict directly
            preds = self.sklearn_model.predict(x_scaled)
            # Reshape to [N, 1] if needed
            if len(preds.shape) == 1:
                preds = preds.reshape(-1, 1)
            output = preds.astype(np.float32)
        
        # Return tensor on same device as input
        return torch.from_numpy(output).to(x.device)
    
    def fit(self, X, y):
        """Fit the sklearn model on training data."""
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_np)
        
        # Fit sklearn model
        self.sklearn_model.fit(X_scaled, y_np)
        self.is_fitted = True


def build_logreg(in_dim: int, out_dim: int):
    """Build Logistic Regression wrapper."""
    # Use default multi_class behavior (deprecated parameter removed)
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    return SkLearnWrapper(model, in_dim, out_dim)


def build_rf(in_dim: int, out_dim: int):
    """Build Random Forest wrapper."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    return SkLearnWrapper(model, in_dim, out_dim)


def build_gb(in_dim: int, out_dim: int):
    """Build Gradient Boosting wrapper."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    return SkLearnWrapper(model, in_dim, out_dim)


# Regression models
def build_linreg(in_dim: int, out_dim: int):
    """Build Linear Regression wrapper."""
    model = LinearRegression()
    return SkLearnWrapper(model, in_dim, out_dim)


def build_ridge(in_dim: int, out_dim: int):
    """Build Ridge Regression wrapper."""
    model = Ridge(alpha=1.0, random_state=42)
    return SkLearnWrapper(model, in_dim, out_dim)


def build_lasso(in_dim: int, out_dim: int):
    """Build Lasso Regression wrapper."""
    model = Lasso(alpha=1.0, random_state=42)
    return SkLearnWrapper(model, in_dim, out_dim)


def build_rf_reg(in_dim: int, out_dim: int):
    """Build Random Forest Regressor wrapper."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    return SkLearnWrapper(model, in_dim, out_dim)


def build_gb_reg(in_dim: int, out_dim: int):
    """Build Gradient Boosting Regressor wrapper."""
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    return SkLearnWrapper(model, in_dim, out_dim)


# SVM models
def build_svm_classifier(in_dim: int, out_dim: int):
    """Build SVM Classifier wrapper."""
    model = SVC(
        kernel="rbf",
        probability=True,
        C=1.0,
        gamma="scale",
        random_state=42,
    )
    return SkLearnWrapper(model, in_dim, out_dim)


def build_svm_regressor(in_dim: int, out_dim: int):
    """Build SVM Regressor wrapper."""
    model = SVR(
        kernel="rbf",
        C=1.0,
        gamma="scale",
    )
    return SkLearnWrapper(model, in_dim, out_dim)
