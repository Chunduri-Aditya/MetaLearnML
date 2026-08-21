import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """
    Standard MLP block with BatchNorm and Dropout.
    This makes MLPs much more competitive on real datasets.
    """

    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        # Handle 1D case (batch_size, features) - BatchNorm1d expects 2D
        if x.dim() == 2:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MLPRegressor(nn.Module):
    """MLP for regression tasks."""

    def __init__(self, in_dim, hidden_dims=None, dropout=0.2):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(MLPBlock(last, h, dropout=dropout))
            last = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(last, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.out(x)


class MLPClassifier(nn.Module):
    """MLP for classification tasks."""

    def __init__(self, in_dim, n_classes, hidden_dims=None, dropout=0.3):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(MLPBlock(last, h, dropout=dropout))
            last = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(last, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.out(x)


# Legacy class names for backward compatibility
class MLP_Small(MLPClassifier):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, hidden_dims=[64, 32], dropout=0.2)


class MLP_Medium(MLPClassifier):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, hidden_dims=[128, 64, 32], dropout=0.3)


class MLP_Deep(MLPClassifier):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, hidden_dims=[256, 128, 64, 32], dropout=0.4)


# Regression variants
class MLP_Small_Reg(MLPRegressor):
    def __init__(self, in_dim, out_dim=1):
        super().__init__(in_dim, hidden_dims=[64, 32], dropout=0.1)


class MLP_Medium_Reg(MLPRegressor):
    def __init__(self, in_dim, out_dim=1):
        super().__init__(in_dim, hidden_dims=[128, 64, 32], dropout=0.2)


class MLP_Deep_Reg(MLPRegressor):
    def __init__(self, in_dim, out_dim=1):
        super().__init__(in_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3)
