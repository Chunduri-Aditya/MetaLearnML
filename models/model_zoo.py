from .mlp_variants import (
    MLP_Deep,
    MLP_Deep_Reg,
    MLP_Medium,
    MLP_Medium_Reg,
    MLP_Small,
    MLP_Small_Reg,
)
from .sk_models import (
    build_gb,
    build_gb_reg,
    build_lasso,
    build_linreg,
    build_logreg,
    build_rf,
    build_rf_reg,
    build_ridge,
    build_svm_classifier,
    build_svm_regressor,
)


def build_mlp_small(in_dim: int, out_dim: int):
    return MLP_Small(in_dim, out_dim)


def build_mlp_medium(in_dim: int, out_dim: int):
    return MLP_Medium(in_dim, out_dim)


def build_mlp_deep(in_dim: int, out_dim: int):
    return MLP_Deep(in_dim, out_dim)


# Regression model builders
def build_mlp_small_reg(in_dim: int, out_dim: int = 1):
    return MLP_Small_Reg(in_dim, out_dim)


def build_mlp_medium_reg(in_dim: int, out_dim: int = 1):
    return MLP_Medium_Reg(in_dim, out_dim)


def build_mlp_deep_reg(in_dim: int, out_dim: int = 1):
    return MLP_Deep_Reg(in_dim, out_dim)


MODEL_REGISTRY = {
    # Neural supervised models
    "mlp_small": {
        "identity": "mlp_small",
        "builder": build_mlp_small,
        "family": "neural_supervised",
        "backend": "torch",
    },
    "mlp_medium": {
        "identity": "mlp_medium",
        "builder": build_mlp_medium,
        "family": "neural_supervised",
        "backend": "torch",
    },
    "mlp_deep": {
        "identity": "mlp_deep",
        "builder": build_mlp_deep,
        "family": "neural_supervised",
        "backend": "torch",
    },
    # Classical supervised models
    "logreg": {
        "identity": "logreg",
        "builder": build_logreg,
        "family": "classical_supervised",
        "backend": "sklearn",
    },
    "rf": {
        "identity": "rf",
        "builder": build_rf,
        "family": "classical_supervised",
        "backend": "sklearn",
    },
    "gb": {
        "identity": "gb",
        "builder": build_gb,
        "family": "classical_supervised",
        "backend": "sklearn",
    },
    # Regression models
    "linreg": {
        "identity": "linreg",
        "builder": build_linreg,
        "family": "linear_regression",
        "backend": "sklearn",
    },
    "ridge": {
        "identity": "ridge",
        "builder": build_ridge,
        "family": "linear_regression",
        "backend": "sklearn",
    },
    "lasso": {
        "identity": "lasso",
        "builder": build_lasso,
        "family": "linear_regression",
        "backend": "sklearn",
    },
    "rf_reg": {
        "identity": "rf_reg",
        "builder": build_rf_reg,
        "family": "tree_regression",
        "backend": "sklearn",
    },
    "gb_reg": {
        "identity": "gb_reg",
        "builder": build_gb_reg,
        "family": "tree_regression",
        "backend": "sklearn",
    },
    "mlp_small_reg": {
        "identity": "mlp_small_reg",
        "builder": build_mlp_small_reg,
        "family": "neural_regression",
        "backend": "torch",
    },
    "mlp_medium_reg": {
        "identity": "mlp_medium_reg",
        "builder": build_mlp_medium_reg,
        "family": "neural_regression",
        "backend": "torch",
    },
    "mlp_deep_reg": {
        "identity": "mlp_deep_reg",
        "builder": build_mlp_deep_reg,
        "family": "neural_regression",
        "backend": "torch",
    },
    # SVM models
    "svm_clf": {
        "identity": "svm_clf",
        "builder": build_svm_classifier,
        "family": "svm_classification",
        "backend": "sklearn",
    },
    "svm_reg": {
        "identity": "svm_reg",
        "builder": build_svm_regressor,
        "family": "svm_regression",
        "backend": "sklearn",
    },
    # Examples for later expansion:
    # "kmeans": {
    #     "builder": build_kmeans_wrapper,
    #     "family": "unsupervised",
    # },
    # "autoencoder": {
    #     "builder": build_autoencoder_wrapper,
    #     "family": "unsupervised_neural",
    # },
}


def list_models():
    """Return list of all model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_entry(name: str):
    """Get model registry entry by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {name}")
    return MODEL_REGISTRY[name]


def validate_model_registry():
    """Validate stable registry fields and return a deterministic snapshot."""
    snapshot = []
    for key, entry in MODEL_REGISTRY.items():
        missing = {
            field for field in ("identity", "family", "backend", "builder") if not entry.get(field)
        }
        if missing:
            raise ValueError(f"Model {key!r} is missing registry fields: {sorted(missing)}")
        if entry["identity"] != key:
            raise ValueError(
                f"Model registry key {key!r} must equal stable identity {entry['identity']!r}"
            )
        snapshot.append((key, entry["family"], entry["backend"]))
    return tuple(sorted(snapshot))


def build_model(model_name: str, in_dim: int, out_dim: int):
    """Build a model by name (backward compatibility)."""
    entry = get_model_entry(model_name)
    return entry["builder"](in_dim, out_dim)


# Backward compatibility: keep MODEL_ZOO as list
MODEL_ZOO = list_models()
