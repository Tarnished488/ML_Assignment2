"""Model implementations for Assignment 2."""

from src.models.cnn import ManualCNNClassifier, build_cnn, build_cnn_8x64, build_cnn_32x16
from src.models.mlp import MLPClassifier, build_mlp, build_mlp_deep, build_mlp_wide

__all__ = [
    "MLPClassifier",
    "ManualCNNClassifier",
    "build_mlp",
    "build_mlp_deep",
    "build_mlp_wide",
    "build_cnn",
    "build_cnn_32x16",
    "build_cnn_8x64",
]
