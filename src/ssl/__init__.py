"""Semi-supervised learning methods."""

from src.ssl.clustering import ClusteringSSL
from src.ssl.label_propagation import LabelPropagationSSL
from src.ssl.self_training import SelfTrainingSSL

__all__ = [
    "ClusteringSSL",
    "LabelPropagationSSL",
    "SelfTrainingSSL",
]
