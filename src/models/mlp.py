import torch
from torch import nn


class MLPClassifier(nn.Module):
    """A student-implemented MLP for 512-dimensional feature classification."""

    def __init__(
        self,
        input_dim=512,
        num_classes=10,
        hidden_dims=(256, 128),
        dropout=0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


def build_mlp(input_dim=512, num_classes=10, hidden_dims=(256, 128), dropout=0.3):
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
