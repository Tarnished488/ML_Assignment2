import torch
from torch import nn


class ResidualBlock(nn.Module):
    """MLP block with residual connection and projection shortcut."""

    def __init__(self, in_dim, out_dim, norm_cls, act_cls, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            norm_cls(out_dim),
            act_cls(),
            nn.Dropout(dropout),
        )
        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class MLPClassifier(nn.Module):
    """An optimized MLP for 512-dimensional feature classification.

    Supports residual connections, multiple normalization types,
    and various activation functions.
    """

    _ACTIVATIONS = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
    }
    _NORMS = {
        "batch": nn.BatchNorm1d,
        "layer": nn.LayerNorm,
    }

    def __init__(
        self,
        input_dim=512,
        num_classes=10,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        activation="gelu",
        use_residual=True,
        norm="batch",
    ):
        super().__init__()

        act_cls = self._ACTIVATIONS.get(activation.lower(), nn.GELU)
        norm_cls = self._NORMS.get(norm.lower(), nn.BatchNorm1d)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(
                ResidualBlock(prev_dim, hidden_dim, norm_cls, act_cls, dropout)
            )
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(prev_dim, num_classes)
        self.use_residual = use_residual
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.float()
        for layer in self.layers:
            identity = layer.shortcut(x)
            x = layer.block(x)
            if self.use_residual:
                x = x + identity
        return self.head(x)


def build_mlp(
    input_dim=512,
    num_classes=10,
    hidden_dims=(256, 128, 64),
    dropout=0.3,
    activation="gelu",
    use_residual=True,
    norm="batch",
):
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        use_residual=use_residual,
        norm=norm,
    )


def build_mlp_deep(
    input_dim=512,
    num_classes=10,
    dropout=0.35,
    activation="gelu",
    use_residual=True,
    norm="batch",
):
    """Deep 6-layer MLP — more capacity for complex patterns."""
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=(512, 256, 128, 64, 32),
        dropout=dropout,
        activation=activation,
        use_residual=use_residual,
        norm=norm,
    )


def build_mlp_wide(
    input_dim=512,
    num_classes=10,
    dropout=0.3,
    activation="gelu",
    use_residual=True,
    norm="batch",
):
    """Wide 3-layer MLP — larger hidden representations."""
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=(512, 256, 128),
        dropout=dropout,
        activation=activation,
        use_residual=use_residual,
        norm=norm,
    )
