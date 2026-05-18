import math

import torch
from torch import nn


class LogisticRegressionClassifier(nn.Module):
    """Student-built multinomial logistic regression baseline.

    This is a single linear decision layer for 512-dimensional features.
    It intentionally does not use sklearn or nn.Linear; the trainable
    weight matrix and bias vector are defined directly as Parameters.
    """

    def __init__(self, input_dim=512, num_classes=10):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.float() @ self.weight + self.bias


def build_logistic_regression(input_dim=512, num_classes=10):
    return LogisticRegressionClassifier(input_dim=input_dim, num_classes=num_classes)
