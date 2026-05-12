import math

import torch
from torch import nn


class ManualReLU(nn.Module):
    """ReLU implemented without using nn.ReLU."""

    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))


class ManualFlatten(nn.Module):
    """Flatten all dimensions except the batch dimension."""

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class ManualLinear(nn.Module):
    """Fully connected layer implemented from basic tensor operations."""

    def __init__(self, in_features, out_features):
        super().__init__()
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-limit, limit))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class ManualConv2d(nn.Module):
    """2D convolution implemented manually with explicit sliding windows."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kernel_h, kernel_w = kernel_size
        fan_in = in_channels * kernel_h * kernel_w
        fan_out = out_channels * kernel_h * kernel_w
        limit = math.sqrt(6.0 / (fan_in + fan_out))

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_h, kernel_w).uniform_(-limit, limit)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _pad_input(self, x):
        pad_h, pad_w = self.padding
        if pad_h == 0 and pad_w == 0:
            return x

        batch_size, channels, height, width = x.shape
        padded = x.new_zeros(batch_size, channels, height + 2 * pad_h, width + 2 * pad_w)
        padded[:, :, pad_h : pad_h + height, pad_w : pad_w + width] = x
        return padded

    def forward(self, x):
        x = self._pad_input(x)

        batch_size, _, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        out_height = (height - kernel_h) // stride_h + 1
        out_width = (width - kernel_w) // stride_w + 1
        output = x.new_zeros(batch_size, self.out_channels, out_height, out_width)

        for out_y in range(out_height):
            start_y = out_y * stride_h
            end_y = start_y + kernel_h
            for out_x in range(out_width):
                start_x = out_x * stride_w
                end_x = start_x + kernel_w
                patch = x[:, :, start_y:end_y, start_x:end_x]
                for out_channel in range(self.out_channels):
                    kernel = self.weight[out_channel].unsqueeze(0)
                    output[:, out_channel, out_y, out_x] = (
                        (patch * kernel).sum(dim=(1, 2, 3)) + self.bias[out_channel]
                    )

        return output


class ManualMaxPool2d(nn.Module):
    """Max pooling implemented with explicit region-wise maxima."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        out_height = (height - kernel_h) // stride_h + 1
        out_width = (width - kernel_w) // stride_w + 1
        output = x.new_zeros(batch_size, channels, out_height, out_width)

        for out_y in range(out_height):
            start_y = out_y * stride_h
            end_y = start_y + kernel_h
            for out_x in range(out_width):
                start_x = out_x * stride_w
                end_x = start_x + kernel_w
                region = x[:, :, start_y:end_y, start_x:end_x]
                output[:, :, out_y, out_x] = region.reshape(batch_size, channels, -1).max(dim=2).values

        return output


class ManualCNNClassifier(nn.Module):
    """CNN built from manually implemented convolution, pooling, and linear layers."""

    def __init__(
        self,
        input_dim=512,
        num_classes=10,
        image_height=32,
        image_width=16,
        conv1_channels=8,
        conv2_channels=16,
        hidden_dim=128,
    ):
        super().__init__()

        if image_height * image_width != input_dim:
            raise ValueError(
                f"image_height * image_width must equal input_dim, got "
                f"{image_height} * {image_width} != {input_dim}"
            )

        self.image_height = image_height
        self.image_width = image_width

        self.conv1 = ManualConv2d(1, conv1_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = ManualReLU()
        self.pool1 = ManualMaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ManualConv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = ManualReLU()
        self.pool2 = ManualMaxPool2d(kernel_size=2, stride=2)

        reduced_height = image_height // 4
        reduced_width = image_width // 4
        flattened_dim = conv2_channels * reduced_height * reduced_width

        self.flatten = ManualFlatten()
        self.fc1 = ManualLinear(flattened_dim, hidden_dim)
        self.relu3 = ManualReLU()
        self.fc2 = ManualLinear(hidden_dim, num_classes)

    def _reshape_input(self, x):
        if x.dim() == 2:
            return x.reshape(x.size(0), 1, self.image_height, self.image_width)
        if x.dim() == 4:
            return x
        raise ValueError("Input to ManualCNNClassifier must have shape [B, D] or [B, C, H, W].")

    def forward(self, x):
        x = x.float()
        x = self._reshape_input(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def build_cnn(
    input_dim=512,
    num_classes=10,
    image_height=32,
    image_width=16,
    conv1_channels=8,
    conv2_channels=16,
    hidden_dim=128,
):
    return ManualCNNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        image_height=image_height,
        image_width=image_width,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        hidden_dim=hidden_dim,
    )


def build_cnn_32x16(
    input_dim=512,
    num_classes=10,
    conv1_channels=8,
    conv2_channels=16,
    hidden_dim=128,
):
    """Build a CNN that reshapes 512-dim inputs to 1 x 32 x 16."""

    return ManualCNNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        image_height=32,
        image_width=16,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        hidden_dim=hidden_dim,
    )


def build_cnn_8x64(
    input_dim=512,
    num_classes=10,
    conv1_channels=8,
    conv2_channels=16,
    hidden_dim=128,
):
    """Build a CNN that reshapes 512-dim inputs to 1 x 8 x 64."""

    return ManualCNNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        image_height=8,
        image_width=64,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        hidden_dim=hidden_dim,
    )
