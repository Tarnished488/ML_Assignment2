"""Visualization utilities for semi-supervised learning experiments.

Provides PCA/t-SNE projections, decision boundaries, loss landscapes,
confusion matrices, and training curves.
"""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn

matplotlib.use("Agg")  # non-interactive backend

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

FONTSIZE_TITLE = 13
FONTSIZE_LABEL = 11
FONTSIZE_TICK = 9
FONTSIZE_LEGEND = 9
DPI = 200
FIG_SIZE = (8, 6)


def _ensure_path(save_path: Optional[str]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def _style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.tick_params(labelsize=FONTSIZE_TICK)


# ===================================================================
# PCA / t-SNE
# ===================================================================


def _reduce_2d(X: np.ndarray, method: str = "pca", random_state: int = 42):
    """Reduce X to 2D via PCA or t-SNE."""
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X) - 1))
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    return reducer.fit_transform(X)


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "PCA Projection",
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
) -> plt.Figure:
    """2D PCA projection with class coloring.

    Parameters
    ----------
    X : shape (N, D)
        Feature matrix.
    y : shape (N,)
        Integer class labels. Use -1 to denote unlabeled points.
    title : str
        Plot title.
    save_path : str or None
        If set, save figure to this path.
    class_names : list of str or None
        Class display names.
    """
    X_2d = _reduce_2d(X, method="pca")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    unique_labels = np.unique(y)

    for label in unique_labels:
        mask = y == label
        if label == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=6, alpha=0.25,
                       c="gray", marker=".", label="Unlabeled")
        else:
            name = class_names[label] if class_names else str(label)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=12, alpha=0.8, label=name)

    _style_ax(ax, "PC 1", "PC 2", title)
    if len(unique_labels) <= 12:
        ax.legend(fontsize=FONTSIZE_LEGEND, markerscale=1.2)
    fig.tight_layout()
    _ensure_path(save_path)
    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


def plot_tsne(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "t-SNE Projection",
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
) -> plt.Figure:
    """2D t-SNE projection with class coloring."""
    X_2d = _reduce_2d(X, method="tsne")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    unique_labels = np.unique(y)

    for label in unique_labels:
        mask = y == label
        if label == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=6, alpha=0.25,
                       c="gray", marker=".", label="Unlabeled")
        else:
            name = class_names[label] if class_names else str(label)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=12, alpha=0.8, label=name)

    _style_ax(ax, "t-SNE 1", "t-SNE 2", title)
    if len(unique_labels) <= 12:
        ax.legend(fontsize=FONTSIZE_LEGEND, markerscale=1.2)
    fig.tight_layout()
    _ensure_path(save_path)
    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ===================================================================
# Decision Boundary  (via PCA projection of the input space)
# ===================================================================


def plot_decision_boundary(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    title: str = "Decision Boundary (PCA view)",
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
    grid_resolution: int = 200,
) -> plt.Figure:
    """Plot decision boundary by projecting input space to 2D via PCA.

    The model is evaluated on a dense 2D grid in PCA space and the predicted
    class is coloured.  Training points are overlaid.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    X : shape (N, D)
        Feature matrix used to fit PCA and as training-point overlay.
    y : shape (N,)
        Integer labels for the points in X.
    device : torch.device
    title : str
    save_path : str or None
    class_names : list of str or None
    grid_resolution : int
        Grid density per axis.
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Build a mesh in PCA space
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]          # (G, 2)
    grid_original = pca.inverse_transform(grid_2d)   # (G, D)

    # Predict on grid
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_original, dtype=torch.float32).to(device)
        logits = model(grid_tensor)
        grid_pred = logits.argmax(dim=1).cpu().numpy()
    grid_pred = grid_pred.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.contourf(xx, yy, grid_pred, alpha=0.3, cmap="tab10", levels=20)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=12, alpha=0.85,
                         cmap="tab10", edgecolors="k", linewidth=0.3)

    if class_names:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, class_names, title="Classes",
                  fontsize=FONTSIZE_LEGEND, title_fontsize=FONTSIZE_LEGEND)

    _style_ax(ax, "PC 1", "PC 2", title)
    fig.tight_layout()
    _ensure_path(save_path)
    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ===================================================================
# Loss Landscape
# ===================================================================


@torch.no_grad()
def plot_loss_landscape(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    title: str = "Loss Landscape",
    save_path: Optional[str] = None,
    grid_resolution: int = 51,
    alpha_range: tuple = (-1.0, 1.0),
    beta_range: tuple = (-1.0, 1.0),
) -> plt.Figure:
    """Visualise loss landscape along two random directions in parameter space.

    Implements the approach from Li et al. "Visualizing the Loss Landscape
    of Neural Nets" (NeurIPS 2018) — two random Gaussian direction vectors
    are sampled, filter-normalised, and the loss is evaluated on a 2D grid.

    Parameters
    ----------
    model : nn.Module
        Trained model (used as the centre point).
    X, y : ndarray
        Evaluation data (e.g. validation set).
    criterion : nn.Module
        Loss function.
    device : torch.device
    title : str
    save_path : str or None
    grid_resolution : int
        Grid density per axis.
    alpha_range, beta_range : tuple
        Multiplier range along each direction.
    """
    # Snapshot centre parameters
    centre = {name: p.clone() for name, p in model.named_parameters()}

    # Generate two random direction vectors with filter-wise normalisation
    direction_a = {}
    direction_b = {}
    for name, p in model.named_parameters():
        rand_a = torch.randn_like(p)
        rand_b = torch.randn_like(p)
        # Filter-normalise: scale each filter to match the centre param norm
        if p.dim() >= 2:  # weight matrix / conv filter
            # Flatten each output filter/row and compute a vector norm so the
            # logic works for both linear weights [out, in] and conv weights
            # [out, in, kH, kW] across PyTorch versions.
            flat_shape = (p.shape[0], -1)
            norm_centre = torch.linalg.vector_norm(
                p.reshape(flat_shape), dim=1, keepdim=True
            ).reshape((p.shape[0],) + (1,) * (p.dim() - 1))
            norm_a = torch.linalg.vector_norm(
                rand_a.reshape(flat_shape), dim=1, keepdim=True
            ).reshape((p.shape[0],) + (1,) * (p.dim() - 1))
            norm_b = torch.linalg.vector_norm(
                rand_b.reshape(flat_shape), dim=1, keepdim=True
            ).reshape((p.shape[0],) + (1,) * (p.dim() - 1))
            rand_a = rand_a / (norm_a + 1e-10) * (norm_centre + 1e-10)
            rand_b = rand_b / (norm_b + 1e-10) * (norm_centre + 1e-10)
        direction_a[name] = rand_a
        direction_b[name] = rand_b

    # Evaluate loss on grid
    alphas = np.linspace(alpha_range[0], alpha_range[1], grid_resolution)
    betas = np.linspace(beta_range[0], beta_range[1], grid_resolution)
    loss_grid = np.zeros((grid_resolution, grid_resolution))

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)

    model.eval()
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for name, p in model.named_parameters():
                p.copy_(centre[name] + alpha * direction_a[name] + beta * direction_b[name])
            logits = model(X_t)
            loss_grid[j, i] = criterion(logits, y_t).item()  # j=row, i=col

    # Restore centre
    for name, p in model.named_parameters():
        p.copy_(centre[name])

    # Plot
    AA, BB = np.meshgrid(alphas, betas)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    contour = ax.contourf(AA, BB, loss_grid, levels=50, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Loss", fontsize=FONTSIZE_LABEL)
    ax.scatter([0], [0], marker="*", c="red", s=120, zorder=5, label="Trained model")
    _style_ax(ax, "Direction α", "Direction β", title)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    fig.tight_layout()
    _ensure_path(save_path)
    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ===================================================================
# Confusion Matrix
# ===================================================================


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64)
        cm /= cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm, nan=0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True,
              values_format=".2f" if normalize else "d",
              xticks_rotation=45)
    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    fig.tight_layout()
    _ensure_path(save_path)
    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ===================================================================
# Training Curves
# ===================================================================


def plot_training_curves(
    history: dict,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot loss, accuracy and macro-F1 from a training history dictionary.

    Parameters
    ----------
    history : dict
        Expected keys:
        - ``train_loss`` (list)
        - ``val_acc`` (list)
        - ``val_macro_f1`` (list)
        Optional:
        - ``val_loss`` (list)
        - ``train_acc`` (list)
        - ``lr`` (list)
    title : str
    save_path : str or None
    """
    n_panels = 2 + int("lr" in history)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Loss
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.2)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.2)
    _style_ax(ax, "Epoch", "Loss", "Loss")
    ax.legend(fontsize=FONTSIZE_LEGEND)

    # Panel 2: Accuracy & Macro-F1
    ax = axes[1]
    ax.plot(epochs, history["val_acc"], label="Val Accuracy", linewidth=1.2)
    ax.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", linewidth=1.2)
    if "train_acc" in history:
        ax.plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=1.2)
    _style_ax(ax, "Epoch", "Score", "Accuracy & Macro-F1")
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.set_ylim(0, 1.05)

    # Panel 3: Learning rate (optional)
    if "lr" in history and len(axes) >= 3:
        ax = axes[2]
        ax.plot(epochs, history["lr"], linewidth=1.2, color="green")
        _style_ax(ax, "Epoch", "LR", "Learning Rate")

    fig.suptitle(title, fontsize=FONTSIZE_TITLE + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _ensure_path(save_path)
    if save_path:
        fig.savefig(save_path, dpi=DPI)
    return fig


# ===================================================================
# Comparison Bar Chart
# ===================================================================


