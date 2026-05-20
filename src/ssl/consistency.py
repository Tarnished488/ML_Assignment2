"""Consistency regularisation losses for semi-supervised training.

VAT (Virtual Adversarial Training): Miyato et al., 2018.
Pi-Model: Laine & Aila, 2017.
FixMatch: Sohn et al., NeurIPS 2020 — weak/strong augmentation + confidence
  masking adapted for embedding-space data via Gaussian noise perturbations.
"""

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# KL divergence helper
# ---------------------------------------------------------------------------

def _kl_div_symmetric(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Numerically stable symmetric KL divergence."""
    log_p = F.log_softmax(logits_p, dim=1)
    log_q = F.log_softmax(logits_q, dim=1)
    kl_pq = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
    kl_qp = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
    return (kl_pq + kl_qp) / 2.0


# ---------------------------------------------------------------------------
# Embedding-space augmentations  (weak / strong)
# ---------------------------------------------------------------------------

def weak_augment(x: torch.Tensor, std: float = 0.10) -> torch.Tensor:
    """Weak augmentation: small isotropic Gaussian noise.

    Analogue to the "weak augmentation" in FixMatch (random flip + crop
    for images).  For 512-d embeddings, a small perturbation simulates
    minor measurement noise in the feature extractor.
    """
    noise = torch.randn_like(x) * std
    return x + noise


def strong_augment(
    x: torch.Tensor,
    std: float = 0.30,
    dropout_rate: float = 0.20,
) -> torch.Tensor:
    """Strong augmentation: larger Gaussian noise + random feature dropout.

    Analogue to RandAugment in FixMatch.  Feature dropout zeros out a
    random subset of dimensions, forcing the model to rely on distributed
    representations rather than memorising a few features.
    """
    noise = torch.randn_like(x) * std
    x_noised = x + noise
    if dropout_rate > 0:
        mask = torch.bernoulli(
            torch.full_like(x_noised, 1.0 - dropout_rate)
        )
        x_noised = x_noised * mask / max(1.0 - dropout_rate, 1e-6)
    return x_noised


# ---------------------------------------------------------------------------
# FixMatch-style masked consistency loss
# ---------------------------------------------------------------------------

def fixmatch_loss(
    model: nn.Module,
    x: torch.Tensor,
    threshold: float = 0.95,
    weak_std: float = 0.10,
    strong_std: float = 0.30,
    dropout_rate: float = 0.20,
    hard_label: bool = True,
) -> tuple[torch.Tensor, int]:
    """FixMatch-style consistency loss adapted for embedding-space data.

    Steps
    -----
    1. Apply **weak** augmentation (small noise) → get pseudo-labels.
    2. Keep only samples whose max predicted probability exceeds
       ``threshold`` (confidence-based masking).
    3. Apply **strong** augmentation (larger noise + dropout) to the same
       batch → enforce prediction consistency with the weak pseudo-labels.

    Parameters
    ----------
    model : nn.Module
    x : Tensor (B, D)
        Mini-batch of unlabeled embeddings.
    threshold : float
        Confidence floor for pseudo-label quality.
    weak_std, strong_std, dropout_rate : float
        Augmentation intensities.
    hard_label : bool
        True → CE with hard pseudo-labels.  False → KL-div with soft labels.

    Returns
    -------
    loss : Tensor (scalar)
        FixMatch consistency loss (0.0 if no sample passes the confidence
        threshold).
    num_used : int
        Number of unlabeled samples that passed the confidence threshold.
    """
    if x.size(0) == 0:
        return torch.tensor(0.0, device=x.device), 0

    # Weak augmentation → pseudo-labels (no gradient)
    x_weak = weak_augment(x, std=weak_std)
    with torch.no_grad():
        logits_weak = model(x_weak)
        probs_weak = torch.softmax(logits_weak, dim=1)
        max_probs, pseudo_labels = probs_weak.max(dim=1)
        mask = max_probs >= threshold

    num_used = int(mask.sum().item())
    if num_used == 0:
        return torch.tensor(0.0, device=x.device), 0

    # Strong augmentation → enforce consistency
    x_strong = strong_augment(x[mask], std=strong_std, dropout_rate=dropout_rate)
    logits_strong = model(x_strong)

    if hard_label:
        loss = F.cross_entropy(logits_strong, pseudo_labels[mask])
    else:
        # Soft consistency: KL(weak_probs || strong_probs)
        loss = _kl_div_symmetric(logits_weak[mask], logits_strong)

    return loss, num_used


# ---------------------------------------------------------------------------
# VAT
# ---------------------------------------------------------------------------

def vat_loss(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float = 2.0,
    xi: float = 1e-6,
    num_power_iters: int = 1,
) -> torch.Tensor:
    """Compute the Virtual Adversarial Training loss for a batch."""
    with torch.no_grad():
        logits_orig = model(x)

    d = torch.randn_like(x)
    d = xi * d / (d.norm(dim=1, keepdim=True) + 1e-12)

    for _ in range(num_power_iters):
        d.requires_grad_(True)
        logits_adv = model(x + d)
        kl = _kl_div_symmetric(logits_orig, logits_adv)
        d_grad = torch.autograd.grad(kl, d)[0]
        d = d_grad / (d_grad.norm(dim=1, keepdim=True) + 1e-12)

    d_adv = epsilon * d.detach()
    logits_adv = model(x + d_adv)
    return _kl_div_symmetric(logits_orig, logits_adv)


# ---------------------------------------------------------------------------
# Pi-Model
# ---------------------------------------------------------------------------

def pi_model_loss(
    model: nn.Module,
    x: torch.Tensor,
    dropout_mask: bool = True,
) -> torch.Tensor:
    """Pi-Model consistency loss: two stochastic forward passes should agree."""
    if not dropout_mask:
        with torch.no_grad():
            logits_1 = model(x)
        logits_2 = model(x)
    else:
        logits_1 = model(x)
        logits_2 = model(x)

    return F.mse_loss(
        F.softmax(logits_1, dim=1),
        F.softmax(logits_2, dim=1),
    )


# ---------------------------------------------------------------------------
# Combined loss classes
# ---------------------------------------------------------------------------

class CombinedSSLLoss(nn.Module):
    """Weighted combination: CE + VAT + Pi-Model.

    total = ce_loss + vat_weight * vat_loss + pi_weight * pi_loss
    """

    def __init__(
        self,
        vat_weight: float = 0.3,
        pi_weight: float = 0.1,
        vat_epsilon: float = 2.0,
        ramp_up_epochs: int = 50,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.vat_weight = vat_weight
        self.pi_weight = pi_weight
        self.vat_epsilon = vat_epsilon
        self.ramp_up_epochs = ramp_up_epochs
        self.current_epoch = 0
        self.class_weights = class_weights

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def _ramp_factor(self) -> float:
        if self.ramp_up_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.ramp_up_epochs)

    def forward(
        self,
        model: nn.Module,
        x_labeled: torch.Tensor,
        y_labeled: torch.Tensor,
        x_unlabeled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = model(x_labeled)
        ce = F.cross_entropy(logits, y_labeled, weight=self.class_weights)

        if x_unlabeled is None or (self.vat_weight == 0 and self.pi_weight == 0):
            return ce

        ramp = self._ramp_factor()
        total = ce

        if self.vat_weight > 0 and x_unlabeled.numel() > 0:
            total = total + ramp * self.vat_weight * vat_loss(
                model, x_unlabeled, epsilon=self.vat_epsilon,
            )

        if self.pi_weight > 0 and x_unlabeled.numel() > 0:
            total = total + ramp * self.pi_weight * pi_model_loss(model, x_unlabeled)

        return total


class FixMatchLoss(nn.Module):
    """FixMatch-style masked consistency loss with curriculum threshold.

    total = CE(labeled) + fm_weight * FixMatch_loss(unlabeled)

    The FixMatch threshold follows a round-based curriculum:
    threshold_round = max(initial_threshold * decay^(round-1), min_threshold)

    Parameters
    ----------
    fm_weight : float
        Weight for the FixMatch consistency loss relative to CE.
    fm_threshold : float
        Confidence floor for weak-augmentation pseudo-labels.
    weak_std : float
        Gaussian noise std for weak augmentation.
    strong_std : float
        Gaussian noise std for strong augmentation.
    dropout_rate : float
        Feature dropout rate for strong augmentation.
    ramp_up_epochs : int
        Epoch-based linear ramp-up for the FixMatch weight.
    class_weights : Tensor or None
        Optional per-class weights for the supervised CE term.
    """

    def __init__(
        self,
        fm_weight: float = 1.0,
        fm_threshold: float = 0.95,
        weak_std: float = 0.10,
        strong_std: float = 0.30,
        dropout_rate: float = 0.20,
        ramp_up_epochs: int = 30,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.fm_weight = fm_weight
        self.fm_threshold = fm_threshold
        self.weak_std = weak_std
        self.strong_std = strong_std
        self.dropout_rate = dropout_rate
        self.ramp_up_epochs = ramp_up_epochs
        self.current_epoch = 0
        self.class_weights = class_weights

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def set_threshold(self, threshold: float) -> None:
        """Update the confidence threshold (for curriculum across rounds)."""
        self.fm_threshold = threshold

    def _ramp_factor(self) -> float:
        if self.ramp_up_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.ramp_up_epochs)

    def forward(
        self,
        model: nn.Module,
        x_labeled: torch.Tensor,
        y_labeled: torch.Tensor,
        x_unlabeled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Compute CE + FixMatch loss.

        Returns
        -------
        total_loss : Tensor
        num_confident : int
            Number of unlabeled samples that passed the confidence threshold
            (useful for logging / monitoring curriculum progress).
        """
        logits = model(x_labeled)
        ce = F.cross_entropy(logits, y_labeled, weight=self.class_weights)

        if x_unlabeled is None or x_unlabeled.numel() == 0:
            return ce, 0

        fm, num_confident = fixmatch_loss(
            model=model,
            x=x_unlabeled,
            threshold=self.fm_threshold,
            weak_std=self.weak_std,
            strong_std=self.strong_std,
            dropout_rate=self.dropout_rate,
        )

        ramp = self._ramp_factor()
        total = ce + ramp * self.fm_weight * fm
        return total, num_confident
