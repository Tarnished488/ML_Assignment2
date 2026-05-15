"""Consistency regularisation losses for semi-supervised training.

VAT (Virtual Adversarial Training): Miyato et al., 2018.
Encourages the model to produce *smooth* predictions — a small perturbation
to the input should not significantly change the output distribution.
This loss applies to **both** labeled and unlabeled data, making it a
powerful way to leverage the 10,000 unlabeled samples during training.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _kl_div_symmetric(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Numerically stable symmetric KL divergence.

    Works entirely in log-space to avoid softmax underflow (which causes
    NaN when the model becomes overconfident and produces extreme logits).
    """
    log_p = F.log_softmax(logits_p, dim=1)
    log_q = F.log_softmax(logits_q, dim=1)

    # KL(P||Q) = sum_i P_i * (log P_i - log Q_i)
    # Use F.kl_div with log_target=True — both input and target are log-probs.
    kl_pq = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
    kl_qp = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
    return (kl_pq + kl_qp) / 2.0


def vat_loss(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float = 2.0,
    xi: float = 1e-6,
    num_power_iters: int = 1,
) -> torch.Tensor:
    """Compute the Virtual Adversarial Training loss for a batch.

    Finds the adversarial perturbation direction that *maximally* changes
    the model's output distribution, then penalises that change.

    Parameters
    ----------
    model : nn.Module
    x : Tensor of shape (B, D)
    epsilon : float
        Perturbation norm bound.
    xi : float
        Small constant for initialising the perturbation direction.
    num_power_iters : int
        Number of power-iteration steps to approximate the dominant
        eigenvector of the local KL Hessian.
    """
    # Original (unperturbed) predictions — treated as fixed targets
    with torch.no_grad():
        logits_orig = model(x)

    # Random unit perturbation
    d = torch.randn_like(x)
    d = xi * d / (d.norm(dim=1, keepdim=True) + 1e-12)

    # Power iteration: find the adversarial direction
    for _ in range(num_power_iters):
        d.requires_grad_(True)
        logits_adv = model(x + d)
        kl = _kl_div_symmetric(logits_orig, logits_adv)
        d_grad = torch.autograd.grad(kl, d)[0]
        d = d_grad / (d_grad.norm(dim=1, keepdim=True) + 1e-12)

    # Scale to epsilon
    d_adv = epsilon * d.detach()

    # Final VAT loss: KL(original || adversarial)
    logits_adv = model(x + d_adv)
    return _kl_div_symmetric(logits_orig, logits_adv)


def pi_model_loss(
    model: nn.Module,
    x: torch.Tensor,
    dropout_mask: bool = True,
) -> torch.Tensor:
    """Pi-Model consistency loss: two stochastic forward passes should agree.

    Applies dropout (and optionally Gaussian noise) to enforce consistency.

    Parameters
    ----------
    model : nn.Module
    x : Tensor of shape (B, D)
    dropout_mask : bool
        Whether to use dropout stochasticity (model must be in train mode).
    """
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


class CombinedSSLLoss(nn.Module):
    """Weighted combination of supervised CE + VAT + Pi-Model losses.

    total = ce_loss + vat_weight * vat_loss + pi_weight * pi_loss

    Parameters
    ----------
    vat_weight : float
        Weight for the VAT consistency loss.
    pi_weight : float
        Weight for the Pi-Model consistency loss.
    vat_epsilon : float
        Perturbation norm bound for VAT.
    ramp_up_epochs : int
        Number of epochs over which to linearly ramp up unsupervised weights
        from 0 to their full values (warm-up).
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
        """Compute the combined loss.

        Parameters
        ----------
        model : nn.Module
        x_labeled : Tensor (B_l, D)
        y_labeled : Tensor (B_l,)
        x_unlabeled : Tensor (B_u, D) or None
            If None, only supervised CE is computed.
        """
        # Supervised CE (with optional class weights for balance)
        logits = model(x_labeled)
        ce = F.cross_entropy(logits, y_labeled, weight=self.class_weights)

        if x_unlabeled is None or (self.vat_weight == 0 and self.pi_weight == 0):
            return ce

        ramp = self._ramp_factor()
        total = ce

        # VAT on unlabeled data
        if self.vat_weight > 0 and x_unlabeled.numel() > 0:
            total = total + ramp * self.vat_weight * vat_loss(
                model, x_unlabeled, epsilon=self.vat_epsilon,
            )

        # Pi-Model on unlabeled data
        if self.pi_weight > 0 and x_unlabeled.numel() > 0:
            total = total + ramp * self.pi_weight * pi_model_loss(model, x_unlabeled)

        return total
