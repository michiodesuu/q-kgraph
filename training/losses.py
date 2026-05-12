"""
training/losses.py — Loss Functions, One Per Model Family  [V1]

Each model was co-designed with its native loss function.
Using the wrong loss disadvantages the baseline and creates an unfair comparison.
Always use the appropriate loss per model type.
"""
from __future__ import annotations
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F


class BinaryCrossEntropyLoss(nn.Module):
    """
    BCE loss for QuantumReasoner and ComplEx.

    Appropriate because Born rule outputs lie in [0, 1].
    Label smoothing: positive labels set to label_smoothing (default 0.9)
    instead of 1.0, for robustness to KG noise.

    Args:
        label_smoothing: Positive label value. 0.9 recommended.
        reduction:       "mean" (default) or "sum".
    """
    def __init__(self, label_smoothing: float = 0.9, reduction: str = "mean"):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,) float — positive triple scores
        neg_scores: torch.Tensor,   # (B, K) float — negative triple scores
    ) -> torch.Tensor:
        pos_labels = torch.full_like(pos_scores, self.label_smoothing)
        neg_labels = torch.zeros_like(neg_scores)

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, reduction=self.reduction)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores.reshape(-1),
            neg_labels.reshape(-1),
            reduction=self.reduction,
        )
        return pos_loss + neg_loss


class MarginRankingLoss(nn.Module):
    """
    Margin ranking loss for TransE (Bordes et al. 2013 original loss).

    L = mean(max(0, gamma - score_pos + score_neg))

    Args:
        margin: The gamma margin. Default 9.0 for FB15k-237 (from paper).
        reduction: "mean" or "sum".
    """
    def __init__(self, margin: float = 9.0, reduction: str = "mean"):
        super().__init__()
        self.margin    = margin
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,) float
        neg_scores: torch.Tensor,   # (B, K) float
    ) -> torch.Tensor:
        B, K = neg_scores.shape
        pos_exp  = pos_scores.unsqueeze(1).expand(B, K)  # (B, K)
        hinge    = F.relu(self.margin - pos_exp + neg_scores)   # (B, K)

        if self.reduction == "mean":
            return hinge.mean()
        return hinge.sum()


class SelfAdversarialLoss(nn.Module):
    """
    Self-adversarial loss for RotatE (Sun et al. 2019 original loss).

    Weights negative samples by their score (hard negatives weighted more).

    L = -log σ(γ + score_pos) - Σᵢ p(neg_i) log σ(-γ - score_neg_i)
    where p(neg_i) = softmax(α · score_neg_i) [hard-negative weighting]

    Args:
        margin:      γ margin parameter. Default 9.0.
        adversarial_temperature: α temperature for negative weighting.
        reduction:   "mean" or "sum".
    """
    def __init__(
        self,
        margin:                 float = 9.0,
        adversarial_temperature: float = 1.0,
        reduction:              str   = "mean",
    ):
        super().__init__()
        self.margin      = margin
        self.temperature = adversarial_temperature
        self.reduction   = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,) float
        neg_scores: torch.Tensor,   # (B, K) float
    ) -> torch.Tensor:
        # Positive loss: -log σ(γ + score_pos)
        pos_loss = -F.logsigmoid(self.margin + pos_scores)

        # Adversarial negative weights: softmax(α · score_neg)
        neg_weights = F.softmax(self.temperature * neg_scores, dim=-1).detach()

        # Negative loss: -Σ p(neg_i) log σ(-γ - score_neg_i)
        neg_loss = -(neg_weights * F.logsigmoid(-self.margin - neg_scores)).sum(dim=-1)

        loss = pos_loss + neg_loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def build_loss(
    loss_type:       str   = "bce",
    label_smoothing: float = 0.9,
    margin:          float = 9.0,
    temperature:     float = 1.0,
) -> nn.Module:
    """
    Factory function for loss construction.

    Args:
        loss_type:       "bce" (QuantumReasoner/ComplEx), "margin" (TransE),
                         "self_adversarial" (RotatE).
        label_smoothing: For BCE loss.
        margin:          For margin and self-adversarial losses.
        temperature:     For self-adversarial loss.

    Returns:
        Loss module instance.
    """
    registry = {
        "bce":              lambda: BinaryCrossEntropyLoss(label_smoothing),
        "margin":           lambda: MarginRankingLoss(margin),
        "self_adversarial": lambda: SelfAdversarialLoss(margin, temperature),
    }
    if loss_type not in registry:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Choose from: {list(registry.keys())}")
    return registry[loss_type]()
