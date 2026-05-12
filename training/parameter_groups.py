"""
training/parameter_groups.py  --  Challenge 1 Fix:
    Separate Learning Rates for Phase Parameters

WHY THIS FILE EXISTS:
    Challenge 1: when imaginary components start learning (epochs 10-30),
    you see loss spikes. The cause: phase angles (theta in DiagonalUnitary)
    receive large gradients because small phase changes cause large amplitude changes.
    A single global LR causes instability.

    Solution: use PyTorch parameter groups with SEPARATE LRs:
        - Entity embeddings (real + imag): standard LR (e.g., 0.001)
        - Phase parameters (unitary angles): lower LR (e.g., 0.0001)
        - Relation bias:                    standard LR

    This gives the optimizer a knob for each parameter type instead of
    forcing all parameters to share one LR.

    Also provides: GradientMonitor -- detects gradient spikes early and
    logs warnings so you can stop a run before it diverges.
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


def build_optimizer_with_groups(
    model:          nn.Module,
    base_lr:        float = 1e-3,
    phase_lr_ratio: float = 0.1,
    bias_lr_ratio:  float = 1.0,
    weight_decay:   float = 1e-5,
    optimizer_type: str   = "adam",
) -> torch.optim.Optimizer:
    """
    Build optimizer with separate learning rates per parameter group.

    Parameter groups:
        Group 0 "phase":     unitary phase angles  -- LR = base_lr * phase_lr_ratio
        Group 1 "entity":    entity embeddings     -- LR = base_lr
        Group 2 "other":     everything else       -- LR = base_lr

    Args:
        model:          QuantumReasoner (or any model with .unitary.phases or similar).
        base_lr:        Base learning rate for entity embeddings.
        phase_lr_ratio: Phase LR = base_lr * phase_lr_ratio.
                        Recommended: 0.1 (10x lower than entity LR).
        bias_lr_ratio:  Bias LR multiplier.
        weight_decay:   L2 regularization (not applied to bias terms).
        optimizer_type: "adam" | "adamw".

    Returns:
        Configured PyTorch optimizer.

    Example:
        >>> optimizer = build_optimizer_with_groups(
        ...     model=quantum_reasoner,
        ...     base_lr=0.001,
        ...     phase_lr_ratio=0.1,  # phase LR = 0.0001
        ... )
        >>> # Check groups
        >>> for g in optimizer.param_groups:
        ...     print(g['name'], g['lr'], sum(p.numel() for p in g['params']))
    """
    phase_params   = []
    entity_params  = []
    bias_params    = []
    other_params   = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Phase angle parameters in unitary operators
        if any(key in name for key in ["phases", "angles", "cluster_phases",
                                        "relation_delta", "primary_phases",
                                        "h_real_triu", "h_imag_triu"]):
            phase_params.append(param)

        # Entity embedding parameters
        elif any(key in name for key in ["real_embeddings", "imag_embeddings",
                                          "entity_embeddings", "entity_real",
                                          "entity_imag"]):
            entity_params.append(param)

        # Bias terms
        elif "bias" in name or "relation_bias" in name:
            bias_params.append(param)

        # Everything else (path weights, cluster logits, etc.)
        else:
            other_params.append(param)

    param_groups = []

    if phase_params:
        param_groups.append({
            "name":         "phase",
            "params":       phase_params,
            "lr":           base_lr * phase_lr_ratio,
            "weight_decay": weight_decay,
        })

    if entity_params:
        param_groups.append({
            "name":         "entity",
            "params":       entity_params,
            "lr":           base_lr,
            "weight_decay": weight_decay,
        })

    if bias_params:
        param_groups.append({
            "name":         "bias",
            "params":       bias_params,
            "lr":           base_lr * bias_lr_ratio,
            "weight_decay": 0.0,   # no weight decay on biases
        })

    if other_params:
        param_groups.append({
            "name":         "other",
            "params":       other_params,
            "lr":           base_lr,
            "weight_decay": weight_decay,
        })

    # Fallback: if no groups matched, use all parameters with base LR
    if not param_groups:
        param_groups = [{"params": list(model.parameters()), "lr": base_lr}]

    opt_map = {
        "adam":   torch.optim.Adam,
        "adamw":  torch.optim.AdamW,
    }
    opt_class = opt_map.get(optimizer_type, torch.optim.Adam)
    return opt_class(param_groups, lr=base_lr, weight_decay=weight_decay)


class GradientMonitor:
    """
    Monitors gradient norms per parameter group to detect training instability.

    Use this alongside build_optimizer_with_groups() to catch gradient spikes
    early (in the first 30 epochs) before they cause divergence.

    Three alert levels:
        INFO:    Gradient norm is normal.
        WARNING: Gradient norm > spike_threshold. Log and continue.
        CRITICAL: Gradient norm > 10 * spike_threshold. Suggests divergence.
                  Caller should reduce LR or restore last checkpoint.

    Args:
        spike_threshold: Gradient norm above this triggers a WARNING.
                         Recommended starting value: 10.0.
        window_size:     Number of recent gradient norms to track per group.
                         Used to compute running mean for adaptive detection.

    Example:
        >>> monitor = GradientMonitor(spike_threshold=10.0)
        >>> # In training loop, AFTER loss.backward() and BEFORE optimizer.step():
        >>> alerts = monitor.check(model)
        >>> for alert in alerts:
        ...     if alert['level'] == 'CRITICAL':
        ...         log.warning(f"Gradient spike: {alert}")
        ...         # optionally: optimizer.zero_grad(); continue
    """

    def __init__(
        self,
        spike_threshold: float = 10.0,
        window_size:     int   = 20,
    ) -> None:
        self.spike_threshold = spike_threshold
        self.window_size     = window_size
        self._history: dict[str, list[float]] = {}

    def check(
        self,
        model: nn.Module,
        clip_norm: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Compute gradient norms, clip them, and check for spikes.

        MUST be called AFTER loss.backward() and BEFORE optimizer.step().

        Args:
            model:     The model being trained.
            clip_norm: Clip gradients to this norm AFTER monitoring.

        Returns:
            List of alert dicts. Each has 'name', 'norm', 'level', 'message'.
            Empty list = all gradients normal.
        """
        alerts = []

        # Compute per-group gradient norms before clipping
        group_norms: dict[str, float] = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            # Determine group
            if any(k in name for k in ["phases", "angles", "cluster_phases",
                                         "relation_delta", "primary_phases"]):
                group = "phase"
            elif any(k in name for k in ["real_embeddings", "imag_embeddings",
                                           "entity_embeddings"]):
                group = "entity"
            else:
                group = "other"

            grad_norm = param.grad.detach().norm().item()
            if group not in group_norms:
                group_norms[group] = 0.0
            group_norms[group] = max(group_norms[group], grad_norm)

        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        # Check for spikes
        for group, norm in group_norms.items():
            if group not in self._history:
                self._history[group] = []

            history = self._history[group]
            history.append(norm)
            if len(history) > self.window_size:
                history.pop(0)

            # Adaptive threshold: mean of recent norms * 3
            recent_mean = sum(history) / len(history) if history else norm
            adaptive_threshold = max(self.spike_threshold, recent_mean * 3.0)

            if norm > 10.0 * self.spike_threshold:
                level = "CRITICAL"
            elif norm > adaptive_threshold:
                level = "WARNING"
            else:
                level = "INFO"

            if level != "INFO":
                alerts.append({
                    "name":    group,
                    "norm":    norm,
                    "level":   level,
                    "mean":    recent_mean,
                    "message": (
                        f"[{level}] Group '{group}': grad_norm={norm:.4f} "
                        f"(recent_mean={recent_mean:.4f}, "
                        f"threshold={adaptive_threshold:.4f})"
                    ),
                })

        return alerts

    def summary(self) -> dict[str, float]:
        """Return mean gradient norm per group over recent history."""
        return {
            group: sum(h) / len(h) if h else 0.0
            for group, h in self._history.items()
        }
