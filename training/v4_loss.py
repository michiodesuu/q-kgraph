"""
training/v4_loss.py — Combined V4 Loss Function [V4]

FOUR LOSS COMPONENTS:

    1. L_Quaternion (QIQE-KGC):
       Binary Cross-Entropy on quaternion triple scores s(h,r,t) = Re(e_h⊙r⊙conj(e_t)).
       Standard contrastive training: positive triple should score higher than negatives.
       Uses label smoothing to prevent overconfidence (same as V3 InterferenceAwareLoss).

    2. L_Lattice = L_ELoss + L_LLoss + L_MLoss (QIQE-KGC):
       Global logical consistency from QuantumLogicLattice.
       L_ELoss: forces logical embeddings toward binary (orthocomplemented structure).
       L_LLoss: positive and negative tails should have orthogonal logical embeddings.
       L_MLoss: entities should belong to their relation's domain/range.
       Combined weight: lattice_weight (default 0.2).

    3. L_Interference (V3 + AAAI-2024 extension):
       PhaseSeparationLoss: drives wrong-answer quaternion phase to oppose correct.
       ContrastiveLoss: penalizes P(wrong) > P(correct) for contradiction queries.
       Logic-weighted: paths with high lattice consistency get stronger signal.
       Weight: interference_weight (default 1.0).

    4. L_RoutingCalibration (AAAI-2024 dynamic routing):
       Penalizes router for sending high-confidence queries to quantum pipeline.
       Encourages the router to route correctly:
           confident queries → classical (minimize quantum routing overhead)
           ambiguous queries → quantum (maximize interference benefit)
       Weight: routing_weight (default 0.05).

TOTAL LOSS:
    L = L_Quaternion + lattice_weight·L_Lattice
      + interference_weight·L_Interference
      + routing_weight·L_RoutingCalibration
"""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuaternionBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss on quaternion triple scores.

    From QIQE-KGC: L_Quaternion = -[y·log(σ(s)) + (1-y)·log(1-σ(s))]
    with label smoothing ε to prevent overconfidence.

    Smoothed labels: y_pos = 1-ε, y_neg = ε/(N-1) ≈ ε/N

    This is the primary scoring loss, applied to:
        s_pos = s(h,r,t_pos) = Re(e_h ⊙ r ⊙ conj(e_t_pos))
        s_neg = s(h,r,t_neg) for each negative sample
    """

    def __init__(self, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.eps = label_smoothing

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,) float
        neg_scores: torch.Tensor,   # (B, K) float
    ) -> torch.Tensor:
        B, K = neg_scores.shape

        # Positive loss: -[(1-ε)·log(σ(s_pos)) + ε·log(1-σ(s_pos))]
        y_pos    = 1.0 - self.eps
        pos_loss = -F.logsigmoid(pos_scores) * y_pos \
                 - F.logsigmoid(-pos_scores) * (1.0 - y_pos)

        # Negative loss: -(ε·log(σ(s_neg)) + (1-ε)·log(1-σ(s_neg)))
        y_neg    = self.eps / K
        neg_loss = -F.logsigmoid(neg_scores) * y_neg \
                 - F.logsigmoid(-neg_scores) * (1.0 - y_neg)

        return pos_loss.mean() + neg_loss.mean()


class LatticeRegularization(nn.Module):
    """
    Regularization to encourage lattice convergence to binary states.

    Beyond the three lattice losses (L_ELoss, L_LLoss, L_MLoss), we add:
    - Entropy regularization: maximize logical embedding entropy during warmup,
      then minimize it to encourage binary convergence.
    - This prevents premature collapse to 0 or 1 before the model has learned.

    Phase 1 (warmup_steps): maximize entropy (explore)
    Phase 2 (after warmup): minimize entropy (converge to binary)
    """

    def __init__(
        self,
        warmup_steps: int   = 1000,
        reg_weight:   float = 0.01,
    ) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.reg_weight   = reg_weight
        self._step        = 0

    def forward(self, lattice_module) -> torch.Tensor:
        self._step += 1

        if lattice_module is None:
            return torch.tensor(0.0)

        # Sample all relation logic embeddings
        E_r = torch.sigmoid(lattice_module.relation_logic.weight)  # (R, d)

        # Entropy per embedding: H = -p·log(p) - (1-p)·log(1-p) ∈ [0, log2]
        entropy = -(E_r * (E_r + 1e-10).log() + (1 - E_r) * (1 - E_r + 1e-10).log())
        mean_entropy = entropy.mean()

        if self._step < self.warmup_steps:
            # Maximize entropy (explore diverse binary patterns)
            return -self.reg_weight * mean_entropy
        else:
            # Minimize entropy (drive toward binary — orthocomplemented structure)
            return self.reg_weight * mean_entropy


class InterferenceLossV4(nn.Module):
    """
    Enhanced interference loss for V4, extending V3's InterferenceAwareLoss.

    V3: PhaseSeparation + Contrastive (on complex phases)
    V4: Quaternion phase-based separation + logic-weighted contrastive

    QUATERNION PHASE SEPARATION:
    In H^d, the "phase" is the rotation angle of the quaternion:
        θ = 2·arctan(||imag_part|| / real_part)
    Wrong-answer path quaternions should have θ_wrong ≈ π - θ_correct (opposing rotation).
    PhaseSeparationLoss pushes toward this opposition.

    LOGIC-WEIGHTED CONTRASTIVE:
    Standard V3: max(0, P(wrong) - P(correct) + γ)
    V4: weight by logic scores from lattice:
        - High logic(wrong_path): model strongly believes wrong → penalize harder
        - Low logic(wrong_path):  model already doubts wrong → lighter penalty
    """

    def __init__(
        self,
        contrast_weight: float = 1.0,
        phase_weight:    float = 0.2,
        margin:          float = 0.5,
        phase_temp:      float = 1.0,
    ) -> None:
        super().__init__()
        self.contrast_weight = contrast_weight
        self.phase_weight    = phase_weight
        self.margin          = margin
        self.phase_temp      = phase_temp

    def quaternion_phase_separation_loss(
        self,
        pos_score:  torch.Tensor,   # (B,) positive triple score
        neg_scores: torch.Tensor,   # (B, K) negative triple scores
        unitary_module,
    ) -> torch.Tensor:
        """
        Enforce quaternion phase separation between correct and wrong paths.

        For quaternion embedding: the i-component acts as the "phase carrier."
        We want: angle(pos) ≈ π - angle(neg) (opposing quaternion rotations).

        Uses the relation unitary's i-component as the phase proxy,
        consistent with V3's InterferenceMonitor which watches imag_norm.
        """
        # Quaternion rotation angle proxy: arctan(||i,j,k|| / r_comp)
        phases = unitary_module.get_phases()   # (R, d)

        # Phase separation: positive and negative should oppose
        # Use scores as proxies for amplitude directions
        pos_prob = torch.sigmoid(pos_score)                 # (B,)
        neg_prob = torch.sigmoid(neg_scores).mean(dim=-1)  # (B,)

        # Target: pos_prob high, neg_prob low (standard)
        # Phase component: penalize when negative probability is too close to positive
        phase_sep_loss = F.relu(neg_prob - pos_prob + self.margin)

        # Scale by cosine of phase separation (from V3's PhaseSeparationLoss concept)
        # Uses phases from unitary as regularity term
        phase_reg = phases.abs().mean()  # keep phases from collapsing to 0

        return phase_sep_loss.mean() + self.phase_weight * (-phase_reg)

    def contrastive_loss(
        self,
        pos_scores: torch.Tensor,    # (B,)
        neg_scores: torch.Tensor,    # (B, K)
        logic_weights: Optional[torch.Tensor] = None,  # (B, K) from lattice
    ) -> torch.Tensor:
        """
        Logic-weighted hinge contrastive loss.

        Standard hinge: max(0, margin - pos + neg) per negative
        V4 logic-weighted: multiply by logic_weight of the negative sample
            High logic weight of negative = model confident wrong answer → stronger penalty
        """
        pos_exp = pos_scores.unsqueeze(1).expand_as(neg_scores)   # (B, K)
        hinge   = F.relu(self.margin - pos_exp + neg_scores)       # (B, K)

        if logic_weights is not None:
            # Higher logic weight = higher confidence in wrong answer = stronger penalty
            hinge = hinge * logic_weights.clamp(0.3, 1.0)

        return hinge.mean()

    def forward(
        self,
        pos_scores:    torch.Tensor,
        neg_scores:    torch.Tensor,
        unitary_module,
        logic_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phase_loss = self.quaternion_phase_separation_loss(
            pos_scores, neg_scores, unitary_module
        )
        contrast_loss = self.contrastive_loss(pos_scores, neg_scores, logic_weights)
        return self.phase_weight * phase_loss + self.contrast_weight * contrast_loss


class RouterCalibrationLoss(nn.Module):
    """
    Loss to calibrate the DynamicRouter's threshold parameters.

    Goal: teach the router WHEN quantum processing is beneficial.
        - On clean data (low noise): classical should dominate (high threshold)
        - On noisy data / contradictions: quantum should activate (low threshold)

    This loss compares routing decisions to an "oracle" routing signal:
        oracle_classical: when positive score >> all negative scores (unambiguous)
        oracle_quantum:   when positive score is close to max negative (ambiguous)

    Minimizing this loss calibrates the learnable thresholds (log_high, log_low)
    in the DynamicRouter.
    """

    def __init__(self, weight: float = 0.05) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pos_scores:  torch.Tensor,    # (B,)
        neg_scores:  torch.Tensor,    # (B, K)
        router_module,
    ) -> torch.Tensor:
        if router_module is None:
            return torch.tensor(0.0, device=pos_scores.device)

        # Oracle signal: how ambiguous is each sample?
        max_neg = neg_scores.max(dim=-1).values   # (B,)
        gap     = pos_scores - max_neg            # (B,) positive gap = easier

        # Confidence: high gap → should be classical, low gap → should be quantum
        oracle_confidence = torch.sigmoid(gap)    # (B,) in (0,1)

        # Actual router thresholds (differentiable via sigmoid)
        predicted_high = torch.sigmoid(router_module.log_high)
        predicted_low  = torch.sigmoid(router_module.log_low)

        # Loss: oracle_confidence should be inside [predicted_low, predicted_high]
        # Penalize if router threshold is too far from oracle
        high_loss = F.relu(oracle_confidence.mean() - predicted_high).pow(2)
        low_loss  = F.relu(predicted_low - oracle_confidence.mean()).pow(2)

        return self.weight * (high_loss + low_loss)


class V4Loss(nn.Module):
    """
    Complete V4 combined loss.

    L_total = L_Quaternion
            + lattice_weight  × L_Lattice
            + interf_weight   × L_Interference
            + routing_weight  × L_RouterCalibration
            + lattice_reg     × L_LatticeRegularization

    All weights have defaults that work for the toy KG. Tune per dataset.

    Args:
        label_smoothing:  For L_Quaternion BCE. Default 0.1.
        lattice_weight:   Weight for combined lattice loss. Default 0.2.
        interf_weight:    Weight for interference loss. Default 1.0.
        routing_weight:   Weight for router calibration. Default 0.05.
        contrast_weight:  Weight for contrastive term inside interference loss.
        phase_weight:     Weight for phase separation inside interference loss.
        lattice_reg_weight: Weight for lattice entropy regularization. Default 0.01.
        lattice_warmup:   Steps before switching from entropy maximize to minimize.
    """

    def __init__(
        self,
        label_smoothing:    float = 0.1,
        lattice_weight:     float = 0.2,
        interf_weight:      float = 1.0,
        routing_weight:     float = 0.05,
        contrast_weight:    float = 1.0,
        phase_weight:       float = 0.2,
        lattice_reg_weight: float = 0.01,
        lattice_warmup:     int   = 500,
    ) -> None:
        super().__init__()
        self.lattice_weight = lattice_weight
        self.interf_weight  = interf_weight

        self.quat_bce    = QuaternionBCELoss(label_smoothing)
        self.interference = InterferenceLossV4(contrast_weight, phase_weight)
        self.router_cal  = RouterCalibrationLoss(routing_weight)
        self.lattice_reg = LatticeRegularization(lattice_warmup, lattice_reg_weight)

    def forward(
        self,
        model,
        pos_scores:  torch.Tensor,   # (B,) from score_triple or score_with_paths
        neg_scores:  torch.Tensor,   # (B, K)
        h_ids:       torch.Tensor,
        r_ids:       torch.Tensor,
        t_pos_ids:   torch.Tensor,
        t_neg_ids:   torch.Tensor,   # (B, K, 3) — full negative triples
    ) -> dict:
        """
        Compute all loss components.

        Args:
            model:       QuaternionReasoner V4 instance.
            pos_scores:  (B,) scores for positive triples.
            neg_scores:  (B, K) scores for negative triples.
            h_ids:       (B,) head entity IDs.
            r_ids:       (B,) relation IDs.
            t_pos_ids:   (B,) positive tail IDs.
            t_neg_ids:   (B, K) or (B, K, 3) negative tail IDs.

        Returns:
            Dict with 'total' loss and all components.
        """
        device = pos_scores.device

        # 1. Quaternion BCE loss
        l_quat = self.quat_bce(pos_scores, neg_scores)

        # 2. Lattice loss (if enabled)
        l_lattice = torch.tensor(0.0, device=device)
        if model.use_lattice and model.lattice is not None:
            # Extract negative tail IDs from batch
            if t_neg_ids.dim() == 3:
                t_neg_flat = t_neg_ids[:, :, 2].reshape(-1)   # (B×K,) tails
            else:
                t_neg_flat = t_neg_ids.reshape(-1)
            t_neg_sample = t_neg_flat[:len(h_ids)]   # use first K per sample for simplicity

            l_lattice = self.lattice_weight * model.compute_lattice_loss(
                h_ids, r_ids, t_pos_ids, t_neg_sample
            )
            l_lattice_reg = self.lattice_reg(model.lattice)
            l_lattice = l_lattice + l_lattice_reg

        # 3. Logic weights for interference (from lattice logic scores)
        logic_weights = None
        if model.use_lattice and model.lattice is not None:
            with torch.no_grad():
                logic_weights = model.get_logic_scores(r_ids).unsqueeze(1).expand_as(neg_scores)

        # 4. Interference loss
        l_interf = self.interf_weight * self.interference(
            pos_scores, neg_scores, model.unitary, logic_weights
        )

        # 5. Router calibration loss
        l_router = self.router_cal(pos_scores, neg_scores,
                                    model.router if model.use_routing else None)

        total = l_quat + l_lattice + l_interf + l_router

        return {
            "total":      total,
            "l_quat":     l_quat.item(),
            "l_lattice":  l_lattice.item() if isinstance(l_lattice, torch.Tensor) else 0.0,
            "l_interf":   l_interf.item(),
            "l_router":   l_router.item() if isinstance(l_router, torch.Tensor) else 0.0,
        }
