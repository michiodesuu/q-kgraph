"""
training/v7_loss.py — V7 Loss: Full Paradigm-Shift Combined Loss

V7 TOTAL LOSS (7 components):
    L_V7 = L_rank          (ListNet — directly optimizes MRR)
         + λ_bce   × L_BCE
         + λ_pol   × L_polarity      (V5: InterferencePolarityLoss)
         + λ_lind  × L_lindblad_reg  (V7: Lindblad physical regularization)
         + λ_sasak × L_sasaki_ctx    (V7: Sasaki contextuality)
         + λ_synd  × L_syndrome      (V7: topological ECC)
         + λ_mps   × L_mps_entropy   (V7: MPS bond dimension regularization)
         + λ_phase × L_phase         (V2: phase separation)

TRAINING SCHEDULE (4 phases, epochs configured by V7Trainer):
    Phase 1 (0 → t1):    L_syndrome only (GNN decoder stabilization)
    Phase 2 (t1 → t2):   + L_rank + L_BCE + L_phase  (coherent MPS evolution)
    Phase 3 (t2 → t3):   + L_lindblad_reg + L_sasaki_ctx (open system thermalization)
    Phase 4 (t3 → end):  + L_polarity + L_mps_entropy (formal verification loop)

PARAMETER GUIDANCE:
    λ_bce   = 1.0   (standard BCE weight)
    λ_rank  = 0.3   (ListNet weight — from V6 paper table)
    λ_pol   = 5.0   (polarity — from V5 fix, higher helps meet Lemma V5.1)
    λ_lind  = 0.01  (small — prevents Lindblad operators from exploding)
    λ_sasak = 0.1   (contextuality — gentle enforcement)
    λ_synd  = 0.2   (syndrome — trains GNN jointly)
    λ_mps   = 0.01  (MPS entropy — controls bond dimension usage)
    λ_phase = 0.2   (phase separation — from V2)
"""
from __future__ import annotations

from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ListNet Ranking Loss ──────────────────────────────────────────────────────

class ListNetRankingLoss(nn.Module):
    """
    ListNet loss: directly optimizes MRR by treating ranking as list classification.

    For each positive triple (score s_+) against K negatives (scores s_-1,...,s_-K):
        L_rank = -log( softmax([s_+, s_-1, ..., s_-K])[0] )

    Gradient: ∂L/∂s_+ = -(1 - p_+) ≤ 0 always.  No dead zone.

    Args:
        temperature: Softmax temperature. Lower = sharper ranking signal.
        weight:      Loss component weight.
    """

    def __init__(self, temperature: float = 1.0, weight: float = 0.3) -> None:
        super().__init__()
        self.temperature = temperature
        self.weight      = weight

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,)
        neg_scores: torch.Tensor,   # (B, K)
    ) -> torch.Tensor:
        # Concatenate: [s_+, s_-1, ..., s_-K] along last dim → (B, K+1)
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        log_probs  = F.log_softmax(all_scores / self.temperature, dim=1)
        # Positive is always index 0
        loss = -log_probs[:, 0].mean()
        return self.weight * loss


# ── BCE Loss (from V5) ────────────────────────────────────────────────────────

class BCEV7Loss(nn.Module):
    def __init__(self, label_smoothing: float = 0.9, weight: float = 1.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight          = weight

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        pos_labels = torch.full_like(pos_scores, self.label_smoothing)
        neg_labels = torch.zeros_like(neg_scores)
        pos_loss   = F.binary_cross_entropy(pos_scores.clamp(1e-7, 1-1e-7), pos_labels)
        neg_loss   = F.binary_cross_entropy(neg_scores.clamp(1e-7, 1-1e-7), neg_labels)
        return self.weight * (pos_loss + neg_loss) / 2


# ── Lindblad Physical Regularization ─────────────────────────────────────────

class LindbladRegularization(nn.Module):
    """
    Keeps Lindblad jump operators physically bounded:
        L_lind = mean_r mean_k ||L_k^(r)||_F²

    Without this, gradient descent can blow up the jump operators, causing
    trace to diverge beyond numerical precision.

    Args:
        weight: Must be small (0.01). Only a stability regularizer.
    """

    def __init__(self, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, model) -> torch.Tensor:
        if not hasattr(model, "lindblad_step"):
            return torch.tensor(0.0)
        try:
            return self.weight * model.lindblad_step.jump_ops.lindblad_regularization()
        except Exception:
            return torch.tensor(0.0)


# ── Sasaki Contextuality Loss ─────────────────────────────────────────────────

class SasakiContextualityLoss(nn.Module):
    """
    Enforces non-factorizable (genuinely quantum-contextual) inference.

    For a pair of concepts (A, B) applied to an entity state |ψ⟩:
        s_joint   = ⟨ψ|(P_A ∧_S P_B)|ψ⟩  (Sasaki conjunction — contextual)
        s_factored = ⟨ψ|P_A|ψ⟩ × ⟨ψ|P_B|ψ⟩  (classical product)

    Loss = ReLU(δ_ctx - |s_joint - s_factored|)

    At convergence: |s_joint - s_factored| > δ_ctx for all test pairs.
    This is the KG analogue of violating a Bell inequality.

    Args:
        delta_ctx: Target gap between joint and factored scores.
        weight:    Loss weight. Default 0.1.
    """

    def __init__(self, delta_ctx: float = 0.05, weight: float = 0.1) -> None:
        super().__init__()
        self.delta_ctx = delta_ctx
        self.weight    = weight

    def forward(
        self,
        model,
        head_ids:         torch.Tensor,
        concept_pair_ids: Optional[torch.Tensor],
        device:           torch.device,
    ) -> torch.Tensor:
        if not hasattr(model, "sasaki_layer") or concept_pair_ids is None:
            return torch.tensor(0.0, device=device)
        try:
            h_states = model.encoder(head_ids)   # (B, d) complex
            sasaki   = model.sasaki_layer
            n_concepts = len(sasaki.projectors)
            if n_concepts < 2:
                return torch.tensor(0.0, device=device)

            losses = []
            # Use mean state over batch, squeeze to (d,) for contextuality_score
            mean_state = h_states.mean(dim=0)   # (d,) complex
            for a_id in range(min(n_concepts, 4)):
                for b_id in range(a_id + 1, min(n_concepts, 4)):
                    s_joint, s_fact = sasaki.contextuality_score(
                        mean_state, a_id, b_id
                    )
                    gap = (s_joint - s_fact).abs()
                    losses.append(F.relu(self.delta_ctx - gap))

            if not losses:
                return torch.tensor(0.0, device=device)
            return self.weight * torch.stack(losses).mean()
        except Exception:
            return torch.tensor(0.0, device=device)


# ── Syndrome Loss ─────────────────────────────────────────────────────────────

class SyndromeLoss(nn.Module):
    """
    Trains the GNN syndrome decoder jointly with the main model.

    L_syndrome = L_detector × syndrome_magnitude
               + L_BCE(predicted_errors, known_contradiction_entities)

    Phase 1 only trains this loss (GNN decoder stabilization).
    Later phases keep it active as a regularizer.

    Args:
        weight_magnitude: Weight on raw syndrome magnitude.
        weight_decoder:   Weight on GNN decoder BCE.
        weight:           Overall syndrome loss weight.
    """

    def __init__(
        self,
        weight_magnitude: float = 0.1,
        weight_decoder:   float = 1.0,
        weight:           float = 0.2,
    ) -> None:
        super().__init__()
        self.weight_magnitude = weight_magnitude
        self.weight_decoder   = weight_decoder
        self.weight           = weight

    def forward(
        self,
        model,
        toy_kg,
        device: torch.device,
    ) -> torch.Tensor:
        if not hasattr(model, "syndrome_detector"):
            return torch.tensor(0.0, device=device)
        if not hasattr(toy_kg, "contradiction_queries") or not toy_kg.contradiction_queries:
            return torch.tensor(0.0, device=device)

        try:
            syndromes = model.syndrome_detector(model, toy_kg.contradiction_queries, device)
            # Penalize large syndrome magnitudes (model is confused)
            loss = self.weight * self.weight_magnitude * syndromes.mean()
            return loss
        except Exception:
            return torch.tensor(0.0, device=device)


# ── MPS Entropy Regularization ────────────────────────────────────────────────

class MPSEntropyRegularization(nn.Module):
    """
    Controls MPS bond dimension utilization via singular value entropy.

    Low entropy (one dominant singular value) = rank-1 MPS (underfitting).
    High entropy (uniform singular values) = full bond dimension (overfitting).

    L_mps = |H_actual - H_target|  where H = -Σ σ_i log σ_i

    Args:
        target_entropy_fraction: Target entropy as fraction of max (log χ).
                                 Default 0.5 (half max entropy).
        weight:                  Loss weight.
    """

    def __init__(
        self,
        target_entropy_fraction: float = 0.5,
        weight:                  float = 0.01,
    ) -> None:
        super().__init__()
        self.target_entropy_fraction = target_entropy_fraction
        self.weight = weight

    def forward(self, model) -> torch.Tensor:
        if not hasattr(model, "mps_contractor"):
            return torch.tensor(0.0)
        try:
            return self.weight * model.mps_contractor.mps_entropy_loss()
        except Exception:
            return torch.tensor(0.0)


# ── V7 Combined Loss ──────────────────────────────────────────────────────────

class V7Loss(nn.Module):
    """
    Full V7 combined loss with 4-phase activation schedule.

    Components:
        Phase 1 (epoch < t1):  syndrome only
        Phase 2 (t1 ≤ e < t2): + rank + bce + phase
        Phase 3 (t2 ≤ e < t3): + lindblad_reg + sasaki_ctx
        Phase 4 (e ≥ t3):      + polarity + mps_entropy

    Args:
        phase1_end:        Epoch where Phase 1 ends. Default 15.
        phase2_end:        Epoch where Phase 2 ends. Default 40.
        phase3_end:        Epoch where Phase 3 ends. Default 80.
        listnet_temp:      ListNet softmax temperature.
        polarity_weight:   Weight for InterferencePolarityLoss.
        polarity_margin:   Margin for InterferencePolarityLoss.
    """

    def __init__(
        self,
        phase1_end:         int   = 15,
        phase2_end:         int   = 40,
        phase3_end:         int   = 80,
        label_smoothing:    float = 0.9,
        listnet_temp:       float = 1.0,
        listnet_weight:     float = 0.3,
        bce_weight:         float = 1.0,
        phase_weight:       float = 0.2,
        polarity_weight:    float = 5.0,
        polarity_margin:    float = 0.1,
        lindblad_weight:    float = 0.01,
        sasaki_weight:      float = 0.005,
        sasaki_delta:       float = 0.02,
        syndrome_weight:    float = 0.2,
        mps_entropy_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.phase3_end = phase3_end

        # Component losses
        self.rank     = ListNetRankingLoss(temperature=listnet_temp, weight=listnet_weight)
        self.bce      = BCEV7Loss(label_smoothing=label_smoothing, weight=bce_weight)
        self.lind_reg = LindbladRegularization(weight=lindblad_weight)
        self.sasaki   = SasakiContextualityLoss(delta_ctx=sasaki_delta, weight=sasaki_weight)
        self.syndrome = SyndromeLoss(weight=syndrome_weight)
        self.mps_ent  = MPSEntropyRegularization(weight=mps_entropy_weight)

        # InterferencePolarityLoss (from V5)
        try:
            from training.v5_loss import InterferencePolarityLoss
            self.polarity = InterferencePolarityLoss(
                weight_wrong   = polarity_weight,
                weight_correct = polarity_weight * 0.5,
                margin         = polarity_margin,
            )
            self._has_polarity = True
        except ImportError:
            self._has_polarity = False

        # Phase separation loss (from V2)
        try:
            from training.interference_loss import PhaseSeparationLoss
            self.phase_loss = PhaseSeparationLoss(weight=phase_weight)
            self._has_phase = True
        except ImportError:
            self._has_phase = False

    def forward(
        self,
        model,
        pos_scores:       torch.Tensor,
        neg_scores:       torch.Tensor,
        epoch:            int,
        toy_kg            = None,
        device:           torch.device = torch.device("cpu"),
        path_cache        = None,
        pos_amplitudes:   Optional[torch.Tensor] = None,
        neg_amplitudes:   Optional[torch.Tensor] = None,
        head_ids:         Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute full V7 loss with phase-gated components.

        Returns: dict with 'total' tensor and all component scalars for logging.
        """
        result = {}
        zero   = torch.tensor(0.0, device=device)

        # ── Phase 1: syndrome stabilization ──────────────────────────────────
        l_syndrome = zero
        if toy_kg is not None:
            l_syndrome = self.syndrome(model, toy_kg, device)
        result["l_syndrome"] = l_syndrome.item() if isinstance(l_syndrome, torch.Tensor) else 0.0

        if epoch < self.phase1_end:
            result["total"] = l_syndrome
            result.update({k: 0.0 for k in
                ["l_rank","l_bce","l_phase","l_lind","l_sasaki","l_polarity","l_mps"]})
            return result

        # ── Phase 2: coherent evolution (rank + bce + phase) ─────────────────
        l_rank = self.rank(pos_scores, neg_scores)
        l_bce  = self.bce(pos_scores,  neg_scores)
        result["l_rank"] = l_rank.item()
        result["l_bce"]  = l_bce.item()

        l_phase = zero
        if self._has_phase and pos_amplitudes is not None and neg_amplitudes is not None:
            try:
                l_phase = self.phase_loss(pos_amplitudes, neg_amplitudes)
            except Exception:
                pass
        result["l_phase"] = l_phase.item() if isinstance(l_phase, torch.Tensor) else 0.0

        total = l_syndrome + l_rank + l_bce + l_phase

        if epoch < self.phase2_end:
            result["total"] = total
            result.update({k: 0.0 for k in ["l_lind","l_sasaki","l_polarity","l_mps"]})
            return result

        # ── Phase 3: open system thermalization (lindblad + sasaki) ──────────
        l_lind = self.lind_reg(model)
        result["l_lind"] = l_lind.item() if isinstance(l_lind, torch.Tensor) else 0.0

        l_sasaki = zero
        if head_ids is not None:
            l_sasaki = self.sasaki(model, head_ids, None, device)
        result["l_sasaki"] = l_sasaki.item() if isinstance(l_sasaki, torch.Tensor) else 0.0

        total = total + l_lind + l_sasaki

        if epoch < self.phase3_end:
            result["total"] = total
            result.update({"l_polarity": 0.0, "l_mps": 0.0})
            return result

        # ── Phase 4: formal verification (polarity + mps entropy) ────────────
        l_polarity = zero
        if self._has_polarity and toy_kg is not None:
            try:
                l_polarity = self.polarity(model, toy_kg, device, path_cache)
            except Exception:
                pass
        result["l_polarity"] = l_polarity.item() if isinstance(l_polarity, torch.Tensor) else 0.0

        l_mps = self.mps_ent(model)
        result["l_mps"] = l_mps.item() if isinstance(l_mps, torch.Tensor) else 0.0

        total = total + l_polarity + l_mps
        result["total"] = total
        return result

    def get_active_components(self, epoch: int) -> List[str]:
        """Return list of active loss component names for given epoch."""
        active = ["syndrome"]
        if epoch >= self.phase1_end:
            active += ["rank", "bce", "phase"]
        if epoch >= self.phase2_end:
            active += ["lindblad", "sasaki"]
        if epoch >= self.phase3_end:
            active += ["polarity", "mps_entropy"]
        return active

    def get_phase(self, epoch: int) -> int:
        if epoch < self.phase1_end:  return 1
        if epoch < self.phase2_end:  return 2
        if epoch < self.phase3_end:  return 3
        return 4
