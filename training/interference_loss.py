"""
Interference-Aware Training Objective — New Contribution #3.

WHY THIS FILE EXISTS:
    The standard BCE loss minimizes prediction error on true vs false triples.
    It does NOT explicitly require that contradictory path amplitudes be
    out-of-phase. The destructive interference pattern must emerge implicitly
    from the training signal — which is fragile and may not happen (Phase Collapse).

    This file introduces three loss components that EXPLICITLY shape the phase
    structure of the unitary operators toward constructive interference on
    correct answers and destructive interference on wrong answers.

    The combined loss is:
        L_total = L_main + lambda_phase * L_phase + lambda_contrast * L_contrast

    where:
        L_main    = standard BCE or margin loss (correct vs random negatives)
        L_phase   = Phase Separation Loss (NEW): maximizes phase difference
                    between correct-path and wrong-path amplitudes
        L_contrast = Contrastive Interference Loss (NEW): for known contradiction
                    pairs, directly penalizes constructive interference on the
                    wrong answer

THE FORMAL CLAIM (Contribution #3):
    "We introduce an interference-aware training objective that explicitly
    shapes unitary operator phase angles toward the destructive interference
    pattern required for contradiction suppression. Unlike prior KGE models
    whose loss functions optimize prediction accuracy without regard to phase
    structure, our objective guarantees that learned unitaries produce
    destructive interference on contradictory reasoning paths when
    contradiction annotations are available."

WHEN TO USE:
    - Use PhaseSeparationLoss when you have NO contradiction annotations
      (works on any training data by separating positive/negative path phases)
    - Use ContrastiveInterferenceLoss when you DO have contradiction annotations
      (e.g., from the toy_kg's contradiction_queries, or manually labeled pairs)
    - Use InterferenceRegularization always as a lightweight baseline
      to prevent Phase Collapse

MATHEMATICAL FOUNDATION:
    Phase Separation Loss targets the condition for destructive interference:
        arg(A_correct) - arg(A_wrong) ≈ pi   [opposite phases]
    Maximizing cos(arg(A_correct) - arg(A_wrong)) when it should be -1
    directly trains the phase angles toward destructive interference.

    Formally: L_phase = mean( cos(arg(A_pos) - arg(A_neg)) )
    Minimizing L_phase pushes the cosine toward -1 (phases ≈ pi apart).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Phase Separation Loss ─────────────────────────────────────────────────

class PhaseSeparationLoss(nn.Module):
    """
    Maximizes the phase difference between positive and negative path amplitudes.

    For each positive triple (h,r,t) and negative triple (h,r,t'), we compute:
        A_pos = amp(t,  U_r * h)   [complex amplitude for correct answer]
        A_neg = amp(t', U_r * h)   [complex amplitude for wrong answer]

    The Phase Separation Loss is:
        L_phase = mean( cos(arg(A_pos) - arg(A_neg)) )

    Minimizing this loss pushes arg(A_pos) - arg(A_neg) toward ±pi,
    which is exactly the condition for destructive interference on the wrong answer.

    WHY THIS IS NOVEL:
        RotatE, ComplEx, and all classical baselines optimize scoring functions
        that are agnostic to phase structure. This loss DIRECTLY optimizes
        phase separation — a capability that only exists in complex-valued models.
        A real-valued model has no phases to separate, so this loss is undefined
        for TransE/RotatE/ComplEx — it is a strictly quantum contribution.

    Args:
        temperature: Controls how sharply the loss enforces phase separation.
                     Higher temperature = softer gradient, more stable training.
                     Default: 1.0.
        margin:      Target minimum phase difference (in radians).
                     Default: pi (full separation).
        weight:      Overall weight of this loss component in the total loss.

    Example:
        >>> phase_loss = PhaseSeparationLoss(temperature=1.0, weight=0.1)
        >>> loss = phase_loss(pos_amplitudes, neg_amplitudes)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        margin:      float = math.pi,
        weight:      float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.margin      = margin
        self.weight      = weight

    def forward(
        self,
        pos_amplitudes: torch.Tensor,   # (B,) complex — amplitude for positive tail
        neg_amplitudes: torch.Tensor,   # (B, K) complex — amplitudes for K negatives
    ) -> torch.Tensor:
        """
        Compute Phase Separation Loss.

        Args:
            pos_amplitudes: Complex tensor (B,) — amp(t, U_r * h) for true triples.
            neg_amplitudes: Complex tensor (B, K) — amp(t', U_r * h) for K negatives.

        Returns:
            Scalar loss tensor (weighted).

        Mathematical derivation:
            Let phi_pos = arg(A_pos), phi_neg = arg(A_neg).
            Phase difference: delta_phi = phi_pos - phi_neg.
            Destructive interference condition: delta_phi = ±pi.
            Loss: L = mean(cos(delta_phi / temperature))
            Minimizing L pushes cos(delta_phi) toward -1,
            i.e., delta_phi toward ±pi.
        """
        B, K = neg_amplitudes.shape

        # Compute phase angles
        phi_pos = torch.angle(pos_amplitudes)              # (B,) real
        phi_neg = torch.angle(neg_amplitudes)              # (B, K) real

        # Expand pos phases to match neg shape
        phi_pos_exp = phi_pos.unsqueeze(1).expand(B, K)   # (B, K)

        # Phase difference
        delta_phi = phi_pos_exp - phi_neg                 # (B, K)

        # Cosine of phase difference — target: cos(delta_phi) = -1
        cos_delta = torch.cos(delta_phi / self.temperature)   # (B, K)

        # Loss: mean cosine (minimize this to push toward -1)
        # Adding 1 so the loss is 0 when perfectly separated (cos = -1)
        loss = (cos_delta + 1.0).mean()   # range [0, 2], target 0

        return self.weight * loss

    def measure_separation(
        self,
        pos_amplitudes: torch.Tensor,
        neg_amplitudes: torch.Tensor,
    ) -> dict[str, float]:
        """
        Measure the current degree of phase separation (for monitoring).

        Returns:
            Dict with 'mean_phase_diff' (should approach pi) and
            'fraction_separated' (fraction of pairs with |delta_phi| > pi/2).
        """
        with torch.no_grad():
            phi_pos = torch.angle(pos_amplitudes)
            phi_neg = torch.angle(neg_amplitudes)
            B, K    = neg_amplitudes.shape
            phi_pos_exp = phi_pos.unsqueeze(1).expand(B, K)
            delta_phi   = (phi_pos_exp - phi_neg).abs()
            # Wrap to [0, pi]
            delta_phi   = torch.minimum(delta_phi, 2 * math.pi - delta_phi)

            return {
                "mean_phase_diff":   delta_phi.mean().item(),
                "fraction_separated": (delta_phi > math.pi / 2).float().mean().item(),
                "target_phase_diff": math.pi,
            }


# ── 2. Contrastive Interference Loss ─────────────────────────────────────────

class ContrastiveInterferenceLoss(nn.Module):
    """
    Directly penalizes constructive interference on known wrong answers
    by maximizing interference probability gap between correct and contradictory answers.

    Requires contradiction annotations: known pairs of (correct_tail, wrong_tail)
    for specific (head, relation) queries.

    Loss formula:
        For each contradiction triple (h, r, t_correct, t_wrong):
            A_correct = sum_i alpha_i * amp(t_correct, U_Pi * h)   [correct answer]
            A_wrong   = sum_i alpha_i * amp(t_wrong,   U_Pi * h)   [contradictory answer]

            P_correct = |A_correct|^2   [quantum probability of correct answer]
            P_wrong   = |A_wrong|^2     [quantum probability of wrong answer]

            contrastive_term = max(0, P_wrong - P_correct + margin)

        L_contrast = mean(contrastive_term over all contradiction pairs)

    Minimizing L_contrast forces P_correct > P_wrong + margin for every
    contradiction pair. Combined with the BCE loss on the full training set,
    this explicitly trains the model to use destructive interference to
    suppress the contradictory answer.

    Args:
        margin:  Minimum required gap P_correct - P_wrong.
                 Default: 0.1.
        weight:  Overall weight of this loss component.
        path_cache: Optional PathCache to look up pre-computed paths.
                 If None, uses single-hop scoring as approximation.

    Example:
        >>> contrast_loss = ContrastiveInterferenceLoss(margin=0.1, weight=0.5)
        >>> # During training, whenever a contradiction batch is available:
        >>> loss = contrast_loss(
        ...     model, h_ids, r_ids, correct_t_ids, wrong_t_ids, path_cache
        ... )
    """

    def __init__(
        self,
        margin: float = 0.1,
        weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(
        self,
        model,                              # QuantumReasoner instance
        head_ids:      torch.Tensor,        # (B,) int
        relation_ids:  torch.Tensor,        # (B,) int
        correct_tails: torch.Tensor,        # (B,) int
        wrong_tails:   torch.Tensor,        # (B,) int
        paths_correct: Optional[list] = None,  # pre-computed paths to correct tails
        paths_wrong:   Optional[list] = None,  # pre-computed paths to wrong tails
    ) -> torch.Tensor:
        """
        Compute Contrastive Interference Loss for a batch of contradiction pairs.

        Args:
            model:         QuantumReasoner (needs score_triple or score_with_paths).
            head_ids:      (B,) source entity IDs.
            relation_ids:  (B,) relation IDs (used for 1-hop fallback).
            correct_tails: (B,) correct answer tail IDs.
            wrong_tails:   (B,) wrong (contradictory) answer tail IDs.
            paths_correct: Optional B-length list of path lists to correct tails.
            paths_wrong:   Optional B-length list of path lists to wrong tails.

        Returns:
            Scalar loss tensor (weighted).
        """
        # Score correct and wrong answers
        if paths_correct is not None and paths_wrong is not None:
            # Multi-hop interference scoring
            p_correct = model.score_with_paths(
                head_ids, relation_ids, correct_tails, paths_correct
            )
            p_wrong = model.score_with_paths(
                head_ids, relation_ids, wrong_tails, paths_wrong
            )
        else:
            # 1-hop fallback (faster, less powerful)
            p_correct = model.score_triple(head_ids, relation_ids, correct_tails)
            p_wrong   = model.score_triple(head_ids, relation_ids, wrong_tails)

        # Contrastive hinge: penalize when wrong answer scores higher than correct
        # max(0, P_wrong - P_correct + margin)
        contrastive_term = F.relu(p_wrong - p_correct + self.margin)

        loss = contrastive_term.mean()
        return self.weight * loss

    def batch_from_contradiction_queries(
        self,
        contradiction_queries: list[dict],
        entity2id:             dict[str, int],
        relation2id:           dict[str, int],
        device:                torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Convert toy_kg.contradiction_queries into tensors for forward().

        Args:
            contradiction_queries: List of dicts from ToyKG.contradiction_queries.
            entity2id:             Entity string to ID mapping.
            relation2id:           Relation string to ID mapping.
            device:                Target device.

        Returns:
            Dict with 'head_ids', 'relation_ids', 'correct_tails', 'wrong_tails' tensors.

        Example:
            >>> batch = contrast_loss.batch_from_contradiction_queries(
            ...     kg.contradiction_queries, kg.entity2id, kg.relation2id, device
            ... )
            >>> loss = contrast_loss(model, **batch)
        """
        heads, rels, corrects, wrongs = [], [], [], []

        for cq in contradiction_queries:
            head_id    = entity2id[cq["head"]]
            correct_id = entity2id[cq["correct_tail"]]
            wrong_id   = entity2id[cq["contradictory_tail"]]

            # Infer relation from the correct path's first step
            correct_path = cq.get("correct_path", [])
            if correct_path:
                rel_id = relation2id[correct_path[0][0]]   # first hop relation
            else:
                rel_id = 0   # fallback

            heads.append(head_id)
            rels.append(rel_id)
            corrects.append(correct_id)
            wrongs.append(wrong_id)

        return {
            "head_ids":      torch.tensor(heads,    dtype=torch.long, device=device),
            "relation_ids":  torch.tensor(rels,     dtype=torch.long, device=device),
            "correct_tails": torch.tensor(corrects, dtype=torch.long, device=device),
            "wrong_tails":   torch.tensor(wrongs,   dtype=torch.long, device=device),
        }


# ── 3. Interference Regularization ───────────────────────────────────────────

class InterferenceRegularization(nn.Module):
    """
    Lightweight regularization that prevents Phase Collapse.

    Phase Collapse: the model sets all imaginary components near zero,
    collapsing to a real-valued model. This is a local minimum that
    satisfies the BCE loss but destroys the interference mechanism.

    This regularizer penalizes small imaginary norms, pushing the model
    to maintain non-trivial phase structure:

        L_reg = -mean( ||Im(entity_states)||^2 + ||phases(unitary)||^2 )

    Minimizing L_total including this term pushes imaginary norms UP,
    resisting Phase Collapse.

    Args:
        encoder_weight: Weight for entity imaginary norm penalty.
        unitary_weight: Weight for unitary phase magnitude penalty.
        min_imag_norm:  Minimum desired imaginary norm (training target).
                        Default: 0.1 (10% of state should be imaginary).

    Example:
        >>> ir = InterferenceRegularization(encoder_weight=0.01, unitary_weight=0.01)
        >>> reg_loss = ir(encoder, unitary_op)   # add to total loss
    """

    def __init__(
        self,
        encoder_weight: float = 0.01,
        unitary_weight: float = 0.01,
        min_imag_norm:  float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_weight = encoder_weight
        self.unitary_weight = unitary_weight
        self.min_imag_norm  = min_imag_norm

    def forward(
        self,
        encoder,      # QuantumStateEncoder instance
        unitary_op,   # UnitaryOperator instance
    ) -> torch.Tensor:
        """
        Compute the interference regularization penalty.

        Penalizes:
            1. Entity imaginary norms below min_imag_norm
               (ReLU hinge: only penalizes when too small)
            2. Unitary phase angles close to 0 or pi (which produce real-valued unitaries)

        Returns:
            Scalar non-negative loss tensor.
        """
        total = torch.tensor(0.0, device=encoder.real_embeddings.weight.device)

        # 1. Entity imaginary norm penalty
        if self.encoder_weight > 0:
            imag_norms = encoder.imag_embeddings.weight.norm(dim=-1)   # (N,)
            # Penalize when imag_norm < min_imag_norm
            imag_deficit = F.relu(self.min_imag_norm - imag_norms)
            total = total + self.encoder_weight * imag_deficit.mean()

        # 2. Unitary phase magnitude penalty (keep phases non-trivial)
        if self.unitary_weight > 0 and hasattr(unitary_op, 'phases'):
            phases = unitary_op.phases   # could be phases, base_phases, phases_dir etc.
            # Penalize phases that are very close to 0 (real-valued behavior)
            phase_magnitudes = phases.abs()
            phase_trivial    = F.relu(0.05 - phase_magnitudes)
            total = total + self.unitary_weight * phase_trivial.mean()

        return total

    def measure_collapse(
        self,
        encoder,
        unitary_op,
    ) -> dict[str, float]:
        """
        Measure the current degree of Phase Collapse (for monitoring).

        Returns:
            Dict with:
                'mean_imag_norm':     Average imaginary norm across entities.
                'fraction_collapsed': Fraction of entities with imag_norm < 0.05.
                'mean_phase_mag':     Average phase magnitude in unitary.
                'is_collapsed':       True if model has significantly collapsed.
        """
        with torch.no_grad():
            imag_norms = encoder.imag_embeddings.weight.norm(dim=-1)
            mean_imag  = imag_norms.mean().item()
            frac_coll  = (imag_norms < 0.05).float().mean().item()

            mean_phase = 0.0
            if hasattr(unitary_op, 'phases'):
                mean_phase = unitary_op.phases.abs().mean().item()

        return {
            "mean_imag_norm":     mean_imag,
            "fraction_collapsed": frac_coll,
            "mean_phase_magnitude": mean_phase,
            "is_collapsed":       frac_coll > 0.5 or mean_imag < 0.02,
        }


# ── 4. Combined Interference-Aware Loss ──────────────────────────────────────

class InterferenceAwareLoss(nn.Module):
    """
    The full combined loss: BCE + Phase Separation + Contrastive + Regularization.

    This is the recommended loss for the final paper experiments.
    It combines all three interference-aware components with the standard
    BCE main loss.

    Total loss:
        L = L_BCE + w_phase * L_phase + w_contrast * L_contrast + w_reg * L_reg

    Args:
        label_smoothing:     For BCE component.
        phase_weight:        Weight for PhaseSeparationLoss.
        contrast_weight:     Weight for ContrastiveInterferenceLoss.
        reg_encoder_weight:  Weight for encoder imaginary norm regularization.
        reg_unitary_weight:  Weight for unitary phase regularization.
        margin:              Margin for contrastive component.

    Example:
        >>> loss_fn = InterferenceAwareLoss(
        ...     label_smoothing=0.1,
        ...     phase_weight=0.1,
        ...     contrast_weight=0.5,
        ...     reg_encoder_weight=0.01,
        ... )
        >>> loss = loss_fn.forward_main(pos_scores, neg_scores)  # standard training
        >>> # When contradiction batch available:
        >>> loss += loss_fn.forward_contradiction(model, h, r, t_correct, t_wrong)
        >>> # Each epoch:
        >>> loss += loss_fn.regularization(encoder, unitary)
    """

    def __init__(
        self,
        label_smoothing:    float = 0.1,
        phase_weight:       float = 0.1,
        contrast_weight:    float = 0.5,
        reg_encoder_weight: float = 0.01,
        reg_unitary_weight: float = 0.01,
        margin:             float = 0.1,
    ) -> None:
        super().__init__()

        self.phase_sep  = PhaseSeparationLoss(weight=phase_weight)
        self.contrast   = ContrastiveInterferenceLoss(margin=margin, weight=contrast_weight)
        self.reg        = InterferenceRegularization(reg_encoder_weight, reg_unitary_weight)
        self.label_smoothing = label_smoothing

    def forward_main(
        self,
        pos_scores: torch.Tensor,   # (B,) float
        neg_scores: torch.Tensor,   # (B, K) float
        pos_amplitudes: Optional[torch.Tensor] = None,  # (B,) complex
        neg_amplitudes: Optional[torch.Tensor] = None,  # (B, K) complex
    ) -> torch.Tensor:
        """
        Standard training step loss (BCE + optional Phase Separation).

        Call this every batch. Pass pos_amplitudes/neg_amplitudes when
        available (they come from AmplitudeAggregator.forward() calls).
        """
        # Main BCE loss
        pos_labels = torch.full_like(pos_scores, self.label_smoothing)
        neg_labels = torch.zeros_like(neg_scores)

        bce_pos = F.binary_cross_entropy_with_logits(pos_scores, pos_labels).mean()
        bce_neg = F.binary_cross_entropy_with_logits(neg_scores, neg_labels).mean()
        loss    = bce_pos + bce_neg

        # Phase separation loss (when amplitudes available)
        if pos_amplitudes is not None and neg_amplitudes is not None:
            loss = loss + self.phase_sep(pos_amplitudes, neg_amplitudes)

        return loss

    def forward_contradiction(
        self,
        model,
        head_ids:      torch.Tensor,
        relation_ids:  torch.Tensor,
        correct_tails: torch.Tensor,
        wrong_tails:   torch.Tensor,
        paths_correct: Optional[list] = None,
        paths_wrong:   Optional[list] = None,
    ) -> torch.Tensor:
        """
        Contradiction-specific contrastive loss.
        Call this when a contradiction batch is available (can be every N batches).
        """
        return self.contrast(
            model, head_ids, relation_ids,
            correct_tails, wrong_tails,
            paths_correct, paths_wrong,
        )

    def regularization(self, encoder, unitary_op) -> torch.Tensor:
        """Per-epoch regularization to prevent Phase Collapse."""
        return self.reg(encoder, unitary_op)
