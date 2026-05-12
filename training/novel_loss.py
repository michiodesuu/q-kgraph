"""
training/novel_loss.py — Novel Loss Components for Quantum KG Reasoning.

PURPOSE
-------
Defines five novel training objectives that work alongside the existing
``InterferenceAwareLoss`` (training/interference_loss.py) to shape the
quantum knowledge graph model toward physically meaningful representations:

1. ``RankingAwareLoss``               — ListNet-style ranking over all candidate tails.
2. ``TeleportationContrastiveLoss``   — Hinge + entropy penalty for teleportation scores.
3. ``EntanglementEntropyRegularizer`` — Push per-relation entanglement toward targets.
4. ``ContextualityLoss``              — Penalise classical factorizability of 2-hop scores.
5. ``DecoherenceLoss``                — Shape decoherence rates by path length / diversity.
6. ``CombinedNovelLoss``              — Orchestrates all five components.

MOTIVATION FOR EACH COMPONENT
------------------------------

RankingAwareLoss
    Standard BCE treats the scoring problem as B independent binary
    classifications. A ranking-aware loss treats it as a distribution over
    candidate tails (ListNet, Cao et al. 2007). This allows the model to
    jointly shape the RELATIVE ordering of candidates, not just push
    individual scores above/below a threshold, directly optimising MRR and
    Hits@K rather than binary accuracy.

TeleportationContrastiveLoss
    The TeleportationScorer introduces Bell-measurement attention weights q_k.
    If these weights are spread uniformly over outcomes (high entropy), the
    model is "uncertain" about which correction branch to use. For correct
    triples, low-entropy (peaked) weights imply the model can identify which
    correction operator leads to constructive interference with the true tail.
    This loss combines a standard hinge with an entropy penalty that
    encourages certainty for correct triples.

EntanglementEntropyRegularizer
    The Von Neumann entanglement entropy S(ρ_A) of a relation's Bell state
    matrix measures relational complexity. Symmetric and inverse relations
    should have LOW entropy (near-classical correlation); compositional /
    many-to-many relations should have HIGH entropy (rich entanglement).
    Without guidance the matrices may converge to trivial solutions.

ContextualityLoss
    A classical model can factorise any multi-hop score into a product of
    single-hop scores. A genuinely quantum model exhibits contextuality —
    its 2-hop scores cannot be factorised. This loss penalises the model when
    its joint scores are too close to the classical factorised scores, pushing
    it toward non-classical (contextual) behaviour.

DecoherenceLoss
    Decoherence rates should (a) remain low for direct/short paths and (b)
    vary across relations to avoid degenerate solutions where all rates
    converge to the same value. This loss enforces both physical priors.

USAGE
-----
    combined = CombinedNovelLoss(complex_dim=8, num_relations=11)
    rank_loss = combined.forward_ranking(pos_scores, neg_scores)
    reg_loss  = combined.forward_entropy_reg(bell_state_relation)
    ctx_loss  = combined.forward_contextuality(joint_scores, factorized_scores)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. RankingAwareLoss
# ---------------------------------------------------------------------------

class RankingAwareLoss(nn.Module):
    """
    ListNet-style ranking loss that optimises the full ranking distribution
    over candidate tails rather than treating each triple independently.

    For each query (h, r), given scores over K+1 tails [t_pos, t_neg_1, …,
    t_neg_{K-1}]:

        P_true[i]  = one-hot on the correct tail (label-smoothed)
        P_pred[i]  = softmax(scores / T)[i]
        L          = −Σ_i P_true[i] · log P_pred[i]

    This cross-entropy between the soft one-hot target and the predicted
    softmax distribution directly optimises MRR / Hits@K rather than binary
    accuracy.

    Temperature *T* controls the sharpness of the predicted distribution:
        T → 0 : score ranking becomes a hard argmax (high variance gradients).
        T → ∞ : predicted distribution becomes uniform (loss → log(K+1)).

    Label smoothing prevents overconfident targets:
        P_true = (1 − ε) · one_hot + ε / (K+1)

    Args:
        temperature    (float): Softmax temperature T > 0. Default 1.0.
        label_smoothing(float): Smoothing factor ε ∈ [0, 1). Default 0.1.
        weight         (float): Scalar multiplier for this loss. Default 1.0.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        label_smoothing: float = 0.1,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ListNet ranking loss.

        The correct tail is placed at column 0 (first position), and the K
        negatives follow. Label smoothing is applied before computing the
        cross-entropy.

        Args:
            pos_scores: (B,) float32 — scores for the correct tail of each
                        query. B is the batch size.
            neg_scores: (B, K) float32 — scores for K negative tails paired
                        with each query.

        Returns:
            Scalar float32 tensor — ``weight`` × mean cross-entropy loss.

        Steps
        -----
        1. all_scores = cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
                      → (B, K+1)
        2. log_probs  = log_softmax(all_scores / T, dim=1)
        3. targets    = zeros(B, K+1); targets[:, 0] = 1.0
        4. Label smooth: targets = (1 − ε)·targets + ε/(K+1)
        5. loss       = −(targets * log_probs).sum(dim=1).mean()
        6. return weight * loss
        """
        B, K = neg_scores.shape

        # Step 1: concatenate — correct tail is always at index 0
        all_scores = torch.cat(
            [pos_scores.unsqueeze(1), neg_scores], dim=1
        )  # (B, K+1)

        # Step 2: temperature-scaled log-softmax
        log_probs = F.log_softmax(all_scores / self.temperature, dim=1)  # (B, K+1)

        # Step 3: one-hot targets — correct tail is column 0
        targets = torch.zeros(B, K + 1, dtype=all_scores.dtype, device=all_scores.device)
        targets[:, 0] = 1.0

        # Step 4: label smoothing
        targets = (1.0 - self.label_smoothing) * targets + self.label_smoothing / (K + 1)

        # Step 5: cross-entropy
        loss = -(targets * log_probs).sum(dim=1).mean()

        # Step 6: scale by weight
        return self.weight * loss

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, "
            f"label_smoothing={self.label_smoothing}, "
            f"weight={self.weight}"
        )


# ---------------------------------------------------------------------------
# 2. TeleportationContrastiveLoss
# ---------------------------------------------------------------------------

class TeleportationContrastiveLoss(nn.Module):
    """
    Contrastive loss specific to the quantum teleportation scorer.

    Combines two complementary objectives:

    (a) **Hinge loss**: The correct triple must score strictly above the
        hardest negative triple by at least *margin*::

            L_hinge = mean(relu(score_neg − score_pos + margin))

    (b) **Bell-outcome entropy penalty** (optional): The Bell measurement
        attention weights q_k output by the teleportation scorer should be
        *concentrated* (peaked, low entropy) for correct triples. High
        entropy implies the model cannot identify which correction operator
        leads to constructive interference with the true tail::

            H(q)          = −Σ_k q_k · log(q_k + ε)
            L_entropy     = entropy_penalty_weight · mean_b H(q_b)

        We penalise high entropy to encourage "certain" Bell outcomes on
        correct answers.

    Total::

        L = weight · (L_hinge + L_entropy)

    Physics motivation:
        In quantum teleportation, Alice's Bell measurement collapses the
        shared EPR pair into one of four orthogonal outcomes. If the model
        has learned the correct relation structure, the measurement outcome
        should be predictable (low entropy) for correct triples. Uncertainty
        (high entropy) indicates the model has not learned to route quantum
        information efficiently.

    Args:
        margin               (float): Minimum required score gap. Default 0.1.
        entropy_penalty_weight (float): Weight on the entropy penalty.
                               Default 0.05. Set to 0 to disable.
        weight               (float): Overall scalar multiplier. Default 1.0.
    """

    def __init__(
        self,
        margin: float = 0.1,
        entropy_penalty_weight: float = 0.05,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.entropy_penalty_weight = entropy_penalty_weight
        self.weight = weight

    def forward(
        self,
        pos_teleport_scores: torch.Tensor,
        neg_teleport_scores: torch.Tensor,
        pos_outcome_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the teleportation-specific contrastive loss.

        Args:
            pos_teleport_scores: (B,) float32 — TeleportationScorer scores
                                 for the correct (positive) triples.
            neg_teleport_scores: (B,) float32 — TeleportationScorer scores
                                 for the hardest negative tails, paired 1-to-1
                                 with ``pos_teleport_scores``.
            pos_outcome_weights: (B, K) float32 — Bell measurement attention
                                 weights q_k for the *correct* triples. These
                                 should already be normalised (sum to 1 along
                                 the K dimension). If None, the entropy penalty
                                 is skipped entirely.

        Returns:
            Scalar float32 tensor — ``weight`` × (hinge + entropy_penalty).
        """
        # (a) Hinge: penalise when neg score ≥ pos score − margin
        hinge = F.relu(neg_teleport_scores - pos_teleport_scores + self.margin).mean()

        entropy_penalty = torch.tensor(0.0, device=pos_teleport_scores.device)

        # (b) Entropy penalty on Bell measurement outcome weights for correct triples
        if pos_outcome_weights is not None and self.entropy_penalty_weight > 0.0:
            # Shannon entropy H(q) = −Σ_k q_k · log(q_k + ε)
            # q_k ∈ (0, 1) and sum to 1 along dim=-1
            entropy = -(
                pos_outcome_weights * (pos_outcome_weights + 1e-10).log()
            ).sum(dim=-1).mean()
            # entropy ≥ 0; we penalise HIGH entropy (uncertainty is bad for correct triples)
            entropy_penalty = self.entropy_penalty_weight * entropy

        return self.weight * (hinge + entropy_penalty)

    def extra_repr(self) -> str:
        return (
            f"margin={self.margin}, "
            f"entropy_penalty_weight={self.entropy_penalty_weight}, "
            f"weight={self.weight}"
        )


# ---------------------------------------------------------------------------
# 3. EntanglementEntropyRegularizer
# ---------------------------------------------------------------------------

class EntanglementEntropyRegularizer(nn.Module):
    """
    Regularises the Von Neumann entanglement entropy of each relation's Bell
    state matrix toward physics-motivated targets.

    Physical targets
    ----------------
    Relations have different intrinsic complexities:

    * **Symmetric** relations (e.g., "similarTo", "isSiblingOf"):
      These represent well-defined, bijective mappings.
      Target: low entropy — S_target ≈ 0.2 · log(d).

    * **Inverse-pair** relations (e.g., "isParentOf" ↔ "isChildOf"):
      Clean invertible mappings. Target: very low entropy — S_target ≈ 0.1 · log(d).

    * **General** relations:
      Many-to-many, ambiguous. Default target: 0.5 · log(d).

    Without regularisation the Bell state matrices may converge to trivial
    solutions (all-diagonal ≡ classical, or maximally mixed ≡ uninformative).
    This regulariser gently pulls entropies toward the known structural targets
    using differentiable proxy values from ``.entanglement_entropy_all()``.

    Implementation
    --------------
    The targets and per-relation weights are stored as non-learnable
    ``nn.Parameter`` buffers (``requires_grad=False``), allowing them to be
    moved across devices with ``.to(device)`` automatically::

        entropy_targets  : (num_relations,) — target entropy for each relation
        target_weights   : (num_relations,) — per-relation regularisation strength

    The forward pass computes::

        diff        = |entropy_r − target_r|            for each r
        loss        = weight · mean_r (target_weight_r · diff_r)

    Args:
        num_relations (int):   Number of distinct KG relation types.
        complex_dim   (int):   Hilbert space dimension d. Used to compute
                               the maximum entropy log(d) as a scale.
        weight        (float): Scalar multiplier for the regularisation term.
                               Default 0.01.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim: int,
        weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.complex_dim = complex_dim
        self.weight = weight

        max_entropy = math.log(complex_dim) if complex_dim > 1 else 1.0

        # Default target: half of the maximum possible entropy log(d)
        self.entropy_targets = nn.Parameter(
            torch.full((num_relations,), max_entropy * 0.5),
            requires_grad=False,
        )
        # Default per-relation regularisation strength: 0.1 (soft)
        self.target_weights = nn.Parameter(
            torch.ones(num_relations) * 0.1,
            requires_grad=False,
        )

        self._max_entropy = max_entropy

    def set_structural_targets(
        self,
        symmetric_ids: list[int],
        inverse_ids: list[int],
        max_entropy: float,
    ) -> None:
        """
        Override entropy targets and regularisation weights for known relation types.

        After calling this method the specified relation IDs will be pulled
        strongly toward their physics-motivated entropy targets during training.

        Args:
            symmetric_ids: Indices of symmetric relations (e.g., "isSiblingOf").
                           Their entropy target is set to 0.2 · max_entropy (LOW),
                           encouraging near-classical, bijective structure.
            inverse_ids:   Indices of inverse-pair relations (e.g., "isChildOf").
                           Their entropy target is set to 0.1 · max_entropy
                           (VERY LOW), encouraging nearly classical correlations.
            max_entropy:   The maximum possible Von Neumann entropy for this
                           Hilbert space dimension, i.e., log(d). Targets are
                           expressed as fractions of this value.
        """
        with torch.no_grad():
            for rid in symmetric_ids:
                if 0 <= rid < self.num_relations:
                    self.entropy_targets[rid] = 0.2 * max_entropy
                    self.target_weights[rid] = 1.0   # enforce strongly

            for rid in inverse_ids:
                if 0 <= rid < self.num_relations:
                    self.entropy_targets[rid] = 0.1 * max_entropy
                    self.target_weights[rid] = 1.0   # enforce strongly

    def forward(
        self,
        bell_state_relation: object,
    ) -> torch.Tensor:
        """
        Compute the entanglement entropy regularisation loss.

        Uses duck typing: ``bell_state_relation`` must expose
        ``entanglement_entropy_all()`` returning a ``(num_relations,)``
        float32 tensor of Von Neumann entropies, one per relation.

        Args:
            bell_state_relation: Any object with an ``entanglement_entropy_all()``
                                 method returning ``(R,)`` float32 — Von Neumann
                                 entanglement entropy for each relation's Bell
                                 state matrix.

        Returns:
            Scalar float32 tensor — ``weight`` × weighted mean absolute deviation
            from entropy targets.
        """
        # (R,) float32 — Von Neumann entanglement entropy for each relation
        entropies: torch.Tensor = bell_state_relation.entanglement_entropy_all()

        # Absolute deviation from target (move parameter tensors to the same device)
        diff = (entropies - self.entropy_targets.to(entropies.device)).abs()  # (R,)

        # Weighted mean deviation
        weighted_diff = (self.target_weights.to(diff.device) * diff).mean()

        return self.weight * weighted_diff

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"complex_dim={self.complex_dim}, "
            f"max_entropy={self._max_entropy:.4f}, "
            f"weight={self.weight}"
        )


# ---------------------------------------------------------------------------
# 4. ContextualityLoss
# ---------------------------------------------------------------------------

class ContextualityLoss(nn.Module):
    """
    Penalises *classical factorizability* of the model's 2-hop scores.

    Background
    ----------
    In quantum mechanics, *contextuality* means that the joint probability of
    outcomes cannot be written as a product of marginal probabilities. Applied
    to KG reasoning: the score of a 2-hop path (h, r₁∘r₂, t) should *not*
    factorise into the product of two 1-hop scores::

        Score(h, r₁∘r₂, t)  ≠  Score(h, r₁, e) · Score(e, r₂, t)

    If the model IS classically factorizable it is not leveraging the
    non-classical structure of the Hilbert space. This loss penalises the model
    when the gap between its joint score and the factorised classical score is
    too small, rewarding genuine quantum interference effects.

    Loss formula::

        gap   = |joint_score − factorised_score|
        L     = weight · mean(relu(margin − gap))

    Zero penalty when |gap| ≥ margin; positive otherwise.

    Note: this does **not** require the joint score to be *higher* than the
    factorised score — only that they *differ* by at least *margin*.

    Args:
        margin (float): Minimum required absolute gap. Default 0.05.
        weight (float): Scalar multiplier. Default 0.1.
    """

    def __init__(
        self,
        margin: float = 0.05,
        weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(
        self,
        joint_scores: torch.Tensor,
        factorized_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contextuality (non-classicality) penalty.

        Args:
            joint_scores:      (B,) float32 — Score(h, r₁∘r₂, t) produced by
                               the TeleportationScorer with EntanglementSwap path
                               composition. These reflect genuine quantum
                               correlations across hops.
            factorized_scores: (B,) float32 — classical factorised scores
                               Score(h, r₁, e) × Score(e, r₂, t). Represents
                               the best-case classical approximation.

        Returns:
            Scalar float32 tensor — ``weight`` × mean contextuality penalty.
        """
        gap = (joint_scores - factorized_scores).abs()  # (B,) float32
        loss = F.relu(self.margin - gap).mean()
        return self.weight * loss

    @staticmethod
    def compute_factorized_scores(
        h_states: torch.Tensor,
        e_states: torch.Tensor,
        t_states: torch.Tensor,
        r1_ids: torch.Tensor,
        r2_ids: torch.Tensor,
        unitary: object,
    ) -> torch.Tensor:
        """
        Compute the classical factorised 2-hop score using Born-rule inner
        products after DiagonalUnitary application.

        The factorised score is::

            Score(h, r₁, e) × Score(e, r₂, t)

        where each single-hop score is the squared magnitude of the inner
        product between the evolved head (or intermediate) state and the
        target state::

            Score(h, r, t) = |⟨t | U_r | h⟩|²
                           = |(conj(U_r |h⟩) · t).sum()|²

        (Using duck typing: ``unitary.apply(states, rel_ids)`` returns the
        evolved complex state vectors.)

        Args:
            h_states: (B, d) complex64 — unit-norm head entity states.
            e_states: (B, d) complex64 — unit-norm intermediate entity states.
            t_states: (B, d) complex64 — unit-norm tail entity states.
            r1_ids:   (B,) int64 — relation IDs for the first hop.
            r2_ids:   (B,) int64 — relation IDs for the second hop.
            unitary:  Any object exposing ``apply(states, rel_ids) → (B, d)``
                      complex64. Typically a DiagonalUnitary instance.

        Returns:
            (B,) float32 — factorised 2-hop scores in [0, 1].
        """
        # Step 1: evolve head states through relation r1
        h_evolved_r1: torch.Tensor = unitary.apply(h_states, r1_ids)   # (B, d) complex

        # Step 2: Born rule — squared magnitude of inner product ⟨e | U_r1 | h⟩
        score_hop1 = (h_evolved_r1.conj() * e_states).sum(dim=-1).abs().pow(2)   # (B,) float

        # Step 3: evolve intermediate states through relation r2
        e_evolved_r2: torch.Tensor = unitary.apply(e_states, r2_ids)   # (B, d) complex

        # Step 4: Born rule — squared magnitude of inner product ⟨t | U_r2 | e⟩
        score_hop2 = (e_evolved_r2.conj() * t_states).sum(dim=-1).abs().pow(2)   # (B,) float

        # Step 5: product of marginal scores = classical factorised score
        return score_hop1 * score_hop2   # (B,) float

    def extra_repr(self) -> str:
        return f"margin={self.margin}, weight={self.weight}"


# ---------------------------------------------------------------------------
# 5. DecoherenceLoss
# ---------------------------------------------------------------------------

class DecoherenceLoss(nn.Module):
    """
    Two-component loss that makes decoherence rates physically meaningful.

    Physical priors
    ---------------
    (A) **Low rates for short/direct paths**: Short reasoning paths (direct
        links) should maintain high coherence. Rates that exceed
        *max_direct_rate* are penalised::

            L_low = relu(ε_r − max_direct_rate).mean()

    (B) **Diversity across relations**: If all rates converge to the same
        value (trivial ε ≡ 0 or ε ≡ 1 for all relations), the decoherence
        channel conveys no structural information. We incentivise diversity
        by penalising low standard deviation of rates::

            L_diversity = −std(ε)     (negative std → maximise std)

    Total::

        L = weight · (low_rate_weight · L_low + diversity_weight · L_diversity)

    Physics motivation:
        Decoherence accumulates over reasoning steps. A model that learns
        calibrated per-relation decoherence rates can represent the
        reliability of different reasoning chains. Without this loss, gradient
        pressure from prediction objectives may collapse all rates to a single
        trivial value, destroying the decoherence channel's interpretability.

    Args:
        max_direct_rate  (float): Maximum tolerated ε for direct/short paths.
                                  Default 0.1.
        diversity_weight (float): Weight on the diversity penalty. Default 0.01.
        low_rate_weight  (float): Weight on the high-rate penalty for short
                                  paths. Default 0.1.
        weight           (float): Overall scalar multiplier. Default 1.0.
    """

    def __init__(
        self,
        max_direct_rate: float = 0.1,
        diversity_weight: float = 0.01,
        low_rate_weight: float = 0.1,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_direct_rate = max_direct_rate
        self.diversity_weight = diversity_weight
        self.low_rate_weight = low_rate_weight
        self.weight = weight

    def forward(
        self,
        decoherence_channel: object,
        path_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the decoherence rate shaping loss.

        Args:
            decoherence_channel: Any object with a ``rates`` property returning
                                 ``(num_relations,)`` float32 ∈ (0, 1) —
                                 per-relation decoherence rates (differentiable,
                                 typically via sigmoid). Duck-typed: the actual
                                 DecoherenceChannel class is not imported here.
            path_lengths:        (B,) int64 — hop counts for the current batch.
                                 If provided, applies a higher penalty to short
                                 paths (length == 1) with high rates; currently
                                 used as a proxy hint (mean-based approach is
                                 used when path_lengths is None).

        Returns:
            Scalar float32 tensor — combined decoherence regularisation loss.
        """
        rates: torch.Tensor = decoherence_channel.rates  # (R,) float32 ∈ (0, 1)

        # ── (A) Penalty for rates above max_direct_rate ──────────────────────
        # Use the full rates tensor as a proxy (mean across all relations).
        # If path_lengths is provided, we could weight by short paths, but the
        # mean-based approach is a valid conservative bound.
        direct_penalty = F.relu(rates - self.max_direct_rate).mean()

        # ── (B) Diversity: maximise std of rates across relations ─────────────
        # Penalise low std (all rates converging to the same value is bad).
        # torch.std uses Bessel's correction by default; need at least 2 elements.
        if rates.numel() > 1:
            diversity_penalty = -rates.std()   # negative → minimising this maximises std
        else:
            diversity_penalty = torch.tensor(0.0, device=rates.device)

        total = self.weight * (
            self.low_rate_weight * direct_penalty
            + self.diversity_weight * diversity_penalty
        )
        return total

    def extra_repr(self) -> str:
        return (
            f"max_direct_rate={self.max_direct_rate}, "
            f"diversity_weight={self.diversity_weight}, "
            f"low_rate_weight={self.low_rate_weight}, "
            f"weight={self.weight}"
        )


# ---------------------------------------------------------------------------
# 6. CombinedNovelLoss
# ---------------------------------------------------------------------------

class CombinedNovelLoss(nn.Module):
    """
    Orchestrates all novel loss terms together with existing interference losses.

    This module owns one instance of each novel loss component and exposes
    both individual ``forward_*`` methods (for selective use in a training loop)
    and a ``get_loss_weights`` method for logging.

    Designed to complement ``InterferenceAwareLoss`` from
    ``training/interference_loss.py``, which handles BCE, phase separation, and
    contrastive interference. This module adds the ranking, teleportation,
    entropy, contextuality, and decoherence objectives.

    Combined loss::

        L = L_bce
          + λ_rank       · L_rank
          + λ_teleport   · L_teleport_contrast
          + λ_entropy    · L_entropy_reg
          + λ_context    · L_context
          + λ_decohere   · L_decohere

    Args:
        complex_dim              (int):   Hilbert space dimension d.
        num_relations            (int):   Number of distinct KG relation types.
        ranking_weight           (float): Weight for RankingAwareLoss. Default 0.3.
        teleport_contrast_weight (float): Weight for TeleportationContrastiveLoss.
                                          Default 0.5.
        entropy_reg_weight       (float): Weight for EntanglementEntropyRegularizer.
                                          Default 0.01.
        contextuality_weight     (float): Weight for ContextualityLoss. Default 0.1.
        decoherence_weight       (float): Weight for DecoherenceLoss. Default 0.05.
        ranking_temperature      (float): Temperature for RankingAwareLoss. Default 1.0.
        teleport_margin          (float): Margin for TeleportationContrastiveLoss.
                                          Default 0.1.
        contextuality_margin     (float): Margin for ContextualityLoss. Default 0.05.

    Submodules:
        ranking_loss      : RankingAwareLoss
        teleport_contrast : TeleportationContrastiveLoss
        entropy_reg       : EntanglementEntropyRegularizer
        contextuality     : ContextualityLoss
        decoherence       : DecoherenceLoss
    """

    def __init__(
        self,
        complex_dim: int,
        num_relations: int,
        ranking_weight: float = 0.3,
        teleport_contrast_weight: float = 0.5,
        entropy_reg_weight: float = 0.01,
        contextuality_weight: float = 0.1,
        decoherence_weight: float = 0.05,
        ranking_temperature: float = 1.0,
        teleport_margin: float = 0.1,
        contextuality_margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.complex_dim = complex_dim
        self.num_relations = num_relations

        self.ranking_loss = RankingAwareLoss(
            temperature=ranking_temperature,
            label_smoothing=0.1,
            weight=ranking_weight,
        )
        self.teleport_contrast = TeleportationContrastiveLoss(
            margin=teleport_margin,
            entropy_penalty_weight=0.05,
            weight=teleport_contrast_weight,
        )
        self.entropy_reg = EntanglementEntropyRegularizer(
            num_relations=num_relations,
            complex_dim=complex_dim,
            weight=entropy_reg_weight,
        )
        self.contextuality = ContextualityLoss(
            margin=contextuality_margin,
            weight=contextuality_weight,
        )
        self.decoherence = DecoherenceLoss(
            max_direct_rate=0.1,
            diversity_weight=0.01,
            low_rate_weight=0.1,
            weight=decoherence_weight,
        )

    # ── Individual forward methods ─────────────────────────────────────────────

    def forward_ranking(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ListNet ranking loss.

        Args:
            pos_scores: (B,) float32 — scores for the correct tails.
            neg_scores: (B, K) float32 — scores for K negative tails.

        Returns:
            Scalar float32 tensor.
        """
        return self.ranking_loss(pos_scores, neg_scores)

    def forward_teleportation_contrast(
        self,
        pos_tp_scores: torch.Tensor,
        neg_tp_scores: torch.Tensor,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the teleportation contrastive loss (hinge + entropy penalty).

        Args:
            pos_tp_scores: (B,) float32 — teleportation scores for correct triples.
            neg_tp_scores: (B,) float32 — teleportation scores for hardest negatives.
            pos_weights:   (B, K) float32 — Bell measurement outcome attention weights
                           for correct triples (optional). If provided, the entropy
                           penalty is computed.

        Returns:
            Scalar float32 tensor.
        """
        return self.teleport_contrast(pos_tp_scores, neg_tp_scores, pos_weights)

    def forward_entropy_reg(
        self,
        bell_state_relation: object,
    ) -> torch.Tensor:
        """
        Compute the entanglement entropy regularisation loss.

        Args:
            bell_state_relation: Any object with ``entanglement_entropy_all()``
                                 returning (num_relations,) float32.

        Returns:
            Scalar float32 tensor.
        """
        return self.entropy_reg(bell_state_relation)

    def forward_contextuality(
        self,
        joint_scores: torch.Tensor,
        factorized_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contextuality (non-classicality) penalty.

        Args:
            joint_scores:      (B,) float32 — 2-hop quantum joint scores.
            factorized_scores: (B,) float32 — classical factorised scores.

        Returns:
            Scalar float32 tensor.
        """
        return self.contextuality(joint_scores, factorized_scores)

    def forward_decoherence(
        self,
        decoherence_channel: object,
        path_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the decoherence rate shaping loss.

        Args:
            decoherence_channel: Any object with a ``rates`` property returning
                                 (num_relations,) float32 ∈ (0, 1).
            path_lengths:        (B,) int64 — hop counts (optional).

        Returns:
            Scalar float32 tensor.
        """
        return self.decoherence(decoherence_channel, path_lengths)

    def get_loss_weights(self) -> dict[str, float]:
        """
        Return a mapping from loss component name to its scalar weight.

        Useful for logging hyperparameters and verifying configuration.

        Returns:
            Dict[str, float] with keys:
                ``'ranking'``, ``'teleport_contrast'``, ``'entropy_reg'``,
                ``'contextuality'``, ``'decoherence'``.
        """
        return {
            "ranking": self.ranking_loss.weight,
            "teleport_contrast": self.teleport_contrast.weight,
            "entropy_reg": self.entropy_reg.weight,
            "contextuality": self.contextuality.weight,
            "decoherence": self.decoherence.weight,
        }

    def extra_repr(self) -> str:
        weights = self.get_loss_weights()
        parts = ", ".join(f"{k}={v}" for k, v in weights.items())
        return (
            f"complex_dim={self.complex_dim}, "
            f"num_relations={self.num_relations}, "
            f"weights=({parts})"
        )
