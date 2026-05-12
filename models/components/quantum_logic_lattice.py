"""
models/components/quantum_logic_lattice.py — Orthocomplemented Lattice Module [V4]

SOURCE: QIQE-KGC (Information Sciences, 2023)
TECHNIQUE: Quantum Logic Embedding with Orthocomplemented Lattices

PURPOSE:
    Addresses the core failure of V1-V3: the model only considers LOCAL path amplitudes.
    A path might have high amplitude leading to tail t, but if t is LOGICALLY IMPOSSIBLE
    given the global graph structure, the model should suppress it.

    The orthocomplemented lattice enforces GLOBAL logical consistency:
    - Every proposition p and its negation ¬p must be orthogonal: p·¬p = 0
    - In KGE terms: if triple (h,r,t) is true, then (h,r,t') for all other t' must be penalized
    - This is the "quantum firewall" against hallucinating logically impossible predictions

THE ORTHOCOMPLEMENTED LATTICE (from QIQE-KGC Section 3.2):
    In quantum logic, propositions are subspaces of a Hilbert space.
    A proposition p and its complement ¬p must satisfy:
        p ⊕ ¬p = I (spans the full space)
        p ∧ ¬p = 0 (no overlap — orthogonal)

    Implementation (soft/differentiable for gradient training):
        E_r ∈ R^{2d}: logical embedding of relation r (sigmoid-activated → [0,1])
        ¬E_r = 1 - E_r:  complement (guaranteed if E_r ∈ [0,1])
        Orthogonality: E_r · (1-E_r) = 0 ← enforced via L_binary loss
        Logic score: f(E_r) = ||(1-E_r) ⊙ E_r||_F (lower = more orthogonal = more logical)

THREE LOSS COMPONENTS (from QIQE-KGC):
    L_ELoss (Entity Loss):
        Entity logical embeddings should map to valid quantum propositions.
        Penalizes entity embeddings that violate orthogonality constraints.

    L_LLoss (Logical Relationship Loss):
        If (h, r, t) is true and (h, r, t') is false,
        their logical embeddings should be orthogonal: E_{h,r,t} · E_{h,r,t'} ≈ 0.

    L_MLoss (Membership Loss):
        Entities should logically "belong" to their relation's domain/range.
        E_h ∈ domain(r) and E_t ∈ range(r).

V4 EXTENSION — PATH-LEVEL LOGIC:
    V4 applies the lattice at the PATH level, not just triple level:
    - For each path Pi = (r1, r2, ..., rn), compute a path logical embedding
    - Paths that violate global constraints get lower logic scores
    - Logic score multiplies the path amplitude weight αᵢ
    - This is the "logical filter" on top of the quantum amplitude
"""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumLogicLattice(nn.Module):
    """
    Orthocomplemented lattice for global logical consistency enforcement.

    From QIQE-KGC: enforces that entity-relation logical embeddings satisfy
    quantum orthogonality constraints, preventing logically impossible predictions.

    In V4, this module works ALONGSIDE the quaternion amplitude aggregator:
    - Quaternion aggregator: WHAT paths lead where (amplitude computation)
    - Logic lattice: WHETHER those paths are globally consistent (logic filter)

    The total V4 score = amplitude_score × logic_weight + logic_penalty

    Args:
        num_entities:   Total entity count.
        num_relations:  Total relation count.
        lattice_dim:    Dimension of logical embeddings (2d in QIQE-KGC notation).
                        Typically set to 2 × quaternion_dim.
        binary_weight:  Weight for L_binary loss (enforces orthogonality).
        logic_weight:   Weight for L_LLoss (logical relationship consistency).
        member_weight:  Weight for L_MLoss (entity-relation membership).
    """

    def __init__(
        self,
        num_entities:   int,
        num_relations:  int,
        lattice_dim:    int   = 64,
        binary_weight:  float = 0.1,
        logic_weight:   float = 0.05,
        member_weight:  float = 0.05,
    ) -> None:
        super().__init__()
        self.num_entities   = num_entities
        self.num_relations  = num_relations
        self.lattice_dim    = lattice_dim
        self.binary_weight  = binary_weight
        self.logic_weight   = logic_weight
        self.member_weight  = member_weight

        # Logical embeddings for entities: E_e ∈ R^{lattice_dim}
        # Passed through sigmoid during forward to constrain to [0,1]
        self.entity_logic = nn.Embedding(num_entities, lattice_dim)

        # Logical embeddings for relations: E_r ∈ R^{lattice_dim}
        self.relation_logic = nn.Embedding(num_relations, lattice_dim)

        # Domain and range projectors for membership loss
        # P_domain_r: head entity should "belong" here
        # P_range_r:  tail entity should "belong" here
        self.domain_proj = nn.Embedding(num_relations, lattice_dim)
        self.range_proj  = nn.Embedding(num_relations, lattice_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize logical embeddings near 0.5 (maximally uncertain, not binary yet)."""
        for emb in [self.entity_logic, self.relation_logic,
                    self.domain_proj, self.range_proj]:
            nn.init.uniform_(emb.weight, 0.4, 0.6)  # near 0.5 = maximum entropy

    def _get_soft_binary(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Convert raw logits to soft binary values in (0, 1) via sigmoid.
        During training: values are soft (gradient flows).
        At inference: values approach 0 or 1 (after L_binary loss converges).
        """
        return torch.sigmoid(embedding)

    def get_entity_logic(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Return soft binary logical embedding for entities. Shape: (..., lattice_dim)."""
        return self._get_soft_binary(self.entity_logic(entity_ids))

    def get_relation_logic(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Return soft binary logical embedding for relations. Shape: (..., lattice_dim)."""
        return self._get_soft_binary(self.relation_logic(relation_ids))

    def logic_score(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum logic score from QIQE-KGC Equation 6:
            f_Quantum(E_r) = ||(1_{2d} − E_r) · E_r||

        LOWER score = MORE orthogonal = MORE logically consistent.
        We return 1 - normalized_score so that HIGHER = more consistent.

        Args:
            relation_ids: (B,) int64 relation IDs.

        Returns:
            (B,) float32 logic consistency score in [0, 1].
            1.0 = perfectly binary/orthogonal (maximum consistency).
            0.5 = uniform values (maximum uncertainty).
        """
        E_r   = self.get_relation_logic(relation_ids)    # (B, lattice_dim)
        neg_E = 1.0 - E_r                                 # complement ¬E_r
        # Orthogonality: E_r ⊙ ¬E_r (element-wise product) → 0 when binary
        ortho = (E_r * neg_E)                             # (B, lattice_dim)
        # ||E_r ⊙ ¬E_r||_F per sample (Frobenius-like per-row norm)
        raw_score = ortho.pow(2).sum(dim=-1).sqrt()       # (B,) — lower = better

        # Normalize to [0, max_possible], where max is at E=0.5 (uniform)
        # At E=0.5: each element contributes 0.5×0.5 = 0.25, sum = 0.25·d
        max_score = 0.25 * self.lattice_dim
        # Invert so that 1.0 = fully binary (most logical), 0.0 = uniform (least logical)
        consistency = 1.0 - (raw_score / (math.sqrt(max_score) + 1e-10)).clamp(0, 1)
        return consistency   # (B,) in [0, 1]

    def path_logic_score(
        self,
        path_relation_ids: list[int],
        device: torch.device,
    ) -> float:
        """
        Compute logic score for a multi-hop path by composing relation logic.

        The path logical embedding is the element-wise AND of relation logics
        (minimum in soft Boolean algebra), then scored for orthogonality.

        Args:
            path_relation_ids: List of relation IDs forming the path.
            device: Torch device.

        Returns:
            Float consistency score for the path (higher = more logical).
        """
        if not path_relation_ids:
            return 1.0

        rel_ids  = torch.tensor(path_relation_ids, dtype=torch.long, device=device)
        rel_logs = self.get_relation_logic(rel_ids)   # (n_hops, lattice_dim)

        # Soft AND (minimum) over path relations — path is consistent only if ALL relations are
        path_logic = rel_logs.min(dim=0).values       # (lattice_dim,)

        # Score the composed path logic
        neg_path = 1.0 - path_logic
        ortho    = (path_logic * neg_path).pow(2).sum().sqrt()
        max_score = 0.25 * self.lattice_dim
        consistency = float(1.0 - (ortho / (math.sqrt(max_score) + 1e-10)).clamp(0, 1))
        return consistency

    # ── Loss Functions (from QIQE-KGC) ─────────────────────────────────────────

    def l_binary_loss(
        self,
        entity_ids:   torch.Tensor,   # (B,) int64
        relation_ids: torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:
        """
        L_binary: Entity Element Loss from QIQE-KGC.

        Forces E_e and E_r toward binary values (0 or 1).
        ||E ⊙ (1-E)||_F → 0 when E is binary.

        This is the training mechanism that makes the lattice "quantum":
        forcing binary values creates the orthocomplemented structure where
        p and ¬p are truly orthogonal (non-overlapping propositions).
        """
        E_e = self.get_entity_logic(entity_ids)      # (B, d)
        E_r = self.get_relation_logic(relation_ids)  # (B, d)

        # L_binary = ||E ⊙ (1-E)||_F
        loss_e = (E_e * (1 - E_e)).pow(2).sum(dim=-1).mean()
        loss_r = (E_r * (1 - E_r)).pow(2).sum(dim=-1).mean()

        return self.binary_weight * (loss_e + loss_r)

    def l_logic_loss(
        self,
        pos_h_ids: torch.Tensor,   # (B,) int64 — positive triple head
        pos_r_ids: torch.Tensor,   # (B,) int64
        pos_t_ids: torch.Tensor,   # (B,) int64
        neg_t_ids: torch.Tensor,   # (B,) int64 — corrupted tail
    ) -> torch.Tensor:
        """
        L_LLoss: Logical Relationship Loss from QIQE-KGC.

        Positive and negative tails for the same (h, r) query should have
        ORTHOGONAL logical embeddings in the relation's subspace.

        If (h, r, t_pos) is true and (h, r, t_neg) is false:
            E_{t_pos} ⊙ E_{t_neg} ≈ 0  (orthogonality of truth values)

        This prevents the model from assigning high logic scores to
        contradictory predictions, acting as the quantum firewall.
        """
        E_pos = self.get_entity_logic(pos_t_ids)  # (B, d)
        E_neg = self.get_entity_logic(neg_t_ids)  # (B, d)

        # Dot product should be near 0 (orthogonal)
        overlap = (E_pos * E_neg).sum(dim=-1)            # (B,)
        # Penalize non-zero overlap (should be orthogonal)
        logic_loss = overlap.pow(2).mean()

        return self.logic_weight * logic_loss

    def l_member_loss(
        self,
        h_ids: torch.Tensor,   # (B,) int64
        r_ids: torch.Tensor,   # (B,) int64
        t_ids: torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:
        """
        L_MLoss: Membership Loss from QIQE-KGC.

        Head entity should be in relation's domain; tail in relation's range.
        E_h ∈ domain(r): high overlap between head embedding and domain projector.
        E_t ∈ range(r):  high overlap between tail embedding and range projector.

        Training signal: maximize E_h · P_domain_r and E_t · P_range_r.
        This ensures the model learns which entity types are valid for each relation.
        """
        E_h      = self.get_entity_logic(h_ids)    # (B, d)
        E_t      = self.get_entity_logic(t_ids)    # (B, d)
        P_domain = self._get_soft_binary(self.domain_proj(r_ids))  # (B, d)
        P_range  = self._get_soft_binary(self.range_proj(r_ids))   # (B, d)

        # Maximize membership overlap (minimize negative overlap)
        domain_overlap = (E_h * P_domain).sum(dim=-1)  # (B,) — want high
        range_overlap  = (E_t * P_range).sum(dim=-1)   # (B,) — want high

        # Loss = -log(sigmoid(overlap)) — negative membership is penalized
        member_loss = -F.logsigmoid(domain_overlap).mean() \
                    - F.logsigmoid(range_overlap).mean()

        return self.member_weight * member_loss

    def total_loss(
        self,
        h_ids:     torch.Tensor,
        r_ids:     torch.Tensor,
        t_pos_ids: torch.Tensor,
        t_neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined quantum logic loss: L_ELoss + L_LLoss + L_MLoss.

        Call this every training step alongside the main BCE/quaternion loss.
        The gradients flow back to entity_logic and relation_logic embeddings.
        """
        l_bin = self.l_binary_loss(h_ids, r_ids)
        l_log = self.l_logic_loss(h_ids, r_ids, t_pos_ids, t_neg_ids)
        l_mem = self.l_member_loss(h_ids, r_ids, t_pos_ids)
        return l_bin + l_log + l_mem

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "entity_logic":   self.entity_logic.weight.numel(),
            "relation_logic": self.relation_logic.weight.numel(),
            "domain_proj":    self.domain_proj.weight.numel(),
            "range_proj":     self.range_proj.weight.numel(),
            "total":          total,
        }

    def extra_repr(self) -> str:
        return (
            f"lattice_dim={self.lattice_dim}, "
            f"binary_weight={self.binary_weight}, "
            f"logic_weight={self.logic_weight}"
        )
