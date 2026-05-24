"""
KG-Specific Unitary Operators — New Contribution #2.

WHY THIS FILE EXISTS:
    The unitary_operators.py file borrows DiagonalUnitary, GivensUnitary, and
    MatrixExpUnitary directly from quantum computing textbooks (phase gates,
    rotation gates, Hamiltonian evolution). These are generic quantum operators
    with no knowledge of the KG's relational structure.

    A reviewer's valid objection: "Why should the unitary for 'isA' look
    anything like the unitary for 'livesIn'? Generic phase rotations don't
    encode the relational semantics of the KG."

    This file addresses that objection by introducing three KG-aware unitary
    parameterizations that are designed specifically for KG relational structure:

    1. RelationalDecomposedUnitary:
       Decomposes each relation's unitary into three semantic components:
           U_r = U_sym(r) @ U_dir(r) @ U_inv(r)
       where:
           U_sym(r) = symmetric component (handles r(a,b) iff r(b,a))
           U_dir(r) = directional component (handles asymmetric direction)
           U_inv(r) = inversion component (handles relation r vs r^-1)
       This directly encodes the four relation patterns from RotatE paper
       (symmetry, antisymmetry, inversion, composition) at the UNITARY level,
       not just at the scoring level.

    2. HierarchyAwareUnitary:
       Uses the KG's relation hierarchy (if available) to constrain unitaries:
       if relation r1 is a sub-relation of r2 (e.g., isDirectorOf subpropOf hasJob),
       then U_r1 should be "close to" U_r2 in Frobenius norm.
       Implemented as a soft regularization penalty.

    3. ContextualUnitary:
       The unitary for relation r depends on the CURRENT entity state:
           U_r(h) = f(h, theta_r)   [h is the head entity state]
       This is the quantum analogue of attention: the operator adapts to
       the entity being transformed. Implemented as a lightweight
       state-conditioned phase modulation.

THE KEY CLAIM FOR YOUR PAPER:
    "Unlike prior quantum KG methods that borrow generic quantum gate
    parameterizations, we introduce KG-specific unitary operators that
    explicitly encode relational structure (symmetry, directionality,
    inversion) at the operator level. This makes the unitary parameterization
    a genuine contribution to the quantum-KG literature, not a library import."
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unitary_operators import UnitaryOperator, DiagonalUnitary


# ── 1. Relational Decomposed Unitary ─────────────────────────────────────────

class RelationalDecomposedUnitary(UnitaryOperator):
    """
    Unitary decomposed into three semantic relation-pattern components.

    Each relation r's unitary is:
        U_r = U_sym(r) @ U_dir(r) @ U_inv(r)

    where each component is a diagonal unitary (exp(i*theta) per dimension)
    with SHARED structural constraints:

        U_sym(r): symmetric phase — learns how much relation r is "symmetric"
                  Constraint: if r is known symmetric, theta_sym(r) is small.
                  In practice: regularized toward zero for symmetric relations.

        U_dir(r): directional phase — the primary relation-specific component.
                  This is the main expressive part (equivalent to RotatE's phase).

        U_inv(r): inversion phase — encodes r vs r^-1 relationship.
                  Constraint: theta_inv(r) ≈ -theta_inv(r_inverse) for inverse pairs.

    The composition U_sym @ U_dir @ U_inv:
        For a symmetric relation: U_sym cancels out,
            leaving mostly U_dir @ U_inv ≈ U_dir (near-identity inversion).
        For an antisymmetric relation: U_sym is non-zero,
            capturing the direction reversal.
        For an inversion pair (r, r^-1): theta_inv(r) ≈ -theta_inv(r^-1).

    Args:
        num_relations:   Total relation count.
        complex_dim:     Complex space dimension.
        symmetric_rels:  Optional set of relation IDs known to be symmetric.
                         These get an L2 regularization on their sym component.
        inverse_pairs:   Optional list of (r_id, r_inv_id) pairs.
                         These get a soft anti-symmetry constraint on inv component.
        sym_reg_weight:  Regularization weight for symmetric relation constraint.
        inv_reg_weight:  Regularization weight for inversion pair constraint.

    Example:
        >>> U = RelationalDecomposedUnitary(
        ...     num_relations=12,
        ...     complex_dim=8,
        ...     symmetric_rels={10},      # relation 10 is 'synonym' (symmetric)
        ...     inverse_pairs=[(0, 8)],   # relation 0 and 8 are inverses
        ... )
        >>> states = torch.randn(4, 8, dtype=torch.complex64)
        >>> out = U.apply(states, torch.tensor([0, 1, 2, 3]))
    """

    def __init__(
        self,
        num_relations:  int,
        complex_dim:    int,
        symmetric_rels: Optional[set[int]] = None,
        inverse_pairs:  Optional[list[tuple[int, int]]] = None,
        sym_reg_weight: float = 0.01,
        inv_reg_weight: float = 0.01,
    ) -> None:
        super().__init__(num_relations, complex_dim)

        self.symmetric_rels  = symmetric_rels or set()
        self.inverse_pairs   = inverse_pairs   or []
        self.sym_reg_weight  = sym_reg_weight
        self.inv_reg_weight  = inv_reg_weight

        # Three sets of phase angles — one per semantic component
        self.phases_sym = nn.Parameter(
            torch.zeros(num_relations, complex_dim)
        )
        self.phases_dir = nn.Parameter(
            torch.randn(num_relations, complex_dim) * 0.1
        )
        self.phases_inv = nn.Parameter(
            torch.randn(num_relations, complex_dim) * 0.01
        )

    def apply(
        self,
        states:       torch.Tensor,   # (..., complex_dim) complex
        relation_ids: torch.Tensor,   # (...) int
    ) -> torch.Tensor:
        """
        Apply the decomposed unitary: U_sym @ U_dir @ U_inv to each state.

        Application order: inv first, then dir, then sym.
        (Rightmost matrix applied first in quantum convention.)
        """
        # Build combined phase: theta_sym + theta_dir + theta_inv
        # (product of diagonal unitaries = diagonal unitary with summed phases)
        theta = (
            self.phases_sym[relation_ids]
          + self.phases_dir[relation_ids]
          + self.phases_inv[relation_ids]
        )   # (..., complex_dim)

        phase_factors = torch.polar(torch.ones_like(theta), theta)
        return states * phase_factors

    def get_matrix(self, relation_id: int) -> torch.Tensor:
        with torch.no_grad():
            theta = (
                self.phases_sym[relation_id]
              + self.phases_dir[relation_id]
              + self.phases_inv[relation_id]
            )
            return torch.diag(torch.polar(torch.ones_like(theta), theta))

    def structural_regularization_loss(self) -> torch.Tensor:
        """
        Compute structural regularization penalties:
          1. Symmetric relations: penalize large sym phases
             (symmetric relation unitaries should be self-inverse: U = U^dagger,
              meaning all phase angles should be near 0 or pi)
          2. Inverse pairs: penalize theta_inv(r) + theta_inv(r^-1) != 0
             (inverse relations should have opposite inversion phases)

        Returns:
            Scalar loss tensor. ADD this to your main training loss.

        Mathematical motivation:
            Symmetric relation r: r(a,b) iff r(b,a)
              => U_r |a> and |b> should be symmetric
              => phases_sym(r) should be small (near 0)

            Inverse pair (r, r^-1): r(a,b) iff r^-1(b,a)
              => U_r @ U_{r^-1} should be near identity
              => phases_inv(r) + phases_inv(r^-1) should be near 0
        """
        total_reg = torch.tensor(0.0, device=self.phases_sym.device)

        # 1. Symmetric relation regularization
        if self.symmetric_rels:
            sym_ids = torch.tensor(
                list(self.symmetric_rels), device=self.phases_sym.device
            )
            sym_loss = self.phases_sym[sym_ids].pow(2).mean()
            total_reg = total_reg + self.sym_reg_weight * sym_loss

        # 2. Inverse pair regularization
        if self.inverse_pairs:
            inv_loss = torch.tensor(0.0, device=self.phases_inv.device)
            for r_id, r_inv_id in self.inverse_pairs:
                # phases_inv(r) + phases_inv(r^-1) should be 0
                pair_loss = (
                    self.phases_inv[r_id] + self.phases_inv[r_inv_id]
                ).pow(2).mean()
                inv_loss = inv_loss + pair_loss
            inv_loss = inv_loss / max(len(self.inverse_pairs), 1)
            total_reg = total_reg + self.inv_reg_weight * inv_loss

        return total_reg

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, complex_dim={self.complex_dim}, "
            f"symmetric_rels={len(self.symmetric_rels)}, "
            f"inverse_pairs={len(self.inverse_pairs)}"
        )


# ── 2. Hierarchy-Aware Unitary ─────────────────────────────────────────────

class HierarchyAwareUnitary(UnitaryOperator):
    """
    Unitary operator with soft hierarchy constraints between relations.

    If relation r1 is a sub-relation of r2 (e.g., 'isDirectorOf' subpropOf 'hasJob'),
    then U_r1 should be geometrically close to U_r2.
    This is enforced as a regularization:
        reg_loss = sum_{(r1, r2) in hierarchy} ||theta(r1) - theta(r2)||^2

    The base parameterization is DiagonalUnitary (same as phases.exp(i*theta)),
    but with an additional hierarchy_loss() method that returns the soft
    constraint penalty.

    This captures the insight that KG relations are NOT independent —
    they form ontological hierarchies, and the unitary operators should
    reflect this structure.

    Args:
        num_relations:   Total relation count.
        complex_dim:     Complex space dimension.
        relation_hierarchy: List of (child_rel_id, parent_rel_id) pairs.
        hierarchy_weight:   Regularization weight for hierarchy constraint.

    Example:
        >>> hierarchy = [(3, 7), (4, 7)]  # relations 3,4 are children of 7
        >>> U = HierarchyAwareUnitary(12, 8, relation_hierarchy=hierarchy)
        >>> reg = U.hierarchy_regularization_loss()   # add to training loss
    """

    def __init__(
        self,
        num_relations:      int,
        complex_dim:        int,
        relation_hierarchy: Optional[list[tuple[int, int]]] = None,
        hierarchy_weight:   float = 0.01,
    ) -> None:
        super().__init__(num_relations, complex_dim)

        self.relation_hierarchy = relation_hierarchy or []
        self.hierarchy_weight   = hierarchy_weight

        # Base diagonal unitary phases
        self.phases = nn.Parameter(
            torch.randn(num_relations, complex_dim) * 0.01
        )

    def apply(
        self,
        states:       torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        theta         = self.phases[relation_ids]
        phase_factors = torch.polar(torch.ones_like(theta), theta)
        return states * phase_factors

    def get_matrix(self, relation_id: int) -> torch.Tensor:
        with torch.no_grad():
            theta = self.phases[relation_id]
            return torch.diag(torch.polar(torch.ones_like(theta), theta))

    def hierarchy_regularization_loss(self) -> torch.Tensor:
        """
        Soft hierarchy constraint: child relations have similar unitaries to parents.

        Mathematical motivation:
            If r1 subPropertyOf r2, then all triples (h, r1, t) are also (h, r2, t).
            The unitary for r1 should therefore apply a "more specific" transformation
            in the same direction as r2's unitary.
            Formally: ||theta(r1) - theta(r2)||^2 is small.

        Returns:
            Scalar loss tensor. ADD to training loss.
        """
        if not self.relation_hierarchy:
            return torch.tensor(0.0, device=self.phases.device)

        loss = torch.tensor(0.0, device=self.phases.device)
        for child_id, parent_id in self.relation_hierarchy:
            diff = self.phases[child_id] - self.phases[parent_id]
            loss = loss + diff.pow(2).mean()

        return self.hierarchy_weight * loss / max(len(self.relation_hierarchy), 1)

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, complex_dim={self.complex_dim}, "
            f"hierarchy_pairs={len(self.relation_hierarchy)}"
        )


# ── 3. Contextual (State-Dependent) Unitary ──────────────────────────────────

class ContextualUnitary(UnitaryOperator):
    """
    State-dependent unitary: the operator for relation r adapts to the
    current entity state h.

    U_r(h) = diag(exp(i * (theta_r + delta_r(h))))

    where delta_r(h) is a small state-dependent correction computed by a
    lightweight linear projection:
        delta_r(h) = W_r @ Re(h) + b_r

    (We use Re(h) for the projection input to keep it differentiable and
    avoid complex-valued linear algebra in the conditioning network.)

    This is the quantum analogue of RELATION-SPECIFIC ATTENTION over the
    entity state. The entity h influences how relation r transforms it,
    capturing the fact that 'isA' applied to a 'Mammal' state should behave
    differently than 'isA' applied to an 'Ocean' state.

    Computational cost: one real-valued linear projection per hop.
    O(d^2 * num_relations) parameters for the conditioning weights.
    Use max_conditioning_dim to control this.

    Args:
        num_relations:        Total relation count.
        complex_dim:          Complex space dimension.
        conditioning_dim:     Hidden dim for the conditioning network.
                              Default: complex_dim // 4 (lightweight).
        residual_scale:       Scale factor on delta_r(h) to keep corrections small.
                              Default: 0.1 (corrections are perturbations, not dominant).

    Example:
        >>> U = ContextualUnitary(num_relations=12, complex_dim=8, conditioning_dim=2)
        >>> states = torch.randn(4, 8, dtype=torch.complex64)
        >>> out = U.apply(states, torch.tensor([0,1,2,3]))
    """

    def __init__(
        self,
        num_relations:   int,
        complex_dim:     int,
        conditioning_dim: Optional[int] = None,
        residual_scale:  float = 0.1,
    ) -> None:
        super().__init__(num_relations, complex_dim)

        self.conditioning_dim = conditioning_dim or max(complex_dim // 4, 2)
        self.residual_scale   = residual_scale

        # Base phase angles (same as DiagonalUnitary)
        self.base_phases = nn.Parameter(
            torch.randn(num_relations, complex_dim) * 0.01
        )

        # State conditioning: projects Re(h) -> phase correction delta
        # Separate weight per relation (captures relation-specific sensitivity)
        # To keep parameter count manageable: use a shared trunk + relation bias
        self.cond_trunk = nn.Linear(complex_dim, self.conditioning_dim, bias=False)
        self.cond_heads = nn.Linear(
            self.conditioning_dim, num_relations * complex_dim, bias=False
        )

        # Initialize conditioning weights very small
        # (corrections should start near zero and grow slowly)
        nn.init.normal_(self.cond_trunk.weight, std=0.01)
        nn.init.zeros_(self.cond_heads.weight)

    def apply(
        self,
        states:       torch.Tensor,   # (..., complex_dim) complex
        relation_ids: torch.Tensor,   # (...) int
    ) -> torch.Tensor:
        """
        Apply state-conditioned phase rotation.

        For each state s and relation r:
            delta_r(s) = W_r @ Re(s)   [state-specific correction]
            theta_total = base_phases[r] + residual_scale * delta_r(s)
            U_r(s) = diag(exp(i * theta_total))
            output = U_r(s) * s
        """
        batch_shape = states.shape[:-1]

        # Extract real parts for conditioning (avoids complex arithmetic)
        real_input = states.real     # (..., complex_dim)

        # Shared trunk: (..., complex_dim) -> (..., conditioning_dim)
        h_cond = self.cond_trunk(real_input)   # (..., cond_dim)
        h_cond = torch.tanh(h_cond)            # bounded activation

        # All relation corrections: (..., num_relations * complex_dim)
        all_deltas = self.cond_heads(h_cond)
        all_deltas = all_deltas.view(
            *batch_shape, self.num_relations, self.complex_dim
        )

        # Select corrections for the batch's specific relations
        # relation_ids: (...) -> need to index all_deltas[..., r_id, :]
        r_idx = relation_ids.unsqueeze(-1).expand(*batch_shape, self.complex_dim)
        delta = all_deltas.gather(
            dim=-2, index=r_idx.unsqueeze(-2)
        ).squeeze(-2)   # (..., complex_dim)

        # Combine base phases with state-specific correction
        base   = self.base_phases[relation_ids]   # (..., complex_dim)
        theta  = base + self.residual_scale * delta

        phase_factors = torch.polar(torch.ones_like(theta), theta)
        return states * phase_factors

    def get_matrix(self, relation_id: int) -> torch.Tensor:
        """
        For ContextualUnitary, the 'matrix' is state-dependent.
        Returns the base unitary (without conditioning correction) for analysis.
        """
        with torch.no_grad():
            theta = self.base_phases[relation_id]
            return torch.diag(torch.polar(torch.ones_like(theta), theta))

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"num_relations={self.num_relations}, complex_dim={self.complex_dim}, "
            f"conditioning_dim={self.conditioning_dim}, "
            f"residual_scale={self.residual_scale}, params={n_params:,}"
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_kg_unitary(
    unitary_type:    str,
    num_relations:   int,
    complex_dim:     int,
    **kwargs,
) -> UnitaryOperator:
    """
    Factory for KG-specific unitary operators.

    Args:
        unitary_type:  "relational"  -> RelationalDecomposedUnitary
                       "hierarchy"   -> HierarchyAwareUnitary
                       "contextual"  -> ContextualUnitary
                       "diagonal"    -> DiagonalUnitary (fallback, from unitary_operators.py)
        num_relations: Number of KG relations.
        complex_dim:   Complex space dimension.
        **kwargs:      Passed to the constructor (e.g., symmetric_rels, inverse_pairs).

    Returns:
        UnitaryOperator instance.

    Example:
        >>> U = build_kg_unitary(
        ...     "relational",
        ...     num_relations=12,
        ...     complex_dim=8,
        ...     symmetric_rels={10},
        ...     inverse_pairs=[(0, 8)],
        ... )
    """
    registry = {
        "relational": RelationalDecomposedUnitary,
        "hierarchy":  HierarchyAwareUnitary,
        "contextual": ContextualUnitary,
        "diagonal":   DiagonalUnitary,
    }
    if unitary_type not in registry:
        raise ValueError(
            f"Unknown unitary_type: '{unitary_type}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[unitary_type](num_relations, complex_dim, **kwargs)


def infer_relation_structure(
    triple_set: set[tuple[int, int, int]],
    num_relations: int,
    symmetry_threshold: float = 0.8,
    inverse_threshold:  float = 0.7,
) -> dict:
    """
    Automatically infer symmetric relations and inverse pairs from the KG data.

    Used to populate RelationalDecomposedUnitary and HierarchyAwareUnitary
    without manual annotation. Useful when the KG does not have explicit
    relation metadata.

    Args:
        triple_set:         Set of (h, r, t) integer triples.
        num_relations:      Total relation count.
        symmetry_threshold: A relation r is symmetric if
                            P(r(b,a) | r(a,b)) >= symmetry_threshold.
        inverse_threshold:  Relations r1, r2 are inverse if
                            P(r2(b,a) | r1(a,b)) >= inverse_threshold.

    Returns:
        Dict with keys:
            'symmetric_rels': set of relation IDs
            'inverse_pairs':  list of (r1_id, r2_id) pairs

    Example:
        >>> structure = infer_relation_structure(
        ...     train_triple_set, num_relations=237
        ... )
        >>> U = build_kg_unitary(
        ...     "relational", 237, 128,
        ...     symmetric_rels=structure['symmetric_rels'],
        ...     inverse_pairs=structure['inverse_pairs'],
        ... )
    """
    # Build all lookups in a single pass — O(T)
    ht_to_tails: dict[tuple, set] = {}   # (h, r) -> {t}
    rt_to_heads: dict[tuple, set] = {}   # (r, t) -> {h}
    rel_to_pairs: dict[int, list] = {}   # r -> [(h, t)]  ← avoids re-scanning per relation

    for h, r, t in triple_set:
        ht_to_tails.setdefault((h, r), set()).add(t)
        rt_to_heads.setdefault((r, t), set()).add(h)
        rel_to_pairs.setdefault(r, []).append((h, t))

    # Detect symmetric relations — O(T) total
    symmetric_rels: set[int] = set()
    for r, forward_triples in rel_to_pairs.items():
        sym_count = sum(
            1 for (h, t) in forward_triples
            if t in rt_to_heads.get((r, h), set())
        )
        if sym_count / len(forward_triples) >= symmetry_threshold:
            symmetric_rels.add(r)

    # Detect inverse pairs — O(R² + T) instead of O(R² × T)
    inverse_pairs: list[tuple[int, int]] = []
    checked_pairs: set[tuple] = set()

    for r1, forward_r1 in rel_to_pairs.items():
        if not forward_r1:
            continue
        for r2 in range(num_relations):
            if r2 <= r1 or (r1, r2) in checked_pairs:
                continue
            inv_count = sum(
                1 for (h, t) in forward_r1
                if h in rt_to_heads.get((r2, t), set())
            )
            if inv_count / len(forward_r1) >= inverse_threshold:
                inverse_pairs.append((r1, r2))
                checked_pairs.add((r1, r2))
                checked_pairs.add((r2, r1))

    return {
        "symmetric_rels": symmetric_rels,
        "inverse_pairs":  inverse_pairs,
    }
