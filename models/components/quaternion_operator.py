"""
models/components/quaternion_operators.py — Hamilton Product Unitary Operators [V4]

SOURCE: QIQE-KGC (Information Sciences, 2023)
TECHNIQUE: Quaternion Relation Operators in H^d

WHY QUATERNION OPERATORS OVER DIAGONAL UNITARY:
    V1-V3 DiagonalUnitary: U_r = diag(exp(iθ)) — rotation in ℂ^d (1 angle/dim)
    V4 QuaternionUnitary:  U_r = q_r ∈ H^d   — rotation in H^d (3 angles/dim)

    The Hamilton product naturally encodes:
    1. Symmetry:     r(a,b) → r(b,a): relation q_r maps to its conjugate
    2. Antisymmetry: q_r ⊙ q_r ≠ identity (non-commutative)
    3. Inversion:    r⁻¹ = conj(r) / ||r||²
    4. Composition:  r₁ ∘ r₂ → q_r1 ⊙ q_r2 (multi-hop path composition)

    This is the formalization used in QIQE-KGC's quaternion spatial module.

MULTI-HOP PATH COMPOSITION:
    U_P = q_rn ⊙ q_r(n-1) ⊙ ... ⊙ q_r1
    Non-commutativity ensures path ORDER matters (Contextual Ontology Theorem).

HARDWARE MAPPING:
    A unit quaternion q = cos(θ/2) + sin(θ/2)(n_i·i + n_j·j + n_k·k)
    encodes 3D rotation by angle θ around axis n = (n_i, n_j, n_k).
    Can be decomposed into 3 sequential RZ gates + 2 CNOT gates on hardware.
    (Still not hardware-native like DiagonalUnitary, but more expressive.)
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .quaternion_states import (
    hamilton_product, quaternion_normalize,
    quaternion_conjugate, quaternion_norm,
)


class QuaternionUnitary(nn.Module):
    """
    Relation operators as unit quaternions in H^d.

    Each relation r stores 4 parameter vectors (r_r, r_i, r_j, r_k) ∈ R^d.
    The unit quaternion constraint ||q_r|| = 1 is enforced during forward.

    The QIQE-KGC formulation treats the relation as a rotation in quaternion space:
        - Entity state evolves as: h_evolved = q_r ⊙ h (Hamilton product)
        - Multi-hop composition: q_P = q_rn ⊙ ... ⊙ q_r1

    Args:
        num_relations:  Number of distinct relations.
        quaternion_dim: Dimension d of quaternion space.
        init_std:       Initialization std for relation phases.
    """

    def __init__(
        self,
        num_relations:  int,
        quaternion_dim: int   = 32,
        init_std:       float = 0.01,
    ) -> None:
        super().__init__()
        self.num_relations  = num_relations
        self.quaternion_dim = quaternion_dim
        # complex_dim for backward compatibility
        self.complex_dim    = quaternion_dim

        # Four component parameter matrices for relation quaternions
        self.rel_r = nn.Embedding(num_relations, quaternion_dim)
        self.rel_i = nn.Embedding(num_relations, quaternion_dim)
        self.rel_j = nn.Embedding(num_relations, quaternion_dim)
        self.rel_k = nn.Embedding(num_relations, quaternion_dim)

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        """
        Initialize near identity quaternion (1, 0, 0, 0).
        Small imaginary parts ensure non-trivial rotation from the start.
        """
        nn.init.normal_(self.rel_r.weight, mean=1.0, std=std)  # bias toward real = 1
        nn.init.normal_(self.rel_i.weight, mean=0.0, std=std)
        nn.init.normal_(self.rel_j.weight, mean=0.0, std=std)
        nn.init.normal_(self.rel_k.weight, mean=0.0, std=std)
        # Normalize immediately
        with torch.no_grad():
            self._normalize_inplace()

    def _normalize_inplace(self) -> None:
        """Enforce unit quaternion for all relations."""
        r = self.rel_r.weight.data
        i = self.rel_i.weight.data
        j = self.rel_j.weight.data
        k = self.rel_k.weight.data
        norms = (r.pow(2) + i.pow(2) + j.pow(2) + k.pow(2)).sum(-1, keepdim=True).sqrt().clamp(1e-10)
        self.rel_r.weight.data = r / norms
        self.rel_i.weight.data = i / norms
        self.rel_j.weight.data = j / norms
        self.rel_k.weight.data = k / norms

    def get_relation_quaternion(
        self,
        relation_ids: torch.Tensor,   # (...) int64
    ) -> tuple[torch.Tensor, ...]:
        """
        Return unit quaternion components for given relations.

        Returns:
            (r, i, j, k) — four float32 tensors (..., quaternion_dim).
        """
        r, i, j, k = (self.rel_r(relation_ids), self.rel_i(relation_ids),
                      self.rel_j(relation_ids), self.rel_k(relation_ids))
        return quaternion_normalize(r, i, j, k)

    def apply(
        self,
        entity_r: torch.Tensor,   # (..., d) float32 — real component
        entity_i: torch.Tensor,   # (..., d) float32 — i component
        entity_j: torch.Tensor,   # (..., d) float32 — j component
        entity_k: torch.Tensor,   # (..., d) float32 — k component
        relation_ids: torch.Tensor,  # (...) int64
    ) -> tuple[torch.Tensor, ...]:
        """
        Apply quaternion rotation to entity state: q_out = q_rel ⊙ q_entity.

        This is the core operation of the QIQE-KGC quaternion spatial module.
        The Hamilton product models the relation as a 3D rotation in H^d,
        mapping the head entity toward where the tail entity should be.

        Args:
            entity_*:     Four float32 components of entity quaternion.
            relation_ids: Relation IDs to look up.

        Returns:
            (r, i, j, k) — rotated entity quaternion, unit-normalized.
        """
        rr, ri, rj, rk = self.get_relation_quaternion(relation_ids)
        # Hamilton product: q_rel ⊙ q_entity
        out_r, out_i, out_j, out_k = hamilton_product(
            rr, ri, rj, rk,
            entity_r, entity_i, entity_j, entity_k,
        )
        # Normalize output to maintain unit quaternion (numeric stability)
        return quaternion_normalize(out_r, out_i, out_j, out_k)

    def compose_path(
        self,
        path_relation_ids: torch.Tensor,   # (n_hops,) int64
    ) -> tuple[torch.Tensor, ...]:
        """
        Compose quaternion operators for a multi-hop path.

        q_P = q_rn ⊙ q_r(n-1) ⊙ ... ⊙ q_r1

        The Hamilton product is ASSOCIATIVE but NOT commutative.
        This ensures different path orderings produce different rotations,
        which is the Contextual Ontology Theorem in quaternion form.

        Args:
            path_relation_ids: 1D tensor of relation IDs in path order.
                               path_relation_ids[0] applied first.

        Returns:
            (r, i, j, k) — composed path quaternion operator, unit-normalized.
        """
        device = self.rel_r.weight.device

        # Start with identity quaternion (1, 0, 0, 0)
        q_r = torch.ones(1,  self.quaternion_dim, device=device)
        q_i = torch.zeros(1, self.quaternion_dim, device=device)
        q_j = torch.zeros(1, self.quaternion_dim, device=device)
        q_k = torch.zeros(1, self.quaternion_dim, device=device)

        for rel_id in path_relation_ids:
            rel_r, rel_i, rel_j, rel_k = self.get_relation_quaternion(rel_id.unsqueeze(0))
            q_r, q_i, q_j, q_k = hamilton_product(
                rel_r, rel_i, rel_j, rel_k,
                q_r,   q_i,   q_j,   q_k,
            )

        return quaternion_normalize(q_r, q_i, q_j, q_k)

    def verify_unit_quaternion(self, relation_id: int, tol: float = 1e-3) -> bool:
        """Check that relation quaternion has unit norm (||q_r|| = 1)."""
        with torch.no_grad():
            rr, ri, rj, rk = self.get_relation_quaternion(
                torch.tensor([relation_id], device=self.rel_r.weight.device)
            )
            norm = (rr.pow(2) + ri.pow(2) + rj.pow(2) + rk.pow(2)).sum(-1).sqrt()
            return bool(abs(norm.item() - 1.0) < tol)

    def get_phases(self) -> torch.Tensor:
        """
        Return equivalent phase angles for backward compatibility with V3's TrainerV2.
        Computed as: θ = 2·arctan(||imag|| / real) — quaternion rotation angle.
        """
        with torch.no_grad():
            r  = self.rel_r.weight
            i  = self.rel_i.weight
            j  = self.rel_j.weight
            k  = self.rel_k.weight
            imag_norm = (i.pow(2) + j.pow(2) + k.pow(2)).sqrt()
            theta = 2.0 * torch.atan2(imag_norm, r.abs())
        return theta  # (num_relations, quaternion_dim)

    # Backward compatibility property for InterferenceMonitor
    @property
    def phases(self) -> nn.Parameter:
        """Synthetic phases property using i-component for monitoring."""
        return self.rel_i  # InterferenceMonitor checks .phases attribute

    def extra_repr(self) -> str:
        return f"num_relations={self.num_relations}, quaternion_dim={self.quaternion_dim}"
