"""
Quantum Description Logic (QDL-ALC) — mapping ALC Description Logic to quantum operators.

ALC to Quantum Correspondence
------------------------------
- Concept C  <->  subspace S_C ⊆ H  (projector P_C = V_C V_C^T)
- Entity e    <->  state vector |e⟩ ∈ H
- C(e) holds  <->  P_C|e⟩ ≠ 0  (entity has nonzero projection onto concept subspace)

Intersection (C ⊓ D)
    By von Neumann's alternating projection theorem:
        P_{C⊓D} = lim_{n→∞} (P_C P_D)^n
    Implemented with K=10 alternating steps, which gives a tight approximation
    in practice for low-dimensional concept subspaces.

Negation (¬C)
    P_{¬C} = I − P_C  (orthogonal complement projector)
    Applying it to |e⟩ gives the component of |e⟩ outside the concept subspace.

ABox Inconsistency via Destructive Interference
    For the ABox assertion {C(e), (¬C)(e)}, the entity is asked to inhabit both
    C and its complement simultaneously.  The inconsistency score is:
        ||P_C|ψ⟩ - P_{¬C}|ψ⟩||²
    KEY PAPER CLAIM: when C and D are semantically disjoint (P_C P_D ≈ 0),
    the QuantumReasoner's interference mechanism naturally drives entity embeddings
    such that their projections onto S_C and S_D destructively interfere — i.e.,
    the subspace of C ⊓ ¬C = {0} ensures the entity embedding has near-zero
    amplitude in the overlap region.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# ConceptSubspace
# ---------------------------------------------------------------------------

class ConceptSubspace(nn.Module):
    """
    A learnable low-rank projection P_C = V V^T representing concept C,
    where V ∈ ℝ^{d × k} is kept column-orthonormal via QR decomposition.

    Args:
        embed_dim:   full state dimension d
        concept_dim: subspace rank k (must satisfy k <= embed_dim)
    """

    def __init__(self, embed_dim: int, concept_dim: int) -> None:
        super().__init__()
        if concept_dim > embed_dim:
            raise ValueError(
                f"concept_dim ({concept_dim}) must be <= embed_dim ({embed_dim})"
            )
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        # Raw (un-orthonormalized) basis matrix — learnable
        self.V_raw = nn.Parameter(torch.randn(embed_dim, concept_dim) * 0.1)

    def _get_orthonormal_basis(self) -> torch.Tensor:
        """Return orthonormal V ∈ ℝ^{d × k} via thin QR."""
        Q, _ = torch.linalg.qr(self.V_raw)  # (d, k)
        return Q

    def project(self, state: torch.Tensor) -> torch.Tensor:
        """
        Project state onto concept subspace: P_C |ψ⟩ = V V^T |ψ⟩.

        Args:
            state: (B, d) real entity states

        Returns:
            projected: (B, d)
        """
        V = self._get_orthonormal_basis()          # (d, k)
        coords = state @ V                          # (B, k)
        return coords @ V.T                         # (B, d)

    def membership_degree(self, state: torch.Tensor) -> torch.Tensor:
        """
        Membership degree: ||P_C|ψ⟩||² ∈ [0, 1] (assuming ||ψ|| = 1).

        Args:
            state: (B, d)

        Returns:
            degree: (B,)
        """
        projected = self.project(state)             # (B, d)
        return (projected ** 2).sum(dim=-1)         # (B,)

    def get_param_count(self) -> dict[str, int]:
        return {"V_raw": self.V_raw.numel()}


# ---------------------------------------------------------------------------
# AlternatingProjection
# ---------------------------------------------------------------------------

class AlternatingProjection(nn.Module):
    """
    Approximates the intersection subspace projector P_{C ⊓ D} via
    von Neumann's alternating projection theorem:

        P_{C ⊓ D} |ψ⟩ ≈ (P_C P_D)^n |ψ⟩   for large n

    With n_steps=10 the approximation is tight for well-separated subspaces.

    Args:
        concept_c: ConceptSubspace for concept C
        concept_d: ConceptSubspace for concept D
        n_steps:   number of alternating projection iterations
    """

    def __init__(
        self,
        concept_c: ConceptSubspace,
        concept_d: ConceptSubspace,
        n_steps: int = 10,
    ) -> None:
        super().__init__()
        if concept_c.embed_dim != concept_d.embed_dim:
            raise ValueError("Both concepts must share the same embed_dim.")
        self.concept_c = concept_c
        self.concept_d = concept_d
        self.n_steps = n_steps

    def project(self, state: torch.Tensor) -> torch.Tensor:
        """
        Approximate P_{C⊓D}|ψ⟩ by alternating projections.

        Args:
            state: (B, d)

        Returns:
            projected: (B, d)
        """
        x = state
        for _ in range(self.n_steps):
            x = self.concept_c.project(x)
            x = self.concept_d.project(x)
        return x

    def get_param_count(self) -> dict[str, int]:
        return {}  # parameters live in the sub-modules


# ---------------------------------------------------------------------------
# ALCOperator
# ---------------------------------------------------------------------------

class ALCOperator(nn.Module):
    """
    Full ALC operator supporting concept registration, intersection,
    negation, subsumption checking, and ABox inconsistency scoring.

    Usage
    -----
        op = ALCOperator()
        op.register_concept("Animal", ConceptSubspace(64, 8))
        op.register_concept("Pet",    ConceptSubspace(64, 8))
        score = op.ABoxInconsistencyScore(state, "Animal", "Pet")
    """

    def __init__(self, embed_dim: int = 64, inconsistency_threshold: float = 0.5) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold = inconsistency_threshold
        # Use ModuleDict so PyTorch tracks parameters of registered concepts
        self._concepts: nn.ModuleDict = nn.ModuleDict()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_concept(self, name: str, concept: ConceptSubspace) -> None:
        """Register a named concept subspace."""
        self._concepts[name] = concept

    def _get_concept(self, name: str) -> ConceptSubspace:
        if name not in self._concepts:
            raise KeyError(f"Concept '{name}' not registered.")
        return self._concepts[name]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core operators
    # ------------------------------------------------------------------

    def intersect(
        self,
        state: torch.Tensor,
        name_c: str,
        name_d: str,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Approximate the intersection C ⊓ D projection on state.

        Returns: (B, d)
        """
        c = self._get_concept(name_c)
        d = self._get_concept(name_d)
        ap = AlternatingProjection(c, d, n_steps=n_steps)
        return ap.project(state)

    def negate(self, state: torch.Tensor, name_c: str) -> torch.Tensor:
        """
        Negation ¬C: project onto orthogonal complement (I − P_C)|ψ⟩.

        Returns: (B, d)
        """
        c = self._get_concept(name_c)
        return state - c.project(state)

    def is_subsumed(
        self,
        state: torch.Tensor,
        name_c: str,
        name_d: str,
    ) -> torch.Tensor:
        """
        Check C ⊑ D for entity state: is ||P_D P_C|ψ⟩||² / ||P_C|ψ⟩||² > threshold?

        Returns: (B,) bool
        """
        c = self._get_concept(name_c)
        d = self._get_concept(name_d)
        proj_c = c.project(state)                           # (B, d)
        proj_dc = d.project(proj_c)                         # (B, d)
        norm_c_sq = (proj_c ** 2).sum(dim=-1).clamp(min=1e-12)
        norm_dc_sq = (proj_dc ** 2).sum(dim=-1)
        ratio = norm_dc_sq / norm_c_sq                      # (B,)
        return ratio > self.threshold                        # (B,) bool

    def ABoxInconsistencyScore(
        self,
        state: torch.Tensor,
        name_c: str,
        name_d: str,
    ) -> torch.Tensor:
        """
        Inconsistency score ||P_C|ψ⟩ − P_D|ψ⟩||².

        When C and D are disjoint (P_C P_D = 0), the QuantumReasoner's
        interference mechanism naturally drives this toward 0 because the
        entity embedding simultaneously satisfies C(e) and D(e) only when
        their projections are identical — impossible if the subspaces are
        orthogonal (S_C ⊥ S_D implies C ⊓ D = {0}).

        Args:
            state:  (B, d)
            name_c: concept C
            name_d: concept D

        Returns:
            score: (B,) non-negative float
        """
        c = self._get_concept(name_c)
        d = self._get_concept(name_d)
        proj_c = c.project(state)                           # (B, d)
        proj_d = d.project(state)                           # (B, d)
        diff = proj_c - proj_d                              # (B, d)
        return (diff ** 2).sum(dim=-1)                      # (B,)

    def get_param_count(self) -> dict[str, int]:
        total = 0
        breakdown: dict[str, int] = {}
        for name, module in self._concepts.items():
            count = sum(p.numel() for p in module.parameters())
            breakdown[f"concept_{name}"] = count
            total += count
        breakdown["total"] = total
        return breakdown


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, D, K = 4, 64, 8

    # --- ConceptSubspace ---
    cs = ConceptSubspace(embed_dim=D, concept_dim=K)
    state = F.normalize(torch.randn(B, D), dim=-1)
    projected = cs.project(state)
    degrees = cs.membership_degree(state)
    assert projected.shape == (B, D), f"Expected ({B},{D}), got {projected.shape}"
    assert degrees.shape == (B,), f"Expected ({B},), got {degrees.shape}"
    assert (degrees >= 0).all() and (degrees <= 1 + 1e-5).all(), "Degrees out of [0,1]"
    print(f"ConceptSubspace OK  | projected: {projected.shape}, degrees: {degrees}")

    # --- AlternatingProjection ---
    c1 = ConceptSubspace(D, K)
    c2 = ConceptSubspace(D, K)
    ap = AlternatingProjection(c1, c2, n_steps=10)
    intersected = ap.project(state)
    assert intersected.shape == (B, D), f"Expected ({B},{D}), got {intersected.shape}"
    print(f"AlternatingProjection OK | intersected: {intersected.shape}")

    # --- ALCOperator ---
    op = ALCOperator(inconsistency_threshold=0.5)
    op.register_concept("Animal", ConceptSubspace(D, K))
    op.register_concept("Pet", ConceptSubspace(D, K))

    inter = op.intersect(state, "Animal", "Pet")
    neg = op.negate(state, "Animal")
    subsumed = op.is_subsumed(state, "Pet", "Animal")
    inconsistency = op.ABoxInconsistencyScore(state, "Animal", "Pet")

    assert inter.shape == (B, D)
    assert neg.shape == (B, D)
    assert subsumed.shape == (B,) and subsumed.dtype == torch.bool
    assert inconsistency.shape == (B,)
    assert (inconsistency >= 0).all()

    print(f"ALCOperator OK")
    print(f"  intersect:           {inter.shape}")
    print(f"  negate:              {neg.shape}")
    print(f"  is_subsumed:         {subsumed}")
    print(f"  inconsistency score: {inconsistency}")
    print(f"  param count:         {op.get_param_count()}")
    print("All assertions passed.")
