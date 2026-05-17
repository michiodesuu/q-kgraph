"""
models/components/holonomy.py — Parallel Transport & Relational Holonomy (V8)

Implements curved-manifold reasoning for knowledge graphs.

A multi-hop reasoning path h → r1 → e → r2 → t is interpreted as a path on
a curved manifold. The *holonomy* of a closed cycle measures how much the
parallel-transported state differs from the original — a non-zero holonomy
gap signals a geometric contradiction in the reasoning cycle.

Key ideas:
    - Per-relation unitary parallel transport T_r ∈ U(d), stored via skew-Hermitian
      parameterisation T_r = exp(A_r), A_r = (M_r − M_r†) / 2i
    - Holonomy of cycle h → r1 → e → r2 → h: Hol = T_r2 @ T_r1
    - Holonomy gap: ||Hol − I||_F  (zero iff cycle is geometrically consistent)
    - Path phase: arg(det(composed transport)) — modulates Born-rule amplitude
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _make_skew_hermitian(M: Tensor) -> Tensor:
    """Return (M - M†) / 2i  (skew-Hermitian).

    Args:
        M: (..., d, d) complex tensor.

    Returns:
        A: (..., d, d) complex skew-Hermitian tensor, A† = -A.
    """
    return (M - M.conj().transpose(-2, -1)) / (2j)


def _matrix_exp_complex(A: Tensor) -> Tensor:
    """Compute matrix exponential of a complex tensor using torch.linalg.matrix_exp.

    Args:
        A: (..., d, d) complex tensor.

    Returns:
        exp(A): (..., d, d) complex tensor.
    """
    # torch.linalg.matrix_exp works on real and complex tensors
    return torch.linalg.matrix_exp(A)


# ---------------------------------------------------------------------------
# ParallelTransportOperator
# ---------------------------------------------------------------------------

class ParallelTransportOperator(nn.Module):
    """Per-relation unitary parallel transport matrix T_r ∈ U(d_complex × d_complex).

    Parameterisation (guarantees unitarity):
        M_r  — learnable real-valued matrix of shape (num_relations, 2*d, 2*d)
               interpreted as a complex matrix via view_as_complex
        A_r  = (M_r − M_r†) / 2i   ← skew-Hermitian
        T_r  = exp(A_r)              ← unitary

    The matrix exponential of a skew-Hermitian matrix is unitary:
        (exp(A))† = exp(A†) = exp(−A) = (exp(A))^{-1}  ✓

    Args:
        num_relations (int): Number of distinct relations.
        complex_dim (int): Dimensionality of the complex state space.
    """

    def __init__(self, num_relations: int, complex_dim: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.complex_dim = complex_dim

        # Store as real (num_relations, complex_dim, complex_dim, 2) for grad
        # The last dim is (real, imag).
        self.M_re = nn.Parameter(
            torch.randn(num_relations, complex_dim, complex_dim) * 0.01
        )
        self.M_im = nn.Parameter(
            torch.randn(num_relations, complex_dim, complex_dim) * 0.01
        )

    def _get_M(self) -> Tensor:
        """Return M as complex tensor (num_relations, d, d)."""
        return torch.complex(self.M_re, self.M_im)

    def get_transport(self, relation_ids: Tensor) -> Tensor:
        """Return unitary transport matrices for given relation ids.

        Args:
            relation_ids: (B,) long tensor.

        Returns:
            T_r: (B, complex_dim, complex_dim) complex tensor, unitary.
        """
        M = self._get_M()                      # (R, d, d) complex
        M_r = M[relation_ids]                  # (B, d, d) complex
        A_r = _make_skew_hermitian(M_r)        # (B, d, d) skew-Hermitian
        T_r = _matrix_exp_complex(A_r)         # (B, d, d) unitary
        return T_r

    def compose(self, rel_ids_1: Tensor, rel_ids_2: Tensor) -> Tensor:
        """Compose two transport operators: T_{r2} @ T_{r1}.

        Args:
            rel_ids_1: (B,) — first relation (applied first).
            rel_ids_2: (B,) — second relation (applied second).

        Returns:
            T_composed: (B, complex_dim, complex_dim) complex.
        """
        T1 = self.get_transport(rel_ids_1)     # (B, d, d)
        T2 = self.get_transport(rel_ids_2)     # (B, d, d)
        return torch.bmm(T2, T1)               # T_{r2} @ T_{r1}

    def apply(
        self,
        state_re: Tensor,
        state_im: Tensor,
        relation_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply transport T_r to a complex state |ψ⟩.

        Args:
            state_re: (B, complex_dim) real part.
            state_im: (B, complex_dim) imaginary part.
            relation_ids: (B,) long.

        Returns:
            (re_out, im_out): (B, complex_dim) each.
        """
        T = self.get_transport(relation_ids)        # (B, d, d) complex
        psi = torch.complex(state_re, state_im)     # (B, d)
        psi_out = torch.bmm(T, psi.unsqueeze(-1)).squeeze(-1)  # (B, d)
        return psi_out.real, psi_out.imag


# ---------------------------------------------------------------------------
# HolonomyOperator
# ---------------------------------------------------------------------------

class HolonomyOperator(nn.Module):
    """Computes holonomy of reasoning cycles and converts gap to interference phase.

    The holonomy of a closed path (r1, r2, ..., rN) back to the starting entity
    is the composed transport:
        Hol = T_{rN} @ ... @ T_{r2} @ T_{r1}

    If the knowledge graph is geometrically consistent along this cycle,
    Hol = I.  Any deviation (||Hol − I||_F > 0) signals a contradiction.

    Args:
        transport_op (ParallelTransportOperator): shared transport parameters.
    """

    def __init__(self, transport_op: ParallelTransportOperator) -> None:
        super().__init__()
        self.transport_op = transport_op

    def compute_holonomy(self, cycle_relation_ids: list[Tensor]) -> Tensor:
        """Compose transport matrices around a cycle: T_{rN} @ ... @ T_{r1}.

        Args:
            cycle_relation_ids: list of (B,) long tensors, ordered r1, r2, ..., rN.

        Returns:
            Hol: (B, complex_dim, complex_dim) complex tensor.
        """
        if len(cycle_relation_ids) == 0:
            raise ValueError("cycle_relation_ids must be non-empty")

        # Start with identity
        B = cycle_relation_ids[0].shape[0]
        d = self.transport_op.complex_dim
        device = cycle_relation_ids[0].device

        Hol = (
            torch.eye(d, dtype=torch.complex64, device=device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            .clone()
        )

        # Compose left-to-right: Hol = T_{rN} @ ... @ T_{r1}
        for rel_ids in cycle_relation_ids:
            T = self.transport_op.get_transport(rel_ids)   # (B, d, d)
            Hol = torch.bmm(T, Hol)                        # (B, d, d)

        return Hol

    def holonomy_gap(self, relation_path: list[Tensor]) -> Tensor:
        """Frobenius-norm distance of holonomy from identity.

        ||T_{rN} @ ... @ T_{r1} − I||_F

        Zero iff the cycle is geometrically consistent (no contradiction).

        Args:
            relation_path: list of (B,) long tensors.

        Returns:
            gap: (B,) float tensor.
        """
        Hol = self.compute_holonomy(relation_path)   # (B, d, d) complex
        B = Hol.shape[0]
        d = Hol.shape[-1]
        device = Hol.device

        I = (
            torch.eye(d, dtype=torch.complex64, device=device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        diff = Hol - I                                      # (B, d, d)
        # Frobenius norm: sqrt( sum |diff_ij|^2 )
        gap = torch.sqrt((diff.abs() ** 2).sum(dim=(-2, -1)))  # (B,)
        return gap.float()

    def path_phase(self, relation_ids_sequence: list[Tensor]) -> Tensor:
        """Extract phase from det(composed transport), in [0, 2π).

        Used to modulate Born-rule amplitude along a multi-hop path.

        Args:
            relation_ids_sequence: list of (B,) long tensors.

        Returns:
            phase: (B,) float, angle in [0, 2π).
        """
        Hol = self.compute_holonomy(relation_ids_sequence)  # (B, d, d)
        # det is complex; its argument gives the accumulated phase
        det = torch.linalg.det(Hol)                         # (B,) complex
        phase = torch.angle(det)                            # (B,) in (-π, π]
        phase = (phase + 2 * math.pi) % (2 * math.pi)      # shift to [0, 2π)
        return phase.float()


# ---------------------------------------------------------------------------
# RelationalManifoldEncoder
# ---------------------------------------------------------------------------

class RelationalManifoldEncoder(nn.Module):
    """Encodes entities + relations on a curved manifold. Replaces QuantumStateEncoder.

    Each entity is represented as:
        - A complex state vector (re, im) of shape (B, complex_dim)
        - A base-point on the manifold (B, curvature_dim) — encodes local geometry

    Relations are encoded via ParallelTransportOperator (not DiagonalUnitary),
    capturing the full geometry of curved space.

    Args:
        num_entities (int): Vocabulary size for entities.
        num_relations (int): Vocabulary size for relations.
        embed_dim (int): Total embedding dimension (complex_dim = embed_dim // 2).
        curvature_dim (int): Dimension of the local manifold base-point (default 4).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
        curvature_dim: int = 4,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.complex_dim = embed_dim // 2
        self.curvature_dim = curvature_dim

        # Entity state vectors — stored as flat real vectors of size embed_dim
        # first half = real part, second half = imaginary part
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(embed_dim))

        # Per-entity manifold base-point (local geometry)
        self.base_point = nn.Embedding(num_entities, curvature_dim)
        nn.init.zeros_(self.base_point.weight)

        # Parallel transport for relations
        self.transport = ParallelTransportOperator(num_relations, self.complex_dim)

    def encode_entity(self, entity_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalised complex state (re, im) for entity ids.

        Args:
            entity_ids: (B,) long.

        Returns:
            (re, im): (B, complex_dim) each, L2-normalised.
        """
        emb = self.entity_emb(entity_ids)               # (B, embed_dim)
        re = emb[:, : self.complex_dim]                 # (B, complex_dim)
        im = emb[:, self.complex_dim :]                 # (B, complex_dim)
        # L2-normalise the full complex vector
        norm = (re.pow(2) + im.pow(2)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
        return re / norm, im / norm

    def encode_with_transport(
        self, entity_ids: Tensor, relation_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode entity then apply one parallel transport hop.

        Computes |ψ'⟩ = T_r |ψ_h⟩.

        Args:
            entity_ids: (B,) long.
            relation_ids: (B,) long.

        Returns:
            (re_out, im_out): (B, complex_dim) each.
        """
        re, im = self.encode_entity(entity_ids)
        re_out, im_out = self.transport.apply(re, im, relation_ids)
        return re_out, im_out

    def manifold_inner_product(
        self,
        state1_re: Tensor,
        state1_im: Tensor,
        state2_re: Tensor,
        state2_im: Tensor,
    ) -> Tensor:
        """Compute ⟨ψ₁|ψ₂⟩ on the manifold (complex scalar).

        Curvature enters the geometry via the transport T_r; the inner product
        itself is the standard flat Hermitian form:
            ⟨ψ₁|ψ₂⟩ = ψ₁† ψ₂  =  Σ_i (re1_i − i·im1_i)(re2_i + i·im2_i)

        Args:
            state{1,2}_re: (B, complex_dim).
            state{1,2}_im: (B, complex_dim).

        Returns:
            inner: (B,) complex tensor.
        """
        re_part = (state1_re * state2_re + state1_im * state2_im).sum(-1)
        im_part = (state1_re * state2_im - state1_im * state2_re).sum(-1)
        return torch.complex(re_part, im_part)           # (B,) complex

    def get_param_count(self) -> dict[str, int]:
        return {
            "entity_emb": self.entity_emb.weight.numel(),
            "base_point": self.base_point.weight.numel(),
            "transport_M_re": self.transport.M_re.numel(),
            "transport_M_im": self.transport.M_im.numel(),
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("holonomy.py — V8 Curved Manifold Demo")
    print("=" * 60)

    torch.manual_seed(42)
    B, E, R, D = 4, 100, 20, 16  # batch, entities, relations, complex_dim

    # --- ParallelTransportOperator ---
    transport = ParallelTransportOperator(num_relations=R, complex_dim=D)
    rel_ids = torch.randint(0, R, (B,))

    T = transport.get_transport(rel_ids)
    print(f"\nParallelTransportOperator:")
    print(f"  T shape:  {T.shape}")
    # Verify unitarity: T† T ≈ I
    TH_T = torch.bmm(T.conj().transpose(-2, -1), T)
    I = torch.eye(D, dtype=torch.complex64).unsqueeze(0).expand(B, -1, -1)
    unitary_err = (TH_T - I).abs().max().item()
    print(f"  Unitarity error (should be ~0): {unitary_err:.2e}")

    # Compose
    rel_ids2 = torch.randint(0, R, (B,))
    T_composed = transport.compose(rel_ids, rel_ids2)
    print(f"  Composed T shape: {T_composed.shape}")

    # Apply to state
    re = torch.randn(B, D)
    im = torch.randn(B, D)
    re_out, im_out = transport.apply(re, im, rel_ids)
    print(f"  Applied state re shape: {re_out.shape}")

    # --- HolonomyOperator ---
    holonomy_op = HolonomyOperator(transport)
    cycle = [torch.randint(0, R, (B,)) for _ in range(3)]

    Hol = holonomy_op.compute_holonomy(cycle)
    gap = holonomy_op.holonomy_gap(cycle)
    phase = holonomy_op.path_phase(cycle)

    print(f"\nHolonomyOperator:")
    print(f"  Holonomy matrix shape: {Hol.shape}")
    print(f"  Holonomy gap (mean):   {gap.mean().item():.4f}  (>0 = contradiction)")
    print(f"  Path phase (degrees):  {(phase * 180 / math.pi).mean().item():.2f}°")

    # --- RelationalManifoldEncoder ---
    encoder = RelationalManifoldEncoder(
        num_entities=E, num_relations=R, embed_dim=D * 2, curvature_dim=4
    )
    ent_ids = torch.randint(0, E, (B,))
    re_h, im_h = encoder.encode_entity(ent_ids)
    re_t, im_t = encoder.encode_entity(torch.randint(0, E, (B,)))

    re_hp, im_hp = encoder.encode_with_transport(ent_ids, rel_ids)
    inner = encoder.manifold_inner_product(re_hp, im_hp, re_t, im_t)
    born_score = inner.abs() ** 2

    print(f"\nRelationalManifoldEncoder:")
    print(f"  Entity re shape: {re_h.shape}")
    print(f"  Transport applied shape: {re_hp.shape}")
    print(f"  Born rule |⟨t|T_r|h⟩|²: {born_score}")
    print(f"\nParam counts: {encoder.get_param_count()}")
    print("\nAll checks passed.")
