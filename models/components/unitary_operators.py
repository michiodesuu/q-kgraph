"""
models/components/unitary_operators.py — Unitary Relation Operators  [V1]

PURPOSE:
    Implements QM Postulate 2: relations are unitary operators that evolve
    entity states while preserving their norms.

    Three parameterizations of increasing expressiveness:
        DiagonalUnitary:    exp(iθⱼ) per dimension. Start here. Hardware-native.
        GivensUnitary:      butterfly 2D rotations. Full unitary coverage.
        MatrixExpUnitary:   exp(iH). Most expressive. Hardest to transpile.

THE UNITARITY REQUIREMENT:
    A unitary operator U satisfies U†U = I.
    This is required (not optional) because:
        1. Preserves Born rule: ||U|e⟩|| = 1 when ||e|| = 1 (Theorem 1)
        2. Preserves inner products: ⟨Ua|Ub⟩ = ⟨a|b⟩
        3. Makes path composition valid: (U_rn ··· U_r1)†(U_rn ··· U_r1) = I

HARDWARE MAPPING (DiagonalUnitary):
    exp(iθⱼ) per dimension → RZ(2θⱼ) gate on qubit j.
    NO CNOT gates needed. This is why DiagonalUnitary is the only hardware-
    feasible option for real IBM/IonQ hardware.

USAGE:
    U = DiagonalUnitary(num_relations=12, complex_dim=8)
    ids = torch.tensor([0, 1, 2])
    states = encoder(ids)                    # (3, 8) complex
    transformed = U.apply(states, ids)      # (3, 8) complex, same norms
    U_matrix = U.get_matrix(0)              # (8, 8) complex — relation 0's matrix
    U_composed = U.compose([0, 1])          # (8, 8) — path operator
    U.verify_unitarity(0)                   # True if U†U ≈ I
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Abstract base class ───────────────────────────────────────────────────────

class UnitaryOperator(ABC, nn.Module):
    """
    Abstract base class for all unitary relation operators.

    All subclasses must implement:
        apply():      Apply operator to a batch of states.
        get_matrix(): Return the full unitary matrix for a relation.
    """

    def __init__(self, num_relations: int, complex_dim: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.complex_dim   = complex_dim

    @abstractmethod
    def apply(
        self,
        states:       torch.Tensor,   # (..., complex_dim) complex
        relation_ids: torch.Tensor,   # (...) int64
    ) -> torch.Tensor:
        """Apply the relation operator to a batch of entity states."""
        ...

    @abstractmethod
    def get_matrix(self, relation_id: int) -> torch.Tensor:
        """Return the (complex_dim × complex_dim) complex unitary matrix."""
        ...

    def compose(
        self,
        relation_ids: torch.Tensor,   # (n_hops,) int64
    ) -> torch.Tensor:
        """
        Compose operators for a multi-hop path: U_P = U_rn @ ... @ U_r1.

        The rightmost operator (r1) is applied first (standard QM convention).
        U_P is also unitary because the product of unitary matrices is unitary.

        Args:
            relation_ids: 1D tensor of relation IDs, in path order.
                          relation_ids[0] = first relation applied.
                          relation_ids[-1] = last relation applied.

        Returns:
            (complex_dim, complex_dim) complex unitary matrix U_P.
        """
        device  = next(self.parameters()).device
        U_P     = torch.eye(self.complex_dim, dtype=torch.complex64, device=device)

        for r_id in relation_ids:
            U_r = self.get_matrix(r_id.item())
            U_P = U_r @ U_P   # right-to-left: r1 applied first → leftmost factor applied last

        return U_P

    def verify_unitarity(
        self,
        relation_id: int,
        tol:         float = 1e-4,
    ) -> bool:
        """
        Check whether U†U ≈ I for a given relation's unitary matrix.

        Computes the Frobenius norm ||U†U - I||_max.
        Returns True if max element-wise error < tol.

        Args:
            relation_id: Relation to check.
            tol:         Tolerance for floating point errors.

        Returns:
            True if the operator is numerically unitary.
        """
        with torch.no_grad():
            U = self.get_matrix(relation_id)
            I = torch.eye(self.complex_dim, dtype=torch.complex64, device=U.device)
            error = (U.conj().T @ U - I).abs().max().item()
            return error < tol


# ── 1. DiagonalUnitary ────────────────────────────────────────────────────────

class DiagonalUnitary(UnitaryOperator):
    """
    Diagonal unitary operator: U_r = diag(exp(iθ₁), ..., exp(iθ_d)).

    Each relation r learns d phase angles θ₁, ..., θ_d ∈ ℝ.
    Applying U_r multiplies each complex dimension j by exp(iθⱼ) —
    a pure phase rotation that preserves the state's norm exactly.

    WHY START HERE:
        - Fastest forward pass: element-wise multiplication (no matrix mul)
        - Exactly unitary by construction (|exp(iθ)| = 1 always)
        - Maps directly to RZ gates on quantum hardware (no CNOTs needed)
        - Sufficient expressiveness for the interference mechanism
          (the interference depends on PHASE DIFFERENCES, not gate complexity)

    HARDWARE MAPPING:
        For a d-dimensional complex state encoded in d qubits:
        exp(iθⱼ) on dimension j → RZ(2θⱼ) on qubit j.
        Total gates per hop: d single-qubit gates, 0 two-qubit gates.
        This is the most shallow possible circuit for a unitary transform.

    PAPER NOTE:
        Reviewers will note this is similar to RotatE's relation parameterization.
        The response: RotatE applies exp(iθ) to compute a SINGLE-HOP SCORE.
        We apply it to EVOLVE THE ENTITY STATE for PATH COMPOSITION,
        then SUM AMPLITUDES ACROSS K PATHS before squaring.
        The interference cross-terms are absent from RotatE by design.

    Args:
        num_relations: Number of distinct relations in the KG.
        complex_dim:   Dimension of entity Hilbert space.
        init_scale:    Scale of random initialization for phase angles.
                       Small values help avoid large initial interference noise.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        init_scale:    float = 0.01,
    ) -> None:
        super().__init__(num_relations, complex_dim)
        self.phases = nn.Parameter(
            torch.randn(num_relations, complex_dim) * init_scale
        )

    def apply(
        self,
        states:       torch.Tensor,   # (..., complex_dim) complex
        relation_ids: torch.Tensor,   # (...) int64
    ) -> torch.Tensor:
        """
        Apply diagonal unitary: multiply each state dimension by exp(iθⱼ).

        This is efficient: no matrix multiplication needed.
        phase_factors = exp(i · phases[relation_ids]) ∈ ℂ^complex_dim.
        output = states * phase_factors (element-wise complex multiplication).

        Args:
            states:       Entity states (..., complex_dim) complex64.
            relation_ids: Relation IDs (...) int64.

        Returns:
            Evolved states (..., complex_dim) complex64. Same norm as input.
        """
        theta         = self.phases[relation_ids]
        ones          = torch.ones_like(theta)
        phase_factors = torch.polar(ones, theta)
        
        # Handle All-to-All broadcasting if batch sizes differ
        # e.g., states: [28, dim], phase_factors: [12, dim]
        if states.dim() == 2 and phase_factors.dim() == 2 and states.size(0) != phase_factors.size(0):
            # Resulting shape: [28, 12, dim]
            return states.unsqueeze(1) * phase_factors.unsqueeze(0)
            
        # Default 1-to-1 batched behavior
        # Resulting shape: [batch_size, dim]
        return states * phase_factors

    def get_matrix(self, relation_id: int) -> torch.Tensor:
        """Return the diagonal unitary matrix for one relation."""
        with torch.no_grad():
            theta = self.phases[relation_id]
            diag  = torch.polar(torch.ones_like(theta), theta)
            return torch.diag(diag)

    def extra_repr(self) -> str:
        return f"num_relations={self.num_relations}, complex_dim={self.complex_dim}"


# ── 2. GivensUnitary ──────────────────────────────────────────────────────────

class GivensUnitary(UnitaryOperator):
    """
    Full unitary via butterfly Givens rotations.

    A Givens rotation G(i, j, θ) rotates the (i, j) plane by angle θ.
    Composing d·log(d) Givens rotations produces any unitary in U(d).

    Use for:
        - Ablation to show DiagonalUnitary is sufficient (if performance is
          similar, the parameterization doesn't matter — the mechanism does)
        - Capturing cross-dimensional interference (entanglement between dims)

    Hardware note: Givens rotations require CNOT gates for qubit pairs.
    NOT hardware-native. Use DiagonalUnitary for hardware experiments.

    Args:
        num_relations: Number of relations.
        complex_dim:   State space dimension.
        n_layers:      Number of butterfly layers. More layers = more expressive.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        n_layers:      int = 2,
    ) -> None:
        super().__init__(num_relations, complex_dim)
        self.n_layers = n_layers

        # Pairs of dimensions to rotate per layer
        # Butterfly pattern: pairs (0,1), (2,3), ... then (1,2), (3,4), ...
        self.pairs: list[list[tuple[int, int]]] = []
        for layer in range(n_layers):
            offset = layer % 2
            layer_pairs = [
                (j, j + 1)
                for j in range(offset, complex_dim - 1, 2)
                if j + 1 < complex_dim
            ]
            self.pairs.append(layer_pairs)

        n_angles = sum(len(p) for p in self.pairs)
        # angles: (num_relations, n_layers, max_pairs_per_layer) float
        # Stored flat for efficiency
        self.angles = nn.Parameter(
            torch.randn(num_relations, n_angles) * 0.01
        )

    def apply(
        self,
        states:       torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Givens rotations to states.

        For each layer: apply G(i, j, α) to all pairs in that layer.
        G(i, j, α)|v⟩ rotates the (i,j) subspace by angle α.
        """
        angles = self.angles[relation_ids]  # (..., n_angles)
        out    = states.clone()
        angle_idx = 0

        for layer_pairs in self.pairs:
            for (i, j) in layer_pairs:
                if angle_idx >= angles.shape[-1]:
                    break
                alpha = angles[..., angle_idx]  # (...,) scalar per batch

                # Apply Givens rotation G(i,j,α)
                cos_a = torch.cos(alpha)
                sin_a = torch.sin(alpha)

                vi = out[..., i].clone()
                vj = out[..., j].clone()

                out[..., i] = cos_a * vi - sin_a * vj
                out[..., j] = sin_a * vi + cos_a * vj

                angle_idx += 1

        return out

    def get_matrix(self, relation_id: int) -> torch.Tensor:
        """Build and return the full unitary matrix for one relation."""
        device  = self.angles.device
        U       = torch.eye(self.complex_dim, dtype=torch.complex64, device=device)
        angles  = self.angles[relation_id]  # (n_angles,)
        angle_idx = 0

        for layer_pairs in self.pairs:
            for (i, j) in layer_pairs:
                if angle_idx >= len(angles):
                    break
                alpha = angles[angle_idx].item()
                cos_a = math.cos(alpha)
                sin_a = math.sin(alpha)

                G = torch.eye(self.complex_dim, dtype=torch.complex64, device=device)
                G[i, i] =  cos_a + 0j
                G[i, j] = -sin_a + 0j
                G[j, i] =  sin_a + 0j
                G[j, j] =  cos_a + 0j

                U = G @ U
                angle_idx += 1

        return U

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"complex_dim={self.complex_dim}, "
            f"n_layers={self.n_layers}"
        )


# ── 3. MatrixExpUnitary ───────────────────────────────────────────────────────

class MatrixExpUnitary(UnitaryOperator):
    """
    Unitary operator: U_r = exp(i · H_r) where H_r is a Hermitian matrix.

    exp(iH) is unitary whenever H is Hermitian (H† = H). This follows from
    the spectral theorem: if H = VΛV†, then exp(iH) = V · exp(iΛ) · V†
    which is clearly unitary.

    This is the generator/Hamiltonian approach from quantum mechanics.
    H_r represents the "Hamiltonian" of relation r — the infinitesimal
    generator of the unitary evolution.

    CAYLEY APPROXIMATION (for efficiency):
        Computing exp(iH) exactly requires eigendecomposition (O(d³)).
        The Cayley approximation is exact (not an approximation!) for
        any skew-Hermitian matrix H:
            exp(iH) ≈ (I + iH/2)(I - iH/2)^{-1}
        This is differentiable via torch.linalg.solve (O(d³) too, but
        avoids eigendecomposition which is less stable for autograd).

    HARDWARE NOTE:
        exp(iH) requires Trotter-Suzuki decomposition for hardware.
        For dense H, this produces O(d²) gate layers.
        NOT hardware-native. Use only for classical experiments.

    Args:
        num_relations: Number of relations.
        complex_dim:   State space dimension.
        use_cayley:    If True, use Cayley approximation (more stable).
                       If False, use torch.matrix_exp (exact but less stable grad).
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        use_cayley:    bool = True,
    ) -> None:
        super().__init__(num_relations, complex_dim)
        self.use_cayley = use_cayley

        # Store upper-triangular + diagonal entries of H (Hermitian parameterization)
        # H is Hermitian: H[i,j] = conj(H[j,i])
        # Upper triangle (including diag): complex_dim*(complex_dim+1)//2 entries
        n_params = complex_dim * complex_dim   # full matrix, enforce Hermitian explicitly
        self.H_params = nn.Parameter(
            torch.randn(num_relations, n_params) * 0.01
        )

    def _get_hermitian(self, relation_id: int) -> torch.Tensor:
        """Build the Hermitian matrix H_r from flat parameters."""
        params = self.H_params[relation_id]   # (d²,)
        H_raw  = params.view(self.complex_dim, self.complex_dim)

        # Split into real and imaginary parts (stored as flat float)
        # Since we store flat params, split in half
        d      = self.complex_dim
        # Use real params as a real symmetric matrix (simplified Hermitian)
        # For full Hermitian: H = 0.5*(H_raw + H_raw.T) makes it symmetric
        H_sym  = 0.5 * (H_raw + H_raw.T)   # symmetric real matrix
        # Convert to complex
        return H_sym.to(dtype=torch.complex64)

    def _compute_unitary(self, relation_id: int) -> torch.Tensor:
        """Compute exp(iH) for one relation."""
        H      = self._get_hermitian(relation_id)
        device = H.device

        if self.use_cayley:
            # Cayley: U = (I + iH/2)(I - iH/2)^{-1}
            I  = torch.eye(self.complex_dim, dtype=torch.complex64, device=device)
            A  = I + 0.5j * H
            B  = I - 0.5j * H
            # Solve B @ U = A  →  U = B^{-1} @ A
            U  = torch.linalg.solve(B, A)
        else:
            U  = torch.matrix_exp(1j * H)

        return U

    def apply(
        self,
        states:       torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exp(iH_r) to each state."""
        batch_shape = states.shape[:-1]
        out         = torch.zeros_like(states)
        flat_states = states.reshape(-1, self.complex_dim)
        flat_rids   = relation_ids.reshape(-1)

        for idx, r_id in enumerate(flat_rids):
            U = self._compute_unitary(r_id.item())
            out_flat_idx = (U @ flat_states[idx].unsqueeze(-1)).squeeze(-1)
            out.reshape(-1, self.complex_dim)[idx] = out_flat_idx

        return out.reshape(*batch_shape, self.complex_dim)

    def get_matrix(self, relation_id: int) -> torch.Tensor:
        with torch.no_grad():
            return self._compute_unitary(relation_id)

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"complex_dim={self.complex_dim}, "
            f"use_cayley={self.use_cayley}"
        )


# ── Factory function ──────────────────────────────────────────────────────────

def build_unitary(
    unitary_type:  str,
    num_relations: int,
    complex_dim:   int,
    **kwargs,
) -> UnitaryOperator:
    """
    Factory function for unitary operator construction.

    Args:
        unitary_type:  "diagonal" (recommended), "givens", or "matrix_exp".
        num_relations: Number of KG relations.
        complex_dim:   Hilbert space dimension.
        **kwargs:      Passed to constructor (e.g., n_layers for GivensUnitary).

    Returns:
        UnitaryOperator instance.

    Example:
        >>> U = build_unitary("diagonal", num_relations=12, complex_dim=8)
        >>> U = build_unitary("givens",   num_relations=12, complex_dim=8, n_layers=3)
        >>> U = build_unitary("matrix_exp", num_relations=12, complex_dim=8)
    """
    registry = {
        "diagonal":   DiagonalUnitary,
        "givens":     GivensUnitary,
        "matrix_exp": MatrixExpUnitary,
    }
    if unitary_type not in registry:
        raise ValueError(
            f"Unknown unitary_type: '{unitary_type}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[unitary_type](num_relations, complex_dim, **kwargs)
