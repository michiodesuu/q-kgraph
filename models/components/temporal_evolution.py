"""
Hamiltonian Time Evolution for Temporal Knowledge Graphs.

Schrödinger Connection
----------------------
The time-dependent Schrödinger equation (ℏ=1):
    d|ψ⟩/dt = −i H |ψ⟩

has the formal solution:
    |ψ(t)⟩ = U(t) |ψ(0)⟩    where U(t) = exp(−i H t)

Each relation r gets its own Hermitian Hamiltonian H_r, so the evolution
of an entity embedding under relation r for time τ is:
    |ψ(τ)⟩ = exp(−i H_r τ) |ψ(0)⟩

Handling TKG Contradictions via Phase
--------------------------------------
Two temporally-separated facts (h, r, t, τ₁) and (h, r, t', τ₂) naturally
acquire different phases because:
    ⟨t|ψ(τ₁)⟩  vs  ⟨t'|ψ(τ₂)⟩

When τ₂ − τ₁ ≈ π / ||H_r||, the two amplitudes are π out of phase and
destructively interfere when their predictions are summed, which is exactly
the desired behaviour: contradictory facts at widely separated times should
not reinforce each other.

Connection to MatrixExpUnitary
-------------------------------
MatrixExpUnitary (elsewhere in this codebase) applies a static per-relation
Hermitian matrix exp(−i H_r) (time step fixed at 1).  TemporalHamiltonianOperator
generalises this with a continuous scalar timestamp τ passed at inference time.
The Hermitian construction is identical: H_r = (A_r + A_r^T)/2.

Cayley Map for Stability
-------------------------
Direct matrix exponential exp(−i H τ) can be expensive and numerically
sensitive for large τ.  The Cayley approximation:
    U_Cayley = (I − i H τ/2)(I + i H τ/2)^{-1}
preserves unitarity exactly and avoids iterative eigendecomposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TemporalHamiltonianOperator
# ---------------------------------------------------------------------------

class TemporalHamiltonianOperator(nn.Module):
    """
    Per-relation Hermitian Hamiltonian H_r for continuous time evolution.

    U_r(τ) = (I − i H_r τ/2)(I + i H_r τ/2)^{-1}   [Cayley map]

    H_r is constructed as the skew-symmetric part of a learnable matrix A_r:
        H_r = (A_r - A_r^T) / 2

    Skew-symmetric matrices are the real analogue of Hermitian operators:
    they generate the Lie algebra so(d), and exp(H_r t) ∈ O(d) exactly,
    guaranteeing U_r preserves the entity state norm at all timestamps.

    Args:
        num_relations: number of relation types R
        complex_dim:   half of the full embed_dim (= embed_dim // 2)
    """

    def __init__(self, num_relations: int, complex_dim: int) -> None:
        super().__init__()
        if complex_dim < 1:
            raise ValueError(f"complex_dim must be >= 1, got {complex_dim}")
        self.num_relations = num_relations
        self.complex_dim = complex_dim

        # Learnable raw (asymmetric) matrices; symmetrised at runtime
        # Shape: (R, complex_dim, complex_dim)
        self.A = nn.Parameter(
            torch.randn(num_relations, complex_dim, complex_dim) * 0.02
        )

    def _get_hamiltonians(self) -> torch.Tensor:
        """
        Return skew-symmetric Hamiltonians H_r = (A_r - A_r^T) / 2.

        Skew-symmetric matrices generate the Lie algebra so(d), whose matrix
        exponential exp(H) is orthogonal (O(d)), making them the correct real
        analogue of Hermitian generators for unitary evolution.

        Returns: (R, d, d) where d = complex_dim
        """
        return (self.A - self.A.transpose(-1, -2)) / 2.0    # (R, d, d)  skew-sym

    def get_evolution_matrix(
        self,
        relation_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample Cayley unitary U_r(τ).

        Args:
            relation_ids: (B,) int
            timestamps:   (B,) float, τ values

        Returns:
            U: (B, d, d) real orthogonal matrices
        """
        if relation_ids.shape != timestamps.shape:
            raise ValueError(
                "relation_ids and timestamps must have the same shape."
            )
        B = relation_ids.shape[0]
        d = self.complex_dim
        H_all = self._get_hamiltonians()                     # (R, d, d)
        H = H_all[relation_ids]                              # (B, d, d)
        tau = timestamps.view(B, 1, 1)                       # (B, 1, 1)
        # Cayley: U = (I - iH·τ/2)(I + iH·τ/2)^{-1}
        # For real matrices, "i" corresponds to swapping real/imag blocks.
        # Here we implement the real Cayley map: U = (I - S)(I + S)^{-1}
        # where S = H * τ / 2 (skew-symmetric part is already handled by
        # taking the symmetric H; for the Cayley map we use S directly).
        I = torch.eye(d, device=H.device, dtype=H.dtype).unsqueeze(0)  # (1,d,d)
        S = H * tau / 2.0                                   # (B, d, d)
        lhs = I - S                                          # (B, d, d)
        rhs = I + S                                          # (B, d, d)
        # Solve rhs @ U = lhs  =>  U = rhs^{-1} @ lhs
        U = torch.linalg.solve(rhs, lhs)                    # (B, d, d)
        return U                                             # (B, d, d)

    def apply(
        self,
        state_re: torch.Tensor,
        state_im: torch.Tensor,
        relation_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply U_r(τ) to complex entity state (state_re, state_im).

        For a real Cayley unitary U, the complex multiplication is:
            (re' + i·im') = U(re + i·im)
                          = U·re + i·U·im

        Args:
            state_re:     (B, complex_dim) real part
            state_im:     (B, complex_dim) imaginary part
            relation_ids: (B,) int
            timestamps:   (B,) float

        Returns:
            state_re': (B, complex_dim)
            state_im': (B, complex_dim)
        """
        U = self.get_evolution_matrix(relation_ids, timestamps)  # (B, d, d)
        new_re = torch.bmm(U, state_re.unsqueeze(-1)).squeeze(-1)  # (B, d)
        new_im = torch.bmm(U, state_im.unsqueeze(-1)).squeeze(-1)  # (B, d)
        return new_re, new_im

    def get_param_count(self) -> dict[str, int]:
        return {"A": self.A.numel()}


# ---------------------------------------------------------------------------
# TemporalFactDecay
# ---------------------------------------------------------------------------

class TemporalFactDecay(nn.Module):
    """
    Models fact validity decay over continuous time via phase precession.

    P(h, r, t, τ) = |Σᵢ αᵢ ⟨t|U_r(τ)|h⟩|²

    As τ advances past the fact's validity window, the phase of
    exp(−i H_r τ) precesses past π, causing destructive interference
    with the baseline amplitude at τ=0 and driving the score toward 0.

    The learnable per-relation decay rate ω_r controls how fast the phase
    wraps: score(τ) = base_score · cos²(ω_r · Δτ / 2)

    Args:
        num_relations: R
        hamiltonian:   TemporalHamiltonianOperator (shared weights)
    """

    def __init__(
        self,
        num_relations: int,
        hamiltonian: TemporalHamiltonianOperator,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.hamiltonian = hamiltonian
        # Per-relation angular frequency (governs decay speed)
        self.omega = nn.Parameter(torch.ones(num_relations) * 0.5)

    def decay_score(
        self,
        score_at_t0: torch.Tensor,
        relation_ids: torch.Tensor,
        delta_tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modulate base score by temporal phase factor cos²(ω_r · Δτ / 2).

        When Δτ = π / ω_r the score reaches its first minimum (destructive
        interference), modelling the end of a fact's temporal validity window.

        Args:
            score_at_t0: (B,) base triple score
            relation_ids: (B,) int
            delta_tau:   (B,) time elapsed since fact was established

        Returns:
            decayed_score: (B,) non-negative float
        """
        omega_r = self.omega[relation_ids]                   # (B,)
        phase = omega_r * delta_tau / 2.0                    # (B,)
        decay = torch.cos(phase) ** 2                        # (B,) ∈ [0,1]
        return score_at_t0 * decay

    def get_param_count(self) -> dict[str, int]:
        h_params = self.hamiltonian.get_param_count()
        h_params["omega"] = self.omega.numel()
        return h_params


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B = 5
    R = 10
    complex_dim = 32

    # --- TemporalHamiltonianOperator ---
    op = TemporalHamiltonianOperator(num_relations=R, complex_dim=complex_dim)
    rel_ids = torch.randint(0, R, (B,))
    timestamps = torch.rand(B) * 5.0            # τ ∈ [0, 5]

    U = op.get_evolution_matrix(rel_ids, timestamps)
    assert U.shape == (B, complex_dim, complex_dim), f"Bad shape: {U.shape}"

    state_re = torch.randn(B, complex_dim)
    state_im = torch.randn(B, complex_dim)
    new_re, new_im = op.apply(state_re, state_im, rel_ids, timestamps)
    assert new_re.shape == (B, complex_dim)
    assert new_im.shape == (B, complex_dim)
    print(f"TemporalHamiltonianOperator OK")
    print(f"  U shape:          {U.shape}")
    print(f"  state_re' shape:  {new_re.shape}")
    print(f"  param count:      {op.get_param_count()}")

    # Verify approximate orthogonality: U U^T ≈ I
    eye_approx = torch.bmm(U, U.transpose(-1, -2))
    I = torch.eye(complex_dim).unsqueeze(0).expand(B, -1, -1)
    max_err = (eye_approx - I).abs().max().item()
    print(f"  max |UU^T - I|:   {max_err:.6f}  (should be small)")

    # --- TemporalFactDecay ---
    decay = TemporalFactDecay(num_relations=R, hamiltonian=op)
    base_score = torch.sigmoid(torch.randn(B))
    delta_tau = torch.rand(B) * 10.0
    decayed = decay.decay_score(base_score, rel_ids, delta_tau)
    assert decayed.shape == (B,)
    assert (decayed >= 0).all()
    print(f"TemporalFactDecay OK")
    print(f"  base score:    {base_score.tolist()}")
    print(f"  decayed score: {decayed.tolist()}")
    print(f"  param count:   {decay.get_param_count()}")
    print("All assertions passed.")
