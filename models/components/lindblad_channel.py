"""
models/components/lindblad_channel.py — Lindblad Open Quantum System Dynamics [V6]

PURPOSE:
    Implements continuous-time open quantum system dynamics for multi-hop knowledge
    graph reasoning.  While decoherence.py models quantum noise as a simple
    depolarising channel (mixing the state with I/d), this file models the full
    Lindblad master equation — the most general completely-positive trace-preserving
    (CPTP) evolution for an open quantum system in continuous time.

QUANTUM MECHANICS BACKGROUND:
    A closed quantum system evolves by a unitary U (reversible, no information loss).
    An OPEN system additionally exchanges information with an environment.  The
    Lindblad master equation (Gorini–Kossakowski–Sudarshan–Lindblad, GKSL) describes
    the most general Markovian time evolution of a density matrix ρ:

        dρ/dt = -i[H, ρ]                               ← coherent (unitary) part
              + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})  ← dissipative (jump) part

    where:
        H            — system Hamiltonian (Hermitian operator)
        L_k          — jump operators encoding environment coupling modes
        γ_k ≥ 0     — decay rates (how strongly each mode couples)
        {A, B}       — anti-commutator: AB + BA

    For KG reasoning, each relation r gets its own Hamiltonian H_r (via
    MatrixExpUnitary) and K learnable jump operators L_k^(r) with rates γ_k^(r).

    Per-hop Euler discretisation (step size dt):
        ρ' = ρ + dt × [ -i[H_r, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ}) ]

    After the Euler step, ρ is explicitly renormalised (Tr(ρ) ← 1) to counteract
    any numerical drift in trace preservation.

WHY LINDBLAD OVER SIMPLE DECOHERENCE:
    DecoherenceChannel:   ρ_out = (1-ε) |ψ⟩⟨ψ| + ε I/d
        — fixed isotropic noise; relation-specific only through ε_r
        — cannot capture directional, structured, or relation-specific noise

    LindbladEvolutionStep: full dissipator Σ_k γ_k^(r) (L_k^(r) ρ L_k^(r)† - ½{…})
        — K independent noise channels per relation
        — jump operators L_k^(r) are learned d×d complex matrices
        — can represent amplitude damping, phase damping, erasure, … all at once
        — reduces to depolarising if all L_k are chosen to be Pauli operators
        — strictly more expressive than DecoherenceChannel

    This is analogous to how MatrixExpUnitary (full U(d)) is strictly more
    expressive than DiagonalUnitary (maximal torus T^d ⊊ U(d)).

CLASSES:
    LindbladJumpOperators      — learnable K jump operators per relation, with rates.
    LindbladEvolutionStep      — one Euler step of the Lindblad equation.
    LindbladPathIntegrator     — integrates Lindblad evolution over a multi-hop path.

USAGE:
    from models.components.lindblad_channel import LindbladPathIntegrator
    from models.components.matrix_exp_unitary import MatrixExpUnitary

    unitary    = MatrixExpUnitary(num_relations=237, complex_dim=16)
    integrator = LindbladPathIntegrator(num_relations=237, complex_dim=16,
                                        num_jumps=4, dt=0.1)

    # Initial pure-state density matrix for head entity
    head = entity_states[head_id]                       # (d,) complex
    rho0 = head.unsqueeze(-1) * head.conj().unsqueeze(-2)  # (d, d)

    # Integrate over path [r1, r2, r3]
    rho_final = integrator.integrate(rho0, [r1, r2, r3], unitary, device)

    # Score against tail entity (Born rule for mixed states)
    tail  = entity_states[tail_id]                      # (d,) complex
    score = integrator.score(rho_final, tail)           # real scalar in [0, 1]

    # Add to training loss
    reg_loss = integrator.lindblad_regularization()
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. LindbladJumpOperators
# ---------------------------------------------------------------------------

class LindbladJumpOperators(nn.Module):
    """
    K learnable complex jump operators per relation, with associated decay rates.

    Each relation r has K jump operators L_k^(r) ∈ C^(d×d) and K scalar rates
    γ_k^(r) ∈ [0, 1].  These parameterise the dissipative (environment-coupling)
    part of the Lindblad master equation.

    Parameterisation:
        Operators: stored as real and imaginary parts separately to keep all
                   arithmetic in real-valued parameters for optimiser compatibility.
                   L_k^(r) = L_real[r, k] + i · L_imag[r, k]

        Rates:     raw_rates[r, k] ∈ ℝ (unconstrained logits).
                   γ_k^(r) = softmax(raw_rates[r, :])[k]  ← sums to 1 per relation.
                   Softmax ensures γ_k^(r) ∈ (0, 1) and avoids unbounded growth
                   of any single rate.  Multiply by a global scale if absolute
                   magnitudes matter (they don't for relative inference).

    Initialisation:
        Both L_real and L_imag are initialised with very small values (std=0.01).
        This means at the start of training each L_k ≈ 0, so the dissipative term
        contributes negligibly — training begins near-unitary and gradually learns
        to inject structured noise.

    Args:
        num_relations: Number of distinct KG relation types R.
        complex_dim:   Hilbert space dimension d.
        num_jumps:     Number of jump operators K per relation (default 4).
                       K=4 matches the number of single-qubit Pauli channels for d=2;
                       for higher d, K controls expressiveness vs. parameter count.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        num_jumps:     int = 4,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.complex_dim   = complex_dim
        self.num_jumps     = num_jumps

        d, K, R = complex_dim, num_jumps, num_relations

        # Jump operator real and imaginary parts: (R, K, d, d)
        # Initialised small so the dissipator starts near zero
        self.L_real = nn.Parameter(
            torch.randn(R, K, d, d) * 0.01
        )
        self.L_imag = nn.Parameter(
            torch.randn(R, K, d, d) * 0.01
        )

        # Raw (unconstrained) decay rate logits: (R, K)
        # Initialised to zero → softmax gives uniform γ_k = 1/K at start
        self.raw_rates = nn.Parameter(
            torch.zeros(R, K)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_operators(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Return jump operators for the given relation batch.

        Args:
            rel_ids: (B,) int64 — relation indices.

        Returns:
            (B, K, d, d) complex64 — jump operator tensors L_k^(r).
        """
        L_r = self.L_real[rel_ids]  # (B, K, d, d)
        L_i = self.L_imag[rel_ids]  # (B, K, d, d)
        return torch.complex(L_r, L_i)  # (B, K, d, d) complex64

    def get_rates(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Return normalised decay rates for the given relation batch.

        Applies softmax over the K rates for each relation so that
        Σ_k γ_k^(r) = 1 and γ_k^(r) > 0.

        Args:
            rel_ids: (B,) int64 — relation indices.

        Returns:
            (B, K) float32 — normalised rates γ_k^(r).
        """
        raw = self.raw_rates[rel_ids]          # (B, K)
        return F.softmax(raw, dim=-1)          # (B, K) — sums to 1 per row

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"complex_dim={self.complex_dim}, "
            f"num_jumps={self.num_jumps}"
        )


# ---------------------------------------------------------------------------
# 2. LindbladEvolutionStep
# ---------------------------------------------------------------------------

class LindbladEvolutionStep(nn.Module):
    """
    One Euler step of the Lindblad master equation for a batch of density matrices.

    Given a batch of density matrices ρ (B, d, d), the corresponding unitary
    operators U_r from MatrixExpUnitary (B, d, d), and relation ids (B,), computes:

        dρ/dt = -i[H_r, ρ] + Σ_k γ_k (L_k ρ L_k† - ½ {L_k†L_k, ρ})

    and updates:

        ρ' = ρ + dt × dρ/dt         (Euler step)
        ρ' = ρ' / Tr(ρ')            (renormalise to fix numerical drift)

    Coherent part:
        We express the commutator -i[H, ρ] in terms of the unitary U = exp(iH):
            U ρ U† ≈ ρ - i[H, ρ] dt + O(dt²)
        So for the coherent part we use the first-order approximation:
            coherent = (U ρ U†  - ρ) / dt  ≈ -i[H, ρ]
        This avoids computing H explicitly from U (which would require matrix log)
        and keeps the implementation compatible with any unitary provider.

        Note: for small dt this approximation is accurate (O(dt²) error per step).

    Dissipative part (Lindblad dissipator):
        D(ρ) = Σ_k γ_k (L_k ρ L_k† - ½ {L_k†L_k, ρ})
             = Σ_k γ_k (L_k ρ L_k† - ½ L_k†L_k ρ - ½ ρ L_k†L_k)

    Trace preservation:
        For a valid Lindblad equation dρ/dt has zero trace:
            Tr(-i[H,ρ]) = 0  (always)
            Tr(D(ρ)) = Σ_k γ_k Tr(L_k ρ L_k† - ½{L_k†L_k, ρ}) = 0
        So in exact arithmetic Tr(ρ + dt × dρ/dt) = Tr(ρ).
        Floating-point errors accumulate over many hops, so we add an explicit
        renormalisation step after each Euler update.

    Args:
        num_relations: Number of KG relation types.
        complex_dim:   Hilbert space dimension d.
        num_jumps:     Number of jump operators per relation K (default 4).
        dt:            Euler step size (default 0.1).  Controls the strength of
                       each hop's dissipation.  Smaller dt → more hops needed for
                       the same total evolution; larger dt is faster but less accurate.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        num_jumps:     int = 4,
        dt:            float = 0.1,
    ) -> None:
        super().__init__()
        self.dt = dt

        # Jump operators and rates — the only learnable parameters in this module
        self.jump_ops = LindbladJumpOperators(
            num_relations=num_relations,
            complex_dim=complex_dim,
            num_jumps=num_jumps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dissipator(
        rho:   torch.Tensor,   # (B, d, d) complex
        L:     torch.Tensor,   # (B, K, d, d) complex
        gamma: torch.Tensor,   # (B, K) float
    ) -> torch.Tensor:
        """
        Compute batched Lindblad dissipator D(ρ) = Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ}).

        Args:
            rho:   (B, d, d) complex — current density matrices.
            L:     (B, K, d, d) complex — jump operators.
            gamma: (B, K) float — decay rates.

        Returns:
            (B, d, d) complex — dissipator D(ρ).
        """
        B, K, d, _ = L.shape
        device = rho.device

        # Promote gamma to complex for mixed-precision arithmetic
        gamma_c = gamma.to(dtype=rho.dtype)  # (B, K) complex

        # L†: conjugate transpose over the last two dims
        Ldag = L.conj().transpose(-2, -1)          # (B, K, d, d)

        # L†L: (B, K, d, d)
        LdagL = torch.matmul(Ldag, L)              # (B, K, d, d)

        # Expand rho for K-wise broadcasting: (B, 1, d, d)
        rho_exp = rho.unsqueeze(1)

        # Three terms of the dissipator for each k:
        #   term1: L_k ρ L_k†   shape (B, K, d, d)
        #   term2: ½ L_k†L_k ρ  shape (B, K, d, d)
        #   term3: ½ ρ L_k†L_k  shape (B, K, d, d)
        term1 = torch.matmul(torch.matmul(L, rho_exp), Ldag)   # (B, K, d, d)
        term2 = 0.5 * torch.matmul(LdagL, rho_exp)              # (B, K, d, d)
        term3 = 0.5 * torch.matmul(rho_exp, LdagL)              # (B, K, d, d)

        # Per-k contribution: (term1 - term2 - term3) weighted by gamma_k
        # gamma_c: (B, K) → reshape to (B, K, 1, 1) for broadcasting
        g = gamma_c.view(B, K, 1, 1)
        D_k = g * (term1 - term2 - term3)          # (B, K, d, d)

        # Sum over jump operators
        D = D_k.sum(dim=1)                          # (B, d, d)
        return D

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        rho:            torch.Tensor,   # (B, d, d) complex64
        unitary_states: torch.Tensor,   # (B, d, d) complex64 — U_r from MatrixExpUnitary
        rel_ids:        torch.Tensor,   # (B,) int64
        device:         torch.device,
    ) -> torch.Tensor:
        """
        Apply one Lindblad Euler step to a batch of density matrices.

        Steps:
            1. Coherent part: Δ_coh = (U ρ U† - ρ) / dt   (≈ -i[H, ρ] for small dt)
            2. Dissipative part: Δ_dis = D(ρ) from LindbladJumpOperators
            3. Euler update: ρ' = ρ + dt × (Δ_coh + Δ_dis)
            4. Renormalise: ρ' ← ρ' / Tr(ρ')  (preserves Tr(ρ) = 1)

        Args:
            rho:            (B, d, d) complex64 — input density matrices.
            unitary_states: (B, d, d) complex64 — unitary operators U_r per sample.
            rel_ids:        (B,) int64 — relation indices.
            device:         torch.device — target device.

        Returns:
            rho_next: (B, d, d) complex64 — updated density matrices with Tr = 1.
        """
        rho = rho.to(device)
        U   = unitary_states.to(device)

        # ── 1. Coherent part ──────────────────────────────────────────────
        # U ρ U†: batched matrix product (B, d, d)
        Udag    = U.conj().transpose(-2, -1)         # (B, d, d)
        U_rho_Udag = torch.bmm(torch.bmm(U, rho), Udag)  # (B, d, d)
        # First-order approximation: (U ρ U† - ρ) / dt  ≈  -i[H, ρ]
        delta_coh = (U_rho_Udag - rho) / (self.dt + 1e-12)   # (B, d, d)

        # ── 2. Dissipative part ───────────────────────────────────────────
        L     = self.jump_ops.get_operators(rel_ids)   # (B, K, d, d)
        gamma = self.jump_ops.get_rates(rel_ids)       # (B, K)
        delta_dis = self._dissipator(rho, L, gamma)    # (B, d, d)

        # ── 3. Euler update ────────────────────────────────────────────────
        rho_next = rho + self.dt * (delta_coh + delta_dis)    # (B, d, d)

        # ── 4. Renormalise trace (numerical stability) ────────────────────
        # Tr(ρ) = sum of diagonal, which must be real for Hermitian ρ
        traces = rho_next.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B,) complex
        traces_real = traces.real.clamp(min=1e-8)                 # (B,) float
        # Reshape for broadcasting: (B, 1, 1)
        rho_next = rho_next / traces_real.to(rho_next.dtype).view(-1, 1, 1)

        return rho_next   # (B, d, d) complex64


# ---------------------------------------------------------------------------
# 3. LindbladPathIntegrator
# ---------------------------------------------------------------------------

class LindbladPathIntegrator(nn.Module):
    """
    Integrates Lindblad evolution over a multi-hop reasoning path.

    This is the top-level module for open-quantum-system KG reasoning.  Given
    an initial density matrix ρ_0 (typically |h⟩⟨h| for a head entity),
    a sequence of relation IDs [r_1, r_2, ..., r_K], and a MatrixExpUnitary
    module, it applies LindbladEvolutionStep at each hop to produce the final
    mixed state ρ_K.

    Scoring then uses the Born rule for mixed states:
        P(t | ρ) = Tr(ρ |t⟩⟨t|) = ⟨t|ρ|t⟩

    This generalises the pure-state Born rule |⟨t|ψ⟩|² to mixed states.
    The score is guaranteed real and in [0, 1] for a valid density matrix
    and unit-norm tail state.

    Args:
        num_relations: Number of distinct KG relation types.
        complex_dim:   Hilbert space dimension d.
        num_jumps:     Number of jump operators K per relation (default 4).
        dt:            Euler step size (default 0.1).
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        num_jumps:     int = 4,
        dt:            float = 0.1,
    ) -> None:
        super().__init__()
        self.complex_dim   = complex_dim
        self.num_relations = num_relations
        self.num_jumps     = num_jumps
        self.dt            = dt

        # Single evolution step module (owns LindbladJumpOperators)
        self.step = LindbladEvolutionStep(
            num_relations=num_relations,
            complex_dim=complex_dim,
            num_jumps=num_jumps,
            dt=dt,
        )

    # ------------------------------------------------------------------
    # Core: multi-hop integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        rho_init:       torch.Tensor,       # (d, d) complex64 — initial density matrix
        path_rel_ids:   List[int],          # relation IDs for each hop
        unitary_module: nn.Module,          # MatrixExpUnitary (or compatible)
        device:         torch.device,
    ) -> torch.Tensor:
        """
        Integrate Lindblad evolution over a multi-hop reasoning path.

        For each hop k with relation r_k:
            1. Compute the unitary U_{r_k} from unitary_module.
            2. Apply LindbladEvolutionStep to update ρ_{k-1} → ρ_k.

        The density matrix is kept as a (1, d, d) batch internally and
        squeezed back to (d, d) at the end.

        Args:
            rho_init:       (d, d) complex64 — starting density matrix.
            path_rel_ids:   List of relation IDs [r_1, ..., r_K].
            unitary_module: Module with get_matrix(rel_id) → (d, d) complex.
            device:         Target device.

        Returns:
            rho_final: (d, d) complex64 — density matrix after all hops.
                       Has Tr(ρ) = 1 and is positive semi-definite (approximately).
        """
        if not path_rel_ids:
            # No hops: return the initial density matrix unchanged
            return rho_init.to(device)

        # Work with a (1, d, d) batch throughout the loop
        rho = rho_init.unsqueeze(0).to(device)   # (1, d, d)

        for rel_id in path_rel_ids:
            rel_tensor = torch.tensor([rel_id], dtype=torch.long, device=device)

            # Get the (d, d) unitary for this relation, add batch dim → (1, d, d)
            U_r = unitary_module.get_matrix(rel_id).unsqueeze(0).to(device)

            # One Lindblad Euler step
            rho = self.step(
                rho=rho,
                unitary_states=U_r,
                rel_ids=rel_tensor,
                device=device,
            )  # (1, d, d)

        return rho.squeeze(0)   # (d, d) complex64

    def integrate_batched(
        self,
        rho_batch:      torch.Tensor,       # (B, d, d) complex64
        path_rel_ids:   List[int],          # shared hop sequence (same path for all)
        unitary_module: nn.Module,
        device:         torch.device,
    ) -> torch.Tensor:
        """
        Integrate the same hop sequence for a batch of density matrices.

        Useful when scoring multiple (head, tail) pairs that share the same
        multi-hop relation path (e.g., within a training batch).

        Args:
            rho_batch:    (B, d, d) complex64 — batch of initial density matrices.
            path_rel_ids: List of relation IDs (same path applied to all B samples).
            unitary_module: Module with get_matrix(rel_id) → (d, d) complex.
            device:       Target device.

        Returns:
            (B, d, d) complex64 — batch of final density matrices.
        """
        if not path_rel_ids:
            return rho_batch.to(device)

        B = rho_batch.shape[0]
        rho = rho_batch.to(device)

        for rel_id in path_rel_ids:
            rel_tensor = torch.full((B,), rel_id, dtype=torch.long, device=device)

            # Broadcast U_r to (B, d, d)
            U_r = unitary_module.get_matrix(rel_id).to(device)
            U_r = U_r.unsqueeze(0).expand(B, -1, -1)   # (B, d, d)

            rho = self.step(
                rho=rho,
                unitary_states=U_r,
                rel_ids=rel_tensor,
                device=device,
            )

        return rho   # (B, d, d)

    # ------------------------------------------------------------------
    # Scoring (Born rule for mixed states)
    # ------------------------------------------------------------------

    def score(
        self,
        rho:        torch.Tensor,   # (d, d) complex64
        tail_state: torch.Tensor,   # (d,) complex64
    ) -> torch.Tensor:
        """
        Born rule score for a mixed state: P(t|ρ) = Tr(ρ |t⟩⟨t|) = ⟨t|ρ|t⟩.

        This is the quantum generalisation of the pure-state score |⟨t|ψ⟩|².
        For a valid density matrix (Hermitian, PSD, Tr=1) and unit-norm tail
        state, the result lies in [0, 1].

        Args:
            rho:        (d, d) complex64 — density matrix.
            tail_state: (d,) complex64 — unit-norm tail entity state.

        Returns:
            Scalar float32 tensor — probability-like score in [0, 1].
        """
        # ⟨t|ρ|t⟩ = t† @ ρ @ t  (scalar complex, then real part)
        score = (tail_state.conj() @ rho @ tail_state).real
        return score

    def score_batched(
        self,
        rho_batch:   torch.Tensor,   # (B, d, d) complex64
        tail_states: torch.Tensor,   # (B, d) complex64
    ) -> torch.Tensor:
        """
        Batched Born rule: P(t_b | ρ_b) = ⟨t_b|ρ_b|t_b⟩ for each b.

        Args:
            rho_batch:   (B, d, d) complex64 — batch of density matrices.
            tail_states: (B, d)   complex64 — corresponding tail entity states.

        Returns:
            (B,) float32 tensor of scores.
        """
        # einsum 'bi,bij,bj->b': t† ρ t per batch element
        scores = torch.einsum(
            "bi,bij,bj->b",
            tail_states.conj(),   # (B, d)
            rho_batch,            # (B, d, d)
            tail_states,          # (B, d)
        ).real   # (B,) float32
        return scores

    def score_vs_all(
        self,
        rho:             torch.Tensor,   # (d, d) complex64
        all_tail_states: torch.Tensor,   # (E, d) complex64
    ) -> torch.Tensor:
        """
        Score a single density matrix against all entity states simultaneously.

        Returns ⟨eᵢ|ρ|eᵢ⟩ for every entity i ∈ [0, E).  Used at evaluation
        time to rank all entities as candidate tails for a (head, path, ?) query.

        Args:
            rho:             (d, d) complex64 — final density matrix.
            all_tail_states: (E, d) complex64 — all entity quantum states.

        Returns:
            (E,) float32 tensor of scores, each in [0, 1] for valid inputs.
        """
        scores = torch.einsum(
            "ei,ij,ej->e",
            all_tail_states.conj(),   # (E, d)
            rho,                      # (d, d)
            all_tail_states,          # (E, d)
        ).real   # (E,) float32
        return scores

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def lindblad_regularization(self) -> torch.Tensor:
        """
        Regularisation loss to prevent jump operators from growing unboundedly.

        Computes the mean Frobenius norm of all jump operators across all
        relations and all K channels:

            reg = (1 / (R × K)) × Σ_{r,k} ||L_k^(r)||_F

        where ||L||_F = sqrt(Σ_{ij} |L_{ij}|²).

        This keeps the dissipator bounded, ensures stable Euler steps, and
        prevents the Lindblad dynamics from overwhelming the coherent evolution.
        Add to training loss with a small coefficient (e.g., λ = 1e-4).

        Returns:
            Scalar float32 tensor — mean Frobenius norm of all jump operators.
        """
        L_complex = torch.complex(
            self.step.jump_ops.L_real,   # (R, K, d, d)
            self.step.jump_ops.L_imag,   # (R, K, d, d)
        )
        # Frobenius norm over the last two dims, then mean over all R×K operators
        frob = L_complex.abs().pow(2).sum(dim=(-2, -1)).sqrt()   # (R, K)
        return frob.mean()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def density_matrix_diagnostics(self, rho: torch.Tensor) -> dict:
        """
        Compute quantum information metrics for a single density matrix.

        Useful for monitoring training: purity falling toward 1/d indicates
        over-decoherence; purity near 1 means the model is near-unitary.

        Args:
            rho: (d, d) complex64 — density matrix.

        Returns:
            Dict with keys:
                'trace'   : float — should be 1.0 for a valid ρ
                'purity'  : float — Tr(ρ²) ∈ [1/d, 1]
                'entropy' : float — von Neumann entropy -Tr(ρ log ρ) ∈ [0, log d]
        """
        with torch.no_grad():
            trace   = rho.diagonal().sum().real.item()
            purity  = (rho @ rho).diagonal().sum().real.item()
            eigvals = torch.linalg.eigvalsh(rho).real.clamp(min=1e-12)
            entropy = -(eigvals * torch.log(eigvals)).sum().item()
        return {
            "trace":   trace,
            "purity":  purity,
            "entropy": entropy,
        }

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"complex_dim={self.complex_dim}, "
            f"num_jumps={self.num_jumps}, "
            f"dt={self.dt}"
        )
