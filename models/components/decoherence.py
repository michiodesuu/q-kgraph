"""
models/components/decoherence.py — Quantum Decoherence Channels for KG Reasoning

PURPOSE:
    Models the physical process of quantum decoherence in multi-hop knowledge graph
    reasoning. In real quantum systems, interaction with an environment causes a
    pure quantum state |ψ⟩ to lose its coherence and become a statistical mixture,
    described by a density matrix ρ instead of a state vector.

    For knowledge graphs, this captures the intuition that reasoning chains
    become "noisier" and less certain over longer paths. After a 3-hop chain,
    confidence should be lower than after a 1-hop link — and this uncertainty
    should be represented not just as a scalar weight, but as a fundamental
    change in the mathematical object (state vector → density matrix).

QUANTUM MECHANICS BACKGROUND:
    Pure state:  |ψ⟩ ∈ ℂ^d  — described by a state vector (zero uncertainty)
    Mixed state: ρ ∈ ℂ^(d×d) — Hermitian, PSD, Tr(ρ)=1 (statistical mixture)

    The depolarising decoherence channel maps:
        ρ_k = (1 - ε)^k |ψ_k⟩⟨ψ_k| + (1 - (1 - ε)^k) · I/d

    where:
        ε ∈ [0, 1]  — per-hop decoherence rate (0 = pure, 1 = fully mixed)
        k           — number of hops / path length
        I/d         — maximally mixed state (uniform probability over all states)

    After k hops, the cumulative decoherence factor (1 - ε)^k shrinks the
    coherent component. Longer paths therefore produce flatter, higher-entropy
    distributions — faithfully representing increased reasoning uncertainty.

SCORING WITH DENSITY MATRICES:
    For a mixed state ρ and a candidate tail entity |t⟩, the Born rule
    generalises naturally (Postulate 3 for mixed states):

        P(t | ρ) = ⟨t|ρ|t⟩ = Tr(ρ |t⟩⟨t|)

    This reduces to the pure-state score |⟨t|ψ⟩|² when ρ = |ψ⟩⟨ψ|,
    and gives a lower, flatter distribution when ρ → I/d (maximum uncertainty).

MULTI-PATH AGGREGATION (Incoherent Mixture):
    For K reasoning paths, each path i produces a density matrix ρᵢ.
    The paths are combined as a classical (incoherent) mixture:

        ρ_total = Σᵢ wᵢ · ρᵢ,   Σᵢ wᵢ = 1,  wᵢ ≥ 0

    Unlike AmplitudeAggregator (which sums complex amplitudes coherently),
    this model cleanly separates QUANTUM decoherence WITHIN each path from
    CLASSICAL uncertainty BETWEEN paths. This is physically motivated:
    the choice of path is a classical probability, while the state evolution
    along a path is quantum.

CLASSES:
    DecoherenceChannel          — Applies per-relation depolarising channel.
    DensityMatrixScorer         — Static-method scorer: Tr(ρ|t⟩⟨t|), purity, entropy.
    DecoherencePathAggregator   — Full pipeline: multi-path mixture scoring.
    DecoherenceRateScheduler    — Anneals decoherence rates toward zero during training.

USAGE:
    channel = DecoherenceChannel(complex_dim=8, num_relations=11, init_rate=0.3)
    rho = channel.apply_decoherence_batched(evolved_states, relation_ids)  # (B, d, d)
    score = DensityMatrixScorer.score(rho[0], tail_state)                  # scalar
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# A path is a list of (relation_id, entity_id) hops
Path = list[tuple[int, int]]


# ---------------------------------------------------------------------------
# 1. DecoherenceChannel
# ---------------------------------------------------------------------------

class DecoherenceChannel(nn.Module):
    """
    Per-relation depolarising decoherence channel.

    After a unitary hop along relation r, the resulting pure state |ψ'⟩ is
    partially mixed with the maximally mixed state I/d at a learned rate ε_r:

        ρ_out = (1 - ε_r) |ψ'⟩⟨ψ'| + ε_r · (I/d)

    The decoherence rate ε_r ∈ (0, 1) is parameterised via a logit:
        ε_r = sigmoid(log_rate_r)

    so that ε_r is always in the open interval (0, 1) regardless of the
    magnitude of the learned logit. This avoids boundary issues during optimisation.

    Physics interpretation:
        ε_r ≈ 0 : relation r is coherence-preserving (e.g., a confident direct link)
        ε_r ≈ 1 : relation r induces maximal uncertainty (e.g., a long-range,
                  ambiguous or noisy relation)

    Over k hops with the same rate ε, the cumulative coherent fraction is (1-ε)^k,
    so longer paths naturally decohere more — without any explicit path-length
    penalty needing to be hand-engineered.

    Args:
        complex_dim:       Hilbert space dimension d. Must equal embed_dim // 2.
        num_relations:     Number of distinct KG relation types R.
        learnable_rates:   If True, ε_r are learned parameters (default).
                           If False, rates are fixed buffers at init_rate.
        init_rate:         Initial decoherence rate in (0, 1). Converted to
                           the corresponding logit via inverse-sigmoid so that
                           sigmoid(logit) ≈ init_rate at initialisation.
    """

    def __init__(
        self,
        complex_dim: int,
        num_relations: int,
        learnable_rates: bool = True,
        init_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.complex_dim = complex_dim
        self.num_relations = num_relations
        self.learnable_rates = learnable_rates

        # Inverse-sigmoid: log(p / (1 - p))
        init_logit = math.log(init_rate / (1.0 - init_rate))

        if learnable_rates:
            # (R,) float32 — optimiser will update these
            self.log_rates = nn.Parameter(
                torch.full((num_relations,), fill_value=init_logit)
            )
        else:
            # Fixed rates stored as a buffer (not a parameter)
            self.register_buffer(
                "log_rates",
                torch.full((num_relations,), fill_value=init_logit),
            )

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def rates(self) -> torch.Tensor:
        """
        Per-relation decoherence rates ε_r = sigmoid(log_rates_r) ∈ (0, 1).

        Returns:
            Tensor of shape (num_relations,) float32.
        """
        return torch.sigmoid(self.log_rates)

    # ------------------------------------------------------------------
    # Single-sample (unbatched) operations
    # ------------------------------------------------------------------

    def apply_decoherence(
        self,
        pure_state: torch.Tensor,
        relation_id: int,
    ) -> torch.Tensor:
        """
        Applies the depolarising channel to a single pure state.

        Computes:
            ρ_out = (1 - ε) |ψ⟩⟨ψ| + ε · (I/d)

        Args:
            pure_state:  (d,) complex64 — a single, unit-norm pure state.
            relation_id: Integer index selecting ε_r from self.rates.

        Returns:
            (d, d) complex64 density matrix. Satisfies ρ ≥ 0, Tr(ρ) = 1
            for a unit-norm input.
        """
        d = self.complex_dim
        device = pure_state.device

        # ρ_pure = |ψ⟩⟨ψ| via outer product: (d,) ⊗ (d,)* → (d, d)
        rho_pure = pure_state.unsqueeze(-1) * pure_state.conj().unsqueeze(-2)
        # Shape: (d, d) complex64

        eps = self.rates[relation_id]  # scalar float32

        # Maximally mixed state: I/d
        I_d = torch.eye(d, dtype=torch.complex64, device=device) / d
        # Shape: (d, d)

        # Depolarising mixture
        eps_c = eps.to(torch.complex64)
        rho_out = (1.0 - eps_c) * rho_pure + eps_c * I_d
        # Shape: (d, d) complex64

        return rho_out

    def evolve_pure_then_decohere(
        self,
        state: torch.Tensor,
        unitary_op: nn.Module,
        relation_id: int,
    ) -> torch.Tensor:
        """
        Applies a unitary evolution followed by the decoherence channel.

        Steps:
            1. Evolve: |ψ'⟩ = U_r |ψ⟩  (via unitary_op.apply)
            2. Decohere: ρ = (1-ε)|ψ'⟩⟨ψ'| + ε·I/d

        The unitary_op.apply interface expects batched inputs (B, d), so
        this method adds and removes the batch dimension automatically.

        Args:
            state:       (d,) complex64 — single pure state |ψ⟩.
            unitary_op:  A module with ``apply(states, relation_ids)`` method.
                         Must match the UnitaryOperator ABC from unitary_operators.py.
            relation_id: Integer relation index for both the unitary and the
                         decoherence rate.

        Returns:
            (d, d) complex64 density matrix ρ after evolution and decoherence.
        """
        device = state.device

        # Add batch dimension → (1, d), call apply, remove → (d,)
        rel_tensor = torch.tensor([relation_id], dtype=torch.long, device=device)
        state_evolved = unitary_op.apply(state.unsqueeze(0), rel_tensor).squeeze(0)
        # Shape: (d,) complex64

        # Normalise for numerical safety (unitaries should preserve norm exactly,
        # but floating-point drift can accumulate over many hops)
        norm = state_evolved.abs().pow(2).sum().sqrt().clamp(min=1e-8)
        state_evolved = state_evolved / norm

        return self.apply_decoherence(state_evolved, relation_id)

    def compose_decoherence(
        self,
        rho_in: torch.Tensor,
        pure_state_next: torch.Tensor,
        relation_id: int,
    ) -> torch.Tensor:
        """
        Chains decoherence: blends an existing density matrix with the next
        decohered pure state to produce the multi-hop mixed state.

        For a state that has already passed through some hops (represented by
        ρ_in), and a new hop that produces pure evolved state |ψ_next⟩, the
        combined output is:

            ρ_out = (1 - ε) |ψ_next⟩⟨ψ_next| + ε · (ρ_in + I/d) / 2

        Physical interpretation:
            - The coherent part tracks the latest evolved pure state |ψ_next⟩,
              weighted by the surviving coherence fraction (1-ε).
            - The incoherent part is an equal mixture of the prior mixed state
              ρ_in (carrying accumulated path noise) and the maximally mixed
              state I/d (representing environmental decoherence at this hop).

        This is a practical approximation that avoids the full density-matrix
        evolution U ρ U† (which would require matrix products over (d,d) matrices
        at every hop). For diagnostic and composition purposes it is sufficient.

        Args:
            rho_in:          (d, d) complex64 — density matrix from previous hops.
            pure_state_next: (d,) complex64 — evolved pure state at the current hop.
            relation_id:     Integer index selecting ε_r.

        Returns:
            (d, d) complex64 — updated density matrix after this decoherence step.
        """
        d = self.complex_dim
        device = rho_in.device

        eps = self.rates[relation_id]
        eps_c = eps.to(torch.complex64)

        # Pure-state component: |ψ_next⟩⟨ψ_next|
        rho_pure_next = (
            pure_state_next.unsqueeze(-1) * pure_state_next.conj().unsqueeze(-2)
        )
        # Shape: (d, d) complex64

        # Maximally mixed state: I/d
        I_d = torch.eye(d, dtype=torch.complex64, device=device) / d

        # Blend: coherent part + incoherent mix of prior state and max-mixed
        rho_out = (
            (1.0 - eps_c) * rho_pure_next
            + eps_c * (rho_in + I_d) / 2.0
        )
        # Shape: (d, d) complex64

        return rho_out

    # ------------------------------------------------------------------
    # Batched operations
    # ------------------------------------------------------------------

    def apply_decoherence_batched(
        self,
        pure_states: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched depolarising channel: applies per-sample relation-specific rates.

        Computes for each b in [0, B):
            ρ_b = (1 - ε_{r_b}) |ψ_b⟩⟨ψ_b| + ε_{r_b} · (I/d)

        Args:
            pure_states:  (B, d) complex64 — batch of unit-norm pure states.
            relation_ids: (B,) int64 — one relation index per sample.

        Returns:
            (B, d, d) complex64 — batch of density matrices.
        """
        B, d = pure_states.shape
        device = pure_states.device

        # ρ_pure[b, i, j] = ψ_b[i] * conj(ψ_b[j])
        rho_pure = torch.einsum("bi,bj->bij", pure_states, pure_states.conj())
        # Shape: (B, d, d) complex64

        # Per-sample decoherence rates: ε_batch[b] = ε_{r_b}
        eps_batch = self.rates[relation_ids]  # (B,) float32

        # Maximally mixed state: I/d, broadcast to (B, d, d)
        I_d = torch.eye(d, dtype=torch.complex64, device=device) / d
        I_d = I_d.unsqueeze(0)  # (1, d, d)

        # Depolarising mixture with per-sample rates
        # Reshape eps for broadcasting: (B, 1, 1)
        eps_c = eps_batch.to(torch.complex64).view(B, 1, 1)
        rho_out = (1.0 - eps_c) * rho_pure + eps_c * I_d
        # Shape: (B, d, d) complex64

        return rho_out

    # ------------------------------------------------------------------
    # nn.Module utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"complex_dim={self.complex_dim}, "
            f"num_relations={self.num_relations}, "
            f"learnable_rates={self.learnable_rates}"
        )


# ---------------------------------------------------------------------------
# 2. DensityMatrixScorer
# ---------------------------------------------------------------------------

class DensityMatrixScorer:
    """
    Stateless scorer: computes Born-rule probabilities from density matrices.

    For a density matrix ρ and a candidate tail entity |t⟩, the generalised
    Born rule gives:

        P(t | ρ) = ⟨t|ρ|t⟩ = Tr(ρ |t⟩⟨t|)
                 = Σᵢⱼ ρ_ij t_j* t_i        (matrix element form)
                 = (t† ρ t).real             (vector-matrix-vector form)

    This reduces to the standard Born rule |⟨t|ψ⟩|² when ρ = |ψ⟩⟨ψ| (pure state),
    and returns 1/d for all t when ρ = I/d (maximally mixed state).

    The result is guaranteed real and non-negative by the positive semi-definiteness
    of valid density matrices.

    This class is NOT an nn.Module — it has no parameters and all methods are
    static, making it lightweight to call anywhere in the pipeline.

    Quantum information diagnostics also provided:
        trace(ρ)              — should equal 1 for a valid density matrix
        purity(ρ)             — Tr(ρ²) ∈ [1/d, 1]; 1=pure, 1/d=maximally mixed
        von_neumann_entropy(ρ) — -Tr(ρ log ρ) ∈ [0, log d]; 0=pure, log(d)=mixed
    """

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    @staticmethod
    def score(
        rho: torch.Tensor,
        tail_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Born-rule score for a single (ρ, t) pair.

        Computes P(t|ρ) = (t† ρ t).real = Σᵢⱼ ρ_ij t_j* t_i.

        Args:
            rho:        (d, d) complex64 — a single density matrix. Must be
                        Hermitian, positive semi-definite, and trace-1 for the
                        result to be a valid probability in [0, 1].
            tail_state: (d,) complex64 — unit-norm candidate tail entity state.

        Returns:
            Scalar float32 tensor. In [0, 1] for a valid density matrix and
            unit-norm tail state.
        """
        # t† ρ t : scalar complex, then take real part
        score = (tail_state.conj() @ rho @ tail_state).real
        return score

    @staticmethod
    def score_batched(
        rho_batch: torch.Tensor,
        tail_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Born-rule scores for a batch of (ρ_b, t_b) pairs.

        Computes P(t_b | ρ_b) = Σᵢⱼ ρ_b_ij t_b_j* t_b_i for each b.

        Args:
            rho_batch:   (B, d, d) complex64 — batch of density matrices.
            tail_states: (B, d) complex64 — corresponding tail entity states.

        Returns:
            (B,) float32 tensor of scores, each in [0, 1] for valid inputs.
        """
        # einsum: 'bi,bij,bj->b' computes t† ρ t per batch element
        scores = torch.einsum(
            "bi,bij,bj->b",
            tail_states.conj(),  # (B, d)
            rho_batch,           # (B, d, d)
            tail_states,         # (B, d)
        ).real  # (B,) float32
        return scores

    @staticmethod
    def score_vs_all(
        rho: torch.Tensor,
        all_tail_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Born-rule score of a single density matrix against all entity states.

        Computes Tr(ρ |eᵢ⟩⟨eᵢ|) = ⟨eᵢ|ρ|eᵢ⟩ for every entity i ∈ [0, E).

        Useful at evaluation time to rank all entities as candidate tails
        for a single (source, relation, ?) query.

        Args:
            rho:             (d, d) complex64 — a single density matrix.
            all_tail_states: (E, d) complex64 — all entity quantum states.

        Returns:
            (E,) float32 tensor of scores, each in [0, 1] for valid inputs.
        """
        # einsum: 'ei,ij,ej->e' computes ⟨eᵢ|ρ|eᵢ⟩ for each entity i
        scores = torch.einsum(
            "ei,ij,ej->e",
            all_tail_states.conj(),  # (E, d)
            rho,                     # (d, d)
            all_tail_states,         # (E, d)
        ).real  # (E,) float32
        return scores

    # ------------------------------------------------------------------
    # Quantum information diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def trace(rho: torch.Tensor) -> torch.Tensor:
        """
        Returns Tr(ρ) — should equal 1 for a valid density matrix.

        Args:
            rho: (d, d) complex64.

        Returns:
            Scalar float32 tensor (real part of the complex trace).
        """
        return rho.diagonal().sum().real

    @staticmethod
    def purity(rho: torch.Tensor) -> torch.Tensor:
        """
        Computes the purity P = Tr(ρ²) ∈ [1/d, 1].

        Interpretation:
            P = 1    : pure state (ρ = |ψ⟩⟨ψ|, zero entropy)
            P = 1/d  : maximally mixed state (ρ = I/d, maximum entropy)
            P ∈ (1/d, 1) : partially decohered state

        The purity is a scalar measure of how far the state is from a
        classical probability distribution. It does NOT require diagonalisation
        (unlike von Neumann entropy), making it cheap to compute.

        Args:
            rho: (d, d) complex64.

        Returns:
            Scalar float32 tensor.
        """
        # Tr(ρ²) = Σᵢⱼ ρ_ij ρ_ji = ||ρ||_F²
        return (rho @ rho).diagonal().sum().real

    @staticmethod
    def von_neumann_entropy(rho: torch.Tensor) -> torch.Tensor:
        """
        Computes the von Neumann entropy S(ρ) = -Tr(ρ log ρ) ∈ [0, log d].

        Interpretation:
            S = 0      : pure state (zero uncertainty)
            S = log(d) : maximally mixed state (maximum uncertainty)

        The von Neumann entropy is the quantum generalisation of Shannon entropy.
        It requires eigendecomposition (O(d³)) and is more expensive than purity.
        Use purity for cheap monitoring; use entropy for precise information measures.

        Numerically: S = -Σᵢ λᵢ log(λᵢ), where λᵢ are the eigenvalues of ρ.
        Eigenvalues are guaranteed real and non-negative for Hermitian PSD matrices;
        small negative values from floating-point error are clamped to zero.

        Args:
            rho: (d, d) complex64.

        Returns:
            Scalar float32 tensor (entropy in nats).
        """
        # Eigendecomposition — eigvalsh assumes Hermitian (more stable than eig)
        eigenvalues = torch.linalg.eigvalsh(rho).real  # (d,) float32
        # Clamp to avoid log(0) or log of tiny negatives from numerical error
        eigenvalues = eigenvalues.clamp(min=1e-12)
        # -Σ λᵢ log(λᵢ)
        entropy = -(eigenvalues * torch.log(eigenvalues)).sum()
        return entropy


# ---------------------------------------------------------------------------
# 3. DecoherencePathAggregator
# ---------------------------------------------------------------------------

class DecoherencePathAggregator(nn.Module):
    """
    Multi-path decoherence-aware scoring via incoherent density matrix mixture.

    Replaces AmplitudeAggregator for settings where decoherence along reasoning
    paths is the dominant modelling concern rather than pure-state interference.

    DESIGN PHILOSOPHY — Quantum vs. Classical Mixing:
        AmplitudeAggregator: coherent superposition ACROSS paths (interference).
            Score = |Σᵢ αᵢ ⟨t|Uᵢ|s⟩|²   ← cross-terms produce interference

        DecoherencePathAggregator: quantum decoherence WITHIN each path +
            classical (incoherent) mixing ACROSS paths:
            ρᵢ     = decoherence_along_path_i(|s⟩)  ← quantum within-path noise
            ρ_total = Σᵢ |αᵢ|² ρᵢ                  ← classical between-path mix
            Score  = ⟨t|ρ_total|t⟩

        This design is physically motivated: the choice of which path to follow
        is a classical uncertainty (no quantum superposition between paths),
        while the noise encountered ALONG a path is a quantum decoherence effect.

    TRAINING DYNAMICS:
        At initialisation, decoherence rates are moderate (init_rate ≈ 0.3),
        so ρᵢ are well-mixed and scores are smooth — easy to optimise.
        Pair with DecoherenceRateScheduler to anneal rates toward zero,
        progressively sharpening the model toward near-pure-state predictions.

    Args:
        complex_dim:     Hilbert space dimension d = embed_dim // 2.
        num_relations:   Number of distinct KG relation types.
        max_paths:       Maximum paths per query to aggregate. Extra paths
                         found by PathEnumerator are truncated.
        learnable_rates: Passed to DecoherenceChannel (see that class).
        init_rate:       Initial decoherence rate (see DecoherenceChannel).
    """

    def __init__(
        self,
        complex_dim: int,
        num_relations: int,
        max_paths: int = 8,
        learnable_rates: bool = True,
        init_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.complex_dim = complex_dim
        self.num_relations = num_relations
        self.max_paths = max_paths

        # Owned DecoherenceChannel — learns per-relation ε_r
        self.decoherence_channel = DecoherenceChannel(
            complex_dim=complex_dim,
            num_relations=num_relations,
            learnable_rates=learnable_rates,
            init_rate=init_rate,
        )

        # Learned path mixture weights (real); softmax-normalised at forward time.
        # Shape: (max_paths,) float32
        self.path_weights = nn.Parameter(torch.ones(max_paths))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evolve_path_with_decoherence(
        self,
        source_state: torch.Tensor,
        path: Path,
        unitary: nn.Module,
    ) -> torch.Tensor:
        """
        Evolves a pure state along a single reasoning path with per-hop decoherence.

        For each hop (rel_id, _) in the path:
            1. Apply unitary U_r to the current (pure) state representation.
            2. Apply decoherence channel to get the hop's density matrix.
            3. For subsequent hops, compose with the running density matrix via
               compose_decoherence to accumulate noise over the full path.

        Args:
            source_state: (d,) complex64 — initial pure entity state |s⟩.
            path:         List of (relation_id, entity_id) tuples. Entity IDs
                          are ignored here (only relation IDs govern evolution).
            unitary:      Module with ``apply(states, relation_ids)`` method.

        Returns:
            (d, d) complex64 density matrix representing the path's mixed state.
            Returns the pure-state density matrix |s⟩⟨s| if path is empty.
        """
        d = self.complex_dim
        device = source_state.device

        if not path:
            # No hops: return the pure-state density matrix of the source
            rho = source_state.unsqueeze(-1) * source_state.conj().unsqueeze(-2)
            return rho  # (d, d) complex64

        current_pure = source_state.clone()  # (d,) complex64
        rho: Optional[torch.Tensor] = None

        for hop_idx, (rel_id, _entity_id) in enumerate(path):
            rel_tensor = torch.tensor([rel_id], dtype=torch.long, device=device)

            # Evolve current pure state through unitary
            evolved = unitary.apply(
                current_pure.unsqueeze(0), rel_tensor
            ).squeeze(0)  # (d,) complex64

            # Normalise (numerical safety)
            norm = evolved.abs().pow(2).sum().sqrt().clamp(min=1e-8)
            evolved = evolved / norm

            if hop_idx == 0:
                # First hop: pure decoherence on the evolved state
                rho = self.decoherence_channel.apply_decoherence(evolved, rel_id)
                # Shape: (d, d) complex64
            else:
                # Subsequent hops: compose with running density matrix
                assert rho is not None
                rho = self.decoherence_channel.compose_decoherence(
                    rho_in=rho,
                    pure_state_next=evolved,
                    relation_id=rel_id,
                )
                # Shape: (d, d) complex64

            # Prepare next iteration: extract a representative pure state
            # from the diagonal of rho (amplitude = sqrt(diagonal probability))
            diag_real = rho.diagonal().real.clamp(min=0.0)  # (d,) float32
            norm_diag = diag_real.sum().sqrt().clamp(min=1e-8)
            current_pure = (diag_real / norm_diag).sqrt().to(torch.complex64)
            # Shape: (d,) complex64 (real amplitudes — phase partially lost,
            # but this is acceptable since decoherence washes out phase anyway)

        assert rho is not None
        return rho

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source_states: torch.Tensor,
        target_states: torch.Tensor,
        paths_batch: list[list[Path]],
        unitary: nn.Module,
    ) -> torch.Tensor:
        """
        Computes decoherence-aware scores for a batch of (source, target) pairs.

        For each sample b:
            1. For each path p ≤ max_paths:
               - Evolve source_states[b] along path p with per-hop decoherence
                 → density matrix ρ_{b,p}.
            2. Compute softmax-normalised mixture weights w_p = softmax(path_weights)[p].
            3. Aggregate: ρ_total_b = Σ_p w_p ρ_{b,p}  (incoherent mixture).
            4. Score: Score_b = ⟨target_b|ρ_total_b|target_b⟩.

        If a sample has no paths, the score falls back to 1/d (maximally uncertain).

        Args:
            source_states: (B, d) complex64 — source entity quantum states.
            target_states: (B, d) complex64 — target entity quantum states.
            paths_batch:   Length-B list of per-sample path lists. Each path is
                           a list of (relation_id, entity_id) tuples.
            unitary:       UnitaryOperator module for state evolution.

        Returns:
            (B,) float32 tensor of scores in [0, 1].
        """
        B = source_states.shape[0]
        d = self.complex_dim
        device = source_states.device

        # Normalise path weights once per forward call
        w_norm = F.softmax(self.path_weights, dim=0)  # (max_paths,) float32

        scores = torch.zeros(B, dtype=torch.float32, device=device)

        for b in range(B):
            paths_b = paths_batch[b][:self.max_paths]  # truncate to max_paths
            n_paths = len(paths_b)

            if n_paths == 0:
                # No paths found: maximally uncertain score
                scores[b] = 1.0 / d
                continue

            # Accumulate incoherent mixture of path density matrices
            rho_total = torch.zeros(d, d, dtype=torch.complex64, device=device)
            weight_sum = 0.0

            for p_idx, path in enumerate(paths_b):
                # Evolve source state along this path with decoherence
                rho_p = self._evolve_path_with_decoherence(
                    source_state=source_states[b],
                    path=path,
                    unitary=unitary,
                )  # (d, d) complex64

                w_p = w_norm[p_idx]  # scalar float32
                rho_total = rho_total + w_p.to(torch.complex64) * rho_p
                weight_sum = weight_sum + w_p.item()

            # Renormalise if fewer than max_paths paths were found
            # (partial softmax weight sum < 1 otherwise)
            if weight_sum > 1e-8:
                rho_total = rho_total / weight_sum

            # Score: ⟨t|ρ_total|t⟩
            scores[b] = DensityMatrixScorer.score(rho_total, target_states[b])

        return scores.clamp(min=0.0, max=1.0 + 1e-6)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_diagnostics(
        self,
        source_state: torch.Tensor,
        target_state: torch.Tensor,
        paths: list[Path],
        unitary: nn.Module,
    ) -> dict:
        """
        Single-sample diagnostic: per-path quantum information metrics.

        Computes per-path density matrices and extracts purity, scores, and
        effective decoherence rates. Useful for understanding how much decoherence
        the model applies to each reasoning path and how it affects the score.

        The 'effective_decoherence_rates' are inferred from per-path purity:
            Tr(ρ²) = (1-ε)² + (1-(1-ε)²) * (1/d)
            → solving for ε given measured purity.
            (Approximate for multi-hop paths; exact for single-hop.)

        Also computes the 'classical_limit_score': what the score would be
        if all paths were maximally decohered (ε=1, ρ = I/d), giving a
        lower bound corresponding to uniform entity scoring.

        Args:
            source_state: (d,) complex64 — single source entity state.
            target_state: (d,) complex64 — single target entity state.
            paths:        List of Path objects for this (source, target) pair.
            unitary:      UnitaryOperator module.

        Returns:
            Dict with keys:
                'path_purities'              : list[float] — Tr(ρᵢ²) per path
                'path_scores'                : list[float] — ⟨t|ρᵢ|t⟩ per path
                'effective_decoherence_rates': list[float] — inferred ε per path
                'total_score'                : float — final aggregated score
                'classical_limit_score'      : float — score if ε=1 (uniform ρ=I/d)
        """
        d = self.complex_dim
        device = source_state.device

        paths_k = paths[:self.max_paths]
        n_paths = len(paths_k)

        if n_paths == 0:
            return {
                "path_purities": [],
                "path_scores": [],
                "effective_decoherence_rates": [],
                "total_score": 1.0 / d,
                "classical_limit_score": 1.0 / d,
            }

        w_norm = F.softmax(self.path_weights[:n_paths], dim=0)  # (n_paths,)

        path_purities: list[float] = []
        path_scores: list[float] = []
        eff_rates: list[float] = []
        rho_total = torch.zeros(d, d, dtype=torch.complex64, device=device)

        with torch.no_grad():
            for p_idx, path in enumerate(paths_k):
                rho_p = self._evolve_path_with_decoherence(
                    source_state=source_state,
                    path=path,
                    unitary=unitary,
                )  # (d, d) complex64

                purity_p = DensityMatrixScorer.purity(rho_p).item()
                score_p = DensityMatrixScorer.score(rho_p, target_state).item()

                # Infer effective ε from measured purity:
                #   purity = (1-ε)² + (1-(1-ε)²) / d
                #   Let x = (1-ε)². Then: purity = x + (1-x)/d = x(1 - 1/d) + 1/d
                #   → x = (purity - 1/d) / (1 - 1/d)
                #   → ε = 1 - sqrt(max(x, 0))
                if d > 1:
                    x = (purity_p - 1.0 / d) / (1.0 - 1.0 / d)
                    x = max(0.0, min(1.0, x))
                    eff_eps = 1.0 - math.sqrt(x)
                else:
                    eff_eps = 0.0

                path_purities.append(purity_p)
                path_scores.append(score_p)
                eff_rates.append(eff_eps)

                w_p = w_norm[p_idx].to(torch.complex64)
                rho_total = rho_total + w_p * rho_p

            # Normalise total density matrix (in case n_paths < max_paths)
            weight_sum = w_norm.sum().item()
            if weight_sum > 1e-8:
                rho_total = rho_total / weight_sum

            total_score = DensityMatrixScorer.score(rho_total, target_state).item()

            # Classical limit: replace all ρᵢ with I/d → score = 1/d
            classical_limit_score = 1.0 / d

        return {
            "path_purities": path_purities,
            "path_scores": path_scores,
            "effective_decoherence_rates": eff_rates,
            "total_score": total_score,
            "classical_limit_score": classical_limit_score,
        }

    def extra_repr(self) -> str:
        return (
            f"complex_dim={self.complex_dim}, "
            f"num_relations={self.num_relations}, "
            f"max_paths={self.max_paths}"
        )


# ---------------------------------------------------------------------------
# 4. DecoherenceRateScheduler
# ---------------------------------------------------------------------------

class DecoherenceRateScheduler:
    """
    Anneals decoherence rates from high (noisy/mixed) toward low (coherent/pure).

    MOTIVATION:
        Starting training with high decoherence rates ε ≈ init_rate gives the
        model smooth, well-mixed density matrices that are easy to optimise
        (less sharp probability distributions → smoother loss landscape).
        Over training, annealing ε → final_rate causes the model to progressively
        sharpen toward near-pure-state predictions, ultimately approaching the
        classical interference regime of AmplitudeAggregator.

        This curriculum from "classical-like" to "quantum" mirrors techniques
        in quantum annealing where a system transitions from easy-to-sample
        thermal states to quantum ground states.

    SCHEDULE OPTIONS:
        'exponential': ε(k) = ε_f + (ε_0 - ε_f) · exp(-5k / K)
            Fast initial decay, then slow approach to ε_f. Preferred because it
            front-loads exploration (high ε early) and smoothly plateaus.

        'linear': ε(k) = ε_0 - (ε_0 - ε_f) · min(k / K, 1)
            Constant decay rate. Simpler; useful for ablations.

    USAGE:
        scheduler = DecoherenceRateScheduler(
            channel=my_decoherence_channel,
            init_rate=0.3, final_rate=0.01,
            anneal_epochs=100, schedule='exponential'
        )
        for epoch in range(num_epochs):
            train_one_epoch(...)
            scheduler.anneal(epoch)

    Args:
        channel:       DecoherenceChannel instance to manage.
        init_rate:     Starting decoherence rate ε₀ ∈ (0, 1).
        final_rate:    Target decoherence rate ε_f ∈ (0, ε₀).
        anneal_epochs: Number of epochs over which to anneal (K in the formulas).
        schedule:      'exponential' (default) or 'linear'.
    """

    def __init__(
        self,
        channel: DecoherenceChannel,
        init_rate: float = 0.3,
        final_rate: float = 0.01,
        anneal_epochs: int = 100,
        schedule: str = "exponential",
    ) -> None:
        if schedule not in ("exponential", "linear"):
            raise ValueError(
                f"Unknown schedule '{schedule}'. Choose 'exponential' or 'linear'."
            )
        if not (0.0 < final_rate < init_rate < 1.0):
            raise ValueError(
                f"Required: 0 < final_rate ({final_rate}) < init_rate ({init_rate}) < 1."
            )

        self.channel = channel
        self.init_rate = init_rate
        self.final_rate = final_rate
        self.anneal_epochs = anneal_epochs
        self.schedule = schedule

    # ------------------------------------------------------------------
    # Core annealing step
    # ------------------------------------------------------------------

    def _compute_rate(self, epoch: int) -> float:
        """
        Compute the target decoherence rate ε at the given epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Target rate ε ∈ [final_rate, init_rate].
        """
        if self.schedule == "exponential":
            # ε(k) = ε_f + (ε_0 - ε_f) · exp(-5k / K)
            decay = math.exp(-5.0 * epoch / max(self.anneal_epochs, 1))
            rate = self.final_rate + (self.init_rate - self.final_rate) * decay
        else:
            # ε(k) = ε_0 - (ε_0 - ε_f) · min(k / K, 1)
            frac = min(epoch / max(self.anneal_epochs, 1), 1.0)
            rate = self.init_rate - (self.init_rate - self.final_rate) * frac

        # Clamp to [final_rate, init_rate] for numerical safety
        return float(max(self.final_rate, min(self.init_rate, rate)))

    def anneal(self, epoch: int) -> None:
        """
        Updates the DecoherenceChannel's log_rates to match the scheduled ε.

        Computes the target rate ε(epoch), converts to logit = log(ε/(1-ε)),
        and sets all log_rates to this value (broadcast to all relations).

        Note: This unconditionally overwrites the current logits. If the
        optimiser has also updated the logits during this epoch, those
        gradient-based updates are discarded. For best results, call anneal()
        AFTER the optimiser step.

        Args:
            epoch: Current epoch index (0-indexed).
        """
        target_rate = self._compute_rate(epoch)

        # Inverse-sigmoid: convert rate to logit, with clamping to avoid ±inf
        target_logit = math.log(
            max(target_rate, 1e-6) / max(1.0 - target_rate, 1e-6)
        )

        with torch.no_grad():
            self.channel.log_rates.data.fill_(target_logit)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_current_rates(self, epoch: int) -> dict:
        """
        Returns a summary of the current decoherence rate status at a given epoch.

        Reports both the scheduled target and the actual (possibly gradient-
        modified) rates in the channel.

        Args:
            epoch: Current epoch index (0-indexed).

        Returns:
            Dict with keys:
                'mean_rate'         : float — mean ε across all relations (actual)
                'min_rate'          : float — min ε across all relations (actual)
                'max_rate'          : float — max ε across all relations (actual)
                'schedule_fraction' : float — fraction through anneal schedule in [0,1]
                'scheduled_rate'    : float — target ε at this epoch from schedule
        """
        actual_rates = self.channel.rates.detach().cpu()

        schedule_fraction = min(epoch / max(self.anneal_epochs, 1), 1.0)

        return {
            "mean_rate": float(actual_rates.mean().item()),
            "min_rate": float(actual_rates.min().item()),
            "max_rate": float(actual_rates.max().item()),
            "schedule_fraction": schedule_fraction,
            "scheduled_rate": self._compute_rate(epoch),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """
        Returns a dict capturing the scheduler's configuration for checkpointing.

        Note: The channel's actual log_rates are saved separately via
        channel.state_dict() (standard PyTorch checkpointing). This method
        saves only the scheduler hyperparameters (which are not nn.Parameters).

        Returns:
            Dict with keys: 'init_rate', 'final_rate', 'anneal_epochs', 'schedule'.
        """
        return {
            "init_rate": self.init_rate,
            "final_rate": self.final_rate,
            "anneal_epochs": self.anneal_epochs,
            "schedule": self.schedule,
        }

    def load_state_dict(self, d: dict) -> None:
        """
        Restores scheduler configuration from a checkpointed state dict.

        Args:
            d: Dict as returned by state_dict().

        Raises:
            KeyError: If required keys are missing from d.
        """
        self.init_rate = d["init_rate"]
        self.final_rate = d["final_rate"]
        self.anneal_epochs = d["anneal_epochs"]
        self.schedule = d["schedule"]

        if self.schedule not in ("exponential", "linear"):
            raise ValueError(
                f"Loaded invalid schedule '{self.schedule}'. "
                "Expected 'exponential' or 'linear'."
            )
