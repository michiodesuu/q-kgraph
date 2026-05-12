"""
models/components/quantum_teleportation.py — Quantum Teleportation-Inspired Scoring

PURPOSE:
    Implements a quantum teleportation analogy for knowledge graph triple scoring.
    Classical quantum teleportation transmits an unknown state |ψ⟩ from Alice to
    Bob via a shared Bell pair and classical communication of a measurement outcome.

    Here we adapt this structure for KGE scoring:
        - head entity |h⟩ plays the role of the state to be "teleported"
        - relation M_r plays the role of the entangling channel
        - Generalized Pauli (Weyl) corrections C_k† account for measurement outcomes
        - tail entity |t⟩ is the "receiver" whose overlap with the reconstructed
          state gives the triple score

THE SCORING EQUATION:
    Score(h, r, t) = |Σₖ √q_k · ⟨t|C_k†M_r|h⟩|²

    where:
        M_r     ∈ ℂ^(d×d)  — Frobenius-normalized relation matrix (not constrained unitary)
        C_k     ∈ ℂ^(d×d)  — k-th Weyl/generalized Pauli correction operator
        q_k     ∈ [0,1]    — learned attention weight for outcome k (Σ_k q_k = 1)
        √q_k               — amplitude weight (so q_k enters Born rule as probability)

    The sum-before-squaring allows quantum interference between correction branches:
        = Σₖ q_k |⟨t|C_k†M_r|h⟩|²                 ← classical sum (no interference)
        + Σₖ≠ₗ √(q_k q_l) Re(⟨t|C_k†M_r|h⟩* ⟨t|C_l†M_r|h⟩)  ← interference terms

CLASSES:
    BellStateRelation           — Learnable unconstrained complex matrices M_r per relation
    GeneralizedPauliCorrections — Precomputed Weyl/Pauli correction operators C_{mn}
    DifferentiableBellMeasurement — Learned content-dependent weights q_k over outcomes
    TeleportationScorer         — Full scoring pipeline; fully batched
    EntanglementSwap            — Static utilities for composing relation operators

QUANTUM MECHANICS CONNECTION:
    Weyl operators C_{mn} form a unitary basis for ℂ^(d×d):
        C_{mn}|j⟩ = ω^(nj)|(j+m) mod d⟩,  ω = exp(2πi/d)
    They satisfy Tr(C_{mn}†C_{m'n'}) = d·δ_{mm'}δ_{nn'}, forming a complete
    orthonormal operator basis (operator Fourier transform on ℤ_d × ℤ_d).

    The entanglement entropy S = -Tr(ρ_A log ρ_A) for ρ_A = M†M / Tr(M†M) measures
    how "channel-like" a relation matrix is. High entropy ≈ maximally entangling
    channel (e.g., 'relatedTo'); low entropy ≈ near-unitary, information-preserving.

USAGE:
    scorer = TeleportationScorer(complex_dim=8, num_relations=11)
    scores = scorer.score_triple(head_states, relation_ids, tail_states)  # (B,)
    all_scores = scorer.score_triple_vs_all(head_states, relation_ids, all_tails)  # (B, E)
    analysis = scorer.analyze_outcomes(h, r_id, t)   # dict with quantum diagnostics
"""

from __future__ import annotations

import math
from functools import reduce
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. BellStateRelation
# ---------------------------------------------------------------------------

class BellStateRelation(nn.Module):
    """
    Encodes each relation r as a learnable, Frobenius-normalized complex matrix
    M_r ∈ ℂ^(d×d).

    Unlike unitary parameterizations (which enforce M†M = I), these matrices are
    unconstrained except for Frobenius normalization Tr(M†M) = 1. This makes them
    more expressive: they can represent entangling channels, damping maps, and
    general linear operations on the Hilbert space.

    The Frobenius normalization is applied at property-access time (not stored),
    so gradients flow through it during training.

    Args:
        num_relations: Number of distinct relation types in the KG.
        complex_dim:   Hilbert space dimension d (= embed_dim // 2).
        init_scale:    Std-dev for random initialization of raw parameters.
                       Small values (0.1) keep initial matrices close to zero-mean.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim: int,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.complex_dim = complex_dim

        # Raw parameters stored as float32 with last dim = (real, imag)
        # Shape: (num_relations, d, d, 2)
        raw = torch.randn(num_relations, complex_dim, complex_dim, 2) * init_scale
        self.raw_params = nn.Parameter(raw)

    @property
    def matrices(self) -> torch.Tensor:
        """
        Returns all relation matrices as Frobenius-normalized complex64 tensors.

        Each matrix M_r is normalized so that Tr(M_r†M_r) = ||M_r||_F² = 1.

        Returns:
            Tensor of shape (num_relations, d, d) with dtype complex64.
        """
        # Convert (R, d, d, 2) float32 → (R, d, d) complex64
        M = torch.view_as_complex(self.raw_params.contiguous())  # (R, d, d) complex64

        # Frobenius norm per relation: sqrt(Tr(M†M)) = ||M||_F
        # ||M||_F² = sum of |M_ij|² over all i,j
        frob_sq = (M.abs() ** 2).sum(dim=(-2, -1), keepdim=True)  # (R, 1, 1)
        frob = frob_sq.sqrt().clamp(min=1e-8)
        return M / frob  # Tr(M†M) = 1 per relation

    def get_matrix(self, relation_id: int) -> torch.Tensor:
        """
        Returns the Frobenius-normalized complex matrix for a single relation.

        Args:
            relation_id: Integer index of the relation (0 ≤ relation_id < num_relations).

        Returns:
            Tensor of shape (d, d) with dtype complex64.
        """
        return self.matrices[relation_id]

    def entanglement_entropy(self, relation_id: int) -> float:
        """
        Computes the von Neumann entropy of the density-like operator ρ_A = M†M.

        Since M is Frobenius-normalized, Tr(M†M) = 1, so ρ_A is a valid density
        matrix (positive semidefinite, trace 1). Its eigenvalues λ_i ∈ [0,1] with
        Σ λ_i = 1. The entropy S = -Σ λ_i log(λ_i) ∈ [0, log(d)].

        High entropy means the relation operator is maximally mixing (like a
        depolarizing channel). Low entropy means it is nearly rank-1 / unitary-like.

        Args:
            relation_id: Integer index of the relation.

        Returns:
            Entropy value as a Python float.
        """
        M = self.get_matrix(relation_id)  # (d, d) complex64
        # ρ_A = M†M is Hermitian positive semidefinite
        rho_A = M.conj().T @ M  # (d, d) complex64
        # eigvalsh returns real eigenvalues in ascending order for Hermitian matrix
        eigvals = torch.linalg.eigvalsh(rho_A)  # (d,) real
        eigvals = eigvals.clamp(min=1e-10, max=1.0)
        entropy = -(eigvals * eigvals.log()).sum()
        return entropy.detach().item()

    def entanglement_entropy_all(self) -> torch.Tensor:
        """
        Computes entanglement entropy for all relations in a batched fashion.

        Returns:
            Tensor of shape (num_relations,) with dtype float32.
        """
        M_all = self.matrices  # (R, d, d) complex64
        # ρ_A[r] = M[r]† @ M[r]  — batched: (R, d, d)
        rho_all = M_all.conj().transpose(-2, -1) @ M_all  # (R, d, d)
        # Batched eigenvalue decomposition
        eigvals = torch.linalg.eigvalsh(rho_all)  # (R, d) real
        eigvals = eigvals.clamp(min=1e-10, max=1.0)
        entropy = -(eigvals * eigvals.log()).sum(dim=-1)  # (R,)
        return entropy


# ---------------------------------------------------------------------------
# 2. GeneralizedPauliCorrections
# ---------------------------------------------------------------------------

class GeneralizedPauliCorrections:
    """
    Precomputes the generalized Pauli (Weyl) correction operators for a d-dimensional
    Hilbert space.

    The Weyl operators form a unitary operator basis for ℂ^(d×d):
        C_{mn}|j⟩ = ω^(nj) |(j+m) mod d⟩,   ω = exp(2πi/d)

    where m, n ∈ {0, 1, ..., d-1}. There are d² such operators, forming an
    orthonormal basis under the Hilbert-Schmidt inner product.

    In quantum teleportation, these are the d² possible Pauli corrections that
    Bob applies depending on which of the d² Bell measurement outcomes Alice reports.
    For d=2, these reduce to the standard {I, X, Y, Z} Pauli operators.

    Args:
        complex_dim:     Hilbert space dimension d.
        max_corrections: Maximum number of operators to precompute. If d² > max_corrections,
                         only the first max_corrections operators (in row-major order over
                         m, n ∈ {0,...,d-1}) are used.

    Attributes:
        operators:     (K, d, d) complex64 tensor of precomputed Weyl operators.
        n_corrections: K = min(d², max_corrections).
    """

    def __init__(self, complex_dim: int, max_corrections: int = 16) -> None:
        self.complex_dim = complex_dim
        d = complex_dim
        total = d * d
        K = min(total, max_corrections)
        self.n_corrections = K

        # ω = exp(2πi/d)
        omega = math.tau / d  # 2π/d (real angle)

        # Build all K operators as dense (d, d) complex64 tensors
        # C_{mn}[i, j] = ω^(n*j) if i == (j + m) % d else 0
        operators = torch.zeros(K, d, d, dtype=torch.complex64)

        for idx in range(K):
            m = idx // d
            n = idx % d
            for j in range(d):
                i = (j + m) % d
                phase = math.cos(omega * n * j) + 1j * math.sin(omega * n * j)
                operators[idx, i, j] = phase

        self.operators: torch.Tensor = operators

    def get_all_corrections(self) -> torch.Tensor:
        """
        Returns all precomputed Weyl correction operators.

        Returns:
            Tensor of shape (K, d, d) with dtype complex64, where K = n_corrections.
        """
        return self.operators


# ---------------------------------------------------------------------------
# 3. DifferentiableBellMeasurement
# ---------------------------------------------------------------------------

class DifferentiableBellMeasurement(nn.Module):
    """
    Learns content-dependent attention weights over Bell measurement outcomes.

    In quantum teleportation, the Bell measurement collapses the joint system into
    one of d² orthogonal outcomes with equal classical probability 1/d². Here we
    replace the uniform distribution with learned content-dependent weights q_k,
    conditioned on the head entity state and the relation type.

    This allows the model to up-weight outcomes that lead to constructive
    interference with the tail entity.

    Network architecture:
        input:   [Re(h) ∥ Im(h) ∥ relation_embed]  ∈ ℝ^(2d + hidden_dim//2)
        hidden:  Linear → ReLU
        output:  Linear → softmax    ∈ ℝ^K

    Args:
        complex_dim:   Hilbert space dimension d.
        num_relations: Number of distinct relation types.
        n_corrections: Number of correction operators K.
        hidden_dim:    Width of the attention network (default 64).
    """

    def __init__(
        self,
        complex_dim: int,
        num_relations: int,
        n_corrections: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.complex_dim = complex_dim
        self.n_corrections = n_corrections

        rel_embed_dim = hidden_dim // 2
        self.relation_embed = nn.Embedding(num_relations, rel_embed_dim)

        # Input: real(h) ∥ imag(h) ∥ relation_embed
        # = complex_dim + complex_dim + hidden_dim//2
        input_dim = 2 * complex_dim + rel_embed_dim
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_corrections),
        )

    def compute_weights(
        self,
        head_states: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes softmax attention weights over Bell measurement outcomes.

        Args:
            head_states:  (B, d) complex64 — head entity quantum states.
            relation_ids: (B,) int64 — relation indices.

        Returns:
            Tensor of shape (B, K) float32, softmax-normalized (Σ_k q_k = 1).
        """
        # Decompose complex head state into real and imaginary float parts
        h_real = head_states.real  # (B, d) float32
        h_imag = head_states.imag  # (B, d) float32

        # Relation embedding
        rel_emb = self.relation_embed(relation_ids)  # (B, hidden_dim//2)

        # Concatenate features
        features = torch.cat([h_real, h_imag, rel_emb], dim=-1)  # (B, 2d + hidden_dim//2)

        # Compute logits and apply softmax
        logits = self.attn_net(features)  # (B, K)
        weights = F.softmax(logits, dim=-1)  # (B, K)
        return weights


# ---------------------------------------------------------------------------
# 4. TeleportationScorer
# ---------------------------------------------------------------------------

class TeleportationScorer(nn.Module):
    """
    Core scorer implementing quantum teleportation-inspired triple scoring.

    Scoring equation:
        Score(h, r, t) = |Σₖ √q_k · ⟨t|C_k†M_r|h⟩|²

    where M_r is the relation matrix, C_k are Weyl correction operators,
    and q_k are learned attention weights summing to 1.

    The sum-before-squaring (Born rule after aggregation) allows quantum interference
    between different correction branches. Constructive interference boosts scores for
    valid triples; destructive interference suppresses scores for invalid ones.

    Args:
        complex_dim:   Hilbert space dimension d (= embed_dim // 2).
        num_relations: Number of distinct relation types.
        n_corrections: Number of Weyl correction operators K.
                       Defaults to min(d², 16).
        hidden_dim:    Width of the attention network in DifferentiableBellMeasurement.

    Submodules:
        bell_states:  BellStateRelation — relation matrices M_r.
        corrections:  GeneralizedPauliCorrections — Weyl operators C_k.
        bell_meas:    DifferentiableBellMeasurement — attention weights q_k.
    """

    def __init__(
        self,
        complex_dim: int,
        num_relations: int,
        n_corrections: Optional[int] = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.complex_dim = complex_dim
        self.num_relations = num_relations

        if n_corrections is None:
            n_corrections = min(complex_dim ** 2, 16)
        self.n_corrections = n_corrections

        self.bell_states = BellStateRelation(num_relations, complex_dim)
        self.corrections = GeneralizedPauliCorrections(complex_dim, max_corrections=n_corrections)
        self.bell_meas = DifferentiableBellMeasurement(
            complex_dim, num_relations, n_corrections, hidden_dim
        )

    def score_triple(
        self,
        head_states: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes teleportation-inspired scores for a batch of (h, r, t) triples.

        Fully batched — no Python loops over the batch dimension.

        Steps:
            1. Retrieve M_r for each sample in the batch.
            2. Retrieve all Weyl correction operators.
            3. Compute effective operators E_k = C_k† @ M_r for each (k, batch) pair.
            4. Apply to head state: evolved_{bk} = E_k |h_b⟩.
            5. Compute amplitudes: a_{bk} = ⟨t_b|evolved_{bk}⟩.
            6. Compute attention weights q_{bk} via DifferentiableBellMeasurement.
            7. Weighted coherent sum: total_b = Σ_k √q_{bk} · a_{bk}.
            8. Born rule: score_b = |total_b|².

        Args:
            head_states:  (B, d) complex64 — head entity states, unit-norm.
            relation_ids: (B,) int64 — relation indices.
            tail_states:  (B, d) complex64 — tail entity states, unit-norm.

        Returns:
            Tensor of shape (B,) float32 — scores in [0, 1+1e-6].
        """
        # Step 1: Relation matrices for this batch
        # (B, d, d) complex64
        M_batch = self.bell_states.matrices[relation_ids]

        # Step 2 & 3: Weyl corrections and their adjoints
        # C_all: (K, d, d) complex64
        C_all = self.corrections.get_all_corrections().to(head_states.device)
        # C_adj[k] = C_k†: (K, d, d) complex64
        C_adj = C_all.conj().transpose(-2, -1)

        # Step 4: Effective operators E_{bk} = C_k† @ M_r_b
        # einsum: for each b and k, E[b,k,i,l] = Σ_j C_adj[k,i,j] * M_batch[b,j,l]
        # E_all: (B, K, d, d) complex64
        E_all = torch.einsum("kij,bjl->bkil", C_adj, M_batch)

        # Step 5: Apply to head states: h_evolved[b,k,i] = Σ_j E_all[b,k,i,j] * h[b,j]
        # h_evolved: (B, K, d) complex64
        h_evolved = torch.einsum("bkij,bj->bki", E_all, head_states)

        # Step 6: Inner product with tail states: ⟨t_b|evolved_{bk}⟩
        # amplitudes[b,k] = Σ_i t[b,i]* · h_evolved[b,k,i]
        # amplitudes: (B, K) complex64
        amplitudes = torch.einsum("bi,bki->bk", tail_states.conj(), h_evolved)

        # Step 7: Compute attention weights
        # weights: (B, K) float32
        weights = self.bell_meas.compute_weights(head_states, relation_ids)

        # Step 8: Amplitude-weighted coherent sum
        # √q_k scales amplitude, so |Σ √q_k a_k|² = Born rule after interference
        w_complex = weights.sqrt().to(torch.complex64)  # (B, K) complex64
        # total[b] = Σ_k √q_{bk} · a_{bk}
        total = (w_complex * amplitudes).sum(dim=-1)  # (B,) complex64

        # Step 9: Born rule — clamp to handle floating-point overflow
        scores = total.abs().pow(2).clamp(0.0, 1.0 + 1e-6)  # (B,) float32
        return scores

    def score_triple_vs_all(
        self,
        head_states: torch.Tensor,
        relation_ids: torch.Tensor,
        all_tail_states: torch.Tensor,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """
        Scores all entities as tails against the given (head, relation) batch.

        Uses chunking over the entity dimension to avoid O(B·E·K·d²) memory peaks.

        Args:
            head_states:     (B, d) complex64 — head entity states.
            relation_ids:    (B,) int64 — relation indices.
            all_tail_states: (E, d) complex64 — all entity states as candidate tails.
            chunk_size:      Number of tail entities processed per chunk.

        Returns:
            Tensor of shape (B, E) float32 — scores in [0, 1+1e-6].
        """
        B = head_states.shape[0]
        E = all_tail_states.shape[0]
        device = head_states.device

        # Precompute quantities shared across all tail chunks
        # M_batch: (B, d, d)
        M_batch = self.bell_states.matrices[relation_ids]

        # C_adj: (K, d, d)
        C_all = self.corrections.get_all_corrections().to(device)
        C_adj = C_all.conj().transpose(-2, -1)

        # E_all: (B, K, d, d) — effective operators
        E_all = torch.einsum("kij,bjl->bkil", C_adj, M_batch)

        # h_evolved: (B, K, d) — evolved head states under each correction
        h_evolved = torch.einsum("bkij,bj->bki", E_all, head_states)

        # weights: (B, K) — attention weights
        weights = self.bell_meas.compute_weights(head_states, relation_ids)
        w_complex = weights.sqrt().to(torch.complex64)  # (B, K)

        # Weighted evolved states: (B, K, d) → weighted_h[b,k,i] = √q[b,k] * h_evolved[b,k,i]
        weighted_h = w_complex.unsqueeze(-1) * h_evolved  # (B, K, d)

        # Aggregate over k: coherent_h[b, i] = Σ_k √q[b,k] * h_evolved[b,k,i]
        # This is the "reconstructed" state at Bob's side
        coherent_h = weighted_h.sum(dim=1)  # (B, d)

        # Score against all tail states in chunks
        scores_list: list[torch.Tensor] = []
        for chunk_start in range(0, E, chunk_size):
            chunk_end = min(chunk_start + chunk_size, E)
            tail_chunk = all_tail_states[chunk_start:chunk_end]  # (C, d)

            # Amplitude: ⟨t_e|coherent_h_b⟩
            # amp[b, e] = Σ_i tail_chunk[e,i]* · coherent_h[b,i]
            amp_chunk = torch.einsum("ei,bi->be", tail_chunk.conj(), coherent_h)  # (B, C)

            # Born rule
            score_chunk = amp_chunk.abs().pow(2).clamp(0.0, 1.0 + 1e-6)  # (B, C)
            scores_list.append(score_chunk)

        return torch.cat(scores_list, dim=-1)  # (B, E)

    def analyze_outcomes(
        self,
        head_state: torch.Tensor,
        relation_id: int,
        tail_state: torch.Tensor,
    ) -> dict:
        """
        Performs a detailed single-sample analysis of the teleportation scoring.

        Decomposes the total score into per-outcome amplitudes and weights, and
        quantifies the quantum interference contribution relative to the classical
        (incoherent) sum.

        Args:
            head_state:  (d,) complex64 — head entity state (no batch dimension).
            relation_id: Integer relation index.
            tail_state:  (d,) complex64 — tail entity state (no batch dimension).

        Returns:
            dict with keys:
                'outcome_amplitudes':  list of K complex numbers ⟨t|C_k†M_r|h⟩
                'outcome_weights':     list of K floats q_k (softmax-normalized)
                'interference_gain':   float = total_score - classical_sum
                                       Positive → constructive interference boosted score.
                                       Negative → destructive interference reduced score.
                'constructive_count':  int — number of outcomes with positive Re contribution
                'destructive_count':   int — number of outcomes with negative Re contribution
                'entanglement_entropy': float — S(ρ_A) for the relation matrix M_r
        """
        device = head_state.device

        # Add batch dimension for shared computation
        h = head_state.unsqueeze(0)  # (1, d)
        t = tail_state.unsqueeze(0)  # (1, d)
        r_ids = torch.tensor([relation_id], dtype=torch.long, device=device)

        # Retrieve relation matrix and corrections
        M = self.bell_states.get_matrix(relation_id)  # (d, d)
        C_all = self.corrections.get_all_corrections().to(device)  # (K, d, d)
        C_adj = C_all.conj().transpose(-2, -1)  # (K, d, d)

        # Compute per-outcome amplitudes: a_k = ⟨t|C_k†M_r|h⟩
        # Apply M_r to head: Mh = M @ h  → (d,)
        Mh = M @ head_state  # (d,)
        # Apply each C_k† to Mh: C_adj[k] @ Mh → (K, d)
        C_Mh = torch.einsum("kij,j->ki", C_adj, Mh)  # (K, d)
        # Inner product with tail: a_k = ⟨t|C_k†M_r|h⟩
        amplitudes = torch.einsum("i,ki->k", tail_state.conj(), C_Mh)  # (K,) complex

        # Compute attention weights
        with torch.no_grad():
            weights = self.bell_meas.compute_weights(h, r_ids).squeeze(0)  # (K,)

        # Weighted coherent sum
        w_complex = weights.sqrt().to(torch.complex64)  # (K,)
        total = (w_complex * amplitudes).sum()  # scalar complex
        total_score = total.detach().abs().pow(2).item()

        # Classical incoherent sum: Σ_k q_k |a_k|²
        classical_sum = (weights * amplitudes.detach().abs().pow(2)).sum().item()

        # Interference gain
        interference_gain = total_score - classical_sum

        # Per-outcome contributions to the total amplitude's real part
        # Contribution of outcome k: √q_k * a_k contributes to total = Σ_k √q_k a_k
        # The real contribution of each term to the final score is:
        # 2 * Re(√q_k * a_k * (total - √q_k a_k)*)  (approximation via full total)
        # Simpler: check Re(√q_k * a_k) relative to mean
        per_outcome_real = (w_complex * amplitudes).real.detach()  # (K,)
        mean_contribution = per_outcome_real.mean().item()
        constructive_count = int((per_outcome_real > mean_contribution).sum().item())
        destructive_count = int((per_outcome_real <= mean_contribution).sum().item())

        # Entanglement entropy for this relation
        entropy = self.bell_states.entanglement_entropy(relation_id)

        return {
            "outcome_amplitudes": [a.detach().item() for a in amplitudes],
            "outcome_weights": weights.tolist(),
            "interference_gain": interference_gain,
            "constructive_count": constructive_count,
            "destructive_count": destructive_count,
            "entanglement_entropy": entropy,
        }


# ---------------------------------------------------------------------------
# 5. EntanglementSwap
# ---------------------------------------------------------------------------

class EntanglementSwap:
    """
    Static utilities for composing relation operators via entanglement swapping.

    In quantum networks, entanglement swapping extends entanglement across nodes
    by performing a Bell measurement on intermediate particles and applying
    corrections. For knowledge graph reasoning, this corresponds to multi-hop
    path reasoning: the operator for a 2-hop path (h→r1→e→r2→t) is obtained
    by composing the individual relation operators.

    The intermediate Bell measurement can produce d² outcomes, each yielding
    a different effective long-range operator. Summing over these outcomes
    (weighted by probabilities) gives the effective 2-hop channel.

    All methods are static — this class has no learnable parameters.
    """

    @staticmethod
    def swap(M_r1: torch.Tensor, M_r2: torch.Tensor) -> torch.Tensor:
        """
        Composes two relation operators via entanglement swap (matrix product).

        The effective operator for the path h→r1→e→r2→t is M_r2 @ M_r1,
        applied right-to-left: first M_r1 acts on |h⟩, then M_r2 acts on the result.

        Args:
            M_r1: (d, d) complex64 — first relation's matrix.
            M_r2: (d, d) complex64 — second relation's matrix.

        Returns:
            (d, d) complex64 — composed operator M_r2 @ M_r1.
        """
        return M_r2 @ M_r1

    @staticmethod
    def path_operator(matrices: list[torch.Tensor]) -> torch.Tensor:
        """
        Computes the composed operator for a multi-hop path.

        For a path with relations r_1, r_2, ..., r_n (left to right),
        the effective operator is M_{r_n} @ ... @ M_{r_2} @ M_{r_1}
        (rightmost matrix applied first to the state).

        Args:
            matrices: List of (d, d) complex64 tensors, ordered left-to-right
                      along the path (first relation first).

        Returns:
            (d, d) complex64 — product of all matrices, applied right-to-left.

        Raises:
            ValueError: If matrices is empty.
        """
        if not matrices:
            raise ValueError("matrices must be a non-empty list of tensors.")
        # reduce with rightmost first: M_n @ ... @ M_1
        # We iterate left-to-right but accumulate as M_new = M_current @ M_acc
        # so the final result is M_n @ M_{n-1} @ ... @ M_1
        return reduce(lambda acc, M: M @ acc, matrices)

    @staticmethod
    def swap_with_intermediate(
        M_r1: torch.Tensor,
        M_r2: torch.Tensor,
        intermediate_weights: torch.Tensor,
        corrections: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes all effective 2-hop operators resulting from a Bell measurement
        on the intermediate entity, weighted by outcome probabilities.

        In entanglement swapping, a Bell measurement on the intermediate particle
        yields one of n_outcomes results. Each result k requires applying correction
        C_k† to the long-range channel. The k-th effective operator is:
            E_k = M_r2 @ C_k† @ M_r1

        Args:
            M_r1:                  (d, d) complex64 — first relation matrix.
            M_r2:                  (d, d) complex64 — second relation matrix.
            intermediate_weights:  (n_outcomes,) float — probability of each
                                   Bell measurement outcome (should sum to 1).
            corrections:           (n_outcomes, d, d) complex64 — Weyl correction
                                   operators C_k.

        Returns:
            (n_outcomes, d, d) complex64 — effective operators E_k = M_r2 @ C_k† @ M_r1.
            Note: the weights are NOT folded in; caller should use them as q_k in
            Score = |Σ_k √q_k ⟨t|E_k|h⟩|².
        """
        n_outcomes = corrections.shape[0]

        # C_k†: (n_outcomes, d, d)
        C_adj = corrections.conj().transpose(-2, -1)

        # Intermediate operators: C_k† @ M_r1 for each k
        # → (n_outcomes, d, d)
        C_M1 = torch.einsum("kij,jl->kil", C_adj, M_r1)

        # Full 2-hop operators: M_r2 @ (C_k† @ M_r1)
        # → (n_outcomes, d, d)
        E_all = torch.einsum("ij,kjl->kil", M_r2, C_M1)

        return E_all
