"""
models/components/matrix_product_state.py — MPS Tensor Networks for KG Reasoning [V6]

PURPOSE:
    Implements Matrix Product State (MPS) tensor networks for scalable multi-hop
    knowledge graph reasoning WITHOUT explicit path enumeration.

    Standard multi-hop KG reasoning requires enumerating all paths of length K
    between a head and tail entity — exponential in K.  MPS tensor networks
    provide a compact representation where the amplitude for ALL paths of a given
    length is computed by a single sequence of matrix contractions, inheriting
    the polynomial efficiency of tensor network methods from quantum many-body physics.

TENSOR NETWORK BACKGROUND:
    An MPS (also called a "tensor train") represents a high-rank tensor as a
    product of low-rank 3-index cores.  For KG reasoning, each relation r gets
    a core tensor A[r] ∈ C^(d × χ × d), where:
        d — physical (embedding) dimension
        χ — bond (virtual) dimension; controls expressiveness vs. cost

    A K-hop path (r_1, r_2, ..., r_K) contributes an amplitude:

        amp(h, r_1,...,r_K, t) = ⟨t| [ Σ_{α_1,...,α_{K-1}}
            A[r_1]_{:,α_1,:} A[r_2]_{:,α_2,:} ... A[r_K]_{:,:,:}
        ] |h⟩

    equivalently written as the contraction:

        v_L^T · A[r_1] · A[r_2] · ... · A[r_K] · v_R

    where v_L, v_R ∈ C^χ are learned boundary vectors that project from bond
    space to a scalar amplitude (after projecting head/tail states into the
    physical indices).

    The bond dimension χ controls the expressiveness:
        χ = 1 : fully separable (product state), O(d) parameters per relation
        χ = d : maximally expressive, O(d³) parameters per relation

WHY MPS OVER PATH ENUMERATION:
    Path enumeration:   exponential in K, infeasible for K > 3 in large KGs
    MPS contraction:    O(K × χ² × d²) — polynomial in all quantities
    Infinite-hop:       Neumann series (I - λM)^{-1} computes Σ_K λ^K M^K exactly

    The MPS representation also naturally captures path ORDER (non-commutativity
    of the matrix product) unlike scalar or diagonal representations.

CLASSES:
    MPSRelationTensor      — learnable MPS core per relation A[r] ∈ C^(d×χ×d).
    MPSPathContractor      — contracts MPS cores along a path for KG scoring.
    InfiniteHopAggregator  — aggregates over ALL paths of ALL lengths via Neumann series.

USAGE:
    from models.components.matrix_product_state import MPSPathContractor, InfiniteHopAggregator

    contractor  = MPSPathContractor(num_relations=237, complex_dim=16, bond_dim=16)
    aggregator  = InfiniteHopAggregator(num_relations=237, complex_dim=16, bond_dim=16)

    # Score a specific 3-hop path
    score = contractor.contract_path(head_state, [r1, r2, r3], tail_state, device)

    # Score over all paths of all lengths (Neumann series)
    agg_state = aggregator.aggregate(head_state, adj, contractor.mps, device)
    score = (tail_state.conj() @ agg_state).real
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. MPSRelationTensor
# ---------------------------------------------------------------------------

class MPSRelationTensor(nn.Module):
    """
    Learnable MPS core tensor per relation: A[r] ∈ C^(d × χ × d).

    Indexing convention: A[r]_{i, α, j} where
        i — input physical index  (entity embedding dimension d)
        α — bond (virtual) index  (bond dimension χ)
        j — output physical index (entity embedding dimension d)

    This can be viewed as a χ-wide "transition matrix" in both the physical
    and bond spaces simultaneously:
        For each bond index α, A[r][:, α, :] is a d×d complex matrix.
        Contracting over α with neighbouring cores couples physical dimensions.

    Boundary vectors v_L, v_R ∈ C^χ:
        v_L projects a head entity state (in bond space) onto the left boundary.
        v_R projects the final accumulated bond vector onto a scalar amplitude.
        Both are shared across all relations (global boundary conditions).

    Initialisation:
        A_real, A_imag: small random (std=0.02) initialisation.
        Boundary vectors: uniform 1/sqrt(χ) — equal weight on all bond indices.

        The small initialisation for A means the path amplitudes start small and
        stable; the optimiser grows them as needed.

    Args:
        num_relations: Number of distinct KG relation types R.
        complex_dim:   Physical (entity embedding) dimension d.
        bond_dim:      Bond (virtual) dimension χ (default 16).
                       χ=16 gives a good expressiveness/cost tradeoff for d=8–32.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        bond_dim:      int = 16,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.complex_dim   = complex_dim
        self.bond_dim      = bond_dim

        d, chi, R = complex_dim, bond_dim, num_relations

        # Core tensors: A_real, A_imag ∈ R^(R × d × χ × d)
        # Small init so path amplitudes start near zero (numerically stable)
        self.A_real = nn.Parameter(
            torch.randn(R, d, chi, d) * 0.02
        )
        self.A_imag = nn.Parameter(
            torch.randn(R, d, chi, d) * 0.02
        )

        # Boundary vectors: v_L, v_R ∈ C^χ  (global, shared across relations)
        # Initialise to uniform 1/sqrt(χ) so all bond indices start equally weighted
        init_val = 1.0 / math.sqrt(chi)
        self.v_L_real = nn.Parameter(torch.full((chi,), init_val))
        self.v_L_imag = nn.Parameter(torch.zeros(chi))
        self.v_R_real = nn.Parameter(torch.full((chi,), init_val))
        self.v_R_imag = nn.Parameter(torch.zeros(chi))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tensor(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Return MPS core tensors for a batch of relations.

        Args:
            rel_ids: (B,) int64 — relation indices.

        Returns:
            (B, d, χ, d) complex64 — core tensors A[r] for each relation in the batch.
        """
        A_r = self.A_real[rel_ids]   # (B, d, chi, d)
        A_i = self.A_imag[rel_ids]   # (B, d, chi, d)
        return torch.complex(A_r, A_i)   # (B, d, chi, d) complex64

    def get_tensor_single(self, rel_id: int) -> torch.Tensor:
        """
        Return the MPS core tensor for a single relation.

        Args:
            rel_id: int — relation index.

        Returns:
            (d, χ, d) complex64 — core tensor A[rel_id].
        """
        A_r = self.A_real[rel_id]   # (d, chi, d)
        A_i = self.A_imag[rel_id]   # (d, chi, d)
        return torch.complex(A_r, A_i)   # (d, chi, d) complex64

    def get_boundaries(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return boundary vectors v_L and v_R.

        Returns:
            v_L: (χ,) complex64 — left boundary vector.
            v_R: (χ,) complex64 — right boundary vector.
        """
        v_L = torch.complex(self.v_L_real, self.v_L_imag)   # (chi,) complex64
        v_R = torch.complex(self.v_R_real, self.v_R_imag)   # (chi,) complex64
        return v_L, v_R

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"complex_dim={self.complex_dim}, "
            f"bond_dim={self.bond_dim}"
        )


# ---------------------------------------------------------------------------
# 2. MPSPathContractor
# ---------------------------------------------------------------------------

class MPSPathContractor(nn.Module):
    """
    Contracts MPS tensors along a reasoning path to compute a path amplitude.

    For a K-hop path with relations [r_1, r_2, ..., r_K], the amplitude is:

        amp(h, r_1,...,r_K, t) = v_L^† (Σ_{i,j} h_i A[r_1]_{i,α_1,j})
                                     (Σ_{j,k} …  A[r_2]_{j,α_2,k})
                                     ...
                                     (Σ  …        A[r_K]_{...,j}) v_R h_j ... t_j

    More concisely, contracting all indices sequentially:

        Algorithm:
            1. Project head into bond space via left boundary:
               bond_vec = v_L  (shape: χ)
               physical_in = h  (shape: d)
            2. For each hop k with relation r_k:
               bond_vec = einsum('i, iαj, α → j', physical_in, A[r_k], bond_vec)
               but we track both physical and bond simultaneously.
            3. Score: ⟨t|result|t⟩ using right boundary.

    The actual contraction order used here:
        - Start with bond vector α = v_L  (χ,)
        - For each hop k:
            α_new[β] = Σ_{i,α} h_i A[r_k]_{i,α,β} α[α]
                      (contract physical with head, bond with α)
            In practice: α_new = einsum('iαβ, i, α → β', A[r_k], h, α)
            But after the first hop, h is replaced by the propagated physical state.
        - Actually the cleanest interpretation: the MPS maps (physical_in, bond_in)
          → (bond_out), and the output physical index is contracted with the tail.
        - Full contraction: scalar = ⟨t|[Σ_α v_L[α] · A[r_1][:, α, :] · A[r_2] · ...]|h⟩

    Implementation follows "left-to-right" bond contraction:
        left = v_L                                # (χ,)
        left = einsum('α, i, iαβ → β', left, h, A[r_1])  # contract physical h, bond
        for k=2..K:
            left = einsum('α, iαβ → iβ', left, A[r_k])   # propagate bond only
        score = left · v_R · t                            # project to scalar

    Wait — the physical dimension needs careful treatment.  Here we implement
    a cleaner version where the physical index is contracted at each step:

        left[α] = v_L[α]               # χ
        left[α] = Σ_{β,i} left[β] A[r_1]_{i,β,α} h[i]   # χ: contracted with h once
        for k=2..K:
            # No more physical contraction — propagate bond only
            # But we need a physical index at each step ...

    The physically correct interpretation for KG:
        The "in" physical index of A[r_k] = the embedding space passed from
        the previous entity, and "out" = embedding space of the next entity.
        Contracting with h at the first step and with t at the last step gives
        a scalar amplitude that scores the entire path (h, r_1,...,r_K, t).

    Final contraction (this implementation):
        state = h                                 # (d,) current "physical" state
        bond  = v_L                               # (χ,) current bond vector
        for A[r_k] in path:
            # A[r_k]: (d, χ, d)
            # contract: new_state[j] = Σ_{i,α} state[i] · A[r_k][i,α,j] · bond[α]
            new_state = einsum('i, iaj, a -> j', state, A[r_k], bond)
            bond = v_L  # reset bond (option A) or keep propagating (option B)

        Actually the cleanest and most powerful option is to carry the bond:
            bond[β] = Σ_{i,α} state[i] · A[r_k][i,α,β] · bond[α]  — no output physical
        but then we need to project onto the output physical at the end.

    This implementation uses the fully-contracted version where at each hop
    the bond AND the physical state are both propagated:
        Effective matrix per hop: M[r_k] = einsum('iaj->j', A[r_k] * bond[:, α])
    yielding a d-dimensional output state after each hop.

    See contract_path() for the actual implementation.

    Args:
        num_relations: Number of KG relation types.
        complex_dim:   Physical dimension d.
        bond_dim:      Bond dimension χ (default 16).
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        bond_dim:      int = 16,
    ) -> None:
        super().__init__()
        self.complex_dim   = complex_dim
        self.bond_dim      = bond_dim
        self.num_relations = num_relations

        # The MPS core tensors and boundaries live here
        self.mps = MPSRelationTensor(
            num_relations=num_relations,
            complex_dim=complex_dim,
            bond_dim=bond_dim,
        )

    # ------------------------------------------------------------------
    # Internal: single-sample contraction
    # ------------------------------------------------------------------

    def _contract_single(
        self,
        head_state:   torch.Tensor,   # (d,) complex
        path_rel_ids: List[int],
        tail_state:   torch.Tensor,   # (d,) complex
        device:       torch.device,
    ) -> torch.Tensor:
        """
        Contract MPS tensors along a single path.

        Contraction algorithm (bond + physical propagation):

            bond  = v_L                              # (χ,) current bond vector
            phys  = head_state                       # (d,) current physical state
            for each relation r_k in path:
                A = A[r_k]                           # (d, χ, d)
                # Map (phys, bond) → new bond: new_bond[β] = Σ_{i,α} phys[i] A[i,α,β] bond[α]
                new_bond = einsum('i, iab, a -> b', phys, A, bond)   # (χ,)
                # Map (phys, bond) → new phys: new_phys[j] = Σ_{i,α} phys[i] A[i,α,j] bond[α]
                #   (here α is summed with bond, i with phys, j is the output)
                # In the standard MPS view the physical output is the right-side index.
                # We marginalise over bond here too to get a d-vector for the next step.
                new_phys = einsum('i, iaj, a -> j', phys, A, bond)   # (d,)
                bond = new_bond
                phys = new_phys (normalised for stability)
            score = ⟨tail|phys⟩ * (bond · v_R)     # complex scalar

        This propagates BOTH a "bond message" (χ-dimensional) AND a "physical message"
        (d-dimensional) through the path. The bond captures long-range correlations
        across hops; the physical captures the entity-space trajectory.

        The amplitude is the product of the physical inner product ⟨t|phys⟩ and
        the bond inner product ⟨v_R|bond⟩.

        Args:
            head_state:   (d,) complex — head entity state.
            path_rel_ids: List[int] — relation IDs for each hop.
            tail_state:   (d,) complex — tail entity state.
            device:       Target device.

        Returns:
            Complex scalar amplitude (differentiable).
        """
        head = head_state.to(device)
        tail = tail_state.to(device)

        if not path_rel_ids:
            # Zero-hop: direct inner product ⟨t|h⟩
            return (tail.conj() @ head)

        v_L, v_R = self.mps.get_boundaries()
        v_L = v_L.to(device)
        v_R = v_R.to(device)

        bond = v_L.clone()            # (χ,) complex
        phys = head.clone()           # (d,) complex

        for rel_id in path_rel_ids:
            A = self.mps.get_tensor_single(rel_id).to(device)   # (d, χ, d)

            # new_bond[β] = Σ_{i,α} phys[i] · A[i,α,β] · bond[α]
            # einsum: 'i, iab, a -> b'
            new_bond = torch.einsum("i, iab, a -> b", phys, A, bond)   # (χ,)

            # new_phys[j] = Σ_{i,α} phys[i] · A[i,α,j] · bond[α]
            # einsum: 'i, iaj, a -> j'
            new_phys = torch.einsum("i, iaj, a -> j", phys, A, bond)   # (d,)

            bond = new_bond

            # Normalise physical state to prevent magnitude explosion
            phys_norm = new_phys.abs().pow(2).sum().sqrt().clamp(min=1e-8)
            phys = new_phys / phys_norm

        # Amplitude = ⟨t|phys⟩ × ⟨v_R|bond⟩
        phys_score = tail.conj() @ phys          # complex scalar
        bond_score = v_R.conj() @ bond           # complex scalar

        return phys_score * bond_score   # complex scalar

    # ------------------------------------------------------------------
    # Public: contract a single path
    # ------------------------------------------------------------------

    def contract_path(
        self,
        head_state:   torch.Tensor,   # (d,) complex64
        path_rel_ids: List[int],
        tail_state:   torch.Tensor,   # (d,) complex64
        device:       torch.device,
    ) -> torch.Tensor:
        """
        Compute the MPS path amplitude for a single (head, path, tail) triple.

        The amplitude is complex; taking its squared modulus gives a probability-
        like score:  P = |amp|².  Taking the real part of the amplitude directly
        is also used in some scoring schemes.

        Args:
            head_state:   (d,) complex64 — head entity state.
            path_rel_ids: List[int] — relation IDs for each hop (length = path length).
            tail_state:   (d,) complex64 — tail entity state.
            device:       Target device.

        Returns:
            Complex scalar tensor — differentiable MPS amplitude.
            Use .abs()**2 for probability score, .real for logit-like score.
        """
        return self._contract_single(head_state, path_rel_ids, tail_state, device)

    def contract_batch(
        self,
        head_states: torch.Tensor,    # (B, d) complex64
        paths:       List[List[int]], # list of B paths (may differ in length)
        tail_states: torch.Tensor,    # (B, d) complex64
        device:      torch.device,
    ) -> torch.Tensor:
        """
        Contract MPS tensors for a batch of (head, path, tail) triples.

        Paths in the batch may have different lengths; they are processed
        independently (no padding needed — each is contracted in a loop).

        For large batches where all paths share the same length and relation
        sequence, see integrate_batched in LindbladPathIntegrator for a
        more efficient batched kernel.

        Args:
            head_states: (B, d) complex64 — head entity states.
            paths:       List of B path-relation-id lists (variable length).
            tail_states: (B, d) complex64 — tail entity states.
            device:      Target device.

        Returns:
            (B,) complex64 tensor of path amplitudes (differentiable).
        """
        B = head_states.shape[0]
        amplitudes = []
        for b in range(B):
            amp = self._contract_single(
                head_state=head_states[b],
                path_rel_ids=paths[b],
                tail_state=tail_states[b],
                device=device,
            )
            amplitudes.append(amp)
        return torch.stack(amplitudes, dim=0)   # (B,) complex64

    def probability_scores(
        self,
        head_states: torch.Tensor,    # (B, d) complex64
        paths:       List[List[int]],
        tail_states: torch.Tensor,    # (B, d) complex64
        device:      torch.device,
    ) -> torch.Tensor:
        """
        Compute |amplitude|² scores for a batch of (head, path, tail) triples.

        Args:
            head_states: (B, d) complex64.
            paths:       List of B path-relation-id lists.
            tail_states: (B, d) complex64.
            device:      Target device.

        Returns:
            (B,) float32 tensor of probability-like scores in [0, ∞).
            Not normalised to [0, 1] — treat as logits for cross-entropy.
        """
        amps = self.contract_batch(head_states, paths, tail_states, device)
        return amps.abs().pow(2).real   # (B,) float32

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def mps_entropy_loss(self) -> torch.Tensor:
        """
        Entropy regularisation that encourages appropriate bond dimension use.

        For each relation r, computes the singular values of the reshaped core
        tensor A[r] (viewed as a (d×χ, d) matrix) and returns the mean singular
        value entropy:

            H = -Σ_i σ_i log(σ_i)  (with σ normalised to sum to 1)

        High entropy → singular values are spread out → bond dimension is well-used.
        Low entropy  → singular values concentrate on few → bond is underutilised.

        Maximising this loss (or penalising low entropy) encourages the model to
        use the full bond dimension rather than collapsing to a lower-rank solution.

        Returns:
            Scalar float32 tensor — mean singular value entropy across all relations.
            Add to loss with a small negative coefficient to maximise entropy
            (e.g., loss -= 1e-3 × mps_entropy_loss()) or use as a positive
            regulariser if you prefer low-rank solutions.
        """
        A = torch.complex(self.mps.A_real, self.mps.A_imag)   # (R, d, χ, d)
        R, d, chi, _ = A.shape

        # Reshape each core to (d×χ, d) for SVD
        A_mat = A.view(R, d * chi, d)   # (R, d*χ, d)

        # Singular values for each relation (the min dimension sets the number)
        # torch.linalg.svdvals returns real non-negative singular values
        sigma = torch.linalg.svdvals(A_mat)   # (R, min(d*χ, d)) = (R, d)

        # Normalise singular values to sum to 1 per relation (like a distribution)
        sigma_norm = sigma / (sigma.sum(dim=-1, keepdim=True).clamp(min=1e-8))

        # Shannon entropy: -Σ σ log(σ),  clamp to avoid log(0)
        sigma_norm = sigma_norm.clamp(min=1e-12)
        entropy_per_rel = -(sigma_norm * torch.log(sigma_norm)).sum(dim=-1)   # (R,)

        return entropy_per_rel.mean()   # scalar float32


# ---------------------------------------------------------------------------
# 3. InfiniteHopAggregator
# ---------------------------------------------------------------------------

class InfiniteHopAggregator(nn.Module):
    """
    Aggregates over ALL paths of ALL lengths without explicit enumeration.

    Uses the Neumann series (geometric series for operators):

        Σ_{K=0}^∞ λ^K M^K = (I - λM)^{-1}   (for ||λM|| < 1)

    where:
        M — transfer matrix built from MPS tensors contracted with graph adjacency
        λ — damping factor in (0, 1) ensuring convergence (default 0.85)

    The aggregated state for a query head h is:

        x* = (I - λM)^{-1} h  =  h + λM h + λ²M²h + ...
           = contribution from paths of length 0, 1, 2, ..., ∞

    Equivalently, x* solves the linear system (I - λM) x = h, computed
    efficiently via power iteration:

        x_{t+1} = h + λ M x_t   (PageRank-style fixed point)

    Convergence: for ||λM||_op < 1 (ensured by λ < 1 and normalising M),
    the iteration converges geometrically in at most 20–50 steps for typical KG sizes.

    Transfer Matrix Construction:
        For each relation r and each directed edge (u, v) with relation r in the graph,
        the MPS core A[r] contributes a block to M.  For a compact implementation
        suitable for small-to-medium KGs (< 50k entities), we build M as a dense
        (E×d, E×d) complex matrix (entity × embedding dimension).

        For large KGs, use sparse representations — the implementation falls back
        to power iteration which only requires M × vector products.

    Args:
        num_relations: Number of distinct KG relation types.
        complex_dim:   Physical (entity embedding) dimension d.
        bond_dim:      Bond dimension χ for the underlying MPS (default 16).
        damping:       Neumann series damping factor λ ∈ (0, 1) (default 0.85).
                       Controls the effective path-length discount; analogous to
                       the damping factor in PageRank.
    """

    def __init__(
        self,
        num_relations: int,
        complex_dim:   int,
        bond_dim:      int = 16,
        damping:       float = 0.85,
    ) -> None:
        super().__init__()
        if not (0.0 < damping < 1.0):
            raise ValueError(f"damping must be in (0, 1), got {damping}")

        self.num_relations = num_relations
        self.complex_dim   = complex_dim
        self.bond_dim      = bond_dim
        self.damping       = damping

        # MPS tensors (shared with MPSPathContractor if desired, or standalone)
        self.mps = MPSRelationTensor(
            num_relations=num_relations,
            complex_dim=complex_dim,
            bond_dim=bond_dim,
        )

    # ------------------------------------------------------------------
    # Transfer matrix construction
    # ------------------------------------------------------------------

    def build_transfer_matrix(
        self,
        adj:         Dict[int, List[Tuple[int, int]]],   # {rel_id: [(h, t), ...]}
        mps_tensors: MPSRelationTensor,
        device:      torch.device,
    ) -> torch.Tensor:
        """
        Build the dense transfer matrix M ∈ C^(E×d, E×d) from MPS cores and adjacency.

        For each edge (u, v) with relation r in the adjacency:
            M[ v*d : (v+1)*d,  u*d : (u+1)*d ] += A_effective[r]

        where A_effective[r] is the d×d effective transition matrix for relation r,
        obtained by marginalising over the bond dimension with boundary vectors:

            A_effective[r]_{ij} = v_R^T · A[r]_{i,:,j} · v_L
                                = Σ_α v_R[α] A[r][i, α, j] v_L[α]
                                = einsum('iαj, α, α -> ij', A[r], v_R, v_L)
                                = einsum('iαj, α -> ij', A[r], v_R*v_L)

        This gives the "mean-field" approximation of the full bond-contracted MPS.

        For full bond fidelity, use block-diagonal M of size (E×d×χ, E×d×χ) —
        but for moderate χ and E this becomes prohibitively large.  The mean-field
        approximation (χ-marginalised) retains the leading contribution and keeps
        M ∈ C^(Ed × Ed) tractable.

        Args:
            adj:         Adjacency dict: {rel_id: list of (head_entity, tail_entity) pairs}.
                         All entity IDs must be in [0, E).
            mps_tensors: MPSRelationTensor module to extract A[r] from.
            device:      Target device.

        Returns:
            (E*d, E*d) complex64 sparse-ish dense matrix M (use with x → Mx).
            For E > 5000 consider using sparse operations instead.
        """
        v_L, v_R = mps_tensors.get_boundaries()
        v_L = v_L.to(device)
        v_R = v_R.to(device)

        # Combined boundary weight vector: w[α] = v_R[α] * v_L[α]
        w = v_R.conj() * v_L   # (χ,)

        # Determine number of entities E from adjacency
        all_entities: set = set()
        for rel_id, edges in adj.items():
            for h, t in edges:
                all_entities.add(h)
                all_entities.add(t)
        E = max(all_entities) + 1 if all_entities else 0

        if E == 0:
            # Empty graph: return zero matrix
            return torch.zeros(1, 1, dtype=torch.complex64, device=device)

        d   = self.complex_dim
        dim = E * d

        M = torch.zeros(dim, dim, dtype=torch.complex64, device=device)

        # Precompute effective d×d matrices for each relation
        # A_eff[r]_{ij} = einsum('iaj, a -> ij', A[r], w)
        rel_ids_present = list(adj.keys())
        if rel_ids_present:
            rel_tensor = torch.tensor(rel_ids_present, dtype=torch.long, device=device)
            A_batch    = mps_tensors.get_tensor(rel_tensor)   # (|R|, d, χ, d)

            # Marginalise over bond: A_eff[r]_{ij} = Σ_α A[r][i,α,j] w[α]
            # einsum: 'riaj, a -> rij'
            w_dev    = w.to(device)
            A_eff    = torch.einsum("riaj, a -> rij", A_batch, w_dev)   # (|R|, d, d)

            for rel_idx, rel_id in enumerate(rel_ids_present):
                A_r = A_eff[rel_idx]   # (d, d)
                for (h_ent, t_ent) in adj[rel_id]:
                    # Block M[ t*d:(t+1)*d, h*d:(h+1)*d ] += A_r
                    M[t_ent*d : (t_ent+1)*d,
                      h_ent*d : (h_ent+1)*d] = (
                        M[t_ent*d : (t_ent+1)*d,
                          h_ent*d : (h_ent+1)*d] + A_r
                    )

        return M   # (E*d, E*d) complex64

    # ------------------------------------------------------------------
    # Power iteration (Neumann series fixed point)
    # ------------------------------------------------------------------

    def aggregate(
        self,
        head_state: torch.Tensor,   # (d,) complex64
        adj:        Dict[int, List[Tuple[int, int]]],
        mps_module: MPSRelationTensor,
        device:     torch.device,
        max_iter:   int = 20,
        tol:        float = 1e-4,
    ) -> torch.Tensor:
        """
        Aggregate over all paths of all lengths via Neumann series power iteration.

        Computes the fixed point of:
            x* = h + λ M x*

        equivalently:  x* = (I - λM)^{-1} h

        using the power iteration:
            x_0 = h
            x_{k+1} = h + λ M x_k

        which converges to x* when λ ||M||_op < 1 (guaranteed for λ < 1
        with spectral normalisation applied).

        The resulting x* is an Ed-dimensional complex vector.  Its block
        corresponding to a query entity e is x*[e*d : (e+1)*d] — the
        d-dimensional aggregated state summing contributions from all paths
        arriving at entity e from head h via any sequence of relations.

        For scoring against a tail entity t:
            score(h, t) = ||x*[t*d : (t+1)*d]||² or ⟨t_state|x*_t⟩

        Args:
            head_state: (d,) complex64 — head entity quantum state.
            adj:        Adjacency dict {rel_id: [(h, t), ...]} — full graph structure.
            mps_module: MPSRelationTensor to build transfer matrix from.
            device:     Target device.
            max_iter:   Maximum power iteration steps (default 20).
            tol:        Convergence tolerance (||x_{k+1} - x_k||_2 < tol → stop).

        Returns:
            (E, d) complex64 — aggregated entity states for all E entities.
            Interpret as: agg_states[e] = Σ_{K,paths ending at e} λ^K amplitude.
        """
        head = head_state.to(device)   # (d,)

        # Build transfer matrix
        M = self.build_transfer_matrix(adj, mps_module, device)   # (E*d, E*d)
        Ed = M.shape[0]
        E  = Ed // self.complex_dim
        d  = self.complex_dim

        if E == 0:
            # Empty graph
            return head.unsqueeze(0)   # (1, d)

        # Spectral normalisation: scale M so λ||M||_op < 1
        # Use Frobenius norm as a proxy (always >= spectral norm)
        M_frob = M.abs().pow(2).sum().sqrt().clamp(min=1e-8)
        scale  = float(M_frob.item())
        if scale > 1e-6:
            M_normalised = M / (scale * (1.0 / self.damping + 1e-3))
        else:
            M_normalised = M

        # Build initial vector x_0: place head_state at entity index 0
        # (The caller should translate their head entity to index 0 or provide
        # a precomputed Ed-dimensional vector.  Here we place head at block 0.)
        x = torch.zeros(Ed, dtype=torch.complex64, device=device)
        x[:d] = head   # place head state at entity block 0

        h_vec = x.clone()   # fixed: the "source" term (RHS = h)

        lam = self.damping

        # Power iteration: x_{k+1} = h + λ M x_k
        for _ in range(max_iter):
            x_new = h_vec + lam * (M_normalised @ x)   # (Ed,)
            # Convergence check
            diff = (x_new - x).abs().pow(2).sum().sqrt()
            x = x_new
            if diff.item() < tol:
                break

        # Reshape to (E, d)
        return x.view(E, d)   # (E, d) complex64

    def aggregate_direct_solve(
        self,
        head_state: torch.Tensor,   # (d,) complex64
        adj:        Dict[int, List[Tuple[int, int]]],
        mps_module: MPSRelationTensor,
        device:     torch.device,
    ) -> torch.Tensor:
        """
        Compute the Neumann series aggregation via direct linear solve.

        Solves (I - λM) x = h directly using torch.linalg.solve.
        More accurate than power iteration for small graphs (E×d < ~1000),
        but O((Ed)³) in complexity — impractical for large KGs.

        Use for:
            - Validation / debugging of the power iteration result
            - Small KGs (E < 100, d < 16) where exact solve is affordable
            - When gradient flow through the solve itself is needed

        Args:
            head_state: (d,) complex64 — head entity state.
            adj:        Adjacency dict.
            mps_module: MPSRelationTensor module.
            device:     Target device.

        Returns:
            (E, d) complex64 — aggregated entity states.
        """
        head = head_state.to(device)

        M = self.build_transfer_matrix(adj, mps_module, device)   # (Ed, Ed)
        Ed = M.shape[0]
        E  = Ed // self.complex_dim
        d  = self.complex_dim

        if E == 0:
            return head.unsqueeze(0)

        # Build (I - λM)
        I   = torch.eye(Ed, dtype=torch.complex64, device=device)
        lhs = I - self.damping * M   # (Ed, Ed)

        # Build RHS: h placed at entity block 0
        rhs = torch.zeros(Ed, dtype=torch.complex64, device=device)
        rhs[:d] = head

        # Solve: lhs @ x = rhs
        # torch.linalg.solve works for complex matrices
        x = torch.linalg.solve(lhs, rhs)   # (Ed,)

        return x.view(E, d)   # (E, d) complex64

    def score_tail(
        self,
        aggregated_states: torch.Tensor,   # (E, d) complex64
        tail_entity_id:    int,
        tail_state:        torch.Tensor,   # (d,) complex64
    ) -> torch.Tensor:
        """
        Score a specific tail entity from the aggregated states.

        Args:
            aggregated_states: (E, d) complex64 — output of aggregate().
            tail_entity_id:    int — entity index of the tail.
            tail_state:        (d,) complex64 — tail entity quantum state.

        Returns:
            Scalar float32 — Born-rule-like score ⟨t|x_tail⟩.real.
            Use abs()**2 for a probability-like score.
        """
        x_tail = aggregated_states[tail_entity_id]   # (d,) complex64
        score  = (tail_state.conj() @ x_tail).real
        return score

    def score_all_tails(
        self,
        aggregated_states: torch.Tensor,    # (E, d) complex64
        all_tail_states:   torch.Tensor,    # (E, d) complex64
    ) -> torch.Tensor:
        """
        Score all entities simultaneously as candidate tails.

        Args:
            aggregated_states: (E, d) complex64 — output of aggregate().
            all_tail_states:   (E, d) complex64 — all entity quantum states.

        Returns:
            (E,) float32 — scores for each entity as the tail.
        """
        # scores[e] = ⟨t_e | x_e⟩.real
        scores = (all_tail_states.conj() * aggregated_states).sum(dim=-1).real
        return scores   # (E,) float32

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"complex_dim={self.complex_dim}, "
            f"bond_dim={self.bond_dim}, "
            f"damping={self.damping}"
        )
