"""
models/components/quantum_states.py — Entity Hilbert Space Embeddings  [V1]

PURPOSE:
    Implements QM Postulate 1: entities are unit-norm complex vectors in H=ℂ^d.
    The central object of the quantum KGE model.

QUANTUM MECHANICS CONNECTION:
    Postulate 1 (State Space): Physical systems are described by unit vectors
    in a Hilbert space. Here: H = ℂ^(embed_dim/2), entities are pure states.

    Born Rule (Postulate 3): The probability of measuring entity t given
    state |h⟩ is |⟨t|h⟩|². This is the scoring function.

IMPLEMENTATION:
    Two nn.Embedding tables store real and imaginary parts separately:
        real_embeddings: (N, complex_dim) float32 — x_e values
        imag_embeddings: (N, complex_dim) float32 — y_e values
    Combined state: |e⟩ = x_e + iy_e ∈ ℂ^complex_dim

    Why two tables instead of torch.complex embeddings?
        - Separate LR for real vs imaginary (TrainerV2 param groups)
        - Standard optimizers work natively with float tensors
        - Individual weight decay control (decay=0 for imag)

USAGE:
    encoder = QuantumStateEncoder(num_entities=28, embed_dim=16)
    ids     = torch.tensor([0, 1, 2])
    states  = encoder(ids)         # (3, 8) complex64
    norms   = states.norm(dim=-1)  # should all be ~1.0
    probs   = encoder.probability(states[:1], states[1:2])  # scalar in [0,1]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumStateEncoder(nn.Module):
    """
    Encodes knowledge graph entities as unit-norm complex quantum states.

    Each entity e is stored as |e⟩ = x_e + iy_e ∈ ℂ^(complex_dim)
    where complex_dim = embed_dim // 2.

    The normalize flag enforces ||e|| = 1 at every forward pass,
    implementing QM Postulate 1 (state vectors live on the unit sphere).

    Args:
        num_entities:  Number of distinct entities in the KG.
        embed_dim:     Total embedding dimension. complex_dim = embed_dim // 2.
                       Must be even.
        dropout:       Dropout rate (0.0 recommended for small toy KG).
        normalize:     If True, normalizes states to unit norm at every forward.
                       Set False only for debugging/visualization.
        init_std:      Standard deviation for random initialization.
                       Small values (0.01-0.1) help avoid Phase Collapse at start.

    Attributes:
        complex_dim:         embed_dim // 2
        real_embeddings:     nn.Embedding (num_entities, complex_dim) — Re(|e⟩)
        imag_embeddings:     nn.Embedding (num_entities, complex_dim) — Im(|e⟩)
    """

    def __init__(
        self,
        num_entities: int,
        embed_dim:    int   = 16,
        dropout:      float = 0.0,
        normalize:    bool  = True,
        init_std:     float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be even (real + imaginary parts). "
                f"Got embed_dim={embed_dim}."
            )

        self.num_entities = num_entities
        self.embed_dim    = embed_dim
        self.complex_dim  = embed_dim // 2
        self.normalize    = normalize
        self.dropout      = nn.Dropout(p=dropout) if dropout > 0 else None

        # Two separate embedding tables for real and imaginary parts
        # This enables independent learning rates (TrainerV2 param groups)
        self.real_embeddings = nn.Embedding(num_entities, self.complex_dim)
        self.imag_embeddings = nn.Embedding(num_entities, self.complex_dim)

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        """
        Initialize embeddings with small random values.

        Small initialization is critical:
            - Large initial imaginary values → unstable early training
            - Zero imaginary values → Phase Collapse from epoch 1
            - Small non-zero imaginary values → stable learning

        The real parts are initialized larger than imaginary parts.
        This matches the quantum convention: the "ground state" is real
        with small imaginary perturbations that grow during training.
        """
        nn.init.normal_(self.real_embeddings.weight, mean=0.0, std=std)
        nn.init.normal_(self.imag_embeddings.weight, mean=0.0, std=std * 0.3)

        # Normalize to unit sphere after initialization only if required
        if self.normalize:
            with torch.no_grad():
                self._normalize_inplace()

    def _normalize_inplace(self) -> None:
        """Normalize all entity embeddings to unit complex norm in-place."""
        re = self.real_embeddings.weight.data
        im = self.imag_embeddings.weight.data
        norms = (re.pow(2) + im.pow(2)).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-10)
        self.real_embeddings.weight.data = re / norms
        self.imag_embeddings.weight.data = im / norms

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(
        self,
        entity_ids: torch.Tensor,   # (...) int64
    ) -> torch.Tensor:
        """
        Look up entity quantum states.

        Args:
            entity_ids: Integer tensor of entity IDs. Can be any shape.

        Returns:
            Complex tensor of shape (..., complex_dim), dtype=complex64.
            If normalize=True, all states have unit norm: ||state|| = 1.
        """
        real = self.real_embeddings(entity_ids)   # (..., complex_dim) float32
        imag = self.imag_embeddings(entity_ids)   # (..., complex_dim) float32

        if self.dropout is not None:
            real = self.dropout(real)
            imag = self.dropout(imag)

        # Combine into complex tensor: |e⟩ = real + i * imag
        states = torch.complex(real, imag)        # (..., complex_dim) complex64

        if self.normalize:
            states = self._normalize(states)

        return states

    def _normalize(self, states: torch.Tensor) -> torch.Tensor:
        """
        Normalize complex states to unit L2 norm.

        ||state|| = sqrt(Σⱼ |state_j|²) = sqrt(Σⱼ (re_j² + im_j²)) = 1

        This implements QM Postulate 1: quantum states have unit norm.
        """
        norms = states.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-10)
        return states / norms

    # ── Quantum operations ──────────────────────────────────────────────────────
    def inner_product(
        self,
        states_a: torch.Tensor,   # (..., complex_dim) complex
        states_b: torch.Tensor,   # (..., complex_dim) complex
    ) -> torch.Tensor:
        """
        Compute quantum inner product ⟨a|b⟩ = Σⱼ conj(a_j) · b_j.

        This is the fundamental operation of Hilbert space theory.
        The result is a complex scalar (or batch of complex scalars).

        Properties:
            ⟨a|b⟩ = conj(⟨b|a⟩)   [conjugate symmetry]
            |⟨a|b⟩| ≤ ||a|| · ||b||  [Cauchy-Schwarz]
            ⟨a|a⟩ = 1 for unit vectors [self-inner product]

        Args:
            states_a, states_b: Complex tensors of same shape (..., complex_dim).

        Returns:
            Complex tensor of shape (...) — the inner products.
        """
        return (states_a.conj() * states_b).sum(dim=-1)

    def probability(
        self,
        states_a: torch.Tensor,   # (..., complex_dim) complex
        states_b: torch.Tensor,   # (..., complex_dim) complex
    ) -> torch.Tensor:
        """
        Compute Born rule probability |⟨a|b⟩|² ∈ [0, 1].

        This is QM Postulate 3 (measurement). The probability of
        measuring state |b⟩ given system in state |a⟩ is |⟨a|b⟩|².

        CRITICAL DISTINCTION FROM COMPLEX:
            ComplEx score: Re(⟨r,h,conj(t)⟩)   [real part — no interference]
            This function:  |⟨a|b⟩|²             [squared magnitude — INTERFERENCE]

        The difference is one line of code but mathematically fundamental.
        |z|² = Re(z)² + Im(z)² ≠ Re(z) in general.
        The extra Im(z)² term enables interference cross-products.

        Args:
            states_a, states_b: Complex tensors of same shape.

        Returns:
            Real tensor of shape (...) with values in [0, 1].
        """
        amplitude = self.inner_product(states_a, states_b)
        return amplitude.abs().pow(2)

    def state_fidelity(
        self,
        entity_a: int,
        entity_b: int,
        device:   Optional[torch.device] = None,
    ) -> float:
        """
        Compute quantum fidelity F(a, b) = |⟨a|b⟩|² between two entities.

        Fidelity measures how "similar" two quantum states are:
            F = 1: identical states (same entity)
            F = 0: orthogonal states (maximally different)
            F ∈ (0,1): partially overlapping states

        After training, semantically similar entities should have higher fidelity.
        E.g.: F(Platypus, Mammal) > F(Platypus, Reptile) after learning correct taxonomy.

        Args:
            entity_a, entity_b: Entity IDs.
            device: Compute device (defaults to model device).

        Returns:
            Float fidelity in [0, 1].
        """
        if device is None:
            device = self.real_embeddings.weight.device

        with torch.no_grad():
            a_id  = torch.tensor([entity_a], device=device)
            b_id  = torch.tensor([entity_b], device=device)
            s_a   = self(a_id).squeeze(0)
            s_b   = self(b_id).squeeze(0)
            return float(self.probability(s_a.unsqueeze(0), s_b.unsqueeze(0)).item())

    def init_from_llm(
        self,
        llm_embeddings: torch.Tensor,
        projection_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize entity states from pre-trained LLM embeddings (e.g., Sentence-BERT).

        This is the "hybrid LLM-quantum" initialization that addresses the
        FQCE failure mode (pure structural topology → low accuracy).
        By grounding the initial Hilbert space in semantic meaning, the model
        starts with entities that are already geometrically separated by meaning,
        making the interference pattern easier to learn.

        Method:
            1. Project LLM embeddings to complex_dim via linear layer (if needed)
            2. Split into real (first half) and imaginary (second half) parts
            3. Normalize to unit complex norm

        Args:
            llm_embeddings: (N, llm_dim) float tensor of pre-trained embeddings.
                            N must equal num_entities.
                            llm_dim is the LLM embedding dimension (e.g., 384 for MiniLM).
            projection_dim: If not None, projects LLM embeddings to this dim first.
                            Default: 2 * complex_dim (full embed_dim).

        Example:
            >>> from sentence_transformers import SentenceTransformer
            >>> sbert = SentenceTransformer("all-MiniLM-L6-v2")
            >>> embs  = torch.tensor(sbert.encode(entity_descriptions))
            >>> encoder.init_from_llm(embs)
        """
        if llm_embeddings.shape[0] != self.num_entities:
            raise ValueError(
                f"llm_embeddings has {llm_embeddings.shape[0]} rows but "
                f"num_entities={self.num_entities}."
            )

        target_dim = 2 * self.complex_dim

        with torch.no_grad():
            embs = llm_embeddings.float()

            # Project if needed
            if embs.shape[1] != target_dim:
                # Simple linear projection (no bias, preserves relative distances)
                W = torch.randn(embs.shape[1], target_dim) / math.sqrt(embs.shape[1])
                embs = embs @ W  # (N, target_dim)

            # Split into real and imaginary
            real_part = embs[:, :self.complex_dim]
            imag_part = embs[:, self.complex_dim:]

            # Normalize to unit complex norm
            norms = (real_part.pow(2) + imag_part.pow(2)).sum(-1, keepdim=True).sqrt().clamp(1e-10)
            real_part = real_part / norms
            imag_part = imag_part / norms

            self.real_embeddings.weight.data.copy_(real_part)
            self.imag_embeddings.weight.data.copy_(imag_part)

    # ── Parameter access ──────────────────────────────────────────────────────
    def get_all_states(
        self,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return all entity states as a single (N, complex_dim) complex tensor.

        Used for:
            - Subspace projection (SubspaceProjector)
            - Interference monitoring (InterferenceMonitor)
            - Visualization (phase_plots.py)

        Args:
            device: Target device.

        Returns:
            (num_entities, complex_dim) complex64 tensor.
        """
        if device is None:
            device = self.real_embeddings.weight.device
        all_ids = torch.arange(self.num_entities, device=device)
        return self(all_ids)

    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_entities={self.num_entities}, "
            f"embed_dim={self.embed_dim}, "
            f"complex_dim={self.complex_dim}, "
            f"normalize={self.normalize}, "
            f"params={self.num_params():,}"
        )
