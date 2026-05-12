"""
models/baselines/rotate.py — RotatE Baseline  [V1]

RotatE (Sun et al., ICLR 2019): entity-relation interaction via complex rotation.
score(h, r, t) = gamma - ||h ∘ r - t||  where |r_j| = 1 (unit modulus)

THE KEY DIFFERENTIATOR FROM THIS PROJECT'S MODEL:
    RotatE applies exp(iθⱼ) element-wise to compute: h ∘ r - t
    This project applies exp(iθⱼ) element-wise to EVOLVE STATE: U_r|h⟩
    Then SUMS AMPLITUDES ACROSS K PATHS before squaring (Born rule).

    RotatE: |score| scores one triple at a time (no path composition, no sum)
    Quantum: |Σᵢ αᵢ ⟨t|U_Pᵢ|h⟩|² scores via amplitude superposition

    The squaring of the SUM produces interference cross-terms.
    RotatE squares each term individually then sums: Σᵢ |Aᵢ|² (classical, no interference).
    These are NOT the same operation.

RELATION PATTERNS RotatE handles (Section 3 of paper):
    Symmetry:     r(a,b) ⟹ r(b,a)   → θ_r = 0 or π
    Antisymmetry: r(a,b) ⟹ ¬r(b,a)  → θ_r ≠ 0, π
    Inversion:    r₁(a,b) ⟺ r₂(b,a)  → θ_r₂ = -θ_r₁
    Composition:  r₁(a,b) ∧ r₂(b,c) ⟹ r₃(a,c) → complex angle addition

USAGE:
    model = RotatE(num_entities=28, num_relations=12, embed_dim=64, gamma=12.0)
    scores = model.score_triple(h, r, t)
    scores = model.score_triple_vs_all(h, r)
"""
from __future__ import annotations
import math
import torch, torch.nn as nn, torch.nn.functional as F


class RotatE(nn.Module):
    """
    RotatE: entities as complex vectors, relations as unit-modulus complex rotations.

    Implementation detail:
        Entities: (num_entities, embed_dim) stored as real float.
                  First half = real parts, second half = imaginary parts.
        Relations: (num_relations, embed_dim//2) phase angles θ ∈ (-π, π].
                   Applied as: r_j = exp(iθ_j) = (cos θ_j, sin θ_j)

    Args:
        num_entities:  Entity count.
        num_relations: Relation count.
        embed_dim:     Total embedding dim. complex_dim = embed_dim // 2.
        gamma:         Margin for self-adversarial loss.
        epsilon:       Embedding range initialization factor.
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int   = 64,
        gamma:         float = 12.0,
        epsilon:       float = 2.0,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.complex_dim   = embed_dim // 2
        self.gamma         = gamma

        # Entity embedding range (RotatE paper formula)
        self.embedding_range = (gamma + epsilon) / self.complex_dim

        self.entity_embeddings = nn.Embedding(num_entities,  embed_dim)
        self.relation_phases   = nn.Embedding(num_relations, self.complex_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize per RotatE paper: uniform in [-embedding_range, embedding_range]."""
        nn.init.uniform_(
            self.entity_embeddings.weight,
            -self.embedding_range, self.embedding_range,
        )
        # Relation phases: uniform in [-π, π]
        nn.init.uniform_(self.relation_phases.weight, -math.pi, math.pi)

    def _get_complex(
        self,
        entity_ids: torch.Tensor,   # (...) int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (real, imaginary) parts of entity embeddings.

        Returns:
            real: (..., complex_dim) float
            imag: (..., complex_dim) float
        """
        emb  = self.entity_embeddings(entity_ids)         # (..., embed_dim)
        real = emb[..., :self.complex_dim]
        imag = emb[..., self.complex_dim:]
        return real, imag

    def _get_relation_complex(
        self,
        relation_ids: torch.Tensor,   # (...) int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (cos θ, sin θ) for relation phase angles (unit modulus).

        |r_j| = 1 is guaranteed: r_j = exp(iθ_j) = cos(θ_j) + i·sin(θ_j).
        This is the RotatE constraint that preserves entity norms.
        """
        theta   = self.relation_phases(relation_ids)       # (..., complex_dim)
        r_real  = torch.cos(theta)
        r_imag  = torch.sin(theta)
        return r_real, r_imag

    def score_triple(
        self,
        head_ids:     torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids:     torch.Tensor,
    ) -> torch.Tensor:
        """
        RotatE score: gamma - ||h ∘ r - t||₂

        Complex element-wise product: (h ∘ r)_j = h_j · r_j
            = (h_re + i·h_im)(r_re + i·r_im)
            = (h_re·r_re - h_im·r_im) + i(h_re·r_im + h_im·r_re)

        Returns:
            (B,) float scores. Higher = more likely true.
        """
        h_re, h_im = self._get_complex(head_ids)
        r_re, r_im = self._get_relation_complex(relation_ids)
        t_re, t_im = self._get_complex(tail_ids)

        # h ∘ r (complex multiplication)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # ||h ∘ r - t||
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        dist    = torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-10).sum(dim=-1)

        return self.gamma - dist

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as tails: gamma - ||h ∘ r - t_e||₂ for all e.

        Returns:
            (B, num_entities) float score matrix.
        """
        B      = head_ids.shape[0]
        device = head_ids.device

        h_re, h_im = self._get_complex(head_ids)
        r_re, r_im = self._get_relation_complex(relation_ids)

        # h ∘ r: (B, complex_dim)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # All entity embeddings
        all_ids = torch.arange(self.num_entities, device=device)
        all_re  = self.entity_embeddings.weight[:, :self.complex_dim]   # (E, d)
        all_im  = self.entity_embeddings.weight[:, self.complex_dim:]   # (E, d)

        # Pairwise distance: (B, E, d) → (B, E)
        # diff_re[b, e, j] = hr_re[b, j] - all_re[e, j]
        diff_re = hr_re.unsqueeze(1) - all_re.unsqueeze(0)   # (B, E, d)
        diff_im = hr_im.unsqueeze(1) - all_im.unsqueeze(0)   # (B, E, d)

        dist = torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-10).sum(dim=-1)  # (B, E)
        return self.gamma - dist

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "entity_emb":    self.entity_embeddings.weight.numel(),
            "relation_phase": self.relation_phases.weight.numel(),
            "total":         total,
        }

    def extra_repr(self) -> str:
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, gamma={self.gamma}"
        )
