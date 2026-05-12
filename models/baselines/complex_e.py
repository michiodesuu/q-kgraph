"""
models/baselines/complex_e.py — ComplEx Baseline  [V1]

ComplEx (Trouillon et al., ICML 2016): complex-valued embeddings, real scoring.
score(h, r, t) = Re(⟨r, h, conj(t)⟩) = Re(Σⱼ r_j · h_j · conj(t_j))

THE FORMAL DIFFERENTIATOR FROM THIS PROJECT:
    ComplEx:        score = Re(amplitude)           [real part — interference-free]
    QuantumReasoner: score = |amplitude|²            [squared magnitude — interference]

    These are different operations on complex number z = a + ib:
        Re(z) = a         — always real, loses imaginary component
        |z|² = a² + b²   — squared magnitude, always ≥ 0

    For two paths with amplitudes z₁, z₂:
        Re(z₁ + z₂) = Re(z₁) + Re(z₂)       — linear, no cross-terms
        |z₁ + z₂|² = |z₁|² + |z₂|² + 2Re(z₁·z̄₂)  — cross-term 2Re(z₁z̄₂) can be negative

    The cross-term 2Re(z₁z̄₂) is NEGATIVE when arg(z₁) - arg(z₂) ≈ π.
    This is the interference that ComplEx structurally cannot produce.

WHAT COMPLEX DOES HANDLE (its real advantage over TransE):
    Asymmetric relations: r(a,b) ≠ r(b,a) — because conj(t) breaks symmetry.
    TransE cannot model asymmetric relations at all.
    ComplEx can, which is why it outperforms TransE on FB15k-237.
    This model builds on ComplEx's complex structure but adds path interference.

USAGE:
    model = ComplEx(num_entities=28, num_relations=12, embed_dim=64)
    scores = model.score_triple(h, r, t)
    scores = model.score_triple_vs_all(h, r)
    reg    = model.regularization_loss()   # add to loss: reg_weight * reg
"""
from __future__ import annotations
import torch, torch.nn as nn


class ComplEx(nn.Module):
    """
    ComplEx: complex-valued embeddings with Re(trilinear) scoring.

    The four-term expansion of Re(⟨r, h, conj(t)⟩):
        = Re(r)·Re(h)·Re(t)  + Re(r)·Im(h)·Im(t)
        + Im(r)·Re(h)·Im(t)  - Im(r)·Im(h)·Re(t)

    This is equation (9) in Trouillon et al. 2016. Note: the result is
    a real scalar — no squared magnitude, no interference possible.

    Args:
        num_entities:   Entity count.
        num_relations:  Relation count.
        embed_dim:      Total embedding dim. complex_dim = embed_dim // 2.
        reg_weight:     L3 regularization weight (original paper uses L3).
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int   = 64,
        reg_weight:    float = 1e-3,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.complex_dim   = embed_dim // 2
        self.reg_weight    = reg_weight

        # Store real and imaginary parts separately (same convention as QuantumReasoner)
        self.entity_re   = nn.Embedding(num_entities,  self.complex_dim)
        self.entity_im   = nn.Embedding(num_entities,  self.complex_dim)
        self.relation_re = nn.Embedding(num_relations, self.complex_dim)
        self.relation_im = nn.Embedding(num_relations, self.complex_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization."""
        for emb in [self.entity_re, self.entity_im,
                    self.relation_re, self.relation_im]:
            nn.init.xavier_uniform_(emb.weight)

    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        ComplEx score: Re(⟨r, h, conj(t)⟩) — the four-term expansion.

        This is a REAL-VALUED score. No squared magnitude. No interference.
        Adding this baseline to your experiments empirically proves that
        complex arithmetic alone is not sufficient — interference (|·|²) is needed.

        Returns:
            (B,) float scores.
        """
        h_re = self.entity_re(head_ids)        # (B, d)
        h_im = self.entity_im(head_ids)        # (B, d)
        r_re = self.relation_re(relation_ids)   # (B, d)
        r_im = self.relation_im(relation_ids)   # (B, d)
        t_re = self.entity_re(tail_ids)        # (B, d)
        t_im = self.entity_im(tail_ids)        # (B, d)

        # Four-term expansion of Re(r ⊙ h ⊙ conj(t))
        # where ⊙ = element-wise complex product
        score = (
              (r_re * h_re * t_re).sum(-1)   # Re·Re·Re
            + (r_re * h_im * t_im).sum(-1)   # Re·Im·Im
            + (r_im * h_re * t_im).sum(-1)   # Im·Re·Im
            - (r_im * h_im * t_re).sum(-1)   # -Im·Im·Re
        )
        return score

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as tails.

        Uses the bilinear form: score[b, e] = Re(r_b ⊙ h_b ⊙ conj(e))
        Vectorized as: Re( (r_b ⊙ h_b) · conj(all_e).T )

        Returns:
            (B, num_entities) float score matrix.
        """
        h_re = self.entity_re(head_ids)        # (B, d)
        h_im = self.entity_im(head_ids)        # (B, d)
        r_re = self.relation_re(relation_ids)   # (B, d)
        r_im = self.relation_im(relation_ids)   # (B, d)

        # r ⊙ h: complex product of r and h → (B, d) complex result
        # (r_re + i·r_im)(h_re + i·h_im) = (r_re·h_re - r_im·h_im) + i(r_re·h_im + r_im·h_re)
        rh_re = r_re * h_re - r_im * h_im   # (B, d)
        rh_im = r_re * h_im + r_im * h_re   # (B, d)

        # All entity embeddings: (E, d)
        all_re = self.entity_re.weight   # (E, d)
        all_im = self.entity_im.weight   # (E, d)

        # Re(⟨rh, conj(all_e)⟩) = rh_re·all_re + rh_im·all_im
        # (because conj(t) = t_re - i·t_im, so Re(rh · conj(t)) = rh_re·t_re + rh_im·t_im)
        scores = (
            rh_re @ all_re.T    # (B, E)
          + rh_im @ all_im.T    # (B, E)
        )
        return scores

    def regularization_loss(self) -> torch.Tensor:
        """
        L3 regularization (nuclear 3-norm) from ComplEx paper.

        L3(h, r, t) = (1/3)(||h||₃³ + ||r||₃³ + ||t||₃³)
        where ||x||₃³ = Σⱼ |x_j|³

        In practice: use L2 regularization on all parameters.
        The full L3 is expensive and usually approximated.
        """
        all_params = torch.cat([
            self.entity_re.weight,
            self.entity_im.weight,
            self.relation_re.weight,
            self.relation_im.weight,
        ], dim=0)
        # L2 approximation of L3 (standard in practice)
        return self.reg_weight * all_params.pow(2).sum()

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "entity_re":   self.entity_re.weight.numel(),
            "entity_im":   self.entity_im.weight.numel(),
            "relation_re": self.relation_re.weight.numel(),
            "relation_im": self.relation_im.weight.numel(),
            "total":       total,
        }

    def extra_repr(self) -> str:
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, complex_dim={self.complex_dim}"
        )
