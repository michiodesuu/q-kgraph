"""
models/baselines/transe.py — TransE Baseline  [V1]

TransE (Bordes et al., NeurIPS 2013): translating embeddings for KGE.
score(h, r, t) = -||h + r - t||_p

WHY THIS IS THE PRIMARY CONTRAST CASE:
    TransE is purely real, purely additive, strictly monotonic.
    No mechanism exists for path cancellation.
    When Platypus has correct and contradictory paths, TransE scores
    both positively and sums them — it cannot distinguish.
    Under 20% noise, TransE should degrade the steepest of all models.
    That widening gap is your paper's Figure 3 key result.

PAPER CONNECTION:
    Every number in Table 2 for TransE comes from this file.
    The fairness constraint: same trainer, same seed, same negative sampling.
    The loss: MarginRankingLoss (original paper loss, gamma=9.0).

USAGE:
    model = TransE(num_entities=28, num_relations=12, embed_dim=64)
    scores = model.score_triple(h, r, t)         # (B,) float
    scores = model.score_triple_vs_all(h, r)     # (B, E) float
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class TransE(nn.Module):
    """
    TransE: entities and relations as real vectors. h + r ≈ t for true triples.

    Args:
        num_entities:  Total entity count.
        num_relations: Total relation count.
        embed_dim:     Embedding dimension (same for entities and relations).
        p_norm:        L_p norm (1 = L1, 2 = L2). L1 usually better for FB15k-237.
        margin:        Margin gamma for MarginRankingLoss. Match loss_fn setting.
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int   = 64,
        p_norm:        int   = 1,
        margin:        float = 9.0,
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.p_norm        = p_norm
        self.margin        = margin

        self.entity_embeddings   = nn.Embedding(num_entities,  embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize per the TransE paper: uniform in [-6/√d, 6/√d].
        Relations normalized to unit L2 norm after init.
        """
        bound = 6.0 / (self.embed_dim ** 0.5)
        nn.init.uniform_(self.entity_embeddings.weight,   -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        # Normalize relation embeddings to unit norm (TransE paper convention)
        with torch.no_grad():
            norms = self.relation_embeddings.weight.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            self.relation_embeddings.weight.data /= norms

    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Compute TransE scores: score = -||h + r - t||_p.

        Higher scores (less negative) mean the triple is more likely true.
        Score range: [-∞, 0]. Exactly 0 only when h + r = t exactly.

        Returns:
            (B,) float scores.
        """
        h = self.entity_embeddings(head_ids)      # (B, d)
        r = self.relation_embeddings(relation_ids) # (B, d)
        t = self.entity_embeddings(tail_ids)      # (B, d)

        # TransE scoring: -||h + r - t||_p
        diff  = h + r - t                          # (B, d)
        score = -diff.norm(p=self.p_norm, dim=-1)  # (B,) negative distances
        return score

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as candidate tails: score = -||h + r - t_e||_p.

        Used during evaluation (filtered ranking).
        Returns (B, num_entities) float matrix.
        """
        h = self.entity_embeddings(head_ids)       # (B, d)
        r = self.relation_embeddings(relation_ids)  # (B, d)
        hr = h + r                                  # (B, d)

        # All entity embeddings: (E, d)
        all_e = self.entity_embeddings.weight       # (E, d)

        # Pairwise distances: (B, E)
        # ||hr_b - all_e_e||_p for all (b, e) pairs
        # = (hr.unsqueeze(1) - all_e.unsqueeze(0)).norm(p, dim=-1)
        diff   = hr.unsqueeze(1) - all_e.unsqueeze(0)  # (B, E, d)
        scores = -diff.norm(p=self.p_norm, dim=-1)      # (B, E)
        return scores

    def regularization_loss(self, weight: float = 1e-3) -> torch.Tensor:
        """L2 regularization on entity and relation embeddings."""
        return weight * (
            self.entity_embeddings.weight.norm(p=2).pow(2)
            + self.relation_embeddings.weight.norm(p=2).pow(2)
        )

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"entity_emb": self.entity_embeddings.weight.numel(),
                "relation_emb": self.relation_embeddings.weight.numel(),
                "total": total}

    def extra_repr(self) -> str:
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, p_norm={self.p_norm}"
        )
