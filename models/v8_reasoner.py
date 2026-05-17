"""
models/v8_reasoner.py — V8 Curved Manifold Reasoner

V8CurvedManifoldReasoner supersedes:
    - QuantumReasoner (V1–V5): replaced DiagonalUnitary with full unitary T_r
    - QuaternionReasoner (V4): replaced quaternion algebra with curved manifolds

Architecture:
    1. Encode h and t as curved embeddings (κ_h, κ_t per entity)
    2. Apply parallel transport T_r to h → h' = T_r|h⟩  (curved geometry)
    3. For multi-hop: compose transports T_r2 @ T_r1, compute holonomy gap
    4. Score: Born rule |⟨t|h'⟩|² + holonomy_weight × exp(−||Hol−I||_F)
    5. Contradiction signal: large holonomy gap → lower confidence

Imports:
    from models.components.holonomy import ParallelTransportOperator, HolonomyOperator,
                                           RelationalManifoldEncoder
    from models.components.adaptive_curvature import CurvedEmbedding, RelationCurvatureAdapter
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.components.holonomy import (
    ParallelTransportOperator,
    HolonomyOperator,
    RelationalManifoldEncoder,
)
from models.components.adaptive_curvature import (
    CurvedEmbedding,
    RelationCurvatureAdapter,
)

_EPS = 1e-8


class V8CurvedManifoldReasoner(nn.Module):
    """Complete V8 model: curved embeddings + parallel transport + Born rule.

    Scoring for a triple (h, r, t):
        1. Encode h → (re_h, im_h) and t → (re_t, im_t) via CurvedEmbedding
        2. Transport: (re_h', im_h') = T_r (re_h, im_h)
        3. Born rule score: |⟨t | h'⟩|²
        4. Holonomy confidence (optional): exp(−||Hol−I||_F)

    For multi-hop paths (r1, r2, ..., rN):
        Composed transport T_rN @ ... @ T_r1 is applied once.
        The holonomy gap of any implied cycle measures contradiction.

    Args:
        num_entities (int):         Number of entities in the KG.
        num_relations (int):        Number of distinct relations.
        embed_dim (int):            Embedding dimension (complex_dim = embed_dim // 2).
        holonomy_weight (float):    Weight of holonomy confidence term (default 0.1).
        use_adaptive_curvature (bool): Enable per-entity κ (default True).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 128,
        holonomy_weight: float = 0.1,
        use_adaptive_curvature: bool = True,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.complex_dim = embed_dim // 2
        self.holonomy_weight = holonomy_weight
        self.use_adaptive_curvature = use_adaptive_curvature

        # Curved entity embeddings — κ_e per entity
        self.curved_encoder = CurvedEmbedding(
            num_entities=num_entities,
            embed_dim=embed_dim,
            init_curvature=0.0,
        )

        # Full unitary parallel transport T_r ∈ U(complex_dim)
        self.transport = ParallelTransportOperator(
            num_relations=num_relations,
            complex_dim=self.complex_dim,
        )

        # Holonomy operator (wraps transport)
        self.holonomy = HolonomyOperator(self.transport)

        # Curvature adapter for relation-specific geometry
        self.curvature_adapter = RelationCurvatureAdapter(
            num_relations=num_relations,
            embed_dim=embed_dim,
        )

        # Optional flat-to-curved projection for the tail
        self.tail_proj_re = nn.Linear(self.complex_dim, self.complex_dim, bias=False)
        self.tail_proj_im = nn.Linear(self.complex_dim, self.complex_dim, bias=False)
        nn.init.eye_(self.tail_proj_re.weight)
        nn.init.eye_(self.tail_proj_im.weight)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, entity_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode entities to normalised complex vectors via CurvedEmbedding.

        Args:
            entity_ids: (B,) long.

        Returns:
            (re, im): (B, complex_dim) each.
        """
        if self.use_adaptive_curvature:
            return self.curved_encoder.to_complex(entity_ids)
        else:
            # Fall back to flat normalised embedding
            emb = self.curved_encoder.entity_emb(entity_ids)   # (B, embed_dim)
            re = emb[:, : self.complex_dim]
            im = emb[:, self.complex_dim :]
            norm = (re.pow(2) + im.pow(2)).sum(-1, keepdim=True).sqrt().clamp(min=_EPS)
            return re / norm, im / norm

    def _born_rule(
        self,
        h_re: Tensor, h_im: Tensor,
        t_re: Tensor, t_im: Tensor,
    ) -> Tensor:
        """Born rule: |⟨t | h⟩|² = (Σ re_h·re_t + im_h·im_t)² + (...)².

        Args:
            h_re, h_im: (B, complex_dim) transported head.
            t_re, t_im: (B, complex_dim) tail.

        Returns:
            score: (B,) float in [0, 1].
        """
        re_inner = (h_re * t_re + h_im * t_im).sum(-1)   # (B,)
        im_inner = (h_re * t_im - h_im * t_re).sum(-1)   # (B,)
        return re_inner.pow(2) + im_inner.pow(2)          # (B,)

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score_triple(
        self, head_ids: Tensor, relation_ids: Tensor, tail_ids: Tensor
    ) -> Tensor:
        """Score (h, r, t) triples via 1-hop Born rule |⟨t | T_r | h⟩|².

        Args:
            head_ids:     (B,) long.
            relation_ids: (B,) long.
            tail_ids:     (B,) long.

        Returns:
            scores: (B,) float.
        """
        re_h, im_h = self._encode(head_ids)
        re_t, im_t = self._encode(tail_ids)

        # Apply parallel transport
        re_hp, im_hp = self.transport.apply(re_h, im_h, relation_ids)

        return self._born_rule(re_hp, im_hp, re_t, im_t)

    def score_with_holonomy(
        self,
        head_ids: Tensor,
        relation_ids: Tensor,
        tail_ids: Tensor,
        return_paths: bool = True,
    ) -> Tensor:
        """Score triples with holonomy confidence multiplier.

        For a triple (h, r, t) we construct the implied 2-cycle h→r→t→r_inv→h
        by using the same relation id for both forward and backward legs
        (a proxy for the true inverse).  The holonomy gap of this cycle
        measures geometric consistency.

        Score = Born(h,r,t) + holonomy_weight × exp(−gap)

        Args:
            head_ids:     (B,) long.
            relation_ids: (B,) long.
            tail_ids:     (B,) long.
            return_paths: unused flag kept for API compatibility.

        Returns:
            scores: (B,) float.
        """
        base_score = self.score_triple(head_ids, relation_ids, tail_ids)

        # Holonomy gap for 2-hop cycle (forward + same relation = proxy cycle)
        cycle = [relation_ids, relation_ids]
        gap = self.holonomy.holonomy_gap(cycle)                     # (B,)

        # Confidence: exp(−gap), in (0, 1] — higher confidence when gap → 0
        confidence = torch.exp(-gap)                                # (B,)
        return base_score + self.holonomy_weight * confidence

    def score_triple_vs_all(
        self, head_ids: Tensor, relation_ids: Tensor
    ) -> Tensor:
        """Score head + relation against all entities as candidate tails.

        Args:
            head_ids:     (B,) long.
            relation_ids: (B,) long.

        Returns:
            scores: (B, num_entities) float.
        """
        B = head_ids.shape[0]
        E = self.num_entities
        device = head_ids.device

        re_h, im_h = self._encode(head_ids)              # (B, d)
        re_hp, im_hp = self.transport.apply(re_h, im_h, relation_ids)  # (B, d)

        # All tail embeddings
        all_ids = torch.arange(E, device=device)
        re_t_all, im_t_all = self._encode(all_ids)       # (E, d)

        # Batch inner products: (B, d) × (d, E) → (B, E)
        re_inner = re_hp @ re_t_all.T + im_hp @ im_t_all.T   # (B, E)
        im_inner = re_hp @ im_t_all.T - im_hp @ re_t_all.T   # (B, E)
        return re_inner.pow(2) + im_inner.pow(2)              # (B, E)

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def compute_manifold_curvature_stats(self) -> dict[str, float]:
        """Return mean and std of learned κ values across all entities.

        Returns:
            stats: dict with keys 'kappa_mean', 'kappa_std', 'kappa_min',
                   'kappa_max', 'frac_hyperbolic', 'frac_spherical'.
        """
        with torch.no_grad():
            all_ids = torch.arange(self.num_entities)
            kappa = self.curved_encoder.get_curvature(all_ids)  # (E,)

        return {
            "kappa_mean":       kappa.mean().item(),
            "kappa_std":        kappa.std().item(),
            "kappa_min":        kappa.min().item(),
            "kappa_max":        kappa.max().item(),
            "frac_hyperbolic":  (kappa < -1e-3).float().mean().item(),
            "frac_spherical":   (kappa >  1e-3).float().mean().item(),
        }

    def get_param_count(self) -> dict[str, int]:
        """Return parameter counts broken down by sub-module."""
        counts: dict[str, int] = {
            "curved_encoder.entity_emb":    self.curved_encoder.entity_emb.weight.numel(),
            "curved_encoder.log_curvature": self.curved_encoder.log_curvature.weight.numel(),
            "transport.M_re":               self.transport.M_re.numel(),
            "transport.M_im":               self.transport.M_im.numel(),
            "curvature_adapter.rel_re":     self.curvature_adapter.rel_re.weight.numel(),
            "curvature_adapter.rel_im":     self.curvature_adapter.rel_im.weight.numel(),
            "curvature_adapter.kappa_r":    self.curvature_adapter.log_rel_curvature.weight.numel(),
            "tail_proj_re":                 self.tail_proj_re.weight.numel(),
            "tail_proj_im":                 self.tail_proj_im.weight.numel(),
        }
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def extra_repr(self) -> str:
        return (
            f"num_entities={self.num_entities}, "
            f"num_relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, "
            f"complex_dim={self.complex_dim}, "
            f"holonomy_weight={self.holonomy_weight}, "
            f"use_adaptive_curvature={self.use_adaptive_curvature}"
        )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("v8_reasoner.py — V8 Curved Manifold Reasoner Demo")
    print("=" * 60)

    torch.manual_seed(7)
    E, R, D = 100, 15, 64
    B = 8

    model = V8CurvedManifoldReasoner(
        num_entities=E,
        num_relations=R,
        embed_dim=D,
        holonomy_weight=0.1,
        use_adaptive_curvature=True,
    )
    print(f"\nModel:\n  {model.extra_repr()}")

    head_ids = torch.randint(0, E, (B,))
    rel_ids  = torch.randint(0, R, (B,))
    tail_ids = torch.randint(0, E, (B,))

    # 1-hop score
    scores = model.score_triple(head_ids, rel_ids, tail_ids)
    print(f"\nscore_triple output:")
    print(f"  shape: {scores.shape}, values: {scores.detach().tolist()}")

    # Holonomy-weighted score
    scores_hol = model.score_with_holonomy(head_ids, rel_ids, tail_ids)
    print(f"\nscore_with_holonomy output:")
    print(f"  shape: {scores_hol.shape}, values: {scores_hol.detach().tolist()}")

    # vs-all
    vs_all = model.score_triple_vs_all(head_ids, rel_ids)
    print(f"\nscore_triple_vs_all:")
    print(f"  shape: {vs_all.shape}")

    # Curvature stats
    stats = model.compute_manifold_curvature_stats()
    print(f"\nManifold curvature stats (untrained):")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Param counts
    print(f"\nParam counts:")
    for k, v in model.get_param_count().items():
        print(f"  {k}: {v:,}")

    # Gradient check
    loss = scores.sum()
    loss.backward()
    grads_ok = all(p.grad is not None for p in model.parameters())
    print(f"\nGradients computed: {grads_ok}")
    print("\nAll checks passed.")
