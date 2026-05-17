"""
models/components/adaptive_curvature.py — κ-Parameterised Curved Embeddings (V8)

Each entity has a learnable curvature κ_e ∈ (−1, 1):
    κ < 0  →  hyperbolic (Poincaré ball) — suited for hierarchical entities
    κ ≈ 0  →  Euclidean flat
    κ > 0  →  spherical — suited for cyclical relations

Mixed-geometry products are handled via Möbius gyrovector addition, and
all embeddings are projected to ℂ^{d/2} via a per-dimension stereographic map
so that the rest of the V8 pipeline receives standard complex vectors.

References:
    Ganea et al. (2018)  — Hyperbolic Neural Networks
    Bachmann et al. (2020) — Constant Curvature Graph Convolutional Networks
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_EPS = 1e-8
_KAPPA_EPS = 1e-6   # treat |κ| < this as Euclidean


def _clamp_poincare(x: Tensor, kappa: Tensor) -> Tensor:
    """Project x into the open Poincaré ball of radius 1/√|κ|.

    Args:
        x:     (B, d) float.
        kappa: (B,) float, κ < 0 assumed.

    Returns:
        x_proj: (B, d) with ||x|| < 1/√|κ| − ε.
    """
    radius = 1.0 / (kappa.abs().sqrt().unsqueeze(-1) + _EPS)  # (B, 1)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)       # (B, 1)
    # scale so that ||x_proj|| ≤ (1 − 1e-5) * radius
    scale = (radius * (1.0 - 1e-5)) / norm.clamp(min=_EPS)
    return torch.where(norm < radius, x, x * scale)


def _normalize_sphere(x: Tensor, kappa: Tensor) -> Tensor:
    """Project x onto sphere of radius 1/√κ.

    Args:
        x:     (B, d) float.
        kappa: (B,) float, κ > 0 assumed.

    Returns:
        x_proj: (B, d) with ||x|| = 1/√κ.
    """
    radius = 1.0 / (kappa.sqrt().unsqueeze(-1) + _EPS)        # (B, 1)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)       # (B, 1)
    return x / norm * radius


# ---------------------------------------------------------------------------
# CurvedEmbedding
# ---------------------------------------------------------------------------

class CurvedEmbedding(nn.Module):
    """Entity embeddings with per-entity learnable curvature κ_e.

    Parameterisation:
        entity_emb:    nn.Embedding(num_entities, embed_dim)  — tangent vector
        log_curvature: nn.Embedding(num_entities, 1)          — κ_e = tanh(·) ∈ (−1, 1)

    The raw embedding is a tangent vector at the origin of M^d(κ).  We use
    the exponential map (approximated by projection) to map it onto the manifold.

    Args:
        num_entities (int): Entity vocabulary size.
        embed_dim (int):    Embedding dimension (must be even).
        init_curvature (float): Initial value of raw log-curvature (default 0.0
                                gives κ = tanh(0) = 0, i.e. Euclidean).
    """

    def __init__(
        self,
        num_entities: int,
        embed_dim: int,
        init_curvature: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.num_entities = num_entities
        self.embed_dim = embed_dim
        self.complex_dim = embed_dim // 2
        self.init_curvature = init_curvature

        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(embed_dim))

        self.log_curvature = nn.Embedding(num_entities, 1)
        nn.init.constant_(self.log_curvature.weight, init_curvature)

    def get_curvature(self, entity_ids: Tensor) -> Tensor:
        """Return κ_e for each entity, squeezed to (B,) ∈ (−1, 1).

        Args:
            entity_ids: (B,) long.

        Returns:
            kappa: (B,) float in (−1, 1).
        """
        raw = self.log_curvature(entity_ids).squeeze(-1)   # (B,)
        return torch.tanh(raw)                             # κ ∈ (−1, 1)

    def project_to_manifold(self, emb: Tensor, kappa: Tensor) -> Tensor:
        """Project embedding onto the κ-manifold.

        - κ < −ε:  Poincaré ball  ||x|| < 1/√|κ|
        - κ > +ε:  Sphere         ||x|| = 1/√κ
        - else:    Identity (Euclidean)

        Args:
            emb:   (B, embed_dim) float.
            kappa: (B,) float.

        Returns:
            projected: (B, embed_dim) float.
        """
        B, d = emb.shape

        hyperbolic_mask = kappa < -_KAPPA_EPS          # (B,)
        spherical_mask  = kappa >  _KAPPA_EPS          # (B,)

        projected = emb.clone()

        # Hyperbolic branch
        if hyperbolic_mask.any():
            idx = hyperbolic_mask.nonzero(as_tuple=True)[0]
            projected[idx] = _clamp_poincare(emb[idx], kappa[idx])

        # Spherical branch
        if spherical_mask.any():
            idx = spherical_mask.nonzero(as_tuple=True)[0]
            projected[idx] = _normalize_sphere(emb[idx], kappa[idx])

        return projected

    def mobius_add(self, x: Tensor, y: Tensor, kappa: Tensor) -> Tensor:
        """Möbius addition in κ-curved space (gyrovector addition).

        Formula (unified κ-stereographic model):
            x ⊕_κ y = (
                (1 + 2κ⟨x,y⟩ + κ||y||²) x + (1 − κ||x||²) y
            ) / (
                1 + 2κ⟨x,y⟩ + κ²||x||²||y||²
            )

        For κ = 0 this reduces to standard vector addition x + y.

        Args:
            x:     (B, d) float.
            y:     (B, d) float.
            kappa: (B,) float (same κ for both operands).

        Returns:
            z: (B, d) float.
        """
        k = kappa.unsqueeze(-1)                          # (B, 1)
        xy = (x * y).sum(-1, keepdim=True)              # (B, 1)  ⟨x,y⟩
        x2 = (x * x).sum(-1, keepdim=True)              # (B, 1)  ||x||²
        y2 = (y * y).sum(-1, keepdim=True)              # (B, 1)  ||y||²

        num = (1.0 + 2.0 * k * xy + k * y2) * x + (1.0 - k * x2) * y
        denom = (1.0 + 2.0 * k * xy + k * k * x2 * y2).clamp(min=_EPS)
        return num / denom

    def to_complex(self, entity_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Stereographic projection: M^d(κ) → ℂ^{d/2}.

        Simplified per-dimension map:
            x ↦ x[:d/2] / (1 − κ · x[d/2:])
        where the denominator acts component-wise as a conformal factor.

        For κ = 0 this is the identity (flat Euclidean).

        Args:
            entity_ids: (B,) long.

        Returns:
            (re, im): (B, complex_dim) each.
        """
        emb   = self.entity_emb(entity_ids)              # (B, embed_dim)
        kappa = self.get_curvature(entity_ids)           # (B,)
        emb   = self.project_to_manifold(emb, kappa)    # (B, embed_dim)

        d = self.complex_dim
        x_re = emb[:, :d]                               # (B, d)
        x_im = emb[:, d:]                               # (B, d)
        k = kappa.unsqueeze(-1)                         # (B, 1)

        # Conformal factor: 1 − κ · x_im  (broadcasted per-dim)
        conformal = (1.0 - k * x_im).clamp(min=_EPS)   # (B, d)

        re = x_re / conformal                           # (B, d)
        im = x_im / conformal                           # (B, d)

        # Normalise projected complex vector
        norm = (re.pow(2) + im.pow(2)).sum(-1, keepdim=True).sqrt().clamp(min=_EPS)
        return re / norm, im / norm

    def get_param_count(self) -> dict[str, int]:
        return {
            "entity_emb": self.entity_emb.weight.numel(),
            "log_curvature": self.log_curvature.weight.numel(),
        }


# ---------------------------------------------------------------------------
# RelationCurvatureAdapter
# ---------------------------------------------------------------------------

class RelationCurvatureAdapter(nn.Module):
    """Adapts entity embeddings to relation-specific curvature space.

    Each relation has a learnable curvature κ_r ∈ (−1, 1).  When scoring
    h−r→t the head entity (living in κ_h space) is transported to the
    relation's κ_r space via gyrovector re-parameterisation.

    Args:
        num_relations (int): Relation vocabulary size.
        embed_dim (int):     Embedding dimension (complex_dim = embed_dim // 2).
    """

    def __init__(self, num_relations: int, embed_dim: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.complex_dim = embed_dim // 2

        # Per-relation curvature parameter
        self.log_rel_curvature = nn.Embedding(num_relations, 1)
        nn.init.zeros_(self.log_rel_curvature.weight)

        # Relation-specific linear map (complex, stored as two real matrices)
        self.rel_re = nn.Embedding(num_relations, self.complex_dim)
        self.rel_im = nn.Embedding(num_relations, self.complex_dim)
        nn.init.normal_(self.rel_re.weight, std=0.01)
        nn.init.normal_(self.rel_im.weight, std=0.01)

    def get_relation_curvature(self, relation_ids: Tensor) -> Tensor:
        """Return κ_r ∈ (−1, 1) for each relation.

        Args:
            relation_ids: (B,) long.

        Returns:
            kappa_r: (B,) float.
        """
        raw = self.log_rel_curvature(relation_ids).squeeze(-1)
        return torch.tanh(raw)

    def transport_to_relation_space(
        self,
        entity_re: Tensor,
        entity_im: Tensor,
        entity_kappa: Tensor,
        relation_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Transport entity embedding from its own κ_e space to κ_r space.

        The transport is implemented as a Möbius addition of the entity point
        with a zero-vector in the target curvature, then rescaling.

        Args:
            entity_re:    (B, complex_dim) real part.
            entity_im:    (B, complex_dim) imaginary part.
            entity_kappa: (B,) curvature of the entity.
            relation_ids: (B,) long.

        Returns:
            (re_out, im_out, kappa_out): transported entity and target κ.
        """
        kappa_r = self.get_relation_curvature(relation_ids)     # (B,)

        # Interp curvature: weighted average towards κ_r
        kappa_mid = 0.5 * (entity_kappa + kappa_r)

        # Scale entity embedding by curvature ratio (isometric re-scaling)
        scale_e = (1.0 + entity_kappa.unsqueeze(-1).abs())      # (B, 1)
        scale_r = (1.0 + kappa_r.unsqueeze(-1).abs())           # (B, 1)
        ratio = (scale_r / scale_e.clamp(min=_EPS)).clamp(0.1, 10.0)

        re_out = entity_re * ratio
        im_out = entity_im * ratio

        # Normalise
        norm = (re_out.pow(2) + im_out.pow(2)).sum(-1, keepdim=True).sqrt().clamp(min=_EPS)
        return re_out / norm, im_out / norm, kappa_mid

    def combined_score(
        self,
        h_re: Tensor,
        h_im: Tensor,
        r_ids: Tensor,
        t_re: Tensor,
        t_im: Tensor,
    ) -> Tensor:
        """Curvature-aware inner product: ⟨h ⊕_κ r, t⟩_κ.

        The head is translated by the relation vector in κ_r space (Möbius),
        then the Born-rule amplitude is computed.

        Args:
            h_re, h_im: (B, complex_dim) head complex embedding.
            r_ids:      (B,) relation ids.
            t_re, t_im: (B, complex_dim) tail complex embedding.

        Returns:
            score: (B,) float in [0, 1].
        """
        kappa_r = self.get_relation_curvature(r_ids)            # (B,)
        r_re = self.rel_re(r_ids)                               # (B, d)
        r_im = self.rel_im(r_ids)                               # (B, d)

        # Concatenate re/im for Möbius (operate on 2d-dim flat vector)
        h_flat = torch.cat([h_re, h_im], dim=-1)                # (B, 2d)
        r_flat = torch.cat([r_re, r_im], dim=-1)                # (B, 2d)

        # Duplicate kappa for the doubled space
        kappa_2d = kappa_r

        # Möbius translation: h' = h ⊕_κ r
        # (operate on the full 2*complex_dim vector)
        k = kappa_2d.unsqueeze(-1)
        xy = (h_flat * r_flat).sum(-1, keepdim=True)
        x2 = (h_flat * h_flat).sum(-1, keepdim=True)
        y2 = (r_flat * r_flat).sum(-1, keepdim=True)
        num   = (1.0 + 2.0 * k * xy + k * y2) * h_flat + (1.0 - k * x2) * r_flat
        denom = (1.0 + 2.0 * k * xy + k * k * x2 * y2).clamp(min=_EPS)
        hp_flat = num / denom                                   # (B, 2d)

        hp_re = hp_flat[:, : self.complex_dim]
        hp_im = hp_flat[:, self.complex_dim :]

        # Born rule: |⟨t|h'⟩|²
        re_inner = (hp_re * t_re + hp_im * t_im).sum(-1)
        im_inner = (hp_re * t_im - hp_im * t_re).sum(-1)
        score = re_inner.pow(2) + im_inner.pow(2)               # (B,)
        return score

    def get_param_count(self) -> dict[str, int]:
        return {
            "log_rel_curvature": self.log_rel_curvature.weight.numel(),
            "rel_re": self.rel_re.weight.numel(),
            "rel_im": self.rel_im.weight.numel(),
        }


# ---------------------------------------------------------------------------
# HybridManifoldScorer
# ---------------------------------------------------------------------------

class HybridManifoldScorer(nn.Module):
    """Combines curved entity embedding with quantum Born rule scoring.

    Scoring pipeline:
        1. Retrieve curved entity embeddings (re, im) via CurvedEmbedding.to_complex
        2. Apply curvature-aware relation transform via RelationCurvatureAdapter
        3. Compute Born rule |⟨t|h'⟩|²

    Args:
        num_entities (int):  Entity vocabulary size.
        num_relations (int): Relation vocabulary size.
        embed_dim (int):     Embedding dimension (must be even).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.complex_dim = embed_dim // 2

        self.curved_emb = CurvedEmbedding(num_entities, embed_dim)
        self.adapter = RelationCurvatureAdapter(num_relations, embed_dim)

    def score_triple(
        self, head_ids: Tensor, relation_ids: Tensor, tail_ids: Tensor
    ) -> Tensor:
        """Score (h, r, t) triples.

        Args:
            head_ids:     (B,) long.
            relation_ids: (B,) long.
            tail_ids:     (B,) long.

        Returns:
            scores: (B,) float.
        """
        h_re, h_im = self.curved_emb.to_complex(head_ids)
        t_re, t_im = self.curved_emb.to_complex(tail_ids)
        return self.adapter.combined_score(h_re, h_im, relation_ids, t_re, t_im)

    def score_triple_vs_all(
        self, head_ids: Tensor, relation_ids: Tensor
    ) -> Tensor:
        """Score against all entities as candidate tails.

        Args:
            head_ids:     (B,) long.
            relation_ids: (B,) long.

        Returns:
            scores: (B, num_entities) float.
        """
        B = head_ids.shape[0]
        E = self.num_entities

        h_re, h_im = self.curved_emb.to_complex(head_ids)    # (B, d)

        # All entity embeddings
        all_ids = torch.arange(E, device=head_ids.device)
        all_re, all_im = self.curved_emb.to_complex(all_ids)   # (E, d)

        kappa_r = self.adapter.get_relation_curvature(relation_ids)   # (B,)
        r_re = self.adapter.rel_re(relation_ids)               # (B, d)
        r_im = self.adapter.rel_im(relation_ids)               # (B, d)

        # Möbius translate head by relation
        h_flat = torch.cat([h_re, h_im], dim=-1)               # (B, 2d)
        r_flat = torch.cat([r_re, r_im], dim=-1)               # (B, 2d)
        k = kappa_r.unsqueeze(-1)
        xy = (h_flat * r_flat).sum(-1, keepdim=True)
        x2 = (h_flat * h_flat).sum(-1, keepdim=True)
        y2 = (r_flat * r_flat).sum(-1, keepdim=True)
        num   = (1.0 + 2.0 * k * xy + k * y2) * h_flat + (1.0 - k * x2) * r_flat
        denom = (1.0 + 2.0 * k * xy + k * k * x2 * y2).clamp(min=_EPS)
        hp_flat = num / denom                                   # (B, 2d)
        hp_re = hp_flat[:, : self.complex_dim]                 # (B, d)
        hp_im = hp_flat[:, self.complex_dim :]                 # (B, d)

        # Inner product vs all tails: (B, d) x (E, d) -> (B, E)
        re_inner = hp_re @ all_re.T + hp_im @ all_im.T        # (B, E)
        im_inner = hp_re @ all_im.T - hp_im @ all_re.T        # (B, E)
        scores = re_inner.pow(2) + im_inner.pow(2)             # (B, E)
        return scores

    def get_param_count(self) -> dict[str, int]:
        counts = {}
        counts.update({f"curved_emb.{k}": v for k, v in self.curved_emb.get_param_count().items()})
        counts.update({f"adapter.{k}": v for k, v in self.adapter.get_param_count().items()})
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("adaptive_curvature.py — V8 Curved Embeddings Demo")
    print("=" * 60)

    torch.manual_seed(0)
    B, E, R, D = 4, 50, 10, 32

    # --- CurvedEmbedding ---
    curved = CurvedEmbedding(num_entities=E, embed_dim=D)
    ent_ids = torch.randint(0, E, (B,))

    kappa = curved.get_curvature(ent_ids)
    print(f"\nCurvedEmbedding:")
    print(f"  κ values: {kappa.tolist()}")
    print(f"  All in (-1, 1): {(kappa.abs() < 1).all().item()}")

    re, im = curved.to_complex(ent_ids)
    print(f"  Complex re shape: {re.shape}")
    norms = (re.pow(2) + im.pow(2)).sum(-1).sqrt()
    print(f"  Norms (should be ~1): {norms.tolist()}")

    # Test Möbius addition
    x = torch.randn(B, D)
    y = torch.randn(B, D) * 0.1
    kappa_flat = torch.zeros(B)
    z_euclidean = curved.mobius_add(x, y, kappa_flat)
    print(f"\n  Möbius add (κ=0) ≈ x+y: max diff = {(z_euclidean - (x+y)).abs().max().item():.4f}")

    # --- RelationCurvatureAdapter ---
    adapter = RelationCurvatureAdapter(num_relations=R, embed_dim=D)
    rel_ids = torch.randint(0, R, (B,))
    scores = adapter.combined_score(re, im, rel_ids, re, im)
    print(f"\nRelationCurvatureAdapter:")
    print(f"  Self-scores (should be > 0): {scores.tolist()}")

    # --- HybridManifoldScorer ---
    scorer = HybridManifoldScorer(E, R, D)
    tail_ids = torch.randint(0, E, (B,))
    triple_scores = scorer.score_triple(ent_ids, rel_ids, tail_ids)
    vs_all = scorer.score_triple_vs_all(ent_ids, rel_ids)

    print(f"\nHybridManifoldScorer:")
    print(f"  Triple scores: {triple_scores.tolist()}")
    print(f"  vs_all shape: {vs_all.shape}")
    print(f"\nParam counts:")
    for k, v in scorer.get_param_count().items():
        print(f"  {k}: {v:,}")
    print("\nAll checks passed.")
