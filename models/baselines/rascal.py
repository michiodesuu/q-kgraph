"""
models/baselines/rascal.py — RASCAL Baseline

RASCAL: Relational Asymmetric Scoring via bilinear Contraction with Adaptive Learned matrices
Follows the RESCAL formulation (Nickel et al., ICML 2011).
score(h, r, t) = h^T · M_r · t   (full d×d bilinear matrix per relation)

WHAT RASCAL CAN MODEL (vs TransE/RotatE/ComplEx):
    TransE:   r encoded as a single d-vector translation. O(|R|d) params.
    RotatE:   r encoded as d rotation angles. O(|R|d) params.
    ComplEx:  r encoded as a d-complex vector. O(|R|d) params.
    RASCAL:   r encoded as a full d×d matrix M_r. O(|R|d²) params.

    The full matrix M_r can capture EVERY bilinear interaction between
    dimensions of h and t, mediated by relation r. It strictly subsumes
    TransE (diagonal of symmetric M_r) and ComplEx (real part of Hermitian M_r).
    RASCAL is the most expressive shallow embedding model.

WHY RASCAL STILL CANNOT PRODUCE INTERFERENCE:
    RASCAL score = h^T M_r t = Σᵢⱼ hᵢ (M_r)ᵢⱼ tⱼ  [single real scalar]

    This is a DIRECT triple score, not a path sum:
    - No summation over K reasoning paths
    - No complex amplitudes
    - No cross-terms between different multi-hop chains
    - Any "path aggregation" would be classical addition: Σₖ h^T M_{rₖ} t ≥ 0

    Even if M_r were complex, RASCAL computes ONE interaction per triple.
    The interference cross-term 2Re(A₁Ā₂) requires SQUARING A SUM, which
    demands at least K=2 paths. RASCAL has K=0 (no path enumeration at all).

    RASCAL proves: even full bilinear matrices per relation cannot produce
    the destructive interference that QuantumReasoner achieves via path sums.

PARAMETER COUNT COMPARISON (embed_dim=256, FB15k-237):
    TransE:        (14541 + 237) × 256           =   3.8M
    ComplEx:       (14541 + 237) × 256 × 2       =   7.6M
    RASCAL:        14541 × 256 + 237 × 256²      =  15.3M  ← memory intensive
    QuantumReasoner: 14541 × 256 × 2 + 237 × 256 =   7.5M  ← matches ComplEx

    For toy KG (28 entities, 12 relations, embed_dim=64):
    RASCAL:        28 × 64 + 12 × 64²           =  51K  (fine)

PUBLISHED NUMBERS (paper Table 2):
    FB15k-237: MRR ≈ 0.356, Hits@1 ≈ 0.264, Hits@10 ≈ 0.530
    WN18RR:    MRR ≈ 0.467, Hits@1 ≈ 0.439, Hits@10 ≈ 0.517

USAGE:
    model = RASCAL(num_entities=28, num_relations=12, embed_dim=64)
    scores = model.score_triple(h, r, t)         # (B,) float
    scores = model.score_triple_vs_all(h, r)     # (B, E) float
    reg    = model.regularization_loss()
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class RASCAL(nn.Module):
    """
    RASCAL: full bilinear scoring. score(h, r, t) = h^T M_r t.

    M_r is a d×d matrix per relation — the most expressive shallow KGE model.
    Memory: O(|R| · d²) for relation matrices. Use embed_dim ≤ 128 for FB15k-237.

    Args:
        num_entities:  Entity count.
        num_relations: Relation count.
        embed_dim:     Entity embedding dimension d.
        reg_weight:    L2 regularization weight.
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int   = 64,
        reg_weight:    float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.reg_weight    = reg_weight

        self.entity_embeddings   = nn.Embedding(num_entities, embed_dim)
        # Relation matrices: (num_relations, embed_dim, embed_dim)
        # Stored as a (num_relations, embed_dim*embed_dim) embedding and reshaped
        self.relation_matrices = nn.Embedding(num_relations, embed_dim * embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for entity embeddings; near-identity init for relation matrices."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        # Initialize M_r near identity to avoid gradient vanishing at start
        with torch.no_grad():
            eye = torch.eye(self.embed_dim).flatten()             # (d²,)
            nn.init.normal_(self.relation_matrices.weight, mean=0.0, std=0.01)
            self.relation_matrices.weight.data += eye.unsqueeze(0)

    def _get_relation_matrices(
        self,
        relation_ids: torch.Tensor,    # (B,) int
    ) -> torch.Tensor:
        """
        Return relation matrices shaped (B, d, d).
        """
        flat = self.relation_matrices(relation_ids)       # (B, d*d)
        return flat.view(-1, self.embed_dim, self.embed_dim)  # (B, d, d)

    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        RASCAL score: h^T M_r t.

        Computes the bilinear form for each triple in the batch.

        Returns:
            (B,) float scores. Unbounded (can be positive or negative).
        """
        h  = self.entity_embeddings(head_ids)           # (B, d)
        t  = self.entity_embeddings(tail_ids)           # (B, d)
        Mr = self._get_relation_matrices(relation_ids)  # (B, d, d)

        # h^T M_r t = h · (M_r t)
        # Step 1: M_r @ t → column vector (B, d, 1)
        Mrt = torch.bmm(Mr, t.unsqueeze(-1))            # (B, d, 1)
        # Step 2: h^T @ (M_r t) → scalar (B, 1, 1) → (B,)
        score = torch.bmm(h.unsqueeze(1), Mrt).squeeze(-1).squeeze(-1)  # (B,)
        return score

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as candidate tails: score[b, e] = h[b]^T M_r[b] all_e[e].

        Efficient vectorized form:
            M_r_h[b, j] = Σᵢ h[b,i] M_r[b,i,j] = (h @ M_r)[b,j]   (row vector)
            score[b, e] = M_r_h[b] · all_e[e] = M_r_h @ all_e.T

        Returns:
            (B, num_entities) float score matrix.
        """
        h  = self.entity_embeddings(head_ids)           # (B, d)
        Mr = self._get_relation_matrices(relation_ids)  # (B, d, d)

        # h^T M_r → row vector (B, d)
        # h.unsqueeze(1) @ Mr = (B, 1, d) @ (B, d, d) = (B, 1, d) → (B, d)
        Mr_h = torch.bmm(h.unsqueeze(1), Mr).squeeze(1)  # (B, d)

        all_e = self.entity_embeddings.weight            # (E, d)
        scores = Mr_h @ all_e.T                          # (B, E)
        return scores

    def regularization_loss(self) -> torch.Tensor:
        """L2 regularization on entity embeddings and relation matrices."""
        return self.reg_weight * (
            self.entity_embeddings.weight.pow(2).sum()
            + self.relation_matrices.weight.pow(2).sum()
        )

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "entity_emb":       self.entity_embeddings.weight.numel(),
            "relation_matrices": self.relation_matrices.weight.numel(),
            "total":            total,
        }

    def extra_repr(self) -> str:
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, "
            f"relation_params={self.num_relations * self.embed_dim ** 2:,}"
        )
