"""
models/baselines/tucker.py — TuckER Baseline

TuckER: Tensor Factorization for Knowledge Graph Completion (Balazevic et al., EMNLP 2019)
arxiv.org/abs/1901.09590

score(h, r, t) = σ( W ×₁ h ×₂ r · t )

Where:
    W      : core tensor of shape (d_e, d_r, d_e)
    ×₁     : mode-1 product with entity vector h ∈ ℝ^d_e
    ×₂     : mode-2 product with relation vector r ∈ ℝ^d_r
    ·      : dot product with tail entity t ∈ ℝ^d_e

Expanding W ×₁ h gives a (d_r, d_e) matrix M_h:
    M_h[j, k] = Σᵢ W[i,j,k] h[i]

Then W ×₁ h ×₂ r gives a d_e vector v_{hr}:
    v_{hr}[k] = Σⱼ M_h[j,k] r[j]

Final score: v_{hr} · t = Σₖ v_{hr}[k] t[k]

WHY TuckER IS AN IMPORTANT ADDITIONAL BASELINE:
    TuckER's core tensor W jointly models all three-way interactions between
    entity dimensions, relation dimensions, and tail entity dimensions.
    It generalises DistMult, RESCAL (RASCAL), and ComplEx as special cases.

    TuckER vs RASCAL:
        RASCAL (RESCAL): M_r is a free d×d matrix → effectively d·d_e "slices" of W
                         TuckER factors W across entity and relation subspaces.
                         TuckER uses d_r ≤ d_e (compressed relation dim), fewer params.

    TuckER vs QuantumReasoner:
        TuckER score = σ( W ×₁ h ×₂ r · t ) — a single real scalar per triple.
        No path enumeration. No complex amplitudes. No cross-terms.
        The W tensor is FIXED per relation — it cannot accumulate K path signals
        and square the sum. Interference requires:
            1. K complex amplitudes  (TuckER has K=0)
            2. Summation before squaring  (TuckER squares each dimension independently)
            3. A learning objective that drives phase angles apart  (no imaginary parts)

        TuckER is the most expressive FACTORIZED embedding baseline.
        It proves that even three-way tensor factorization cannot replace
        the quantum path interference mechanism.

PARAMETER COMPARISON (d_e=200, d_r=200, FB15k-237):
    W core tensor:       200 × 200 × 200 = 8M params  (dominant)
    Entity embeddings:   14541 × 200     = 2.9M
    Relation embeddings: 237   × 200     = 47K
    Total:               ~11M params

    Practical setting: d_r=200 or d_r=64 (smaller d_r reduces W without harming much)

PUBLISHED NUMBERS (paper Table 2):
    FB15k-237: MRR ≈ 0.358, Hits@1 ≈ 0.266, Hits@10 ≈ 0.544  ← best shallow model
    WN18RR:    MRR ≈ 0.470, Hits@1 ≈ 0.443, Hits@10 ≈ 0.526

USAGE:
    model = TuckER(num_entities=28, num_relations=12, d_entity=200, d_relation=200)
    scores = model.score_triple(h, r, t)          # (B,) float
    scores = model.score_triple_vs_all(h, r)      # (B, E) float

IMPLEMENTATION EFFICIENCY:
    Naive three-mode product: O(d_e² · d_r) per triple — expensive.
    Batched efficient form via einsum:
        Step 1: W × h → (B, d_r, d_e)   via einsum "ijk,bi->bjk"
        Step 2: × r  → (B, d_e)         via einsum "bjk,bj->bk"
        Step 3: · t  → (B,)             via einsum "bk,bk->b"
    Or score_triple_vs_all via:
        Step 1+2 → (B, d_e) feature
        Step 3   → (B, E) via matmul with entity weight matrix
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TuckER(nn.Module):
    """
    TuckER: Tucker tensor factorization scoring.

    score(h, r, t) = σ( W ×₁ h ×₂ r · t )

    W is a shared d_entity × d_relation × d_entity core tensor.
    Entity embeddings: (num_entities, d_entity).
    Relation embeddings: (num_relations, d_relation). d_relation can differ from d_entity.

    Args:
        num_entities:      Entity count.
        num_relations:     Relation count.
        d_entity:          Entity embedding dimension (also tail scoring dimension).
        d_relation:        Relation embedding dimension. ≤ d_entity typically.
        input_dropout:     Dropout on input entity/relation embeddings.
        hidden_dropout1:   Dropout after W × h step.
        hidden_dropout2:   Dropout before final dot product.
        reg_weight:        L2 regularization weight.
    """

    def __init__(
        self,
        num_entities:    int,
        num_relations:   int,
        d_entity:        int   = 200,
        d_relation:      int   = 200,
        input_dropout:   float = 0.3,
        hidden_dropout1: float = 0.4,
        hidden_dropout2: float = 0.5,
        reg_weight:      float = 0.0,
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.d_entity      = d_entity
        self.d_relation    = d_relation
        self.reg_weight    = reg_weight

        self.entity_embeddings   = nn.Embedding(num_entities,  d_entity)
        self.relation_embeddings = nn.Embedding(num_relations, d_relation)

        # Core tensor W: (d_entity, d_relation, d_entity)
        self.W = nn.Parameter(torch.empty(d_entity, d_relation, d_entity))

        # Batch normalization on embeddings (from original TuckER paper)
        self.bn0 = nn.BatchNorm1d(d_entity)
        self.bn1 = nn.BatchNorm1d(d_entity)

        # Dropout layers
        self.drop_input    = nn.Dropout(input_dropout)
        self.drop_hidden1  = nn.Dropout(hidden_dropout1)
        self.drop_hidden2  = nn.Dropout(hidden_dropout2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for embeddings; normal for core tensor."""
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)
        nn.init.xavier_normal_(self.W)

    def _get_tucker_feature(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Compute W ×₁ h ×₂ r, the d_entity-dimensional feature vector.

        Returns:
            (B, d_entity) float feature vectors before dot product with t.
        """
        h = self.entity_embeddings(head_ids)        # (B, d_e)
        r = self.relation_embeddings(relation_ids)  # (B, d_r)

        h = self.bn0(h)
        h = self.drop_input(h)

        # W ×₁ h: "ijk,bi->bjk" → (B, d_r, d_e)
        # For each batch item b: Wh[b, j, k] = Σᵢ W[i,j,k] h[b,i]
        Wh = torch.einsum("ijk,bi->bjk", self.W, h)      # (B, d_r, d_e)
        Wh = self.drop_hidden1(Wh)

        # W ×₁ h ×₂ r: "bjk,bj->bk" → (B, d_e)
        # For each batch item b: Whr[b, k] = Σⱼ Wh[b,j,k] r[b,j]
        Whr = torch.einsum("bjk,bj->bk", Wh, r)          # (B, d_e)
        Whr = self.bn1(Whr)
        Whr = self.drop_hidden2(Whr)
        return Whr

    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        TuckER score: σ( W ×₁ h ×₂ r · t ).

        Returns:
            (B,) float scores in (0, 1) after sigmoid.
        """
        Whr = self._get_tucker_feature(head_ids, relation_ids)  # (B, d_e)
        t   = self.entity_embeddings(tail_ids)                   # (B, d_e)

        score = (Whr * t).sum(dim=-1)                            # (B,) logit
        return torch.sigmoid(score)

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as candidate tails.

        Returns:
            (B, num_entities) float score matrix (sigmoid-applied).
        """
        Whr   = self._get_tucker_feature(head_ids, relation_ids)  # (B, d_e)
        all_e = self.entity_embeddings.weight                      # (E, d_e)

        logits = Whr @ all_e.T                                     # (B, E)
        return torch.sigmoid(logits)

    def regularization_loss(self) -> torch.Tensor:
        """L2 regularization on entity embeddings, relation embeddings, and W."""
        if self.reg_weight == 0.0:
            return torch.tensor(0.0, device=self.W.device)
        return self.reg_weight * (
            self.entity_embeddings.weight.pow(2).sum()
            + self.relation_embeddings.weight.pow(2).sum()
            + self.W.pow(2).sum()
        )

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "entity_emb":    self.entity_embeddings.weight.numel(),
            "relation_emb":  self.relation_embeddings.weight.numel(),
            "core_tensor_W": self.W.numel(),
            "total":         total,
        }

    def extra_repr(self) -> str:
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"d_entity={self.d_entity}, d_relation={self.d_relation}, "
            f"W={self.d_entity}×{self.d_relation}×{self.d_entity}={self.W.numel():,}"
        )
