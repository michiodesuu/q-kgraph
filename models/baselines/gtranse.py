"""
models/baselines/gtranse.py — GTransE Baseline

GTransE: Generalizing Translation-based Model on Uncertain Knowledge Graph Embedding
Kertkeidkachorn, Liu, Ichise (AIST / NII, Tokyo)

GTransE is NOT a new scoring function. It is a new LOSS FUNCTION for uncertain KGs.
The scoring function is identical to TransE: f(h, r, t) = ||h + r - t||₁

THE KEY INNOVATION — Confidence-Scaled Margin Loss (Equation 2 in the paper):

    L = Σ_{(h,r,t,s) ∈ Q} Σ_{(h',r,t',s) ∈ Q'}  [ f(h,r,t) − f(h',r,t') + s^α · M ]+

Where:
    s ∈ [0,1]  : confidence score of the POSITIVE quadruple (h,r,t,s)
    α ≥ 0      : hyperparameter amplifying the influence of confidence
    M          : global margin constant
    [x]+       : max(0, x) — hinge loss

INTUITION (directly from the paper):
    Standard TransE margin loss uses a FIXED global margin M for every triple.
    But in uncertain KGs, each triple has a confidence s ∈ [0,1].

    The confidence-scaled margin s^α · M means:
        High confidence (s ≈ 1):  s^α · M ≈ M   → large margin, push hard
        Low confidence  (s ≈ 0):  s^α · M ≈ 0   → tiny margin, barely push
                                                   (this triple might be noise)

    The higher the uncertainty (low s), the less force is applied.
    GTransE uses ALL training data — it never removes uncertain triples.
    Instead, it lets them contribute with reduced influence.

SPECIAL CASES:
    α = 0: s^0 = 1 for all s → identical to standard TransE margin loss
    α → ∞: only triples with s = 1.0 contribute; equivalent to TransE(s=1)
    α = 2: best on NELL dataset (Table 2 in paper)
    α = 3: best on synthetic FB15k-237 datasets (Table 1 in paper)

CONTRAST WITH QuantumReasoner:
    Both models address uncertainty / noise in KGs:

    GTransE approach — CONFIDENCE MARGIN:
        "Don't push uncertain facts as hard."
        Mechanism: loss-level weighting by confidence score.
        Effect: gradients from low-s triples are scaled down proportionally.
        Limitation: still uses TransE scoring — no path cancellation, no
                    negative cross-terms between competing reasoning chains.

    QuantumReasoner approach — DESTRUCTIVE INTERFERENCE:
        "Actively cancel wrong paths with opposite-phase amplitudes."
        Mechanism: Born rule |Σᵢ αᵢ ⟨t|U_{Pᵢ}|h⟩|² produces cross-terms.
        Effect: wrong reasoning paths are driven to phase ≈ π and SUBTRACT
                from the probability, not just contribute less.
        Key difference: GTransE can only reduce wrong contribution;
                        QuantumReasoner can make wrong contribution NEGATIVE.

    Both are better than plain TransE on noisy/uncertain data.
    The question the paper answers: at 20% noise, which retains more accuracy?
    Expected: GTransE (α=3) > TransE; QuantumReasoner > GTransE at high noise
    because interference is a fundamentally stronger noise-rejection mechanism.

RELEVANCE TO NELL-995:
    NELL-995 provides per-triple confidence scores in [0.9, 1.0].
    GTransE was evaluated on exactly this dataset (Table 2 in paper).
    This makes GTransE the MOST DIRECTLY RELEVANT baseline for the NELL
    experiments introduced in V5. If QuantumReasoner > GTransE on NELL,
    that is the strongest empirical argument in the paper.

PUBLISHED NUMBERS (Table 2 — NELL dataset):
    GTransE α=1: Hits@1=11.11%, Hits@10=30.47%, MR=0.18
    GTransE α=2: Hits@1=12.08%, Hits@10=31.22%, MR=0.19  ← best on NELL
    GTransE α=3: Hits@1=12.20%, Hits@10=31.49%, MR=0.19
    GTransE α=4: Hits@1=12.21%, Hits@10=31.81%, MR=0.19

PUBLISHED NUMBERS (Table 1 — FB15k-237 synthetic, ρ=1.0, full uncertainty):
    GTransE α=3: Hits@1=10.8%, Hits@10=31.3%, MRR=0.18  ← best at ρ=1.0
    vs TransE(s≥0): Hits@1=3.5%, Hits@10=24.5%, MRR=0.10

USAGE:
    # Standard usage: scoring is TransE-identical
    model  = GTransE(num_entities=28, num_relations=12, embed_dim=64)
    scores = model.score_triple(h, r, t)          # (B,) float — same as TransE
    scores = model.score_triple_vs_all(h, r)      # (B, E) float

    # Confidence-scaled margin loss (requires confidence per positive triple)
    loss = model.confidence_margin_loss(
        pos_scores  = model.score_triple(h, r, t),       # (B,)
        neg_scores  = model.score_triple(h_neg, r, t_neg),  # (B*K,) or (B, K)
        confidence  = conf,                              # (B,) in [0,1]
        alpha       = 3.0,
    )

    # When confidence is unavailable (no uncertain KG), use standard margin loss:
    loss = model.margin_loss(pos_scores, neg_scores)  # α=0 equivalent

NOTE ON TRAINER INTEGRATION:
    The standard Trainer (training/trainer.py) does not pass confidence scores.
    For NELL-995 experiments: use NELLDataset (data/nell_dataset.py) which provides
    per-triple confidence. Wire confidence from batch["confidence"] to
    confidence_margin_loss() in the training loop.
    For non-NELL datasets: call margin_loss() or set confidence = torch.ones(B).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GTransE(nn.Module):
    """
    GTransE: TransE scoring + confidence-scaled margin loss for uncertain KGs.

    Scoring:  f(h, r, t) = ||h + r - t||_p   (identical to TransE)
    Loss:     L = Σ max(0, f_pos - f_neg + s^α · M)

    The model itself is TransE. The contribution of GTransE is the loss function
    that weights the margin by the per-triple confidence score.

    Args:
        num_entities:   Entity count.
        num_relations:  Relation count.
        embed_dim:      Embedding dimension.
        p_norm:         L_p norm for scoring. Default: 1 (L1, as used in paper).
        margin:         Global margin M. Default: 9.0.
        alpha:          Default confidence exponent. Paper best: 2–3.
                        Override per-call in confidence_margin_loss().
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int   = 64,
        p_norm:        int   = 1,
        margin:        float = 9.0,
        alpha:         float = 2.0,
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.p_norm        = p_norm
        self.margin        = margin
        self.alpha         = alpha

        self.entity_embeddings   = nn.Embedding(num_entities,  embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """TransE paper initialization: uniform in [-6/√d, 6/√d]; relations normalized."""
        bound = 6.0 / (self.embed_dim ** 0.5)
        nn.init.uniform_(self.entity_embeddings.weight,   -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        with torch.no_grad():
            norms = self.relation_embeddings.weight.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            self.relation_embeddings.weight.data /= norms

    # ── Scoring (identical to TransE) ────────────────────────────────────────

    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        TransE score: -||h + r - t||_p.  (SAME AS TransE — GTransE does not change this.)

        Returns:
            (B,) float scores. Higher (less negative) = more likely true.
        """
        h = self.entity_embeddings(head_ids)        # (B, d)
        r = self.relation_embeddings(relation_ids)  # (B, d)
        t = self.entity_embeddings(tail_ids)        # (B, d)
        diff  = h + r - t
        score = -diff.norm(p=self.p_norm, dim=-1)
        return score

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as candidate tails: -||h + r - e||_p for all e.

        Returns:
            (B, num_entities) float score matrix.
        """
        h  = self.entity_embeddings(head_ids)        # (B, d)
        r  = self.relation_embeddings(relation_ids)  # (B, d)
        hr = h + r                                    # (B, d)

        all_e = self.entity_embeddings.weight         # (E, d)
        diff  = hr.unsqueeze(1) - all_e.unsqueeze(0)  # (B, E, d)
        return -diff.norm(p=self.p_norm, dim=-1)      # (B, E)

    # ── Loss functions ────────────────────────────────────────────────────────

    def confidence_margin_loss(
        self,
        pos_scores:  torch.Tensor,   # (B,) — score_triple on positives
        neg_scores:  torch.Tensor,   # (B, K) or (B*K,) — score_triple on negatives
        confidence:  torch.Tensor,   # (B,) — per-positive confidence s ∈ [0,1]
        alpha:       float | None = None,
    ) -> torch.Tensor:
        """
        GTransE confidence-scaled margin loss (Equation 2 in the paper).

        L = Σ_b Σ_k max(0, f(h_b,r_b,t_b) − f(h'_k,r_b,t'_k) + s_b^α · M)

        Where f = -score (positive distance), so:
            f_pos = -pos_score  (small for correct triples)
            f_neg = -neg_score  (large for corrupted triples)

        Expanding:
            L = max(0, f_pos − f_neg + s^α·M)
              = max(0, (−pos_score) − (−neg_score) + s^α·M)
              = max(0, neg_score − pos_score + s^α·M)

        Args:
            pos_scores: (B,) TransE scores on positive triples. Higher = better.
            neg_scores: (B, K) or (B*K,) TransE scores on negative triples.
            confidence: (B,) confidence scores for each positive triple. ∈ [0, 1].
            alpha:      Confidence exponent. None → use self.alpha.

        Returns:
            Scalar loss.
        """
        alpha = alpha if alpha is not None else self.alpha

        if neg_scores.dim() == 1:
            # (B*K,) → infer K and reshape to (B, K)
            B = pos_scores.shape[0]
            K = neg_scores.shape[0] // B
            neg_scores = neg_scores.view(B, K)

        # Confidence-scaled effective margin: s^alpha * M — shape (B,)
        eff_margin = confidence.clamp(0.0, 1.0).pow(alpha) * self.margin  # (B,)

        # Loss per (positive, negative) pair: (B, K)
        # max(0, neg_score - pos_score + eff_margin)
        loss = F.relu(
            neg_scores - pos_scores.unsqueeze(1) + eff_margin.unsqueeze(1)
        )  # (B, K)

        return loss.mean()

    def margin_loss(
        self,
        pos_scores: torch.Tensor,   # (B,)
        neg_scores: torch.Tensor,   # (B, K) or (B*K,)
    ) -> torch.Tensor:
        """
        Standard (α=0) margin loss — identical to TransE.

        Equivalent to confidence_margin_loss(... , confidence=ones(B), alpha=0).
        Use this when confidence scores are unavailable (non-uncertain KG datasets).

        Returns:
            Scalar loss.
        """
        if neg_scores.dim() == 1:
            B = pos_scores.shape[0]
            K = neg_scores.shape[0] // B
            neg_scores = neg_scores.view(B, K)

        loss = F.relu(
            neg_scores - pos_scores.unsqueeze(1) + self.margin
        )
        return loss.mean()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def regularization_loss(self, weight: float = 1e-3) -> torch.Tensor:
        """L2 regularization on entity and relation embeddings (same as TransE)."""
        return weight * (
            self.entity_embeddings.weight.norm(p=2).pow(2)
            + self.relation_embeddings.weight.norm(p=2).pow(2)
        )

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "entity_emb":   self.entity_embeddings.weight.numel(),
            "relation_emb": self.relation_embeddings.weight.numel(),
            "total":        total,
        }

    def extra_repr(self) -> str:
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, p_norm={self.p_norm}, "
            f"margin={self.margin}, alpha={self.alpha}"
        )
