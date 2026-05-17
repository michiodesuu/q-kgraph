"""
Entanglement-Driven Graph Pruning for Efficient KGE Inference.

Connection to the AAAI-2024 "Complexity Paradox"
-------------------------------------------------
The AAAI-2024 analysis shows that quantum-inspired KGE models have a
"complexity paradox": for simple, deterministic (functional) relations,
the full quantum interference machinery is overkill — a bilinear scoring
function (DistMult, ComplEx) achieves near-identical accuracy at a fraction
of the compute cost.

This module resolves the paradox by ROUTING queries:
- Simple / functional relations → classical bilinear scoring
- Complex / many-to-many relations → full quantum interference scoring

Entropy as a Complexity Signal
-------------------------------
The BellStateRelation (V6) represents each relation r as a Bell-state matrix
M_r ∈ ℂ^{d×d}.  Its entanglement entropy S(r) = -Tr(ρ_r log ρ_r) where
ρ_r = M_r M_r† / ||M_r||_F² is the reduced density matrix.

- Low  S(r) → M_r is approximately rank-1 → relation is functional (1-to-1)
  → classical scoring is sufficient
- High S(r) → M_r is maximally entangled → relation is many-to-many
  → quantum interference adds measurable value

Integration with V6 BellStateRelation
--------------------------------------
EntanglementPruner expects pre-computed per-relation entropies from
BellStateRelation.entanglement_entropy_all() and uses them to make
routing decisions without re-computing expensive eigendecompositions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


# ---------------------------------------------------------------------------
# EntanglementPruner
# ---------------------------------------------------------------------------

class EntanglementPruner(nn.Module):
    """
    Uses Bell-state entanglement entropy S(r) to decide which relation
    branches to prune during inference, routing each query to either
    quantum or classical scoring.

    S(r) < entropy_threshold  →  prune / route to classical
    S(r) >= entropy_threshold  →  keep / route to quantum

    Args:
        num_relations:       R
        entropy_threshold:   prune relations with S(r) below this value
        prune_ratio:         hard cap — prune at most this fraction of branches
    """

    def __init__(
        self,
        num_relations: int,
        entropy_threshold: float = 0.5,
        prune_ratio: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 <= prune_ratio <= 1.0:
            raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}")
        self.num_relations = num_relations
        self.entropy_threshold = entropy_threshold
        self.prune_ratio = prune_ratio

    # ------------------------------------------------------------------
    # Entropy computation
    # ------------------------------------------------------------------

    def compute_relation_entropy(self, M_r: torch.Tensor) -> torch.Tensor:
        """
        Compute per-relation Von Neumann entanglement entropy.

        S(r) = -Tr(ρ_r log ρ_r)   where ρ_r = M_r M_r† / ||M_r||_F²

        For numerical stability we compute S from the singular values of M_r:
            ρ_r has the same eigenvalues as σᵢ² / Σᵢ σᵢ²
            S(r) = -Σᵢ pᵢ log pᵢ    where pᵢ = σᵢ² / Σᵢ σᵢ²

        Args:
            M_r: (R, d, d) or (R, d, d) complex Bell-state matrices

        Returns:
            entropies: (R,) float in [0, log(d)]
        """
        if M_r.ndim != 3 or M_r.shape[1] != M_r.shape[2]:
            raise ValueError(
                f"M_r must be (R, d, d), got {tuple(M_r.shape)}"
            )
        # Work with real view if input is complex
        if M_r.is_complex():
            M_r_real = M_r.real
        else:
            M_r_real = M_r

        # Singular values of each M_r: (R, d)
        try:
            sv = torch.linalg.svdvals(M_r_real)             # (R, d)
        except Exception:
            # Fallback: eigenvalues of M M^T
            MtM = torch.bmm(M_r_real, M_r_real.transpose(-1, -2))
            eigvals = torch.linalg.eigvalsh(MtM).clamp(min=0)
            sv = eigvals.sqrt()

        sv2 = sv ** 2                                        # (R, d)
        total = sv2.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        p = sv2 / total                                      # (R, d) probabilities
        # S = -Σ p log p  (entropy in nats; convention: 0 log 0 = 0)
        log_p = torch.log(p.clamp(min=1e-12))
        entropy = -(p * log_p).sum(dim=-1)                  # (R,)
        return entropy                                       # (R,)

    # ------------------------------------------------------------------
    # Per-query pruning decisions
    # ------------------------------------------------------------------

    def should_prune(
        self,
        relation_ids: torch.Tensor,
        entropies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decide per query whether to prune (low entropy → prune).

        Respects the global prune_ratio cap: at most floor(B * prune_ratio)
        queries are pruned per batch, choosing the lowest-entropy ones first.

        Args:
            relation_ids: (B,) int
            entropies:    (R,) float — one value per relation

        Returns:
            prune_mask: (B,) bool  — True means prune this query
        """
        B = relation_ids.shape[0]
        rel_entropy = entropies[relation_ids]                # (B,)
        # Threshold-based mask
        below_thresh = rel_entropy < self.entropy_threshold  # (B,) bool

        # Enforce prune_ratio cap
        max_prune = int(B * self.prune_ratio)
        if below_thresh.sum() <= max_prune:
            return below_thresh

        # Keep only the max_prune lowest-entropy relations
        _, sorted_idx = rel_entropy.sort()
        prune_mask = torch.zeros(B, dtype=torch.bool, device=relation_ids.device)
        prune_mask[sorted_idx[:max_prune]] = True
        # Intersect with threshold condition
        return prune_mask & below_thresh

    def prune_path_set(
        self,
        paths: list[Any],
        relation_entropies: torch.Tensor,
        path_relation_ids: torch.Tensor,
    ) -> list[Any]:
        """
        Filter a list of paths, removing those whose key relation has low entropy.

        Args:
            paths:               list of length M (any path objects)
            relation_entropies:  (R,) float
            path_relation_ids:   (M,) int — key relation id for each path

        Returns:
            filtered paths: subset of `paths`
        """
        if len(paths) != path_relation_ids.shape[0]:
            raise ValueError(
                f"paths length ({len(paths)}) must match "
                f"path_relation_ids ({path_relation_ids.shape[0]})"
            )
        keep_mask = ~self.should_prune(path_relation_ids, relation_entropies)
        return [p for p, keep in zip(paths, keep_mask.tolist()) if keep]

    def routing_decision(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        entropies: torch.Tensor,
    ) -> list[str]:
        """
        Return "quantum" or "classical" routing label per query.

        Args:
            head_ids:     (B,) int  (available for future complexity features)
            relation_ids: (B,) int
            entropies:    (R,) float

        Returns:
            decisions: list of B strings, each "quantum" or "classical"
        """
        prune = self.should_prune(relation_ids, entropies)  # (B,) bool
        return ["classical" if p else "quantum" for p in prune.tolist()]

    def get_param_count(self) -> dict[str, int]:
        return {"trainable": 0}  # threshold / ratio are hyperparameters, not params


# ---------------------------------------------------------------------------
# AdaptivePruningPolicy
# ---------------------------------------------------------------------------

class AdaptivePruningPolicy(nn.Module):
    """
    Learns WHEN to prune based on query complexity (head embedding,
    relation embedding, entropy features).

    The policy outputs a probability P_quantum(q) ∈ [0, 1] per query.
    Training minimises the routing loss:
        L_route = Σ [P_quantum · cost_quantum + (1-P_quantum) · cost_classical]
                    · indicator(answer_correct)

    In practice, cost_quantum = 1 (normalised) and cost_classical = α < 1
    (cheap bilinear), so the policy learns to be selective.

    Args:
        embed_dim:   dimension of head and relation embeddings
        hidden_dim:  MLP hidden size
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        # Input: head_emb || relation_emb || entropy_features (scalar)
        # entropy_features = [S(r), S(r)^2, log(1+S(r))]
        self.input_dim = embed_dim * 2 + 3
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        head_embedding: torch.Tensor,
        relation_embedding: torch.Tensor,
        entropy_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-query probability of routing to quantum scoring.

        Args:
            head_embedding:     (B, embed_dim)
            relation_embedding: (B, embed_dim)
            entropy_features:   (B,) scalar per-relation entropy S(r)

        Returns:
            prune_probability:  (B,) float ∈ [0, 1]
                                (probability of using quantum scoring;
                                 1 - this is the probability of classical routing)
        """
        B = head_embedding.shape[0]
        s = entropy_features                                 # (B,)
        # Entropy feature map: [S, S², log(1+S)]
        ent_feat = torch.stack([s, s ** 2, torch.log1p(s)], dim=-1)  # (B, 3)
        x = torch.cat([head_embedding, relation_embedding, ent_feat], dim=-1)  # (B, input_dim)
        logit = self.mlp(x).squeeze(-1)                     # (B,)
        return torch.sigmoid(logit)                          # (B,)

    def routing_loss(
        self,
        p_quantum: torch.Tensor,
        is_correct: torch.Tensor,
        cost_quantum: float = 1.0,
        cost_classical: float = 0.3,
    ) -> torch.Tensor:
        """
        Differentiable routing loss encouraging cheap routing when possible.

        L = Σ [p_q · c_q + (1−p_q) · c_cl] · correct

        Args:
            p_quantum:       (B,) float ∈ [0, 1]
            is_correct:      (B,) float ∈ {0, 1} — whether triple is positive
            cost_quantum:    scalar cost for quantum path (normalised to 1)
            cost_classical:  scalar cost for classical path (< 1)

        Returns:
            loss: scalar
        """
        expected_cost = p_quantum * cost_quantum + (1.0 - p_quantum) * cost_classical
        return (expected_cost * is_correct).mean()

    def get_param_count(self) -> dict[str, int]:
        return {name: p.numel() for name, p in self.named_parameters()}


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(99)
    R = 16
    d = 8
    B = 6
    embed_dim = 32

    # Synthetic Bell-state matrices
    M_r = torch.randn(R, d, d)

    pruner = EntanglementPruner(
        num_relations=R,
        entropy_threshold=0.5,
        prune_ratio=0.3,
    )

    # compute_relation_entropy
    entropies = pruner.compute_relation_entropy(M_r)
    assert entropies.shape == (R,), f"Bad shape: {entropies.shape}"
    assert (entropies >= 0).all(), "Entropy must be non-negative"
    print(f"EntanglementPruner.compute_relation_entropy: {entropies.shape}")
    print(f"  entropies (first 4): {entropies[:4].tolist()}")

    # should_prune
    rel_ids = torch.randint(0, R, (B,))
    prune_mask = pruner.should_prune(rel_ids, entropies)
    assert prune_mask.shape == (B,) and prune_mask.dtype == torch.bool
    print(f"should_prune: {prune_mask.tolist()}")

    # prune_path_set
    dummy_paths = [f"path_{i}" for i in range(B)]
    path_rel_ids = torch.randint(0, R, (B,))
    filtered = pruner.prune_path_set(dummy_paths, entropies, path_rel_ids)
    assert len(filtered) <= B
    print(f"prune_path_set: {len(dummy_paths)} → {len(filtered)} paths")

    # routing_decision
    decisions = pruner.routing_decision(rel_ids, rel_ids, entropies)
    assert len(decisions) == B
    assert all(d in ("quantum", "classical") for d in decisions)
    print(f"routing_decision: {decisions}")

    # AdaptivePruningPolicy
    policy = AdaptivePruningPolicy(embed_dim=embed_dim, hidden_dim=64)
    head_emb = torch.randn(B, embed_dim)
    rel_emb  = torch.randn(B, embed_dim)
    entropy_b = entropies[rel_ids]                          # (B,)

    p_q = policy(head_emb, rel_emb, entropy_b)
    assert p_q.shape == (B,)
    assert (p_q >= 0).all() and (p_q <= 1).all()
    print(f"AdaptivePruningPolicy p_quantum: {p_q.tolist()}")

    is_correct = torch.ones(B)
    loss = policy.routing_loss(p_q, is_correct)
    assert loss.item() >= 0
    print(f"routing_loss: {loss.item():.4f}")

    print(f"param count: {policy.get_param_count()}")
    print("All assertions passed.")
