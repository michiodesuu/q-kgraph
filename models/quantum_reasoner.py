"""
models/quantum_reasoner.py — Full Assembled Quantum KG Model  [V1]

Wires: QuantumStateEncoder + UnitaryOperator + AmplitudeAggregator.

USAGE:
    model = QuantumReasoner(num_entities=28, num_relations=12, embed_dim=16)
    scores = model.score_triple(h, r, t)              # (B,) training
    scores = model.score_triple_vs_all(h, r)          # (B, E) evaluation
    analysis = model.analyze_interference(h, r, t, paths)  # dict
"""
from __future__ import annotations
from typing import Optional
import torch, torch.nn as nn

from models.components.quantum_states import QuantumStateEncoder
from models.components.unitary_operators import UnitaryOperator, build_unitary
from models.components.path_aggregator import PathEnumerator, AmplitudeAggregator


class QuantumReasoner(nn.Module):
    """
    Complete quantum knowledge graph reasoning model.

    Two scoring modes:
        score_triple():      Fast 1-hop Born rule. Used during training.
        score_with_paths():  Multi-hop interference. Used at inference.

    Ablation modes (for paper Table 3):
        "full":     Normal operation.
        "no_phase": Zeroes imaginary components (tests phase contribution).
        "no_paths": 1-hop only (tests multi-hop contribution).
        "classical": Re(amplitude) instead of |amplitude|² (tests Born rule).

    Args:
        num_entities:   Total entity count.
        num_relations:  Total relation count.
        embed_dim:      Total embedding dimension (complex_dim = embed_dim // 2).
        unitary_type:   "diagonal" (default), "givens", "matrix_exp".
        max_paths:      Max paths per query for interference computation.
        max_hops:       Max BFS depth for path enumeration.
        dropout:        Dropout rate on entity states.
        ablation_mode:  "full", "no_phase", "no_paths", "classical".
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int   = 16,
        unitary_type:  str   = "diagonal",
        max_paths:     int   = 8,
        max_hops:      int   = 2,
        dropout:       float = 0.0,
        ablation_mode: str   = "full",
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.complex_dim   = embed_dim // 2
        self.max_paths     = max_paths
        self.max_hops      = max_hops
        self.ablation_mode = ablation_mode

        # ── Components ──────────────────────────────────────────────────────
        self.encoder = QuantumStateEncoder(
            num_entities = num_entities,
            embed_dim    = embed_dim,
            dropout      = dropout,
            normalize    = True,
        )
        self.unitary = build_unitary(unitary_type, num_relations, self.complex_dim)
        self.aggregator = AmplitudeAggregator(
            complex_dim        = self.complex_dim,
            max_paths          = max_paths,
            learn_path_weights = True,
        )

        # Relation-specific bias (scalar per relation)
        self.relation_bias = nn.Parameter(torch.zeros(num_relations))

    # ── Scoring methods ────────────────────────────────────────────────────
    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Fast 1-hop Born rule scoring: |⟨t|U_r|h⟩|² + bias_r.

        Used during training for efficiency.
        Ablation modes alter this computation.

        Returns:
            (B,) float scores (logits for BCE loss).
        """
        h_states = self.encoder(head_ids)    # (B, d) complex
        t_states = self.encoder(tail_ids)    # (B, d) complex

        # Ablation: no_phase → zero imaginary components
        if self.ablation_mode == "no_phase":
            h_states = torch.complex(h_states.real, torch.zeros_like(h_states.real))
            t_states = torch.complex(t_states.real, torch.zeros_like(t_states.real))

        # Apply relation unitary: |h'⟩ = U_r|h⟩
        h_transformed = self.unitary.apply(h_states, relation_ids)  # (B, d) complex

        # Inner product: ⟨t|h'⟩
        amplitude = (t_states.conj() * h_transformed).sum(-1)   # (B,) complex

        # Born rule or classical (ablation)
        if self.ablation_mode == "classical":
            score = amplitude.real
        else:
            score = amplitude.abs().pow(2)

        return score + self.relation_bias[relation_ids]

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as candidate tails for (head, relation) queries.

        Used during evaluation. Returns a (B, E) matrix where
        entry [b, e] is the score of entity e as tail for query b.

        Memory note: For large E (WN18RR: 40k, YAGO3-10: 123k),
        this may cause OOM. Use ChunkedEvaluator (V2) instead.

        Returns:
            (B, num_entities) float score matrix.
        """
        B  = head_ids.shape[0]
        device = head_ids.device

        h_states      = self.encoder(head_ids)                          # (B, d) complex
        h_transformed = self.unitary.apply(h_states, relation_ids)     # (B, d) complex

        # All entity states: (E, d)
        all_ids    = torch.arange(self.num_entities, device=device)
        all_states = self.encoder(all_ids)                             # (E, d) complex

        # Batch inner product: (B, d) conj × (d, E) = (B, E)
        # h_transformed: (B, d) → (B, d)
        # all_states:    (E, d) → conj.T = (d, E)
        scores_complex = torch.matmul(
            h_transformed,
            all_states.conj().T,
        )   # (B, E) complex

        if self.ablation_mode == "classical":
            scores = scores_complex.real
        else:
            scores = scores_complex.abs().pow(2)

        bias = self.relation_bias[relation_ids].unsqueeze(1)  # (B, 1)
        return scores + bias   # (B, E)

    def score_with_paths(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
        paths_batch:  list,           # B lists of Path objects
    ) -> torch.Tensor:
        """
        Multi-hop interference scoring: P(t|s) = |Σᵢ αᵢ ⟨t|U_Pi|s⟩|².

        Used at inference time. Slower than score_triple() but uses the
        full interference mechanism across all paths.

        Ablation "no_paths": falls back to score_triple().

        Returns:
            (B,) float probabilities in [0, 1].
        """
        if self.ablation_mode == "no_paths":
            return self.score_triple(head_ids, relation_ids, tail_ids)

        h_states = self.encoder(head_ids)   # (B, d)
        t_states = self.encoder(tail_ids)   # (B, d)

        return self.aggregator.forward(h_states, t_states, paths_batch, self.unitary)

    # ── Analysis methods ───────────────────────────────────────────────────
    def analyze_interference(
        self,
        head_id:   int,
        tail_id:   int,
        paths:     list,
        device:    Optional[torch.device] = None,
    ) -> dict:
        """
        Run interference decomposition for one (head, tail, paths) query.

        Returns the compute_interference_terms() dict.
        Used in run_toy.py Step 7 and by InterferenceMonitor.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            h_state = self.encoder(torch.tensor([head_id], device=device)).squeeze(0)
            t_state = self.encoder(torch.tensor([tail_id],  device=device)).squeeze(0)
            return self.aggregator.compute_interference_terms(
                h_state, t_state, paths, self.unitary
            )

    # ── Utilities ──────────────────────────────────────────────────────────
    def get_param_count(self) -> dict[str, int]:
        """Return parameter counts per component."""
        enc_params  = sum(p.numel() for p in self.encoder.parameters())
        uni_params  = sum(p.numel() for p in self.unitary.parameters())
        agg_params  = sum(p.numel() for p in self.aggregator.parameters())
        bias_params = self.relation_bias.numel()
        total       = enc_params + uni_params + agg_params + bias_params

        return {
            "encoder":    enc_params,
            "unitary":    uni_params,
            "aggregator": agg_params,
            "bias":       bias_params,
            "total":      total,
        }

    def extra_repr(self) -> str:
        counts = self.get_param_count()
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, complex_dim={self.complex_dim}, "
            f"total_params={counts['total']:,}"
        )
