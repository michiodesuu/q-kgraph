"""
models/components/topological_ecc.py — Topological Quantum Error Correction for KGs  [V6]

PURPOSE:
    Adapts the Tanner-graph / syndrome-measurement framework from topological
    quantum error-correcting codes (surface codes, toric codes) to detect and
    correct logical contradictions in a knowledge graph.

ANALOGY TO QUANTUM ERROR CORRECTION:
    QEC concept          →  KG analogue
    ─────────────────────────────────────────────────────────────────────────
    Data qubit           →  Entity (node in the KG)
    Stabiliser check     →  "Entity e should have a unique answer for relation r"
    Syndrome bit = 1     →  Constraint violated (e.g. conflicting targets for r)
    Error (Pauli flip)   →  Noisy or contradictory triple in training data
    Decoder (MWPM/GNN)   →  GNNSyndromeDecoder learns which entities to distrust
    Correction operator  →  Down-weight or prune high-error-probability triples
    ─────────────────────────────────────────────────────────────────────────

ARCHITECTURE OVERVIEW:
    1. TannerGraphBuilder  — constructs bipartite (entity ↔ check) graph from
                             raw triples; no external graph libraries needed.
    2. SyndromeDetector    — uses model's current predictions to score each
                             constraint check; high score = potential error.
    3. GNNSyndromeDecoder  — 2-layer message-passing GNN over the Tanner graph
                             that maps syndromes → per-entity error probabilities.
    4. SyndromeLoss        — joint training objective: GNN BCE + syndrome penalty.

DESIGN PRINCIPLES:
    - No external GNN library (torch_geometric, dgl): message passing is
      implemented with simple scatter operations over Python index lists.
    - Differentiable throughout: all syndrome scores and GNN activations are
      smooth (no hard thresholds), so gradients flow to the main model.
    - Numerically stable: complex entity states are converted to real features
      via |state| before entering the (real-valued) GNN.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  TannerGraphBuilder
# ──────────────────────────────────────────────────────────────────────────────

class TannerGraphBuilder:
    """
    Builds a Tanner (bipartite) graph from a knowledge graph.

    Graph structure
    ───────────────
    Left nodes  (type 0) = entities         (data qubits in QEC)
    Right nodes (type 1) = constraint checks (stabilisers in QEC)

    A "constraint check" for KG is the proposition:
        "Entity e participates in a UNIQUE role for relation r"

    Formally, for each (entity e, relation r) pair that appears more than once
    with DIFFERENT tail (or head) entities, we create one check node.  Triples
    that share an (h, r) prefix but disagree on t are potential contradictions.

    The check_matrix (C × E) encodes which entities participate in which checks:
        check_matrix[c, e] = 1  iff  entity e is involved in check c.

    For the GNN we need an explicit list of (entity_idx, check_idx) edges.

    Args (to build()):
        triples:               list of (h, r, t) integer tuples.
        num_entities:          total number of entities.
        num_relations:         total number of relation types.
        contradiction_queries: optional pre-specified list of conflict queries
                               {head, rel, wrong_tail, correct_tail}.
                               If None, contradictions are inferred from triples.

    Returns (from build()):
        check_matrix:     (num_checks, num_entities) bool list-of-lists
                          (row c, col e) = True if entity e is in check c.
        tanner_edges:     list of (entity_idx, check_idx) int pairs.
        entity_features:  (num_entities, feature_dim) float32 tensor.
                          Initialized with degree statistics for the GNN.
        check_info:       list of dicts describing each check node.
    """

    # feature_dim = [out_degree, in_degree, total_degree, log_degree]
    FEATURE_DIM: int = 4

    @staticmethod
    def build(
        triples: List[Tuple[int, int, int]],
        num_entities: int,
        num_relations: int,
        contradiction_queries: Optional[List[Dict]] = None,
    ) -> Tuple[List[List[bool]], List[Tuple[int, int]], torch.Tensor, List[Dict]]:
        """
        Main construction method.

        Returns:
            check_matrix:    list of length num_checks, each element is a set
                             of entity indices belonging to that check.
            tanner_edges:    flat list of (entity_idx, check_idx) pairs.
            entity_features: (num_entities, FEATURE_DIM) float32 tensor.
            check_info:      metadata list, one dict per check.
        """
        # ── Step 1: count entity degrees ──────────────────────────────────────
        out_degree = [0] * num_entities
        in_degree  = [0] * num_entities
        for h, r, t in triples:
            if h < num_entities:
                out_degree[h] += 1
            if t < num_entities:
                in_degree[t]  += 1

        # ── Step 2: build check nodes from (head, relation) conflicts ──────────
        # Group triples by (h, r) prefix
        hr_to_tails: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for h, r, t in triples:
            hr_to_tails[(h, r)].append(t)

        check_entity_sets: List[set] = []
        check_info: List[Dict] = []

        # Checks from structural multi-answer (h, r) pairs
        for (h, r), tails in hr_to_tails.items():
            unique_tails = list(set(tails))
            if len(unique_tails) > 1:
                # Contradiction: same head + relation → multiple tails
                involved = {h} | set(unique_tails)
                check_entity_sets.append(involved)
                check_info.append({
                    "type": "multi_tail",
                    "head": h,
                    "relation": r,
                    "conflicting_tails": unique_tails,
                })

        # Optional: checks from explicitly provided contradiction queries
        if contradiction_queries:
            for q in contradiction_queries:
                h  = q.get("head", -1)
                wt = q.get("wrong_tail", -1)
                ct = q.get("correct_tail", -1)
                involved = set()
                for idx in (h, wt, ct):
                    if 0 <= idx < num_entities:
                        involved.add(idx)
                if involved:
                    check_entity_sets.append(involved)
                    check_info.append({
                        "type": "explicit_contradiction",
                        **q,
                    })

        # If there are no detected contradictions, add one "trivial" check
        # covering all entities (so the graph is never empty).
        if not check_entity_sets:
            trivial = set(range(min(num_entities, 8)))
            check_entity_sets.append(trivial)
            check_info.append({"type": "trivial", "entities": list(trivial)})

        num_checks = len(check_entity_sets)

        # ── Step 3: build flat edge list (entity_idx, check_idx) ──────────────
        tanner_edges: List[Tuple[int, int]] = []
        for check_idx, entity_set in enumerate(check_entity_sets):
            for entity_idx in sorted(entity_set):
                tanner_edges.append((entity_idx, check_idx))

        # ── Step 4: entity feature matrix ─────────────────────────────────────
        feats = torch.zeros(num_entities, TannerGraphBuilder.FEATURE_DIM)
        for e in range(num_entities):
            total = out_degree[e] + in_degree[e]
            feats[e, 0] = float(out_degree[e])
            feats[e, 1] = float(in_degree[e])
            feats[e, 2] = float(total)
            feats[e, 3] = math.log1p(float(total))  # log(1 + degree)

        # Normalise each feature column to [0, 1] for GNN stability
        col_max = feats.max(dim=0).values.clamp(min=1.0)
        feats   = feats / col_max

        return check_entity_sets, tanner_edges, feats, check_info


# ──────────────────────────────────────────────────────────────────────────────
#  SyndromeDetector
# ──────────────────────────────────────────────────────────────────────────────

class SyndromeDetector(nn.Module):
    """
    Measures syndromes (parity violations) from model score predictions.

    Analogy to QEC:
        In a surface code, syndrome measurement tells us which stabiliser
        checks are violated by the current error pattern — without revealing
        the state itself.  Here, we "measure" a constraint check by asking:
            "Does the model rank the WRONG answer above the CORRECT answer?"
        A high syndrome score means the model is confused — indicating either
        a noisy training triple or a poorly trained region of the embedding space.

    Syndrome formula:
        For a contradiction query (h, r, wrong_t, correct_t):
            syndrome = max(0,  score(h, r, wrong_t) - score(h, r, correct_t) + margin)

        syndrome = 0:   model correctly prefers correct_t (no violation)
        syndrome > 0:   model incorrectly prefers wrong_t (parity violation)

    The margin parameter makes the detector sensitive to near-miss confusions
    where the model is nearly indifferent between the two answers.

    Args:
        margin: Minimum required score gap between correct and wrong tails.
                Default 0.1 — consistent with standard KGE training margins.
    """

    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        model: nn.Module,
        contradiction_queries: List[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute syndrome vector for the given set of contradiction queries.

        The model must implement a score(h, r, t) method (or accept tensors
        of head, relation, tail indices and return scores).  We accept two
        calling conventions:

            Convention A: model.score(head_tensor, rel_tensor, tail_tensor) → (N,)
            Convention B: model(head_tensor, rel_tensor, tail_tensor) → (N,)

        Args:
            model:                 The KGE model being trained.
            contradiction_queries: List of dicts with keys:
                                       head, rel, wrong_tail, correct_tail
                                   All values are integer entity/relation IDs.
            device:                Compute device.

        Returns:
            (num_queries,) float32 tensor of syndrome scores ≥ 0.
            Differentiable w.r.t. model parameters.
        """
        if not contradiction_queries:
            return torch.zeros(0, device=device)

        heads_w  = torch.tensor([q["head"]       for q in contradiction_queries], device=device)
        rels     = torch.tensor([q["rel"]         for q in contradiction_queries], device=device)
        wrongs   = torch.tensor([q["wrong_tail"]  for q in contradiction_queries], device=device)
        corrects = torch.tensor([q["correct_tail"] for q in contradiction_queries], device=device)

        # Score wrong and correct tails — try both calling conventions
        try:
            score_wrong   = model.score(heads_w, rels, wrongs)
            score_correct = model.score(heads_w, rels, corrects)
        except AttributeError:
            score_wrong   = model(heads_w, rels, wrongs)
            score_correct = model(heads_w, rels, corrects)

        # Syndrome:  ReLU(score_wrong - score_correct + margin)
        # High when the model incorrectly favours the wrong tail.
        syndrome = F.relu(score_wrong - score_correct + self.margin)  # (N,)
        return syndrome


# ──────────────────────────────────────────────────────────────────────────────
#  GNNSyndromeDecoder
# ──────────────────────────────────────────────────────────────────────────────

class GNNSyndromeDecoder(nn.Module):
    """
    Two-layer message-passing GNN over the Tanner graph that converts syndrome
    measurements into per-entity error probabilities.

    Architecture
    ────────────
    Input node features:
        Entity nodes:  |entity_state| (real amplitudes) + static degree features
                       Shape: (E, entity_dim + FEATURE_DIM)  — we use amplitudes
                       because they are real, positive, and physically meaningful.
        Check  nodes:  syndrome scores expanded to a vector.
                       Shape: (C, 1)

    Message-passing (2 rounds):
        Round 1 — entity → check aggregation:
            For each check node c:
                m_c = mean of { entity_linear(h_e) for e in neighbours(c) }
                h_c_new = ReLU(m_c + syndrome_embed(syndrome_c))

        Round 2 — check → entity aggregation:
            For each entity node e:
                m_e = mean of { check_linear(h_c) for c in neighbours(e) }
                h_e_new = ReLU(m_e + entity_residual(h_e_old))

    Output:
        error_probs = Sigmoid(output_linear(h_e_new))   shape: (E,)

    This deliberately avoids fancy attention mechanisms to keep the decoder
    interpretable and dependency-free.  Simple mean-aggregation (like GraphSAGE)
    is sufficient for syndrome decoding, which is a relatively structured task.

    Args:
        entity_dim:  Dimension of entity state vectors (complex_dim of QuantumStateEncoder).
        hidden_dim:  Hidden dimension for GNN layers.
        num_layers:  Number of message-passing rounds (currently only 2 supported).
    """

    STATIC_DIM = TannerGraphBuilder.FEATURE_DIM  # 4 static degree features

    def __init__(
        self,
        entity_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input dimension: amplitude features (entity_dim real) + static features
        input_dim = entity_dim + self.STATIC_DIM

        # ── Entity input projection ────────────────────────────────────────────
        self.entity_input = nn.Linear(input_dim, hidden_dim)

        # ── Check node: syndrome embedding (scalar → hidden_dim) ──────────────
        self.syndrome_embed = nn.Linear(1, hidden_dim)

        # ── Round 1: entity → check (entity features → check hidden) ──────────
        self.entity_to_check = nn.Linear(hidden_dim, hidden_dim)
        self.check_update    = nn.Linear(hidden_dim, hidden_dim)

        # ── Round 2: check → entity (check hidden → entity hidden) ────────────
        self.check_to_entity  = nn.Linear(hidden_dim, hidden_dim)
        self.entity_residual  = nn.Linear(hidden_dim, hidden_dim)

        # ── Output: entity error probability ──────────────────────────────────
        self.output_layer = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Scatter helpers (no external libraries) ────────────────────────────────

    @staticmethod
    def _mean_aggregate(
        source_feats: torch.Tensor,
        source_indices: List[int],
        target_indices: List[int],
        num_targets: int,
    ) -> torch.Tensor:
        """
        Mean-aggregate source features into target nodes.

        For each target node t, collects all source features h_s where
        (s, t) is an edge and computes their mean.  If a target has no
        incoming edges, its aggregated feature is the zero vector.

        This implements:
            h_target[t] = mean_{s: (s,t) ∈ E} h_source[s]

        Args:
            source_feats:   (|source_nodes|, F) float tensor.
            source_indices: list of source node indices (one per edge).
            target_indices: list of target node indices (one per edge).
            num_targets:    number of target nodes.

        Returns:
            (num_targets, F) float tensor.
        """
        F_dim = source_feats.shape[1]
        device = source_feats.device

        # Accumulator and count
        agg   = torch.zeros(num_targets, F_dim, device=device)
        count = torch.zeros(num_targets, 1,     device=device)

        if not source_indices:
            return agg

        src_idx = torch.tensor(source_indices, dtype=torch.long, device=device)  # (|E|,)
        tgt_idx = torch.tensor(target_indices, dtype=torch.long, device=device)  # (|E|,)

        # Gather source features for each edge
        edge_feats = source_feats[src_idx]  # (|E|, F)

        # Scatter-add into target nodes
        agg.scatter_add_(0, tgt_idx.unsqueeze(1).expand_as(edge_feats), edge_feats)
        count.scatter_add_(0, tgt_idx.unsqueeze(1), torch.ones(len(tgt_idx), 1, device=device))

        # Safe mean (avoid div-by-zero for isolated nodes)
        count = count.clamp(min=1.0)
        return agg / count

    def forward(
        self,
        entity_states: torch.Tensor,
        syndrome_scores: torch.Tensor,
        tanner_adjacency: List[Tuple[int, int]],
        device: torch.device,
        entity_static_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run 2-round message passing and return error probabilities.

        Args:
            entity_states:          (E, entity_dim) complex tensor.
                                    Converted to real via amplitude |state|.
            syndrome_scores:        (C,) float tensor of syndrome values.
            tanner_adjacency:       List of (entity_idx, check_idx) int pairs.
            device:                 Compute device.
            entity_static_features: Optional (E, STATIC_DIM) float tensor of
                                    pre-computed degree features.  If None,
                                    zeros are used.

        Returns:
            (E,) float tensor of error probabilities ∈ [0, 1].
        """
        E = entity_states.shape[0]
        C = syndrome_scores.shape[0] if syndrome_scores.numel() > 0 else 1

        # ── Convert complex entity states to real amplitude features ───────────
        # |ψ|  =  component-wise absolute value → real non-negative vector
        entity_amp = entity_states.abs()   # (E, entity_dim) real

        # Concatenate static degree features if provided
        if entity_static_features is not None:
            static = entity_static_features.to(device)
        else:
            static = torch.zeros(E, self.STATIC_DIM, device=device)

        entity_feat_raw = torch.cat([entity_amp, static], dim=-1)  # (E, entity_dim+STATIC_DIM)

        # ── Build edge lists: entity→check and check→entity ───────────────────
        if tanner_adjacency:
            ent_in_edges  = [e for (e, c) in tanner_adjacency]   # entity source
            chk_in_edges  = [c for (e, c) in tanner_adjacency]   # check target
            chk_out_edges = [c for (e, c) in tanner_adjacency]   # check source
            ent_out_edges = [e for (e, c) in tanner_adjacency]   # entity target
        else:
            ent_in_edges = chk_in_edges = chk_out_edges = ent_out_edges = []

        # ── Initial node embeddings ────────────────────────────────────────────
        # Entity initial hidden state
        h_entity = F.relu(self.entity_input(entity_feat_raw))  # (E, H)

        # Check initial hidden state from syndrome scores
        syn_input = syndrome_scores.view(C, 1)                 # (C, 1)
        h_check   = F.relu(self.syndrome_embed(syn_input))     # (C, H)

        # ── Round 1: entity → check ────────────────────────────────────────────
        # Aggregate entity messages into check nodes
        entity_msgs = self.entity_to_check(h_entity)           # (E, H)
        agg_at_check = self._mean_aggregate(
            entity_msgs, ent_in_edges, chk_in_edges, C
        )                                                       # (C, H)
        # Update check nodes: combine aggregated entity info with syndrome embed
        h_check_new = F.relu(self.check_update(agg_at_check) + h_check)  # (C, H)

        # ── Round 2: check → entity ────────────────────────────────────────────
        # Aggregate check messages into entity nodes
        check_msgs = self.check_to_entity(h_check_new)         # (C, H)
        agg_at_entity = self._mean_aggregate(
            check_msgs, chk_out_edges, ent_out_edges, E
        )                                                       # (E, H)
        # Update entity nodes: combine neighbour check info with residual
        h_entity_new = F.relu(
            self.entity_residual(h_entity) + agg_at_entity
        )                                                       # (E, H)

        # ── Output: sigmoid for probability in [0, 1] ──────────────────────────
        logits      = self.output_layer(h_entity_new).squeeze(-1)  # (E,)
        error_probs = torch.sigmoid(logits)                         # (E,)

        return error_probs

    def extra_repr(self) -> str:
        return (
            f"entity_dim={self.entity_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}"
        )


# ──────────────────────────────────────────────────────────────────────────────
#  SyndromeLoss
# ──────────────────────────────────────────────────────────────────────────────

class SyndromeLoss(nn.Module):
    """
    Joint training loss for the GNN syndrome decoder and the main KGE model.

    Combines two terms:

    (a) Binary Cross-Entropy on GNN error predictions:
        L_bce = BCE(error_probs[e], label[e])
        where label[e] = 1 if entity e is in known_error_entities, else 0.
        This trains the GNN to identify which entities are "corrupted".

    (b) Syndrome magnitude penalty:
        L_syn = mean(syndrome_scores)
        This drives the main model towards a state where fewer syndromes fire,
        i.e. the model correctly ranks correct triples above wrong ones.
        The gradient flows back to the main model through SyndromeDetector.

    Combined:
        L_total = weight_bce * L_bce + weight_syndrome * L_syn

    Usage pattern:
        # In training loop:
        loss = syndrome_loss(
            detector, gnn, model,
            contradiction_queries, error_entity_set,
            entity_states, tanner_edges, device
        )
        loss.backward()

    Args:
        weight_bce:      Weight for the GNN BCE loss.
        weight_syndrome: Weight for the syndrome magnitude penalty.
    """

    def __init__(
        self,
        weight_bce:      float = 1.0,
        weight_syndrome: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight_bce      = weight_bce
        self.weight_syndrome = weight_syndrome

    def forward(
        self,
        syndrome_detector:       SyndromeDetector,
        gnn_decoder:             GNNSyndromeDecoder,
        model:                   nn.Module,
        contradiction_queries:   List[Dict],
        known_error_entities:    List[int],
        entity_states:           torch.Tensor,
        tanner_adjacency:        List[Tuple[int, int]],
        device:                  torch.device,
        entity_static_features:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined syndrome + decoder loss.

        Args:
            syndrome_detector:      SyndromeDetector module.
            gnn_decoder:            GNNSyndromeDecoder module.
            model:                  Main KGE model (for syndrome measurement).
            contradiction_queries:  List of {head, rel, wrong_tail, correct_tail}.
            known_error_entities:   List of entity indices known to be "corrupted"
                                    (ground-truth labels for GNN training).
            entity_states:          (E, d) complex entity states for the GNN.
            tanner_adjacency:       List of (entity_idx, check_idx) Tanner edges.
            device:                 Compute device.
            entity_static_features: Optional (E, STATIC_DIM) degree features.

        Returns:
            Scalar loss tensor (differentiable w.r.t. all parameters).
        """
        E = entity_states.shape[0]

        # ── Step 1: measure syndromes ──────────────────────────────────────────
        syndrome_scores = syndrome_detector(model, contradiction_queries, device)
        # syndrome_scores: (num_queries,) or (0,) if empty

        # Pad to at least 1 element so the GNN check nodes are defined
        if syndrome_scores.numel() == 0:
            syndrome_scores = torch.zeros(1, device=device)

        num_checks = syndrome_scores.shape[0]

        # ── Step 2: run GNN decoder ────────────────────────────────────────────
        error_probs = gnn_decoder(
            entity_states,
            syndrome_scores,
            tanner_adjacency,
            device,
            entity_static_features=entity_static_features,
        )  # (E,) in [0, 1]

        # ── Step 3: BCE loss against known error labels ────────────────────────
        labels = torch.zeros(E, device=device)
        for e in known_error_entities:
            if 0 <= e < E:
                labels[e] = 1.0

        if E > 0:
            l_bce = F.binary_cross_entropy(
                error_probs.clamp(1e-7, 1 - 1e-7),
                labels,
            )
        else:
            l_bce = torch.zeros(1, device=device).squeeze()

        # ── Step 4: syndrome magnitude penalty ────────────────────────────────
        # This gradient flows back to the main model encouraging it to produce
        # consistent rankings (correct > wrong).
        l_syn = syndrome_scores.mean()

        # ── Step 5: combine ────────────────────────────────────────────────────
        total_loss = self.weight_bce * l_bce + self.weight_syndrome * l_syn
        return total_loss


# ──────────────────────────────────────────────────────────────────────────────
#  Utility: build a simple check-to-entity adjacency from triples (convenience)
# ──────────────────────────────────────────────────────────────────────────────

def build_tanner_graph(
    triples: List[Tuple[int, int, int]],
    num_entities: int,
    num_relations: int,
    contradiction_queries: Optional[List[Dict]] = None,
) -> Tuple[List[Tuple[int, int]], torch.Tensor, List[Dict]]:
    """
    Convenience wrapper around TannerGraphBuilder.build().

    Returns:
        tanner_edges:     list of (entity_idx, check_idx) edges.
        entity_features:  (num_entities, 4) static feature tensor.
        check_info:       metadata per check node.
    """
    _, tanner_edges, entity_features, check_info = TannerGraphBuilder.build(
        triples=triples,
        num_entities=num_entities,
        num_relations=num_relations,
        contradiction_queries=contradiction_queries,
    )
    return tanner_edges, entity_features, check_info
