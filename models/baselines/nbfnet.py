"""
models/baselines/nbfnet.py — NBFNet Baseline  [V3]

NBFNet: Neural Bellman-Ford Networks (Zhu et al., NeurIPS 2021)
arxiv.org/abs/2106.06935

PURPOSE IN THIS PROJECT:
    NBFNet is the strongest modern GNN baseline. Without it, reviewers at
    NeurIPS/ICLR/ICML will reject the paper for "strawman baselines"
    (only comparing against 2013-2019 models). Including NBFNet and showing
    that QuantumReasoner outperforms it specifically at high noise levels
    (≥10%) is the core empirical argument of the V3 paper.

    Expected experimental pattern (paper Figure 3 key result):
        Clean data (0% noise):    NBFNet > QuantumReasoner   [acceptable]
        10% noise:                NBFNet ≈ QuantumReasoner    [crossover]
        20% noise:                NBFNet < QuantumReasoner    [key result]

    This crossover is the argument: NBFNet sees more of the graph but cannot
    cancel contradictory evidence. QuantumReasoner uses less of the graph but
    the interference mechanism provides superior noise robustness.

HOW NBFNet WORKS (contrast with QuantumReasoner):
    NBFNet uses generalized Bellman-Ford propagation across the FULL graph:
        h_v^(l+1) = AGG({MSG(h_u^(l), e_{uv}) : (u,r,v) ∈ G})
    where MSG and AGG are learned functions and l is the layer count.

    This is equivalent to considering ALL paths of length l, not just
    BFS-truncated paths. NBFNet sees more of the graph at inference time,
    which is why it outperforms on clean data.

    QuantumReasoner uses BFS with max_paths=8. The interference mechanism
    compensates for the path truncation by actively suppressing wrong paths.

IMPLEMENTATION NOTES:
    Full NBFNet requires message passing over the entire KG adjacency,
    which is memory-intensive for large graphs. This implementation uses
    the sparse edge representation and sparse matrix multiplication.

    For the paper: train NBFNet at the same embed_dim and number of layers.
    Compare at multiple noise levels using run_ablation.py noise_experiment.

USAGE:
    model = NBFNet(num_entities=28, num_relations=12, embed_dim=64, n_layers=3)
    scores = model.score_triple(h, r, t)          # (B,) float
    scores = model.score_triple_vs_all(h, r)      # (B, E) float
    model.set_graph(edge_index, edge_type)         # must call before forward
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BellmanFordLayer(nn.Module):
    """
    One Bellman-Ford message passing layer.

    For each node v, aggregates messages from all neighbors u
    connected by any relation r, conditioned on the query relation.

    h_v^(l+1) = GRU(h_v^(l), AGG_{(u,r,v) ∈ G}[MSG(h_u^(l), e_r, r_query)])

    The conditioning on r_query is what makes NBFNet a conditional model:
    it computes different node representations for different query relations.

    Args:
        embed_dim:      Node embedding dimension.
        num_relations:  Number of relation types.
        aggr:           Aggregation function: "sum", "mean", or "max".
    """

    def __init__(
        self,
        embed_dim:     int,
        num_relations: int,
        aggr:          str = "sum",
    ) -> None:
        super().__init__()
        self.embed_dim     = embed_dim
        self.num_relations = num_relations
        self.aggr          = aggr

        # Message function: f(h_u, e_r, r_query) → message
        # Concatenate neighbor embedding + relation embedding + query embedding
        self.msg_linear = nn.Linear(embed_dim * 3, embed_dim)

        # Update function: GRU-style update
        self.update_gru = nn.GRUCell(embed_dim, embed_dim)

        # Relation embeddings (shared across layers, like RotatE)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)

    def forward(
        self,
        node_h:     torch.Tensor,     # (N, embed_dim) current node representations
        query_emb:  torch.Tensor,     # (B, embed_dim) query relation embeddings
        edge_index: torch.Tensor,     # (2, E) source and target node indices
        edge_type:  torch.Tensor,     # (E,) relation type for each edge
        batch_ids:  torch.Tensor,     # (N,) which query batch each node belongs to
    ) -> torch.Tensor:
        """
        Perform one Bellman-Ford propagation step.

        Args:
            node_h:     Current node representations (N, d).
            query_emb:  Query relation embedding per batch item (B, d).
            edge_index: COO format edge list (2, E).
            edge_type:  Relation type per edge (E,).
            batch_ids:  Batch assignment per node (N,). Used to broadcast
                        the query embedding to the right nodes.

        Returns:
            Updated node representations (N, embed_dim).
        """
        src, dst = edge_index[0], edge_index[1]   # source and destination nodes

        # Gather source node representations
        h_src = node_h[src]                        # (E, d)

        # Gather relation embeddings
        h_rel = self.relation_emb(edge_type)       # (E, d)

        # Gather query embeddings for each edge's source node's batch
        h_query = query_emb[batch_ids[src]]        # (E, d)

        # Message: f(h_u, e_r, r_query)
        msg_input = torch.cat([h_src, h_rel, h_query], dim=-1)  # (E, 3d)
        msg       = F.relu(self.msg_linear(msg_input))           # (E, d)

        # Aggregate messages at each destination node
        N   = node_h.shape[0]
        agg = torch.zeros(N, self.embed_dim, device=node_h.device)

        if self.aggr == "sum":
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        elif self.aggr == "mean":
            count = torch.zeros(N, 1, device=node_h.device)
            count.scatter_add_(0, dst.unsqueeze(1), torch.ones(msg.shape[0], 1, device=msg.device))
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            agg = agg / (count + 1e-10)
        elif self.aggr == "max":
            agg = agg.scatter_reduce(0, dst.unsqueeze(1).expand_as(msg), msg,
                                     reduce="amax", include_self=True)

        # Update: GRU-style
        return self.update_gru(agg, node_h)


class NBFNet(nn.Module):
    """
    Neural Bellman-Ford Network for knowledge graph link prediction.

    Key difference from QuantumReasoner:
        - Sees ALL paths (full Bellman-Ford propagation)
        - Real-valued (no quantum interference)
        - Stronger on clean data, weaker under noise

    This is the modern GNN baseline that replaces the "strawman" argument.
    Include it in V3 paper Table 2 and Figure 3.

    Args:
        num_entities:   Total entity count.
        num_relations:  Total relation count.
        embed_dim:      Embedding dimension.
        n_layers:       Number of Bellman-Ford propagation layers.
                        Equivalent to maximum reasoning path length.
                        Use 3 to match QuantumReasoner's max_hops=2.
        aggr:           Message aggregation: "sum", "mean", or "max".
        dropout:        Dropout rate.

    USAGE:
        model = NBFNet(num_entities=28, num_relations=12, embed_dim=64, n_layers=3)
        # Must set graph structure before scoring (do once after building dataset)
        model.set_graph(edge_index, edge_type, num_entities=28)
        scores = model.score_triple(h, r, t)
        scores = model.score_triple_vs_all(h, r)
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        embed_dim:     int  = 64,
        n_layers:      int  = 3,
        aggr:          str  = "sum",
        dropout:       float = 0.0,
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.n_layers      = n_layers

        # Entity and relation embeddings (initialization representations)
        self.entity_emb   = nn.Embedding(num_entities,  embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)

        # Bellman-Ford layers
        self.layers = nn.ModuleList([
            BellmanFordLayer(embed_dim, num_relations, aggr)
            for _ in range(n_layers)
        ])

        # Final scoring MLP: [h_init, h_final, r] → score
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Stored graph structure (set via set_graph())
        self._edge_index: Optional[torch.Tensor] = None
        self._edge_type:  Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def set_graph(
        self,
        edge_index:  torch.Tensor,     # (2, E) COO edge list
        edge_type:   torch.Tensor,     # (E,) relation type per edge
        num_entities: Optional[int] = None,
    ) -> None:
        """
        Set the graph structure for Bellman-Ford propagation.

        Must be called once before any score_triple() calls.
        Call again if the graph changes (e.g., noise injection).

        Args:
            edge_index:   (2, E) tensor — [source_ids, target_ids].
            edge_type:    (E,) tensor — relation type for each edge.
            num_entities: Override num_entities (if graph has different size).

        Example:
            # Build from toy_kg
            adj   = kg.get_adjacency()
            edges = [(h, t, r) for h, neighbors in adj.items()
                     for r, t in neighbors]
            edge_index = torch.tensor([[h, t] for h, t, r in edges]).T
            edge_type  = torch.tensor([r for h, t, r in edges])
            model.set_graph(edge_index, edge_type)
        """
        self._edge_index = edge_index
        self._edge_type  = edge_type
        if num_entities is not None:
            self.num_entities = num_entities

    def _build_node_representations(
        self,
        query_heads:    torch.Tensor,    # (B,) head entity IDs
        query_relations: torch.Tensor,   # (B,) query relation IDs
        device:         torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run Bellman-Ford propagation for a batch of queries.

        For each query (h_b, r_b), initializes the source node h_b with
        a special "source" representation (entity embedding + relation embedding),
        then propagates messages through n_layers steps across the full graph.

        Returns:
            (node_representations, batch_ids) where:
                node_representations: (B*N, embed_dim) — all nodes, all queries
                batch_ids: (B*N,) — which query each node row belongs to
        """
        B = query_heads.shape[0]
        N = self.num_entities

        if self._edge_index is None:
            raise RuntimeError(
                "Graph not set. Call model.set_graph(edge_index, edge_type) first."
            )

        edge_index = self._edge_index.to(device)
        edge_type  = self._edge_type.to(device)

        # Initialize all node representations as entity embeddings
        # Replicate N nodes for each of B queries → (B*N, embed_dim)
        all_entity_ids = torch.arange(N, device=device)
        base_h = self.entity_emb(all_entity_ids)   # (N, embed_dim)
        node_h = base_h.unsqueeze(0).expand(B, N, -1).reshape(B * N, -1).clone()

        # Set source nodes: h[b*N + head_b] = entity_emb[head_b] + relation_emb[rel_b]
        # This gives the source node a special initialization conditioned on the query
        query_emb  = self.relation_emb(query_relations)   # (B, embed_dim)
        source_ids = query_heads + torch.arange(B, device=device) * N   # (B,) global indices
        head_embs  = self.entity_emb(query_heads)         # (B, embed_dim)
        source_init = head_embs + query_emb               # (B, embed_dim) — head + query
        node_h[source_ids] = source_init

        # Build batch_ids: which query each node belongs to
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(B * N)

        # Expand edge_index for all B queries
        # Original: (2, E) → (2, B*E) by repeating and offsetting node IDs
        E           = edge_index.shape[1]
        offsets     = (torch.arange(B, device=device) * N).unsqueeze(1).expand(B, E).reshape(-1)
        src_expanded = edge_index[0].unsqueeze(0).expand(B, E).reshape(-1) + offsets
        dst_expanded = edge_index[1].unsqueeze(0).expand(B, E).reshape(-1) + offsets
        edge_type_expanded = edge_type.unsqueeze(0).expand(B, E).reshape(-1)
        edge_index_expanded = torch.stack([src_expanded, dst_expanded], dim=0)

        # Run Bellman-Ford layers
        for layer in self.layers:
            node_h = layer(
                node_h          = node_h,
                query_emb       = query_emb,
                edge_index      = edge_index_expanded,
                edge_type       = edge_type_expanded,
                batch_ids       = batch_ids,
            )

        return node_h.reshape(B, N, self.embed_dim), query_emb

    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score specific (head, relation, tail) triples.

        Runs Bellman-Ford propagation from each head, then scores
        the specific tail node using the final representation.

        Returns:
            (B,) float scores. Higher = more likely true.
        """
        device = head_ids.device
        B      = head_ids.shape[0]

        node_reps, query_emb = self._build_node_representations(
            head_ids, relation_ids, device
        )   # (B, N, d), (B, d)

        # Gather tail representations for each query
        tail_reps = node_reps[
            torch.arange(B, device=device),
            tail_ids,
        ]   # (B, d)

        # Score: MLP over [initial_head, final_tail, relation]
        head_init = self.entity_emb(head_ids)   # (B, d)
        features  = torch.cat([head_init, tail_reps, query_emb], dim=-1)   # (B, 3d)
        scores    = self.score_mlp(features).squeeze(-1)   # (B,)
        return scores

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score all entities as candidate tails.

        Runs Bellman-Ford propagation from each head, then scores
        all N entity nodes using their final representations.

        Returns:
            (B, num_entities) float score matrix.
        """
        device = head_ids.device
        B      = head_ids.shape[0]
        N      = self.num_entities

        node_reps, query_emb = self._build_node_representations(
            head_ids, relation_ids, device
        )   # (B, N, d), (B, d)

        # Score all tail nodes
        head_init  = self.entity_emb(head_ids)                             # (B, d)
        head_exp   = head_init.unsqueeze(1).expand(B, N, -1)               # (B, N, d)
        query_exp  = query_emb.unsqueeze(1).expand(B, N, -1)               # (B, N, d)
        features   = torch.cat([head_exp, node_reps, query_exp], dim=-1)   # (B, N, 3d)

        # Reshape for MLP: (B*N, 3d) → (B*N,) → reshape to (B, N)
        scores = self.score_mlp(
            features.reshape(B * N, -1)
        ).reshape(B, N)
        return scores

    def get_graph_from_toy_kg(self, toy_kg, device: torch.device) -> None:
        """
        Convenience: set graph from a ToyKG instance.

        Builds the COO edge list from toy_kg.get_adjacency() and
        calls self.set_graph() automatically.

        Args:
            toy_kg: ToyKG instance.
            device: Target device.
        """
        adj = toy_kg.get_adjacency()
        src_list, dst_list, rel_list = [], [], []

        for node_id, neighbors in adj.items():
            for rel_id, neighbor_id in neighbors:
                src_list.append(node_id)
                dst_list.append(neighbor_id)
                rel_list.append(rel_id)

        if not src_list:
            # Empty graph — create dummy edge
            src_list, dst_list, rel_list = [0], [0], [0]

        edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long, device=device
        )
        edge_type = torch.tensor(rel_list, dtype=torch.long, device=device)

        self.set_graph(edge_index, edge_type)

    def get_param_count(self) -> dict[str, int]:
        """Return parameter counts per component."""
        total = sum(p.numel() for p in self.parameters())
        layer_params = sum(p.numel() for layer in self.layers for p in layer.parameters())
        return {
            "entity_emb":    self.entity_emb.weight.numel(),
            "relation_emb":  self.relation_emb.weight.numel(),
            "bf_layers":     layer_params,
            "score_mlp":     sum(p.numel() for p in self.score_mlp.parameters()),
            "total":         total,
        }

    def extra_repr(self) -> str:
        counts = self.get_param_count()
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim}, n_layers={self.n_layers}, "
            f"total_params={counts['total']:,}"
        )
