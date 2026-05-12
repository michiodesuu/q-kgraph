"""
models/baselines/conve.py — ConvE Baseline

ConvE: Convolutional 2D Knowledge Graph Embeddings (Dettmers et al., AAAI 2018)
arxiv.org/abs/1707.01476

score(h, r, t) = σ( vec( f( [ē_h ; r̄_r] * ω ) ) W ) · e_t

Where:
    ē_h, r̄_r : h and r embeddings reshaped to 2D "images" (embed_h × embed_w)
    [; ]      : concatenation along the row axis → (2·embed_h × embed_w) input
    * ω       : 2D convolution with learned filters ω
    f(·)      : ReLU + batch normalization
    W         : linear projection (conv_out_size → embed_dim)
    · e_t     : dot product with all candidate tail embeddings

WHY ConvE IS A KEY ADDITIONAL BASELINE:
    All of TransE, RotatE, ComplEx apply a SINGLE relation operator to h and score
    against t. ConvE applies a 2D CONVOLUTION over the concatenated [h; r] image.
    This is qualitatively different: convolution captures local interaction patterns
    between adjacent h and r dimensions. ConvE is non-linear (ReLU) and uses batch
    normalization, making it strictly more powerful than the linear baselines.

    ConvE vs QuantumReasoner:
        ConvE:           scores one (h, r, t) directly via convolution
                         No path enumeration. No amplitude accumulation.
                         Destructive interference? IMPOSSIBLE. Real-valued output.

        QuantumReasoner: accumulates K complex path amplitudes and applies Born rule.
                         Cross-terms 2Re(AᵢĀⱼ) can be NEGATIVE.
                         ConvE's ReLU ensures all feature map values are ≥ 0,
                         structurally preventing negative contributions.

INTERACTION PATTERNS ConvE CAPTURES:
    Because the 2D kernel sees adjacent pixels (dimensions), ConvE captures:
    - How dimension i of h interacts with adjacent dimensions of r
    - Cross-dimensional patterns that dot products and bilinear models miss
    This is why ConvE outperforms TransE/RotatE on clean data.

    But: it still cannot produce NEGATIVE scores between competing reasoning
    paths, because convolution + ReLU + projection = a non-negative pipeline.

PUBLISHED NUMBERS (paper Table 2):
    FB15k-237: MRR ≈ 0.325, Hits@1 ≈ 0.237, Hits@10 ≈ 0.501
    WN18RR:    MRR ≈ 0.430, Hits@1 ≈ 0.400, Hits@10 ≈ 0.520

USAGE:
    model = ConvE(num_entities=28, num_relations=12, embed_dim=200)
    scores = model.score_triple(h, r, t)          # (B,) float
    scores = model.score_triple_vs_all(h, r)      # (B, E) float
    reg    = model.regularization_loss()

IMPLEMENTATION NOTES:
    - embed_h × embed_w must equal embed_dim (default: 10×20 for embed_dim=200)
    - For embed_dim=64: embed_h=8, embed_w=8
    - Input to conv: (B, 1, 2·embed_h, embed_w) after stacking h and r images
    - Output channels: num_filters (default=32)
    - After conv + flatten + linear: (B, embed_dim) feature vector
    - Scored via dot product with all entity embeddings: (B, E)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvE(nn.Module):
    """
    ConvE: 2D convolutional KGE scoring.

    Args:
        num_entities:       Entity count.
        num_relations:      Relation count.
        embed_dim:          Entity and relation embedding dimension.
        embed_h:            Height of the 2D reshape (embed_h × embed_w = embed_dim).
        embed_w:            Width of the 2D reshape.
        num_filters:        Number of convolutional filters. Default: 32.
        kernel_size:        Convolutional kernel size. Default: (3, 3).
        input_dropout:      Dropout on input embeddings.
        hidden_dropout:     Dropout on conv feature maps.
        feature_map_dropout: Dropout after flattening conv output.
        reg_weight:         L2 regularization weight.
    """

    def __init__(
        self,
        num_entities:         int,
        num_relations:        int,
        embed_dim:            int   = 200,
        embed_h:              int   = 10,
        embed_w:              int   = 20,
        num_filters:          int   = 32,
        kernel_size:          tuple = (3, 3),
        input_dropout:        float = 0.2,
        hidden_dropout:       float = 0.3,
        feature_map_dropout:  float = 0.2,
        reg_weight:           float = 1e-3,
    ) -> None:
        super().__init__()
        if embed_h * embed_w != embed_dim:
            raise ValueError(
                f"embed_h × embed_w must equal embed_dim. "
                f"Got {embed_h}×{embed_w}={embed_h*embed_w} ≠ {embed_dim}. "
                f"Common choices: (10,20) for 200, (8,8) for 64, (4,4) for 16."
            )

        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.embed_h       = embed_h
        self.embed_w       = embed_w
        self.num_filters   = num_filters
        self.reg_weight    = reg_weight

        self.entity_embeddings   = nn.Embedding(num_entities,  embed_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)

        # Batch normalization on input images and on features before scoring
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm1d(embed_dim)

        # Dropout layers
        self.drop_input       = nn.Dropout(input_dropout)
        self.drop_hidden      = nn.Dropout(hidden_dropout)
        self.drop_feature_map = nn.Dropout2d(feature_map_dropout)

        # 2D convolution: input (1, 2*embed_h, embed_w) → (num_filters, H_out, W_out)
        kh, kw = kernel_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )

        # Compute flattened feature size after convolution
        # Input height: 2 * embed_h, width: embed_w
        # After conv (no padding, stride=1): H_out = H_in - kh + 1, W_out = W_in - kw + 1
        h_out = 2 * embed_h - kh + 1
        w_out = embed_w - kw + 1
        self.flat_size = num_filters * h_out * w_out

        # Projection from conv features to embedding space
        self.fc = nn.Linear(self.flat_size, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init for embeddings; default init for conv and linear."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def _get_conv_features(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Forward pass through conv pipeline for a batch of (h, r) pairs.

        Returns:
            (B, embed_dim) float feature vectors.
        """
        B = head_ids.shape[0]
        h = self.entity_embeddings(head_ids)        # (B, d)
        r = self.relation_embeddings(relation_ids)  # (B, d)

        # Apply embedding-level batch norm: treat each embedding as a 2D image
        h_2d = h.view(B, 1, self.embed_h, self.embed_w)  # (B, 1, embed_h, embed_w)
        r_2d = r.view(B, 1, self.embed_h, self.embed_w)  # (B, 1, embed_h, embed_w)

        # Stack h and r vertically: (B, 1, 2*embed_h, embed_w)
        stacked = torch.cat([h_2d, r_2d], dim=2)          # (B, 1, 2H, W)
        stacked = self.bn0(stacked)
        stacked = self.drop_input(stacked)

        # 2D convolution
        x = self.conv(stacked)                            # (B, num_filters, H_out, W_out)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop_feature_map(x)

        # Flatten and project
        x = x.view(B, -1)                                 # (B, flat_size)
        x = self.drop_hidden(x)
        x = self.fc(x)                                    # (B, embed_dim)
        x = self.bn2(x)
        x = F.relu(x)
        return x                                           # (B, embed_dim)

    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        ConvE score: σ( vec(f([ē_h; r̄_r] * ω)) W ) · e_t.

        Returns:
            (B,) float scores in (0, 1) after sigmoid.
        """
        features = self._get_conv_features(head_ids, relation_ids)  # (B, d)
        t = self.entity_embeddings(tail_ids)                         # (B, d)

        # Dot product + sigmoid
        score = (features * t).sum(dim=-1)                           # (B,) logit
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
        features = self._get_conv_features(head_ids, relation_ids)  # (B, d)
        all_e = self.entity_embeddings.weight                        # (E, d)

        logits = features @ all_e.T                                  # (B, E)
        return torch.sigmoid(logits)

    def regularization_loss(self) -> torch.Tensor:
        """L2 regularization on entity and relation embeddings."""
        return self.reg_weight * (
            self.entity_embeddings.weight.pow(2).sum()
            + self.relation_embeddings.weight.pow(2).sum()
        )

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "entity_emb":    self.entity_embeddings.weight.numel(),
            "relation_emb":  self.relation_embeddings.weight.numel(),
            "conv":          sum(p.numel() for p in self.conv.parameters()),
            "fc":            sum(p.numel() for p in self.fc.parameters()),
            "total":         total,
        }

    def extra_repr(self) -> str:
        return (
            f"entities={self.num_entities}, relations={self.num_relations}, "
            f"embed_dim={self.embed_dim} ({self.embed_h}×{self.embed_w}), "
            f"filters={self.num_filters}, flat_size={self.flat_size}"
        )
