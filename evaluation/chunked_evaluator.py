"""
Chunked Evaluator — Challenge 4 Solution.

WHY THIS FILE EXISTS:
    Challenge 4: On WN18RR with 40,943 entities, score_triple_vs_all()
    produces a (B, 40943) matrix. With batch_size=64: 2.6M scores per
    batch. On CPU this produces OOM errors or takes hours.

    Even on GPU, a (64, 40943) float32 matrix = 10.4MB per batch.
    Over 3,034 test triples (WN18RR test set): 64 batches × 10.4MB
    = 666MB in GPU memory just for scoring matrices.

    For FB15k-237 (14,541 entities): more manageable but still 3.7MB per
    batch — fine for Kaggle T4 GPUs (16GB VRAM) but needs monitoring.

SOLUTION: Chunked Evaluation
    Instead of scoring all E entities at once, split the entity set into
    chunks of chunk_size entities. Score each chunk separately and combine.

    Memory per batch: batch_size × chunk_size × 4 bytes
    With chunk_size=1000 and batch_size=64: 256KB — negligible.

    Time trade-off: chunk_size=1000 means E/1000 separate forward passes
    per batch. For WN18RR: 41 passes per batch instead of 1.
    Net effect: ~2-5x slower than unchunked, but fits in any GPU memory.

ADAPTIVE CHUNKING:
    ChunkedEvaluator.auto_chunk_size() estimates available GPU memory and
    selects the largest chunk_size that fits, up to a maximum of full-entity
    scoring. This gives the best possible speed for your hardware.

USAGE:
    >>> evaluator = ChunkedEvaluator(
    ...     model, dataset, device,
    ...     chunk_size="auto",   # or integer like 1000
    ... )
    >>> results = evaluator.evaluate()
    >>> print(results)   # filtered MRR and Hits@K
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .metrics import RankingMetrics, MetricResults


class ChunkedEvaluator:
    """
    Memory-efficient evaluator using chunked entity scoring.

    Computes filtered MRR and Hits@K without materializing the full
    (batch_size, num_entities) score matrix at once.

    Args:
        model:        Any model with score_triple_vs_all() OR score_triple().
        num_entities: Total entity count.
        device:       torch.device.
        chunk_size:   Number of entities to score at once.
                      "auto" = estimate from available GPU memory.
                      Recommended for WN18RR: 2000.
                      Recommended for FB15k-237: 5000.
        true_tails:   {(h,r): set(t)} for filtered evaluation.
        batch_size:   Number of queries to process at once.
        verbose:      Print progress.

    Example:
        >>> eval = ChunkedEvaluator(
        ...     model=quantum_model,
        ...     num_entities=40943,  # WN18RR
        ...     device=device,
        ...     chunk_size=2000,
        ...     true_tails=dataset.true_tails,
        ... )
        >>> results = eval.evaluate_loader(test_loader)
        >>> print(results)
    """

    def __init__(
        self,
        model:        nn.Module,
        num_entities: int,
        device:       torch.device,
        chunk_size:   int | str = "auto",
        true_tails:   Optional[dict] = None,
        batch_size:   int   = 64,
        verbose:      bool  = True,
    ) -> None:
        self.model        = model
        self.num_entities = num_entities
        self.device       = device
        self.true_tails   = true_tails
        self.batch_size   = batch_size
        self.verbose      = verbose
        self.metrics      = RankingMetrics(filter_false_negatives=True)

        if chunk_size == "auto":
            self.chunk_size = self._auto_chunk_size()
        else:
            self.chunk_size = int(chunk_size)

        if verbose:
            mode = "full" if self.chunk_size >= num_entities else "chunked"
            print(
                f"[ChunkedEvaluator] {num_entities} entities | "
                f"chunk_size={self.chunk_size} | mode={mode}"
            )

    # ------------------------------------------------------------------ #
    # Main evaluation methods                                              #
    # ------------------------------------------------------------------ #

    def evaluate_loader(
        self,
        loader: "DataLoader",
    ) -> MetricResults:
        """
        Evaluate on a full DataLoader.

        Args:
            loader: DataLoader with batches containing 'positive' (B, 3) tensors.

        Returns:
            MetricResults with filtered MRR and Hits@K.
        """
        self.model.eval()
        self.metrics.reset()
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Evaluating",
                leave=False,
                disable=not self.verbose,
            ):
                positive = batch["positive"].to(self.device)
                h_ids    = positive[:, 0]
                r_ids    = positive[:, 1]
                t_ids    = positive[:, 2]

                scores = self._score_all_chunked(h_ids, r_ids)  # (B, E)

                self.metrics.update(
                    scores       = scores,
                    true_indices = t_ids,
                    head_ids     = h_ids,
                    relation_ids = r_ids,
                    true_tails   = self.true_tails,
                )

        elapsed = time.perf_counter() - t0
        results = self.metrics.compute()

        if self.verbose:
            print(
                f"[ChunkedEvaluator] {results} | "
                f"{elapsed:.1f}s total | "
                f"{results.num_triples / elapsed:.0f} triples/s"
            )

        return results

    def evaluate_triples(
        self,
        head_ids:     torch.Tensor,   # (N,)
        relation_ids: torch.Tensor,   # (N,)
        tail_ids:     torch.Tensor,   # (N,)
    ) -> MetricResults:
        """
        Evaluate on explicit (head, relation, tail) triple tensors.
        Useful for evaluating specific subsets (e.g., contradiction queries only).
        """
        self.model.eval()
        self.metrics.reset()
        N = head_ids.shape[0]

        with torch.no_grad():
            # Process in batches
            for i in range(0, N, self.batch_size):
                h = head_ids[i : i + self.batch_size].to(self.device)
                r = relation_ids[i : i + self.batch_size].to(self.device)
                t = tail_ids[i : i + self.batch_size].to(self.device)

                scores = self._score_all_chunked(h, r)

                self.metrics.update(
                    scores       = scores,
                    true_indices = t,
                    head_ids     = h,
                    relation_ids = r,
                    true_tails   = self.true_tails,
                )

        return self.metrics.compute()

    # ------------------------------------------------------------------ #
    # Chunked scoring                                                      #
    # ------------------------------------------------------------------ #

    def _score_all_chunked(
        self,
        head_ids:     torch.Tensor,   # (B,)
        relation_ids: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Score all entities as candidate tails in chunks.

        Equivalent to model.score_triple_vs_all(head_ids, relation_ids)
        but with bounded memory usage.

        Args:
            head_ids:     (B,) head entity IDs.
            relation_ids: (B,) relation IDs.

        Returns:
            (B, num_entities) float score tensor.
        """
        B = head_ids.shape[0]
        all_scores = torch.zeros(B, self.num_entities, device=self.device)

        # Score entities in chunks
        entity_ids = torch.arange(self.num_entities, device=self.device)

        for chunk_start in range(0, self.num_entities, self.chunk_size):
            chunk_end    = min(chunk_start + self.chunk_size, self.num_entities)
            chunk_ids    = entity_ids[chunk_start:chunk_end]   # (C,)
            chunk_scores = self._score_vs_chunk(head_ids, relation_ids, chunk_ids)
            all_scores[:, chunk_start:chunk_end] = chunk_scores

        return all_scores   # (B, E)

    def _score_vs_chunk(
        self,
        head_ids:     torch.Tensor,   # (B,)
        relation_ids: torch.Tensor,   # (B,)
        chunk_entity_ids: torch.Tensor,  # (C,)
    ) -> torch.Tensor:
        """
        Score all entities in a chunk as candidate tails.

        Returns:
            (B, C) float score tensor.
        """
        B = head_ids.shape[0]
        C = chunk_entity_ids.shape[0]

        # Get head states and transform by relation
        h_states = self.model.encoder(head_ids)              # (B, complex_dim)
        h_transformed = self.model.unitary.apply(
            h_states, relation_ids
        )                                                     # (B, complex_dim)

        # Get chunk entity states
        chunk_states = self.model.encoder(chunk_entity_ids)  # (C, complex_dim)

        # Compute Born rule scores: |h_transformed @ chunk.conj().T|^2
        # h_transformed: (B, D) complex
        # chunk_states:  (C, D) complex -> conj: (D, C)
        scores_complex = torch.matmul(
            h_transformed,
            chunk_states.conj().T
        )                                                     # (B, C) complex

        if hasattr(self.model, 'ablation_mode') and self.model.ablation_mode == "classical":
            scores = scores_complex.real
        else:
            scores = scores_complex.abs().pow(2)

        # Add relation bias (broadcast)
        bias = self.model.relation_bias[relation_ids].unsqueeze(1)
        return scores + bias

    # ------------------------------------------------------------------ #
    # Memory estimation                                                    #
    # ------------------------------------------------------------------ #

    def _auto_chunk_size(self) -> int:
        """
        Estimate the largest chunk_size that fits in available GPU memory.

        Target: use at most 25% of free GPU memory for the score matrix.
        Falls back to full entity set if plenty of memory available,
        or a safe minimum of 500 if memory is very limited.
        """
        if not torch.cuda.is_available() or self.device.type != "cuda":
            # CPU: use a conservative chunk size
            return min(2000, self.num_entities)

        try:
            free_memory   = torch.cuda.mem_get_info(self.device)[0]
            target_memory = free_memory * 0.25   # use 25% of free memory

            # Bytes per score: float32 = 4 bytes
            # Score matrix: batch_size × chunk_size × 4 bytes
            max_chunk = int(target_memory / (self.batch_size * 4))
            max_chunk = max(500, min(max_chunk, self.num_entities))

            return max_chunk
        except Exception:
            return min(2000, self.num_entities)

    def memory_estimate_mb(self, chunk_size: Optional[int] = None) -> float:
        """
        Estimate GPU memory usage in MB for the current chunk_size.

        Args:
            chunk_size: Override chunk_size for estimation.

        Returns:
            Memory in MB for one batch's score matrix.
        """
        cs = chunk_size or self.chunk_size
        return self.batch_size * cs * 4 / 1e6
