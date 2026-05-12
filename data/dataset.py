"""
data/dataset.py — PyTorch Dataset and DataLoader Factory  [V1]

PURPOSE:
    Wraps knowledge graph triple lists into PyTorch-compatible datasets.
    Handles negative sampling, filtered evaluation protocol, and batching.

KEY DESIGN DECISIONS:
    Negative sampling: For each positive triple (h, r, t), generate K corrupted
    triples by randomly replacing either the head or tail entity.
    False-negative avoidance: reject corrupted triples that are actually true.

    Filtered evaluation: Before ranking all entities for (h, r, ?),
    mask out known-true tails with score=-inf. This prevents penalising
    the model for correctly ranking other valid answers above the test tail.
    Published baselines use filtered MRR. Always use filter=True.

USAGE:
    from data.dataset import build_dataloaders
    kg = build_toy_kg()
    train_dl, val_dl, test_dl = build_dataloaders(kg, batch_size=32, num_negatives=8)
    for batch in train_dl:
        positive  = batch["positive"]   # (B, 3) — [h_id, r_id, t_id]
        negatives = batch["negatives"]  # (B, K, 3)
        label     = batch["label"]      # (B,) — 0.9 for positive, 0.0 for negative
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class KGBatch:
    """
    Typed container for one training batch.
    The raw dict form is also valid (DataLoader returns dicts).
    """
    positive:  torch.Tensor   # (B, 3)   int64 — [h_id, r_id, t_id]
    negatives: torch.Tensor   # (B, K, 3) int64
    label:     torch.Tensor   # (B,)      float32 — smoothed positive labels
    idx:       torch.Tensor   # (B,)      int64 — index within dataset


class KGDataset(Dataset):
    """
    PyTorch Dataset for knowledge graph link prediction.

    Each __getitem__ returns one positive triple and K corrupted negatives.

    Args:
        triples:       List of (h_id, r_id, t_id) integer tuples.
        num_entities:  Total entity count (for corruption sampling).
        num_relations: Total relation count.
        num_negatives: Number of corrupted triples to generate per positive.
        true_set:      Set of all (h_id, r_id, t_id) known-true triples.
                       Used for false-negative avoidance during sampling.
        label_smoothing: Positive label value (default 0.9 instead of 1.0).
                         Robustness to noisy KG labels.
        mode:          "train" — generate negatives via sampling.
                       "eval"  — return only the positive (evaluation uses
                                 score_triple_vs_all instead of negatives).
        seed:          Reproducibility seed for negative sampling.
        corrupt_head_prob: Fraction of negatives that corrupt the head
                           (vs tail). Default 0.5 (half head, half tail).
    """

    def __init__(
        self,
        triples:            list[tuple[int, int, int]],
        num_entities:       int,
        num_relations:      int,
        num_negatives:      int   = 8,
        true_set:           Optional[set] = None,
        label_smoothing:    float = 0.9,
        mode:               str   = "train",
        seed:               int   = 42,
        corrupt_head_prob:  float = 0.5,
    ) -> None:
        self.triples           = triples
        self.num_entities      = num_entities
        self.num_relations     = num_relations
        self.num_negatives     = num_negatives
        self.true_set          = true_set or set()
        self.label_smoothing   = label_smoothing
        self.mode              = mode
        self.corrupt_head_prob = corrupt_head_prob
        self._rng              = random.Random(seed)

        # Build (h, r) → set(t) for false-negative avoidance
        self.true_tails: dict[tuple, set] = defaultdict(set)
        self.true_heads: dict[tuple, set] = defaultdict(set)
        for h_id, r_id, t_id in self.true_set:
            self.true_tails[(h_id, r_id)].add(t_id)
            self.true_heads[(r_id, t_id)].add(h_id)

    # ── Dataset interface ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        h_id, r_id, t_id = self.triples[idx]

        if self.mode == "eval":
            return {
                "positive": torch.tensor([h_id, r_id, t_id], dtype=torch.long),
                "negatives": torch.zeros(0, 3, dtype=torch.long),
                "label":    torch.tensor(self.label_smoothing),
                "idx":      torch.tensor(idx, dtype=torch.long),
            }

        # Generate K negative triples
        negatives = self._sample_negatives(h_id, r_id, t_id)

        return {
            "positive":  torch.tensor([h_id, r_id, t_id], dtype=torch.long),
            "negatives": torch.tensor(negatives, dtype=torch.long),
            "label":     torch.tensor(self.label_smoothing),
            "idx":       torch.tensor(idx, dtype=torch.long),
        }

    # ── Negative sampling ──────────────────────────────────────────────────────
    def _sample_negatives(
        self,
        h_id: int,
        r_id: int,
        t_id: int,
        max_attempts: int = 8,
    ) -> list[tuple[int, int, int]]:
        """
        Sample K negative triples by corrupting head or tail.

        Strategy:
            - ceil(K * corrupt_head_prob) negatives corrupt the head
            - floor(K * (1 - corrupt_head_prob)) corrupt the tail
            - False negatives: if corrupted triple is in true_set, resample
              up to max_attempts times before accepting

        Args:
            h_id, r_id, t_id: The positive triple to corrupt.
            max_attempts: Max resampling attempts per negative.

        Returns:
            List of K (h_id, r_id, t_id) corrupted triples.
        """
        negatives = []
        n_head = int(self.num_negatives * self.corrupt_head_prob)
        n_tail = self.num_negatives - n_head

        # Corrupt tail
        for _ in range(n_tail):
            for _ in range(max_attempts):
                t_neg = self._rng.randint(0, self.num_entities - 1)
                if t_neg != t_id and t_neg not in self.true_tails.get((h_id, r_id), set()):
                    break
            negatives.append((h_id, r_id, t_neg))

        # Corrupt head
        for _ in range(n_head):
            for _ in range(max_attempts):
                h_neg = self._rng.randint(0, self.num_entities - 1)
                if h_neg != h_id and h_neg not in self.true_heads.get((r_id, t_id), set()):
                    break
            negatives.append((h_neg, r_id, t_id))

        return negatives

    def _get_all_tail_candidates(
        self,
        h_id: int,
        r_id: int,
    ) -> list[tuple[int, int, int]]:
        """
        Return ALL entities as candidate tails for (h_id, r_id).
        Used in evaluation mode (filtered ranking).
        """
        return [(h_id, r_id, t) for t in range(self.num_entities)]

    # ── Factory class methods ──────────────────────────────────────────────────
    @classmethod
    def from_toy_kg(
        cls,
        toy_kg,
        split:          str   = "train",
        num_negatives:  int   = 8,
        label_smoothing: float = 0.9,
        seed:           int   = 42,
    ) -> "KGDataset":
        """
        Build a KGDataset from a ToyKG instance.

        Args:
            toy_kg:   ToyKG instance from build_toy_kg().
            split:    "train", "val", or "test".
            num_negatives: K negatives per positive.
            label_smoothing: Positive label value.
            seed:     RNG seed.

        Returns:
            KGDataset instance for the specified split.
        """
        split_map = {
            "train": toy_kg.train_triples,
            "val":   toy_kg.val_triples,
            "test":  toy_kg.test_triples,
        }
        if split not in split_map:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        triples = [toy_kg.triple_to_ids(t) for t in split_map[split]]
        true_set = toy_kg.get_true_set(include_contradictions=False)

        return cls(
            triples         = triples,
            num_entities    = toy_kg.num_entities,
            num_relations   = toy_kg.num_relations,
            num_negatives   = num_negatives,
            true_set        = true_set,
            label_smoothing = label_smoothing,
            mode            = "train" if split == "train" else "eval",
            seed            = seed,
        )

    @classmethod
    def from_text_file(
        cls,
        file_path:      str,
        entity2id:      dict[str, int],
        relation2id:    dict[str, int],
        all_triples_path: Optional[str] = None,
        num_negatives:  int   = 128,
        label_smoothing: float = 0.9,
        seed:           int   = 42,
        mode:           str   = "train",
    ) -> "KGDataset":
        """
        Build a KGDataset from a tab-separated triple file.

        File format (one triple per line):
            head_entity TAB relation TAB tail_entity

        Args:
            file_path:     Path to the split file (train.txt, valid.txt, test.txt).
            entity2id:     Entity string → ID mapping.
            relation2id:   Relation string → ID mapping.
            all_triples_path: Path to all triples (for building true_set).
                             If None, uses file_path only.
            num_negatives: K negatives per positive (larger for real datasets).
            label_smoothing: Positive label value.
            seed:          RNG seed.
            mode:          "train" or "eval".

        Returns:
            KGDataset instance.
        """
        def _read_file(path: str) -> list[tuple[int, int, int]]:
            triples = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 3:
                        continue
                    h, r, t = parts
                    if h in entity2id and r in relation2id and t in entity2id:
                        triples.append((entity2id[h], relation2id[r], entity2id[t]))
            return triples

        triples  = _read_file(file_path)
        true_set = set(triples)

        if all_triples_path and all_triples_path != file_path:
            all_triples = _read_file(all_triples_path)
            true_set   = set(all_triples)

        return cls(
            triples         = triples,
            num_entities    = len(entity2id),
            num_relations   = len(relation2id),
            num_negatives   = num_negatives,
            true_set        = true_set,
            label_smoothing = label_smoothing,
            mode            = mode,
            seed            = seed,
        )


# ── DataLoader factory ─────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate function for KGDataset batches.

    Stacks tensors correctly for both train and eval modes.
    Eval mode negatives have shape (0, 3) — filtered by DataLoader.
    """
    positive  = torch.stack([item["positive"]  for item in batch])
    label     = torch.stack([item["label"]     for item in batch])
    idx       = torch.stack([item["idx"]       for item in batch])

    # Handle train vs eval mode (eval has no negatives)
    if batch[0]["negatives"].shape[0] == 0:
        negatives = torch.zeros(len(batch), 0, 3, dtype=torch.long)
    else:
        negatives = torch.stack([item["negatives"] for item in batch])

    return {
        "positive":  positive,
        "negatives": negatives,
        "label":     label,
        "idx":       idx,
    }


def build_dataloaders(
    toy_kg,
    batch_size:     int   = 32,
    num_negatives:  int   = 8,
    label_smoothing: float = 0.9,
    num_workers:    int   = 0,
    pin_memory:     bool  = True,
    seed:           int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from a ToyKG.

    This is the standard entry point for toy KG experiments.
    For real datasets (FB15k-237, WN18RR), use KGDataset.from_text_file()
    and build DataLoaders manually.

    Args:
        toy_kg:         ToyKG instance.
        batch_size:     Batch size for all splits.
        num_negatives:  K negatives per positive (train only).
        label_smoothing: Positive label smoothing.
        num_workers:    DataLoader worker count (0 = main process).
        pin_memory:     Enable GPU pinned memory (auto-disabled on CPU).
        seed:           RNG seed.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    import torch

    # Auto-disable pin_memory if no CUDA
    use_pin_memory = pin_memory and torch.cuda.is_available()

    train_ds = KGDataset.from_toy_kg(
        toy_kg, split="train",
        num_negatives=num_negatives,
        label_smoothing=label_smoothing,
        seed=seed,
    )
    val_ds = KGDataset.from_toy_kg(
        toy_kg, split="val",
        num_negatives=0,  # eval mode
        seed=seed,
    )
    test_ds = KGDataset.from_toy_kg(
        toy_kg, split="test",
        num_negatives=0,  # eval mode
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = use_pin_memory,
        collate_fn  = collate_fn,
        drop_last   = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = use_pin_memory,
        collate_fn  = collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = use_pin_memory,
        collate_fn  = collate_fn,
    )

    return train_loader, val_loader, test_loader


def build_true_tails_dict(
    toy_kg,
    include_contradictions: bool = False,
) -> dict[tuple, set]:
    """
    Build the true_tails dict for filtered evaluation.

    Maps (h_id, r_id) → set of all known-true tail IDs.
    Used by RankingMetrics to mask out true tails before ranking.

    Args:
        toy_kg: ToyKG instance.
        include_contradictions: Include contradiction triples in the true set.

    Returns:
        Dict mapping (h_id, r_id) → set(t_id).
    """
    true_tails: dict[tuple, set] = defaultdict(set)
    for triple in toy_kg.triples:
        if triple.is_contradiction and not include_contradictions:
            continue
        h_id = toy_kg.entity2id[triple.head]
        r_id = toy_kg.relation2id[triple.relation]
        t_id = toy_kg.entity2id[triple.tail]
        true_tails[(h_id, r_id)].add(t_id)
    return dict(true_tails)
