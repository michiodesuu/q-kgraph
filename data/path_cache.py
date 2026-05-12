"""
Path Cache — Pre-computation and Disk Storage of BFS Paths.

WHY THIS FILE EXISTS (Challenge 2):
    BFS on FB15k-237 (14,541 entities, 237 relations) for K=3-hop paths during
    training is catastrophically slow when done naively per-sample. Even with
    max_paths=16, running BFS for every training triple at every epoch means:

        ~272,000 training triples × BFS_time ≈ hours per epoch

    This module solves the problem by running BFS ONCE before training starts,
    storing all paths to disk in a compact binary format, and loading them
    into a fast lookup dict that fits in RAM.

HOW IT WORKS:
    1. PathCacheBuilder.build() enumerates all unique (head, tail) pairs in the
       training set and runs BFS for each pair. Takes 10-30 minutes once.
    2. Saves to disk as a compressed numpy .npz file (compact, fast load).
    3. PathCache.load() reads the file and provides O(1) path lookup by
       (head_id, tail_id) key during training.
    4. KGDataset integration: when a PathCache is attached, each batch includes
       pre-computed paths alongside the triples.

STORAGE ESTIMATE:
    FB15k-237 with max_paths=8, max_hops=2:
        ~272k training triples × 8 paths × avg_path_len=2 steps × 2 ints
        ≈ ~17MB compressed — fits easily in RAM.

    FB15k-237 with max_paths=16, max_hops=3:
        ≈ ~80MB compressed — still fine.

USAGE:
    # One-time build (run before training):
    builder = PathCacheBuilder(adjacency, max_hops=2, max_paths=8)
    cache = builder.build(triple_pairs, save_path="data/cache/fb15k237_paths.npz")

    # During training (instantaneous lookup):
    cache = PathCache.load("data/cache/fb15k237_paths.npz")
    paths = cache.get(head_id=3, tail_id=42)   # list of Path objects, immediate
"""

from __future__ import annotations

import os
import time
import pickle
import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# A single reasoning path: list of (relation_id, entity_id) steps
PathSteps = list[tuple[int, int]]


@dataclass
class PathCache:
    """
    In-memory path lookup table loaded from disk.

    After loading, paths[(head_id, tail_id)] returns a list of PathSteps.
    Missing (head, tail) pairs return an empty list.

    Attributes:
        paths:       Dict mapping (head_id, tail_id) -> list of PathSteps.
        num_entities: Total entity count for validation.
        max_hops:    Maximum path length stored.
        max_paths:   Maximum paths per pair stored.
        source_file: Path this cache was loaded from.
    """
    paths:        dict[tuple[int, int], list[PathSteps]] = field(default_factory=dict)
    num_entities: int  = 0
    max_hops:     int  = 2
    max_paths:    int  = 8
    source_file:  str  = ""

    def get(
        self,
        head_id: int,
        tail_id: int,
        fallback_direct: bool = True,
    ) -> list[PathSteps]:
        """
        Return pre-computed paths from head to tail.

        Args:
            head_id:          Source entity ID.
            tail_id:          Target entity ID.
            fallback_direct:  If True and no paths found, returns a single
                              direct "empty path" so the model can still score
                              the triple using the 1-hop Born rule fallback.

        Returns:
            List of PathSteps. Each PathStep is [(rel_id, ent_id), ...].
        """
        result = self.paths.get((head_id, tail_id), [])
        if not result and fallback_direct:
            # Return empty path list — quantum_reasoner.score_triple() handles this
            # by falling back to the direct 1-hop unitary scoring
            return []
        return result

    def get_batch(
        self,
        head_ids: list[int],
        tail_ids: list[int],
    ) -> list[list[PathSteps]]:
        """
        Return paths for a batch of (head, tail) pairs.

        Args:
            head_ids: List of head entity IDs (length B).
            tail_ids: List of tail entity IDs (length B).

        Returns:
            List of length B, each element is a list of PathSteps.
        """
        return [self.get(h, t) for h, t in zip(head_ids, tail_ids)]

    @property
    def size(self) -> int:
        """Number of (head, tail) pairs with cached paths."""
        return len(self.paths)

    @property
    def total_paths(self) -> int:
        """Total number of individual paths stored."""
        return sum(len(v) for v in self.paths.values())

    def coverage(self, triple_pairs: list[tuple[int, int]]) -> float:
        """
        Fraction of given (head, tail) pairs that have at least one cached path.

        Use this to verify the cache covers your training set.
        A coverage below 0.85 means many triples will use the 1-hop fallback.

        Args:
            triple_pairs: List of (head_id, tail_id) tuples from your dataset.

        Returns:
            Float in [0, 1] — higher is better.
        """
        if not triple_pairs:
            return 0.0
        covered = sum(1 for (h, t) in triple_pairs if (h, t) in self.paths)
        return covered / len(triple_pairs)

    def summary(self) -> str:
        return (
            f"PathCache: {self.size} pairs | "
            f"{self.total_paths} total paths | "
            f"max_hops={self.max_hops} | max_paths={self.max_paths} | "
            f"source={self.source_file}"
        )

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        Save the cache to disk using pickle (fastest for nested list-of-tuples).

        Why pickle over HDF5/numpy:
            Paths are ragged (different lengths) and nested (list of lists of tuples).
            Pickle handles this natively with no padding overhead.
            For 80MB cache, pickle load time is ~0.3 seconds.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "paths":        self.paths,
            "num_entities": self.num_entities,
            "max_hops":     self.max_hops,
            "max_paths":    self.max_paths,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = path.stat().st_size / 1e6
        print(f"[PathCache] Saved to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str | Path) -> "PathCache":
        """
        Load a pre-built cache from disk.

        Args:
            path: Path to the .pkl file saved by PathCacheBuilder.build().

        Returns:
            PathCache instance ready for lookup.

        Example:
            >>> cache = PathCache.load("data/cache/fb15k237_2hop_8paths.pkl")
            >>> print(cache.summary())
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Path cache not found: {path}\n"
                f"Run PathCacheBuilder.build() first to create it."
            )

        t0 = time.perf_counter()
        with open(path, "rb") as f:
            payload = pickle.load(f)

        cache = cls(
            paths        = payload["paths"],
            num_entities = payload["num_entities"],
            max_hops     = payload["max_hops"],
            max_paths    = payload["max_paths"],
            source_file  = str(path),
        )
        elapsed = time.perf_counter() - t0
        print(f"[PathCache] Loaded {cache.summary()} in {elapsed:.2f}s")
        return cache

    @classmethod
    def load_or_build(
        cls,
        cache_path: str | Path,
        adjacency:  dict[int, list[tuple[int, int]]],
        triple_pairs: list[tuple[int, int]],
        num_entities: int,
        max_hops:   int = 2,
        max_paths:  int = 8,
        force_rebuild: bool = False,
    ) -> "PathCache":
        """
        Convenience: load from disk if exists, else build and save.

        This is the recommended entry point. Call this once before training:

            cache = PathCache.load_or_build(
                cache_path   = "data/cache/fb15k237_2hop_8paths.pkl",
                adjacency    = toy_kg.get_adjacency(),
                triple_pairs = [(h, t) for h, r, t in train_triples],
                num_entities = toy_kg.num_entities,
                max_hops     = 2,
                max_paths    = 8,
            )

        Args:
            cache_path:    Where to save/load the cache file.
            adjacency:     KG adjacency dict (from KGDataset or toy_kg.get_adjacency()).
            triple_pairs:  (head_id, tail_id) pairs to compute paths for.
            num_entities:  Total entity count.
            max_hops:      Max BFS depth.
            max_paths:     Max paths per pair.
            force_rebuild: If True, always rebuild even if cache exists.

        Returns:
            PathCache ready for lookup.
        """
        cache_path = Path(cache_path)

        if cache_path.exists() and not force_rebuild:
            return cls.load(cache_path)

        print(f"[PathCache] Building cache for {len(triple_pairs)} pairs...")
        builder = PathCacheBuilder(adjacency, max_hops=max_hops, max_paths=max_paths)
        cache = builder.build(triple_pairs, num_entities=num_entities)
        cache.save(cache_path)
        return cache


class PathCacheBuilder:
    """
    Builds a PathCache by running BFS for all (head, tail) pairs.

    This is the expensive one-time operation. Run it before training starts.
    On FB15k-237 with max_hops=2 and max_paths=8, expect ~15-30 minutes.
    On the toy KG, expect ~0.1 seconds.

    Args:
        adjacency:   KG adjacency dict: entity_id -> list of (rel_id, neighbor_id).
        max_hops:    Maximum BFS depth (2 recommended for training, 3 for analysis).
        max_paths:   Maximum paths per (head, tail) pair to store.
        allow_cycles: If False, BFS does not revisit entities in a single path.
        verbose:     If True, prints progress every 10k pairs.

    Example:
        >>> kg = build_toy_kg()
        >>> adj = kg.get_adjacency()
        >>> train_pairs = [(kg.triple_to_ids(t)[0], kg.triple_to_ids(t)[2])
        ...                for t in kg.train_triples]
        >>> builder = PathCacheBuilder(adj, max_hops=2, max_paths=8)
        >>> cache = builder.build(train_pairs, num_entities=kg.num_entities)
        >>> print(cache.summary())
    """

    def __init__(
        self,
        adjacency:    dict[int, list[tuple[int, int]]],
        max_hops:     int  = 2,
        max_paths:    int  = 8,
        allow_cycles: bool = False,
        verbose:      bool = True,
    ) -> None:
        self.adjacency    = adjacency
        self.max_hops     = max_hops
        self.max_paths    = max_paths
        self.allow_cycles = allow_cycles
        self.verbose      = verbose

    def build(
        self,
        triple_pairs:  list[tuple[int, int]],
        num_entities:  int = 0,
    ) -> PathCache:
        """
        Run BFS for all (head, tail) pairs and return a PathCache.

        Args:
            triple_pairs:  List of (head_id, tail_id) pairs.
                           Typically from training triples: [(h, t) for (h,r,t) in train].
            num_entities:  Total entity count (for metadata).

        Returns:
            Populated PathCache.
        """
        # Deduplicate pairs to avoid redundant BFS runs
        unique_pairs = list(set(triple_pairs))
        total = len(unique_pairs)
        paths_dict: dict[tuple[int, int], list[PathSteps]] = {}

        t0 = time.perf_counter()

        for i, (head_id, tail_id) in enumerate(unique_pairs):
            if self.verbose and i % 10000 == 0 and i > 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                eta = (total - i) / rate
                print(
                    f"[PathCacheBuilder] {i}/{total} pairs "
                    f"({100*i/total:.1f}%) | "
                    f"{rate:.0f} pairs/s | "
                    f"ETA {eta/60:.1f} min"
                )

            found = self._bfs(head_id, tail_id)
            if found:
                paths_dict[(head_id, tail_id)] = found

        elapsed = time.perf_counter() - t0
        covered = len(paths_dict)

        if self.verbose:
            print(
                f"[PathCacheBuilder] Done in {elapsed/60:.1f} min | "
                f"{covered}/{total} pairs covered "
                f"({100*covered/total:.1f}%)"
            )

        return PathCache(
            paths        = paths_dict,
            num_entities = num_entities,
            max_hops     = self.max_hops,
            max_paths    = self.max_paths,
        )

    def _bfs(self, source: int, target: int) -> list[PathSteps]:
        """
        BFS from source to target, returning up to max_paths paths of
        length up to max_hops.

        Returns:
            List of paths. Each path is a list of (rel_id, entity_id) steps.
            The source entity is NOT included (it's the starting point).
            The last entity in the path IS the target.
        """
        found: list[PathSteps] = []

        # Queue: (current_entity, path_so_far, visited_entities)
        queue: deque = deque()
        queue.append((source, [], {source}))

        while queue and len(found) < self.max_paths:
            current, path, visited = queue.popleft()

            # Record if we've reached target (and moved at least one step)
            if current == target and len(path) > 0:
                found.append(path[:])   # copy
                if len(found) >= self.max_paths:
                    break
                # Continue BFS to find longer paths too
                # (don't stop — longer paths may have different phases)

            # Don't expand beyond max_hops
            if len(path) >= self.max_hops:
                continue

            # Expand neighbors
            for rel_id, neighbor_id in self.adjacency.get(current, []):
                if not self.allow_cycles and neighbor_id in visited:
                    continue
                new_path    = path + [(rel_id, neighbor_id)]
                new_visited = visited | {neighbor_id}
                queue.append((neighbor_id, new_path, new_visited))

        # If no multi-hop paths found, check for direct single-hop connection
        if not found:
            for rel_id, neighbor_id in self.adjacency.get(source, []):
                if neighbor_id == target:
                    found.append([(rel_id, neighbor_id)])
                    if len(found) >= self.max_paths:
                        break

        return found

    def build_symmetric(
        self,
        triple_pairs:  list[tuple[int, int]],
        num_entities:  int = 0,
    ) -> PathCache:
        """
        Build cache for both (h, t) and (t, h) directions simultaneously.

        Useful when you want to score both tail prediction and head prediction
        during evaluation (both directions are required for the full MRR protocol).

        Args:
            triple_pairs:  List of (head_id, tail_id) from training triples.
            num_entities:  Total entity count.

        Returns:
            PathCache covering both h->t and t->h paths.
        """
        # Add reverse pairs
        both_directions = list(set(
            [(h, t) for h, t in triple_pairs] +
            [(t, h) for h, t in triple_pairs]
        ))
        return self.build(both_directions, num_entities=num_entities)


def build_training_cache(
    toy_kg,                         # ToyKG or equivalent
    cache_dir:  str | Path = "data/cache",
    max_hops:   int        = 2,
    max_paths:  int        = 8,
    force:      bool       = False,
) -> PathCache:
    """
    Convenience function: build and cache paths for a ToyKG training set.

    This is the one-liner you call at the start of training:

        cache = build_training_cache(toy_kg, cache_dir="data/cache")
        # Then pass cache to Trainer or use cache.get(h, t) in your loop.

    Args:
        toy_kg:    ToyKG instance (or any object with get_adjacency() and
                   train_triples).
        cache_dir: Directory to store the .pkl cache file.
        max_hops:  BFS depth.
        max_paths: Max paths per pair.
        force:     If True, rebuild even if cache exists.

    Returns:
        PathCache ready for training.
    """
    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build a stable filename from the parameters
    tag        = f"hops{max_hops}_paths{max_paths}"
    cache_path = cache_dir / f"train_paths_{tag}.pkl"

    adjacency    = toy_kg.get_adjacency()
    train_pairs  = [
        (toy_kg.entity2id[t.head], toy_kg.entity2id[t.tail])
        for t in toy_kg.train_triples
    ]

    return PathCache.load_or_build(
        cache_path   = cache_path,
        adjacency    = adjacency,
        triple_pairs = train_pairs,
        num_entities = toy_kg.num_entities,
        max_hops     = max_hops,
        max_paths    = max_paths,
        force_rebuild= force,
    )
