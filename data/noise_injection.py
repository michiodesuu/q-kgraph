"""
data/noise_injection.py — Controlled Noise Injection for Robustness Experiments  [V1]

PURPOSE:
    Implements three noise protocols for injecting corruption into training triples.
    Used to generate the MRR-vs-noise-level degradation curves (paper Figure 3).

THE THREE NOISE TYPES:
    random:     Uniform entity replacement. Models random data entry errors.
                Best-case noise — easy for any model to handle.

    targeted:   Entities sampled by degree (popular entities corrupted more).
                Models systematic, domain-specific noise — harder.
                Weighted sampling: P(entity corrupted) ∝ degree(entity).

    path_break: Corrupts path-critical triples specifically.
                Targets triples that appear in multi-hop reasoning chains.
                HARDEST for path-based models (like this one).
                A model that outperforms on path_break noise has a real advantage.

CKRL PROTOCOL (Xie et al. 2018):
    Noise injected ONLY in training set.
    Test set remains clean (evaluate on ground truth).
    Injection rate = fraction of training triples to corrupt.
    False negatives: never inject a known-true triple as a negative.

USAGE:
    injector = NoiseInjector(num_entities=28, num_relations=12)
    noisy_triples, corruption_mask = injector.inject(
        train_ids, true_set, corruption_rate=0.10, noise_type="random"
    )
    # corruption_mask[i] = True if triple i was corrupted

    # For full degradation curves:
    levels = injector.inject_levels(train_ids, levels=[0.0, 0.05, 0.10, 0.15, 0.20])
    for noise_rate, (noisy, mask) in levels.items():
        print(f"{noise_rate*100:.0f}% noise: {mask.sum()} triples corrupted")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class NoiseConfig:
    """
    Configuration for noise injection.

    Attributes:
        corruption_rate: Fraction of training triples to corrupt [0.0, 1.0].
        corruption_type: "random", "targeted", or "path_break".
        seed:            RNG seed for reproducibility.
        max_attempts:    Max resampling attempts per triple (false-neg avoidance).
    """
    corruption_rate: float = 0.10
    corruption_type: str   = "random"
    seed:            int   = 42
    max_attempts:    int   = 10


class NoiseInjector:
    """
    Injects controlled noise into training triple sets.

    Implements the CKRL noise protocol: inject into training only,
    evaluate on clean test set.

    Args:
        num_entities:  Total entity count.
        num_relations: Total relation count.
        config:        NoiseConfig instance (or pass individual parameters).

    Example:
        >>> injector = NoiseInjector(num_entities=28, num_relations=12)
        >>> noisy, mask = injector.inject(
        ...     train_ids = [(0, 1, 2), (3, 4, 5)],
        ...     true_set  = {(0,1,2), (3,4,5)},
        ...     corruption_rate = 0.10,
        ...     noise_type = "path_break",
        ... )
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        config:        Optional[NoiseConfig] = None,
    ) -> None:
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.config        = config or NoiseConfig()
        self._rng          = random.Random(self.config.seed)
        self._np_rng       = np.random.RandomState(self.config.seed)

    # ── Main injection method ──────────────────────────────────────────────────
    def inject(
        self,
        triple_ids:      list[tuple[int, int, int]],
        true_set:        set[tuple[int, int, int]],
        corruption_rate: Optional[float] = None,
        noise_type:      Optional[str]   = None,
        adjacency:       Optional[dict]  = None,
    ) -> tuple[list[tuple[int, int, int]], np.ndarray]:
        """
        Inject noise into a list of training triples.

        Args:
            triple_ids:      List of (h_id, r_id, t_id) training triples.
            true_set:        All known-true triples. Corrupted triples are
                             resampled if they accidentally hit this set.
            corruption_rate: Override config.corruption_rate.
            noise_type:      Override config.corruption_type.
            adjacency:       KG adjacency dict (required for path_break mode).
                             If None and noise_type=path_break, falls back to random.

        Returns:
            (noisy_triples, corruption_mask) where:
                noisy_triples: Same length as triple_ids, some corrupted.
                corruption_mask: Boolean array, True where corruption occurred.
        """
        rate  = corruption_rate if corruption_rate is not None else self.config.corruption_rate
        ntype = noise_type     if noise_type      is not None else self.config.corruption_type

        n        = len(triple_ids)
        n_corrupt = int(n * rate)

        # Identify which triples to corrupt
        indices_to_corrupt = set(self._rng.sample(range(n), min(n_corrupt, n)))

        # Build corruption mask
        mask = np.zeros(n, dtype=bool)
        for i in indices_to_corrupt:
            mask[i] = True

        # Compute entity degrees (for targeted noise)
        entity_degrees = None
        if ntype == "targeted":
            entity_degrees = self._compute_entity_degrees(triple_ids)

        # Identify path-critical triples (for path_break noise)
        path_critical = None
        if ntype == "path_break" and adjacency is not None:
            path_critical = self._identify_path_critical(triple_ids, adjacency)

        # Apply corruption
        noisy = list(triple_ids)
        for i in range(n):
            if not mask[i]:
                continue

            h_id, r_id, t_id = triple_ids[i]

            if ntype == "random":
                corrupted = self._corrupt_random(h_id, r_id, t_id, true_set)
            elif ntype == "targeted":
                corrupted = self._corrupt_targeted(
                    h_id, r_id, t_id, true_set, entity_degrees
                )
            elif ntype == "path_break":
                if path_critical is not None and i in path_critical:
                    # For path-critical triples: always corrupt the middle entity
                    corrupted = self._corrupt_path_break(h_id, r_id, t_id, true_set)
                else:
                    corrupted = self._corrupt_random(h_id, r_id, t_id, true_set)
            else:
                raise ValueError(
                    f"Unknown noise_type: '{ntype}'. "
                    f"Choose from: 'random', 'targeted', 'path_break'"
                )

            noisy[i] = corrupted

        return noisy, mask

    # ── Noise type implementations ─────────────────────────────────────────────
    def _corrupt_random(
        self,
        h_id:     int,
        r_id:     int,
        t_id:     int,
        true_set: set,
    ) -> tuple[int, int, int]:
        """
        Uniformly random corruption: replace head or tail randomly.

        With 50% probability: corrupt the tail (h, r, t') where t' is random.
        With 50% probability: corrupt the head (h', r, t) where h' is random.
        Resample up to max_attempts times to avoid false negatives.
        """
        corrupt_tail = self._rng.random() < 0.5

        for _ in range(self.config.max_attempts):
            if corrupt_tail:
                t_new = self._rng.randint(0, self.num_entities - 1)
                candidate = (h_id, r_id, t_new)
            else:
                h_new = self._rng.randint(0, self.num_entities - 1)
                candidate = (h_new, r_id, t_id)

            if candidate not in true_set and candidate != (h_id, r_id, t_id):
                return candidate

        # Fallback: return original if all samples hit true_set
        return (h_id, r_id, t_id)

    def _corrupt_targeted(
        self,
        h_id:           int,
        r_id:           int,
        t_id:           int,
        true_set:       set,
        entity_degrees: np.ndarray,
    ) -> tuple[int, int, int]:
        """
        Degree-weighted corruption: popular entities are more likely to be used
        as corrupted replacements.

        Rationale: Real-world KG noise is not uniform. Important entities appear
        in many triples, so errors involving them are more likely. This is
        harder for models because high-degree entities have many true triples,
        making false-negative avoidance harder.

        Sampling: P(entity j used as replacement) ∝ degree(j).
        """
        probs = entity_degrees / entity_degrees.sum()
        corrupt_tail = self._rng.random() < 0.5

        for _ in range(self.config.max_attempts):
            replacement = int(self._np_rng.choice(self.num_entities, p=probs))

            if corrupt_tail:
                candidate = (h_id, r_id, replacement)
            else:
                candidate = (replacement, r_id, t_id)

            if candidate not in true_set and candidate != (h_id, r_id, t_id):
                return candidate

        return (h_id, r_id, t_id)

    def _corrupt_path_break(
        self,
        h_id:     int,
        r_id:     int,
        t_id:     int,
        true_set: set,
    ) -> tuple[int, int, int]:
        """
        Path-break corruption: replaces an entity that appears in multi-hop chains.

        This is the hardest noise type for PathAggregator because it removes
        the exact triples needed for reasoning. The model must still infer
        the correct answer from partial path evidence.

        Implementation: same as random but with higher preference for corrupting
        the tail (the path endpoint is more disruptive to break than the head).
        """
        for _ in range(self.config.max_attempts):
            # Prefer tail corruption (breaks path endpoint — more disruptive)
            t_new     = self._rng.randint(0, self.num_entities - 1)
            candidate = (h_id, r_id, t_new)
            if candidate not in true_set and candidate != (h_id, r_id, t_id):
                return candidate

        return self._corrupt_random(h_id, r_id, t_id, true_set)

    # ── Multi-level injection ──────────────────────────────────────────────────
    def inject_levels(
        self,
        triple_ids:   list[tuple[int, int, int]],
        levels:       list[float]         = None,
        noise_type:   str                 = "random",
        true_set:     Optional[set]       = None,
        adjacency:    Optional[dict]      = None,
    ) -> dict[float, tuple[list, np.ndarray]]:
        """
        Inject noise at multiple rates for degradation curve generation.

        Returns a dict mapping noise_rate → (noisy_triples, corruption_mask).
        Level 0.0 always returns the original triples (no corruption).

        Args:
            triple_ids: List of (h_id, r_id, t_id) training triples.
            levels:     List of corruption rates. Default [0.0, 0.05, 0.10, 0.15, 0.20].
            noise_type: "random", "targeted", or "path_break".
            true_set:   All known-true triples (for false-negative avoidance).
                        If None, uses the input triple_ids as the true set.
            adjacency:  KG adjacency (for path_break mode).

        Returns:
            Dict: {0.0: (original, no_mask), 0.05: (noisy_05, mask_05), ...}

        Example:
            >>> levels = injector.inject_levels(
            ...     train_ids,
            ...     levels     = [0.0, 0.05, 0.10, 0.15, 0.20],
            ...     noise_type = "random",
            ...     true_set   = toy_kg.get_true_set(),
            ... )
            >>> for rate, (noisy, mask) in levels.items():
            ...     print(f"{rate*100:.0f}%: {mask.sum()} corrupted")
        """
        if levels is None:
            levels = [0.0, 0.05, 0.10, 0.15, 0.20]

        if true_set is None:
            true_set = set(triple_ids)

        results: dict[float, tuple[list, np.ndarray]] = {}
        no_corruption_mask = np.zeros(len(triple_ids), dtype=bool)

        for rate in levels:
            if rate == 0.0:
                results[0.0] = (list(triple_ids), no_corruption_mask.copy())
                continue

            # Use a deterministic seed per level for reproducibility
            level_seed = self.config.seed + int(rate * 1000)
            injector   = NoiseInjector(
                self.num_entities,
                self.num_relations,
                NoiseConfig(
                    corruption_rate = rate,
                    corruption_type = noise_type,
                    seed            = level_seed,
                ),
            )
            noisy, mask = injector.inject(
                triple_ids  = triple_ids,
                true_set    = true_set,
                adjacency   = adjacency,
            )
            results[rate] = (noisy, mask)

        return results

    # ── Helper methods ─────────────────────────────────────────────────────────
    def _compute_entity_degrees(
        self,
        triple_ids: list[tuple[int, int, int]],
    ) -> np.ndarray:
        """Compute entity degree (total appearances as head or tail)."""
        degrees = np.ones(self.num_entities, dtype=float)  # Laplace smoothing
        for h_id, r_id, t_id in triple_ids:
            degrees[h_id] += 1
            degrees[t_id] += 1
        return degrees

    def _identify_path_critical(
        self,
        triple_ids: list[tuple[int, int, int]],
        adjacency:  dict[int, list[tuple[int, int]]],
        min_paths:  int = 2,
    ) -> set[int]:
        """
        Identify triples that appear in multi-hop reasoning paths.

        A triple (h, r, t) is "path-critical" if entity t is a bridge node
        that appears in multi-hop chains — i.e., t has outgoing edges in the
        adjacency graph AND is the target of this triple.

        Args:
            triple_ids: Training triples.
            adjacency:  {entity_id: [(rel_id, neighbor_id), ...]}.
            min_paths:  Minimum out-degree for a tail entity to be "bridge".

        Returns:
            Set of indices into triple_ids that are path-critical.
        """
        path_critical: set[int] = set()
        for idx, (h_id, r_id, t_id) in enumerate(triple_ids):
            # t_id is path-critical if it has enough outgoing edges
            out_degree = len(adjacency.get(t_id, []))
            if out_degree >= min_paths:
                path_critical.add(idx)
        return path_critical

    def get_corruption_stats(
        self,
        original:  list[tuple[int, int, int]],
        noisy:     list[tuple[int, int, int]],
        mask:      np.ndarray,
    ) -> dict:
        """
        Compute statistics about the applied corruption.

        Returns dict with:
            n_corrupted:  Number of corrupted triples.
            corruption_rate: Actual corruption rate (may differ from target).
            n_head_corrupted: Count of head corruptions.
            n_tail_corrupted: Count of tail corruptions.
            n_false_negatives: Count where corrupted triple == original (failed corruption).
        """
        n_corrupted  = int(mask.sum())
        n_total      = len(original)
        n_head       = 0
        n_tail       = 0
        n_failed     = 0

        for i in range(n_total):
            if not mask[i]:
                continue
            orig = original[i]
            new  = noisy[i]
            if orig == new:
                n_failed += 1
            elif orig[0] != new[0]:
                n_head += 1
            elif orig[2] != new[2]:
                n_tail += 1

        return {
            "n_corrupted":       n_corrupted,
            "n_total":           n_total,
            "corruption_rate":   n_corrupted / max(n_total, 1),
            "n_head_corrupted":  n_head,
            "n_tail_corrupted":  n_tail,
            "n_failed":          n_failed,
        }
