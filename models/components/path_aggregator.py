"""
models/components/path_aggregator.py — BFS Path Enumeration + Quantum Interference  [V1]

PURPOSE:
    THE CORE FILE. This file is the paper's contribution in code.
    Every other file either feeds data into this one or evaluates its output.

THE CENTRAL EQUATION:
    P(t|s) = |Σᵢ αᵢ ⟨t|U_Pᵢ|s⟩|²

    Expanded:
        = Σᵢ |αᵢ|²|Aᵢ|²            ← classical sum (per-path Born rule probs)
        + Σᵢ≠ⱼ Re(αᵢαⱼ* AᵢAⱼ*)    ← interference cross-terms (can be NEGATIVE)

    The interference cross-terms are NEGATIVE when Aᵢ and Aⱼ have phase
    difference near π radians. This destroys the probability of wrong answers.

TWO CLASSES:
    PathEnumerator:      BFS over the KG adjacency graph.
                         Finds all paths up to max_hops between entity pairs.
                         Returns: list of [(rel_id, entity_id), ...] per path.

    AmplitudeAggregator: Applies path operators to source state, sums amplitudes,
                         applies Born rule, decomposes into classical + interference.
                         THE KEY METHOD: compute_interference_terms()

THE 27% PROBLEM EXPLAINED:
    Pre-training: 4 correct paths, 7 contradictory paths, random phase angles.
    The 7 wrong-path amplitudes dominate the sum → P(correct) ≈ 27%.
    After training with InterferenceAwareLoss:
        - Correct paths cluster in phase space (constructive)
        - Wrong paths point in opposite direction (destructive)
        - P(wrong) → 0, P(correct) → high

USAGE:
    adj        = kg.get_adjacency()
    enumerator = PathEnumerator(adj, max_hops=3, max_paths=8)
    paths      = enumerator.find_paths(h_id=0, t_id=5)
    # paths = [[(r1, e1), (r2, e2)], [(r3, e3)], ...]

    aggregator = AmplitudeAggregator(complex_dim=8, max_paths=8)
    prob       = aggregator.forward(h_state, t_state, paths, unitary)
    analysis   = aggregator.compute_interference_terms(h_state, t_state, paths, unitary)
    print(analysis["interference"])   # negative = destructive (target after training)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# A path is a list of (relation_id, entity_id) steps
Path = list[tuple[int, int]]


# ── PathEnumerator ─────────────────────────────────────────────────────────────

class PathEnumerator:
    """
    BFS-based path enumeration over a knowledge graph adjacency.

    Finds all paths of length 1 to max_hops between entity pairs.
    Returns at most max_paths paths (shortest first — BFS order).

    WHY BFS INSTEAD OF DFS:
        BFS guarantees shortest paths first. Shorter paths have higher
        information density per amplitude computation.
        For interference: shorter paths are more likely to be causally
        meaningful and less likely to be spurious coincidences.

    SCALABILITY NOTE:
        BFS on FB15k-237 (14k entities, 237 relations) for max_hops=2
        generates paths in ~15-30 minutes when run for all training pairs.
        Use data/path_cache.py (V2) to pre-compute paths once and cache to disk.
        Without the cache: BFS is called per-sample during training → hours/epoch.

    Args:
        adjacency:   Dict: entity_id → list of (relation_id, neighbor_entity_id).
                     Built by toy_kg.get_adjacency() or from training triples.
        max_hops:    Maximum path length (number of relation steps).
                     2 is recommended for training efficiency.
                     3 for analysis (finds more contradictory paths).
        max_paths:   Maximum paths to return per (source, target) pair.
                     More paths = richer interference but slower computation.
        allow_cycles: If False, BFS does not revisit entities within a path.
                      Default False (cycles rarely add meaningful reasoning).
    """

    def __init__(
        self,
        adjacency:    dict[int, list[tuple[int, int]]],
        max_hops:     int  = 2,
        max_paths:    int  = 8,
        allow_cycles: bool = False,
    ) -> None:
        self.adjacency    = adjacency
        self.max_hops     = max_hops
        self.max_paths    = max_paths
        self.allow_cycles = allow_cycles

    def find_paths(
        self,
        source: int,
        target: int,
    ) -> list[Path]:
        """
        Find all paths from source to target via BFS.

        Args:
            source: Source entity ID.
            target: Target entity ID.

        Returns:
            List of paths. Each path is [(rel_id, entity_id), ...].
            The source entity is NOT included (implicit starting point).
            The final entity in each path IS the target.
            Returns empty list if no path found within max_hops.

        Example:
            >>> enumerator.find_paths(platypus_id, warm_blooded_id)
            [[(isa_id, mammal_id), (hasprop_id, warm_blooded_id)],   # correct path
             [(isa_id, reptile_id), (hasprop_id, cold_blooded_id)]]  # contradiction path
        """
        found: list[Path] = []

        # BFS queue: (current_entity, path_so_far, visited_entities)
        queue: deque = deque()
        queue.append((source, [], frozenset([source])))

        while queue and len(found) < self.max_paths:
            current, path, visited = queue.popleft()

            # Check if we reached the target (and moved at least one step)
            if current == target and len(path) > 0:
                found.append(path)
                if len(found) >= self.max_paths:
                    break
                # Continue BFS to find longer paths too
                # (different paths have different phases → richer interference)

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

        # If BFS found nothing via multi-hop, check for direct 1-hop connection
        if not found:
            for rel_id, neighbor_id in self.adjacency.get(source, []):
                if neighbor_id == target:
                    found.append([(rel_id, neighbor_id)])
                    if len(found) >= self.max_paths:
                        break

        return found

    def find_paths_batch(
        self,
        source_ids: list[int],
        target_ids: list[int],
    ) -> list[list[Path]]:
        """
        Find paths for a batch of (source, target) pairs.

        Args:
            source_ids: List of B source entity IDs.
            target_ids: List of B target entity IDs.

        Returns:
            List of B path lists (one per pair).
        """
        return [
            self.find_paths(h, t)
            for h, t in zip(source_ids, target_ids)
        ]

    def count_paths(
        self,
        triple_ids: list[tuple[int, int, int]],
    ) -> dict:
        """
        Count path statistics across a triple set.

        Useful for understanding how many paths exist between training pairs
        and whether the path cache covers them adequately.

        Returns:
            Dict with 'total_pairs', 'pairs_with_paths', 'mean_paths_per_pair',
            'max_paths_found', 'coverage'.
        """
        total     = len(triple_ids)
        covered   = 0
        path_counts = []

        for h_id, r_id, t_id in triple_ids:
            paths = self.find_paths(h_id, t_id)
            if paths:
                covered += 1
            path_counts.append(len(paths))

        return {
            "total_pairs":         total,
            "pairs_with_paths":    covered,
            "coverage":            covered / max(total, 1),
            "mean_paths_per_pair": sum(path_counts) / max(total, 1),
            "max_paths_found":     max(path_counts) if path_counts else 0,
        }


# ── AmplitudeAggregator ────────────────────────────────────────────────────────

class AmplitudeAggregator(nn.Module):
    """
    Computes quantum interference probability P(t|s) = |Σᵢ αᵢ ⟨t|U_Pᵢ|s⟩|².

    This is the paper's core contribution in code form.
    It applies path operators to the source state, sums the complex amplitudes
    with learned weights αᵢ, and applies the Born rule to the total amplitude.

    The Born rule applied AFTER summing (not before) is what produces
    the interference cross-terms. This is the mathematical difference from
    all classical models that either (a) don't sum paths, or (b) apply
    Born rule per-path and then sum.

    Args:
        complex_dim:        Hilbert space dimension.
        max_paths:          Maximum paths to aggregate (truncates if more found).
        learn_path_weights: If True, learn complex path weights αᵢ.
                            If False, use uniform weights αᵢ = 1/√K.
        weight_init_std:    Init std for path weights.
        eps:                Small value for numerical stability.
    """

    def __init__(
        self,
        complex_dim:         int,
        max_paths:           int   = 8,
        learn_path_weights:  bool  = True,
        weight_init_std:     float = 0.01,
        eps:                 float = 1e-10,
    ) -> None:
        super().__init__()
        self.complex_dim        = complex_dim
        self.max_paths          = max_paths
        self.learn_path_weights = learn_path_weights
        self.eps                = eps

        if learn_path_weights:
            # Learned complex path weights α₁, ..., α_K
            # Stored as (max_paths, 2) float [real, imag]
            self.path_weight_real = nn.Parameter(
                torch.randn(max_paths) * weight_init_std
            )
            self.path_weight_imag = nn.Parameter(
                torch.randn(max_paths) * weight_init_std
            )
        else:
            self.register_buffer(
                "uniform_weight",
                torch.ones(max_paths) / math.sqrt(max_paths),
            )

    def _get_weights(self, k: int) -> torch.Tensor:
        """
        Return complex path weights for k paths, normalized.

        Args:
            k: Number of actual paths to aggregate.

        Returns:
            (k,) complex64 weight tensor.
        """
        if self.learn_path_weights:
            real = self.path_weight_real[:k]
            imag = self.path_weight_imag[:k]
            weights = torch.complex(real, imag)
        else:
            w = self.uniform_weight[:k]
            weights = torch.complex(w, torch.zeros_like(w))

        # Normalize: weights / ||weights|| (ensures stable training)
        norm    = weights.abs().pow(2).sum().sqrt().clamp(min=self.eps)
        return weights / norm

    def _evolve_state(
        self,
        source_state: torch.Tensor,   # (complex_dim,) complex
        path:         Path,
        unitary:      "UnitaryOperator",
    ) -> torch.Tensor:
        """
        Evolve source state through a path: |s'⟩ = U_rn ··· U_r1 |s⟩.

        Applies unitary operators in sequence (left-to-right in path order).
        r1 is applied first, rn is applied last.

        Args:
            source_state: Initial entity state |s⟩.
            path:         [(rel_id, entity_id), ...] steps.
            unitary:      UnitaryOperator instance.

        Returns:
            Evolved state |s'⟩ = U_P|s⟩ with same norm as source_state.
        """
        state = source_state.clone()
        for rel_id, _ in path:
            rel_tensor = torch.tensor(
                [rel_id],
                dtype=torch.long,
                device=source_state.device,
            )
            state = unitary.apply(state.unsqueeze(0), rel_tensor).squeeze(0)
        return state

    def forward(
        self,
        source_states: torch.Tensor,          # (B, complex_dim) complex
        target_states: torch.Tensor,          # (B, complex_dim) complex
        paths_batch:   list[list[Path]],      # B lists of paths
        unitary:       "UnitaryOperator",
    ) -> torch.Tensor:
        """
        Compute P(t|s) = |Σᵢ αᵢ ⟨t|U_Pᵢ|s⟩|² for a batch.

        This is the Born rule applied to the weighted superposition of
        path amplitudes. The SUMMING BEFORE SQUARING is what produces
        the interference cross-terms.

        Args:
            source_states: (B, complex_dim) complex — source entity states.
            target_states: (B, complex_dim) complex — target entity states.
            paths_batch:   B lists of Path objects (from PathEnumerator).
            unitary:       UnitaryOperator for relation evolution.

        Returns:
            (B,) float tensor of quantum interference probabilities in [0, 1].
        """
        B      = source_states.shape[0]
        device = source_states.device
        probs  = torch.zeros(B, device=device)

        for b in range(B):
            paths = paths_batch[b][:self.max_paths]

            if not paths:
                # No paths found: fall back to direct 1-hop Born rule
                # (will be handled by quantum_reasoner's score_triple fallback)
                probs[b] = 0.0
                continue

            k       = len(paths)
            weights = self._get_weights(k)     # (k,) complex

            # Compute amplitude A_i = ⟨t|U_Pi|s⟩ for each path
            amplitudes = torch.zeros(k, dtype=torch.complex64, device=device)

            for i, path in enumerate(paths):
                s_evolved = self._evolve_state(source_states[b], path, unitary)
                # Inner product ⟨t|s'⟩ = Σⱼ conj(t_j) * s'_j
                A_i = (target_states[b].conj() * s_evolved).sum()
                amplitudes[i] = A_i

            # Weighted superposition: Σᵢ αᵢ · Aᵢ
            total_amplitude = (weights * amplitudes).sum()

            # Born rule: |Σᵢ αᵢ Aᵢ|²
            probs[b] = total_amplitude.abs().pow(2)

        return probs.clamp(min=0.0, max=1.0 + 1e-6)

    def compute_interference_terms(
        self,
        source_state:  torch.Tensor,   # (complex_dim,) complex
        target_state:  torch.Tensor,   # (complex_dim,) complex
        paths:         list[Path],
        unitary:       "UnitaryOperator",
    ) -> dict:
        """
        Compute and decompose the interference probability into components.

        Returns the classical sum AND the interference cross-terms separately.
        This is the diagnostic method used in:
            - run_toy.py Step 7 (shows 27% pre-training)
            - InterferenceMonitor (detects Phase Collapse)
            - Paper Figure 2 (interference decomposition bar chart)

        Mathematical decomposition:
            P(t|s) = |Σᵢ αᵢ Aᵢ|²
                   = Σᵢ |αᵢ|²|Aᵢ|²              ← classical_sum (always ≥ 0)
                   + Σᵢ≠ⱼ Re(αᵢαⱼ* AᵢAⱼ*)      ← interference (can be < 0)

            interference < 0  →  DESTRUCTIVE  (suppresses this answer)
            interference > 0  →  CONSTRUCTIVE (amplifies this answer)
            interference ≈ 0  →  Phase Collapse (model is classically behaving)

        Args:
            source_state:  Single source entity state (complex_dim,).
            target_state:  Single target entity state (complex_dim,).
            paths:         List of Path objects.
            unitary:       UnitaryOperator.

        Returns:
            Dict with keys:
                'amplitudes':       list of complex path amplitudes Aᵢ
                'path_probabilities': list of |Aᵢ|² per path
                'weights':          list of complex path weights αᵢ
                'total_probability': float — P(t|s) = |Σαᵢ Aᵢ|²
                'classical_sum':    float — Σ|αᵢ|²|Aᵢ|² (monotonic baseline)
                'interference':     float — total - classical (can be negative)
                'interference_sign': "constructive" / "destructive" / "negligible"
                'n_paths':          int — number of paths used
        """
        paths = paths[:self.max_paths]
        if not paths:
            return {
                "amplitudes": [], "path_probabilities": [], "weights": [],
                "total_probability": 0.0, "classical_sum": 0.0,
                "interference": 0.0, "interference_sign": "none", "n_paths": 0,
            }

        k       = len(paths)
        device  = source_state.device
        weights = self._get_weights(k)

        with torch.no_grad():
            # Compute per-path amplitudes
            amplitudes_list = []
            for path in paths:
                s_evolved = self._evolve_state(source_state, path, unitary)
                A_i = (target_state.conj() * s_evolved).sum()
                amplitudes_list.append(A_i)

            amplitudes = torch.stack(amplitudes_list)  # (k,) complex

            # Total probability (Born rule with interference)
            total_amplitude   = (weights * amplitudes).sum()
            total_probability = total_amplitude.abs().pow(2).item()

            # Classical sum (no interference, monotonic)
            classical_sum = ((weights.abs().pow(2)) * (amplitudes.abs().pow(2))).sum().item()

            # Interference = total - classical
            interference = total_probability - classical_sum

            # Determine sign
            if interference < -1e-6:
                sign = "destructive"
            elif interference > 1e-6:
                sign = "constructive"
            else:
                sign = "negligible"

        # Path entropy: H = -Σ pᵢ log(pᵢ) over normalised path probabilities
        path_probs_raw = [abs(a.item())**2 for a in amplitudes]
        prob_sum       = sum(path_probs_raw) + 1e-10
        path_probs_norm = [p / prob_sum for p in path_probs_raw]
        import math
        path_entropy = -sum(
            p * math.log(p + 1e-10) for p in path_probs_norm
        )
        max_entropy  = math.log(k) if k > 1 else 1.0   # uniform baseline
        norm_entropy = path_entropy / max_entropy        # 0=concentrated, 1=uniform

        return {
            "amplitudes":         [a.item() for a in amplitudes],
            "path_probabilities": path_probs_raw,
            "weights":            [w.item() for w in weights],
            "total_probability":  max(0.0, total_probability),
            "classical_sum":      max(0.0, classical_sum),
            "interference":       interference,
            "interference_sign":  sign,
            "n_paths":            k,
            "path_entropy":       path_entropy,          # ← new: raw entropy H
            "path_entropy_norm":  norm_entropy,          # ← new: 0=focused, 1=uniform
        }

    def extra_repr(self) -> str:
        return (
            f"complex_dim={self.complex_dim}, "
            f"max_paths={self.max_paths}, "
            f"learn_path_weights={self.learn_path_weights}"
        )
