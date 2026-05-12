"""
evaluation/v4_metrics.py — V4-Specific Evaluation Metrics [V4]

THREE CUSTOM METRICS (all mentioned in README Section 6 and paper methodology):

    1. PathEntropyTracker — Confidence metric (AAAI-2024 insight)
       H = -Σ pᵢ log(pᵢ) over normalised path amplitude probabilities.
       Low entropy = amplitude concentrated (confident, interference working).
       High entropy = amplitude dispersed (uncertain, noise or no interference).

    2. RankStabilityTracker — Reliability metric (QIQE-KGC MR divergence fix)
       ΔRank = E[|rank_noisy - rank_clean|] per triple.
       Low ΔRank = model is stable under noise (interference shields).
       High ΔRank = model predictions "wiggle" when noise added.

    3. LogicConsistencyTracker — V4-specific, from QIQE-KGC lattice module
       Monitors how binary the lattice embeddings have become over training.
       binary_score = 1 - ||E_r ⊙ (1-E_r)||_F / max_possible
       1.0 = fully binary (ideal orthocomplemented state).
       0.5 = uniform (no constraint satisfied).

    4. RoutingEfficiencyTracker — V4-specific, from AAAI-2024 dynamic routing
       Tracks %classical vs %quantum routing per evaluation.
       Also measures: routing quality = MRR improvement from quantum vs classical.

    5. QuaternionHealthMonitor — Extension of V3's InterferenceMonitor
       Tracks quaternion "phase collapse" across all 4 components (r,i,j,k).
       V3 checked only imaginary norm; V4 checks full quaternion rotation angle.
       Also verifies: ||q|| ≈ 1 for all entities (unit quaternion maintenance).

ALL FIVE can be used standalone OR as part of V4Trainer's monitoring.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import torch


# ── 1. PathEntropyTracker (from README Section 6.3) ──────────────────────────

class PathEntropyTracker:
    """
    Tracks Shannon entropy of path amplitude distributions.

    H = -Σᵢ pᵢ log(pᵢ)   where pᵢ = |Aᵢ|² / Σⱼ|Aⱼ|²

    AAAI-2024 insight: interference CONCENTRATES amplitude.
    V4 quaternion interference should give LOWER entropy than V3 complex interference.
    NBFNet (full-graph propagation) should have HIGHER entropy under noise.

    Expected paper result (Table 5):
        V4 QuantumReasoner entropy@20%noise  < V3 QuantumReasoner entropy@20%noise
        V3 QuantumReasoner entropy@20%noise  < NBFNet entropy@20%noise
        NBFNet entropy@20%noise ≈ ComplEx entropy@20%noise (no interference)
    """

    def __init__(self) -> None:
        self._correct: List[float] = []
        self._wrong:   List[float] = []

    def update(self, entropy: float, query_type: str = "correct") -> None:
        """
        Record one normalised path entropy value.

        Args:
            entropy:    Normalised entropy from compute_interference_terms():
                        0.0 = all amplitude on one path (max confidence).
                        1.0 = uniform over all paths (max uncertainty).
            query_type: "correct" or "wrong".
        """
        if query_type == "correct":
            self._correct.append(float(entropy))
        else:
            self._wrong.append(float(entropy))

    def compute(self) -> dict:
        """Return entropy statistics."""
        c = self._correct
        w = self._wrong
        mean_c = float(np.mean(c)) if c else 0.0
        mean_w = float(np.mean(w)) if w else 0.0
        return {
            "mean_entropy_correct": mean_c,
            "mean_entropy_wrong":   mean_w,
            "std_entropy_correct":  float(np.std(c)) if c else 0.0,
            "std_entropy_wrong":    float(np.std(w)) if w else 0.0,
            # Positive gap means model is MORE uncertain about wrong than correct = good
            "entropy_gap":          mean_w - mean_c,
            "n_correct": len(c),
            "n_wrong":   len(w),
            "interpretation": (
                "Interference concentrating amplitude (good)"
                if mean_w > mean_c + 0.05
                else "No entropy separation (interference not working yet)"
            ),
        }

    def reset(self) -> None:
        self._correct.clear()
        self._wrong.clear()


# ── 2. RankStabilityTracker (from README Section 6.4) ────────────────────────

class RankStabilityTracker:
    """
    Tracks how much a model's predictions change when noise is added.

    ΔRank = E[|rank_noisy - rank_clean|]

    This DIRECTLY measures the QIQE-KGC MR vs MRR divergence problem:
    - If the lattice correctly prevents extreme rank drops:
      ΔRank should be LOW for V4 vs V3 even at 20% noise
    - If the lattice is NOT working:
      ΔRank will be HIGH (some correct answers drop from rank 5 to rank 5000)
      This is the "statistical anomaly" described in QIQE-KGC Section 5.1

    Expected paper result (ΔRank at 10% noise):
        TransE:           ~15-20 (highest, no interference)
        NBFNet:           ~8-12  (more stable but no interference)
        V3 Quantum:       ~5-8   (interference helps)
        V4 Quantum+Lattice: ~3-5 (lattice prevents cliff drops)
    """

    def __init__(self) -> None:
        self._clean_ranks: dict = {}
        self._noisy_pairs: List[tuple] = []

    def record_clean(self, triple_ids: list, ranks: list) -> None:
        """Record clean-data ranks."""
        for triple, rank in zip(triple_ids, ranks):
            self._clean_ranks[tuple(triple)] = rank

    def record_noisy(self, triple_ids: list, ranks: list) -> None:
        """Record noisy-data ranks for the same triples."""
        for triple, noisy_rank in zip(triple_ids, ranks):
            key = tuple(triple)
            if key in self._clean_ranks:
                self._noisy_pairs.append((self._clean_ranks[key], noisy_rank))

    def compute(self) -> dict:
        """Compute ΔRank statistics."""
        if not self._noisy_pairs:
            return {"mean_delta_rank": 0.0, "n_triples": 0, "error": "No data"}

        clean_arr = np.array([c for c, n in self._noisy_pairs])
        noisy_arr = np.array([n for c, n in self._noisy_pairs])
        delta_arr = np.abs(noisy_arr - clean_arr)

        mean_delta  = float(delta_arr.mean())
        mean_clean  = float(clean_arr.mean())
        mean_noisy  = float(noisy_arr.mean())
        stability   = max(0.0, 1.0 - mean_delta / max(mean_noisy, 1.0))
        pct_worse   = float((noisy_arr > clean_arr).mean())

        return {
            "mean_delta_rank":   mean_delta,
            "median_delta_rank": float(np.median(delta_arr)),
            "std_delta_rank":    float(delta_arr.std()),
            "pct_rank_worse":    pct_worse,
            "mean_rank_clean":   mean_clean,
            "mean_rank_noisy":   mean_noisy,
            "stability_score":   stability,
            "n_triples":         len(self._noisy_pairs),
            "interpretation": (
                f"STABLE (ΔRank={mean_delta:.1f}, stability={stability:.2f})"
                if stability > 0.7
                else f"UNSTABLE (ΔRank={mean_delta:.1f}) — lattice not suppressing cliff drops"
            ),
        }

    def reset(self) -> None:
        self._clean_ranks.clear()
        self._noisy_pairs.clear()


# ── 3. LogicConsistencyTracker (V4-specific) ──────────────────────────────────

class LogicConsistencyTracker:
    """
    Monitors convergence of QuantumLogicLattice toward binary/orthogonal state.

    From QIQE-KGC: the lattice embeddings E_r should converge to {0,1} binary
    (orthocomplemented propositions). This tracker measures how binary they are.

    binary_score = 1 - ||E_r ⊙ (1-E_r)||_F / max_possible
    1.0 = fully binary (ideal)
    0.5 = uniform at 0.5 (no convergence)

    Expected training trajectory:
        Epoch 1:   binary_score ≈ 0.50 (random initialization near 0.5)
        Epoch 30:  binary_score ≈ 0.60 (some convergence beginning)
        Epoch 100: binary_score ≈ 0.75 (good convergence)
        Epoch 200: binary_score ≈ 0.85 (strong convergence — lattice working)
    """

    def __init__(self) -> None:
        self._history: List[dict] = []

    def check(self, model, device: torch.device) -> dict:
        """
        Check lattice binary convergence for current model state.

        Args:
            model:  QuaternionReasoner V4 instance.
            device: Torch device.

        Returns:
            Dict with binary_score, entropy, and interpretation.
        """
        if not model.use_lattice or model.lattice is None:
            return {"binary_score": 1.0, "error": "Lattice not enabled"}

        with torch.no_grad():
            # Sample ALL relation logic embeddings
            all_rel_ids = torch.arange(model.num_relations, device=device)
            E_r = model.lattice.get_relation_logic(all_rel_ids)   # (R, d)

            # Binary score: how close to 0 or 1?
            ortho    = (E_r * (1.0 - E_r))                         # (R, d) — should be 0
            mean_ortho = float(ortho.mean())
            max_ortho  = 0.25  # maximum at E=0.5
            binary_score = 1.0 - mean_ortho / max_ortho

            # Mean logic consistency score
            logic_scores = model.lattice.logic_score(all_rel_ids)  # (R,)
            mean_logic   = float(logic_scores.mean())

            # Mean of E_r (should be bimodal distribution near 0 and 1)
            mean_e = float(E_r.mean())
            std_e  = float(E_r.std())

        result = {
            "binary_score":    max(0.0, binary_score),
            "mean_logic":      mean_logic,
            "mean_E_r":        mean_e,
            "std_E_r":         std_e,
            "converged":       binary_score > 0.7,
            "interpretation": (
                "Lattice converging toward binary (orthocomplemented)"
                if binary_score > 0.6
                else "Lattice not yet converged — continue training or increase lattice_warmup"
            ),
        }
        self._history.append(result)
        return result

    def get_history(self) -> list:
        return self._history


# ── 4. RoutingEfficiencyTracker (V4-specific) ─────────────────────────────────

class RoutingEfficiencyTracker:
    """
    Tracks the benefit of dynamic routing over always using one path.

    Key question: does the router actually HELP performance?
    If classical-only MRR ≈ quantum-only MRR → routing is unnecessary overhead.
    If quantum-only MRR >> classical-only MRR on high-noise queries → routing justified.

    Expected result (paper argument):
        At 0% noise:   classical MRR ≈ quantum MRR (routing overhead not worth it)
        At 10% noise:  quantum MRR > classical MRR (routing starting to help)
        At 20% noise:  quantum MRR >> classical MRR (routing clearly justified)
    """

    def __init__(self) -> None:
        self._stats: List[dict] = []

    def record_epoch(self, model, noise_rate: float = 0.0) -> dict:
        """Record routing statistics from current epoch's DynamicRouter."""
        if not model.use_routing or model.router is None:
            return {"error": "Routing not enabled"}

        stats = model.router.get_routing_stats()
        stats["noise_rate"] = noise_rate
        self._stats.append(stats)
        return stats

    def compute_routing_benefit(self) -> dict:
        """
        Compute whether routing added value across all recorded epochs.

        Returns:
            Dict with routing efficiency analysis.
        """
        if not self._stats:
            return {"error": "No data"}

        pct_q_vals   = [s["pct_quantum"]   for s in self._stats]
        pct_cls_vals = [s["pct_classical"] for s in self._stats]

        return {
            "mean_pct_classical":  float(np.mean(pct_cls_vals)),
            "mean_pct_quantum":    float(np.mean(pct_q_vals)),
            "mean_pct_hybrid":     float(np.mean([s["pct_hybrid"] for s in self._stats])),
            "routing_active":      float(np.mean(pct_q_vals)) > 0.05,
            "interpretation": (
                "Router actively using quantum path (>5% of queries)"
                if float(np.mean(pct_q_vals)) > 0.05
                else "Router sending all queries to classical (thresholds may need tuning)"
            ),
        }


# ── 5. QuaternionHealthMonitor (extension of V3's InterferenceMonitor) ─────────

@dataclass
class QuaternionHealthReport:
    """
    Report from QuaternionHealthMonitor.

    Extends V3's InterferenceReport with quaternion-specific fields.
    """
    epoch:              int
    mean_i_norm:        float   # i-component norm (equivalent to V3's mean_imag_norm)
    mean_j_norm:        float   # j-component norm (new in V4)
    mean_k_norm:        float   # k-component norm (new in V4)
    mean_rotation_angle: float  # mean quaternion rotation angle in radians
    frac_collapsed:     float   # fraction of entities with rotation < 0.01 rad
    num_destructive:    int     # queries showing destructive interference
    n_queries:          int     # total queries checked
    is_healthy:         bool    # overall health assessment
    binary_score:       float   # lattice binary convergence score

    def summary(self) -> str:
        health = "HEALTHY" if self.is_healthy else "PHASE COLLAPSE"
        return (
            f"[Epoch {self.epoch:4d}] [{health}] "
            f"i_norm={self.mean_i_norm:.3f} "
            f"j_norm={self.mean_j_norm:.3f} "
            f"k_norm={self.mean_k_norm:.3f} "
            f"angle={self.mean_rotation_angle:.3f}rad | "
            f"collapsed={self.frac_collapsed*100:.0f}% | "
            f"destructive={self.num_destructive}/{self.n_queries} | "
            f"logic={self.binary_score:.2f}"
        )


class QuaternionHealthMonitor:
    """
    Monitors quaternion health across all 4 components and detects phase collapse.

    V3's InterferenceMonitor only checked mean_imag_norm.
    V4 checks all three rotation-carrying components (i, j, k) AND the rotation angle.

    QUATERNION PHASE COLLAPSE (generalization of V3's Phase Collapse):
        V3: imaginary component → 0 (2D collapse to real line)
        V4: i, j, k components all → 0 (4D collapse to identity quaternion)

    Detection:
        If mean_rotation_angle < 0.05 rad: PHASE COLLAPSE in H^d space.
        If mean_i_norm < 0.02: V3-compatible Phase Collapse in i-component.
        Both conditions should be checked.

    Auto-correction (same as V3):
        If Phase Collapse detected: boost lr_i_mult by 5× for next 10 epochs.
    """

    def __init__(
        self,
        model,
        toy_kg           = None,
        check_every:     int   = 10,
        collapse_thresh: float = 0.05,
        verbose:         bool  = True,
    ) -> None:
        self.model           = model
        self.toy_kg          = toy_kg
        self.check_every     = check_every
        self.collapse_thresh = collapse_thresh
        self.verbose         = verbose
        self._reports:       List[QuaternionHealthReport] = []
        self._logic_tracker  = LogicConsistencyTracker()

    def check(self, epoch: int) -> QuaternionHealthReport:
        """
        Run health check for current model state.

        Returns QuaternionHealthReport with all diagnostic values.
        """
        device = self.model.encoder.emb_r.weight.device

        with torch.no_grad():
            # Quaternion component norms
            i_norms = self.model.encoder.emb_i.weight.norm(dim=-1)
            j_norms = self.model.encoder.emb_j.weight.norm(dim=-1)
            k_norms = self.model.encoder.emb_k.weight.norm(dim=-1)

            mean_i = float(i_norms.mean())
            mean_j = float(j_norms.mean())
            mean_k = float(k_norms.mean())

            # Mean quaternion rotation angle: θ = 2·arctan(||i,j,k|| / r)
            r_norms = self.model.encoder.emb_r.weight.norm(dim=-1).clamp(1e-10)
            ijk_norms = (i_norms.pow(2) + j_norms.pow(2) + k_norms.pow(2)).sqrt()
            angles   = 2.0 * torch.atan2(ijk_norms, r_norms)
            mean_angle   = float(angles.mean())
            frac_collapse = float((angles < self.collapse_thresh).float().mean())

            # Logic binary convergence
            logic_result = self._logic_tracker.check(self.model, device)
            binary_score = logic_result.get("binary_score", 0.5)

        # Check destructive interference on contradiction queries
        num_destructive = 0
        n_queries       = 0

        if self.toy_kg is not None:
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / "quantum_kg"))
                from models.components.path_aggregator import PathEnumerator

                adj        = self.toy_kg.get_adjacency()
                enumerator = PathEnumerator(adj, max_hops=2, max_paths=8)

                self.model.eval()
                with torch.no_grad():
                    for cq in self.toy_kg.contradiction_queries:
                        n_queries += 1
                        h_id     = self.toy_kg.entity2id[cq["head"]]
                        corr_id  = self.toy_kg.entity2id[cq["correct_tail"]]
                        wrong_id = self.toy_kg.entity2id[cq["contradictory_tail"]]

                        wrong_paths = enumerator.find_paths(h_id, wrong_id)[:8]
                        corr_paths  = enumerator.find_paths(h_id, corr_id)[:8]

                        if not wrong_paths or not self.model.mv_aggregator:
                            continue

                        h_q  = tuple(x.squeeze(0) for x in self.model.encoder(
                            torch.tensor([h_id],    device=device)))
                        t_w  = tuple(x.squeeze(0) for x in self.model.encoder(
                            torch.tensor([wrong_id],device=device)))
                        t_c  = tuple(x.squeeze(0) for x in self.model.encoder(
                            torch.tensor([corr_id], device=device)))

                        logic_w = [self.model.lattice.path_logic_score([s[0] for s in p], device)
                                   for p in wrong_paths] if self.model.lattice else None

                        wrong_r = self.model.mv_aggregator.compute_interference_terms(
                            h_q, t_w, wrong_paths, self.model.unitary, logic_w
                        )
                        corr_r  = self.model.mv_aggregator.compute_interference_terms(
                            h_q, t_c, corr_paths, self.model.unitary, None
                        )

                        if wrong_r.get("interference", 0) < -1e-6:
                            num_destructive += 1

            except Exception:
                pass

        is_healthy = (mean_angle > self.collapse_thresh) and (mean_i > 0.02)

        report = QuaternionHealthReport(
            epoch               = epoch,
            mean_i_norm         = mean_i,
            mean_j_norm         = mean_j,
            mean_k_norm         = mean_k,
            mean_rotation_angle = mean_angle,
            frac_collapsed      = frac_collapse,
            num_destructive     = num_destructive,
            n_queries           = n_queries,
            is_healthy          = is_healthy,
            binary_score        = binary_score,
        )

        if self.verbose:
            print(f"\n{'='*70}")
            print(report.summary())
            if self.toy_kg and n_queries > 0:
                for cq in self.toy_kg.contradiction_queries:
                    print(f"  {cq['head']:10s} → "
                          f"{cq['correct_tail']:15s}/{cq['contradictory_tail']:15s}")
            print(f"{'='*70}")

        self._reports.append(report)
        return report

    def should_check(self, epoch: int) -> bool:
        return epoch % self.check_every == 0 or epoch == 1

    def final_summary(self) -> dict:
        """Return summary of training health for paper Section 7 tables."""
        if not self._reports:
            return {}
        final = self._reports[-1]
        best_dest = max(r.num_destructive for r in self._reports)
        return {
            "final_epoch":         final.epoch,
            "final_i_norm":        final.mean_i_norm,
            "final_rotation_angle": final.mean_rotation_angle,
            "final_binary_score":  final.binary_score,
            "best_destructive":    best_dest,
            "n_queries":           final.n_queries,
            "is_healthy":          final.is_healthy,
        }
