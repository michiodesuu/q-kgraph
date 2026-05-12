"""
evaluation/metrics.py — Filtered MRR and Hits@K  [V1]

ALWAYS USE FILTERED EVALUATION for FB15k-237 and WN18RR.
All published baselines use filtered MRR. Mixing is a methodological error.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch


@dataclass
class MetricResults:
    mrr:      float = 0.0
    hits_at_1: float = 0.0
    hits_at_3: float = 0.0
    hits_at_10: float = 0.0
    num_triples: int = 0
    mean_rank: float = 0.0

    def to_dict(self) -> dict:
        return {
            "MRR": self.mrr, "H@1": self.hits_at_1,
            "H@3": self.hits_at_3, "H@10": self.hits_at_10,
            "MeanRank": self.mean_rank, "N": self.num_triples,
        }

    def __str__(self) -> str:
        return (
            f"MRR={self.mrr:.4f} | H@1={self.hits_at_1:.4f} | "
            f"H@3={self.hits_at_3:.4f} | H@10={self.hits_at_10:.4f}"
        )

    def __gt__(self, other): return self.mrr > other.mrr
    def __lt__(self, other): return self.mrr < other.mrr


class RankingMetrics:
    """
    Computes filtered MRR and Hits@K for knowledge graph link prediction.

    Filtered evaluation:
        Before ranking all entities as candidate tails for query (h, r, ?),
        set scores of ALL other known-true tails to -∞.
        This prevents penalizing the model for ranking valid answers above
        the specific test tail.

    Args:
        filter_false_negatives: If True (always use), applies filtering.
        hits_k: List of K values for Hits@K. Default [1, 3, 10].
    """

    def __init__(
        self,
        filter_false_negatives: bool = True,
        hits_k: list[int] = None,
    ) -> None:
        self.filter = filter_false_negatives
        self.hits_k = hits_k or [1, 3, 10]
        self.reset()

    def reset(self) -> None:
        self._ranks:     list[int]   = []
        self._rr:        list[float] = []
        self._hits:      dict[int, list] = {k: [] for k in self.hits_k}

    def update(
        self,
        scores:       torch.Tensor,        # (B, num_entities) float
        true_indices: torch.Tensor,        # (B,) int — true tail IDs
        head_ids:     Optional[torch.Tensor] = None,   # (B,) int
        relation_ids: Optional[torch.Tensor] = None,   # (B,) int
        true_tails:   Optional[dict] = None,           # {(h,r): set(t)}
    ) -> None:
        """
        Update metrics for one batch.

        Args:
            scores:       (B, E) score matrix from score_triple_vs_all().
            true_indices: (B,) true tail entity IDs.
            head_ids:     (B,) head entity IDs (for filtering).
            relation_ids: (B,) relation IDs (for filtering).
            true_tails:   {(h_id, r_id): set(t_id)} for filtered evaluation.
        """
        B = scores.shape[0]
        scores = scores.clone().detach().float()

        if torch.isnan(scores).any():
            raise RuntimeError(
                "NaN detected in score matrix — model embeddings have diverged. "
                "Check loss config (label_smoothing, margin) and learning rate."
            )

        for b in range(B):
            # Apply filter: mask out other true tails
            if self.filter and true_tails is not None and head_ids is not None and relation_ids is not None:
                h_id = head_ids[b].item()
                r_id = relation_ids[b].item()
                t_true = true_indices[b].item()

                other_true = true_tails.get((h_id, r_id), set()) - {t_true}
                for t_other in other_true:
                    if t_other < scores.shape[1]:
                        scores[b, t_other] = float("-inf")

            # Compute rank of true tail
            t_id   = true_indices[b].item()
            score_t = scores[b, t_id].item()
            rank    = int((scores[b] > score_t).sum().item()) + 1  # 1-indexed

            self._ranks.append(rank)
            self._rr.append(1.0 / rank)
            for k in self.hits_k:
                self._hits[k].append(1.0 if rank <= k else 0.0)

    def compute(self) -> MetricResults:
        """Compute and return aggregated metrics."""
        if not self._ranks:
            return MetricResults()

        mrr       = float(np.mean(self._rr))
        mean_rank = float(np.mean(self._ranks))
        hits      = {k: float(np.mean(v)) for k, v in self._hits.items()}

        return MetricResults(
            mrr         = mrr,
            hits_at_1   = hits.get(1,  0.0),
            hits_at_3   = hits.get(3,  0.0),
            hits_at_10  = hits.get(10, 0.0),
            num_triples = len(self._ranks),
            mean_rank   = mean_rank,
        )

    def compute_and_reset(self) -> MetricResults:
        result = self.compute()
        self.reset()
        return result


def compare_versions(
    v1_csv: str,
    v2_csv: Optional[str] = None,
    v3_csv: Optional[str] = None,
) -> None:
    """
    Load result CSVs and print a comparison table across versions.
    Called after all three versions have been run.
    """
    import csv
    from pathlib import Path

    all_results = {}

    for label, path in [("V1", v1_csv), ("V2", v2_csv), ("V3", v3_csv)]:
        if path is None or not Path(path).exists():
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get('model', '?')} [{label}]"
                all_results[key] = {k: float(v) for k, v in row.items() if k != "model"}

    if not all_results:
        print("No result CSVs found. Run experiments first.")
        return

    print("\n" + "="*80)
    print("VERSION COMPARISON")
    print("="*80)
    header = ["Model/Version"] + list(next(iter(all_results.values())).keys())
    col_w  = [max(len(h), 20) for h in header]
    print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print("-"*80)
    for model, metrics in all_results.items():
        row = [model] + [f"{v:.4f}" for v in metrics.values()]
        print("  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))
    print("="*80)

class PathEntropyTracker:
    """
    Tracks path entropy across queries to compare model confidence.

    Path entropy H = -Σ pᵢ log(pᵢ) over normalised path amplitude probabilities.

    Low entropy  → amplitude concentrated on a few paths (confident prediction)
    High entropy → amplitude dispersed across all paths (uncertain, noise-like)

    Expected pattern (paper argument):
        QuantumReasoner at high noise: LOW entropy on correct queries
            because interference concentrates amplitude.
        NBFNet at high noise: HIGH entropy on all queries
            because noisy graph propagation spreads activation uniformly.

    Usage:
        tracker = PathEntropyTracker()
        # After compute_interference_terms() for each query:
        tracker.update(analysis["path_entropy_norm"], query_type="correct")
        tracker.update(analysis["path_entropy_norm"], query_type="wrong")
        summary = tracker.compute()
        print(summary["mean_entropy_correct"])   # lower = model is more confident
    """

    def __init__(self) -> None:
        self._correct_entropies: list[float] = []
        self._wrong_entropies:   list[float] = []

    def update(self, entropy: float, query_type: str = "correct") -> None:
        """
        Record one path entropy value.

        Args:
            entropy:    Normalised path entropy from compute_interference_terms().
                        0.0 = all amplitude on one path (max confidence).
                        1.0 = uniform amplitude across all paths (min confidence).
            query_type: "correct" or "wrong".
        """
        if query_type == "correct":
            self._correct_entropies.append(entropy)
        else:
            self._wrong_entropies.append(entropy)

    def compute(self) -> dict:
        """
        Return summary statistics.

        Returns:
            Dict with mean/std entropy for correct and wrong queries,
            and the entropy_gap (wrong - correct).
            Positive entropy_gap = model is more uncertain about wrong answers
            than correct answers = good (interference is working).
        """
        import numpy as np
        c = self._correct_entropies
        w = self._wrong_entropies

        mean_c = float(np.mean(c)) if c else 0.0
        mean_w = float(np.mean(w)) if w else 0.0
        std_c  = float(np.std(c))  if c else 0.0
        std_w  = float(np.std(w))  if w else 0.0

        return {
            "mean_entropy_correct": mean_c,
            "mean_entropy_wrong":   mean_w,
            "std_entropy_correct":  std_c,
            "std_entropy_wrong":    std_w,
            "entropy_gap":          mean_w - mean_c,   # positive = good
            "n_correct_queries":    len(c),
            "n_wrong_queries":      len(w),
            "interpretation": (
                "Interference concentrating amplitude (good)"
                if mean_w > mean_c + 0.1 else
                "No entropy separation yet (Phase Collapse likely)"
            ),
        }

    def reset(self) -> None:
        self._correct_entropies.clear()
        self._wrong_entropies.clear()

class RankStabilityTracker:
    """
    Measures how much a model's rankings change when noise is added.

    ΔRank = average change in the rank of a true triple between clean and noisy evaluation.

    Low ΔRank  → model's predictions are stable under noise (interference shielding)
    High ΔRank → model's predictions are fragile (noise buries the true answer)

    Expected pattern (paper argument):
        TransE ΔRank at 10% noise:        HIGH (~15-20 rank positions lost)
        NBFNet ΔRank at 10% noise:        MEDIUM (~8-12 positions)
        QuantumReasoner ΔRank at 10%:     LOW (~3-6 positions)

    The argument: destructive interference acts as a shield. Even when noise
    corrupts some paths, the interference pattern keeps the correct answer's
    amplitude high relative to wrong answers.

    Usage:
        tracker = RankStabilityTracker()
        # Evaluate same triples on clean model:
        tracker.record_clean(triple_ids, clean_ranks)
        # Evaluate same triples on noise-corrupted model:
        tracker.record_noisy(triple_ids, noisy_ranks)
        # Get ΔRank:
        summary = tracker.compute()
        print(summary["mean_delta_rank"])
    """

    def __init__(self) -> None:
        self._clean_ranks:      dict[tuple, int] = {}
        self._noisy_ranks:      list[tuple[int, int]] = []   # (clean_rank, noisy_rank)

    def record_clean(
        self,
        triple_ids: list[tuple[int, int, int]],
        ranks:      list[int],
    ) -> None:
        """
        Record clean-data ranks for a set of triples.

        Must be called before record_noisy() for the same triples.

        Args:
            triple_ids: List of (h_id, r_id, t_id) triples.
            ranks:      Corresponding filtered ranks (1-indexed).
        """
        for triple, rank in zip(triple_ids, ranks):
            self._clean_ranks[triple] = rank

    def record_noisy(
        self,
        triple_ids: list[tuple[int, int, int]],
        ranks:      list[int],
    ) -> None:
        """
        Record noisy-data ranks for the same triples.

        Must be called after record_clean() has been called for these triples.

        Args:
            triple_ids: Same triples as record_clean().
            ranks:      Ranks computed after noise injection.
        """
        for triple, noisy_rank in zip(triple_ids, ranks):
            if triple in self._clean_ranks:
                self._noisy_ranks.append((self._clean_ranks[triple], noisy_rank))

    def compute(self) -> dict:
        """
        Compute ΔRank statistics.

        Returns:
            Dict with:
                mean_delta_rank:   Average |rank_noisy - rank_clean|.
                median_delta_rank: Median of |ΔRank| (less sensitive to outliers).
                pct_rank_increase: Fraction of triples that got a WORSE rank.
                mean_rank_clean:   Average rank on clean data (reference).
                mean_rank_noisy:   Average rank on noisy data.
                stability_score:   1 - (mean_delta_rank / mean_rank_noisy),
                                   higher = more stable. Range [0, 1].
        """
        import numpy as np

        if not self._noisy_ranks:
            return {"mean_delta_rank": 0.0, "n_triples": 0,
                    "error": "No noisy ranks recorded. Call record_noisy() first."}

        clean_arr = np.array([c for c, n in self._noisy_ranks], dtype=float)
        noisy_arr = np.array([n for c, n in self._noisy_ranks], dtype=float)
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
            "pct_rank_increase": pct_worse,           # fraction that got worse
            "mean_rank_clean":   mean_clean,
            "mean_rank_noisy":   mean_noisy,
            "stability_score":   stability,           # 1.0 = perfectly stable
            "n_triples":         len(self._noisy_ranks),
            "interpretation": (
                f"Stable (ΔRank={mean_delta:.1f}, stability={stability:.2f})"
                if stability > 0.7 else
                f"Unstable (ΔRank={mean_delta:.1f}, stability={stability:.2f})"
            ),
        }

    def reset(self) -> None:
        self._clean_ranks.clear()
        self._noisy_ranks.clear()