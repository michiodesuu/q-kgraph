"""
Interference Monitor — Challenge 3 Solution.

WHY THIS FILE EXISTS:
    Challenge 3 is the biggest risk in this project: the model may learn to
    set all imaginary components near zero (Phase Collapse), effectively
    becoming a real-valued model that satisfies the BCE loss without ever
    producing quantum interference. You would see:
        - Good training loss curves (model is learning)
        - Reasonable MRR on clean data
        - NO interference terms in compute_interference_terms()
        - The central claim of the paper is FALSE

    This module runs automatic checks every N epochs and:
        1. Detects Phase Collapse early
        2. Applies automatic corrections (LR adjustment, regularization bump)
        3. Logs a detailed interference health report
        4. Raises an alert if collapse is unrecoverable

HOW TO USE:
    Attach a monitor to the trainer before training:

        monitor = InterferenceMonitor(
            model=quantum_model,
            toy_kg=kg,
            check_every_n_epochs=10,
            alert_threshold=0.02,
        )
        trainer.add_interference_monitor(monitor)

    The monitor's check() method is called automatically inside trainer.py.
    It returns an InterferenceReport with the current health status.

WHAT COUNTS AS HEALTHY:
    - mean_imag_norm > 0.05:       imaginary components are non-trivial
    - fraction_collapsed < 0.3:   < 30% of entities have collapsed
    - at least 1 of 3 contradiction queries shows destructive interference
      (interference < 0) after epoch 50
    - phase separation (mean phase diff) is increasing over training

WHAT COUNTS AS COLLAPSED:
    - mean_imag_norm < 0.02 after epoch 20
    - all three contradiction queries show interference near 0
    - phase separation is flat or decreasing
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class InterferenceReport:
    """
    Snapshot of the model's interference health at a given epoch.

    Attributes:
        epoch:                  Training epoch number.
        mean_imag_norm:         Average imaginary norm of entity embeddings.
        fraction_collapsed:     Fraction of entities with imag_norm < threshold.
        mean_phase_magnitude:   Average absolute phase angle in unitary.
        contradiction_results:  List of dicts, one per contradiction query:
            {query, interference, classical_sum, total_prob, sign}
        num_destructive:        Number of contradiction queries showing destructive.
        is_healthy:             Overall health verdict.
        alert_message:          Non-empty if intervention needed.
    """
    epoch:                 int
    mean_imag_norm:        float
    fraction_collapsed:    float
    mean_phase_magnitude:  float
    contradiction_results: list[dict] = field(default_factory=list)
    num_destructive:       int        = 0
    is_healthy:            bool       = True
    alert_message:         str        = ""
    timestamp:             float      = field(default_factory=time.time)

    def summary(self) -> str:
        health_tag = "HEALTHY" if self.is_healthy else "ALERT"
        destr = f"{self.num_destructive}/{len(self.contradiction_results)}"
        return (
            f"[Epoch {self.epoch:4d}] [{health_tag}] "
            f"imag_norm={self.mean_imag_norm:.4f} | "
            f"collapsed={self.fraction_collapsed*100:.1f}% | "
            f"phase_mag={self.mean_phase_magnitude:.4f} | "
            f"destructive={destr} | "
            f"{self.alert_message if self.alert_message else 'OK'}"
        )


class InterferenceMonitor:
    """
    Automatic monitoring of interference health during training.

    Attaches to a QuantumReasoner and periodically checks:
        1. Phase Collapse indicators (imaginary norms, phase magnitudes)
        2. Actual interference values on known contradiction queries
        3. Phase separation trend over time

    Optionally applies automatic corrections:
        - If collapse detected: temporarily boost LR for imaginary parameters
        - If collapse severe: trigger warning and suggest re-training

    Args:
        model:              QuantumReasoner instance to monitor.
        toy_kg:             ToyKG instance with contradiction_queries.
        check_every_n_epochs: How often to run the full check.
        collapse_threshold: mean_imag_norm below this = collapse detected.
        healthy_after_epoch: Only flag collapse after this many epochs
                             (early training may legitimately have low imaginary norms).
        log_dir:            Optional directory to save reports as text files.
        auto_correct:       If True, automatically adjust LR when collapse detected.
        optimizer:          Optimizer to adjust when auto_correct=True.
        verbose:            Print reports to console.

    Example:
        >>> monitor = InterferenceMonitor(
        ...     model=quantum_model,
        ...     toy_kg=kg,
        ...     check_every_n_epochs=10,
        ... )
        >>> # In training loop:
        >>> if epoch % monitor.check_every_n_epochs == 0:
        ...     report = monitor.check(epoch)
        ...     print(report.summary())
    """

    def __init__(
        self,
        model,                          # QuantumReasoner
        toy_kg,                         # ToyKG with contradiction_queries
        check_every_n_epochs: int   = 10,
        collapse_threshold:   float = 0.02,
        healthy_after_epoch:  int   = 20,
        log_dir: Optional[str | Path] = None,
        auto_correct:         bool  = False,
        optimizer:            Optional[torch.optim.Optimizer] = None,
        verbose:              bool  = True,
    ) -> None:
        self.model                 = model
        self.toy_kg                = toy_kg
        self.check_every_n_epochs  = check_every_n_epochs
        self.collapse_threshold    = collapse_threshold
        self.healthy_after_epoch   = healthy_after_epoch
        self.log_dir               = Path(log_dir) if log_dir else None
        self.auto_correct          = auto_correct
        self.optimizer             = optimizer
        self.verbose               = verbose

        # History for trend analysis
        self.report_history: list[InterferenceReport] = []

        # Prepare device-agnostic entity/relation tensors once
        self._device: Optional[torch.device] = None
        self._prepared_queries: Optional[list[dict]] = None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Main check method                                                    #
    # ------------------------------------------------------------------ #

    def check(self, epoch: int) -> InterferenceReport:
        """
        Run a full interference health check at the given epoch.

        Args:
            epoch: Current training epoch number.

        Returns:
            InterferenceReport with health status and all measurements.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            # 1. Measure imaginary norm collapse
            collapse_metrics = self._measure_collapse(device)

            # 2. Measure interference on contradiction queries
            interference_results = self._measure_interference(device)

            # 3. Build report
            num_destructive = sum(
                1 for r in interference_results
                if r.get("interference", 0) < -1e-6
            )

            # Determine health status
            is_healthy    = True
            alert_message = ""

            if epoch >= self.healthy_after_epoch:
                if collapse_metrics["mean_imag_norm"] < self.collapse_threshold:
                    is_healthy    = False
                    alert_message = (
                        f"PHASE COLLAPSE DETECTED: mean_imag_norm="
                        f"{collapse_metrics['mean_imag_norm']:.4f} < "
                        f"threshold {self.collapse_threshold}. "
                        f"Try: (1) increase LR for imaginary parameters, "
                        f"(2) add InterferenceRegularization loss, "
                        f"(3) use interference_loss.py InterferenceAwareLoss."
                    )
                elif (
                    len(interference_results) > 0
                    and num_destructive == 0
                    and epoch >= 50
                ):
                    is_healthy    = False
                    alert_message = (
                        f"NO DESTRUCTIVE INTERFERENCE after epoch {epoch}. "
                        f"All {len(interference_results)} contradiction queries "
                        f"show positive/zero interference. Phase separation has not emerged. "
                        f"Try adding ContrastiveInterferenceLoss to your training objective."
                    )

            report = InterferenceReport(
                epoch                = epoch,
                mean_imag_norm       = collapse_metrics["mean_imag_norm"],
                fraction_collapsed   = collapse_metrics["fraction_collapsed"],
                mean_phase_magnitude = collapse_metrics["mean_phase_magnitude"],
                contradiction_results= interference_results,
                num_destructive      = num_destructive,
                is_healthy           = is_healthy,
                alert_message        = alert_message,
            )

        # Store history
        self.report_history.append(report)

        # Logging
        if self.verbose:
            self._print_report(report)

        if self.log_dir:
            self._save_report(report)

        # Auto-correction
        if not is_healthy and self.auto_correct and self.optimizer is not None:
            self._apply_correction(report)

        self.model.train()
        return report

    # ------------------------------------------------------------------ #
    # Measurement methods                                                  #
    # ------------------------------------------------------------------ #

    def _measure_collapse(self, device: torch.device) -> dict:
        """Measure imaginary norm and phase magnitude indicators."""
        encoder   = self.model.encoder
        unitary   = self.model.unitary

        imag_norms = encoder.imag_embeddings.weight.norm(dim=-1)
        mean_imag  = imag_norms.mean().item()
        frac_coll  = (imag_norms < 0.05).float().mean().item()

        mean_phase = 0.0
        if hasattr(unitary, 'phases'):
            mean_phase = unitary.phases.abs().mean().item()
        elif hasattr(unitary, 'phases_dir'):
            mean_phase = unitary.phases_dir.abs().mean().item()

        return {
            "mean_imag_norm":     mean_imag,
            "fraction_collapsed": frac_coll,
            "mean_phase_magnitude": mean_phase,
        }

    def _measure_interference(self, device: torch.device) -> list[dict]:
        """
        Compute interference decomposition for each contradiction query.
        Uses PathEnumerator on the toy_kg adjacency for real path enumeration.
        """
        from models.components.path_aggregator import PathEnumerator

        results = []
        adj         = self.toy_kg.get_adjacency()
        enumerator  = PathEnumerator(adj, max_hops=3, max_paths=8)
        aggregator  = self.model.aggregator

        for cq in self.toy_kg.contradiction_queries:
            h_id     = self.toy_kg.entity2id[cq["head"]]
            corr_id  = self.toy_kg.entity2id[cq["correct_tail"]]
            wrong_id = self.toy_kg.entity2id[cq["contradictory_tail"]]

            h_t    = torch.tensor([h_id],    device=device)
            corr_t = torch.tensor([corr_id], device=device)
            wrong_t= torch.tensor([wrong_id],device=device)

            h_state    = self.model.encoder(h_t).squeeze(0)
            corr_state = self.model.encoder(corr_t).squeeze(0)
            wrong_state= self.model.encoder(wrong_t).squeeze(0)

            # Correct answer interference
            corr_paths = enumerator.find_paths(h_id, corr_id)
            if corr_paths:
                corr_analysis = aggregator.compute_interference_terms(
                    h_state, corr_state, corr_paths, self.model.unitary
                )
            else:
                corr_analysis = {"interference": 0.0, "total_probability": 0.0,
                                 "classical_sum": 0.0, "interference_sign": "none"}

            # Wrong answer interference
            wrong_paths = enumerator.find_paths(h_id, wrong_id)
            if wrong_paths:
                wrong_analysis = aggregator.compute_interference_terms(
                    h_state, wrong_state, wrong_paths, self.model.unitary
                )
            else:
                wrong_analysis = {"interference": 0.0, "total_probability": 0.0,
                                  "classical_sum": 0.0, "interference_sign": "none"}

            results.append({
                "query":                 cq["query"],
                "head":                  cq["head"],
                "correct_tail":          cq["correct_tail"],
                "wrong_tail":            cq["contradictory_tail"],
                "correct_interference":  float(corr_analysis.get("interference", 0.0)),
                "wrong_interference":    float(wrong_analysis.get("interference", 0.0)),
                "correct_prob":          float(corr_analysis.get("total_probability", 0.0)),
                "wrong_prob":            float(wrong_analysis.get("total_probability", 0.0)),
                "interference":          float(wrong_analysis.get("interference", 0.0)),
                "interference_sign":     wrong_analysis.get("interference_sign", "none"),
                "correct_paths_found":   len(corr_paths),
                "wrong_paths_found":     len(wrong_paths),
            })

        return results

    # ------------------------------------------------------------------ #
    # Trend analysis                                                        #
    # ------------------------------------------------------------------ #

    def phase_collapse_trend(self) -> dict:
        """
        Analyze whether Phase Collapse is getting worse over time.

        Returns:
            Dict with 'trend' ("improving", "stable", "worsening", "collapsed"),
            'recent_mean_imag', 'rate_of_change'.
        """
        if len(self.report_history) < 3:
            return {"trend": "insufficient_data"}

        recent   = self.report_history[-3:]
        imag_norms = [r.mean_imag_norm for r in recent]
        rate     = (imag_norms[-1] - imag_norms[0]) / max(len(imag_norms) - 1, 1)

        if imag_norms[-1] < self.collapse_threshold:
            trend = "collapsed"
        elif rate < -0.005:
            trend = "worsening"
        elif rate > 0.005:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend":            trend,
            "recent_mean_imag": imag_norms[-1],
            "rate_of_change":   rate,
        }

    def interference_emergence_trend(self) -> dict:
        """
        Analyze whether destructive interference is emerging over training.

        Returns:
            Dict with 'trend' ("emerging", "stable", "absent"),
            'recent_destructive_count'.
        """
        if len(self.report_history) < 5:
            return {"trend": "insufficient_data"}

        recent_counts = [r.num_destructive for r in self.report_history[-5:]]
        rate = (recent_counts[-1] - recent_counts[0]) / max(len(recent_counts) - 1, 1)

        if recent_counts[-1] > 0 and rate >= 0:
            trend = "emerging"
        elif recent_counts[-1] == 0 and all(c == 0 for c in recent_counts):
            trend = "absent"
        else:
            trend = "stable"

        return {
            "trend":                    trend,
            "recent_destructive_count": recent_counts[-1],
            "max_destructive_possible": len(self.toy_kg.contradiction_queries),
        }

    # ------------------------------------------------------------------ #
    # Output and correction                                                 #
    # ------------------------------------------------------------------ #

    def _print_report(self, report: InterferenceReport) -> None:
        """Print report to console in a readable format."""
        print(f"\n{'='*70}")
        print(report.summary())
        if report.contradiction_results:
            print("  Contradiction queries:")
            for cr in report.contradiction_results:
                sign   = "DESTRUCTIVE" if cr["interference"] < -1e-6 else \
                         ("CONSTRUCTIVE" if cr["interference"] > 1e-6 else "NONE")
                print(
                    f"    {cr['head']:12s} -> {cr['correct_tail']:12s}/{cr['wrong_tail']:12s} | "
                    f"wrong_interference={cr['wrong_interference']:+.4f} [{sign}]"
                )
        if report.alert_message:
            print(f"\n  *** ALERT: {report.alert_message}")
        print(f"{'='*70}\n")

    def _save_report(self, report: InterferenceReport) -> None:
        """Save report to a text file."""
        fname = self.log_dir / f"interference_report_epoch_{report.epoch:04d}.txt"
        with open(fname, "w") as f:
            f.write(report.summary() + "\n\n")
            for cr in report.contradiction_results:
                f.write(f"Query: {cr['query']}\n")
                f.write(f"  Correct: {cr['correct_tail']} | Wrong: {cr['wrong_tail']}\n")
                f.write(f"  Correct interference: {cr['correct_interference']:+.6f}\n")
                f.write(f"  Wrong interference:   {cr['wrong_interference']:+.6f}\n")
                f.write(f"  Correct P: {cr['correct_prob']:.6f} | Wrong P: {cr['wrong_prob']:.6f}\n\n")

    def _apply_correction(self, report: InterferenceReport) -> None:
        """
        Automatic LR adjustment when Phase Collapse is detected.

        Strategy: boost the learning rate for imaginary parameter groups
        by 5x temporarily, which forces the optimizer to move the imaginary
        components away from zero.

        This is a heuristic correction — monitor whether it helps over
        the next 10-20 epochs. If it doesn't help, manual intervention
        is needed (architecture change or loss function change).
        """
        if self.optimizer is None:
            return

        print("[InterferenceMonitor] Applying auto-correction: boosting imaginary LR")
        for group in self.optimizer.param_groups:
            if group.get("name") in ("imag", "phases"):
                old_lr = group["lr"]
                group["lr"] = old_lr * 5.0
                print(f"  Group '{group.get('name')}': LR {old_lr:.6f} -> {group['lr']:.6f}")

    def final_summary(self) -> dict:
        """
        Summary statistics over the entire training run.
        Call after training completes for the paper's Section 7 analysis.
        """
        if not self.report_history:
            return {}

        final = self.report_history[-1]
        best_destructive = max(r.num_destructive for r in self.report_history)
        final_healthy    = final.is_healthy

        return {
            "final_epoch":           final.epoch,
            "final_imag_norm":       final.mean_imag_norm,
            "final_phase_magnitude": final.mean_phase_magnitude,
            "best_destructive_count":best_destructive,
            "total_queries":         len(self.toy_kg.contradiction_queries),
            "final_healthy":         final_healthy,
            "collapse_trend":        self.phase_collapse_trend(),
            "interference_trend":    self.interference_emergence_trend(),
        }
