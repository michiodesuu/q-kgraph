"""
training/v7_training.py — V7 Trainer: 4-Phase Quantum Reasoning Engine

4-PHASE TRAINING PROTOCOL:
    Phase 1 (0 → phase1_end):
        Topological Stabilization.
        Only L_syndrome active. GNN decoder learns to detect contradiction patterns.
        Branch weights: tel=0, dec=0 (pure unitary for stability).
        Lindblad jump operators held near-zero via weight decay.

    Phase 2 (phase1_end → phase2_end):
        Coherent MPS Evolution.
        L_rank + L_BCE + L_phase active. Lindblad dissipation near-zero.
        MPS tensors learn to compress path amplitudes.
        Branch weights annealed: tel → tel_target, dec → 0.

    Phase 3 (phase2_end → phase3_end):
        Open System Thermalization.
        L_lindblad_reg + L_sasaki_ctx added.
        Decoherence rates unspooled. Sasaki conjunction enforced.
        Branch weights: dec → dec_target.

    Phase 4 (phase3_end → end):
        Formal Verification Loop.
        L_polarity + L_mps_entropy added.
        InterferencePolarityLoss directly targets Lemma V5.1.
        MPS entropy regularization controls bond dimension.

PARAMETER GROUPS (8 groups):
    real (1×):         Entity real embeddings
    imag (3×):         Entity imaginary embeddings (anti-Phase-Collapse)
    matrix_real (2×):  MatrixExpUnitary L_real
    matrix_imag (3×):  MatrixExpUnitary L_imag (anti-collapse)
    lindblad (0.1×):   Lindblad jump operators (slow start, physically bounded)
    mps (1×):          MPS core tensors
    sasaki (0.5×):     Concept projectors (near-zero LR, logic-constrained)
    decoder (2×):      GNN syndrome decoder (fast adaptation in Phase 1)
    misc (1×):         Bias, aggregator, branch weights
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from evaluation.metrics import RankingMetrics, MetricResults
    from utils.logger       import RichLogger
    from utils.checkpoint   import CheckpointManager
    _BASE_AVAILABLE = True
except ImportError:
    _BASE_AVAILABLE = False
    MetricResults = None

from training.v7_loss import V7Loss


class V7Trainer:
    """
    Full V7 trainer with 4-phase protocol and 8 parameter groups.

    Args:
        model:              V7QuantumReasoner.
        train_loader:       Training DataLoader.
        val_loader:         Validation DataLoader.
        toy_kg:             ToyKG for interference polarity loss and syndrome detection.
        path_cache:         Optional pre-computed PathCache.
        epochs:             Total epochs.
        lr_base:            Base learning rate.
        lr_imag_mult:       LR multiplier for imaginary embeddings (anti-collapse).
        lr_matrix_mult:     LR multiplier for matrix imag params.
        lr_decoder_mult:    LR multiplier for GNN decoder (fast in Phase 1).
        checkpoint_dir:     Path to save checkpoints.
        device:             Torch device.
        phase1_end:         Epoch where Phase 1 ends. Default 15.
        phase2_end:         Epoch where Phase 2 ends. Default 40.
        phase3_end:         Epoch where Phase 3 ends. Default 80.
        check_every:        How often to run guarantee verification. Default 10.
        tel_target:         Final teleportation branch weight (annealed to from 0).
        dec_target:         Final decoherence branch weight.
    """

    def __init__(
        self,
        model,
        train_loader:      DataLoader,
        val_loader:        DataLoader,
        toy_kg             = None,
        path_cache         = None,
        epochs:            int   = 150,
        lr_base:           float = 1e-3,
        lr_imag_mult:      float = 3.0,
        lr_matrix_mult:    float = 2.0,
        lr_decoder_mult:   float = 2.0,
        checkpoint_dir:    str   = "outputs/checkpoints/v7",
        device:            str   = "cpu",
        phase1_end:        int   = 15,
        phase2_end:        int   = 40,
        phase3_end:        int   = 80,
        check_every:       int   = 10,
        tel_target:        float = 0.15,
        dec_target:        float = 0.10,
        polarity_weight:   float = 5.0,
        polarity_margin:   float = 0.1,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.toy_kg       = toy_kg
        self.path_cache   = path_cache
        self.epochs       = epochs
        self.device       = torch.device(device)
        self.phase1_end   = phase1_end
        self.phase2_end   = phase2_end
        self.phase3_end   = phase3_end
        self.check_every  = check_every
        self.tel_target   = tel_target
        self.dec_target   = dec_target
        self._history: list = []

        self.model.to(self.device)

        self.loss_fn = V7Loss(
            phase1_end       = phase1_end,
            phase2_end       = phase2_end,
            phase3_end       = phase3_end,
            polarity_weight  = polarity_weight,
            polarity_margin  = polarity_margin,
        )

        self.optimizer = self._build_optimizer(
            lr_base, lr_imag_mult, lr_matrix_mult, lr_decoder_mult
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr_base * 0.01
        )

        if _BASE_AVAILABLE:
            self.ckpt   = CheckpointManager(checkpoint_dir)
            self.logger = RichLogger(name="V7")
        else:
            self.ckpt   = None
            self.logger = None

    # ── Parameter groups ───────────────────────────────────────────────────
    def _build_optimizer(
        self,
        lr_base:         float,
        lr_imag_mult:    float,
        lr_matrix_mult:  float,
        lr_decoder_mult: float,
    ) -> torch.optim.Optimizer:
        """
        Build Adam optimizer with 8 parameter groups.

        Group       LR mult     Weight decay    Purpose
        ───────────────────────────────────────────────────────────────────
        real        1×          1e-4            Entity real embeddings
        imag        3×          0               Anti-Phase-Collapse
        mat_real    2×          1e-5            MatrixExpUnitary generator (real)
        mat_imag    3×          0               MatrixExpUnitary generator (imag)
        lindblad    0.1×        1e-3            Jump operators (physically bounded)
        mps         1×          1e-5            MPS core tensors
        sasaki      0.5×        0               Concept projectors
        decoder     2×          1e-4            GNN syndrome decoder
        misc        1×          1e-4            Bias, aggregator, branch weights
        """
        groups = [
            {"params": self.model.real_params(),
             "lr": lr_base, "weight_decay": 1e-4, "name": "real"},
            {"params": self.model.imag_params(),
             "lr": lr_base * lr_imag_mult, "weight_decay": 0.0, "name": "imag"},
            {"params": self.model.matrix_real_params(),
             "lr": lr_base * lr_matrix_mult, "weight_decay": 1e-5, "name": "mat_real"},
            {"params": self.model.matrix_imag_params(),
             "lr": lr_base * lr_matrix_mult * lr_imag_mult, "weight_decay": 0.0, "name": "mat_imag"},
        ]

        # Optional groups — only add if model has those components
        lind_p = self.model.lindblad_params()
        if lind_p:
            groups.append({"params": lind_p,
                           "lr": lr_base * 0.1, "weight_decay": 1e-3, "name": "lindblad"})

        mps_p = self.model.mps_params()
        if mps_p:
            groups.append({"params": mps_p,
                           "lr": lr_base, "weight_decay": 1e-5, "name": "mps"})

        sasaki_p = self.model.sasaki_params()
        if sasaki_p:
            groups.append({"params": sasaki_p,
                           "lr": lr_base * 0.5, "weight_decay": 0.0, "name": "sasaki"})

        decoder_p = self.model.decoder_params()
        if decoder_p:
            groups.append({"params": decoder_p,
                           "lr": lr_base * lr_decoder_mult, "weight_decay": 1e-4, "name": "decoder"})

        misc_p = self.model.misc_params()
        valid_misc = [p for p in misc_p if p.requires_grad]
        if valid_misc:
            groups.append({"params": valid_misc,
                           "lr": lr_base, "weight_decay": 1e-4, "name": "misc"})

        # Filter out empty groups
        groups = [g for g in groups if len(g["params"]) > 0]
        return torch.optim.Adam(groups)

    # ── Branch weight annealing ────────────────────────────────────────────
    def _anneal_branch_weights(self, epoch: int) -> None:
        """
        Linearly anneal teleportation and decoherence branch weights.
        Phase 1: both = 0 (pure unitary, stability)
        Phase 2: tel ramps to tel_target
        Phase 3: dec ramps to dec_target
        """
        if epoch < self.phase1_end:
            target_tel = 0.0
            target_dec = 0.0
        elif epoch < self.phase2_end:
            frac = (epoch - self.phase1_end) / max(1, self.phase2_end - self.phase1_end)
            target_tel = frac * self.tel_target
            target_dec = 0.0
        else:
            target_tel = self.tel_target
            frac = min(1.0, (epoch - self.phase2_end) / max(1, self.phase3_end - self.phase2_end))
            target_dec = frac * self.dec_target

        # Set raw logit values so sigmoid(logit) ≈ target
        # sigmoid(x) = target  →  x = logit(target) = log(target/(1-target))
        def to_logit(w):
            w = max(min(w, 0.95), 0.001)
            import math
            return math.log(w / (1.0 - w))

        with torch.no_grad():
            self.model._tel_weight.fill_(to_logit(target_tel))
            self.model._dec_weight.fill_(to_logit(target_dec))

    # ── Training loop ──────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        totals = {k: 0.0 for k in
            ["total","l_rank","l_bce","l_phase","l_syndrome",
             "l_lind","l_sasaki","l_polarity","l_mps"]}
        n_batches = 0

        for batch in self.train_loader:
            h   = batch["head"].to(self.device)
            r   = batch["relation"].to(self.device)
            t   = batch["tail"].to(self.device)
            neg = batch["negatives"].to(self.device)   # (B, K)

            self.optimizer.zero_grad()

            pos_scores = self.model.score_triple(h, r, t)
            B, K       = neg.shape
            neg_flat   = neg.view(-1)
            h_rep      = h.unsqueeze(1).expand(-1, K, -1).reshape(-1) if h.dim() > 1 \
                         else h.unsqueeze(1).expand(-1, K).reshape(-1)
            r_rep      = r.unsqueeze(1).expand(-1, K).reshape(-1)
            neg_scores = self.model.score_triple(h_rep, r_rep, neg_flat).view(B, K)

            loss_dict = self.loss_fn(
                model       = self.model,
                pos_scores  = pos_scores,
                neg_scores  = neg_scores,
                epoch       = epoch,
                toy_kg      = self.toy_kg,
                device      = self.device,
                path_cache  = self.path_cache,
                head_ids    = h,
            )

            total_loss = loss_dict["total"]
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            for k in totals:
                v = loss_dict.get(k, 0.0)
                totals[k] += float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            n_batches += 1

        if n_batches > 0:
            totals = {k: v / n_batches for k, v in totals.items()}
        return totals

    def _evaluate(self, loader: DataLoader):
        if not _BASE_AVAILABLE:
            return None
        self.model.eval()
        metrics = RankingMetrics(hits_k=[1, 3, 10])
        with torch.no_grad():
            for batch in loader:
                h = batch["head"].to(self.device)
                r = batch["relation"].to(self.device)
                t = batch["tail"].to(self.device)
                scores = self.model.score_triple_vs_all(h, r)   # (B, E)
                metrics.update(scores, t)
        return metrics.compute()

    def _log_epoch(self, epoch: int, train: Dict, val, elapsed: float) -> None:
        phase = self.loss_fn.get_phase(epoch)
        active = "+".join(self.loss_fn.get_active_components(epoch))
        mrr_str = f"MRR={val.mrr:.4f}" if val is not None and hasattr(val, "mrr") else "MRR=--"
        pol_str = f"pol={train.get('l_polarity', 0.0):.4f}"
        synd_str = f"synd={train.get('l_syndrome', 0.0):.4f}"
        print(
            f"[V7 P{phase}] E {epoch:4d} | loss={train['total']:.4f} "
            f"(bce={train.get('l_bce',0.0):.3f} rank={train.get('l_rank',0.0):.3f} "
            f"{pol_str} {synd_str}) | {mrr_str} | "
            f"tel={self.model.tel_weight:.2f} dec={self.model.dec_weight:.2f} | "
            f"active=[{active}] | {elapsed:.1f}s"
        )

    # ── Health check ───────────────────────────────────────────────────────
    def _check_interference(self, epoch: int) -> None:
        """Print interference status for contradiction queries (from toy_kg)."""
        if self.toy_kg is None or not hasattr(self.toy_kg, "contradiction_queries"):
            return
        from models.components.path_aggregator import PathEnumerator
        adj = self.toy_kg.get_adjacency()
        enumerator = PathEnumerator(adj, max_hops=2, max_paths=8)
        self.model.eval()
        print(f"\n[V7 Guarantee Check @ Epoch {epoch}]")
        with torch.no_grad():
            for cq in self.toy_kg.contradiction_queries:
                h_id     = self.toy_kg.entity2id[cq["head"]]
                wrong_id = self.toy_kg.entity2id[cq["contradictory_tail"]]
                corr_id  = self.toy_kg.entity2id[cq["correct_tail"]]
                h_str    = cq["head"]
                r_str    = cq.get("relation", "?")

                wrong_paths = enumerator.find_paths(h_id, wrong_id)
                corr_paths  = enumerator.find_paths(h_id, corr_id)

                if not wrong_paths:
                    continue

                h = self.model.encoder(torch.tensor([h_id], device=self.device)).squeeze(0)
                w = self.model.encoder(torch.tensor([wrong_id], device=self.device)).squeeze(0)
                c_t = self.model.encoder(torch.tensor([corr_id], device=self.device)).squeeze(0)

                amps_w = []
                for path in wrong_paths[:8]:
                    state = h.clone()
                    for rel_id, _ in path:
                        r_t = torch.tensor([rel_id], device=self.device)
                        state = self.model.unitary.apply(state.unsqueeze(0), r_t).squeeze(0)
                    amp = (w.conj() * state).sum()
                    amps_w.append(amp)

                if amps_w:
                    total = sum(amps_w)
                    int_wrong = total.abs().pow(2) - sum(a.abs().pow(2) for a in amps_w)
                    tag = "DESTRUCTIVE" if int_wrong.real < 0 else "CONSTRUCTIVE"
                    # Pairwise cross-term check (Lemma V5.1)
                    destructive_pairs = 0
                    total_pairs = 0
                    for i in range(len(amps_w)):
                        for j in range(i+1, len(amps_w)):
                            cross = 2.0 * (amps_w[i] * amps_w[j].conj()).real
                            total_pairs += 1
                            if cross.item() < 0:
                                destructive_pairs += 1
                    phi_vals = []
                    for i in range(len(amps_w)):
                        for j in range(i+1, len(amps_w)):
                            phi = torch.angle(amps_w[i] * amps_w[j].conj()).item()
                            phi_vals.append(abs(phi) * 180.0 / 3.14159)
                    phi_mean = sum(phi_vals)/len(phi_vals) if phi_vals else 0.0
                    lemma = "✓" if destructive_pairs == total_pairs else "✗ NOT YET"
                    print(
                        f"  Lemma V5.1 [{lemma}] {h_str} {r_str} ?: "
                        f"φ_mean={phi_mean:.1f}°  destructive={destructive_pairs}/{total_pairs}  "
                        f"int={int_wrong.real.item():.4f} [{tag}]"
                    )
        print()

    # ── Main train loop ────────────────────────────────────────────────────
    def train(self) -> Dict:
        """
        Run full 4-phase V7 training.

        Returns dict with best_mrr, training_history.
        """
        best_mrr = 0.0
        print(f"\n{'='*70}")
        print(f"[V7 TRAINING] {self.epochs} epochs | "
              f"Phases: P1=0-{self.phase1_end} P2={self.phase1_end}-{self.phase2_end} "
              f"P3={self.phase2_end}-{self.phase3_end} P4={self.phase3_end}-end")
        print(f"  Components: lindblad={self.model._has_lindblad} "
              f"mps={self.model._has_mps} sasaki={self.model._has_sasaki} "
              f"ecc={self.model._has_ecc}")
        print(f"{'='*70}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Anneal branch weights according to phase
            self._anneal_branch_weights(epoch)

            # Train one epoch
            train_stats = self._train_epoch(epoch)

            # Validate
            val_result = None
            if epoch % 5 == 0 or epoch == self.epochs:
                val_result = self._evaluate(self.val_loader)
                if val_result is not None and hasattr(val_result, "mrr"):
                    if val_result.mrr > best_mrr:
                        best_mrr = val_result.mrr
                    if self.ckpt is not None:
                        self.ckpt.save(self.model, self.optimizer, epoch,
                                       {"mrr": val_result.mrr})

            # Guarantee checks
            if epoch % self.check_every == 0:
                self._check_interference(epoch)

            elapsed = time.time() - t0
            self._log_epoch(epoch, train_stats, val_result, elapsed)

            self._history.append({
                "epoch": epoch,
                "train": train_stats,
                "val_mrr": val_result.mrr if val_result is not None and hasattr(val_result, "mrr") else None,
                "phase": self.loss_fn.get_phase(epoch),
            })

            self.scheduler.step()

        print(f"\n{'='*70}")
        print(f"[V7 COMPLETE] Best val MRR: {best_mrr:.4f}")
        print(f"{'='*70}\n")
        return {"best_mrr": best_mrr, "history": self._history}

    def get_training_history(self) -> list:
        return self._history
