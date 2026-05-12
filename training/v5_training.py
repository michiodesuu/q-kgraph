"""
training/v5_trainer.py — V5 Trainer: Full Theoretical Guarantee Training [V5]

KEY DIFFERENCES FROM V4TRAINER:
    1. Uses MatrixExpUnitary as default (closes RotatE distinction)
    2. Uses V5Loss with InterferencePolarityLoss (closes Phase Collapse guarantee gap)
    3. Three-phase training schedule:
       Phase 1 (epochs 1 to warmup):         BCE + spread + matrix_reg only
                                               Let entity states and MatrixExpUnitary stabilize
       Phase 2 (epochs warmup to full_loss):  Add V2 phase + contrastive losses
                                               Standard interference shaping
       Phase 3 (epochs full_loss to end):     Add InterferencePolarityLoss
                                               FORCE interference sign — the formal guarantee
    4. Verifies Lemma V5.1 and Theorem V5.2 every check_every epochs
    5. Supports NELL real-world noisy data alongside toy KG

PARAMETER GROUPS (6 groups — extending V2's 5):
    real (1×)         — Entity real embeddings
    imag (3×)         — Entity imaginary embeddings (anti-Phase-Collapse)
    matrix_real (2×)  — MatrixExpUnitary L_real (Hermitian real part)
    matrix_imag (2×)  — MatrixExpUnitary L_imag (Hermitian imaginary part, anti-collapse)
    aggregator (1×)   — Path weight aggregator
    bias (1×)         — Relation bias terms
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from evaluation.metrics import RankingMetrics, MetricResults
    from utils.logger       import RichLogger
    from utils.checkpoint   import CheckpointManager
    from training.interference_monitor import InterferenceMonitor
    _BASE_AVAILABLE = True
except ImportError:
    _BASE_AVAILABLE = False
    MetricResults = None

from .v5_loss import V5Loss


class V5Trainer:
    """
    Full V5 trainer with formal guarantee verification.

    Args:
        model:             QuantumReasoner or QuaternionReasoner.
        train_loader:      Training DataLoader.
        val_loader:        Validation DataLoader.
        device:            Torch device.
        toy_kg:            ToyKG with contradiction_queries (required for V5 loss).
        path_cache:        PathCache for faster BFS lookup.
        lr_base:           Base learning rate.
        lr_imag_mult:      LR multiplier for imaginary components. Default 3.0.
        lr_matrix_mult:    LR multiplier for MatrixExpUnitary. Default 2.0.
        warmup_epochs:     Epochs before adding phase/contrast losses.
        full_loss_epoch:   Epochs before InterferencePolarityLoss activates.
        epochs:            Total training epochs.
        grad_clip:         Gradient clipping norm.
        check_every:       How often to run guarantee verification.
        loss_kwargs:       Extra kwargs passed to V5Loss.
        checkpoint_dir:    Directory for checkpoints.
        run_name:          Experiment name.
        true_tails:        Dict for filtered evaluation.
        log_every_n:       Print every N epochs.
        nell_dataset:      Optional NELL dataset for real-world noise experiments.
    """

    def __init__(
        self,
        model,
        train_loader:      DataLoader,
        val_loader:        DataLoader,
        device:            torch.device,
        toy_kg=None,
        path_cache=None,
        lr_base:           float = 0.005,
        lr_imag_mult:      float = 3.0,
        lr_matrix_mult:    float = 2.0,
        warmup_epochs:     int   = 20,
        full_loss_epoch:   int   = 40,
        epochs:            int   = 150,
        grad_clip:         float = 0.5,
        loss_kwargs:       Optional[dict] = None,
        checkpoint_dir:    str   = "outputs/checkpoints",
        run_name:          str   = "v5_run",
        true_tails:        Optional[dict] = None,
        log_every_n:       int   = 10,
        nell_dataset=None,
    ) -> None:
        self.model           = model.to(device)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.device          = device
        self.toy_kg          = toy_kg
        self.path_cache      = path_cache
        self.warmup_epochs   = warmup_epochs
        self.full_loss_epoch = full_loss_epoch
        self.epochs          = epochs
        self.grad_clip       = grad_clip
        self.run_name        = run_name
        self.true_tails      = true_tails or {}
        self.log_every_n     = log_every_n
        self.nell_dataset    = nell_dataset

        # Build V5 loss
        lk = loss_kwargs or {}
        lk.setdefault("warmup_epochs",   warmup_epochs)
        lk.setdefault("full_loss_epoch", full_loss_epoch)
        self.loss_fn = V5Loss(**lk)

        # Checkpoint + logging
        ckpt_path = Path(checkpoint_dir) / run_name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if _BASE_AVAILABLE:
            self.ckpt = CheckpointManager(ckpt_path)
        else:
            self.ckpt = None

        # Optimizer
        self.optimizer = self._build_optimizer(lr_base, lr_imag_mult, lr_matrix_mult)

        # LR scheduler: cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr_base * 0.01
        )

        # Interference monitor (V2 component)
        if _BASE_AVAILABLE and toy_kg is not None:
            self.monitor = InterferenceMonitor(
                model            = model,
                toy_kg           = toy_kg,
                check_every_n_epochs = max(10, epochs // 15),
                collapse_threshold    = 0.02,
                verbose          = True,
            )
        else:
            self.monitor = None

        # State
        self.best_mrr    = 0.0
        self.best_result = None
        self._history    = []

        # Guarantee verification results
        self._guarantee_history = []

    def _build_optimizer(
        self,
        lr_base:        float,
        lr_imag_mult:   float,
        lr_matrix_mult: float,
    ) -> torch.optim.Adam:
        """
        Build Adam optimizer with 6 parameter groups.

        Detects whether model uses MatrixExpUnitary or DiagonalUnitary
        and sets groups accordingly.
        """
        model = self.model
        groups = []

        # ── Entity embedding groups ──────────────────────────────────────────
        try:
            groups.append({
                "name":         "real",
                "params":       list(model.encoder.real_embeddings.parameters()),
                "lr":           lr_base,
                "weight_decay": 1e-5,
            })
            groups.append({
                "name":         "imag",
                "params":       list(model.encoder.imag_embeddings.parameters()),
                "lr":           lr_base * lr_imag_mult,
                "weight_decay": 0.0,
            })
        except AttributeError:
            # V4 QuaternionReasoner has emb_r, emb_i, emb_j, emb_k
            try:
                groups.append({
                    "name":   "emb_r",
                    "params": list(model.encoder.emb_r.parameters()),
                    "lr":     lr_base, "weight_decay": 1e-5,
                })
                groups.append({
                    "name":   "emb_i",
                    "params": list(model.encoder.emb_i.parameters()),
                    "lr":     lr_base * lr_imag_mult, "weight_decay": 0.0,
                })
                groups.append({
                    "name":   "emb_jk",
                    "params": (list(model.encoder.emb_j.parameters()) +
                               list(model.encoder.emb_k.parameters())),
                    "lr":     lr_base * 2.0, "weight_decay": 0.0,
                })
            except AttributeError:
                groups.append({
                    "name":   "encoder",
                    "params": list(model.encoder.parameters()),
                    "lr":     lr_base, "weight_decay": 1e-5,
                })

        # ── Unitary groups (MatrixExpUnitary or DiagonalUnitary) ──────────────
        unitary = model.unitary
        if hasattr(unitary, "L_real") and hasattr(unitary, "L_imag"):
            # MatrixExpUnitary
            groups.append({
                "name":         "matrix_real",
                "params":       [unitary.L_real],
                "lr":           lr_base * lr_matrix_mult,
                "weight_decay": 1e-5,
            })
            groups.append({
                "name":         "matrix_imag",
                "params":       [unitary.L_imag],
                "lr":           lr_base * lr_matrix_mult * 1.5,
                "weight_decay": 0.0,   # NO weight decay on imaginary Hermitian part
            })
        elif hasattr(unitary, "phases"):
            # DiagonalUnitary
            groups.append({
                "name":         "phases",
                "params":       [unitary.phases],
                "lr":           lr_base * 2.0,
                "weight_decay": 0.0,
            })
        else:
            # Generic fallback
            groups.append({
                "name":         "unitary",
                "params":       list(unitary.parameters()),
                "lr":           lr_base * lr_matrix_mult,
                "weight_decay": 0.0,
            })

        # ── Aggregator + bias ────────────────────────────────────────────────
        if hasattr(model, "aggregator") and model.aggregator is not None:
            groups.append({
                "name":         "aggregator",
                "params":       list(model.aggregator.parameters()),
                "lr":           lr_base,
                "weight_decay": 1e-5,
            })
        if hasattr(model, "relation_bias"):
            groups.append({
                "name":         "bias",
                "params":       [model.relation_bias],
                "lr":           lr_base,
                "weight_decay": 0.0,
            })

        optimizer = torch.optim.Adam(groups)
        group_names = " | ".join(f"{g['name']}@{g['lr']:.5f}" for g in groups)
        print(f"[V5Trainer] Groups: {group_names}")
        return optimizer

    def _train_epoch(self, epoch: int) -> dict:
        """One training epoch. Returns loss breakdown."""
        self.model.train()
        totals  = {"total": 0.0, "l_bce": 0.0, "l_phase": 0.0,
                   "l_contrast": 0.0, "l_polarity": 0.0, "l_matrix": 0.0}
        n = 0

        for batch in self.train_loader:
            h   = batch["positive"][:, 0].to(self.device)
            r   = batch["positive"][:, 1].to(self.device)
            t   = batch["positive"][:, 2].to(self.device)
            neg = batch["negatives"].to(self.device)
            B, K, _ = neg.shape

            neg_h = neg[:, :, 0].reshape(-1)
            neg_r = neg[:, :, 1].reshape(-1)
            neg_t = neg[:, :, 2].reshape(-1)

            pos_scores = self.model.score_triple(h, r, t)
            neg_scores = self.model.score_triple(neg_h, neg_r, neg_t).view(B, K)

            loss_dict = self.loss_fn(
                model      = self.model,
                pos_scores = pos_scores,
                neg_scores = neg_scores,
                epoch      = epoch,
                toy_kg     = self.toy_kg,
                device     = self.device,
                path_cache = self.path_cache,
            )
            loss = loss_dict["total"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            for k, v in totals.items():
                val = loss_dict.get(k, 0.0)
                totals[k] += val.item() if isinstance(val, torch.Tensor) else val
            n += 1

        return {k: v / max(n, 1) for k, v in totals.items()}

    def _evaluate(self, loader: DataLoader) -> Optional["MetricResults"]:
        """Filtered MRR evaluation."""
        if not _BASE_AVAILABLE:
            return None
        metrics = RankingMetrics(filter_false_negatives=True)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                h = batch["positive"][:, 0].to(self.device)
                r = batch["positive"][:, 1].to(self.device)
                t = batch["positive"][:, 2].to(self.device)
                scores = self.model.score_triple_vs_all(h, r)
                metrics.update(scores, t, h, r, self.true_tails)
        return metrics.compute()

    def _verify_guarantees(self, epoch: int) -> None:
        """Run V5 theoretical guarantee verification."""
        try:
            from theory.inteference_guarantee import verify_lemma_v51, verify_theorem_v52
            if self.toy_kg is None:
                return

            print(f"\n[V5 Guarantee Check @ Epoch {epoch}]")
            l51 = verify_lemma_v51(self.model, self.toy_kg, self.device)
            for r in l51:
                print(f"  Lemma V5.1: {r.summary()}")

            t52 = verify_theorem_v52(self.model, self.toy_kg, self.device)
            for r in t52:
                print(f"  Theorem V5.2: {r.summary()}")

            self._guarantee_history.append({"epoch": epoch, "l51": l51, "t52": t52})
        except Exception as e:
            print(f"  [Guarantee check failed: {e}]")

    def _log_epoch(self, epoch: int, train: dict, val, elapsed: float) -> None:
        """Print epoch summary."""
        mrr = val.mrr if val else 0.0
        h1  = val.hits_at_1 if val else 0.0
        h10 = val.hits_at_10 if val else 0.0

        # Imaginary norm check (Phase Collapse detector)
        try:
            inorm = float(self.model.encoder.imag_embeddings.weight.norm(dim=-1).mean())
        except AttributeError:
            try:
                inorm = float(self.model.encoder.emb_i.weight.norm(dim=-1).mean())
            except AttributeError:
                inorm = -1.0

        active = self.loss_fn.get_active_components(epoch)
        active_str = "+".join(c.replace("l_", "") for c in active)

        print(
            f"[{self.run_name}] E{epoch:4d} | "
            f"loss={train['total']:.4f} "
            f"(bce={train.get('l_bce',0):.3f} "
            f"pol={train.get('l_polarity',0):.3f} "
            f"mat={train.get('l_matrix',0):.4f}) | "
            f"MRR={mrr:.4f} H@1={h1:.4f} H@10={h10:.4f} | "
            f"imag={inorm:.3f} | active=[{active_str}] | {elapsed:.1f}s"
        )
        self._history.append({
            "epoch": epoch, "loss": train["total"], "l_polarity": train.get("l_polarity", 0),
            "mrr": mrr, "h1": h1, "h10": h10, "imag_norm": inorm,
        })

    def train(self):
        """
        Full V5 training with three-phase schedule.

        Returns:
            Best validation MetricResults.
        """
        print(f"\n{'='*70}")
        print(f"V5 Training: {self.run_name}")
        print(f"  Epochs: {self.epochs}  |  Warmup: {self.warmup_epochs}  |  FullLoss: {self.full_loss_epoch}")
        print(f"  Phase 1 (ep 1-{self.warmup_epochs}):         BCE + matrix_reg + phase_spread")
        print(f"  Phase 2 (ep {self.warmup_epochs}-{self.full_loss_epoch}): + V2 PhaseSep + Contrastive")
        print(f"  Phase 3 (ep {self.full_loss_epoch}-{self.epochs}): + InterferencePolarityLoss (V5)")
        print(f"{'='*70}\n")

        guarantee_check_every = max(20, self.epochs // 7)

        for epoch in range(1, self.epochs + 1):
            t0          = time.time()
            train_stats = self._train_epoch(epoch)
            val_result  = self._evaluate(self.val_loader)
            elapsed     = time.time() - t0

            self.scheduler.step()

            # Save checkpoint
            if val_result and val_result.mrr > self.best_mrr:
                self.best_mrr    = val_result.mrr
                self.best_result = val_result
                if self.ckpt:
                    self.ckpt.save(self.model, self.optimizer, epoch, {"mrr": val_result.mrr})

            # Log
            if epoch % self.log_every_n == 0 or epoch == 1:
                self._log_epoch(epoch, train_stats, val_result, elapsed)

            # Interference monitor
            if self.monitor and epoch % max(10, self.epochs // 15) == 0:
                self.monitor.check(epoch)

            # V5 guarantee verification
            if epoch % guarantee_check_every == 0 or epoch == self.epochs:
                self._verify_guarantees(epoch)

        print(f"\nV5 Training complete. Best val MRR: {self.best_mrr:.4f}")
        return self.best_result

    def get_training_history(self) -> list:
        return self._history

    def get_guarantee_history(self) -> list:
        return self._guarantee_history
