"""
TrainerV2 — Upgraded Training Loop.

Addresses Challenge 1 (Training instability with complex gradients) by:
    1. Parameter groups with DIFFERENT learning rates per component:
       - Entity real embeddings:      lr_base (e.g., 0.001)
       - Entity imaginary embeddings: lr_imag (default: 3x lr_base)
                                      Higher LR fights Phase Collapse
       - Unitary phases:              lr_phase (default: 2x lr_base)
       - Aggregator path weights:     lr_agg (default: lr_base)
       - Relation bias:               lr_base

    2. Per-group gradient norm monitoring:
       Logs the gradient norm for each parameter group every N batches.
       Spikes in the imaginary group norm signal impending instability.

    3. InterferenceMonitor integration:
       Automatically runs the interference health check every N epochs
       and logs the result.

    4. Optional InterferenceAwareLoss:
       Drops in as a replacement for standard BinaryCrossEntropyLoss
       when use_interference_loss=True.

WHY DIFFERENT LRs:
    The imaginary components and phase angles have a different loss landscape
    than the real components. Because the BCE loss is smooth in prediction
    space (scores) but potentially flat in phase space (angles), the optimizer
    tends to leave phases near zero — Phase Collapse.

    Giving imaginary components a 2-5x higher learning rate compensates for
    this imbalance. The ratio 3x for imaginary, 2x for phases is a starting
    point — monitor interference_monitor.py reports to tune this.

RECOMMENDED HYPERPARAMETERS (FB15k-237, embed_dim=256):
    lr_base  = 0.001
    lr_imag  = 0.003   (3x)
    lr_phase = 0.002   (2x)
    grad_clip = 0.5    (tighter than V1's 1.0 for imaginary stability)
    warmup_epochs = 20 (longer warmup for imaginary components)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import RankingMetrics, MetricResults
from utils.logger import RichLogger
from utils.checkpoint import CheckpointManager
from .losses import build_loss
from .interference_loss import InterferenceAwareLoss
from .interference_monitor import InterferenceMonitor


class TrainerV2:
    """
    Upgraded trainer with parameter groups, gradient monitoring,
    and interference health checking.

    Args:
        model:              QuantumReasoner instance.
        train_loader:       Training DataLoader.
        val_loader:         Validation DataLoader.
        device:             torch.device.
        lr_base:            Base learning rate (real components, bias).
        lr_imag:            Learning rate for imaginary embedding components.
                            Default: 3 * lr_base.
        lr_phase:           Learning rate for unitary phase angles.
                            Default: 2 * lr_base.
        lr_agg:             Learning rate for aggregator path weights.
                            Default: lr_base.
        grad_clip:          Max gradient norm.
        epochs:             Total training epochs.
        use_interference_loss: If True, use InterferenceAwareLoss instead of BCE.
        interference_check_every: Run InterferenceMonitor check every N epochs.
        toy_kg:             ToyKG instance (required for InterferenceMonitor).
        contradiction_every: Run ContrastiveInterferenceLoss every N batches.
                             Default: 0 (disabled; enable when contradiction
                             annotations are available).
        checkpoint_dir:     Where to save checkpoints.
        run_name:           Experiment name.
        true_tails:         For filtered evaluation.

    Example:
        >>> trainer = TrainerV2(
        ...     model=quantum_model,
        ...     train_loader=train_dl,
        ...     val_loader=val_dl,
        ...     device=device,
        ...     lr_base=0.001,
        ...     lr_imag=0.003,   # 3x: fights Phase Collapse
        ...     lr_phase=0.002,  # 2x: keeps phases non-trivial
        ...     use_interference_loss=True,
        ...     interference_check_every=10,
        ...     toy_kg=kg,       # for monitoring
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model:              nn.Module,
        train_loader:       DataLoader,
        val_loader:         DataLoader,
        device:             torch.device,
        lr_base:            float = 1e-3,
        lr_imag:            float = 3e-3,
        lr_phase:           float = 2e-3,
        lr_agg:             float = 1e-3,
        weight_decay:       float = 1e-5,
        grad_clip:          float = 0.5,
        epochs:             int   = 200,
        warmup_epochs:      int   = 20,
        use_interference_loss: bool = False,
        interference_loss_kwargs: Optional[dict] = None,
        interference_check_every: int = 10,
        toy_kg              = None,
        contradiction_every: int  = 0,   # 0 = disabled
        checkpoint_dir:     str   = "outputs/checkpoints",
        run_name:           str   = "run_v2",
        true_tails:         Optional[dict] = None,
        use_wandb:          bool  = False,
        patience:           int   = 50,
        resume_from:        Optional[str] = None,
        chunked_evaluator   = None,   # ChunkedEvaluator instance (required for large KGs)
    ) -> None:

        self.model         = model.to(device)
        self.device        = device
        self.epochs        = epochs
        self.patience      = patience
        self.resume_from   = resume_from
        self.grad_clip     = grad_clip
        self.true_tails    = true_tails
        self.run_name      = run_name
        self.use_wandb     = use_wandb
        self.contradiction_every = contradiction_every

        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.chunked_evaluator  = chunked_evaluator   # None → use score_triple_vs_all

        # ── Loss function ──────────────────────────────────────────────
        if use_interference_loss:
            kw = interference_loss_kwargs or {}
            self.loss_fn = InterferenceAwareLoss(**kw)
            self.use_interference_loss = True
        else:
            self.loss_fn = build_loss("bce", label_smoothing=0.1)
            self.use_interference_loss = False

        # ── Parameter groups with different LRs ───────────────────────
        self.optimizer = self._build_param_group_optimizer(
            lr_base, lr_imag, lr_phase, lr_agg, weight_decay
        )

        # ── Scheduler ─────────────────────────────────────────────────
        # Warm restarts every ~100 epochs so the model can escape plateaus
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0      = max((epochs - warmup_epochs) // 4, 50),
            T_mult   = 1,
            eta_min  = lr_base * 0.01,
        )

        # ── Interference Monitor ───────────────────────────────────────
        self.monitor = None
        if toy_kg is not None:
            self.monitor = InterferenceMonitor(
                model                = self.model,
                toy_kg               = toy_kg,
                check_every_n_epochs = interference_check_every,
                auto_correct         = True,
                optimizer            = self.optimizer,
                verbose              = True,
                log_dir              = Path("outputs/logs") / run_name / "interference",
            )

        # ── Checkpoint Manager ─────────────────────────────────────────
        self.ckpt = CheckpointManager(
            checkpoint_dir   = Path(checkpoint_dir) / run_name,
            keep_last_n      = 3,
            metric_name      = "MRR",
            higher_is_better = True,
        )

        # ── Metrics ───────────────────────────────────────────────────
        self.val_metrics = RankingMetrics(filter_false_negatives=True)

        # ── Logger ────────────────────────────────────────────────────
        self.log = RichLogger(
            f"trainer_v2.{run_name}",
            log_file = Path("outputs/logs") / f"{run_name}_train.log",
        )

        # ── History ───────────────────────────────────────────────────
        self.history: dict[str, list] = {
            "train_loss": [], "val_mrr": [],
            "val_hits@1": [], "val_hits@10": [],
            "grad_norm_real": [], "grad_norm_imag": [], "grad_norm_phase": [],
        }

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #

    def train(self) -> MetricResults:
        """
        Run the full training loop with parameter-group monitoring.

        Returns:
            Best validation MetricResults.
        """
        self.log.print_banner(f"TrainerV2: {self.run_name}")
        self.log.info(
            f"Parameter groups: "
            f"real lr={self._get_lr('real'):.5f}, "
            f"imag lr={self._get_lr('imag'):.5f}, "
            f"phase lr={self._get_lr('phases'):.5f}"
        )

        best_results = None
        epochs_no_improve = 0
        start_epoch = 1

        # Resume from checkpoint
        if self.resume_from:
            ckpt = torch.load(self.resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            self.log.info(f"Resumed from epoch {ckpt['epoch']} ({self.resume_from})")

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()

            # Training step
            train_loss, grad_norms = self._train_epoch(epoch)

            # Validation step
            val_results = self._val_epoch(epoch)

            # Scheduler step
            self.scheduler.step()

            # Interference monitor check
            if self.monitor and epoch % self.monitor.check_every_n_epochs == 0:
                report = self.monitor.check(epoch)
                if not report.is_healthy:
                    self.log.warning(f"[Epoch {epoch}] {report.alert_message}")

            # Logging
            epoch_time = time.time() - t0
            self.log.info(
                f"Epoch {epoch:4d} | loss={train_loss:.4f} | "
                f"{val_results} | {epoch_time:.1f}s | "
                f"grad_imag={grad_norms.get('imag', 0):.3f}"
            )

            # Checkpointing
            self.ckpt.save(
                model     = self.model,
                optimizer = self.optimizer,
                epoch     = epoch,
                metrics   = val_results.to_dict(),
                scheduler = self.scheduler,
            )

            # Track best and early stopping
            if best_results is None or val_results.mrr > best_results.mrr:
                best_results = val_results
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.patience > 0 and epochs_no_improve >= self.patience:
                    self.log.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["val_mrr"].append(val_results.mrr)
            self.history["val_hits@1"].append(val_results.hits_at_1)
            self.history["val_hits@10"].append(val_results.hits_at_10)
            for k, v in grad_norms.items():
                self.history.setdefault(f"grad_norm_{k}", []).append(v)

        self.log.print_metrics(best_results.to_dict(), title="Best Validation Results")

        # Final interference summary
        if self.monitor:
            summary = self.monitor.final_summary()
            self.log.info(f"Final interference summary: {summary}")

        return best_results

    # ------------------------------------------------------------------ #
    # Training epoch                                                       #
    # ------------------------------------------------------------------ #

    def _train_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        """Run one training epoch. Returns (mean_loss, grad_norm_dict)."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        grad_norms: dict[str, list] = {}

        for batch in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:4d} [train]",
            leave=False,
        ):
            loss, batch_grad_norms = self._train_step(batch)
            total_loss += loss
            n_batches  += 1
            for k, v in batch_grad_norms.items():
                grad_norms.setdefault(k, []).append(v)

        mean_grad_norms = {k: sum(v)/len(v) for k, v in grad_norms.items() if v}
        return total_loss / max(n_batches, 1), mean_grad_norms

    def _train_step(self, batch: dict) -> tuple[float, dict[str, float]]:
        """Single training step with per-group gradient monitoring."""
        positive  = batch["positive"].to(self.device)   # (B, 3)
        negatives = batch["negatives"].to(self.device)  # (B, K, 3)
        B, K, _   = negatives.shape

        h_pos = positive[:, 0]
        r_pos = positive[:, 1]
        t_pos = positive[:, 2]

        h_neg = negatives[:, :, 0].reshape(B * K)
        r_neg = negatives[:, :, 1].reshape(B * K)
        t_neg = negatives[:, :, 2].reshape(B * K)

        self.optimizer.zero_grad()

        pos_scores = self.model.score_triple(h_pos, r_pos, t_pos)
        neg_scores = self.model.score_triple(h_neg, r_neg, t_neg).view(B, K)

        if self.use_interference_loss:
            loss = self.loss_fn.forward_main(pos_scores, neg_scores)
            # Regularization (every step)
            loss = loss + self.loss_fn.regularization(
                self.model.encoder, self.model.unitary
            )
        else:
            loss = self.loss_fn(pos_scores, neg_scores)

        loss.backward()

        # Measure gradient norms per group BEFORE clipping
        grad_norm_dict = self._measure_grad_norms()

        # Gradient clipping
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item(), grad_norm_dict

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def _val_epoch(self, epoch: int) -> MetricResults:
        self.model.eval()

        # Large KGs (e.g. WN18RR with 40k entities): use ChunkedEvaluator to avoid OOM
        if self.chunked_evaluator is not None:
            return self.chunked_evaluator.evaluate_loader(self.val_loader)

        self.val_metrics.reset()
        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc=f"Epoch {epoch:4d} [val]  ",
                leave=False,
            ):
                positive = batch["positive"].to(self.device)
                h, r, t  = positive[:, 0], positive[:, 1], positive[:, 2]
                scores   = self.model.score_triple_vs_all(h, r)
                self.val_metrics.update(
                    scores=scores, true_indices=t,
                    head_ids=h, relation_ids=r,
                    true_tails=self.true_tails,
                )

        return self.val_metrics.compute()

    # ------------------------------------------------------------------ #
    # Parameter groups                                                     #
    # ------------------------------------------------------------------ #

    def _build_param_group_optimizer(
        self,
        lr_base:      float,
        lr_imag:      float,
        lr_phase:     float,
        lr_agg:       float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """
        Build Adam optimizer with separate learning rates per component.

        Parameter groups:
            'real':    encoder.real_embeddings        -> lr_base
            'imag':    encoder.imag_embeddings        -> lr_imag   (higher)
            'phases':  unitary.phases (any variant)   -> lr_phase  (higher)
            'agg':     aggregator parameters          -> lr_agg
            'bias':    relation_bias                  -> lr_base
        """
        encoder  = self.model.encoder
        unitary  = self.model.unitary
        aggr     = self.model.aggregator

        param_groups = [
            {
                "name":   "real",
                "params": list(encoder.real_embeddings.parameters()),
                "lr":     lr_base,
                "weight_decay": weight_decay,
            },
            {
                "name":   "imag",
                "params": list(encoder.imag_embeddings.parameters()),
                "lr":     lr_imag,      # HIGHER LR: fights Phase Collapse
                "weight_decay": 0.0,   # no decay on imaginary (don't push toward 0)
            },
            {
                "name":   "phases",
                "params": [p for n, p in unitary.named_parameters()],
                "lr":     lr_phase,
                "weight_decay": 0.0,   # no decay on phase angles
            },
            {
                "name":   "agg",
                "params": list(aggr.parameters()),
                "lr":     lr_agg,
                "weight_decay": weight_decay,
            },
            {
                "name":   "bias",
                "params": [self.model.relation_bias],
                "lr":     lr_base,
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.Adam(param_groups)

    def _measure_grad_norms(self) -> dict[str, float]:
        """Measure gradient norms per parameter group."""
        norms = {}
        for group in self.optimizer.param_groups:
            name = group.get("name", "unknown")
            grads = [
                p.grad.detach().norm().item()
                for p in group["params"]
                if p.grad is not None
            ]
            norms[name] = sum(grads) / max(len(grads), 1)
        return norms

    def _get_lr(self, group_name: str) -> float:
        for group in self.optimizer.param_groups:
            if group.get("name") == group_name:
                return group["lr"]
        return 0.0

    def load_best_and_evaluate(self, test_loader: DataLoader) -> MetricResults:
        """Load best checkpoint and run test evaluation."""
        self.ckpt.load_best(self.model, device=self.device)
        test_metrics = RankingMetrics(filter_false_negatives=True)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test evaluation"):
                positive = batch["positive"].to(self.device)
                h, r, t  = positive[:, 0], positive[:, 1], positive[:, 2]
                scores   = self.model.score_triple_vs_all(h, r)
                test_metrics.update(
                    scores=scores, true_indices=t,
                    head_ids=h, relation_ids=r,
                    true_tails=self.true_tails,
                )
        results = test_metrics.compute()
        self.log.print_metrics(results.to_dict(), title="TEST SET Results")
        return results
