"""
training/v4_trainer.py — V4 Trainer: Quaternion + Lattice + Routing [V4]

PARAMETER GROUPS (6 groups, extending V3's TrainerV2's 5):

    V3 had: real(1×) | imag(3×) | phases(2×) | agg(1×) | bias(1×)
    V4 has: real(1×) | i_comp(3×) | j_comp(2×) | k_comp(2×) | unitary(2×) | lattice(0.5×) | router(0.1×)

    Why 6 groups?
    - i_comp at 3×: same anti-collapse mechanism as V3's lr_imag — preserves quaternion rotation
    - j_comp, k_comp at 2×: new quaternion dimensions need moderate push
    - unitary at 2×: relation quaternions must rotate (not stay near identity)
    - lattice at 0.5×: logic lattice should converge SLOWLY toward binary
                        (too fast → premature collapse to degenerate logic states)
    - router at 0.1×: routing thresholds should adapt slowly (avoid oscillation)

LATTICE TRAINING PROTOCOL (from QIQE-KGC):
    Phase 1 — Quaternion pre-training (epochs 1-warmup):
        Train only quaternion components, freeze lattice.
        Reason: let entity embeddings stabilize before imposing global constraints.
    Phase 2 — Joint training (epochs warmup+1 to end):
        Train all components jointly with full V4 loss.
        Lattice entropy regularization gradually shifts from maximize to minimize.

MONITORING:
    Extends V3's InterferenceMonitor with:
    - Logic score monitoring: mean lattice consistency per epoch
    - Routing statistics: %classical vs %quantum per epoch
    - Quaternion phase distribution: mean rotation angle per relation type
    - Rank stability tracking: ΔRank between epochs (from QIQE-KGC MR divergence fix)
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from evaluation.metrics import RankingMetrics, MetricResults
    from utils.logger import RichLogger
    from utils.checkpoint import CheckpointManager
    _KG_AVAILABLE = True
except ImportError:
    _KG_AVAILABLE = False
    MetricResults = None

from .v4_loss import V4Loss


class V4Trainer:
    """
    Trainer for QuaternionReasoner V4.

    Extends V3's TrainerV2 protocol with:
    1. 6-group parameter groups (quaternion r/i/j/k, lattice, router separate)
    2. Two-phase training: quaternion warmup then joint lattice training
    3. Enhanced monitoring: logic scores, routing stats, rank stability
    4. Rank stability tracking to quantify the MR vs MRR divergence fix

    Args:
        model:             QuaternionReasoner V4 instance.
        train_loader:      Training DataLoader.
        val_loader:        Validation DataLoader.
        device:            Torch device.
        lr_base:           Base learning rate for real embeddings.
        lr_i_mult:         LR multiplier for quaternion i-component (default 3.0).
        lr_j_mult:         LR multiplier for quaternion j-component (default 2.0).
        lr_k_mult:         LR multiplier for quaternion k-component (default 2.0).
        lr_unitary_mult:   LR multiplier for relation unitary (default 2.0).
        lr_lattice_mult:   LR multiplier for logic lattice (default 0.5).
        lr_router_mult:    LR multiplier for router thresholds (default 0.1).
        lattice_warmup:    Epochs of quaternion-only training before lattice activates.
        epochs:            Total training epochs.
        grad_clip:         Gradient clipping norm.
        loss_kwargs:       Dict passed to V4Loss constructor.
        checkpoint_dir:    Directory for best.pt checkpoints.
        run_name:          Experiment name.
        true_tails:        Dict for filtered evaluation.
        toy_kg:            ToyKG for InterferenceMonitor (optional).
    """

    def __init__(
        self,
        model,
        train_loader:      DataLoader,
        val_loader:        DataLoader,
        device:            torch.device,
        lr_base:           float = 0.005,
        lr_i_mult:         float = 3.0,
        lr_j_mult:         float = 2.0,
        lr_k_mult:         float = 2.0,
        lr_unitary_mult:   float = 2.0,
        lr_lattice_mult:   float = 0.5,
        lr_router_mult:    float = 0.1,
        lattice_warmup:    int   = 20,
        epochs:            int   = 100,
        grad_clip:         float = 0.5,
        loss_kwargs:       Optional[dict] = None,
        checkpoint_dir:    str   = "outputs/checkpoints",
        run_name:          str   = "v4_run",
        true_tails:        Optional[dict] = None,
        toy_kg             = None,
        log_every_n:       int   = 10,
    ) -> None:
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.epochs         = epochs
        self.grad_clip      = grad_clip
        self.lattice_warmup = lattice_warmup
        self.run_name       = run_name
        self.true_tails     = true_tails or {}
        self.toy_kg         = toy_kg
        self.log_every_n    = log_every_n

        # Loss function
        loss_kwargs = loss_kwargs or {}
        self.loss_fn = V4Loss(**loss_kwargs)

        # Checkpoint manager
        ckpt_path = Path(checkpoint_dir) / run_name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if _KG_AVAILABLE:
            self.ckpt = CheckpointManager(ckpt_path)
            self.log  = RichLogger(f"v4_trainer.{run_name}")
        else:
            self.ckpt = None
            self.log  = None

        # Build optimizer with 7 parameter groups
        self.optimizer = self._build_optimizer(
            lr_base, lr_i_mult, lr_j_mult, lr_k_mult,
            lr_unitary_mult, lr_lattice_mult, lr_router_mult,
        )
        # Cosine LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr_base * 0.01
        )

        # Training state
        self.best_mrr   = 0.0
        self.best_result = None
        self._history   = []

    def _build_optimizer(
        self,
        lr_base:         float,
        lr_i_mult:       float,
        lr_j_mult:       float,
        lr_k_mult:       float,
        lr_unitary_mult: float,
        lr_lattice_mult: float,
        lr_router_mult:  float,
    ) -> torch.optim.Adam:
        """
        Build Adam optimizer with 7 independent LR parameter groups.

        GROUP RATIONALE:
            real (1×):    Real components — stable, no anti-collapse needed
            i_comp (3×):  i-component of quaternion — main "phase carrier," needs boost
            j_comp (2×):  j-component — secondary rotation axis
            k_comp (2×):  k-component — tertiary rotation axis
            unitary (2×): Relation quaternions — must rotate away from identity
            lattice (0.5×): Logic embeddings — must converge slowly to binary
            router (0.1×):  Routing thresholds — tiny LR to prevent oscillation

        weight_decay=0 for i,j,k,unitary: decay would push quaternion toward zero (Phase Collapse)
        weight_decay=1e-5 for real: standard regularization on entity magnitude
        """
        enc = self.model.encoder
        uni = self.model.unitary

        groups = [
            {
                "name":         "real",
                "params":       list(enc.emb_r.parameters()),
                "lr":           lr_base,
                "weight_decay": 1e-5,
            },
            {
                "name":         "i_comp",
                "params":       list(enc.emb_i.parameters()),
                "lr":           lr_base * lr_i_mult,
                "weight_decay": 0.0,   # CRITICAL: no decay on phase-carrying component
            },
            {
                "name":         "j_comp",
                "params":       list(enc.emb_j.parameters()),
                "lr":           lr_base * lr_j_mult,
                "weight_decay": 0.0,
            },
            {
                "name":         "k_comp",
                "params":       list(enc.emb_k.parameters()),
                "lr":           lr_base * lr_k_mult,
                "weight_decay": 0.0,
            },
            {
                "name":         "unitary",
                "params":       list(uni.parameters()),
                "lr":           lr_base * lr_unitary_mult,
                "weight_decay": 0.0,
            },
        ]

        # Lattice and router groups (only if modules exist)
        if self.model.use_lattice and self.model.lattice:
            groups.append({
                "name":         "lattice",
                "params":       list(self.model.lattice.parameters()),
                "lr":           lr_base * lr_lattice_mult,
                "weight_decay": 1e-5,
            })

        if self.model.use_routing and self.model.router:
            groups.append({
                "name":         "router",
                "params":       list(self.model.router.parameters()),
                "lr":           lr_base * lr_router_mult,
                "weight_decay": 0.0,
            })

        if self.model.mv_aggregator:
            groups.append({
                "name":         "mv_aggregator",
                "params":       list(self.model.mv_aggregator.parameters()),
                "lr":           lr_base,
                "weight_decay": 1e-5,
            })

        # Relation bias
        groups.append({
            "name":         "bias",
            "params":       [self.model.relation_bias],
            "lr":           lr_base,
            "weight_decay": 0.0,
        })

        optimizer = torch.optim.Adam(groups)
        group_str = " | ".join(f"{g['name']} lr={g['lr']:.5f}" for g in groups)
        print(f"[V4Trainer] {self.run_name} | Groups: {group_str}")
        return optimizer

    def _freeze_lattice(self) -> None:
        """Freeze lattice parameters during quaternion warmup phase."""
        if self.model.use_lattice and self.model.lattice:
            for p in self.model.lattice.parameters():
                p.requires_grad_(False)

    def _unfreeze_lattice(self) -> None:
        """Unfreeze lattice parameters for joint training."""
        if self.model.use_lattice and self.model.lattice:
            for p in self.model.lattice.parameters():
                p.requires_grad_(True)

    def _train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Returns loss breakdown."""
        self.model.train()
        total_loss = 0.0
        loss_components = {"l_quat": 0.0, "l_lattice": 0.0, "l_interf": 0.0, "l_router": 0.0}
        n_batches = 0

        # Reset routing stats at start of epoch
        if self.model.use_routing and self.model.router:
            self.model.router.reset_stats()

        for batch in self.train_loader:
            h   = batch["positive"][:, 0].to(self.device)
            r   = batch["positive"][:, 1].to(self.device)
            t   = batch["positive"][:, 2].to(self.device)
            neg = batch["negatives"].to(self.device)   # (B, K, 3)
            B, K, _ = neg.shape

            # Flatten negatives for scoring
            neg_h = neg[:, :, 0].reshape(-1)
            neg_r = neg[:, :, 1].reshape(-1)
            neg_t = neg[:, :, 2].reshape(-1)

            # Scores
            pos_scores = self.model.score_triple(h, r, t)
            neg_scores = self.model.score_triple(neg_h, neg_r, neg_t).view(B, K)

            # Full V4 loss
            loss_dict = self.loss_fn(
                model      = self.model,
                pos_scores = pos_scores,
                neg_scores = neg_scores,
                h_ids      = h,
                r_ids      = r,
                t_pos_ids  = t,
                t_neg_ids  = neg,
            )
            loss = loss_dict["total"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_components.items():
                loss_components[k] += loss_dict.get(k, 0.0)
            n_batches += 1

        n_batches = max(n_batches, 1)
        return {
            "loss":       total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()},
        }

    def _evaluate(self, loader: DataLoader) -> "MetricResults":
        """Run filtered MRR evaluation."""
        if not _KG_AVAILABLE:
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

    def _log_epoch(self, epoch: int, train_stats: dict, val_result, elapsed: float) -> None:
        """Print epoch summary."""
        mrr = val_result.mrr if val_result else 0.0
        h1  = val_result.hits_at_1 if val_result else 0.0
        h10 = val_result.hits_at_10 if val_result else 0.0

        # Quaternion health check: mean imag norm of i-component
        with torch.no_grad():
            i_norms = self.model.encoder.emb_i.weight.norm(dim=-1)
            mean_i_norm = float(i_norms.mean())

        # Routing stats
        routing_str = ""
        if self.model.use_routing and self.model.router:
            stats = self.model.router.get_routing_stats()
            routing_str = (
                f" | route: {stats['pct_classical']*100:.0f}%cls "
                f"{stats['pct_quantum']*100:.0f}%q"
            )

        # Logic score
        logic_str = ""
        if self.model.use_lattice and self.model.lattice:
            sample_rels = torch.arange(
                min(10, self.model.num_relations), device=self.device
            )
            logic_scores = self.model.lattice.logic_score(sample_rels)
            logic_str = f" | logic={logic_scores.mean():.3f}"

        print(
            f"[{self.run_name}] Epoch {epoch:4d} | "
            f"loss={train_stats['loss']:.4f} "
            f"(q={train_stats.get('l_quat',0):.3f} "
            f"lat={train_stats.get('l_lattice',0):.3f} "
            f"int={train_stats.get('l_interf',0):.3f}) | "
            f"MRR={mrr:.4f} H@1={h1:.4f} H@10={h10:.4f} | "
            f"i_norm={mean_i_norm:.3f}{logic_str}{routing_str} | "
            f"{elapsed:.1f}s"
        )

        self._history.append({
            "epoch": epoch, "loss": train_stats["loss"],
            "mrr": mrr, "h1": h1, "h10": h10,
            "i_norm": mean_i_norm,
        })

    def train(self):
        """
        Full V4 training loop with two-phase protocol.

        Phase 1 (epochs 1 to lattice_warmup): quaternion pre-training, lattice frozen.
        Phase 2 (epochs lattice_warmup+1 to end): joint training, all components active.

        Returns:
            Best validation MetricResults.
        """
        print(f"\n{'='*70}")
        print(f"V4 Training: {self.run_name}")
        print(f"  Epochs: {self.epochs} | Lattice warmup: {self.lattice_warmup}")
        print(f"  Quaternion dim: {self.model.quaternion_dim}")
        print(f"  Lattice: {'ON' if self.model.use_lattice else 'OFF'}")
        print(f"  Dynamic Routing: {'ON' if self.model.use_routing else 'OFF'}")
        print(f"  Multi-View Interference: {'ON' if self.model.use_mv_interference else 'OFF'}")
        print(f"{'='*70}\n")

        # Phase 1: freeze lattice during quaternion warmup
        self._freeze_lattice()
        lattice_active = False

        for epoch in range(1, self.epochs + 1):
            # Phase transition
            if epoch > self.lattice_warmup and not lattice_active:
                self._unfreeze_lattice()
                lattice_active = True
                print(f"  [Epoch {epoch}] → Phase 2: Lattice training activated")

            t0 = time.time()
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

            if epoch % self.log_every_n == 0 or epoch == 1:
                self._log_epoch(epoch, train_stats, val_result, elapsed)

        print(f"\nTraining complete. Best val MRR: {self.best_mrr:.4f}")
        return self.best_result

    def load_best_and_evaluate(self, test_loader: DataLoader):
        """Load best checkpoint and evaluate on test set."""
        if self.ckpt and (self.ckpt.dir / "best.pt").exists():
            self.ckpt.load_best(self.model, device=self.device)
        return self._evaluate(test_loader)

    def get_training_history(self) -> list:
        """Return per-epoch training history for paper figures."""
        return self._history
