"""
training/trainer.py — Standard Training Loop  [V1]

Single learning rate for all parameters. Used for V1 experiments.
For V2 (with parameter groups and interference monitoring), use trainer_v2.py.

USAGE:
    trainer = Trainer(model, train_dl, val_dl, device, lr=0.001, epochs=100)
    best_results = trainer.train()
"""
from __future__ import annotations
import time, csv
from pathlib import Path
from typing import Optional

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import RankingMetrics, MetricResults
from training.losses import build_loss
from utils.logger import RichLogger
from utils.checkpoint import CheckpointManager


class Trainer:
    """
    Standard training loop. Works with ALL models via score_triple() interface.

    Args:
        model:          Any model with score_triple(h,r,t) and score_triple_vs_all(h,r).
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        device:         torch.device.
        loss_type:      "bce", "margin", or "self_adversarial".
        lr:             Learning rate (single LR for all parameters).
        weight_decay:   L2 regularization.
        grad_clip:      Max gradient norm (0 = no clipping).
        epochs:         Training epochs.
        warmup_epochs:  Linear LR warmup.
        checkpoint_dir: Where to save checkpoints.
        run_name:       Experiment name for logging.
        true_tails:     {(h,r): set(t)} for filtered evaluation.
        log_every_n:    Print metrics every N epochs.
    """

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        device:         torch.device,
        loss_type:      str   = "bce",
        loss_kwargs:    Optional[dict] = None,
        lr:             float = 1e-3,
        weight_decay:   float = 1e-5,
        grad_clip:      float = 1.0,
        epochs:         int   = 100,
        warmup_epochs:  int   = 5,
        checkpoint_dir: str   = "outputs/checkpoints",
        run_name:       str   = "run_v1",
        true_tails:     Optional[dict] = None,
        log_every_n:    int   = 10,
        label_smoothing: float = 0.9,
        patience:       int   = 50,
        resume_from:    Optional[str] = None,
    ) -> None:
        self.model         = model.to(device)
        self.device        = device
        self.epochs        = epochs
        self.patience      = patience
        self.resume_from   = resume_from
        self.grad_clip     = grad_clip
        self.true_tails    = true_tails
        self.run_name      = run_name
        self.log_every_n   = log_every_n
        self.train_loader  = train_loader
        self.val_loader    = val_loader

        kw = {"label_smoothing": label_smoothing}
        if loss_kwargs:
            kw.update(loss_kwargs)
        self.loss_fn = build_loss(loss_type, **kw)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=lr * 0.01
        )

        self.ckpt = CheckpointManager(
            Path(checkpoint_dir) / run_name, keep_last_n=3, metric_name="MRR"
        )
        self.val_metrics = RankingMetrics(filter_false_negatives=True)
        self.log         = RichLogger(f"trainer.{run_name}")

        self.history: dict[str, list] = {
            "train_loss": [], "val_mrr": [], "val_hits1": [], "val_hits10": []
        }

    def train(self) -> MetricResults:
        """Run the full training loop. Returns best validation MetricResults."""
        self.log.print_banner(f"V1 Training: {self.run_name}")
        best_results = None
        epochs_no_improve = 0
        start_epoch = 1

        if self.resume_from:
            ckpt = torch.load(self.resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            self.log.info(f"Resumed from epoch {ckpt['epoch']} ({self.resume_from})")

        for epoch in range(start_epoch, self.epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch()
            val_results = self._val_epoch()
            self.scheduler.step()

            self.ckpt.save(
                model=self.model, optimizer=self.optimizer,
                epoch=epoch, metrics=val_results.to_dict(), scheduler=self.scheduler,
            )

            if epoch % self.log_every_n == 0 or epoch == 1:
                self.log.info(
                    f"[Epoch {epoch:4d}/{self.epochs}] "
                    f"loss={train_loss:.4f} | {val_results} | "
                    f"{time.time()-t0:.1f}s"
                )

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

            self.history["train_loss"].append(train_loss)
            self.history["val_mrr"].append(val_results.mrr)

        self.log.print_metrics(best_results.to_dict(), title="Best Validation")
        return best_results

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in tqdm(self.train_loader, desc="Train", leave=False):
            h    = batch["positive"][:, 0].to(self.device)
            r    = batch["positive"][:, 1].to(self.device)
            t    = batch["positive"][:, 2].to(self.device)
            negs = batch["negatives"].to(self.device)   # (B, K, 3)

            B, K, _ = negs.shape
            h_neg   = negs[:, :, 0].reshape(B * K)
            r_neg   = negs[:, :, 1].reshape(B * K)
            t_neg   = negs[:, :, 2].reshape(B * K)

            self.optimizer.zero_grad()

            pos_scores = self.model.score_triple(h, r, t)
            neg_scores = self.model.score_triple(h_neg, r_neg, t_neg).view(B, K)

            loss = self.loss_fn(pos_scores, neg_scores)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self) -> MetricResults:
        self.model.eval()
        self.val_metrics.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                h = batch["positive"][:, 0].to(self.device)
                r = batch["positive"][:, 1].to(self.device)
                t = batch["positive"][:, 2].to(self.device)
                scores = self.model.score_triple_vs_all(h, r)
                self.val_metrics.update(
                    scores=scores, true_indices=t,
                    head_ids=h, relation_ids=r,
                    true_tails=self.true_tails,
                )

        return self.val_metrics.compute()

    def load_best_and_evaluate(
        self,
        test_loader: DataLoader,
    ) -> MetricResults:
        """Load best checkpoint and evaluate on test set."""
        self.ckpt.load_best(self.model, device=self.device)
        test_metrics = RankingMetrics(filter_false_negatives=True)
        self.model.eval()

        with torch.no_grad():
            for batch in test_loader:
                h = batch["positive"][:, 0].to(self.device)
                r = batch["positive"][:, 1].to(self.device)
                t = batch["positive"][:, 2].to(self.device)
                scores = self.model.score_triple_vs_all(h, r)
                test_metrics.update(
                    scores=scores, true_indices=t,
                    head_ids=h, relation_ids=r,
                    true_tails=self.true_tails,
                )

        results = test_metrics.compute()
        self.log.print_metrics(results.to_dict(), title="TEST SET")
        return results

    def save_results_csv(
        self,
        results: MetricResults,
        model_name: str,
        output_path: str,
    ) -> None:
        """Save results to CSV for comparison tables."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model"] + list(results.to_dict().keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({"model": model_name, **results.to_dict()})
