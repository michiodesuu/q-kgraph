"""
utils/checkpoint.py — Model Checkpointing  [V1]
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Optional, Any
import torch
import torch.nn as nn


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir:   str | Path,
        keep_last_n:      int  = 3,
        metric_name:      str  = "mrr",
        higher_is_better: bool = True,
    ):
        self.dir              = Path(checkpoint_dir)
        self.keep_last_n      = keep_last_n
        self.metric_name      = metric_name
        self.higher_is_better = higher_is_better
        self._best_metric     = float("-inf") if higher_is_better else float("inf")
        self._saved_epochs:   list[int] = []
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        metrics:   dict[str, float],
        scheduler: Optional[Any] = None,
    ) -> None:
        payload = {
            "epoch":     epoch,
            "metrics":   metrics,
            "model":     self._unwrap_state_dict(model),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()

        # Save epoch checkpoint
        epoch_path = self.dir / f"epoch_{epoch:04d}.pt"
        torch.save(payload, epoch_path)
        self._saved_epochs.append(epoch)

        # Check if best
        current = metrics.get(self.metric_name, float("-inf"))
        is_best = (current > self._best_metric) if self.higher_is_better else (current < self._best_metric)
        if is_best:
            self._best_metric = current
            shutil.copy2(epoch_path, self.dir / "best.pt")

        # Save metadata
        with open(self.dir / "metadata.json", "w") as f:
            json.dump({"best_metric": self._best_metric, "epoch": epoch, "metrics": metrics}, f, indent=2)

        # Prune old checkpoints
        if len(self._saved_epochs) > self.keep_last_n:
            old_epoch = self._saved_epochs.pop(0)
            old_path  = self.dir / f"epoch_{old_epoch:04d}.pt"
            if old_path.exists():
                old_path.unlink()

    def load_best(self, model: nn.Module, device: torch.device) -> dict:
        best_path = self.dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best.pt found in {self.dir}")
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["model"])
        return payload

    def _unwrap_state_dict(self, model: nn.Module) -> dict:
        sd = model.state_dict()
        # Strip torch.compile() prefix if present
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
