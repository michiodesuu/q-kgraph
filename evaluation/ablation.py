"""
evaluation/ablation.py — Ablation Study Automation  [V1]

Trains four ablation conditions under identical settings and reports
MRR at multiple noise levels. Generates paper Table 3 and Figure 3 data.

THE FOUR CONDITIONS:
    full:      Complete QuantumReasoner with interference (the actual model).
    no_phase:  Zero imaginary components — tests whether complex phases help.
    no_paths:  Single-hop only — tests whether multi-hop paths help.
    classical: Re(amplitude) instead of |amplitude|² — tests Born rule squaring.

CRITICAL INTERPRETATION:
    For the paper's claim to hold:
        - full > no_phase:    phase angles carry information
        - full > no_paths:    multi-hop reasoning matters
        - full > classical:   Born rule squaring (not just complex arithmetic) is key
        - gap widens with noise: interference specifically helps under noise

USAGE:
    runner = AblationRunner(kg, device, run_name="toy_ablation_v1")
    results = runner.run_all(epochs=100, noise_levels=[0.0, 0.10, 0.20])
    runner.print_ablation_table(results)
    runner.save_results_csv(results, "outputs/results/ablation_results.csv")
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from evaluation.metrics import RankingMetrics, MetricResults
from data.dataset import build_dataloaders, build_true_tails_dict
from data.noise_injection import NoiseInjector
from models.quantum_reasoner import QuantumReasoner
from training.losses import build_loss
from utils.logger import RichLogger

# Standard ablation conditions (paper Table 3)
STANDARD_ABLATIONS = ["full", "no_phase", "no_paths", "classical"]
NOISE_LEVELS       = [0.0, 0.05, 0.10, 0.15, 0.20]


@dataclass
class AblationConfig:
    """Configuration for one ablation run."""
    mode:           str   = "full"
    noise_rate:     float = 0.0
    noise_type:     str   = "random"
    epochs:         int   = 100
    lr:             float = 0.005
    batch_size:     int   = 8
    num_negatives:  int   = 4
    embed_dim:      int   = 16
    seed:           int   = 42


@dataclass
class AblationResult:
    """Result of one ablation run."""
    config:     AblationConfig
    metrics:    MetricResults
    train_time: float = 0.0
    triple_ids: list  = None    # for RankStability tracking
    test_ranks: list  = None    # 1-indexed ranks on test set

    @property
    def label(self) -> str:
        return f"{self.config.mode}_noise{self.config.noise_rate:.2f}"


class AblationRunner:
    """
    Runs all four ablation conditions at multiple noise levels.

    Supports two usage modes:
    - Toy KG mode: pass toy_kg; data loaders are built internally.
    - Real dataset mode: pass base_model + pre-built data loaders.

    Args:
        toy_kg:       ToyKG instance (toy mode).
        device:       torch.device.
        run_name:     Experiment label for logging and file naming.
        verbose:      Print per-epoch progress.
        base_model:   Pre-built QuantumReasoner (real dataset mode).
        train_loader: DataLoader for training triples.
        val_loader:   DataLoader for validation triples.
        test_loader:  DataLoader for test triples.
        epochs:       Training epochs per condition.
        lr:           Learning rate.
        output_dir:   Directory for saving results.
        true_tails:   {(h,r): set(t)} mapping for filtered ranking.
    """

    def __init__(
        self,
        toy_kg                   = None,
        device:   torch.device   = None,
        run_name: str            = "ablation",
        verbose:  bool           = True,
        base_model               = None,
        train_loader             = None,
        val_loader               = None,
        test_loader              = None,
        epochs:   int            = 100,
        lr:       float          = 0.005,
        output_dir: str          = "outputs/results/ablation",
        true_tails: dict         = None,
    ) -> None:
        self.kg           = toy_kg
        self.device       = device
        self.run_name     = run_name
        self.verbose      = verbose
        self.log          = RichLogger(f"ablation.{run_name}")
        self.base_model   = base_model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.epochs       = epochs
        self.lr           = lr
        self.output_dir   = Path(output_dir)
        self.true_tails   = true_tails

        if toy_kg is not None:
            self.injector = NoiseInjector(toy_kg.num_entities, toy_kg.num_relations)
        elif base_model is not None:
            self.injector = NoiseInjector(base_model.num_entities, base_model.num_relations)
        else:
            self.injector = None

    def run_all(
        self,
        modes:        list[str]   = None,
        noise_levels: list[float] = None,
        epochs:       int         = 100,
        embed_dim:    int         = 16,
        seed:         int         = 42,
    ) -> list[AblationResult]:
        """
        Run all ablation conditions at all noise levels.

        Args:
            modes:        Ablation modes to run. Default: all 4 standard conditions.
            noise_levels: Noise rates to test. Default: [0, 0.05, 0.10, 0.15, 0.20].
            epochs:       Training epochs per condition.
            embed_dim:    Model embedding dimension.
            seed:         Random seed.

        Returns:
            List of AblationResult, one per (mode, noise_level) combination.
        """
        modes        = modes        or STANDARD_ABLATIONS
        noise_levels = noise_levels or NOISE_LEVELS

        total  = len(modes) * len(noise_levels)
        done   = 0
        results: list[AblationResult] = []

        self.log.print_banner(
            f"Ablation Study: {len(modes)} modes × {len(noise_levels)} noise levels "
            f"= {total} runs"
        )

        for mode in modes:
            for noise_rate in noise_levels:
                done += 1
                config = AblationConfig(
                    mode        = mode,
                    noise_rate  = noise_rate,
                    epochs      = epochs,
                    embed_dim   = embed_dim,
                    seed        = seed,
                )
                self.log.info(
                    f"[{done}/{total}] mode={mode}, noise={noise_rate:.0%}"
                )
                result = self._run_single(config)
                results.append(result)

        return results

    def _run_single(self, config: AblationConfig) -> AblationResult:
        """Train and evaluate one (mode, noise_level) condition."""
        from utils.seed import set_seed
        set_seed(config.seed)

        # Build noisy training data
        train_ids = [self.kg.triple_to_ids(t) for t in self.kg.train_triples]
        true_set  = self.kg.get_true_set()

        if config.noise_rate > 0:
            train_ids, _ = self.injector.inject(
                train_ids, true_set,
                corruption_rate=config.noise_rate,
                noise_type=config.noise_type,
            )

        # Rebuild dataset with (possibly noisy) training triples
        from data.dataset import KGDataset, collate_fn
        from torch.utils.data import DataLoader

        train_ds = KGDataset(
            triples         = train_ids,
            num_entities    = self.kg.num_entities,
            num_relations   = self.kg.num_relations,
            num_negatives   = config.num_negatives,
            true_set        = true_set,
            label_smoothing = 0.9,
            mode            = "train",
            seed            = config.seed,
        )
        val_ds = KGDataset.from_toy_kg(self.kg, split="val", seed=config.seed)
        test_ds= KGDataset.from_toy_kg(self.kg, split="test", seed=config.seed)

        train_dl = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
        val_dl   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_fn)
        test_dl  = DataLoader(test_ds,  batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_fn)

        # Build model with ablation mode
        model = QuantumReasoner(
            num_entities  = self.kg.num_entities,
            num_relations = self.kg.num_relations,
            embed_dim     = config.embed_dim,
            unitary_type  = "diagonal",
            max_paths     = 8,
            max_hops      = 2,
            ablation_mode = config.mode,
        ).to(self.device)

        true_tails = build_true_tails_dict(self.kg)
        loss_fn    = build_loss("bce")
        optimizer  = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Train
        t0         = time.perf_counter()
        best_mrr   = 0.0
        val_metrics = RankingMetrics(filter_false_negatives=True)

        for epoch in range(1, config.epochs + 1):
            model.train()
            for batch in train_dl:
                h   = batch["positive"][:, 0].to(self.device)
                r   = batch["positive"][:, 1].to(self.device)
                t   = batch["positive"][:, 2].to(self.device)
                neg = batch["negatives"].to(self.device)
                B, K, _ = neg.shape

                optimizer.zero_grad()
                pos_scores = model.score_triple(h, r, t)
                neg_scores = model.score_triple(
                    neg[:,:,0].reshape(-1),
                    neg[:,:,1].reshape(-1),
                    neg[:,:,2].reshape(-1),
                ).view(B, K)
                loss = loss_fn(pos_scores, neg_scores)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Quick val check every 20 epochs
            if epoch % 20 == 0 or epoch == config.epochs:
                model.eval()
                val_metrics.reset()
                with torch.no_grad():
                    for batch in val_dl:
                        h = batch["positive"][:, 0].to(self.device)
                        r = batch["positive"][:, 1].to(self.device)
                        t = batch["positive"][:, 2].to(self.device)
                        scores = model.score_triple_vs_all(h, r)
                        val_metrics.update(scores, t, h, r, true_tails)
                vr = val_metrics.compute()
                if vr.mrr > best_mrr:
                    best_mrr = vr.mrr

        # Final test evaluation
        model.eval()
        test_metrics = RankingMetrics(filter_false_negatives=True)
        with torch.no_grad():
            for batch in test_dl:
                h = batch["positive"][:, 0].to(self.device)
                r = batch["positive"][:, 1].to(self.device)
                t = batch["positive"][:, 2].to(self.device)
                scores = model.score_triple_vs_all(h, r)
                test_metrics.update(scores, t, h, r, true_tails)

        # ADD a rank collection step inside the same torch.no_grad() block:
        # Collect per-triple ranks for RankStability
        triple_ids_test = []
        test_ranks      = []
        model.eval()
        with torch.no_grad():
            for batch in test_dl:
                h = batch["positive"][:, 0].to(self.device)
                r = batch["positive"][:, 1].to(self.device)
                t = batch["positive"][:, 2].to(self.device)
                scores = model.score_triple_vs_all(h, r)
                B = scores.shape[0]
                for b in range(B):
                    h_id, r_id, t_id = h[b].item(), r[b].item(), t[b].item()
                    # Apply filter
                    s_b = scores[b].clone()
                    for other_t in true_tails.get((h_id, r_id), set()):
                        if other_t != t_id:
                            s_b[other_t] = float("-inf")
                    rank = int((s_b > s_b[t_id]).sum().item()) + 1
                    triple_ids_test.append((h_id, r_id, t_id))
                    test_ranks.append(rank)

        elapsed = time.perf_counter() - t0
        return AblationResult(
            config     = config,
            metrics    = test_metrics.compute(),
            train_time = elapsed,
        )

    # ── Real-dataset methods ───────────────────────────────────────────────────

    def run_noise_experiment(
        self,
        modes:        list[str]   = None,
        noise_levels: list[float] = None,
    ) -> dict:
        """
        Run all ablation conditions at all noise levels for a real dataset.

        Returns:
            Nested dict: {mode: {noise_level: AblationResult}}
        """
        import copy
        from torch.utils.data import DataLoader
        from data.dataset import KGDataset, collate_fn

        modes        = modes        or STANDARD_ABLATIONS
        noise_levels = noise_levels or NOISE_LEVELS

        n_ent = self.base_model.num_entities
        n_rel = self.base_model.num_relations

        train_ds    = self.train_loader.dataset
        train_triples = list(train_ds.triples)
        batch_size  = self.train_loader.batch_size or 512

        true_set = {(h, r, t)
                    for (h, r), tails in self.true_tails.items()
                    for t in tails}

        results: dict = {}
        total = len(modes) * len(noise_levels)
        done  = 0

        for mode in modes:
            results[mode] = {}
            for noise_level in noise_levels:
                done += 1
                self.log.info(
                    f"[{done}/{total}] mode={mode}, noise={noise_level:.0%}"
                )
                config = AblationConfig(
                    mode       = mode,
                    noise_rate = noise_level,
                    epochs     = self.epochs,
                    lr         = self.lr,
                )

                # Inject noise into training triples if requested
                if noise_level > 0 and self.injector is not None:
                    noisy_triples, _ = self.injector.inject(
                        train_triples, true_set,
                        corruption_rate=noise_level,
                        noise_type=config.noise_type,
                    )
                else:
                    noisy_triples = train_triples

                noisy_ds = KGDataset(
                    triples        = noisy_triples,
                    num_entities   = n_ent,
                    num_relations  = n_rel,
                    num_negatives  = getattr(train_ds, "num_negatives", 4),
                    true_set       = true_set,
                    label_smoothing= getattr(train_ds, "label_smoothing", 0.9),
                    mode           = "train",
                    seed           = config.seed,
                )
                noisy_dl = DataLoader(
                    noisy_ds, batch_size=batch_size,
                    shuffle=True, collate_fn=collate_fn,
                )

                model = copy.deepcopy(self.base_model)
                model.ablation_mode = mode
                model = model.to(self.device)

                loss_fn   = build_loss("bce")
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

                t0 = time.perf_counter()
                for epoch in range(1, config.epochs + 1):
                    model.train()
                    for batch in noisy_dl:
                        h   = batch["positive"][:, 0].to(self.device)
                        r   = batch["positive"][:, 1].to(self.device)
                        t   = batch["positive"][:, 2].to(self.device)
                        neg = batch["negatives"].to(self.device)
                        B, K, _ = neg.shape

                        optimizer.zero_grad()
                        pos_scores = model.score_triple(h, r, t)
                        neg_scores = model.score_triple(
                            neg[:, :, 0].reshape(-1),
                            neg[:, :, 1].reshape(-1),
                            neg[:, :, 2].reshape(-1),
                        ).view(B, K)
                        loss = loss_fn(pos_scores, neg_scores)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                model.eval()
                test_metrics = RankingMetrics(filter_false_negatives=True)
                with torch.no_grad():
                    for batch in self.test_loader:
                        h = batch["positive"][:, 0].to(self.device)
                        r = batch["positive"][:, 1].to(self.device)
                        t = batch["positive"][:, 2].to(self.device)
                        scores = model.score_triple_vs_all(h, r)
                        test_metrics.update(scores, t, h, r, self.true_tails)

                elapsed = time.perf_counter() - t0
                results[mode][noise_level] = AblationResult(
                    config     = config,
                    metrics    = test_metrics.compute(),
                    train_time = elapsed,
                )

        return results

    def print_noise_table(self, noise_results: dict) -> None:
        """Print noise experiment results as paper Table 3 format.

        Args:
            noise_results: {mode: {noise_level: AblationResult}}
        """
        modes        = [m for m in STANDARD_ABLATIONS if m in noise_results]
        noise_levels = sorted(next(iter(noise_results.values())).keys())

        header = ["Mode"] + [f"MRR@{n:.0%}" for n in noise_levels] + \
                 [f"H@1@{n:.0%}" for n in noise_levels] + \
                 [f"H@10@{n:.0%}" for n in noise_levels]
        col_w  = [12] + [10] * len(noise_levels) * 3

        sep = "=" * (sum(col_w) + len(col_w) * 2)
        print("\n" + sep)
        print("ABLATION STUDY — MRR (filtered, test set)")
        print(sep)
        print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
        print("-" * (sum(col_w) + len(col_w) * 2))

        for mode in modes:
            row = [mode]
            for attr in ["mrr", "hits_at_1", "hits_at_10"]:
                for noise_level in noise_levels:
                    result = noise_results.get(mode, {}).get(noise_level)
                    row.append(
                        f"{getattr(result.metrics, attr):.4f}"
                        if result else "N/A"
                    )
            print("  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))

        print(sep)

    def print_ablation_table(self, results: list[AblationResult]) -> None:
        """Print ablation results as paper Table 3 format."""
        # Group by noise level
        noise_levels = sorted(set(r.config.noise_rate for r in results))
        modes        = [m for m in STANDARD_ABLATIONS
                        if any(r.config.mode == m for r in results)]

        # Header
        header = ["Mode"] + [f"MRR@{n:.0%}" for n in noise_levels] + \
                    [f"H@1@{n:.0%}" for n in noise_levels] + \
                    [f"H@10@{n:.0%}" for n in noise_levels]
        col_w  = [12] + [10] * len(noise_levels) * 3

        print("\n" + "=" * (sum(col_w) + len(col_w) * 2))
        print("ABLATION STUDY — MRR (filtered, test set)")
        print("=" * (sum(col_w) + len(col_w) * 2))
        print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
        print("-" * (sum(col_w) + len(col_w) * 2))

        for mode in modes:
            row = [mode]
            for attr in ["mrr", "hits_at_1", "hits_at_10"]:
                for noise_rate in noise_levels:
                    match = [r for r in results
                            if r.config.mode == mode
                            and abs(r.config.noise_rate - noise_rate) < 1e-6]
                    if match:
                        row.append(f"{getattr(match[0].metrics, attr):.4f}")
                    else:
                        row.append("N/A")
            print("  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))

        print("=" * (sum(col_w) + len(col_w) * 2))

    def save_results_csv(
        self,
        results:     list[AblationResult],
        output_path: str | Path = None,
        filename:    str        = None,
    ) -> None:
        """Save ablation results to CSV for LaTeX import."""
        if output_path is None and filename is not None:
            output_path = self.output_dir / filename
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["mode", "noise_rate", "noise_type",
                      "MRR", "H@1", "H@3", "H@10", "N", "train_time_s"]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "mode":        r.config.mode,
                    "noise_rate":  f"{r.config.noise_rate:.2f}",
                    "noise_type":  r.config.noise_type,
                    "MRR":         f"{r.metrics.mrr:.4f}",
                    "H@1":         f"{r.metrics.hits_at_1:.4f}",
                    "H@3":         f"{r.metrics.hits_at_3:.4f}",
                    "H@10":        f"{r.metrics.hits_at_10:.4f}",
                    "N":           r.metrics.num_triples,
                    "train_time_s":f"{r.train_time:.1f}",
                })

        self.log.info(f"Ablation results saved: {output_path}")
