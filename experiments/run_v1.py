"""
experiments/run_v1.py — V1 Experiment: Standard Training, All Models  [V1]

PURPOSE:
    Runs the complete V1 experiment pipeline. Trains QuantumReasoner and
    all three baselines (TransE, RotatE, ComplEx) on the toy KG using the
    standard V1 trainer (single LR, BCE/margin/self-adversarial loss).
    Optionally extends to FB15k-237.

    V1 EXPECTED RESULTS:
        QuantumReasoner with standard BCE loss will likely suffer Phase Collapse.
        TransE, RotatE, ComplEx will outperform QuantumReasoner.
        Interference will NOT emerge from standard training.
        All models degrade comparably under noise.

    This is NOT a failure — it establishes the baseline that V2 improves upon.
    Save the V1 CSV, then run run_v2.py and compare.

WHAT THIS PRODUCES:
    outputs/results/v1_comparison.csv     All models, MRR+Hits@K on test set
    outputs/results/v1_ablation.csv       4 ablation conditions × 5 noise levels
    outputs/figures/v1_noise_degradation.pdf  Figure 3 (V1 version — flat curves expected)
    outputs/figures/v1_training_curves.pdf    Training dynamics comparison

USAGE:
    python experiments/run_v1.py                        # toy KG, all models
    python experiments/run_v1.py --dataset fb15k237     # FB15k-237 (slow!)
    python experiments/run_v1.py --quick                # 20 epochs, fast test
    python experiments/run_v1.py --models transe rotate # specific models only
    python experiments/run_v1.py --ablation             # also run ablation study
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from rich.console import Console
from rich.panel import Panel

from data.toy_kg import build_toy_kg
from data.dataset import build_dataloaders, build_true_tails_dict
from models.quantum_reasoner import QuantumReasoner
from models.baselines.transe    import TransE
from models.baselines.rotate    import RotatE
from models.baselines.complex_e import ComplEx
from training.trainer import Trainer
from training.losses  import build_loss
from evaluation.metrics import RankingMetrics, MetricResults, compare_versions
from evaluation.ablation import AblationRunner, STANDARD_ABLATIONS, NOISE_LEVELS
from visualization.phase_plots import set_paper_style
from utils.seed import set_seed, get_device
from utils.logger import RichLogger

console = Console()
log     = RichLogger("run_v1")


# ── Model registry ─────────────────────────────────────────────────────────────

def build_model(model_name: str, kg, embed_dim: int = 16):
    """Build the requested model with appropriate defaults."""
    if model_name == "quantum":
        return QuantumReasoner(
            num_entities  = kg.num_entities,
            num_relations = kg.num_relations,
            embed_dim     = embed_dim,
            unitary_type  = "diagonal",
            max_paths     = 8,
            max_hops      = 2,
            ablation_mode = "full",
        )
    elif model_name == "transe":
        return TransE(
            num_entities  = kg.num_entities,
            num_relations = kg.num_relations,
            embed_dim     = embed_dim,
        )
    elif model_name == "rotate":
        return RotatE(
            num_entities  = kg.num_entities,
            num_relations = kg.num_relations,
            embed_dim     = embed_dim,
        )
    elif model_name == "complex":
        return ComplEx(
            num_entities  = kg.num_entities,
            num_relations = kg.num_relations,
            embed_dim     = embed_dim,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_loss_type(model_name: str) -> str:
    """Return the appropriate loss type for each model."""
    return {
        "quantum": "bce",
        "transe":  "margin",
        "rotate":  "self_adversarial",
        "complex": "bce",
    }[model_name]


def get_display_name(model_name: str) -> str:
    return {
        "quantum": "QuantumReasoner",
        "transe":  "TransE",
        "rotate":  "RotatE",
        "complex": "ComplEx",
    }[model_name]


# ── Main experiment ───────────────────────────────────────────────────────────

def run_model(
    model_name:   str,
    kg,
    device:       torch.device,
    train_dl,
    val_dl,
    test_dl,
    true_tails:   dict,
    epochs:       int   = 100,
    lr:           float = 0.005,
    embed_dim:    int   = 16,
    run_name:     str   = "v1",
) -> MetricResults:
    """Train and evaluate one model. Return test MetricResults."""
    display = get_display_name(model_name)
    log.print_banner(f"Training {display} [V1]", color="blue")

    model     = build_model(model_name, kg, embed_dim).to(device)
    loss_type = get_loss_type(model_name)

    # Different LR for margin-based losses
    actual_lr = lr * 0.1 if loss_type == "margin" else lr

    trainer = Trainer(
        model          = model,
        train_loader   = train_dl,
        val_loader     = val_dl,
        device         = device,
        loss_type      = loss_type,
        lr             = actual_lr,
        grad_clip      = 1.0,
        epochs         = epochs,
        checkpoint_dir = str(ROOT / "outputs" / "checkpoints"),
        run_name       = f"{run_name}_{model_name}",
        true_tails     = true_tails,
        log_every_n    = max(epochs // 10, 5),
    )

    trainer.train()
    results = trainer.load_best_and_evaluate(test_dl)
    log.info(f"{display} test results: {results}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 experiment: standard training, all models")
    parser.add_argument("--dataset",  type=str, default="toy",
                        choices=["toy", "fb15k237", "wn18rr"],
                        help="Dataset to use.")
    parser.add_argument("--models",   nargs="+",
                        default=["quantum", "transe", "rotate", "complex"],
                        choices=["quantum", "transe", "rotate", "complex"],
                        help="Models to train.")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--embed_dim",type=int, default=16)
    parser.add_argument("--lr",       type=float, default=0.005)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--quick",    action="store_true",
                        help="Quick mode: 20 epochs for testing.")
    parser.add_argument("--ablation", action="store_true",
                        help="Also run ablation study after main training.")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20

    set_seed(args.seed)
    device = get_device("cuda")

    console.print(Panel(
        f"[bold blue]V1 Experiment[/bold blue]\n"
        f"Dataset: {args.dataset} | Models: {args.models} | "
        f"Epochs: {args.epochs} | Device: {device}",
        border_style="blue",
    ))

    # ── Build data ─────────────────────────────────────────────────────────────
    if args.dataset == "toy":
        kg         = build_toy_kg(seed=args.seed)
        train_dl, val_dl, test_dl = build_dataloaders(
            kg, batch_size=8, num_negatives=4
        )
        true_tails = build_true_tails_dict(kg)
        log.info(kg.summary())
    else:
        # Real dataset — requires prior download
        from data.download import load_entity_relation_maps, load_vocab
        from data.dataset import KGDataset, collate_fn
        from torch.utils.data import DataLoader

        raw_dir = ROOT / "data" / "raw" / args.dataset
        if not raw_dir.exists():
            console.print(
                f"[red]Dataset not found: {raw_dir}[/red]\n"
                f"Run first: python data/download.py --dataset {args.dataset}"
            )
            return 1

        entity2id, relation2id = load_entity_relation_maps(raw_dir)
        all_triples_path       = str(raw_dir / "train.txt")

        train_ds = KGDataset.from_text_file(
            str(raw_dir / "train.txt"), entity2id, relation2id,
            all_triples_path=all_triples_path, num_negatives=128, mode="train",
        )
        val_ds = KGDataset.from_text_file(
            str(raw_dir / "valid.txt"), entity2id, relation2id,
            all_triples_path=all_triples_path, num_negatives=0, mode="eval",
        )
        test_ds = KGDataset.from_text_file(
            str(raw_dir / "test.txt"), entity2id, relation2id,
            all_triples_path=all_triples_path, num_negatives=0, mode="eval",
        )

        train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True,
                              collate_fn=collate_fn, num_workers=(0 if __import__("sys").platform == "win32" else 2))
        val_dl   = DataLoader(val_ds,   batch_size=256,  shuffle=False,
                              collate_fn=collate_fn, num_workers=(0 if __import__("sys").platform == "win32" else 2))
        test_dl  = DataLoader(test_ds,  batch_size=256,  shuffle=False,
                              collate_fn=collate_fn, num_workers=(0 if __import__("sys").platform == "win32" else 2))

        true_tails = dict(train_ds.true_tails)

        # Fake kg interface for compatibility
        class SimpleKG:
            num_entities  = len(entity2id)
            num_relations = len(relation2id)
        kg = SimpleKG()

        args.embed_dim = 256
        args.lr        = 0.001
        log.info(f"Dataset: {args.dataset} — {len(entity2id):,} entities, "
                 f"{len(relation2id):,} relations")

    # ── Train all models ────────────────────────────────────────────────────────
    all_results: dict[str, MetricResults] = {}
    run_name = f"v1_{args.dataset}"

    t0_total = time.time()

    for model_name in args.models:
        results = run_model(
            model_name = model_name,
            kg         = kg,
            device     = device,
            train_dl   = train_dl,
            val_dl     = val_dl,
            test_dl    = test_dl,
            true_tails = true_tails,
            epochs     = args.epochs,
            lr         = args.lr,
            embed_dim  = args.embed_dim,
            run_name   = run_name,
        )
        all_results[get_display_name(model_name)] = results

    total_time = time.time() - t0_total

    # ── Save results ───────────────────────────────────────────────────────────
    results_dir = ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "v1_comparison.csv"

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["model", "MRR", "H@1", "H@3", "H@10", "N"]
        writer     = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in all_results.items():
            writer.writerow({
                "model": model_name,
                "MRR":   f"{metrics.mrr:.4f}",
                "H@1":   f"{metrics.hits_at_1:.4f}",
                "H@3":   f"{metrics.hits_at_3:.4f}",
                "H@10":  f"{metrics.hits_at_10:.4f}",
                "N":     metrics.num_triples,
            })

    log.info(f"V1 results saved: {csv_path}")

    # ── Print comparison ───────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]V1 Results Summary[/bold] ({total_time/60:.1f} min total)\n"
        f"Results saved: {csv_path}\n\n"
        "Expected: QuantumReasoner may underperform baselines (Phase Collapse).\n"
        "This is correct — V2 fixes it with TrainerV2 + InterferenceAwareLoss.\n\n"
        "Next step: python experiments/run_v2.py",
        border_style="blue",
    ))
    log.print_comparison_table(
        {k: v.to_dict() for k, v in all_results.items()},
        title="V1 Test Results"
    )

    # ── Optional ablation ──────────────────────────────────────────────────────
    if args.ablation and args.dataset == "toy":
        log.print_banner("Running V1 Ablation Study", color="yellow")
        runner  = AblationRunner(kg, device, run_name=f"ablation_{run_name}")
        ablation_results = runner.run_all(
            modes        = STANDARD_ABLATIONS,
            noise_levels = NOISE_LEVELS if not args.quick else [0.0, 0.10, 0.20],
            epochs       = args.epochs,
        )
        runner.print_ablation_table(ablation_results)
        runner.save_results_csv(
            ablation_results,
            str(results_dir / "v1_ablation.csv"),
        )

        # Generate noise degradation figure
        try:
            set_paper_style()
            from visualization.phase_plots import plot_noise_degradation
            import matplotlib.pyplot as plt

            # Reformat ablation results for plotting
            noise_mrr: dict[str, dict] = {}
            for r in ablation_results:
                label = r.config.mode
                if label not in noise_mrr:
                    noise_mrr[label] = {}
                noise_mrr[label][r.config.noise_rate] = r.metrics.mrr

            fig = plot_noise_degradation(
                noise_mrr,
                save_path      = ROOT / "outputs" / "figures" / "v1_noise_degradation.pdf",
                highlight_model= "full",
            )
            plt.close(fig)
            log.info("Noise degradation figure saved.")
        except Exception as e:
            log.warning(f"Figure generation failed (non-critical): {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
