"""
experiments/run_v2.py — V2 Experiment: Interference-Aware Training  [V2]

PURPOSE:
    Runs the complete V2 experiment pipeline. Trains QuantumReasoner with
    TrainerV2 (parameter groups, lr_imag=3x) and InterferenceAwareLoss
    (PhaseSeparationLoss + ContrastiveInterferenceLoss).

    V2 EXPECTED RESULTS vs V1:
        QuantumReasoner V2 > QuantumReasoner V1 (Phase Collapse fixed)
        QuantumReasoner V2 approaches or beats TransE/RotatE at noise >= 10%
        InterferenceMonitor reports num_destructive = 3/3 by epoch 50-70
        compare_versions() shows improvement between V1 and V2 CSVs

    KEY DIFFERENCE FROM V1:
        V1: single LR for all params, BCE loss only → Phase Collapse likely
        V2: lr_imag=3x, lr_phase=2x, InterferenceAwareLoss with ContrastiveLoss
            → interference emerges, wrong-answer paths suppressed

WHAT THIS PRODUCES:
    outputs/results/v2_comparison.csv      All models + V2 quantum, MRR+Hits@K
    outputs/results/v2_ablation.csv        4 ablation conditions × 5 noise levels
    outputs/results/v2_theorem.csv         Theorem 8.3 verification per query
    outputs/figures/v2_phase_diagram.pdf   Post-training phase diagram (money figure)
    outputs/figures/v2_noise_degradation.pdf  Figure 3 showing quantum advantage
    outputs/figures/v2_training_curves.pdf    Ablation training dynamics

USAGE:
    python experiments/run_v2.py                     # toy KG, all models
    python experiments/run_v2.py --dataset fb15k237  # full benchmark
    python experiments/run_v2.py --quick             # 20 epochs, fast test
    python experiments/run_v2.py --no_baselines      # quantum only (faster)
    python experiments/run_v2.py --compare v1        # show V1 vs V2 diff table

WORKFLOW:
    1. Build toy KG + infer relation structure
    2. Pre-compute path cache (BFS once, O(1) during training)
    3. Train QuantumReasoner with TrainerV2 + InterferenceAwareLoss
    4. Train all baselines with appropriate V1 losses (fair comparison)
    5. Run ablation study (4 conditions × 5 noise levels)
    6. Verify Theorem 8.3 conditions on trained model
    7. Generate all four paper figures
    8. Save comparison CSV
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
from rich.panel   import Panel
from rich.table   import Table

from data.toy_kg     import build_toy_kg
from data.dataset    import build_dataloaders, build_true_tails_dict
from data.path_cache import build_training_cache, PathCache
from data.noise_injection import NoiseInjector

from models.quantum_reasoner            import QuantumReasoner
from models.baselines.transe            import TransE
from models.baselines.rotate            import RotatE
from models.baselines.complex_e         import ComplEx
from models.components.kg_unitary       import RelationalDecomposedUnitary, infer_relation_structure

from training.trainer    import Trainer
from training.trainer_v2 import TrainerV2
from training.losses     import build_loss

from evaluation.metrics   import RankingMetrics, MetricResults, compare_versions
from evaluation.ablation  import AblationRunner, STANDARD_ABLATIONS, NOISE_LEVELS
from evaluation.chunked_evaluator import ChunkedEvaluator

from theory.noise_guarantee import verify_theorem_conditions

from visualization.phase_plots import (
    set_paper_style, plot_noise_degradation,
    generate_all_paper_figures, plot_training_curves,
)

from utils.seed    import set_seed, get_device
from utils.logger  import RichLogger

console = Console()
log     = RichLogger("run_v2")


# ── Baseline helpers (reused from run_v1 logic) ───────────────────────────────

def train_baseline(
    model_class,
    model_kwargs:  dict,
    loss_type:     str,
    kg,
    train_dl,
    val_dl,
    test_dl,
    true_tails:    dict,
    device:        torch.device,
    epochs:        int,
    lr:            float,
    run_name:      str,
    display_name:  str,
) -> MetricResults:
    """Train one baseline model using the standard V1 trainer."""
    model = model_class(**model_kwargs).to(device)
    trainer = Trainer(
        model          = model,
        train_loader   = train_dl,
        val_loader     = val_dl,
        device         = device,
        loss_type      = loss_type,
        lr             = lr * 0.1 if loss_type == "margin" else lr,
        grad_clip      = 1.0,
        epochs         = epochs,
        checkpoint_dir = str(ROOT / "outputs" / "checkpoints"),
        run_name       = f"{run_name}_{display_name.lower()}",
        true_tails     = true_tails,
        log_every_n    = max(epochs // 10, 5),
    )
    log.print_banner(f"Training {display_name} [V1 trainer, fair comparison]", color="green")
    trainer.train()
    return trainer.load_best_and_evaluate(test_dl)


# ── V2 QuantumReasoner training ────────────────────────────────────────────────

def train_quantum_v2(
    kg,
    train_dl,
    val_dl,
    test_dl,
    true_tails:   dict,
    device:       torch.device,
    epochs:       int,
    embed_dim:    int,
    run_name:     str,
    path_cache:   PathCache,
    use_relational_unitary: bool = True,
    lr_base:      float = 0.005,
    lr_imag_mult: float = 3.0,
    lr_phase_mult: float = 2.0,
    contrast_weight: float = 1.0,
    phase_weight:    float = 0.2,
) -> tuple[QuantumReasoner, MetricResults]:
    """
    Train QuantumReasoner with V2 protocol: parameter groups + InterferenceAwareLoss.

    Returns:
        (trained_model, test_results)
    """
    log.print_banner("Training QuantumReasoner [V2: TrainerV2 + InterferenceAwareLoss]",
                     color="blue")

    # Auto-detect relation structure for KG-specific unitary
    triple_set = {kg.triple_to_ids(t) for t in kg.triples}
    structure  = infer_relation_structure(triple_set, kg.num_relations)
    log.info(
        f"Relation structure: {len(structure['symmetric_rels'])} symmetric, "
        f"{len(structure['inverse_pairs'])} inverse pairs"
    )

    # Build model
    model = QuantumReasoner(
        num_entities  = kg.num_entities,
        num_relations = kg.num_relations,
        embed_dim     = embed_dim,
        unitary_type  = "diagonal",
        max_paths     = 8,
        max_hops      = 2,
        ablation_mode = "full",
    ).to(device)

    # Upgrade to KG-specific relational unitary (New Contribution #2)
    if use_relational_unitary:
        model.unitary = RelationalDecomposedUnitary(
            num_relations  = kg.num_relations,
            complex_dim    = embed_dim // 2,
            symmetric_rels = structure["symmetric_rels"],
            inverse_pairs  = structure["inverse_pairs"],
            sym_reg_weight = 0.01,
            inv_reg_weight = 0.01,
        ).to(device)
        log.info("Using RelationalDecomposedUnitary (V2 KG-specific)")

    # Log path cache coverage
    train_pairs = [(kg.entity2id[t.head], kg.entity2id[t.tail]) for t in kg.train_triples]
    coverage    = path_cache.coverage(train_pairs)
    log.info(f"Path cache: {path_cache.summary()} | coverage={coverage:.1%}")

    # Build TrainerV2
    trainer = TrainerV2(
        model              = model,
        train_loader       = train_dl,
        val_loader         = val_dl,
        device             = device,
        lr_base            = lr_base,
        lr_imag            = lr_base * lr_imag_mult,
        lr_phase           = lr_base * lr_phase_mult,
        grad_clip          = 0.5,
        epochs             = epochs,
        warmup_epochs      = min(10, epochs // 5),
        use_interference_loss = True,
        interference_loss_kwargs = {
            "label_smoothing":    0.1,
            "phase_weight":       phase_weight,
            "contrast_weight":    contrast_weight,
            "reg_encoder_weight": 0.01,
            "reg_unitary_weight": 0.01,
        },
        interference_check_every = 10,
        toy_kg             = kg,
        checkpoint_dir     = str(ROOT / "outputs" / "checkpoints"),
        run_name           = f"{run_name}_quantum_v2",
        true_tails         = true_tails,
    )

    log.info(
        f"V2 training: lr_base={lr_base}, "
        f"lr_imag={lr_base*lr_imag_mult:.4f} ({lr_imag_mult:.1f}×), "
        f"lr_phase={lr_base*lr_phase_mult:.4f} ({lr_phase_mult:.1f}×)"
    )
    log.info(
        f"Loss: contrast_weight={contrast_weight} (high=better for 7-wrong-vs-4-correct), "
        f"phase_weight={phase_weight}"
    )

    best_val = trainer.train()
    test_results = trainer.load_best_and_evaluate(test_dl)

    log.info(f"QuantumReasoner V2 test: {test_results}")

    # Interference monitor final summary
    if trainer.monitor:
        summary = trainer.monitor.final_summary()
        log.info(
            f"Interference summary: "
            f"destructive={summary.get('best_destructive_count',0)}/3, "
            f"final_healthy={summary.get('final_healthy', False)}, "
            f"trend={summary.get('interference_trend',{}).get('trend','?')}"
        )

    return model, test_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="V2 experiment: TrainerV2 + InterferenceAwareLoss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",   type=str, default="toy",
                        choices=["toy", "fb15k237", "wn18rr"],
                        help="Dataset to use.")
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--embed_dim", type=int,   default=16)
    parser.add_argument("--lr_base",   type=float, default=0.005)
    parser.add_argument("--lr_imag_mult", type=float, default=3.0,
                        help="lr_imag = lr_base × this. Default 3.0.")
    parser.add_argument("--lr_phase_mult", type=float, default=2.0)
    parser.add_argument("--contrast_weight", type=float, default=1.0,
                        help="ContrastiveInterferenceLoss weight. Higher = stronger fix for 27%% problem.")
    parser.add_argument("--phase_weight", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--quick",     action="store_true",
                        help="20 epochs, 2 noise levels — for fast testing.")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip baselines (train quantum only, faster).")
    parser.add_argument("--no_ablation", action="store_true",
                        help="Skip ablation study.")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--compare",   type=str,   default="",
                        help="Path to V1 CSV for comparison (e.g., outputs/results/v1_comparison.csv).")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20

    set_seed(args.seed)
    device = get_device("cuda")

    console.print(Panel(
        f"[bold blue]V2 Experiment[/bold blue]\n"
        f"Dataset: {args.dataset} | Epochs: {args.epochs} | Device: {device}\n"
        f"lr_base={args.lr_base} | lr_imag={args.lr_imag_mult}× | "
        f"contrast_weight={args.contrast_weight}",
        border_style="blue",
        title="[bold]quantum_kg V2[/bold]",
    ))

    # ── Build data ─────────────────────────────────────────────────────────────
    if args.dataset == "toy":
        kg         = build_toy_kg(seed=args.seed)
        train_dl, val_dl, test_dl = build_dataloaders(
            kg, batch_size=8, num_negatives=4
        )
        true_tails = build_true_tails_dict(kg)
        log.info(kg.summary())

        # Pre-compute path cache (toy KG: ~0.1s)
        path_cache = build_training_cache(
            kg,
            cache_dir = ROOT / "data" / "cache",
            max_hops  = 2,
            max_paths = 8,
        )

    else:
        from data.download import load_entity_relation_maps, build_adjacency_from_file
        from data.dataset import KGDataset, collate_fn
        from torch.utils.data import DataLoader

        raw_dir = ROOT / "data" / "raw" / args.dataset
        if not raw_dir.exists():
            console.print(
                f"[red]Dataset not found: {raw_dir}[/red]\n"
                f"Run: python data/download.py --dataset {args.dataset}"
            )
            return 1

        entity2id, relation2id = load_entity_relation_maps(raw_dir)

        train_ds = KGDataset.from_text_file(
            str(raw_dir / "train.txt"), entity2id, relation2id,
            all_triples_path=str(raw_dir / "train.txt"),
            num_negatives=128, mode="train",
        )
        val_ds = KGDataset.from_text_file(
            str(raw_dir / "valid.txt"), entity2id, relation2id,
            all_triples_path=str(raw_dir / "train.txt"),
            num_negatives=0, mode="eval",
        )
        test_ds = KGDataset.from_text_file(
            str(raw_dir / "test.txt"), entity2id, relation2id,
            all_triples_path=str(raw_dir / "train.txt"),
            num_negatives=0, mode="eval",
        )

        train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True,
                              collate_fn=collate_fn, num_workers=(0 if __import__("sys").platform == "win32" else 2), pin_memory=(__import__("sys").platform != "win32"))
        val_dl   = DataLoader(val_ds,   batch_size=256,  shuffle=False,
                              collate_fn=collate_fn, num_workers=(0 if __import__("sys").platform == "win32" else 2))
        test_dl  = DataLoader(test_ds,  batch_size=256,  shuffle=False,
                              collate_fn=collate_fn, num_workers=(0 if __import__("sys").platform == "win32" else 2))

        true_tails = dict(train_ds.true_tails)

        # Build adjacency for path cache
        adj = build_adjacency_from_file(
            str(raw_dir / "train.txt"), entity2id, relation2id
        )
        train_pairs = [(h, t) for h, r, t in train_ds.triples]

        cache_path = ROOT / "data" / "cache" / f"{args.dataset}_hops2_paths8.pkl"
        path_cache = PathCache.load_or_build(
            cache_path   = cache_path,
            adjacency    = adj,
            triple_pairs = train_pairs,
            num_entities = len(entity2id),
            max_hops     = 2,
            max_paths    = 8,
        )

        class SimpleKG:
            num_entities  = len(entity2id)
            num_relations = len(relation2id)
            def triple_to_ids(self, t): return t
            @property
            def triples(self): return []
            @property
            def contradiction_queries(self): return []
        kg = SimpleKG()
        args.embed_dim = 256
        args.lr_base   = 0.001
        log.info(f"Dataset: {args.dataset} — {len(entity2id):,} entities, "
                 f"{len(relation2id):,} relations")

    # ── Prepare output directories ─────────────────────────────────────────────
    results_dir = ROOT / "outputs" / "results"
    figures_dir = ROOT / "outputs" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"v2_{args.dataset}"

    # ── Train QuantumReasoner with V2 protocol ────────────────────────────────
    t0 = time.time()
    quantum_model, quantum_results = train_quantum_v2(
        kg               = kg,
        train_dl         = train_dl,
        val_dl           = val_dl,
        test_dl          = test_dl,
        true_tails       = true_tails,
        device           = device,
        epochs           = args.epochs,
        embed_dim        = args.embed_dim,
        run_name         = run_name,
        path_cache       = path_cache,
        lr_base          = args.lr_base,
        lr_imag_mult     = args.lr_imag_mult,
        lr_phase_mult    = args.lr_phase_mult,
        contrast_weight  = args.contrast_weight,
        phase_weight     = args.phase_weight,
    )
    quantum_time = time.time() - t0

    # ── Train baselines ────────────────────────────────────────────────────────
    all_results: dict[str, MetricResults] = {
        "QuantumReasoner [V2]": quantum_results,
    }

    if not args.no_baselines:
        baseline_configs = [
            (TransE,    "TransE",    "margin",            {"num_entities": kg.num_entities, "num_relations": kg.num_relations, "embed_dim": args.embed_dim}),
            (RotatE,    "RotatE",    "self_adversarial",  {"num_entities": kg.num_entities, "num_relations": kg.num_relations, "embed_dim": args.embed_dim}),
            (ComplEx,   "ComplEx",   "bce",               {"num_entities": kg.num_entities, "num_relations": kg.num_relations, "embed_dim": args.embed_dim}),
        ]
        for model_class, name, loss_type, kwargs in baseline_configs:
            result = train_baseline(
                model_class  = model_class,
                model_kwargs = kwargs,
                loss_type    = loss_type,
                kg           = kg,
                train_dl     = train_dl,
                val_dl       = val_dl,
                test_dl      = test_dl,
                true_tails   = true_tails,
                device       = device,
                epochs       = args.epochs,
                lr           = args.lr_base,
                run_name     = run_name,
                display_name = name,
            )
            all_results[name] = result

    total_time = time.time() - t0

    # ── Save V2 results CSV ────────────────────────────────────────────────────
    v2_csv = results_dir / "v2_comparison.csv"
    with open(v2_csv, "w", newline="") as f:
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
    log.info(f"V2 results saved: {v2_csv}")

    # ── Theorem 8.3 verification ───────────────────────────────────────────────
    thm_results = []
    if args.dataset == "toy" and hasattr(kg, "contradiction_queries"):
        log.print_banner("Theorem 8.3 Verification", color="yellow")
        thm_results = verify_theorem_conditions(quantum_model, kg, device)
        thm_csv = results_dir / "v2_theorem.csv"
        with open(thm_csv, "w", newline="") as f:
            fieldnames = ["query", "phi", "phi_gt_pi2", "applicable",
                          "gap_0pct", "gap_10pct", "gap_20pct"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in thm_results:
                writer.writerow({
                    "query":     r.query,
                    "phi":       f"{r.phi:.3f}",
                    "phi_gt_pi2": r.phi_satisfies_condition,
                    "applicable": r.theorem_applicable,
                    "gap_0pct":  f"{r.predicted_gap_0pct:.4f}",
                    "gap_10pct": f"{r.predicted_gap_10pct:.4f}",
                    "gap_20pct": f"{r.predicted_gap_20pct:.4f}",
                })
        log.info(f"Theorem results: {sum(1 for r in thm_results if r.theorem_applicable)}"
                 f"/{len(thm_results)} applicable")

    # ── Ablation study ─────────────────────────────────────────────────────────
    ablation_results = []
    if not args.no_ablation and args.dataset == "toy":
        log.print_banner("V2 Ablation Study", color="yellow")
        noise_levels = NOISE_LEVELS if not args.quick else [0.0, 0.10, 0.20]

        runner = AblationRunner(kg, device, run_name=f"ablation_{run_name}")
        ablation_results = runner.run_all(
            modes        = STANDARD_ABLATIONS,
            noise_levels = noise_levels,
            epochs       = args.epochs,
            embed_dim    = args.embed_dim,
        )
        runner.print_ablation_table(ablation_results)
        runner.save_results_csv(ablation_results, str(results_dir / "v2_ablation.csv"))

    # ── Figure generation ──────────────────────────────────────────────────────
    if not args.no_figures and args.dataset == "toy":
        try:
            set_paper_style()
            import matplotlib.pyplot as plt

            # Figure 1 & 2: Phase diagram + interference decomposition
            saved = generate_all_paper_figures(
                model      = quantum_model,
                toy_kg     = kg,
                device     = device,
                output_dir = figures_dir,
            )
            for name, path in saved.items():
                log.info(f"Figure saved: {path}")

            # Figure 3: Noise degradation
            if ablation_results:
                noise_mrr: dict[str, dict] = {}
                for r in ablation_results:
                    label = r.config.mode
                    if label not in noise_mrr:
                        noise_mrr[label] = {}
                    noise_mrr[label][r.config.noise_rate] = r.metrics.mrr

                # Add baselines at noise levels if available
                # (run_noise_experiment would be needed for full comparison)
                fig = plot_noise_degradation(
                    noise_mrr,
                    save_path      = figures_dir / "v2_noise_degradation.pdf",
                    highlight_model= "full",
                )
                plt.close(fig)
                log.info(f"Noise degradation figure: {figures_dir / 'v2_noise_degradation.pdf'}")

        except Exception as e:
            log.warning(f"Figure generation failed (non-critical): {e}")

    # ── Print summary ──────────────────────────────────────────────────────────
    log.print_comparison_table(
        {k: v.to_dict() for k, v in all_results.items()},
        title="V2 Results Summary",
    )

    # Show V1 vs V2 comparison if V1 CSV available
    v1_csv = args.compare or str(results_dir / "v1_comparison.csv")
    if Path(v1_csv).exists():
        log.print_banner("V1 vs V2 Comparison", color="cyan")
        compare_versions(v1_csv=v1_csv, v2_csv=str(v2_csv))

    # ── Recommendations for next steps ────────────────────────────────────────
    thm_ok = sum(1 for r in thm_results if r.theorem_applicable)
    n_dest = 0
    if args.dataset == "toy":
        try:
            from training.interference_monitor import InterferenceMonitor
            monitor = InterferenceMonitor(
                model=quantum_model, toy_kg=kg, verbose=False
            )
            report = monitor.check(epoch=args.epochs)
            n_dest = report.num_destructive
        except Exception:
            pass

    console.print(Panel(
        f"[bold]V2 Complete[/bold] ({total_time/60:.1f} min)\n\n"
        f"Results: {v2_csv}\n"
        f"Interference emerged: {n_dest}/3 contradiction queries show destructive\n"
        f"Theorem 8.3 applicable: {thm_ok}/{len(thm_results)} queries\n\n"
        + (
            "[green]✓ Phase separation achieved. Proceed to V3.[/green]\n"
            "  python experiments/run_v3.py"
            if n_dest >= 2
            else
            "[yellow]⚠ Interference not fully emerged.[/yellow]\n"
            "  Try: python experiments/run_v2.py "
            f"--contrast_weight {args.contrast_weight*2:.1f} "
            f"--lr_imag_mult {args.lr_imag_mult+1:.0f}.0 "
            f"--epochs {args.epochs*2}"
        ),
        border_style="green" if n_dest >= 2 else "yellow",
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
