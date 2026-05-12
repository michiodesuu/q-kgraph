"""
experiments/train_toy.py — Toy KG Training to Fix the 27% Problem  [V2]

PURPOSE:
    Trains QuantumReasoner on the toy biology KG using TrainerV2 and
    InterferenceAwareLoss. The goal is to turn Phase Collapse into
    Phase Separation — making wrong-answer paths destructively interfere.

THE 27% PROBLEM:
    Pre-training: 4 correct paths, 7 contradictory paths, random phases.
    Born rule result: P(correct) ≈ 27% — lower than P(wrong).
    Root cause: 7 wrong-path amplitudes dominate the sum by sheer count.

THE FIX:
    ContrastiveInterferenceLoss (contrast_weight=1.0) directly penalizes
    P(wrong) > P(correct). After 50-70 epochs, wrong-answer queries
    should show interference < 0 (destructive).

EXPECTED MILESTONES:
    Epoch 10:  mean_imag_norm > 0.05 (imaginary components learning)
    Epoch 20:  PhaseSeparationLoss decreasing (phases diverging)
    Epoch 50:  num_destructive ≥ 1 (first query shows destructive pattern)
    Epoch 70+: num_destructive = 3/3 (all queries suppressed)

SUCCESS CRITERION:
    After training:
        P(Platypus → WarmBlooded) > P(Platypus → ColdBlooded)
        P(Bat → FourLimbs) > P(Bat → TwoLimbs)
        P(Whale → Lungs) > P(Whale → Gills)

USAGE:
    python experiments/train_toy.py               # standard 100 epochs
    python experiments/train_toy.py --epochs 200  # more epochs
    python experiments/train_toy.py --seed 43     # different initialization
    python experiments/train_toy.py --quick       # 15 epochs for testing
    python experiments/train_toy.py --verify_only # just check theorem conditions

OUTPUTS:
    outputs/checkpoints/toy_v2/best.pt            trained model weights
    outputs/results/toy_theorem_verification.csv  Theorem 8.3 check
    outputs/figures/toy_phase_diagram.pdf         paper Figure 1
    outputs/figures/toy_interference_decomp.pdf   paper Figure 2
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

from data.toy_kg     import build_toy_kg
from data.dataset    import build_dataloaders, build_true_tails_dict
from data.path_cache import build_training_cache

from models.quantum_reasoner      import QuantumReasoner
from models.components.kg_unitary import RelationalDecomposedUnitary, infer_relation_structure

from training.trainer_v2 import TrainerV2
from training.interference_monitor import InterferenceMonitor

from evaluation.metrics        import RankingMetrics
from theory.noise_guarantee    import (
    verify_theorem_conditions, print_verification_table,
)
from theory.noise_bound        import NoiseBoundAnalyzer
from visualization.phase_plots import set_paper_style, generate_all_paper_figures
from utils.seed      import set_seed, get_device
from utils.logger    import RichLogger
from utils.checkpoint import CheckpointManager

console = Console()
log     = RichLogger("train_toy")


def build_true_tails(kg) -> dict:
    """Build true_tails dict for filtered evaluation."""
    from collections import defaultdict
    true_tails: dict = defaultdict(set)
    for triple in kg.triples:
        if triple.is_contradiction:
            continue
        h_id = kg.entity2id[triple.head]
        r_id = kg.relation2id[triple.relation]
        t_id = kg.entity2id[triple.tail]
        true_tails[(h_id, r_id)].add(t_id)
    return dict(true_tails)


def run_post_training_analysis(
    model,
    kg,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """
    Post-training analysis: check interference pattern and generate figures.

    Returns dict with:
        n_destructive:     count of queries showing destructive interference on wrong answer
        n_correct_wins:    count of queries where P(correct) > P(wrong)
        thm_applicable:    count of queries where Theorem 8.3 conditions hold
        phase_diagram_path, decomp_path: figure paths
    """
    from models.components.path_aggregator import PathEnumerator

    log.print_banner("Post-Training Analysis", color="green")
    model.eval()

    adj        = kg.get_adjacency()
    enumerator = PathEnumerator(adj, max_hops=3, max_paths=8)

    results = []
    with torch.no_grad():
        for cq in kg.contradiction_queries:
            h_id     = kg.entity2id[cq["head"]]
            corr_id  = kg.entity2id[cq["correct_tail"]]
            wrong_id = kg.entity2id[cq["contradictory_tail"]]

            h_state    = model.encoder(torch.tensor([h_id],    device=device)).squeeze(0)
            corr_state = model.encoder(torch.tensor([corr_id], device=device)).squeeze(0)
            wrong_state= model.encoder(torch.tensor([wrong_id],device=device)).squeeze(0)

            corr_paths  = enumerator.find_paths(h_id, corr_id)
            wrong_paths = enumerator.find_paths(h_id, wrong_id)

            corr_a  = model.aggregator.compute_interference_terms(
                h_state, corr_state, corr_paths, model.unitary
            ) if corr_paths  else {}
            wrong_a = model.aggregator.compute_interference_terms(
                h_state, wrong_state, wrong_paths, model.unitary
            ) if wrong_paths else {}

            is_destructive = float(wrong_a.get("interference", 0)) < -1e-6
            correct_wins   = float(corr_a.get("total_probability", 0)) > \
                             float(wrong_a.get("total_probability", 0))

            results.append({
                "query":       cq["query"],
                "head":        cq["head"],
                "correct":     cq["correct_tail"],
                "wrong":       cq["contradictory_tail"],
                "P_correct":   float(corr_a.get("total_probability", 0)),
                "P_wrong":     float(wrong_a.get("total_probability", 0)),
                "interf_wrong":float(wrong_a.get("interference", 0)),
                "sign_wrong":  wrong_a.get("interference_sign", "none"),
                "destructive": is_destructive,
                "correct_wins":correct_wins,
            })

    n_destructive  = sum(1 for r in results if r["destructive"])
    n_correct_wins = sum(1 for r in results if r["correct_wins"])

    # Print results table
    console.print("\n[bold]Post-Training Interference Results:[/bold]")
    for r in results:
        dest_str = "[green]DESTRUCTIVE ✓[/green]" if r["destructive"] else "[red]NOT YET[/red]"
        win_str  = "[green]✓[/green]" if r["correct_wins"] else "[red]✗[/red]"
        console.print(
            f"  [bold]{r['head']:10s}[/bold] "
            f"P_c={r['P_correct']:.4f} {win_str} "
            f"P_w={r['P_wrong']:.4f} | "
            f"interference={r['interf_wrong']:+.4f} {dest_str}"
        )

    # Theorem verification
    thm_results = verify_theorem_conditions(model, kg, device)
    print_verification_table(thm_results)
    n_applicable = sum(1 for r in thm_results if r.theorem_applicable)

    # Save theorem CSV
    thm_csv = output_dir / "toy_theorem_verification.csv"
    thm_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(thm_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "phi", "phi_ok", "applicable",
            "gap_0pct", "gap_10pct", "gap_20pct",
        ])
        writer.writeheader()
        for r in thm_results:
            writer.writerow({
                "query":   r.query,
                "phi":     f"{r.phi:.4f}",
                "phi_ok":  r.phi_satisfies_condition,
                "applicable": r.theorem_applicable,
                "gap_0pct":  f"{r.predicted_gap_0pct:.4f}",
                "gap_10pct": f"{r.predicted_gap_10pct:.4f}",
                "gap_20pct": f"{r.predicted_gap_20pct:.4f}",
            })
    log.info(f"Theorem verification saved: {thm_csv}")

    # Print NoiseBoundAnalyzer summary for applicable queries
    for r in thm_results:
        if r.theorem_applicable:
            analyzer = NoiseBoundAnalyzer(
                r       = r.r_correct,
                K       = float(r.K_correct),
                phi     = r.phi,
                K_prime = float(r.K_wrong),
            )
            analyzer.print_summary()

    # Generate paper figures
    set_paper_style()
    fig_dir = output_dir.parent.parent / "outputs" / "figures"
    saved   = {}
    try:
        saved = generate_all_paper_figures(
            model      = model,
            toy_kg     = kg,
            device     = device,
            output_dir = fig_dir,
        )
        for name, path in saved.items():
            log.info(f"Figure: {path}")
    except Exception as e:
        log.warning(f"Figure generation failed: {e}")

    return {
        "n_destructive":    n_destructive,
        "n_correct_wins":   n_correct_wins,
        "thm_applicable":   n_applicable,
        "phase_diagram":    saved.get("phase_diagram"),
        "decomp":           saved.get("interference_decomp"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train QuantumReasoner on toy KG — fixes the 27% problem"
    )
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--lr_base",         type=float, default=0.005)
    parser.add_argument("--lr_imag_mult",    type=float, default=3.0)
    parser.add_argument("--lr_phase_mult",   type=float, default=2.0)
    parser.add_argument("--contrast_weight", type=float, default=1.0,
                        help="High value (1.0+) needed because 7 wrong vs 4 correct paths.")
    parser.add_argument("--phase_weight",    type=float, default=0.2)
    parser.add_argument("--embed_dim",       type=int,   default=16)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--run_name",        type=str,   default="toy_v2")
    parser.add_argument("--quick",           action="store_true",
                        help="15 epochs for fast testing.")
    parser.add_argument("--verify_only",     action="store_true",
                        help="Skip training, just verify theorem conditions on saved model.")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 15

    set_seed(args.seed)
    device     = get_device("cpu")
    ckpt_dir   = ROOT / "outputs" / "checkpoints" / args.run_name
    results_dir = ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Build KG ───────────────────────────────────────────────────────────────
    kg = build_toy_kg(seed=args.seed)
    log.info(kg.summary())

    # ── Detect relation structure for KG-specific unitary ──────────────────────
    triple_set = {kg.triple_to_ids(t) for t in kg.triples}
    structure  = infer_relation_structure(triple_set, kg.num_relations)
    log.info(
        f"Relation structure: {len(structure['symmetric_rels'])} symmetric, "
        f"{len(structure['inverse_pairs'])} inverse pairs"
    )

    # ── Build path cache ───────────────────────────────────────────────────────
    cache = build_training_cache(
        kg,
        cache_dir = ROOT / "data" / "cache",
        max_hops  = 2,
        max_paths = 8,
    )
    train_pairs = [(kg.entity2id[t.head], kg.entity2id[t.tail]) for t in kg.train_triples]
    log.info(f"Path cache: {cache.summary()} | coverage={cache.coverage(train_pairs):.1%}")

    # ── Build model ────────────────────────────────────────────────────────────
    model = QuantumReasoner(
        num_entities  = kg.num_entities,
        num_relations = kg.num_relations,
        embed_dim     = args.embed_dim,
        unitary_type  = "diagonal",
        max_paths     = 8,
        max_hops      = 2,
        dropout       = 0.0,
    ).to(device)

    # Upgrade to RelationalDecomposedUnitary (V2 contribution)
    model.unitary = RelationalDecomposedUnitary(
        num_relations  = kg.num_relations,
        complex_dim    = args.embed_dim // 2,
        symmetric_rels = structure["symmetric_rels"],
        inverse_pairs  = structure["inverse_pairs"],
    ).to(device)

    param_counts = model.get_param_count()
    log.info(f"Model: {param_counts}")

    # ── Handle verify_only mode ────────────────────────────────────────────────
    if args.verify_only:
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            log.info(f"Loading saved model: {best_path}")
            ckpt_manager = CheckpointManager(ckpt_dir)
            ckpt_manager.load_best(model, device=device)
        else:
            log.warning("No saved model found. Running analysis on untrained model.")
        analysis = run_post_training_analysis(model, kg, device, results_dir)
        return 0

    # ── Build DataLoaders ──────────────────────────────────────────────────────
    train_dl, val_dl, test_dl = build_dataloaders(
        kg,
        batch_size    = 8,
        num_negatives = 4,
        num_workers   = 0,
    )
    true_tails = build_true_tails(kg)

    # ── Print pre-training interference baseline ───────────────────────────────
    console.print("\n[bold yellow]Pre-Training Interference (27% problem):[/bold yellow]")
    pre_monitor = InterferenceMonitor(
        model=model, toy_kg=kg, verbose=True,
        healthy_after_epoch=999,   # suppress health alerts pre-training
    )
    pre_report = pre_monitor.check(epoch=0)
    console.print(
        f"[dim]This is the baseline before any training. "
        f"P(correct) < P(wrong) is expected at epoch 0.[/dim]\n"
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = TrainerV2(
        model               = model,
        train_loader        = train_dl,
        val_loader          = val_dl,
        device              = device,
        lr_base             = args.lr_base,
        lr_imag             = args.lr_base * args.lr_imag_mult,
        lr_phase            = args.lr_base * args.lr_phase_mult,
        grad_clip           = 0.5,
        epochs              = args.epochs,
        warmup_epochs       = min(10, args.epochs // 5),
        use_interference_loss = True,
        interference_loss_kwargs = {
            "label_smoothing":    0.1,
            "phase_weight":       args.phase_weight,
            "contrast_weight":    args.contrast_weight,
            "reg_encoder_weight": 0.01,
            "reg_unitary_weight": 0.01,
        },
        interference_check_every = 10,
        toy_kg              = kg,
        checkpoint_dir      = str(ROOT / "outputs" / "checkpoints"),
        run_name            = args.run_name,
        true_tails          = true_tails,
    )

    console.print(Panel(
        f"[bold blue]Starting V2 Training[/bold blue]\n"
        f"Epochs: {args.epochs} | lr_base={args.lr_base} | "
        f"lr_imag={args.lr_base*args.lr_imag_mult:.4f} ({args.lr_imag_mult:.1f}×) | "
        f"lr_phase={args.lr_base*args.lr_phase_mult:.4f} ({args.lr_phase_mult:.1f}×)\n"
        f"contrast_weight={args.contrast_weight} | phase_weight={args.phase_weight}\n\n"
        f"Monitor every 10 epochs for:\n"
        f"  • mean_imag_norm > 0.05 (not collapsing)\n"
        f"  • num_destructive ≥ 1 by epoch 50",
        border_style="blue",
    ))

    t0           = time.time()
    best_results = trainer.train()
    elapsed      = time.time() - t0

    # ── Test evaluation ────────────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(ckpt_dir)
    try:
        ckpt_mgr.load_best(model, device=device)
        log.info("Loaded best checkpoint for evaluation")
    except Exception:
        log.warning("No checkpoint to load; using current model weights")

    test_metrics = RankingMetrics(filter_false_negatives=True)
    model.eval()
    with torch.no_grad():
        for batch in test_dl:
            h = batch["positive"][:, 0].to(device)
            r = batch["positive"][:, 1].to(device)
            t = batch["positive"][:, 2].to(device)
            scores = model.score_triple_vs_all(h, r)
            test_metrics.update(scores, t, h, r, true_tails)
    test_results = test_metrics.compute()
    log.info(f"Test results: {test_results}")

    # ── Post-training analysis ─────────────────────────────────────────────────
    analysis = run_post_training_analysis(model, kg, device, results_dir)

    n_dest     = analysis["n_destructive"]
    n_wins     = analysis["n_correct_wins"]
    thm_ok     = analysis["thm_applicable"]
    int_summary = trainer.monitor.final_summary() if trainer.monitor else {}

    # ── Final verdict ──────────────────────────────────────────────────────────
    if n_dest == 3 and n_wins == 3:
        console.print(Panel(
            "[bold green]✓ TRAINING SUCCESSFUL — 27% PROBLEM FIXED[/bold green]\n\n"
            "All 3 contradiction queries show:\n"
            "  ✓ Destructive interference on wrong-answer paths\n"
            "  ✓ P(correct) > P(wrong)\n\n"
            f"Test MRR: {test_results.mrr:.4f} | Hits@10: {test_results.hits_at_10:.4f}\n"
            f"Theorem 8.3 conditions met: {thm_ok}/3 queries\n"
            f"Training time: {elapsed/60:.1f} min\n\n"
            "Next: python experiments/run_v2.py --quick\n"
            "Or:   python experiments/run_v3.py (full benchmark with modern baselines)",
            border_style="green",
        ))
        return 0

    elif n_dest >= 1:
        console.print(Panel(
            f"[bold yellow]PARTIAL SUCCESS — {n_dest}/3 queries show destructive interference[/bold yellow]\n\n"
            f"P(correct) > P(wrong): {n_wins}/3 queries\n"
            f"Test MRR: {test_results.mrr:.4f}\n\n"
            "Suggestions to achieve full success:\n"
            f"  --contrast_weight {args.contrast_weight*1.5:.1f}  (increase from {args.contrast_weight})\n"
            f"  --epochs {int(args.epochs * 1.5)}  (more training)\n"
            f"  --lr_imag_mult {args.lr_imag_mult+1:.0f}.0  (stronger imaginary drive)",
            border_style="yellow",
        ))
        return 0

    else:
        console.print(Panel(
            "[bold red]PHASE COLLAPSE — No destructive interference emerged[/bold red]\n\n"
            "The model trained but the interference mechanism did not activate.\n"
            f"Phase magnitude: {int_summary.get('final_phase_magnitude', 0):.4f} "
            f"(should be > 0.1)\n\n"
            "Aggressive interventions:\n"
            f"  --lr_imag_mult 5.0  (current: {args.lr_imag_mult})\n"
            f"  --contrast_weight 2.0  (current: {args.contrast_weight})\n"
            f"  --epochs {args.epochs * 2}  (more training)\n\n"
            "Check: outputs/logs/interference/ for per-epoch health reports",
            border_style="red",
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
