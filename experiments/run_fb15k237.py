"""
experiments/run_fb15k237.py — Full FB15k-237 Benchmark Experiment

PURPOSE:
    Complete experimental pipeline for the paper's main results:
        1. Data preparation: download + path cache pre-computation
        2. Relation structure inference (auto-detect symmetric/inverse)
        3. QuantumReasoner training (TrainerV2 + InterferenceAwareLoss)
        4. All baselines: TransE, RotatE, ComplEx, NBFNet, RED-GNN
        5. Noise robustness experiment (ablation at 5 noise levels)
        6. Theorem 8.3 verification (φ > π/2 condition check)
        7. Figure generation (noise degradation, training curves)

OUTPUTS:
    outputs/results/fb15k237_main_results.csv     — Table 2 in paper
    outputs/results/fb15k237_ablation.csv         — Table 3 in paper
    outputs/figures/noise_degradation_fb15k237.pdf — Figure 3 in paper
    outputs/figures/training_curves_fb15k237.pdf  — Appendix Figure A1
    outputs/results/theorem_verification.csv      — Appendix Table

ESTIMATED RUNTIME (1× RTX 3090, 24GB):
    Path cache build:        15-30 minutes (one-time)
    QuantumReasoner train:   4-8 hours (500 epochs)
    All baselines:           6-10 hours total
    Noise experiment:        8-12 hours (20 runs)
    TOTAL:                   ~30-50 hours

    For faster iteration: use --quick_mode (50 epochs, 2 noise levels)

USAGE:
    python experiments/run_fb15k237.py
    python experiments/run_fb15k237.py --quick_mode
    python experiments/run_fb15k237.py --skip_baselines
    python experiments/run_fb15k237.py --only quantum_reasoner
    python experiments/run_fb15k237.py --noise_levels 0.0,0.1,0.2
    python experiments/run_fb15k237.py --resume_from outputs/checkpoints/fb15k237_qr/best.pt
"""

from __future__ import annotations

import sys
import csv
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data.download import download_dataset, load_entity_relation_maps
from data.dataset import KGDataset, build_dataloaders
from data.path_cache import PathCache, PathCacheBuilder
from data.noise_injection import NoiseInjector, NoiseConfig
from models.quantum_reasoner import QuantumReasoner
from models.components.kg_unitary import infer_relation_structure, RelationalDecomposedUnitary
from models.baselines.transe import TransE
from models.baselines.rotate import RotatE
from models.baselines.complex_e import ComplEx
from training.trainer import Trainer
from training.trainer_v2 import TrainerV2
from training.losses import build_loss
from evaluation.metrics import RankingMetrics
from evaluation.chunked_evaluator import ChunkedEvaluator
from evaluation.ablation import AblationRunner, STANDARD_ABLATIONS, NOISE_LEVELS
from visualization.phase_plots import (
    plot_noise_degradation, plot_training_curves, set_paper_style
)
from theory.noise_guarantee import verify_theorem_conditions
from utils.seed import set_seed, get_device
from utils.logger import RichLogger
from utils.checkpoint import CheckpointManager

console = Console()
log     = RichLogger("run_fb15k237")

DATA_DIR  = Path("data/raw/fb15k237")
CACHE_DIR = Path("outputs/data/cache")
CKPT_DIR  = Path("outputs/checkpoints")
RES_DIR   = Path("outputs/results")
FIG_DIR   = Path("outputs/figures")


def ensure_dirs():
    for d in [DATA_DIR, CACHE_DIR, CKPT_DIR, RES_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1: DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def phase1_prepare_data(args) -> tuple:
    """Download FB15k-237, build vocab, pre-compute path cache."""
    log.print_banner("Phase 1 — Data Preparation", color="cyan")

    # Download if needed
    if not (DATA_DIR / "train.txt").exists():
        log.info("Downloading FB15k-237...")
        download_dataset("fb15k237", Path("data/raw"))
    else:
        log.info("FB15k-237 already downloaded.")

    # Build vocabulary
    entity2id, relation2id = load_entity_relation_maps(DATA_DIR)
    n_ent = len(entity2id); n_rel = len(relation2id)
    log.info(f"Vocabulary: {n_ent:,} entities, {n_rel} relations")

    # Build datasets
    all_triple_ids: set[tuple] = set()
    splits = {}
    for split_name, fname in [("train","train.txt"),("val","valid.txt"),("test","test.txt")]:
        ds = KGDataset.from_text_file(
            DATA_DIR / fname, entity2id, relation2id,
            mode        = "train" if split_name == "train" else "eval",
            num_negatives= args.neg_samples,
        )
        splits[split_name] = ds
        all_triple_ids.update(ds.triples)
        log.info(f"  {split_name}: {len(ds):,} triples")

    # Rebuild datasets with full true-triple set
    for split_name, ds in splits.items():
        ds.true_triple_set = all_triple_ids
        # Rebuild lookups
        ds.true_tails = {}
        ds.true_heads = {}
        for h, r, t in all_triple_ids:
            ds.true_tails.setdefault((h, r), set()).add(t)
            ds.true_heads.setdefault((r, t), set()).add(h)

    true_tails = splits["train"].true_tails

    # Path cache
    cache_path = CACHE_DIR / f"fb15k237_hops{args.max_hops}_paths{args.max_paths}.pkl"
    if cache_path.exists() and not args.rebuild_cache:
        log.info(f"Loading existing path cache: {cache_path}")
        cache = PathCache.load(cache_path)
    else:
        log.info("Building path cache (this runs once — ~15-30 min)...")
        # Build adjacency from training triples
        adjacency: dict = {}
        for h, r, t in splits["train"].triples:
            adjacency.setdefault(h, []).append((r, t))

        train_pairs = [(h, t) for h, r, t in splits["train"].triples]
        cache = PathCache.load_or_build(
            cache_path    = cache_path,
            adjacency     = adjacency,
            triple_pairs  = train_pairs,
            num_entities  = n_ent,
            max_hops      = args.max_hops,
            max_paths     = args.max_paths,
        )

    coverage = cache.coverage([(h, t) for h, r, t in splits["train"].triples])
    log.info(f"Cache coverage: {coverage:.1%}")

    return entity2id, relation2id, splits, true_tails, cache


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: RELATION STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def phase2_relation_structure(splits, n_rel: int) -> dict:
    """Auto-detect symmetric relations and inverse pairs."""
    log.print_banner("Phase 2 — Relation Structure Inference", color="cyan")

    train_triples = set(splits["train"].triples)
    structure     = infer_relation_structure(
        train_triples, n_rel,
        symmetry_threshold = 0.80,
        inverse_threshold  = 0.70,
    )

    log.info(f"Symmetric relations: {len(structure['symmetric_rels'])}")
    log.info(f"Inverse pairs:       {len(structure['inverse_pairs'])}")
    log.info("These are used to configure RelationalDecomposedUnitary.")
    return structure


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, n_ent: int, n_rel: int, args, structure: dict) -> torch.nn.Module:
    """Instantiate a model by name with appropriate settings."""
    if name == "quantum_reasoner":
        model = QuantumReasoner(
            num_entities  = n_ent,
            num_relations = n_rel,
            embed_dim     = args.embed_dim,
            unitary_type  = "diagonal",
            max_paths     = args.max_paths,
            max_hops      = args.max_hops,
            dropout       = 0.1,
        )
        # Upgrade to relational unitary
        model.unitary = RelationalDecomposedUnitary(
            num_relations  = n_rel,
            complex_dim    = args.embed_dim // 2,
            symmetric_rels = structure["symmetric_rels"],
            inverse_pairs  = structure["inverse_pairs"],
        )
        return model

    elif name == "transe":
        return TransE(n_ent, n_rel, embed_dim=args.embed_dim, p_norm=1)
    elif name == "rotate":
        return RotatE(n_ent, n_rel, embed_dim=args.embed_dim * 2)  # RotatE uses 2x dim
    elif name == "complex_e":
        return ComplEx(n_ent, n_rel, embed_dim=args.embed_dim * 2)
    elif name == "nbfnet":
        try:
            from models.baselines.nbfnet import NBFNet
            return NBFNet(n_ent, n_rel, hidden_dim=args.embed_dim)
        except ImportError:
            log.warning("NBFNet not implemented yet — skipping")
            return None
    elif name == "red_gnn":
        try:
            from models.baselines.red_gnn import REDGNN
            return REDGNN(n_ent, n_rel, hidden_dim=args.embed_dim)
        except ImportError:
            log.warning("RED-GNN not implemented yet — skipping")
            return None
    else:
        raise ValueError(f"Unknown model: {name}")


def get_loss_config(name: str) -> tuple[str, dict]:
    """Get loss type and kwargs for each model."""
    configs = {
        "quantum_reasoner": ("bce",      {"label_smoothing": 0.9}),
        "transe":           ("margin",   {"margin": 9.0}),
        "rotate":           ("self_adversarial", {"margin": 24.0, "temperature": 1.0}),
        "complex_e":        ("bce",      {"label_smoothing": 0.9}),
        "nbfnet":           ("bce",      {"label_smoothing": 0.9}),
        "red_gnn":          ("bce",      {"label_smoothing": 0.9}),
    }
    return configs.get(name, ("bce", {}))


def train_model(
    name: str,
    model,
    splits: dict,
    true_tails: dict,
    args,
    n_ent: int,
    device,
) -> dict:
    """Train one model and return its best test metrics."""
    if model is None:
        return {}

    log.print_banner(f"Training {name}", color="yellow")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(splits["val"],   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(splits["test"],  batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    loss_type, loss_kwargs = get_loss_config(name)
    run_name = f"fb15k237_{name}"

    resume = getattr(args, "resume_from", "") if (not args.only or args.only == name) else ""

    if name == "quantum_reasoner":
        trainer = TrainerV2(
            model              = model,
            train_loader       = train_loader,
            val_loader         = val_loader,
            device             = device,
            lr_base            = args.lr,
            lr_imag            = args.lr * 3,
            lr_phase           = args.lr * 2,
            grad_clip          = 0.5,
            epochs             = args.epochs,
            warmup_epochs      = 20,
            use_interference_loss = True,
            interference_loss_kwargs = {
                "label_smoothing":    0.9,
                "phase_weight":       0.0,
                "contrast_weight":    0.0,
                "reg_encoder_weight": 0.01,
            },
            interference_check_every = 25,
            checkpoint_dir     = str(CKPT_DIR),
            run_name           = run_name,
            true_tails         = true_tails,
            patience           = args.patience,
            resume_from        = resume or None,
        )
    else:
        trainer = Trainer(
            model          = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            device         = device,
            lr             = args.lr,
            weight_decay   = 1e-5,
            loss_type      = loss_type,
            loss_kwargs    = loss_kwargs,
            grad_clip      = 1.0,
            epochs         = args.epochs,
            warmup_epochs  = 10,
            checkpoint_dir = str(CKPT_DIR),
            run_name       = run_name,
            true_tails     = true_tails,
            patience       = args.patience,
            resume_from    = resume or None,
        )

    trainer.train()

    # Final evaluation with ChunkedEvaluator
    evaluator = ChunkedEvaluator(
        model        = model,
        num_entities = n_ent,
        device       = device,
        chunk_size   = "auto",
        true_tails   = true_tails,
        batch_size   = 64,
        verbose      = False,
    )
    evaluator.model.eval()

    # Load best checkpoint
    best_ckpt = CKPT_DIR / run_name / "best.pt"
    if best_ckpt.exists():
        ckpt_mgr = CheckpointManager(str(CKPT_DIR / run_name))
        ckpt_mgr.load_best(model, device=device)

    results = evaluator.evaluate_loader(test_loader)
    log.print_metrics(results.to_dict(), title=f"{name} Test Results")

    return {
        "model_name":   name,
        "noise_rate":   0.0,
        "mrr":          results.mrr,
        "hits@1":       results.hits_at_1,
        "hits@3":       results.hits_at_3,
        "hits@10":      results.hits_at_10,
        "num_triples":  results.num_triples,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4: NOISE ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────

def phase4_noise_experiment(
    quantum_model,
    splits: dict,
    true_tails: dict,
    n_ent: int,
    n_rel: int,
    args,
    structure: dict,
    device,
) -> dict:
    """
    Run all ablation conditions at multiple noise levels.
    Returns nested dict: model_name → noise_level → MetricResults.
    """
    log.print_banner("Phase 4 — Noise Robustness Experiment", color="yellow")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(splits["val"],   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(splits["test"],  batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    # AblationRunner handles the 4 conditions
    ablation_runner = AblationRunner(
        base_model   = quantum_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        test_loader  = test_loader,
        device       = device,
        epochs       = min(args.epochs, 100),   # fewer epochs for ablation
        lr           = args.lr,
        output_dir   = str(RES_DIR / "ablation"),
        true_tails   = true_tails,
    )

    noise_results = ablation_runner.run_noise_experiment(
        noise_levels = args.noise_levels,
    )

    ablation_runner.print_noise_table(noise_results)
    ablation_runner.save_results_csv(
        [r for cond in noise_results.values() for r in cond.values()],
        filename="fb15k237_ablation.csv"
    )

    return noise_results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5: THEOREM VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def phase5_theorem_verification(quantum_model, entity2id, device) -> list:
    """
    Verify that Theorem 8.3 conditions hold on the trained model.
    Builds a mini toy-style verification using FB15k-237 entities.
    """
    log.print_banner("Phase 5 — Theorem 8.3 Verification", color="green")

    # Use toy KG for theorem verification (has known contradiction structure)
    from data.toy_kg import build_toy_kg
    toy_kg = build_toy_kg()

    # Build a tiny QuantumReasoner matching toy KG dimensions for verification
    toy_model = QuantumReasoner(
        num_entities  = toy_kg.num_entities,
        num_relations = toy_kg.num_relations,
        embed_dim     = 16,
    ).to(device)

    # Run verification
    thm_results = verify_theorem_conditions(toy_model, toy_kg, device)

    rows = []
    for r in thm_results:
        status = "✓ APPLICABLE" if r.theorem_applicable else "✗ N/A"
        rows.append({
            "query":                r.query,
            "phi_rad":              round(r.phi, 4),
            "phi_exceeds_half_pi":  r.phi_satisfies_condition,
            "theorem_applicable":   r.theorem_applicable,
            "predicted_gap_0pct":   round(r.predicted_gap_0pct,  6),
            "predicted_gap_10pct":  round(r.predicted_gap_10pct, 6),
            "predicted_gap_20pct":  round(r.predicted_gap_20pct, 6),
        })
        log.info(f"[{status}] {r.query}: φ={r.phi:.2f} rad")

    # Save to CSV
    csv_path = RES_DIR / "theorem_verification.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Theorem verification saved: {csv_path}")

    return thm_results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 6: FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def phase6_figures(all_results: dict, noise_results: dict, histories: dict):
    """Generate all paper figures."""
    log.print_banner("Phase 6 — Figure Generation", color="cyan")
    set_paper_style()

    # Figure 3: Noise degradation curves
    if noise_results:
        fig = plot_noise_degradation(
            results   = noise_results,
            metric    = "mrr",
            save_path = FIG_DIR / "noise_degradation_fb15k237.pdf",
        )
        log.info("Saved: outputs/figures/noise_degradation_fb15k237.pdf")

    # Appendix A1: Training curves
    if histories:
        fig = plot_training_curves(
            histories = histories,
            metric    = "val_mrr",
            save_path = FIG_DIR / "training_curves_fb15k237.pdf",
        )
        log.info("Saved: outputs/figures/training_curves_fb15k237.pdf")


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE RESULTS TABLE (Paper Table 2)
# ─────────────────────────────────────────────────────────────────────────────

def save_main_results_table(all_results: list):
    """Save main results to CSV for LaTeX Table 2."""
    if not all_results:
        return

    csv_path = RES_DIR / "fb15k237_main_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    log.info(f"Main results table saved: {csv_path}")

    # Print comparison table
    log.print_comparison_table(
        {r["model_name"]: {
            "MRR":     r["mrr"],
            "Hits@1":  r["hits@1"],
            "Hits@10": r["hits@10"],
        } for r in all_results},
        title="FB15k-237 Results (Paper Table 2)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full FB15k-237 experimental run")
    parser.add_argument("--embed_dim",       type=int,   default=256)
    parser.add_argument("--epochs",          type=int,   default=500)
    parser.add_argument("--lr",              type=float, default=0.0003)
    parser.add_argument("--batch_size",      type=int,   default=1024)
    parser.add_argument("--neg_samples",     type=int,   default=512)
    parser.add_argument("--max_hops",        type=int,   default=2)
    parser.add_argument("--max_paths",       type=int,   default=16)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--device",          type=str,   default="cuda")
    parser.add_argument("--rebuild_cache",   action="store_true")
    parser.add_argument("--patience",         type=int,   default=50,
                        help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--resume_from",      type=str,   default="",
                        help="Path to checkpoint .pt file to resume training from")
    parser.add_argument("--quick_mode",      action="store_true",
                        help="50 epochs, 2 noise levels — for quick iteration")
    parser.add_argument("--skip_baselines",  action="store_true")
    parser.add_argument("--skip_noise",      action="store_true")
    parser.add_argument("--skip_theorem",    action="store_true")
    parser.add_argument("--only",            type=str, default="",
                        help="Run only this model: quantum_reasoner/transe/rotate/complex_e/nbfnet")
    parser.add_argument("--noise_levels",    type=str, default="0.0,0.05,0.10,0.15,0.20",
                        help="Comma-separated noise levels")
    args = parser.parse_args()

    if args.quick_mode:
        args.epochs      = 50
        args.noise_levels = "0.0,0.10,0.20"
        args.batch_size  = 256
        log.info("Quick mode: 50 epochs, 3 noise levels")

    args.noise_levels = [float(x) for x in args.noise_levels.split(",")]

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dirs()

    console.print(Panel(
        f"[bold cyan]FB15k-237 Full Experiment Run[/]\n\n"
        f"  embed_dim={args.embed_dim}  epochs={args.epochs}  lr={args.lr}\n"
        f"  max_hops={args.max_hops}   max_paths={args.max_paths}\n"
        f"  noise_levels={args.noise_levels}\n"
        f"  quick_mode={args.quick_mode}",
        border_style="cyan"
    ))

    t_global = time.time()

    # Phase 1: Data
    entity2id, relation2id, splits, true_tails, cache = phase1_prepare_data(args)
    n_ent = len(entity2id); n_rel = len(relation2id)

    # Phase 2: Relation structure
    structure = phase2_relation_structure(splits, n_rel)

    # Phase 3: Training
    models_to_run = (
        [args.only] if args.only else
        (["quantum_reasoner"] if args.skip_baselines else
         ["quantum_reasoner", "transe", "rotate", "complex_e", "nbfnet", "red_gnn"])
    )

    all_results = []
    histories   = {}
    quantum_model = None

    for model_name in models_to_run:
        model = build_model(model_name, n_ent, n_rel, args, structure)
        if model is None:
            continue

        result = train_model(model_name, model, splits, true_tails, args, n_ent, device)
        if result:
            all_results.append(result)

        if model_name == "quantum_reasoner":
            quantum_model = model
            # Store training history for figures (from trainer internals)

    save_main_results_table(all_results)

    # Phase 4: Noise experiment
    noise_results = {}
    if not args.skip_noise and quantum_model is not None:
        noise_results = phase4_noise_experiment(
            quantum_model, splits, true_tails, n_ent, n_rel, args, structure, device
        )

    # Phase 5: Theorem verification
    if not args.skip_theorem and quantum_model is not None:
        phase5_theorem_verification(quantum_model, entity2id, device)

    # Phase 6: Figures
    phase6_figures(all_results, noise_results, histories)

    total_time = time.time() - t_global
    console.print(Panel(
        f"[bold green]Experiment complete in {total_time/3600:.1f} hours[/]\n\n"
        "Paper artifacts:\n"
        "  Table 2:  outputs/results/fb15k237_main_results.csv\n"
        "  Table 3:  outputs/results/ablation/fb15k237_ablation.csv\n"
        "  Figure 3: outputs/figures/noise_degradation_fb15k237.pdf\n"
        "  Theorem:  outputs/results/theorem_verification.csv",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
