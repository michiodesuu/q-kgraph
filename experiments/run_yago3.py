"""
experiments/run_yago3.py — Full YAGO3-10 Benchmark Experiment

PURPOSE:
    Baseline and QuantumReasoner evaluation on YAGO3-10.
    YAGO3-10 is a large-scale factual KG scraped from Wikipedia/YAGO.

KEY DIFFERENCES from FB15k-237 / WN18RR:
    - 123,182 entities  → ChunkedEvaluator MANDATORY (would OOM otherwise)
    - 37 relations       → few but factual (isLocatedIn, playsFor, actedIn, ...)
    - 1,079,040 train triples → very large; use batch_size=2048
    - Flat factual structure  → no WordNet hierarchy; RelationalDecomposedUnitary preferred
    - Path cache at max_hops=2 (dense graph; 3-hop cache would take 6+ hours)

EXPECTED RESULTS (literature, filtered MRR):
    TransE:  MRR ~0.50,  H@10 ~0.67
    RotatE:  MRR ~0.495, H@10 ~0.670
    ComplEx: MRR ~0.59,  H@10 ~0.71
    QR V7:   Target MRR >= 0.40 (scalability test)

USAGE:
    python experiments/run_yago3.py
    python experiments/run_yago3.py --quick_mode
    python experiments/run_yago3.py --only rotate
    python experiments/run_yago3.py --skip_baselines
    python experiments/run_yago3.py --only quantum_reasoner
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

from data.download import load_entity_relation_maps
from data.dataset import KGDataset
from data.path_cache import PathCache
from models.quantum_reasoner import QuantumReasoner
from models.components.kg_unitary import infer_relation_structure, RelationalDecomposedUnitary
from models.baselines.transe import TransE
from models.baselines.rotate import RotatE
from models.baselines.complex_e import ComplEx
from training.trainer import Trainer
from training.trainer_v2 import TrainerV2
from evaluation.metrics import RankingMetrics
from evaluation.chunked_evaluator import ChunkedEvaluator
from utils.seed import set_seed, get_device
from utils.logger import RichLogger
from utils.checkpoint import CheckpointManager

console = Console()
log     = RichLogger("run_yago3")

DATA_DIR  = Path("data/raw/yago3_10")
CACHE_DIR = Path("outputs/data/cache")
CKPT_DIR  = Path("outputs/checkpoints")
RES_DIR   = Path("outputs/results")


def ensure_dirs() -> None:
    for d in [DATA_DIR, CACHE_DIR, CKPT_DIR, RES_DIR,
              Path("outputs/figures"), Path("outputs/logs")]:
        d.mkdir(parents=True, exist_ok=True)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(args) -> tuple:
    """Load YAGO3-10, build vocab, build splits, build path cache."""
    log.print_banner("Phase 1 — YAGO3-10 Data Preparation", color="cyan")

    if not (DATA_DIR / "train.txt").exists():
        log.error(
            f"YAGO3-10 data not found at {DATA_DIR}. "
            "Extract YAGO3-10.tar into data/raw/yago3_10/ first:\n"
            "  tar -xf YAGO3-10.tar -C data/raw/yago3_10/"
        )
        sys.exit(1)

    entity2id, relation2id = load_entity_relation_maps(DATA_DIR)
    n_ent = len(entity2id)
    n_rel = len(relation2id)
    log.info(f"YAGO3-10: {n_ent:,} entities, {n_rel} relations")
    log.info("Relations: " + str(list(relation2id.keys())))
    log.warning(
        f"IMPORTANT: {n_ent:,} entities — ChunkedEvaluator is MANDATORY. "
        "score_triple_vs_all() would require ~57GB RAM without chunking."
    )

    neg_samples = args.neg_samples
    all_triple_ids: set = set()
    splits: dict = {}

    for split_name, fname in [("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt")]:
        ds = KGDataset.from_text_file(
            DATA_DIR / fname,
            entity2id,
            relation2id,
            mode          = "train" if split_name == "train" else "eval",
            num_negatives = neg_samples,
        )
        splits[split_name] = ds
        all_triple_ids.update(ds.triples)
        log.info(f"  {split_name}: {len(ds):,} triples")

    # Rebuild true-triple lookups across all splits for filtered evaluation
    for ds in splits.values():
        ds.true_triple_set = all_triple_ids
        ds.true_tails = {}
        ds.true_heads = {}
        for h, r, t in all_triple_ids:
            ds.true_tails.setdefault((h, r), set()).add(t)
            ds.true_heads.setdefault((r, t), set()).add(h)

    true_tails = splits["train"].true_tails

    # Path cache — max_hops=2 for YAGO3-10 (dense graph; 3-hop cache ~6 hrs)
    cache_path = CACHE_DIR / f"yago3_10_hops{args.max_hops}_paths{args.max_paths}.pkl"
    adjacency: dict = {}
    for h, r, t in splits["train"].triples:
        adjacency.setdefault(h, []).append((r, t))
    train_pairs = [(h, t) for h, r, t in splits["train"].triples]

    if cache_path.exists() and not args.rebuild_cache:
        log.info(f"Loading path cache: {cache_path}")
        cache = PathCache.load(cache_path)
    else:
        log.info(
            f"Building YAGO3-10 path cache (max_hops={args.max_hops}, "
            f"max_paths={args.max_paths}) — first time ~2-6 hours. "
            "Consider running overnight."
        )
        cache = PathCache.load_or_build(
            cache_path    = cache_path,
            adjacency     = adjacency,
            triple_pairs  = train_pairs,
            num_entities  = n_ent,
            max_hops      = args.max_hops,
            max_paths     = args.max_paths,
        )

    coverage = cache.coverage(train_pairs)
    log.info(f"Cache coverage: {coverage:.1%}")

    return entity2id, relation2id, splits, true_tails, cache, n_ent, n_rel


# ── Model construction ────────────────────────────────────────────────────────

def build_model(name: str, n_ent: int, n_rel: int, args, structure: dict):
    """Instantiate model by name."""
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
        # Upgrade to RelationalDecomposedUnitary (no hierarchy for YAGO3-10)
        model.unitary = RelationalDecomposedUnitary(
            num_relations  = n_rel,
            complex_dim    = args.embed_dim // 2,
            symmetric_rels = structure.get("symmetric_rels", set()),
            inverse_pairs  = structure.get("inverse_pairs", []),
        )
        return model

    elif name == "transe":
        return TransE(n_ent, n_rel, embed_dim=args.embed_dim, p_norm=1)
    elif name == "rotate":
        return RotatE(n_ent, n_rel, embed_dim=args.embed_dim * 2)
    elif name == "complex_e":
        return ComplEx(n_ent, n_rel, embed_dim=args.embed_dim * 2)
    else:
        raise ValueError(f"Unknown model: {name}")


# ── Training + evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    model,
    splits:     dict,
    true_tails: dict,
    n_ent:      int,
    args,
    device,
) -> dict:
    """Train one model and evaluate with ChunkedEvaluator (YAGO3-10 mandatory)."""
    if model is None:
        return {}

    log.print_banner(f"Training {model_name} on YAGO3-10", color="yellow")
    log.warning(
        f"YAGO3-10: {n_ent:,} entities — using ChunkedEvaluator. "
        "Do NOT call score_triple_vs_all() directly."
    )

    from torch.utils.data import DataLoader
    from data.dataset import collate_fn

    train_loader = DataLoader(
        splits["train"], batch_size=args.batch_size,
        shuffle=True, num_workers=(0 if __import__("sys").platform == "win32" else 2), pin_memory=(__import__("sys").platform != "win32"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        splits["val"], batch_size=64,
        shuffle=False, num_workers=(0 if __import__("sys").platform == "win32" else 2),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        splits["test"], batch_size=64,
        shuffle=False, num_workers=(0 if __import__("sys").platform == "win32" else 2),
        collate_fn=collate_fn,
    )

    run_name = f"yago3_{model_name}"
    ckpt_dir = str(CKPT_DIR / run_name)

    loss_map = {
        "quantum_reasoner": ("bce",            {"label_smoothing": 0.9}),
        "transe":           ("margin",          {"margin": 9.0}),
        "rotate":           ("self_adversarial",{"margin": 24.0, "temperature": 1.0}),
        "complex_e":        ("bce",             {"label_smoothing": 0.1}),
    }

    if model_name == "quantum_reasoner":
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
            checkpoint_dir     = ckpt_dir,
            run_name           = run_name,
            true_tails         = true_tails,
            patience           = args.patience,
        )
    else:
        lt, lk = loss_map.get(model_name, ("bce", {}))
        trainer = Trainer(
            model          = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            device         = device,
            lr             = args.lr,
            weight_decay   = 1e-5,
            loss_type      = lt,
            loss_kwargs    = lk,
            grad_clip      = 1.0,
            epochs         = args.epochs,
            warmup_epochs  = 10,
            checkpoint_dir = ckpt_dir,
            run_name       = run_name,
            true_tails     = true_tails,
            patience       = args.patience,
        )

    trainer.train()

    # Load best checkpoint
    best_ckpt = Path(ckpt_dir) / "best.pt"
    if best_ckpt.exists():
        ckpt = CheckpointManager(ckpt_dir)
        ckpt.load_best(model, device=device)

    # Evaluate — ChunkedEvaluator is mandatory for 123k entities
    chunk_size = args.chunk_size if args.chunk_size > 0 else "auto"
    evaluator = ChunkedEvaluator(
        model        = model,
        num_entities = n_ent,
        device       = device,
        chunk_size   = chunk_size,
        true_tails   = true_tails,
        batch_size   = 32,    # smaller batch for 123k entity scoring
        verbose      = True,
    )
    model.eval()

    log.info(f"Running chunked test evaluation (chunk_size={chunk_size}) ...")
    t0      = time.time()
    results = evaluator.evaluate_loader(test_loader)
    elapsed = time.time() - t0
    log.info(f"Evaluation complete in {elapsed:.1f}s")
    log.print_metrics(results.to_dict(), title=f"{model_name} YAGO3-10 Test Results")

    return {
        "model_name": model_name,
        "dataset":    "YAGO3-10",
        "mrr":        results.mrr,
        "hits@1":     results.hits_at_1,
        "hits@3":     results.hits_at_3,
        "hits@10":    results.hits_at_10,
        "num_triples": results.num_triples,
    }


# ── Results saving ────────────────────────────────────────────────────────────

def save_results(all_results: list) -> None:
    out_path = RES_DIR / "yago3_main_results.csv"
    fieldnames = ["model_name", "dataset", "mrr", "hits@1", "hits@3", "hits@10", "num_triples"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    log.info(f"Results saved: {out_path}")

    # Print summary table
    table = Table(title="YAGO3-10 Results", show_header=True)
    table.add_column("Model",   style="cyan")
    table.add_column("MRR",    justify="right")
    table.add_column("H@1",    justify="right")
    table.add_column("H@3",    justify="right")
    table.add_column("H@10",   justify="right")
    for r in all_results:
        table.add_row(
            r["model_name"],
            f"{r['mrr']:.4f}",
            f"{r['hits@1']:.4f}",
            f"{r['hits@3']:.4f}",
            f"{r['hits@10']:.4f}",
        )
    console.print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YAGO3-10 Benchmark Experiment")
    parser.add_argument("--embed_dim",     type=int,   default=256,
                        help="Embedding dimension (RotatE/ComplEx use 2x)")
    parser.add_argument("--epochs",        type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=0.0003)
    parser.add_argument("--batch_size",    type=int,   default=2048,
                        help="Large batch recommended for 1M+ training triples")
    parser.add_argument("--neg_samples",   type=int,   default=512)
    parser.add_argument("--max_hops",      type=int,   default=2,
                        help="2 recommended (3-hop cache takes 6+ hours for YAGO3-10)")
    parser.add_argument("--max_paths",     type=int,   default=8)
    parser.add_argument("--chunk_size",    type=int,   default=0,
                        help="ChunkedEvaluator chunk size. 0 = auto.")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--patience",      type=int,   default=30,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--quick_mode",    action="store_true",
                        help="50 epochs, 1 noise level — quick iteration")
    parser.add_argument("--skip_baselines",action="store_true",
                        help="Skip TransE/RotatE/ComplEx; run QR only")
    parser.add_argument("--only",          type=str,   default="",
                        help="Run only this model: quantum_reasoner/transe/rotate/complex_e")
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()

    if args.quick_mode:
        args.epochs     = 50
        args.batch_size = 512
        log.info("Quick mode: 50 epochs, batch_size=512")

    set_seed(args.seed)
    device = get_device(args.device)
    ensure_dirs()

    console.print(Panel(
        f"[bold cyan]YAGO3-10 Benchmark[/]\n\n"
        f"  embed_dim={args.embed_dim}  epochs={args.epochs}  lr={args.lr}\n"
        f"  batch_size={args.batch_size}  neg_samples={args.neg_samples}\n"
        f"  max_hops={args.max_hops}   max_paths={args.max_paths}\n"
        f"  chunk_size={'auto' if args.chunk_size==0 else args.chunk_size}\n"
        f"  [yellow]MANDATORY: ChunkedEvaluator for 123,182 entities[/]",
        border_style="cyan"
    ))

    # Phase 1: Data
    (entity2id, relation2id, splits, true_tails,
     cache, n_ent, n_rel) = prepare_data(args)

    # Phase 2: Auto-detect relation structure
    structure = infer_relation_structure(
        set(splits["train"].triples), n_rel,
        symmetry_threshold = 0.80,
        inverse_threshold  = 0.70,
    )
    log.info(
        f"Auto-detected: {len(structure['symmetric_rels'])} symmetric, "
        f"{len(structure['inverse_pairs'])} inverse pairs"
    )

    # Phase 3: Choose models to run
    if args.only:
        models_to_run = [args.only]
    elif args.skip_baselines:
        models_to_run = ["quantum_reasoner"]
    else:
        models_to_run = ["transe", "rotate", "complex_e", "quantum_reasoner"]

    all_results = []
    t_global    = time.time()

    for model_name in models_to_run:
        log.info(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        try:
            model = build_model(model_name, n_ent, n_rel, args, structure)
        except Exception as e:
            log.warning(f"  Failed to build {model_name}: {e}")
            continue

        result = train_and_evaluate(
            model_name, model, splits, true_tails, n_ent, args, device
        )
        if result:
            all_results.append(result)

    # Save all results
    if all_results:
        save_results(all_results)

    elapsed_h = (time.time() - t_global) / 3600
    log.info(f"\nTotal runtime: {elapsed_h:.1f} hours")
    log.info("Results: outputs/results/yago3_main_results.csv")


if __name__ == "__main__":
    main()
