"""
experiments/run_wn18rr.py — Full WN18RR Benchmark Experiment

PURPOSE:
    WN18RR-specific experimental run. Critically different from FB15k-237 in
    three ways that affect experimental setup:

    1. Entity count: 40,943 (vs 14,541) — REQUIRES ChunkedEvaluator.
       Without chunking, score_triple_vs_all() produces a (B, 40943) float32 matrix.
       At batch_size=64: 10.5MB per batch → CUDA OOM on most GPUs.

    2. Relation count: 11 (vs 237) — very few, sparse relation coverage.
       RelationalDecomposedUnitary has fewer parameters but may benefit more
       from HierarchyAwareUnitary since WordNet has explicit hypernym hierarchy.

    3. Graph topology: tree-like hierarchy (hypernym/hyponym chains).
       BFS paths are longer and more structured than FB15k-237.
       max_hops=3 recommended (vs 2 for FB15k-237).
       Path cache will be larger: ~25MB with K=3, max_paths=8.

WORDNET RELATION SEMANTICS:
    WN18RR has 11 relations:
        _hypernym, _hyponym             — class hierarchy (inverse pair)
        _member_meronym, _member_holonym — part-whole (inverse pair)
        _synset_domain_topic_of         — topical grouping
        _has_part, _part_of             — physical composition (inverse pair)
        _derivationally_related_form    — morphological (symmetric)
        _also_see                       — semantic association (symmetric)
        _similar_to                     — similarity (symmetric)
        _verb_group                     — verb clustering (symmetric)

    The HierarchyAwareUnitary is strongly motivated here:
        _hypernym subPropertyOf _synset_domain_topic_of (in some ontologies)
        _hyponym ≈ inverse(_hypernym)
    infer_relation_structure() should detect all inverse pairs automatically.

KEY EXPECTED RESULTS (literature baselines):
    TransE:          MRR 0.226, Hits@10 0.501
    RotatE:          MRR 0.476, Hits@10 0.571
    ComplEx:         MRR 0.480, Hits@10 0.572
    NBFNet:          MRR 0.551, Hits@10 0.666
    QuantumReasoner: Target MRR 0.490-0.530 on clean data,
                     outperform NBFNet on noisy data (>10% corruption)

USAGE:
    python experiments/run_wn18rr.py
    python experiments/run_wn18rr.py --quick_mode
    python experiments/run_wn18rr.py --unitary hierarchy   # WordNet-optimized
    python experiments/run_wn18rr.py --chunk_size 2000     # OOM fix override
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
from models.quantum_reasoner import QuantumReasoner
from models.components.kg_unitary import (
    infer_relation_structure, RelationalDecomposedUnitary,
    HierarchyAwareUnitary, build_kg_unitary,
)
from models.baselines.transe import TransE
from models.baselines.rotate import RotatE
from models.baselines.complex_e import ComplEx
from training.trainer import Trainer
from training.trainer_v2 import TrainerV2
from evaluation.metrics import RankingMetrics
from evaluation.chunked_evaluator import ChunkedEvaluator
from evaluation.ablation import AblationRunner
from visualization.phase_plots import plot_noise_degradation, set_paper_style
from utils.seed import set_seed, get_device
from utils.logger import RichLogger

console = Console()
log     = RichLogger("run_wn18rr")

DATA_DIR  = Path("data/raw/wn18rr")
CACHE_DIR = Path("outputs/data/cache")
CKPT_DIR  = Path("outputs/checkpoints")
RES_DIR   = Path("outputs/results")
FIG_DIR   = Path("outputs/figures")

# WN18RR WordNet hierarchy — manually specified for HierarchyAwareUnitary
# (r_child, r_parent) pairs from WordNet ontology structure
WN18RR_HIERARCHY = [
    # hypernym is a specialization of synset_domain_topic_of
    # (in some WordNet schema interpretations)
    # These can be left empty and infer_relation_structure() will populate
    # automatically based on empirical co-occurrence patterns.
]

# Known inverse pairs in WN18RR (for RelationalDecomposedUnitary)
WN18RR_KNOWN_INVERSES_NAMES = [
    ("_hypernym",         "_hyponym"),
    ("_member_meronym",   "_member_holonym"),
    ("_has_part",         "_part_of"),
]

# Known symmetric relations in WN18RR
WN18RR_SYMMETRIC_NAMES = [
    "_derivationally_related_form",
    "_also_see",
    "_similar_to",
    "_verb_group",
]


def ensure_dirs():
    for d in [DATA_DIR, CACHE_DIR, CKPT_DIR, RES_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(args) -> tuple:
    """Download WN18RR, build vocab, resolve relation IDs, cache paths."""
    log.print_banner("Phase 1 — WN18RR Data Preparation", color="cyan")

    if not (DATA_DIR / "train.txt").exists():
        log.info("Downloading WN18RR...")
        download_dataset("wn18rr", Path("data/raw"))
    else:
        log.info("WN18RR already downloaded.")

    entity2id, relation2id = load_entity_relation_maps(DATA_DIR)
    n_ent = len(entity2id)
    n_rel = len(relation2id)

    log.info(f"WN18RR: {n_ent:,} entities, {n_rel} relations")
    log.info("Relations: " + str(list(relation2id.keys())))

    # Resolve known inverse/symmetric relation names to IDs
    known_inverses = [
        (relation2id[r1], relation2id[r2])
        for r1, r2 in WN18RR_KNOWN_INVERSES_NAMES
        if r1 in relation2id and r2 in relation2id
    ]
    known_symmetric = {
        relation2id[r]
        for r in WN18RR_SYMMETRIC_NAMES
        if r in relation2id
    }
    log.info(f"Known inverse pairs: {len(known_inverses)}")
    log.info(f"Known symmetric rels: {len(known_symmetric)}")

    # Build datasets
    all_triple_ids = set()
    splits = {}
    for split_name, fname in [("train","train.txt"),("val","valid.txt"),("test","test.txt")]:
        ds = KGDataset.from_text_file(
            DATA_DIR / fname, entity2id, relation2id,
            mode          = "train" if split_name == "train" else "eval",
            num_negatives = args.neg_samples,
        )
        splits[split_name] = ds
        all_triple_ids.update(ds.triples)
        log.info(f"  {split_name}: {len(ds):,} triples")

    # Rebuild true-triple lookups
    for ds in splits.values():
        ds.true_triple_set = all_triple_ids
        ds.true_tails = {}
        ds.true_heads = {}
        for h, r, t in all_triple_ids:
            ds.true_tails.setdefault((h, r), set()).add(t)
            ds.true_heads.setdefault((r, t), set()).add(h)

    true_tails = splits["train"].true_tails

    # Path cache — WN18RR needs max_hops=3 due to deep hypernym chains
    cache_path = CACHE_DIR / f"wn18rr_hops{args.max_hops}_paths{args.max_paths}.pkl"
    log.info(f"Building WN18RR path cache (max_hops={args.max_hops}) ...")
    log.info("This takes ~20-45 minutes the first time. Go get coffee.")

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
        force_rebuild = args.rebuild_cache,
    )

    coverage = cache.coverage(train_pairs)
    log.info(f"Cache coverage: {coverage:.1%}")
    if coverage < 0.85:
        log.warning(
            f"Low coverage ({coverage:.0%}). Many WN18RR paths may be >3 hops. "
            "Consider increasing max_hops to 4 (much slower cache build)."
        )

    return entity2id, relation2id, splits, true_tails, cache, known_inverses, known_symmetric


# ─────────────────────────────────────────────────────────────────────────────
#  UNITARY SELECTION (WN18RR-optimized)
# ─────────────────────────────────────────────────────────────────────────────

def build_wn18rr_unitary(
    n_rel:           int,
    complex_dim:     int,
    unitary_type:    str,
    structure:       dict,
    known_inverses:  list,
    known_symmetric: set,
    relation2id:     dict,
):
    """
    Build the WN18RR-optimized unitary operator.

    For WN18RR, HierarchyAwareUnitary is the strongest choice because:
        - WordNet IS an explicit ontological hierarchy (hypernym chains)
        - The hierarchy structure is known and can be encoded directly
        - Hierarchy regularization should improve generalization on unseen paths

    RelationalDecomposedUnitary is second-best:
        - Encodes inverse pairs (hypernym/hyponym) explicitly
        - Encodes symmetry for derivationally_related_form, similar_to, etc.

    DiagonalUnitary is the baseline — no structural constraints.
    """
    if unitary_type == "hierarchy":
        # Build hierarchy pairs: (child_rel_id, parent_rel_id)
        # In WN18RR: _hypernym is "more specific" than _synset_domain_topic_of
        hierarchy_pairs = []

        # Add any additional hierarchy from infer_relation_structure
        # For WN18RR, use known pairs + empirically detected ones
        combined_hierarchy = hierarchy_pairs   # extend if needed

        unitary = HierarchyAwareUnitary(
            num_relations      = n_rel,
            complex_dim        = complex_dim,
            relation_hierarchy = combined_hierarchy,
            hierarchy_weight   = 0.005,
        )
        log.info(f"Using HierarchyAwareUnitary with {len(combined_hierarchy)} hierarchy pairs")
        return unitary

    elif unitary_type == "relational":
        # Use known structure + auto-detected
        all_symmetric = known_symmetric | structure.get("symmetric_rels", set())
        all_inverses  = list(known_inverses) + structure.get("inverse_pairs", [])
        # Deduplicate
        seen_pairs = set()
        dedup_inv  = []
        for pair in all_inverses:
            key = tuple(sorted(pair))
            if key not in seen_pairs:
                seen_pairs.add(key)
                dedup_inv.append(pair)

        unitary = RelationalDecomposedUnitary(
            num_relations   = n_rel,
            complex_dim     = complex_dim,
            symmetric_rels  = all_symmetric,
            inverse_pairs   = dedup_inv,
        )
        log.info(
            f"Using RelationalDecomposedUnitary: "
            f"{len(all_symmetric)} symmetric, {len(dedup_inv)} inverse pairs"
        )
        return unitary

    else:
        # Default diagonal
        from models.components.unitary_operators import DiagonalUnitary
        return DiagonalUnitary(n_rel, complex_dim)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name:    str,
    model,
    splits:        dict,
    true_tails:    dict,
    n_ent:         int,
    args,
    device,
    resume_from:   str = "",
) -> dict:
    """Train one model and evaluate with ChunkedEvaluator (OOM-safe)."""
    if model is None:
        return {}

    log.print_banner(f"Training {model_name} on WN18RR", color="yellow")
    log.warning(
        "REMINDER: WN18RR evaluation REQUIRES ChunkedEvaluator (40,943 entities). "
        "Do NOT use score_triple_vs_all() directly — it will OOM."
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(splits["train"], batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(splits["val"],   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(splits["test"],  batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    run_name = f"wn18rr_{model_name}"

    # Use TrainerV2 for QuantumReasoner, Trainer for baselines
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
            warmup_epochs      = 25,
            use_interference_loss = True,
            interference_loss_kwargs = {
                "label_smoothing": 0.9,
                "phase_weight":    0.1,
                "contrast_weight": 0.5,
                "reg_encoder_weight": 0.01,
            },
            interference_check_every = 20,
            checkpoint_dir     = str(CKPT_DIR),
            run_name           = run_name,
            true_tails         = true_tails,
            patience           = args.patience,
            resume_from        = resume_from or None,
        )
    else:
        loss_map = {
            "transe":    ("margin",   {"margin": 9.0}),
            "rotate":    ("self_adversarial", {"margin": 24.0}),
            "complex_e": ("bce",      {"label_smoothing": 0.1}),
        }
        lt, lk = loss_map.get(model_name, ("bce", {}))
        trainer = Trainer(
            model          = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            device         = device,
            lr             = args.lr,
            loss_type      = lt,
            loss_kwargs    = lk,
            grad_clip      = 1.0,
            epochs         = args.epochs,
            checkpoint_dir = str(CKPT_DIR),
            run_name       = run_name,
            true_tails     = true_tails,
            patience       = args.patience,
            resume_from    = resume_from or None,
        )

    trainer.train()

    # Load best checkpoint
    best_ckpt = CKPT_DIR / run_name / "best.pt"
    if best_ckpt.exists():
        from utils.checkpoint import CheckpointManager
        ckpt = CheckpointManager(str(CKPT_DIR / run_name))
        ckpt.load_best(model, device=device)

    # CHUNKED EVALUATION — required for WN18RR
    chunk_size = args.chunk_size if args.chunk_size > 0 else "auto"
    evaluator = ChunkedEvaluator(
        model        = model,
        num_entities = n_ent,
        device       = device,
        chunk_size   = chunk_size,
        true_tails   = true_tails,
        batch_size   = args.batch_size,
        verbose      = True,
    )
    model.eval()

    log.info(f"Running chunked test evaluation (chunk_size={chunk_size}) ...")
    t0      = time.time()
    results = evaluator.evaluate_loader(test_loader)
    elapsed = time.time() - t0
    log.info(f"Evaluation complete in {elapsed:.1f}s")
    log.print_metrics(results.to_dict(), title=f"{model_name} WN18RR Test Results")

    return {
        "model_name":  model_name,
        "dataset":     "WN18RR",
        "mrr":         results.mrr,
        "hits@1":      results.hits_at_1,
        "hits@3":      results.hits_at_3,
        "hits@10":     results.hits_at_10,
        "num_triples": results.num_triples,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full WN18RR experimental run")
    parser.add_argument("--embed_dim",     type=int,   default=256)
    parser.add_argument("--epochs",        type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=0.0003)
    parser.add_argument("--batch_size",    type=int,   default=1024)
    parser.add_argument("--neg_samples",   type=int,   default=512)
    parser.add_argument("--max_hops",      type=int,   default=3,
                        help="3 recommended for WN18RR hypernym chains")
    parser.add_argument("--max_paths",     type=int,   default=8)
    parser.add_argument("--chunk_size",    type=int,   default=0,
                        help="ChunkedEvaluator chunk size. 0 = auto from VRAM.")
    parser.add_argument("--unitary",       type=str,   default="hierarchy",
                        choices=["diagonal", "relational", "hierarchy"],
                        help="'hierarchy' recommended for WN18RR WordNet structure")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--patience",       type=int,   default=50,
                        help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--resume_from",    type=str,   default="",
                        help="Path to checkpoint .pt file to resume training from")
    parser.add_argument("--quick_mode",    action="store_true")
    parser.add_argument("--skip_baselines",action="store_true")
    parser.add_argument("--skip_noise",    action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--noise_levels",  type=str, default="0.0,0.05,0.10,0.15,0.20")
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
        f"[bold cyan]WN18RR Experiment[/]\n\n"
        f"  embed_dim={args.embed_dim}  epochs={args.epochs}\n"
        f"  max_hops={args.max_hops}   unitary={args.unitary}\n"
        f"  chunk_size={'auto' if args.chunk_size==0 else args.chunk_size}\n"
        f"  [yellow]IMPORTANT: ChunkedEvaluator active for 40,943 entity eval[/]",
        border_style="cyan"
    ))

    # Phase 1: Data
    (entity2id, relation2id, splits, true_tails,
     cache, known_inverses, known_symmetric) = prepare_data(args)

    n_ent = len(entity2id)
    n_rel = len(relation2id)

    # Auto-detect additional structure
    structure = infer_relation_structure(
        set(splits["train"].triples), n_rel,
        symmetry_threshold = 0.80,
        inverse_threshold  = 0.70,
    )
    log.info(f"Auto-detected: {len(structure['symmetric_rels'])} symmetric, "
             f"{len(structure['inverse_pairs'])} inverse pairs")

    # Phase 2: Models and training
    models_to_run = (
        ["quantum_reasoner"] if args.skip_baselines
        else ["quantum_reasoner", "transe", "rotate", "complex_e"]
    )

    all_results = []

    for model_name in models_to_run:
        if model_name == "quantum_reasoner":
            base_model = QuantumReasoner(
                num_entities  = n_ent,
                num_relations = n_rel,
                embed_dim     = args.embed_dim,
                unitary_type  = "diagonal",
                max_paths     = args.max_paths,
                max_hops      = args.max_hops,
                dropout       = 0.1,
            )
            # Upgrade to WN18RR-optimized unitary
            base_model.unitary = build_wn18rr_unitary(
                n_rel, args.embed_dim // 2, args.unitary,
                structure, known_inverses, known_symmetric, relation2id,
            )
        elif model_name == "transe":
            base_model = TransE(n_ent, n_rel, embed_dim=args.embed_dim)
        elif model_name == "rotate":
            base_model = RotatE(n_ent, n_rel, embed_dim=args.embed_dim * 2)
        elif model_name == "complex_e":
            base_model = ComplEx(n_ent, n_rel, embed_dim=args.embed_dim * 2)
        else:
            continue

        resume = getattr(args, "resume_from", "") if model_name == "quantum_reasoner" else ""
        result = train_and_evaluate(
            model_name, base_model, splits, true_tails, n_ent, args, device,
            resume_from=resume,
        )
        if result:
            all_results.append(result)

    # Save main results
    if all_results:
        csv_path = RES_DIR / "wn18rr_main_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Results saved: {csv_path}")

        t = Table(title="WN18RR Results (Paper Table 2)", show_header=True)
        t.add_column("Model"); t.add_column("MRR"); t.add_column("Hits@1")
        t.add_column("Hits@3"); t.add_column("Hits@10")
        for r in all_results:
            t.add_row(r["model_name"], f"{r['mrr']:.4f}", f"{r['hits@1']:.4f}",
                      f"{r['hits@3']:.4f}", f"{r['hits@10']:.4f}")
        console.print(t)

    console.print(Panel(
        "[bold green]WN18RR experiment complete.[/]\n\n"
        "Outputs:\n"
        "  outputs/results/wn18rr_main_results.csv",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
