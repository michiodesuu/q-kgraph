"""
experiments/run_baselines.py — Unified Baseline Runner

Trains all 9 baseline models on any supported dataset and saves a unified
CSV so every version's results can be compared on a common table.

MODELS
------
    1. TransE       — translation-based embedding (Bordes et al. 2013)
    2. RotatE       — complex rotation (Sun et al. 2019)
    3. ComplEx      — complex bilinear (Trouillon et al. 2016)
    4. RASCAL       — full bilinear RESCAL (Nickel et al. 2011)
    5. ConvE        — 2D convolutional (Dettmers et al. 2018)
    6. TuckER       — Tucker tensor factorization (Balazevic et al. 2019)
    7. GTransE      — confidence-scaled margin loss (Kertkeidkachorn et al.)
    8. NBFNet       — neural Bellman-Ford GNN (Zhu et al. 2021)
    9. RED-GNN      — relational digraph GNN (Zhang et al. 2022)

USAGE
-----
    # Toy KG (fast sanity check):
    python experiments/run_baselines.py --dataset toy --quick

    # Full FB15k-237 run:
    python experiments/run_baselines.py --dataset fb15k237

    # All real datasets (fb15k237, wn18rr, nell995, yago3_10, codex_m, codex_l):
    python experiments/run_baselines.py --all

    # Subset of models:
    python experiments/run_baselines.py --dataset toy --models transe rotate

    # Custom embed dim / epochs:
    python experiments/run_baselines.py --dataset toy --embed_dim 64 --epochs 50

OUTPUT
------
    outputs/results/baselines_{dataset}.csv   — per-model MRR/H@1/H@3/H@10/MeanRank
    Each row: model, dataset, MRR, Hits@1, Hits@3, Hits@10, MeanRank, Params, Time(s)

IMPLEMENTATION NOTES
--------------------
    All models are trained with the same Trainer loop (score_triple + margin/BCE loss).
    NBFNet and RED-GNN require set_graph() to be called once with the training
    edge index before any forward pass; this is done automatically.

    Loss function per model (follows each paper's recommendation):
        TransE     → margin loss (hinge)
        RotatE     → self-adversarial loss
        ComplEx    → binary cross-entropy
        RASCAL     → binary cross-entropy
        ConvE      → binary cross-entropy
        TuckER     → binary cross-entropy
        GTransE    → margin loss  (confidence_margin_loss used on NELL;
                                    for other datasets falls back to margin)
        NBFNet     → binary cross-entropy
        RED-GNN    → binary cross-entropy

    Model-specific param requirements:
        RotatE/ComplEx : embed_dim is forced to be even (rounded up if odd)
        ConvE          : embed_h=8, embed_w=8 for embed_dim=64 (8×8=64);
                         embed_h=14, embed_w=15 for embed_dim=210 (14×15=210)
        TuckER         : d_entity=d_relation=embed_dim
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import KGDataset, collate_fn, build_dataloaders
from data.toy_kg import build_toy_kg
from data.download import load_entity_relation_maps
from evaluation.metrics import RankingMetrics
from training.trainer import Trainer
from utils.seed import set_seed, get_device

# ── Baselines ─────────────────────────────────────────────────────────────────
from models.baselines.transe    import TransE
from models.baselines.rotate    import RotatE
from models.baselines.complex_e import ComplEx
from models.baselines.rascal    import RASCAL
from models.baselines.conve     import ConvE
from models.baselines.tucker    import TuckER
from models.baselines.gtranse   import GTransE
from models.baselines.nbfnet    import NBFNet
from models.baselines.red_gnn   import REDGNN

# ── Our models ────────────────────────────────────────────────────────────────
from models.quantum_reasoner import QuantumReasoner
from models.v8_reasoner      import V8CurvedManifoldReasoner

from collections import defaultdict


def _true_tails_from_list(triples: list[tuple[int, int, int]]) -> dict:
    """Build {(h, r): set(t)} from a list of (h, r, t) int tuples."""
    tt: dict = defaultdict(set)
    for h, r, t in triples:
        tt[(h, r)].add(t)
    return dict(tt)


# ── constants ─────────────────────────────────────────────────────────────────
ALL_MODELS = ["transe", "rotate", "complex", "rascal", "conve", "tucker",
              "gtranse", "nbfnet", "redgnn",
              "quantumreasoner_v7", "quantumreasoner_v8"]

DATASET_DIRS = {
    "fb15k237": ROOT / "data/raw/fb15k237",
    "wn18rr":   ROOT / "data/raw/wn18rr",
    "nell995":  ROOT / "data/raw/nell995",
    "yago3_10": ROOT / "data/raw/yago3_10",
    "codex_m":  ROOT / "data/raw/codex_m",
    "codex_l":  ROOT / "data/raw/codex_l",
}

RES_DIR  = ROOT / "outputs/results"
CKPT_DIR = ROOT / "outputs/checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_toy(args) -> tuple[DataLoader, DataLoader, DataLoader, int, int, list, dict]:
    """Build toy KG loaders. Returns (train_dl, val_dl, test_dl, N, R, train_triples, true_tails)."""
    kg = build_toy_kg(seed=args.seed)
    train_dl, val_dl, test_dl = build_dataloaders(
        kg,
        batch_size    = args.batch_size,
        num_negatives = args.neg_samples,
        seed          = args.seed,
    )
    train_triples = [
        (kg.entity2id[t.head], kg.relation2id[t.relation], kg.entity2id[t.tail])
        for t in kg.train_triples
    ]
    all_triples = [
        (kg.entity2id[t.head], kg.relation2id[t.relation], kg.entity2id[t.tail])
        for t in kg.triples
    ]
    true_tails = _true_tails_from_list(all_triples)
    return (train_dl, val_dl, test_dl,
            kg.num_entities, kg.num_relations,
            train_triples, true_tails)


def load_real(dataset: str, args) -> tuple[DataLoader, DataLoader, DataLoader, int, int, list, dict]:
    """Build real-dataset loaders. Returns same tuple as load_toy."""
    raw_dir = DATASET_DIRS[dataset]
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_dir}. "
            f"Run: python data/download.py --dataset {dataset}"
        )

    entity2id, relation2id = load_entity_relation_maps(raw_dir)
    num_entities  = len(entity2id)
    num_relations = len(relation2id)

    splits = {}
    for split_name, fname in [("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt")]:
        fpath = raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing {fpath}")
        splits[split_name] = KGDataset.from_text_file(
            str(fpath), entity2id, relation2id,
            mode         = "train" if split_name == "train" else "eval",
            num_negatives = args.neg_samples if split_name == "train" else 0,
        )

    make_dl = lambda ds, shuffle, bs: DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=0, collate_fn=collate_fn,
    )
    train_dl = make_dl(splits["train"], True,  args.batch_size)
    val_dl   = make_dl(splits["val"],   False, args.batch_size * 2)
    test_dl  = make_dl(splits["test"],  False, args.batch_size * 2)

    train_triples = splits["train"].triples
    all_triples   = (train_triples
                     + splits["val"].triples
                     + splits["test"].triples)
    true_tails = _true_tails_from_list(all_triples)

    return (train_dl, val_dl, test_dl,
            num_entities, num_relations,
            train_triples, true_tails)


# ─────────────────────────────────────────────────────────────────────────────
#  EDGE INDEX (for GNN models)
# ─────────────────────────────────────────────────────────────────────────────

def build_edge_index(
    train_triples: list[tuple[int, int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build COO edge index from training triples.
    Adds reverse edges with shifted relation IDs (2*r+1).

    Returns:
        edge_index: (2, 2E) on device
        edge_type:  (2E,) on device
    """
    srcs, dsts, rels = [], [], []
    for h, r, t in train_triples:
        srcs.append(h); dsts.append(t); rels.append(r)
        srcs.append(t); dsts.append(h); rels.append(r)  # symmetric edges

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
    edge_type  = torch.tensor(rels,         dtype=torch.long, device=device)
    return edge_index, edge_type


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _even(d: int) -> int:
    """Round up to nearest even number (required by RotatE/ComplEx)."""
    return d if d % 2 == 0 else d + 1


def _conve_dims(embed_dim: int) -> tuple[int, int]:
    """
    Find (embed_h, embed_w) such that embed_h * embed_w == embed_dim.
    Falls back to (8, 8) → 64 if exact factoring is impossible.
    Common: 64→(8,8), 128→(8,16), 200→(10,20), 256→(16,16).
    """
    import math
    for h in range(int(math.isqrt(embed_dim)), 0, -1):
        if embed_dim % h == 0:
            return h, embed_dim // h
    return 8, 8  # fallback


def build_model(
    name: str,
    num_entities:  int,
    num_relations: int,
    embed_dim:     int,
) -> tuple[nn.Module, str]:
    """
    Instantiate a named baseline model.

    Returns:
        (model, loss_type) where loss_type ∈ {"margin", "self_adversarial", "bce"}
    """
    d  = embed_dim
    de = _even(d)   # even embed_dim for RotatE/ComplEx

    if name == "transe":
        return TransE(num_entities, num_relations, embed_dim=d), "margin"

    if name == "rotate":
        return RotatE(num_entities, num_relations, embed_dim=de), "self_adversarial"

    if name == "complex":
        return ComplEx(num_entities, num_relations, embed_dim=de), "bce"

    if name == "rascal":
        return RASCAL(num_entities, num_relations, embed_dim=d), "bce"

    if name == "conve":
        eh, ew = _conve_dims(d)
        return ConvE(num_entities, num_relations,
                     embed_dim=eh * ew, embed_h=eh, embed_w=ew), "bce"

    if name == "tucker":
        return TuckER(num_entities, num_relations,
                      d_entity=d, d_relation=d), "bce"

    if name == "gtranse":
        return GTransE(num_entities, num_relations, embed_dim=d), "margin"

    if name == "nbfnet":
        return NBFNet(num_entities, num_relations, embed_dim=d), "bce"

    if name == "redgnn":
        return REDGNN(num_entities, num_relations, embed_dim=d), "bce"

    if name == "quantumreasoner_v7":
        # QuantumReasoner requires even embed_dim (complex_dim = embed_dim // 2)
        return QuantumReasoner(num_entities, num_relations, embed_dim=de), "bce"

    if name == "quantumreasoner_v8":
        # V8CurvedManifoldReasoner: full unitary parallel transport + adaptive curvature
        # embed_dim must be even (complex_dim = embed_dim // 2)
        return V8CurvedManifoldReasoner(num_entities, num_relations, embed_dim=de), "bce"

    raise ValueError(f"Unknown model: {name!r}. Choose from {ALL_MODELS}")


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    true_tails: dict,
) -> dict:
    """Filtered MRR/H@1/H@3/H@10/MeanRank on a DataLoader split."""
    metrics = RankingMetrics(filter_false_negatives=True)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            h = batch["positive"][:, 0].to(device)
            r = batch["positive"][:, 1].to(device)
            t = batch["positive"][:, 2].to(device)
            scores = model.score_triple_vs_all(h, r)  # (B, E)
            metrics.update(
                scores=scores, true_indices=t,
                head_ids=h, relation_ids=r,
                true_tails=true_tails,
            )

    result = metrics.compute()
    return {
        "mrr":       result.mrr,
        "hits_at_1": result.hits_at_1,
        "hits_at_3": result.hits_at_3,
        "hits_at_10": result.hits_at_10,
        "mean_rank": result.mean_rank,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PER-MODEL TRAINING + EVAL
# ─────────────────────────────────────────────────────────────────────────────

def run_model(
    model_name:    str,
    dataset:       str,
    train_dl:      DataLoader,
    val_dl:        DataLoader,
    test_dl:       DataLoader,
    num_entities:  int,
    num_relations: int,
    train_triples: list,
    true_tails:    dict,
    device:        torch.device,
    args,
) -> dict:
    """Train one model and return its test metrics + metadata."""
    print(f"\n  [{model_name.upper()}]  dataset={dataset}  "
          f"entities={num_entities}  relations={num_relations}")

    model, loss_type = build_model(
        model_name, num_entities, num_relations, args.embed_dim
    )
    model = model.to(device)

    # GNN models need the graph wired before any forward pass
    if model_name in ("nbfnet", "redgnn"):
        edge_index, edge_type = build_edge_index(train_triples, device)
        model.set_graph(edge_index, edge_type)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"    params={num_params:,}  loss={loss_type}  epochs={args.epochs}")

    t0 = time.time()
    trainer = Trainer(
        model          = model,
        train_loader   = train_dl,
        val_loader     = val_dl,
        device         = device,
        loss_type      = loss_type,
        lr             = args.lr,
        epochs         = args.epochs,
        checkpoint_dir = str(CKPT_DIR),
        run_name       = f"baseline_{model_name}_{dataset}",
        true_tails     = true_tails,
        patience       = 30,
        log_every_n    = max(1, args.epochs // 5),
        grad_clip      = 1.0,
    )
    best_val = trainer.train()
    train_time = time.time() - t0

    # Load best checkpoint and eval on test set
    test_metrics = evaluate(model, test_dl, device, true_tails)

    row = {
        "model":     model_name,
        "dataset":   dataset,
        "MRR":       round(test_metrics.get("mrr", 0.0), 4),
        "Hits@1":    round(test_metrics.get("hits_at_1", 0.0), 4),
        "Hits@3":    round(test_metrics.get("hits_at_3", 0.0), 4),
        "Hits@10":   round(test_metrics.get("hits_at_10", 0.0), 4),
        "MeanRank":  round(test_metrics.get("mean_rank", 0.0), 1),
        "Params":    num_params,
        "Time(s)":   round(train_time, 1),
        "ValMRR":    round(best_val.mrr, 4),
    }
    print(f"    → Test  MRR={row['MRR']:.4f}  H@1={row['Hits@1']:.4f}  "
          f"H@10={row['Hits@10']:.4f}  MR={row['MeanRank']:.1f}  "
          f"time={row['Time(s)']}s")
    return row


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset: str, models: list[str], device: torch.device, args) -> list[dict]:
    """Train all requested models on one dataset; return list of result rows."""
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset}")
    print(f"{'='*60}")

    if dataset == "toy":
        (train_dl, val_dl, test_dl,
         num_entities, num_relations,
         train_triples, true_tails) = load_toy(args)
    else:
        (train_dl, val_dl, test_dl,
         num_entities, num_relations,
         train_triples, true_tails) = load_real(dataset, args)

    rows = []
    for model_name in models:
        try:
            row = run_model(
                model_name, dataset,
                train_dl, val_dl, test_dl,
                num_entities, num_relations,
                train_triples, true_tails,
                device, args,
            )
            rows.append(row)
        except Exception as exc:
            print(f"    [SKIP] {model_name} failed: {exc}")
            rows.append({
                "model": model_name, "dataset": dataset,
                "MRR": "ERROR", "Hits@1": "ERROR", "Hits@3": "ERROR",
                "Hits@10": "ERROR", "MeanRank": "ERROR",
                "Params": 0, "Time(s)": 0, "ValMRR": "ERROR",
                "note": str(exc),
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = ["model", "dataset", "MRR", "Hits@1", "Hits@3", "Hits@10",
              "MeanRank", "Params", "Time(s)", "ValMRR"]


def save_csv(rows: list[dict], dataset: str) -> Path:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = RES_DIR / f"baselines_{dataset}.csv"

    # Read existing rows (append/update rather than overwrite)
    existing: list[dict] = []
    if out.exists():
        with open(out, newline="") as f:
            existing = list(csv.DictReader(f))

    # Replace rows with same (model, dataset) key
    key = lambda r: (r.get("model"), r.get("dataset"))
    new_keys = {key(r) for r in rows}
    kept = [r for r in existing if key(r) not in new_keys]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept + rows)

    print(f"\n  Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train all 9 baselines + QuantumReasoner V7/V8 and save a unified CSV."
    )
    p.add_argument(
        "--dataset", default="toy",
        choices=["toy", "fb15k237", "wn18rr", "nell995", "yago3_10",
                 "codex_m", "codex_l"],
        help="Dataset to run on. Use --all to run all real datasets.",
    )
    p.add_argument("--all",      action="store_true",
                   help="Run on all real datasets (fb15k237, wn18rr, nell995, "
                        "yago3_10, codex_m, codex_l).")
    p.add_argument("--quick",    action="store_true",
                   help="20 epochs — fast smoke test.")
    p.add_argument("--embed_dim", type=int, default=64,
                   help="Embedding dimension (default 64; RotatE/ComplEx auto-rounded to even).")
    p.add_argument("--epochs",   type=int, default=200,
                   help="Training epochs (overridden to 20 by --quick).")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--neg_samples", type=int, default=8,
                   help="Negative samples per positive (train only).")
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument(
        "--models", nargs="*", default=None,
        metavar="MODEL",
        help=f"Subset of models to run (default: all). Choices: {ALL_MODELS}",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.epochs = 20

    models = args.models or ALL_MODELS
    # validate model names
    for m in models:
        if m not in ALL_MODELS:
            print(f"Unknown model '{m}'. Valid choices: {ALL_MODELS}")
            sys.exit(1)

    datasets = (["fb15k237", "wn18rr", "nell995", "yago3_10", "codex_m", "codex_l"]
                if args.all else [args.dataset])

    set_seed(args.seed)
    device = get_device()
    print(f"\nrun_baselines: device={device}  embed_dim={args.embed_dim}  "
          f"epochs={args.epochs}  models={models}")

    all_rows: list[dict] = []
    for ds in datasets:
        try:
            rows = run_dataset(ds, models, device, args)
            all_rows.extend(rows)
            save_csv(rows, ds)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {ds}: {e}")

    # Combined CSV across all datasets run in this invocation
    if len(datasets) > 1 and all_rows:
        save_csv(all_rows, "combined")

    print("\nDone.")


if __name__ == "__main__":
    main()
