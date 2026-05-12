"""
experiments/run_v7.py — V7 QuantumReasoner: Full Benchmark Suite

DATASETS:
    toy       — 13 entities, deliberate contradictions (theory verification)
    fb15k237  — 14,541 entities, 237 relations (primary benchmark)
    wn18rr    — 40,943 entities, 11 relations (hierarchical WordNet)
    nell995   — 75,492 entities, 200 relations (multi-hop)
    yago3_10  — 123,182 entities, 37 relations (scalability)

USAGE:
    python experiments/run_v7.py --dataset toy
    python experiments/run_v7.py --dataset fb15k237 --epochs 300
    python experiments/run_v7.py --dataset fb15k237 --quick
    python experiments/run_v7.py --dataset wn18rr
    python experiments/run_v7.py --dataset nell995
    python experiments/run_v7.py --dataset yago3_10
    python experiments/run_v7.py --dataset all          # all in sequence
    python experiments/run_v7.py --toy_only             # backward compat
    python experiments/run_v7.py --verify_only
    python experiments/run_v7.py --compare
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Core imports ──────────────────────────────────────────────────────────────
from data.toy_kg import build_toy_kg
from models.v7_reasoner import V7QuantumReasoner, build_v7_model
from training.v7_training import V7Trainer
from training.v7_loss import V7Loss

try:
    from evaluation.metrics import RankingMetrics, MetricResults
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False

try:
    from data.path_cache import PathCache, PathCacheBuilder
    _HAS_CACHE = True
except ImportError:
    _HAS_CACHE = False

try:
    from evaluation.chunked_evaluator import ChunkedEvaluator
    _HAS_CHUNKED = True
except ImportError:
    _HAS_CHUNKED = False

try:
    from data.download import download_dataset, load_entity_relation_maps
    from data.dataset import KGDataset, collate_fn as kg_collate_fn
    from models.components.kg_unitary import infer_relation_structure
    from utils.checkpoint import CheckpointManager
    _HAS_REAL_DATA = True
except ImportError:
    _HAS_REAL_DATA = False
    kg_collate_fn = None

# ── Per-dataset configuration ─────────────────────────────────────────────────

DATASET_CONFIGS: dict[str, dict] = {
    "fb15k237": {
        "dataset_key": "fb15k237",
        "data_dir":    "data/raw/fb15k237",
        "max_hops":    2,
        "max_paths":   16,
        "default_embed_dim": 128,
        "default_bond_dim":  32,
        "default_batch_size": 512,
        "default_neg_samples": 128,
        "default_epochs": 300,
        "need_chunked": False,   # 14,541 entities — fits in VRAM
        "phase_schedule": (20, 80, 160),
        "lr": 5e-4,
        "expected_mrr_range": (0.30, 0.40),
    },
    "wn18rr": {
        "dataset_key": "wn18rr",
        "data_dir":    "data/raw/wn18rr",
        "max_hops":    3,
        "max_paths":   8,
        "default_embed_dim": 128,
        "default_bond_dim":  32,
        "default_batch_size": 512,
        "default_neg_samples": 128,
        "default_epochs": 300,
        "need_chunked": True,   # 40,943 entities — MUST chunk
        "phase_schedule": (20, 80, 160),
        "lr": 5e-4,
        "expected_mrr_range": (0.38, 0.50),
    },
    "nell995": {
        "dataset_key": "nell995",
        "data_dir":    "data/raw/nell995",
        "max_hops":    3,
        "max_paths":   8,
        "default_embed_dim": 128,
        "default_bond_dim":  32,
        "default_batch_size": 256,
        "default_neg_samples": 64,
        "default_epochs": 200,
        "need_chunked": True,   # 75,492 entities
        "phase_schedule": (15, 60, 120),
        "lr": 5e-4,
        "expected_mrr_range": (0.30, 0.45),
    },
    "yago3_10": {
        "dataset_key": "yago3_10",
        "data_dir":    "data/raw/yago3_10",
        "max_hops":    2,
        "max_paths":   8,
        "default_embed_dim": 128,
        "default_bond_dim":  32,
        "default_batch_size": 256,
        "default_neg_samples": 64,
        "default_epochs": 200,
        "need_chunked": True,   # 123,182 entities
        "phase_schedule": (15, 60, 120),
        "lr": 5e-4,
        "expected_mrr_range": (0.35, 0.50),
    },
}

CKPT_DIR = Path("outputs/checkpoints")
RES_DIR  = Path("outputs/results")
LOG_DIR  = Path("outputs/logs")
CACHE_DIR = Path("outputs/data/cache")


def ensure_dirs() -> None:
    for d in [CKPT_DIR, RES_DIR, LOG_DIR, CACHE_DIR,
              Path("outputs/figures"), Path("data/raw")]:
        d.mkdir(parents=True, exist_ok=True)


# ── Toy KG DataLoader ─────────────────────────────────────────────────────────

def _kg_to_dataloader(kg, batch_size: int = 8, neg_per_pos: int = 16, train: bool = True):
    """Convert ToyKG triples to a DataLoader with on-the-fly random negatives."""
    triples = kg.triples
    if not train:
        triples = triples[int(len(triples) * 0.8):]
    else:
        triples = triples[:int(len(triples) * 0.8)]

    heads, rels, tails = [], [], []
    for t in triples:
        h, r, t_id = kg.triple_to_ids(t)
        heads.append(h); rels.append(r); tails.append(t_id)

    heads = torch.tensor(heads)
    rels  = torch.tensor(rels)
    tails = torch.tensor(tails)
    B = len(heads)
    negatives = torch.randint(0, kg.num_entities, (B, neg_per_pos))

    ds = TensorDataset(heads, rels, tails, negatives)

    class _Collate:
        def __call__(self, batch):
            h, r, t, n = zip(*batch)
            return {
                "head":      torch.stack(h),
                "relation":  torch.stack(r),
                "tail":      torch.stack(t),
                "negatives": torch.stack(n),
            }

    return DataLoader(ds, batch_size=batch_size, shuffle=train, collate_fn=_Collate())


# ── Real dataset collate (KGDataset → V7Trainer format) ─────────────────────

def _v7_collate(batch: list) -> dict:
    """
    Convert KGDataset batch format to V7Trainer's expected batch format.

    KGDataset returns: {"positive": (3,), "negatives": (K, 3)}
    V7Trainer expects: {"head": (B,), "relation": (B,), "tail": (B,), "negatives": (B, K)}

    All negatives are tail-corrupted (corrupt_head_prob=0), so we extract tail IDs only.
    """
    positives = torch.stack([b["positive"] for b in batch])   # (B, 3)
    negs_list = [b["negatives"] for b in batch]               # list of (K, 3)

    K = negs_list[0].shape[0] if negs_list[0].numel() > 0 else 1
    if all(n.shape[0] == K for n in negs_list):
        negs = torch.stack(negs_list)   # (B, K, 3)
    else:
        # Pad to uniform K
        K_max = max(n.shape[0] for n in negs_list)
        padded = []
        for n in negs_list:
            if n.shape[0] < K_max:
                pad = n[-1:].expand(K_max - n.shape[0], -1)
                n = torch.cat([n, pad], dim=0)
            padded.append(n)
        negs = torch.stack(padded)   # (B, K_max, 3)

    # All negatives are tail-corrupted → extract tail column
    neg_tails = negs[:, :, 2]   # (B, K)

    return {
        "head":      positives[:, 0],
        "relation":  positives[:, 1],
        "tail":      positives[:, 2],
        "negatives": neg_tails,
    }


# ── V7 Trainer subclass for real datasets (ChunkedEvaluator) ────────────────

class _V7TrainerReal(V7Trainer):
    """
    V7Trainer variant for real datasets:
        - Uses ChunkedEvaluator for validation (avoids OOM on large entity sets)
        - Skips interference health-check (requires toy_kg)
    """

    def __init__(
        self,
        *args,
        n_ent:      int,
        true_tails: dict,
        chunk_size  = "auto",
        val_loader_for_eval: DataLoader = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._n_ent     = n_ent
        self._true_tails = true_tails
        self._chunk_size = chunk_size
        # Separate loader for ChunkedEvaluator (may differ from training val_loader)
        self._val_eval_loader = val_loader_for_eval or self.val_loader

    def _evaluate(self, loader: DataLoader):
        """Override: ChunkedEvaluator for large entity sets."""
        if not _HAS_CHUNKED or not _HAS_METRICS:
            return None
        evaluator = ChunkedEvaluator(
            model        = self.model,
            num_entities = self._n_ent,
            device       = self.device,
            chunk_size   = self._chunk_size,
            true_tails   = self._true_tails,
            batch_size   = 64,
            verbose      = False,
        )
        self.model.eval()
        try:
            return evaluator.evaluate_loader(self._val_eval_loader)
        except Exception as e:
            print(f"  [V7 eval warning] {e}")
            return None


# ── Toy KG evaluation ─────────────────────────────────────────────────────────

def _evaluate_v7(model, kg, device: torch.device) -> dict:
    """Full filtered evaluation on ToyKG test triples."""
    if not _HAS_METRICS:
        return {"mrr": 0.0, "h1": 0.0, "h3": 0.0, "h10": 0.0}

    model.eval()
    true_set = kg.get_true_set(include_contradictions=False)

    mrr_sum, h1, h3, h10, n = 0.0, 0, 0, 0, 0

    with torch.no_grad():
        test_triples = [t for t in kg.triples
                        if not t.is_contradiction][-max(1, len(kg.triples) // 5):]
        for triple in test_triples:
            h_id, r_id, t_id = kg.triple_to_ids(triple)
            h_t = torch.tensor([h_id], device=device)
            r_t = torch.tensor([r_id], device=device)
            scores = model.score_triple_vs_all(h_t, r_t).squeeze(0)   # (E,)

            for (hh, rr, tt) in true_set:
                if hh == h_id and rr == r_id and tt != t_id:
                    scores[tt] = -1e9

            true_score = scores[t_id]
            rank = (scores > true_score).sum().item() + 1

            mrr_sum += 1.0 / rank
            h1  += int(rank <= 1)
            h3  += int(rank <= 3)
            h10 += int(rank <= 10)
            n   += 1

    if n == 0:
        return {"mrr": 0.0, "h1": 0.0, "h3": 0.0, "h10": 0.0, "n": 0}
    return {
        "mrr": mrr_sum / n,
        "h1":  h1 / n,
        "h3":  h3 / n,
        "h10": h10 / n,
        "n":   n,
    }


# ── Theory verification (toy KG only) ────────────────────────────────────────

def _verify_v7_theory(model, kg, device: torch.device) -> dict:
    from models.components.path_aggregator import PathEnumerator
    adj = kg.get_adjacency()
    enumerator = PathEnumerator(adj, max_hops=2, max_paths=8)
    model.eval()

    report: dict = {
        "lemma_v51": {},
        "lindblad_trace": None,
        "mps_health": None,
        "sasaki_contextuality": None,
    }

    print(f"\n{'='*70}")
    print("[V7 THEORY VERIFICATION]")
    print(f"{'='*70}")

    print("\n[1] Lemma V5.1 — Pairwise destructive interference")
    all_pass = True
    with torch.no_grad():
        for cq in kg.contradiction_queries:
            h_id     = kg.entity2id[cq["head"]]
            wrong_id = kg.entity2id[cq["contradictory_tail"]]
            wrong_paths = enumerator.find_paths(h_id, wrong_id)
            if not wrong_paths:
                continue

            h = model.encoder(torch.tensor([h_id], device=device)).squeeze(0)
            w = model.encoder(torch.tensor([wrong_id], device=device)).squeeze(0)

            amps = []
            for path in wrong_paths[:8]:
                state = h.clone()
                for rel_id, _ in path:
                    r_t = torch.tensor([rel_id], device=device)
                    state = model.unitary.apply(state.unsqueeze(0), r_t).squeeze(0)
                amps.append((w.conj() * state).sum())

            destructive = 0
            total_pairs = 0
            for i in range(len(amps)):
                for j in range(i + 1, len(amps)):
                    cross = 2.0 * (amps[i] * amps[j].conj()).real
                    total_pairs += 1
                    if cross.item() < 0:
                        destructive += 1

            passes = destructive == total_pairs
            all_pass &= passes
            tag = "✓ HOLDS" if passes else "✗ NOT YET"
            report["lemma_v51"][cq["head"]] = {"destructive": destructive, "total": total_pairs}
            print(f"  [{tag}] {cq['head']}: {destructive}/{total_pairs} cross-terms destructive")

    print("\n[2] Lindblad trace preservation")
    if model._has_lindblad and kg.contradiction_queries:
        try:
            cq    = kg.contradiction_queries[0]
            h_id  = kg.entity2id[cq["head"]]
            h     = model.encoder(torch.tensor([h_id], device=device)).squeeze(0)
            rho   = torch.outer(h, h.conj())
            paths = enumerator.find_paths(h_id, kg.entity2id[cq["correct_tail"]])
            if paths:
                rel_ids   = [rel for rel, _ in paths[0]]
                rho_final = model.lindblad_integrator.integrate(rho, rel_ids, model.unitary, device)
                trace = rho_final.diagonal().real.sum().item()
                ok  = abs(trace - 1.0) < 0.05
                report["lindblad_trace"] = trace
                tag = "✓ HOLDS" if ok else "✗ DRIFT"
                print(f"  [{tag}] Tr(ρ_final) = {trace:.4f}  (target: 1.000)")
        except Exception as e:
            print(f"  [SKIP] Lindblad: {e}")
    else:
        print("  [SKIP] Lindblad not enabled or no paths")

    print("\n[3] MPS tensor entropy")
    if model._has_mps:
        try:
            entropy = model.mps_contractor.mps_entropy_loss()
            report["mps_health"] = float(entropy.item())
            print(f"  MPS entropy loss: {entropy.item():.4f}")
        except Exception as e:
            print(f"  [SKIP] MPS entropy: {e}")
    else:
        print("  [SKIP] MPS not enabled")

    print("\n[4] Sasaki contextuality")
    if model._has_sasaki and hasattr(model, "sasaki_layer"):
        try:
            cq      = kg.contradiction_queries[0]
            h_id    = kg.entity2id[cq["head"]]
            h_state = model.encoder(torch.tensor([h_id], device=device)).squeeze(0)
            n_concepts = len(model.sasaki_layer.projectors)
            if n_concepts >= 2:
                s_joint, s_fact = model.sasaki_layer.contextuality_score(h_state, 0, 1)
                gap = (s_joint - s_fact).abs().mean().item()
                contextual = gap > 0.01
                report["sasaki_contextuality"] = gap
                tag = "✓ CONTEXTUAL" if contextual else "✗ FACTORIZABLE"
                print(f"  [{tag}] |s_joint - s_factored| = {gap:.4f}  (need > 0.01)")
        except Exception as e:
            print(f"  [SKIP] Sasaki: {e}")
    else:
        print("  [SKIP] Sasaki not enabled")

    print(f"\n{'='*70}\n")
    return report


# ── Real dataset loading ──────────────────────────────────────────────────────

def _load_real_dataset(dataset_name: str, args) -> tuple:
    """
    Download (if needed), load vocab, build KGDatasets, build path cache.

    Returns: (entity2id, relation2id, splits, true_tails, cache)
    """
    if not _HAS_REAL_DATA:
        raise ImportError("Missing data/training infrastructure for real datasets.")

    cfg      = DATASET_CONFIGS[dataset_name]
    data_dir = Path(cfg["data_dir"])
    max_hops  = getattr(args, "max_hops", None) or cfg["max_hops"]
    max_paths = getattr(args, "max_paths", None) or cfg["max_paths"]

    # Download if needed
    if not (data_dir / "train.txt").exists():
        print(f"  Downloading {dataset_name}...")
        download_dataset(cfg["dataset_key"], Path("data/raw"))
    else:
        print(f"  {dataset_name} data: {data_dir}")

    entity2id, relation2id = load_entity_relation_maps(data_dir)
    n_ent = len(entity2id)
    n_rel = len(relation2id)
    print(f"  {dataset_name}: {n_ent:,} entities, {n_rel} relations")

    neg_samples = getattr(args, "neg_samples", None) or cfg["default_neg_samples"]

    all_triple_ids: set = set()
    splits: dict = {}
    for split_name, fname in [("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt")]:
        ds = KGDataset.from_text_file(
            data_dir / fname,
            entity2id,
            relation2id,
            mode          = "train" if split_name == "train" else "eval",
            num_negatives = neg_samples,
        )
        ds.corrupt_head_prob = 0.0   # all tail-corrupted → compatible with V7Trainer
        splits[split_name] = ds
        all_triple_ids.update(ds.triples)
        print(f"    {split_name}: {len(ds):,} triples")

    # Rebuild true-triple lookup across all splits (for filtered evaluation)
    for ds in splits.values():
        ds.true_triple_set = all_triple_ids
        ds.true_tails = {}
        ds.true_heads = {}
        for h, r, t in all_triple_ids:
            ds.true_tails.setdefault((h, r), set()).add(t)
            ds.true_heads.setdefault((r, t), set()).add(h)

    true_tails = splits["train"].true_tails

    # Path cache
    cache_path = CACHE_DIR / f"{dataset_name}_hops{max_hops}_paths{max_paths}.pkl"
    adjacency: dict = {}
    for h, r, t in splits["train"].triples:
        adjacency.setdefault(h, []).append((r, t))
    train_pairs = [(h, t) for h, r, t in splits["train"].triples]

    if cache_path.exists() and not getattr(args, "rebuild_cache", False):
        print(f"  Loading path cache: {cache_path}")
        cache = PathCache.load(cache_path)
    else:
        est = cfg.get("cache_build_minutes", 60)
        print(f"  Building path cache (max_hops={max_hops}) — first time ~{est} min ...")
        cache = PathCache.load_or_build(
            cache_path   = cache_path,
            adjacency    = adjacency,
            triple_pairs = train_pairs,
            num_entities = n_ent,
            max_hops     = max_hops,
            max_paths    = max_paths,
        )

    coverage = cache.coverage(train_pairs)
    print(f"  Cache coverage: {coverage:.1%}")

    return entity2id, relation2id, splits, true_tails, cache, n_ent, n_rel


# ── V7 model builder for real datasets ───────────────────────────────────────

def _build_v7_real(n_ent: int, n_rel: int, args, dataset_name: str) -> V7QuantumReasoner:
    """Build V7 model with dataset-appropriate dimensions."""
    cfg       = DATASET_CONFIGS[dataset_name]
    embed_dim = getattr(args, "embed_dim", None) or cfg["default_embed_dim"]
    bond_dim  = getattr(args, "bond_dim",  None) or cfg["default_bond_dim"]
    num_jumps    = getattr(args, "num_jumps",    4)
    num_concepts = getattr(args, "num_concepts", 8)
    # Use actual available device, not the requested one (handles CUDA unavailable)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Building V7 for {dataset_name}: embed={embed_dim} bond={bond_dim} "
          f"jumps={num_jumps} concepts={num_concepts}")

    # Minimal KG-like object that build_v7_model needs
    class _FakeKG:
        num_entities  = n_ent
        num_relations = n_rel

    model = build_v7_model(
        _FakeKG(),
        embed_dim    = embed_dim,
        bond_dim     = bond_dim,
        num_jumps    = num_jumps,
        num_concepts = num_concepts,
        device       = device_str,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Components: lindblad={model._has_lindblad} mps={model._has_mps} "
          f"sasaki={model._has_sasaki} ecc={model._has_ecc}")
    return model


# ── Run V7 on a real benchmark dataset ───────────────────────────────────────

def run_real_dataset_v7(dataset_name: str, args) -> dict:
    """
    Full V7 training + evaluation on a real benchmark dataset.

    Returns dict with MRR, H@1, H@3, H@10, dataset, n.
    """
    print("\n" + "=" * 70)
    print(f"[V7 on {dataset_name.upper()}]")
    print("=" * 70)

    if not _HAS_REAL_DATA:
        print("  ERROR: Real data infrastructure not available. "
              "Check imports: data.download, data.dataset, evaluation.chunked_evaluator")
        return {}

    cfg    = DATASET_CONFIGS[dataset_name]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device: {device}")

    t_start = time.time()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    (entity2id, relation2id, splits, true_tails,
     cache, n_ent, n_rel) = _load_real_dataset(dataset_name, args)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = _build_v7_real(n_ent, n_rel, args, dataset_name)
    model.to(device)

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    # Use larger batch_size for training (KGDataset already generates negatives)
    bs = getattr(args, "batch_size", None) or cfg["default_batch_size"]

    # Training: tail-only negatives, collated to V7Trainer format
    train_loader = DataLoader(
        splits["train"],
        batch_size   = bs,
        shuffle      = True,
        num_workers  = 2,
        pin_memory   = True,
        collate_fn   = _v7_collate,
    )
    # Validation/test: standard KGDataset collate for ChunkedEvaluator
    val_loader = DataLoader(
        splits["val"],
        batch_size  = 64,
        shuffle     = False,
        num_workers = 2,
        collate_fn  = kg_collate_fn,
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size  = 64,
        shuffle     = False,
        num_workers = 2,
        collate_fn  = kg_collate_fn,
    )

    # ── 4. Phase schedule ─────────────────────────────────────────────────────
    epochs  = getattr(args, "epochs", None) or cfg["default_epochs"]
    _p1, p2, p3 = cfg["phase_schedule"]
    # For real datasets: skip Phase 1 (syndrome-only) — no toy_kg contradiction queries.
    # Start directly in Phase 2 (rank + BCE + phase separation).
    p1 = 0
    # Scale phase boundaries proportionally if epochs changed from default
    default_epochs = cfg["default_epochs"]
    if epochs != default_epochs:
        scale = epochs / default_epochs
        p2 = max(10, int(p2 * scale))
        p3 = max(20, int(p3 * scale))

    lr = getattr(args, "lr", None) or cfg["lr"]
    run_name = f"{dataset_name}_v7"
    ckpt_dir = str(CKPT_DIR / run_name)

    need_chunked = cfg["need_chunked"]

    # ── 5. Trainer ────────────────────────────────────────────────────────────
    trainer = _V7TrainerReal(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        toy_kg           = None,       # syndrome loss disabled for real datasets
        path_cache       = cache,
        epochs           = epochs,
        lr_base          = lr,
        checkpoint_dir   = ckpt_dir,
        device           = str(device),
        phase1_end       = p1,
        phase2_end       = p2,
        phase3_end       = p3,
        check_every      = max(10, epochs // 20),
        polarity_weight  = 2.0,    # reduced (no toy_kg contradiction queries)
        polarity_margin  = 0.05,
        # ChunkedEvaluator params
        n_ent            = n_ent,
        true_tails       = true_tails,
        chunk_size       = "auto" if need_chunked else n_ent,
        val_loader_for_eval = val_loader,
    )

    print(f"\n  Training V7 on {dataset_name}: {epochs} epochs | "
          f"phases P1=0-{p1} P2={p1}-{p2} P3={p2}-{p3} P4={p3}-end")

    trainer.train()

    # ── 6. Load best checkpoint and final test evaluation ─────────────────────
    best_ckpt = Path(ckpt_dir) / "best.pt"
    if best_ckpt.exists():
        print("  Loading best checkpoint for final test evaluation...")
        ckpt_mgr = CheckpointManager(ckpt_dir)
        ckpt_mgr.load_best(model, device=device)

    print(f"\n  Running final test evaluation on {dataset_name}...")
    if _HAS_CHUNKED:
        evaluator = ChunkedEvaluator(
            model        = model,
            num_entities = n_ent,
            device       = device,
            chunk_size   = "auto" if need_chunked else n_ent,
            true_tails   = true_tails,
            batch_size   = 64,
            verbose      = True,
        )
        model.eval()
        test_result = evaluator.evaluate_loader(test_loader)
        result = {
            "dataset":  dataset_name,
            "model":    "V7 QuantumReasoner (Lindblad+MPS+Sasaki+ECC)",
            "mrr":      test_result.mrr,
            "hits@1":   test_result.hits_at_1,
            "hits@3":   test_result.hits_at_3,
            "hits@10":  test_result.hits_at_10,
            "n":        test_result.num_triples,
            "elapsed_h": round((time.time() - t_start) / 3600, 2),
        }
    else:
        print("  WARNING: ChunkedEvaluator not available, skipping test eval.")
        result = {"dataset": dataset_name, "model": "V7", "mrr": None}

    # ── 7. Save results ───────────────────────────────────────────────────────
    out_path = RES_DIR / f"v7_{dataset_name}_results.csv"
    fieldnames = ["dataset", "model", "mrr", "hits@1", "hits@3", "hits@10", "n", "elapsed_h"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in fieldnames})

    mrr_str = f"{result['mrr']:.4f}" if result.get("mrr") is not None else "N/A"
    print(f"\n  [{dataset_name.upper()}] V7 Test MRR: {mrr_str}")
    print(f"  Results saved: {out_path}")
    exp = cfg.get("expected_mrr_range", (None, None))
    if exp[0] is not None and result.get("mrr") is not None:
        in_range = exp[0] <= result["mrr"] <= exp[1]
        tag = "✓ in range" if in_range else "⚠ outside expected range"
        print(f"  Expected MRR {exp[0]}–{exp[1]}: {tag}")

    return result


# ── Toy KG experiment ─────────────────────────────────────────────────────────

def run_toy_kg_v7(args) -> dict:
    """Train and evaluate V7 on the Toy KG."""
    print("\n" + "=" * 70)
    print("[V7 Experiment: Toy KG]")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kg     = build_toy_kg()

    path_cache = None
    if _HAS_CACHE:
        cache_path = Path("data/cache/toy_kg_paths.pkl")
        try:
            path_cache = PathCache.load(cache_path)
            print(f"  Loaded path cache: {cache_path}")
        except Exception:
            path_cache = None

    model = build_v7_model(
        kg,
        embed_dim    = args.embed_dim,
        bond_dim     = args.bond_dim,
        num_jumps    = args.num_jumps,
        num_concepts = args.num_concepts,
        device       = str(device),
    )
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Components: lindblad={model._has_lindblad} mps={model._has_mps} "
          f"sasaki={model._has_sasaki} ecc={model._has_ecc}")

    train_loader = _kg_to_dataloader(kg, batch_size=args.batch_size, train=True)
    val_loader   = _kg_to_dataloader(kg, batch_size=args.batch_size, train=False)

    trainer = V7Trainer(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        toy_kg           = kg,
        path_cache       = path_cache,
        epochs           = args.epochs,
        lr_base          = args.lr,
        checkpoint_dir   = "outputs/checkpoints/v7_toy",
        device           = str(device),
        phase1_end       = max(5,  args.epochs // 10),
        phase2_end       = max(15, args.epochs // 4),
        phase3_end       = max(30, args.epochs // 2),
        check_every      = max(10, args.epochs // 10),
        polarity_weight  = 5.0,
        polarity_margin  = 0.1,
    )
    trainer.train()

    results = _evaluate_v7(model, kg, device)
    print(f"\n[V7 Toy KG Final Results]")
    print(f"  MRR:  {results['mrr']:.4f}")
    print(f"  H@1:  {results['h1']:.4f}")
    print(f"  H@3:  {results['h3']:.4f}")
    print(f"  H@10: {results['h10']:.4f}")
    print(f"  N:    {results.get('n', '?')}")

    _verify_v7_theory(model, kg, device)

    return {"model": model, "kg": kg, "device": device, "results": results}


# ── V5 vs V7 comparison ───────────────────────────────────────────────────────

def run_comparison_v5_v7(args) -> None:
    """Side-by-side V5 vs V7 comparison on Toy KG."""
    print("\n" + "=" * 70)
    print("[V7 Experiment: V5 vs V7 Comparison on Toy KG]")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kg     = build_toy_kg()
    results: dict = {}

    try:
        from models.quantum_reasoner import QuantumReasoner
        from training.v5_training import V5Trainer

        v5_model = QuantumReasoner(
            num_entities  = kg.num_entities,
            num_relations = kg.num_relations,
            embed_dim     = args.embed_dim,
            unitary_type  = "matrix_exp",
        ).to(device)

        train_loader = _kg_to_dataloader(kg, batch_size=args.batch_size, train=True)
        val_loader   = _kg_to_dataloader(kg, batch_size=args.batch_size, train=False)

        v5_trainer = V5Trainer(
            model        = v5_model,
            train_loader = train_loader,
            val_loader   = val_loader,
            toy_kg       = kg,
            epochs       = args.epochs,
            checkpoint_dir = "outputs/checkpoints/v5_compare",
            device       = str(device),
        )
        v5_trainer.train()
        results["v5"] = _evaluate_v7(v5_model, kg, device)
        print(f"  V5 MRR: {results['v5']['mrr']:.4f}  H@10: {results['v5']['h10']:.4f}")
    except Exception as e:
        print(f"  V5 training failed: {e}")
        results["v5"] = None

    v7_model = build_v7_model(
        kg,
        embed_dim    = args.embed_dim,
        bond_dim     = args.bond_dim,
        num_concepts = args.num_concepts,
        device       = str(device),
    )
    train_loader = _kg_to_dataloader(kg, batch_size=args.batch_size, train=True)
    val_loader   = _kg_to_dataloader(kg, batch_size=args.batch_size, train=False)

    v7_trainer = V7Trainer(
        model            = v7_model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        toy_kg           = kg,
        epochs           = args.epochs,
        checkpoint_dir   = "outputs/checkpoints/v7_compare",
        device           = str(device),
        phase1_end       = max(5, args.epochs // 10),
        phase2_end       = max(15, args.epochs // 4),
        phase3_end       = max(30, args.epochs // 2),
    )
    v7_trainer.train()
    results["v7"] = _evaluate_v7(v7_model, kg, device)
    print(f"  V7 MRR: {results['v7']['mrr']:.4f}  H@10: {results['v7']['h10']:.4f}")

    out_dir = RES_DIR
    rows = []
    for version, r in results.items():
        if r is None:
            continue
        rows.append({
            "model": f"V{version[-1]} QuantumReasoner (Toy KG)",
            "MRR":  round(r["mrr"], 4),
            "H@1":  round(r["h1"],  4),
            "H@3":  round(r["h3"],  4),
            "H@10": round(r["h10"], 4),
            "N":    r.get("n", "?"),
        })

    with open(out_dir / "v7_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "MRR", "H@1", "H@3", "H@10", "N"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Comparison saved: {out_dir / 'v7_comparison.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="V7 QuantumReasoner — Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets: toy | fb15k237 | wn18rr | nell995 | yago3_10 | all

Examples:
  python experiments/run_v7.py --dataset toy
  python experiments/run_v7.py --dataset fb15k237 --epochs 300
  python experiments/run_v7.py --dataset fb15k237 --quick
  python experiments/run_v7.py --dataset all
        """,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="toy",
        choices=["toy", "fb15k237", "wn18rr", "nell995", "yago3_10", "all"],
        help="Dataset to run V7 on. 'all' runs all real datasets in sequence.",
    )

    # Model hyperparameters (override per-dataset defaults)
    parser.add_argument("--epochs",       type=int,   default=None,
                        help="Training epochs (default: per-dataset)")
    parser.add_argument("--embed_dim",    type=int,   default=None,
                        help="Embedding dimension (default: 32 for toy, 128 for real)")
    parser.add_argument("--bond_dim",     type=int,   default=None,
                        help="MPS bond dimension (default: 16 for toy, 32 for real)")
    parser.add_argument("--num_jumps",    type=int,   default=4,
                        help="Lindblad jump operators per relation")
    parser.add_argument("--num_concepts", type=int,   default=8,
                        help="Sasaki concept projectors")
    parser.add_argument("--batch_size",   type=int,   default=None,
                        help="Training batch size (default: per-dataset)")
    parser.add_argument("--neg_samples",  type=int,   default=None,
                        help="Negative samples per positive (default: per-dataset)")
    parser.add_argument("--lr",           type=float, default=None,
                        help="Base learning rate (default: per-dataset)")
    parser.add_argument("--max_hops",     type=int,   default=None,
                        help="Max path hops for path cache (default: per-dataset)")
    parser.add_argument("--max_paths",    type=int,   default=None,
                        help="Max paths per pair (default: per-dataset)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       type=str,   default="cuda")

    # Toy KG specific (legacy)
    parser.add_argument("--toy_only",    action="store_true",
                        help="[legacy] Run Toy KG only (same as --dataset toy)")
    parser.add_argument("--verify_only", action="store_true",
                        help="Theory verification only (requires --dataset toy)")
    parser.add_argument("--compare",     action="store_true",
                        help="V5 vs V7 comparison on Toy KG")

    # Convenience
    parser.add_argument("--quick",        action="store_true",
                        help="Fast test: 50 epochs, small batch")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="Force rebuild path cache")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ensure_dirs()

    # Quick mode overrides
    if args.quick:
        if args.epochs is None:
            args.epochs = 50
        if args.batch_size is None:
            args.batch_size = 128
        print("  Quick mode: 50 epochs")

    # Toy KG legacy flags
    if args.toy_only:
        args.dataset = "toy"

    # Toy-specific defaults
    if args.dataset == "toy":
        if args.embed_dim is None:
            args.embed_dim = 32
        if args.bond_dim is None:
            args.bond_dim = 16
        if args.batch_size is None:
            args.batch_size = 8
        if args.lr is None:
            args.lr = 1e-3
        if args.epochs is None:
            args.epochs = 150

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.verify_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kg     = build_toy_kg()
        model  = build_v7_model(kg, embed_dim=args.embed_dim or 32,
                                bond_dim=args.bond_dim or 16)
        model.to(device)
        _verify_v7_theory(model, kg, device)
        return

    if args.compare:
        run_comparison_v5_v7(args)
        return

    if args.dataset == "toy":
        out     = run_toy_kg_v7(args)
        results = out["results"]

        with open(RES_DIR / "v7_full_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "MRR", "H@1", "H@3", "H@10", "N"])
            writer.writeheader()
            writer.writerow({
                "model": "V7 QuantumReasoner (Lindblad+MPS+Sasaki+ECC)",
                "MRR":  round(results["mrr"],  4),
                "H@1":  round(results["h1"],   4),
                "H@3":  round(results["h3"],   4),
                "H@10": round(results["h10"],  4),
                "N":    results.get("n", "?"),
            })
        print(f"\n  Results saved: outputs/results/v7_full_results.csv")
        print("\n" + "=" * 70)
        print("V7 TOY KG COMPLETE")
        print(f"  MRR: {results['mrr']:.4f}  H@1: {results['h1']:.4f}"
              f"  H@10: {results['h10']:.4f}")
        print("=" * 70)

    elif args.dataset == "all":
        all_results = []
        for ds_name in ["fb15k237", "wn18rr", "nell995", "yago3_10"]:
            data_dir = Path(DATASET_CONFIGS[ds_name]["data_dir"])
            if not data_dir.exists():
                print(f"\n  Skipping {ds_name}: data not downloaded "
                      f"(run: python data/download.py --dataset "
                      f"{DATASET_CONFIGS[ds_name]['dataset_key']})")
                continue
            try:
                r = run_real_dataset_v7(ds_name, args)
                if r:
                    all_results.append(r)
            except Exception as e:
                print(f"\n  ERROR on {ds_name}: {e}")

        # Save combined summary
        if all_results:
            summary_path = RES_DIR / "v7_all_datasets_results.csv"
            fieldnames = ["dataset", "model", "mrr", "hits@1", "hits@3", "hits@10", "n"]
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in all_results:
                    writer.writerow({k: r.get(k, "") for k in fieldnames})
            print(f"\n  All-dataset summary: {summary_path}")

    else:
        # Single real dataset
        data_dir = Path(DATASET_CONFIGS[args.dataset]["data_dir"])
        if not data_dir.exists():
            ds_key = DATASET_CONFIGS[args.dataset]["dataset_key"]
            print(f"\n  Data not found: {data_dir}")
            print(f"  Download first:  python data/download.py --dataset {ds_key}")
            print(f"  Then re-run:     python experiments/run_v7.py --dataset {args.dataset}")
            return

        run_real_dataset_v7(args.dataset, args)


if __name__ == "__main__":
    main()
