"""
experiments/run_v8.py — V8 Curved Manifold Experiment Runner

V8: Relational Holonomy + Adaptive Curvature + Schrödinger-Dirac Loss

New in V8:
    - Knowledge Graph as Curved Relational Manifold
    - Parallel transport T_r ∈ U(d) per relation (replaces DiagonalUnitary)
    - Holonomy gap ||T_rN...T_r1 − I||_F as contradiction signal
    - Adaptive curvature κ_e per entity (hyperbolic for hierarchies, spherical for cycles)
    - Schrödinger-Dirac loss: Lorentz-invariant scoring via Dirac spinors

Usage:
    python experiments/run_v8.py           # full experiment
    python experiments/run_v8.py --quick   # 20 epochs
    python experiments/run_v8.py --compare_v6  # V6 vs V8 comparison table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Optional V8 component imports — degrade gracefully if unavailable
# ---------------------------------------------------------------------------

_V8_OK = False
try:
    from models.v8_reasoner import V8CurvedManifoldReasoner
    _V8_OK = True
    print("[V8] V8CurvedManifoldReasoner loaded OK.")
except Exception as _e:
    print(f"[V8] WARNING: Could not import V8CurvedManifoldReasoner: {_e}")
    print("[V8] Will fall back to TransE baseline.")

_DIRAC_LOSS_OK = False
try:
    from training.schrodinger_dirac_loss import SchrodingerDiracLoss, DiracSpinorEncoder, DiracRelationOperator
    _DIRAC_LOSS_OK = True
    print("[V8] SchrodingerDiracLoss loaded OK.")
except Exception as _e:
    print(f"[V8] WARNING: Could not import SchrodingerDiracLoss: {_e}")

_METRICS_OK = False
try:
    from evaluation.metrics import RankingMetrics
    _METRICS_OK = True
except Exception as _e:
    print(f"[V8] WARNING: Could not import RankingMetrics: {_e}")

# ---------------------------------------------------------------------------
# TransE fallback model
# ---------------------------------------------------------------------------

class _TransEFallback(nn.Module):
    """Minimal TransE for graceful degradation when V8 imports fail."""

    def __init__(self, num_entities: int, num_relations: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.embed_dim = embed_dim
        self.ent = nn.Embedding(num_entities, embed_dim)
        self.rel = nn.Embedding(num_relations, embed_dim)

    def score_triple(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -((self.ent(h) + self.rel(r) - self.ent(t)).norm(dim=-1))

    def score_triple_vs_all(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        hr = self.ent(h) + self.rel(r)                      # (B, d)
        all_t = self.ent.weight                             # (E, d)
        return -torch.cdist(hr.unsqueeze(1), all_t.unsqueeze(0)).squeeze(1)

    def get_param_count(self) -> dict[str, int]:
        return {
            "ent": self.ent.weight.numel(),
            "rel": self.rel.weight.numel(),
            "total": sum(p.numel() for p in self.parameters()),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CKPT_DIR = ROOT / "outputs" / "checkpoints"
RES_DIR  = ROOT / "outputs" / "results"


def _ensure_dirs() -> None:
    for d in [CKPT_DIR, RES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _kg_to_loaders(kg, batch_size: int = 8, neg_per_pos: int = 16):
    """Build train/val DataLoaders from a ToyKG."""
    triples = kg.triples
    n_train = max(1, int(len(triples) * 0.8))

    def _split_to_tensors(tris):
        heads, rels, tails = [], [], []
        for tri in tris:
            h, r, t = kg.triple_to_ids(tri)
            heads.append(h); rels.append(r); tails.append(t)
        return (
            torch.tensor(heads, dtype=torch.long),
            torch.tensor(rels,  dtype=torch.long),
            torch.tensor(tails, dtype=torch.long),
        )

    h_tr, r_tr, t_tr = _split_to_tensors(triples[:n_train])
    h_va, r_va, t_va = _split_to_tensors(triples[n_train:])

    # Random negatives sampled offline
    neg_tr = torch.randint(0, kg.num_entities, (len(h_tr), neg_per_pos))
    neg_va = torch.randint(0, kg.num_entities, (max(1, len(h_va)), neg_per_pos))

    train_ds = TensorDataset(h_tr, r_tr, t_tr, neg_tr)
    val_ds   = TensorDataset(h_va, r_va, t_va, neg_va)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


def _build_model(kg, args) -> nn.Module:
    """Build V8 model (or TransE fallback)."""
    if _V8_OK:
        model = V8CurvedManifoldReasoner(
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
            embed_dim=args.embed_dim,
            holonomy_weight=0.1,
            use_adaptive_curvature=True,
        )
        print(f"[V8] Model: V8CurvedManifoldReasoner  |  {model.extra_repr()}")
    else:
        model = _TransEFallback(kg.num_entities, kg.num_relations, args.embed_dim)
        print(f"[V8] Model: TransE fallback (V8 import failed)")
    return model


def _build_loss(num_entities: int, embed_dim: int):
    """Build SchrodingerDiracLoss or BCE fallback."""
    if _DIRAC_LOSS_OK:
        loss_fn = SchrodingerDiracLoss(dirac_weight=1.0, schrodinger_weight=0.1, lorentz_weight=0.05)
        print("[V8] Loss: SchrodingerDiracLoss (Dirac + Schrödinger + Lorentz)")
        return loss_fn, None, None
    else:
        print("[V8] Loss: BCE fallback (SchrodingerDiracLoss import failed)")
        return None, None, None


def _train_epoch(model, loader, optimizer, loss_fn, epoch: int, use_dirac: bool) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        h, r, t, neg = batch
        optimizer.zero_grad()

        pos_scores = model.score_triple(h, r, t)

        # Flatten negatives: (B, neg_per_pos) → (B * neg_per_pos,)
        B, N = neg.shape
        h_exp = h.unsqueeze(1).expand(-1, N).reshape(-1)
        r_exp = r.unsqueeze(1).expand(-1, N).reshape(-1)
        neg_flat = neg.reshape(-1)
        neg_scores = model.score_triple(h_exp, r_exp, neg_flat)

        if use_dirac and loss_fn is not None:
            loss = loss_fn(pos_scores, neg_scores.view(B, N).mean(-1))
        else:
            # Simple margin / BCE
            labels_pos = torch.ones_like(pos_scores)
            labels_neg = torch.zeros_like(neg_scores)
            loss = (
                F.binary_cross_entropy_with_logits(pos_scores, labels_pos)
                + F.binary_cross_entropy_with_logits(neg_scores, labels_neg)
            )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def _evaluate(model, val_loader, kg) -> dict[str, float]:
    """Simple ranking evaluation on validation triples."""
    model.eval()
    mrr_sum, hits1_sum, hits3_sum, hits10_sum = 0.0, 0.0, 0.0, 0.0
    n = 0

    for batch in val_loader:
        h, r, t, _ = batch
        all_scores = model.score_triple_vs_all(h, r)    # (B, E)

        for i in range(h.shape[0]):
            true_score = all_scores[i, t[i]].item()
            rank = int((all_scores[i] > true_score).sum().item()) + 1
            mrr_sum   += 1.0 / rank
            hits1_sum  += float(rank <= 1)
            hits3_sum  += float(rank <= 3)
            hits10_sum += float(rank <= 10)
            n += 1

    if n == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    return {
        "MRR":    mrr_sum   / n,
        "Hits@1": hits1_sum / n,
        "Hits@3": hits3_sum / n,
        "Hits@10":hits10_sum/ n,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_v8_experiment(args) -> dict:
    """Run the full V8 experiment on the toy KG."""
    torch.manual_seed(args.seed)
    _ensure_dirs()

    print("\n" + "=" * 60)
    print("V8 Curved Manifold Reasoner — Experiment")
    print("=" * 60)
    print("  Architecture: Parallel transport T_r ∈ U(d) + adaptive κ")
    print("  Holonomy gap: ||T_rN...T_r1 − I||_F (contradiction signal)")
    print("  Loss: Schrödinger-Dirac (Lorentz-invariant)")
    print("=" * 60)

    # 1. Build toy KG
    from data.toy_kg import build_toy_kg
    kg = build_toy_kg()
    print(f"\n[V8] Toy KG: {kg.num_entities} entities, {kg.num_relations} relations, "
          f"{len(kg.triples)} triples")

    # 2. DataLoaders
    train_loader, val_loader = _kg_to_loaders(kg, batch_size=8, neg_per_pos=16)

    # 3. Model
    model = _build_model(kg, args)

    # 4. Loss
    loss_fn, _, _ = _build_loss(kg.num_entities, args.embed_dim)
    use_dirac = _DIRAC_LOSS_OK and loss_fn is not None

    # 5. Optimiser — separate LR for curvature parameters (want slow adaptation)
    param_groups = [{"params": model.parameters(), "lr": args.lr}]
    if _V8_OK and hasattr(model, "curved_encoder"):
        curvature_params = list(model.curved_encoder.log_curvature.parameters())
        other_params = [p for p in model.parameters()
                        if not any(p is cp for cp in curvature_params)]
        param_groups = [
            {"params": other_params,      "lr": args.lr},
            {"params": curvature_params,  "lr": args.lr * 0.1},
        ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)

    # 6. Training loop
    print(f"\n[V8] Training for {args.epochs} epochs  (lr={args.lr})")
    best_mrr = 0.0
    train_losses = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        avg_loss = _train_epoch(model, train_loader, optimizer, loss_fn, epoch, use_dirac)
        train_losses.append(avg_loss)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            metrics = _evaluate(model, val_loader, kg)
            best_mrr = max(best_mrr, metrics["MRR"])
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:4d}/{args.epochs}  "
                f"loss={avg_loss:.4f}  "
                f"MRR={metrics['MRR']:.4f}  "
                f"Hits@10={metrics['Hits@10']:.4f}  "
                f"[{elapsed:.1f}s]"
            )

    # 7. Final evaluation
    final_metrics = _evaluate(model, val_loader, kg)
    elapsed_total = time.time() - t0

    # 8. Curvature stats (V8 specific)
    curvature_stats: dict = {}
    if _V8_OK and hasattr(model, "compute_manifold_curvature_stats"):
        curvature_stats = model.compute_manifold_curvature_stats()
        print(f"\n[V8] Manifold curvature stats (after training):")
        for k, v in curvature_stats.items():
            print(f"     {k}: {v:.4f}")
        print(f"  Interpretation:")
        print(f"     frac_hyperbolic={curvature_stats.get('frac_hyperbolic', 0):.2%} of entities → hyperbolic (hierarchies)")
        print(f"     frac_spherical ={curvature_stats.get('frac_spherical',  0):.2%} of entities → spherical  (cycles)")

    # 9. Param counts
    param_counts = model.get_param_count()
    print(f"\n[V8] Parameter counts:")
    for k, v in param_counts.items():
        print(f"     {k}: {v:,}")

    # 10. Holonomy gap demo
    if _V8_OK:
        from models.components.holonomy import HolonomyOperator
        hol_op = model.holonomy
        r_ids = torch.randint(0, kg.num_relations, (4,))
        cycle = [r_ids, r_ids]
        with torch.no_grad():
            gap = hol_op.holonomy_gap(cycle)
        print(f"\n[V8] Holonomy gap for a 2-cycle (trained model): "
              f"mean={gap.mean().item():.4f}  (lower = more consistent KG)")

    # 11. Save results
    results = {
        "model":           "V8CurvedManifoldReasoner" if _V8_OK else "TransE_fallback",
        "embed_dim":       args.embed_dim,
        "epochs":          args.epochs,
        "final_MRR":       final_metrics["MRR"],
        "final_Hits@1":    final_metrics["Hits@1"],
        "final_Hits@3":    final_metrics["Hits@3"],
        "final_Hits@10":   final_metrics["Hits@10"],
        "best_MRR":        best_mrr,
        "train_time_s":    elapsed_total,
        "param_counts":    param_counts,
        "curvature_stats": curvature_stats,
        "v8_components": {
            "parallel_transport": _V8_OK,
            "adaptive_curvature": _V8_OK,
            "dirac_loss":         _DIRAC_LOSS_OK,
        },
    }
    out_path = RES_DIR / "v8_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[V8] Results saved to {out_path}")

    print("\n" + "=" * 60)
    print("V8 Final Results:")
    print(f"  MRR:     {final_metrics['MRR']:.4f}")
    print(f"  Hits@1:  {final_metrics['Hits@1']:.4f}")
    print(f"  Hits@3:  {final_metrics['Hits@3']:.4f}")
    print(f"  Hits@10: {final_metrics['Hits@10']:.4f}")
    print(f"  Total training time: {elapsed_total:.1f}s")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# V6 vs V8 comparison
# ---------------------------------------------------------------------------

def compare_v6_v8(args) -> None:
    """Print a comparison table of V6 and V8 results (from saved JSON if available)."""
    print("\n" + "=" * 60)
    print("V6 vs V8 Comparison")
    print("=" * 60)

    v6_path = RES_DIR / "v6_results.json"
    v8_path = RES_DIR / "v8_results.json"

    # Load existing V8 result or run a quick one
    if not v8_path.exists():
        print("[compare] No V8 results found — running quick experiment first...")
        args_quick = argparse.Namespace(**vars(args))
        args_quick.epochs = 20
        run_v8_experiment(args_quick)

    v6_results: dict = {}
    if v6_path.exists():
        with open(v6_path) as fh:
            v6_results = json.load(fh)
    else:
        print("[compare] No V6 results found at outputs/results/v6_results.json")
        print("[compare] Run experiments/run_v6.py first for a full comparison.")
        v6_results = {
            "model": "V6 (not run)", "final_MRR": float("nan"),
            "final_Hits@1": float("nan"), "final_Hits@10": float("nan"),
        }

    with open(v8_path) as fh:
        v8_results = json.load(fh)

    header = f"{'Metric':<20} {'V6':>10} {'V8':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    metrics_to_compare = [
        ("MRR",     "final_MRR"),
        ("Hits@1",  "final_Hits@1"),
        ("Hits@3",  "final_Hits@3"),
        ("Hits@10", "final_Hits@10"),
    ]
    for label, key in metrics_to_compare:
        v6_val = v6_results.get(key, float("nan"))
        v8_val = v8_results.get(key, float("nan"))
        try:
            delta = v8_val - v6_val
            delta_str = f"{delta:+.4f}"
        except TypeError:
            delta_str = "N/A"
        print(f"{label:<20} {v6_val:>10.4f} {v8_val:>10.4f} {delta_str:>10}")

    print()
    print(f"  V6 model: {v6_results.get('model', 'unknown')}")
    print(f"  V8 model: {v8_results.get('model', 'unknown')}")
    print()
    print("V8 innovations:")
    print("  + Parallel transport T_r ∈ U(d)  — full unitary, not diagonal")
    print("  + Holonomy gap — geometric contradiction signal")
    print("  + Adaptive curvature κ_e per entity")
    print("  + Schrödinger-Dirac loss — Lorentz-invariant scoring")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V8 Curved Manifold Experiment Runner"
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Run only 20 training epochs (smoke test).",
    )
    p.add_argument(
        "--compare_v6", action="store_true",
        help="Print V6 vs V8 comparison table.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (default: 200).",
    )
    p.add_argument(
        "--embed_dim", type=int, default=64,
        help="Embedding dimension (default: 64, must be even).",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.quick:
        args.epochs = 20
        print("[V8] Quick mode: 20 epochs only.")

    if args.compare_v6:
        compare_v6_v8(args)
        return 0

    run_v8_experiment(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
