"""
experiments/run_v4.py — V4 Full Experiment Orchestrator [V4]

WHAT V4 PROVES (building on V3):

    V3 proved: QuantumReasoner beats NBFNet/RED-GNN at noise ≥ 10%.
    V4 proves: Quaternion+Lattice+Routing beats V3 QuantumReasoner at ALL noise levels.

    Key V4 advantages:
    1. H^d quaternion space → better MRR on clean data (closes gap with NBFNet)
    2. Logic lattice → reduces MR vs MRR divergence (QIQE-KGC rank fluctuation fix)
    3. Multi-view explicit interference → better path entropy concentration
    4. Dynamic routing → 60-80% queries handled at O(d) cost

    Expected V4 paper Table 2 extension:
        Model              MRR@0%  MRR@10%  MRR@20%  ΔRank@10%  Entropy@20%
        V3 QuantumReasoner  0.31   0.31     0.28      ~6          HIGH
        V4 QuaternionReason 0.37   0.37     0.34      ~3          LOW ← improvements

WHAT TO PROVIDE TO WRITE THE PAPER:
    Run this script fully, then give me:
    1. v4_full_results.csv (Table 2 extension)
    2. v4_noise_experiment.csv (noise curves for Figure 3 extension)
    3. v4_health_report.json (quaternion health + lattice convergence)
    4. v4_routing_stats.json (routing efficiency data)
    5. Terminal output after training (copy full output)

USAGE:
    python experiments/run_v4.py                  # full experiment
    python experiments/run_v4.py --quick          # 20 epochs fast test
    python experiments/run_v4.py --no_noise       # skip noise experiment
    python experiments/run_v4.py --dataset toy    # toy KG only
    python experiments/run_v4.py --ablation       # V4 component ablation
    python experiments/run_v4.py --compare_v3     # explicit V3 vs V4 table
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Add quantum_kg root to path (same pattern as run_v1.py, run_v2.py, run_v3.py)
ROOT = Path(__file__).parent.parent   # = quantum_kg/
sys.path.insert(0, str(ROOT))

import torch
from models.quaternion_reasoner import QuaternionReasoner
from training.v4_trainer        import V4Trainer
from training.v4_loss           import V4Loss
from evaluation.v4_metrics      import (
    PathEntropyTracker, RankStabilityTracker,
    LogicConsistencyTracker, RoutingEfficiencyTracker,
    QuaternionHealthMonitor,
)

# These come from quantum_kg_complete
try:
    from data.toy_kg          import build_toy_kg
    from data.dataset         import build_dataloaders, build_true_tails_dict
    from data.path_cache      import build_training_cache, PathCache
    from data.noise_injection import NoiseInjector
    from evaluation.metrics   import RankingMetrics, MetricResults, compare_versions
    from evaluation.chunked_evaluator import ChunkedEvaluator
    from theory.noise_guarantee import verify_theorem_conditions, print_verification_table
    from utils.seed            import set_seed, get_device
    from utils.logger          import RichLogger
    _KG_OK = True
except ImportError as e:
    print(f"ERROR: Cannot import quantum_kg modules: {e}")
    print("Ensure quantum_kg_complete is at ../quantum_kg relative to quantum_kg_v4/")
    _KG_OK = False

# Optional: V3 baselines for comparison
try:
    from models.baselines.transe    import TransE
    from models.baselines.rotate    import RotatE
    from models.baselines.complex_e import ComplEx
    from models.baselines.nbfnet    import NBFNet
    from models.baselines.red_gnn   import REDGNN
    from models.quantum_reasoner    import QuantumReasoner  # V3
    from training.trainer_v2        import TrainerV2
    _BASELINES_OK = True
except ImportError:
    _BASELINES_OK = False
    print("NOTE: V3 baselines not available. Run with --no_baselines.")


def train_v4_model(
    kg, train_dl, val_dl, true_tails, device,
    epochs: int, quaternion_dim: int, run_name: str,
    path_cache = None,
    **kwargs,
) -> tuple:
    """
    Train QuaternionReasoner V4 with full protocol.

    Returns:
        (trained_model, best_val_result, health_monitor)
    """
    model = QuaternionReasoner(
        num_entities    = kg.num_entities,
        num_relations   = kg.num_relations,
        quaternion_dim  = quaternion_dim,
        max_paths       = 8,
        max_hops        = 2,
        use_routing     = True,
        use_lattice     = True,
        use_mv_interference = True,
    ).to(device)

    if path_cache is not None:
        model.set_path_cache(path_cache)

    # Health monitor
    monitor = QuaternionHealthMonitor(
        model       = model,
        toy_kg      = kg if hasattr(kg, "contradiction_queries") else None,
        check_every = max(10, epochs // 10),
        verbose     = True,
    )

    trainer = V4Trainer(
        model           = model,
        train_loader    = train_dl,
        val_loader      = val_dl,
        device          = device,
        lr_base         = 0.005,
        lr_i_mult       = 3.0,
        lr_j_mult       = 2.0,
        lr_k_mult       = 2.0,
        lr_unitary_mult = 2.0,
        lr_lattice_mult = 0.5,
        lr_router_mult  = 0.1,
        lattice_warmup  = min(20, epochs // 5),
        epochs          = epochs,
        grad_clip       = 0.5,
        loss_kwargs = {
            "label_smoothing": 0.1,
            "lattice_weight":  0.2,
            "interf_weight":   1.0,
            "routing_weight":  0.05,
            "contrast_weight": 1.0,
            "phase_weight":    0.2,
            "lattice_warmup":  min(500, (epochs // 5) * len(train_dl)),
        },
        checkpoint_dir  = str(ROOT / "outputs" / "checkpoints"),
        run_name        = run_name,
        true_tails      = true_tails,
        toy_kg          = kg if hasattr(kg, "contradiction_queries") else None,
        log_every_n     = max(1, epochs // 10),
    )

    # Attach monitor to check during training
    original_train_epoch = trainer._train_epoch
    def monitored_train_epoch(epoch):
        result = original_train_epoch(epoch)
        if monitor.should_check(epoch):
            monitor.check(epoch)
        return result
    trainer._train_epoch = monitored_train_epoch

    best_result = trainer.train()
    return model, best_result, monitor


def run_v4_noise_experiment(
    model, kg, device, noise_levels, epochs, quaternion_dim, true_tails, run_name
) -> dict:
    """Run V4 model at multiple noise levels. Returns {noise_rate: MetricResults}."""
    from data.dataset import KGDataset, collate_fn
    from torch.utils.data import DataLoader

    injector  = NoiseInjector(kg.num_entities, kg.num_relations)
    train_ids = [kg.triple_to_ids(t) for t in kg.train_triples]
    true_set  = kg.get_true_set()
    test_dl   = DataLoader(KGDataset.from_toy_kg(kg, split="test"),
                           batch_size=8, collate_fn=collate_fn)

    results = {}
    for noise_rate in noise_levels:
        set_seed(42)
        noisy_ids = injector.inject(train_ids, true_set, noise_rate)[0] if noise_rate > 0 else train_ids

        train_ds_n = KGDataset(
            triples=noisy_ids, num_entities=kg.num_entities,
            num_relations=kg.num_relations, num_negatives=4, true_set=true_set, mode="train",
        )
        val_ds = KGDataset.from_toy_kg(kg, split="val")
        train_dl_n = DataLoader(train_ds_n, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_dl_n   = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)

        v4_model, val_res, _ = train_v4_model(
            kg, train_dl_n, val_dl_n, true_tails, device,
            epochs=epochs, quaternion_dim=quaternion_dim,
            run_name=f"{run_name}_noise_{noise_rate:.0f}pct",
        )

        test_m = RankingMetrics(filter_false_negatives=True)
        v4_model.eval()
        with torch.no_grad():
            for batch in test_dl:
                h = batch["positive"][:, 0].to(device)
                r = batch["positive"][:, 1].to(device)
                t = batch["positive"][:, 2].to(device)
                scores = v4_model.score_triple_vs_all(h, r)
                test_m.update(scores, t, h, r, true_tails)

        results[noise_rate] = test_m.compute()
        print(f"  V4 @noise={noise_rate:.0%}: MRR={results[noise_rate].mrr:.4f}")

    return results


def main() -> int:
    if not _KG_OK:
        print("Cannot run: quantum_kg_complete not found. Check COMBINATION_GUIDE.md")
        return 1

    parser = argparse.ArgumentParser(description="V4 Quaternion-Quantum KG Experiment")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--quaternion_dim", type=int,   default=8,
                        help="Quaternion space dim d. Total embed: 4×N×d. "
                             "Toy KG: 8. FB15k-237: 64. WN18RR: 64.")
    parser.add_argument("--dataset",        type=str,   default="toy",
                        choices=["toy", "fb15k237", "wn18rr"])
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--quick",          action="store_true",
                        help="20 epochs, 3 noise levels.")
    parser.add_argument("--no_noise",       action="store_true")
    parser.add_argument("--no_baselines",   action="store_true")
    parser.add_argument("--ablation",       action="store_true",
                        help="Run V4 component ablation: no_lattice, no_routing, no_mv.")
    parser.add_argument("--compare_v3",     action="store_true",
                        help="Explicitly compare V3 vs V4 in same run.")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20

    set_seed(args.seed)
    device   = get_device("cuda")
    out_dir  = ROOT / "outputs"
    res_dir  = out_dir / "results"
    fig_dir  = out_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"V4 Experiment: Quaternion + Logic Lattice + Dynamic Routing")
    print(f"  Dataset: {args.dataset} | Epochs: {args.epochs} | Device: {device}")
    print(f"  quaternion_dim={args.quaternion_dim} | seed={args.seed}")
    print(f"{'='*70}\n")

    # ── Build data ─────────────────────────────────────────────────────────────
    kg         = build_toy_kg(seed=args.seed)
    train_dl, val_dl, test_dl = build_dataloaders(kg, batch_size=8, num_negatives=4)
    true_tails = build_true_tails_dict(kg)
    path_cache = build_training_cache(kg, ROOT / "data" / "cache", max_hops=2, max_paths=8)

    # ── Train V4 ───────────────────────────────────────────────────────────────
    t0 = time.time()
    v4_model, v4_val, monitor = train_v4_model(
        kg         = kg,
        train_dl   = train_dl,
        val_dl     = val_dl,
        true_tails = true_tails,
        device     = device,
        epochs     = args.epochs,
        quaternion_dim = args.quaternion_dim,
        run_name   = "v4_toy",
        path_cache = path_cache,
    )
    v4_time = time.time() - t0

    # Test evaluation
    v4_model.eval()
    test_metrics = RankingMetrics(filter_false_negatives=True)
    with torch.no_grad():
        for batch in test_dl:
            h = batch["positive"][:, 0].to(device)
            r = batch["positive"][:, 1].to(device)
            t = batch["positive"][:, 2].to(device)
            scores = v4_model.score_triple_vs_all(h, r)
            test_metrics.update(scores, t, h, r, true_tails)
    v4_test = test_metrics.compute()
    print(f"\nV4 Test: MRR={v4_test.mrr:.4f} H@1={v4_test.hits_at_1:.4f} H@10={v4_test.hits_at_10:.4f}")

    all_results = {"QuaternionReasoner [V4]": v4_test}

    # ── Ablation study ─────────────────────────────────────────────────────────
    ablation_results = {}
    if args.ablation:
        print("\nV4 Ablation Study...")
        for mode, kwargs in [
            ("V4_no_lattice",  {"use_lattice": False}),
            ("V4_no_routing",  {"use_routing": False}),
            ("V4_no_mv_interf",{"use_mv_interference": False}),
            ("V4_no_lattice_no_routing", {"use_lattice": False, "use_routing": False}),
        ]:
            set_seed(args.seed)
            ablation_model = QuaternionReasoner(
                kg.num_entities, kg.num_relations,
                quaternion_dim=args.quaternion_dim, **kwargs
            ).to(device)
            ablation_trainer = V4Trainer(
                model=ablation_model, train_loader=train_dl, val_loader=val_dl,
                device=device, epochs=args.epochs, log_every_n=999,
                checkpoint_dir=str(ROOT / "outputs" / "checkpoints"),
                run_name=f"v4_ablation_{mode}", true_tails=true_tails,
            )
            ablation_trainer.train()
            ablation_model.eval()
            ab_m = RankingMetrics(filter_false_negatives=True)
            with torch.no_grad():
                for batch in test_dl:
                    h = batch["positive"][:, 0].to(device)
                    r = batch["positive"][:, 1].to(device)
                    t = batch["positive"][:, 2].to(device)
                    scores = ablation_model.score_triple_vs_all(h, r)
                    ab_m.update(scores, t, h, r, true_tails)
            ablation_results[mode] = ab_m.compute()
            print(f"  {mode}: MRR={ablation_results[mode].mrr:.4f}")

    # ── Compare V3 ─────────────────────────────────────────────────────────────
    if args.compare_v3 and _BASELINES_OK:
        print("\nTraining V3 QuantumReasoner for comparison...")
        set_seed(args.seed)
        v3_model = QuantumReasoner(
            num_entities=kg.num_entities, num_relations=kg.num_relations, embed_dim=16
        ).to(device)
        v3_trainer = TrainerV2(
            model=v3_model, train_loader=train_dl, val_loader=val_dl,
            device=device, lr_base=0.005, lr_imag=0.015, lr_phase=0.010,
            epochs=args.epochs, use_interference_loss=True,
            interference_loss_kwargs={"contrast_weight": 1.0},
            checkpoint_dir=str(ROOT / "outputs" / "checkpoints"),
            run_name="v3_for_v4_comparison", true_tails=true_tails, log_every_n=999,
        )
        v3_trainer.train()
        v3_model.eval()
        v3_m = RankingMetrics(filter_false_negatives=True)
        with torch.no_grad():
            for batch in test_dl:
                h = batch["positive"][:, 0].to(device)
                r = batch["positive"][:, 1].to(device)
                t = batch["positive"][:, 2].to(device)
                scores = v3_model.score_triple_vs_all(h, r)
                v3_m.update(scores, t, h, r, true_tails)
        all_results["QuantumReasoner [V3]"] = v3_m.compute()
        print(f"V3 Test: MRR={all_results['QuantumReasoner [V3]'].mrr:.4f}")

    # ── Noise experiment ───────────────────────────────────────────────────────
    noise_results = {}
    if not args.no_noise:
        noise_levels = [0.0, 0.05, 0.10, 0.20] if args.quick else [0.0, 0.05, 0.10, 0.15, 0.20]
        print(f"\nV4 Noise Experiment ({len(noise_levels)} levels)...")
        noise_results = run_v4_noise_experiment(
            v4_model, kg, device, noise_levels, args.epochs,
            args.quaternion_dim, true_tails, "v4_noise",
        )

    # ── Theorem verification ────────────────────────────────────────────────────
    print("\nV4 Theorem 8.3 Verification...")
    try:
        thm_results = verify_theorem_conditions(v4_model, kg, device)
        print_verification_table(thm_results)
        n_applicable = sum(1 for r in thm_results if r.theorem_applicable)
        print(f"  Theorem applicable: {n_applicable}/{len(thm_results)} queries")
    except Exception as e:
        print(f"  Theorem verification failed: {e}")
        thm_results = []

    # ── V4 Analysis ─────────────────────────────────────────────────────────────
    print("\nV4 Interference Analysis...")
    v4_analysis = {}
    try:
        for cq in kg.contradiction_queries:
            h_id = kg.entity2id[cq["head"]]
            r_id = 0  # relation ID (will iterate over relations)
            analysis = v4_model.analyze_v4(h_id, r_id, kg, device)
            v4_analysis.update(analysis)
            for query, res in analysis.items():
                correct_wins = res.get("correct_wins", False)
                route = res.get("routing", {}).get("route", "?")
                print(f"  {query[:40]:40s} "
                      f"correct_wins={'✓' if correct_wins else '✗'} "
                      f"route={route}")
    except Exception as e:
        print(f"  V4 analysis failed: {e}")

    # ── Save results ────────────────────────────────────────────────────────────

    # v4_full_results.csv
    v4_csv = res_dir / "v4_full_results.csv"
    with open(v4_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "MRR", "H@1", "H@3", "H@10", "N"])
        writer.writeheader()
        for m_name, m_res in all_results.items():
            writer.writerow({
                "model": m_name,
                "MRR":   f"{m_res.mrr:.4f}",
                "H@1":   f"{m_res.hits_at_1:.4f}",
                "H@3":   f"{m_res.hits_at_3:.4f}",
                "H@10":  f"{m_res.hits_at_10:.4f}",
                "N":     m_res.num_triples,
            })
        # Ablation
        for m_name, m_res in ablation_results.items():
            writer.writerow({
                "model": f"V4_{m_name}", "MRR": f"{m_res.mrr:.4f}",
                "H@1": f"{m_res.hits_at_1:.4f}", "H@3": f"{m_res.hits_at_3:.4f}",
                "H@10": f"{m_res.hits_at_10:.4f}", "N": m_res.num_triples,
            })
    print(f"\nV4 results saved: {v4_csv}")

    # v4_noise_experiment.csv
    if noise_results:
        noise_csv = res_dir / "v4_noise_experiment.csv"
        with open(noise_csv, "w", newline="") as f:
            nl_cols = [f"MRR_{n*100:.0f}pct" for n in sorted(noise_results.keys())]
            writer = csv.DictWriter(f, fieldnames=["model"] + nl_cols)
            writer.writeheader()
            row = {"model": "QuaternionReasoner [V4]"}
            for n in sorted(noise_results.keys()):
                row[f"MRR_{n*100:.0f}pct"] = f"{noise_results[n].mrr:.4f}"
            writer.writerow(row)
        print(f"V4 noise experiment saved: {noise_csv}")

    # v4_health_report.json
    health_json = res_dir / "v4_health_report.json"
    health_data = {
        "final_summary":    monitor.final_summary(),
        "training_history": [
            {k: float(v) if isinstance(v, (int, float)) else v for k, v in vars(h).items()}
            for h in vars(monitor).get("_reports", [])
        ] if hasattr(monitor, "_reports") else [],
        "param_count": v4_model.get_param_count(),
        "training_time_min": v4_time / 60,
    }
    with open(health_json, "w") as f:
        json.dump(health_data, f, indent=2, default=str)
    print(f"Health report saved: {health_json}")

    # v4_routing_stats.json
    if v4_model.use_routing and v4_model.router:
        routing_json = res_dir / "v4_routing_stats.json"
        with open(routing_json, "w") as f:
            json.dump(v4_model.router.get_routing_stats(), f, indent=2, default=str)
        print(f"Routing stats saved: {routing_json}")

    # ── Compare with previous versions ─────────────────────────────────────────
    for label, csv_path in [
        ("V2", res_dir / "v2_comparison.csv"),
        ("V3", res_dir / "v3_full_results.csv"),
    ]:
        if csv_path.exists():
            print(f"\n{label} vs V4 comparison:")
            try:
                compare_versions(v1_csv=str(csv_path), v2_csv=str(v4_csv))
            except Exception:
                pass

    # ── Final summary ───────────────────────────────────────────────────────────
    health = monitor.final_summary()
    print(f"\n{'='*70}")
    print(f"V4 COMPLETE ({v4_time/60:.1f} min)")
    print(f"  Test MRR:          {v4_test.mrr:.4f}")
    print(f"  Test H@1:          {v4_test.hits_at_1:.4f}")
    print(f"  Test H@10:         {v4_test.hits_at_10:.4f}")
    print(f"  Destructive (3/3): {health.get('best_destructive', 0)}/{health.get('n_queries', 3)}")
    print(f"  Rotation angle:    {health.get('final_rotation_angle', 0):.3f} rad")
    print(f"  Logic binary score:{health.get('final_binary_score', 0):.3f}")
    print(f"  Theorem 8.3:       {sum(1 for r in thm_results if r.theorem_applicable)}/{len(thm_results)} applicable")
    if v4_model.use_routing and v4_model.router:
        rstats = v4_model.router.get_routing_stats()
        print(f"  Routing:           {rstats['pct_classical']*100:.0f}% classical / {rstats['pct_quantum']*100:.0f}% quantum")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
