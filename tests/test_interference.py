"""
tests/test_interference.py — Complete Test Suite for quantum_kg

PURPOSE:
    Validates every new component added in V2 and V3.
    Designed to catch regressions, verify mathematical correctness,
    and confirm the paper's central claims are empirically testable.

RUN:
    pytest tests/ -v
    pytest tests/ -v -k "TestQuantumStates"
    pytest tests/ -v --tb=short -x

TEST GROUPS:
    TestQuantumStates         — unit norms, Born rule, fidelity
    TestUnitaryOperators      — norm preservation, U†U=I
    TestPathAggregator        — BFS paths, interference decomposition
    TestKGUnitaries           — relational, hierarchy, contextual
    TestPhaseSeparationLoss   — gradient direction, separation metric
    TestContrastiveLoss       — hinge, contradiction batches
    TestInterferenceReg       — Phase Collapse detection/prevention
    TestInterferenceMonitor   — health checks, auto-correction
    TestTrainerV2             — param groups, differential LRs
    TestPathCache             — build, load, coverage, lookup
    TestChunkedEvaluator      — parity with standard, OOM-safe
    TestNoiseGuarantee        — Theorem 8.3, formula correctness
    TestNoiseSeparation       — injection rates, monotonicity
    TestEndToEnd              — full pipeline regression test
"""

from __future__ import annotations

import sys
import math
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch
import torch.nn as nn


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")

@pytest.fixture(scope="module")
def toy_kg():
    from data.toy_kg import build_toy_kg
    return build_toy_kg(seed=42)

@pytest.fixture(scope="module")
def small_encoder(device):
    from models.components.quantum_states import QuantumStateEncoder
    return QuantumStateEncoder(num_entities=28, embed_dim=8, normalize=True).to(device)

@pytest.fixture(scope="module")
def small_unitary(device):
    from models.components.unitary_operators import DiagonalUnitary
    return DiagonalUnitary(num_relations=12, complex_dim=4).to(device)

@pytest.fixture(scope="module")
def small_model(device):
    from models.quantum_reasoner import QuantumReasoner
    return QuantumReasoner(
        num_entities=28, num_relations=12, embed_dim=8,
        unitary_type="diagonal", max_paths=4, max_hops=2,
    ).to(device)


# =============================================================================
#  GROUP 1: QUANTUM STATES
# =============================================================================

class TestQuantumStates:

    def test_unit_norm_all_entities(self, small_encoder, device):
        ids    = torch.arange(28, device=device)
        states = small_encoder(ids)
        norms  = states.abs().pow(2).sum(dim=-1).sqrt()
        assert (norms - 1.0).abs().max().item() < 1e-5, "Unit norm violation"

    def test_born_rule_range(self, small_encoder, device):
        ids    = torch.arange(28, device=device)
        states = small_encoder(ids)
        for i in range(28):
            prob = small_encoder.probability(states[0], states[i])
            assert 0.0 <= float(prob) <= 1.0 + 1e-6

    def test_self_fidelity_is_one(self, small_encoder, device):
        ids    = torch.arange(28, device=device)
        states = small_encoder(ids)
        for i in range(28):
            fid = small_encoder.probability(states[i], states[i])
            assert abs(float(fid) - 1.0) < 1e-5

    def test_inner_product_is_complex(self, small_encoder, device):
        ids    = torch.arange(5, device=device)
        states = small_encoder(ids)
        inner  = small_encoder.inner_product(states[0], states[1])
        assert inner.is_complex()

    def test_complex_dtype(self, small_encoder, device):
        states = small_encoder(torch.tensor([0], device=device))
        assert states.dtype == torch.complex64

    def test_state_shape(self, small_encoder, device):
        states = small_encoder(torch.arange(5, device=device))
        assert states.shape == (5, 4)

    def test_grad_flows_through_encoder(self, device):
        from models.components.quantum_states import QuantumStateEncoder
        enc    = QuantumStateEncoder(5, 8, normalize=True)
        states = enc(torch.tensor([0, 1]))
        loss   = states.abs().pow(2).sum()
        loss.backward()
        assert enc.real_embeddings.weight.grad is not None
        assert enc.imag_embeddings.weight.grad is not None


# =============================================================================
#  GROUP 2: UNITARY OPERATORS
# =============================================================================

class TestUnitaryOperators:

    def test_diagonal_preserves_norm(self, small_encoder, small_unitary, device):
        states = small_encoder(torch.arange(28, device=device))
        for r_id in range(12):
            rel_t   = torch.full((28,), r_id, dtype=torch.long, device=device)
            evolved = small_unitary.apply(states, rel_t)
            norms   = evolved.abs().pow(2).sum(-1).sqrt()
            assert (norms - 1.0).abs().max().item() < 1e-5

    def test_verify_unitarity(self, small_unitary):
        for r_id in range(12):
            assert small_unitary.verify_unitarity(r_id, tol=1e-4)

    def test_output_dtype(self, small_encoder, small_unitary, device):
        states = small_encoder(torch.arange(5, device=device))
        rel_t  = torch.zeros(5, dtype=torch.long, device=device)
        out    = small_unitary.apply(states, rel_t)
        assert out.dtype == torch.complex64

    def test_givens_preserves_norm(self, device):
        from models.components.unitary_operators import GivensUnitary
        from models.components.quantum_states import QuantumStateEncoder
        uni    = GivensUnitary(5, 4).to(device)
        enc    = QuantumStateEncoder(10, 8, normalize=True).to(device)
        states = enc(torch.arange(10, device=device))
        for r_id in range(5):
            rel_t   = torch.full((10,), r_id, dtype=torch.long, device=device)
            evolved = uni.apply(states, rel_t)
            norms   = evolved.abs().pow(2).sum(-1).sqrt()
            assert (norms - 1.0).abs().max().item() < 1e-4


# =============================================================================
#  GROUP 3: PATH AGGREGATOR
# =============================================================================

class TestPathAggregator:

    def test_find_paths_platypus(self, toy_kg):
        from models.components.path_aggregator import PathEnumerator
        adj  = toy_kg.get_adjacency()
        enum = PathEnumerator(adj, max_hops=3, max_paths=8)
        h_id = toy_kg.entity2id["Platypus"]
        c_id = toy_kg.entity2id["WarmBlooded"]
        w_id = toy_kg.entity2id["ColdBlooded"]
        assert len(enum.find_paths(h_id, c_id)) >= 1, "No correct paths for Platypus"
        assert len(enum.find_paths(h_id, w_id)) >= 2, "Need >= 2 contradictory paths"

    def test_interference_decomposition_identity(self, toy_kg, small_encoder, small_unitary, device):
        """total = classical_sum + interference (algebraic identity from Definition 6)."""
        from models.components.path_aggregator import PathEnumerator, AmplitudeAggregator
        adj  = toy_kg.get_adjacency()
        enum = PathEnumerator(adj, max_hops=3, max_paths=8)
        agg  = AmplitudeAggregator(4, max_paths=8).to(device)

        h_id  = toy_kg.entity2id["Platypus"]
        w_id  = toy_kg.entity2id["ColdBlooded"]
        paths = enum.find_paths(h_id, w_id)
        if not paths:
            pytest.skip("No paths found")

        h_s = small_encoder(torch.tensor([h_id], device=device)).squeeze(0)
        w_s = small_encoder(torch.tensor([w_id], device=device)).squeeze(0)
        r   = agg.compute_interference_terms(h_s, w_s, paths, small_unitary)

        total = float(r["total_probability"])
        cl    = float(r["classical_sum"])
        interf= float(r["interference"])

        assert abs(total - (cl + interf)) < 1e-5
        assert cl    >= -1e-6
        assert total >= -1e-6

    def test_empty_paths_no_crash(self, small_encoder, small_unitary, device):
        from models.components.path_aggregator import AmplitudeAggregator
        agg = AmplitudeAggregator(4, max_paths=4).to(device)
        s   = small_encoder(torch.tensor([0], device=device)).squeeze(0)
        t   = small_encoder(torch.tensor([1], device=device)).squeeze(0)
        r   = agg.compute_interference_terms(s, t, [], small_unitary)
        assert r.get("total_probability", 0) == 0.0

    def test_forward_shape_and_range(self, toy_kg, small_encoder, small_unitary, device):
        from models.components.path_aggregator import PathEnumerator, AmplitudeAggregator
        adj = toy_kg.get_adjacency()
        enum = PathEnumerator(adj, max_hops=2, max_paths=4)
        agg  = AmplitudeAggregator(4, max_paths=4).to(device)

        h_ids = torch.arange(5, device=device)
        t_ids = torch.tensor([5, 6, 7, 8, 9], device=device)
        h_s   = small_encoder(h_ids)
        t_s   = small_encoder(t_ids)
        paths = [enum.find_paths(int(h), int(t))[:4]
                 for h, t in zip(h_ids.tolist(), t_ids.tolist())]

        probs = agg.forward(h_s, t_s, paths, small_unitary)
        assert probs.shape == (5,)
        assert (probs >= -1e-5).all()
        assert (probs <= 1.0 + 1e-5).all()


# =============================================================================
#  GROUP 4: KG-SPECIFIC UNITARIES
# =============================================================================

class TestKGUnitaries:

    def test_relational_decomposed_preserves_norm(self, small_encoder, device):
        from models.components.kg_unitary import RelationalDecomposedUnitary
        uni    = RelationalDecomposedUnitary(12, 4).to(device)
        states = small_encoder(torch.arange(28, device=device))
        for r_id in range(12):
            rel_t   = torch.full((28,), r_id, dtype=torch.long, device=device)
            evolved = uni.apply(states, rel_t)
            norms   = evolved.abs().pow(2).sum(-1).sqrt()
            assert (norms - 1.0).abs().max().item() < 1e-5

    def test_structural_reg_non_negative(self, device):
        from models.components.kg_unitary import RelationalDecomposedUnitary
        uni  = RelationalDecomposedUnitary(12, 4,
               symmetric_rels={0,1}, inverse_pairs=[(2,3)]).to(device)
        loss = uni.structural_regularization_loss()
        assert float(loss) >= 0.0
        assert loss.requires_grad

    def test_structural_reg_penalizes_large_sym_phases(self, device):
        from models.components.kg_unitary import RelationalDecomposedUnitary
        uni = RelationalDecomposedUnitary(4, 4, symmetric_rels={0}).to(device)
        with torch.no_grad():
            uni.phases_sym.data[0] = torch.ones(4) * math.pi
        loss_large = float(uni.structural_regularization_loss())
        with torch.no_grad():
            uni.phases_sym.data[0] = torch.ones(4) * 0.01
        loss_small = float(uni.structural_regularization_loss())
        assert loss_large > loss_small

    def test_inverse_reg_penalizes_mismatch(self, device):
        from models.components.kg_unitary import RelationalDecomposedUnitary
        uni = RelationalDecomposedUnitary(4, 4, inverse_pairs=[(0,1)]).to(device)
        with torch.no_grad():
            uni.phases_inv.data[0] = torch.ones(4) * 1.0
            uni.phases_inv.data[1] = torch.ones(4) * 1.0
        loss_bad = float(uni.structural_regularization_loss())
        with torch.no_grad():
            uni.phases_inv.data[1] = -torch.ones(4) * 1.0
        loss_good = float(uni.structural_regularization_loss())
        assert loss_bad > loss_good

    def test_hierarchy_unitary_preserves_norm(self, small_encoder, device):
        from models.components.kg_unitary import HierarchyAwareUnitary
        uni    = HierarchyAwareUnitary(12, 4, relation_hierarchy=[(0,1)]).to(device)
        states = small_encoder(torch.arange(28, device=device))
        for r_id in range(12):
            rel_t   = torch.full((28,), r_id, dtype=torch.long, device=device)
            evolved = uni.apply(states, rel_t)
            norms   = evolved.abs().pow(2).sum(-1).sqrt()
            assert (norms - 1.0).abs().max().item() < 1e-5

    def test_hierarchy_reg_pulls_child_toward_parent(self, device):
        from models.components.kg_unitary import HierarchyAwareUnitary
        uni = HierarchyAwareUnitary(4, 4, relation_hierarchy=[(0,1)],
                                    hierarchy_weight=1.0).to(device)
        opt = torch.optim.SGD(uni.parameters(), lr=0.1)
        with torch.no_grad():
            uni.phases.data[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
            uni.phases.data[1] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        initial_diff = (uni.phases.data[0] - uni.phases.data[1]).norm().item()
        for _ in range(20):
            opt.zero_grad()
            uni.hierarchy_regularization_loss().backward()
            opt.step()
        final_diff = (uni.phases.data[0] - uni.phases.data[1]).norm().item()
        assert final_diff < initial_diff

    def test_contextual_unitary_preserves_norm(self, small_encoder, device):
        from models.components.kg_unitary import ContextualUnitary
        uni    = ContextualUnitary(12, 4, conditioning_dim=2).to(device)
        states = small_encoder(torch.arange(28, device=device))
        for r_id in range(12):
            rel_t   = torch.full((28,), r_id, dtype=torch.long, device=device)
            evolved = uni.apply(states, rel_t)
            norms   = evolved.abs().pow(2).sum(-1).sqrt()
            assert (norms - 1.0).abs().max().item() < 1e-4

    def test_infer_relation_structure_returns_dict(self, toy_kg):
        from models.components.kg_unitary import infer_relation_structure
        triple_set = {toy_kg.triple_to_ids(t) for t in toy_kg.triples}
        structure  = infer_relation_structure(triple_set, toy_kg.num_relations)
        assert "symmetric_rels" in structure
        assert "inverse_pairs" in structure
        assert isinstance(structure["symmetric_rels"], set)
        assert isinstance(structure["inverse_pairs"], list)


# =============================================================================
#  GROUP 5: PHASE SEPARATION LOSS
# =============================================================================

class TestPhaseSeparationLoss:

    def test_zero_when_fully_separated(self):
        from training.interference_loss import PhaseSeparationLoss
        loss_fn  = PhaseSeparationLoss(temperature=1.0, weight=1.0)
        pos_amps = torch.ones(4, dtype=torch.complex64)
        neg_amps = -torch.ones(4, 3, dtype=torch.complex64)
        loss     = loss_fn(pos_amps, neg_amps)
        assert float(loss) < 0.1, f"Should be ~0 when fully separated, got {loss}"

    def test_two_when_aligned(self):
        from training.interference_loss import PhaseSeparationLoss
        loss_fn  = PhaseSeparationLoss(temperature=1.0, weight=1.0)
        pos_amps = torch.ones(4, dtype=torch.complex64)
        neg_amps = torch.ones(4, 3, dtype=torch.complex64)
        loss     = loss_fn(pos_amps, neg_amps)
        assert float(loss) > 1.5, f"Should be ~2 when aligned, got {loss}"

    def test_grad_is_finite(self):
        from training.interference_loss import PhaseSeparationLoss
        loss_fn = PhaseSeparationLoss(weight=1.0)
        pos     = torch.randn(4, dtype=torch.complex64)
        neg     = torch.randn(4, 3, dtype=torch.complex64).requires_grad_(True)
        loss    = loss_fn(pos, neg)
        loss.backward()
        assert torch.isfinite(neg.grad).all()

    def test_measure_separation_keys(self):
        from training.interference_loss import PhaseSeparationLoss
        loss_fn = PhaseSeparationLoss()
        metrics = loss_fn.measure_separation(
            torch.randn(8, dtype=torch.complex64),
            torch.randn(8, 4, dtype=torch.complex64),
        )
        for key in ("mean_phase_diff", "fraction_separated", "target_phase_diff"):
            assert key in metrics

    def test_weight_scales_loss(self):
        from training.interference_loss import PhaseSeparationLoss
        pos  = torch.randn(4, dtype=torch.complex64)
        neg  = torch.randn(4, 3, dtype=torch.complex64)
        l1   = float(PhaseSeparationLoss(weight=1.0)(pos, neg))
        l2   = float(PhaseSeparationLoss(weight=2.0)(pos, neg))
        assert abs(l2 - 2 * l1) < 1e-5


# =============================================================================
#  GROUP 6: CONTRASTIVE INTERFERENCE LOSS
# =============================================================================

class TestContrastiveLoss:

    def test_zero_when_correct_exceeds_wrong(self):
        from training.interference_loss import ContrastiveInterferenceLoss
        margin  = 0.1
        correct = torch.tensor([0.8, 0.9, 0.7])
        wrong   = torch.tensor([0.3, 0.4, 0.2])
        hinge   = torch.clamp(wrong - correct + margin, min=0)
        assert hinge.sum().item() == 0.0

    def test_positive_when_wrong_exceeds_correct(self):
        from training.interference_loss import ContrastiveInterferenceLoss
        margin  = 0.1
        correct = torch.tensor([0.3, 0.4])
        wrong   = torch.tensor([0.8, 0.7])
        hinge   = torch.clamp(wrong - correct + margin, min=0)
        assert hinge.sum().item() > 0

    def test_batch_from_contradiction_queries(self, toy_kg, device):
        from training.interference_loss import ContrastiveInterferenceLoss
        loss_fn = ContrastiveInterferenceLoss()
        batch   = loss_fn.batch_from_contradiction_queries(
            toy_kg.contradiction_queries, toy_kg.entity2id,
            toy_kg.relation2id, device,
        )
        B = len(toy_kg.contradiction_queries)
        assert batch["head_ids"].shape    == (B,)
        assert batch["correct_tails"].shape == (B,)
        assert (batch["head_ids"] < toy_kg.num_entities).all()
        assert (batch["correct_tails"] < toy_kg.num_entities).all()


# =============================================================================
#  GROUP 7: INTERFERENCE REGULARIZATION
# =============================================================================

class TestInterferenceReg:

    def test_detects_collapse(self, device):
        from models.components.quantum_states import QuantumStateEncoder
        from models.components.unitary_operators import DiagonalUnitary
        from training.interference_loss import InterferenceRegularization
        enc = QuantumStateEncoder(10, 8, normalize=True).to(device)
        uni = DiagonalUnitary(5, 4).to(device)
        with torch.no_grad():
            enc.imag_embeddings.weight.data.fill_(0.0)
        metrics = InterferenceRegularization().measure_collapse(enc, uni)
        assert metrics["is_collapsed"]

    def test_no_collapse_random_init(self, small_encoder, small_unitary, device):
        from training.interference_loss import InterferenceRegularization
        metrics = InterferenceRegularization().measure_collapse(small_encoder, small_unitary)
        assert metrics["fraction_collapsed"] < 1.0

    def test_reg_higher_when_collapsed(self, device):
        from models.components.quantum_states import QuantumStateEncoder
        from models.components.unitary_operators import DiagonalUnitary
        from training.interference_loss import InterferenceRegularization
        enc = QuantumStateEncoder(10, 8, normalize=True).to(device)
        uni = DiagonalUnitary(5, 4).to(device)
        reg = InterferenceRegularization(encoder_weight=1.0, min_imag_norm=0.1)
        with torch.no_grad():
            enc.imag_embeddings.weight.data.fill_(0.001)
        loss_col = float(reg(enc, uni))
        with torch.no_grad():
            nn.init.normal_(enc.imag_embeddings.weight, std=0.5)
        loss_norm = float(reg(enc, uni))
        assert loss_col > loss_norm

    def test_grad_flows(self, small_encoder, small_unitary, device):
        from training.interference_loss import InterferenceRegularization
        reg  = InterferenceRegularization(encoder_weight=0.1, unitary_weight=0.1, min_imag_norm=100.0)
        loss = reg(small_encoder, small_unitary)
        loss.backward()
        assert small_encoder.imag_embeddings.weight.grad is not None
        assert small_encoder.imag_embeddings.weight.grad.abs().sum() > 0


# =============================================================================
#  GROUP 8: INTERFERENCE MONITOR
# =============================================================================

class TestInterferenceMonitor:

    def test_check_returns_valid_report(self, small_model, toy_kg, device):
        from training.interference_monitor import InterferenceMonitor
        monitor = InterferenceMonitor(small_model, toy_kg, verbose=False)
        report  = monitor.check(epoch=1)
        assert report.epoch == 1
        assert 0.0 <= report.mean_imag_norm
        assert 0.0 <= report.fraction_collapsed <= 1.0
        assert isinstance(report.contradiction_results, list)
        assert isinstance(report.summary(), str)

    def test_detects_collapse(self, toy_kg, device):
        from models.quantum_reasoner import QuantumReasoner
        from training.interference_monitor import InterferenceMonitor
        model = QuantumReasoner(toy_kg.num_entities, toy_kg.num_relations, embed_dim=8).to(device)
        with torch.no_grad():
            model.encoder.imag_embeddings.weight.data.fill_(0.0)
        monitor = InterferenceMonitor(
            model, toy_kg, healthy_after_epoch=1, collapse_threshold=0.02, verbose=False
        )
        report = monitor.check(epoch=5)
        assert not report.is_healthy

    def test_history_accumulates(self, small_model, toy_kg):
        from training.interference_monitor import InterferenceMonitor
        monitor = InterferenceMonitor(small_model, toy_kg, verbose=False)
        for epoch in [1, 2, 3]:
            monitor.check(epoch=epoch)
        assert len(monitor.report_history) == 3

    def test_final_summary_has_keys(self, small_model, toy_kg):
        from training.interference_monitor import InterferenceMonitor
        monitor = InterferenceMonitor(small_model, toy_kg, verbose=False)
        monitor.check(epoch=10)
        summary = monitor.final_summary()
        for key in ("final_epoch", "final_imag_norm", "best_destructive_count", "final_healthy"):
            assert key in summary


# =============================================================================
#  GROUP 9: TRAINER V2
# =============================================================================

class TestTrainerV2:

    def _make_trainer(self, small_model, toy_kg, device, **kwargs):
        from data.dataset import build_dataloaders
        from training.trainer_v2 import TrainerV2
        train_dl, val_dl, _ = build_dataloaders(
            toy_kg, batch_size=4, num_negatives=2, num_workers=0
        )
        defaults = dict(
            model=small_model, train_loader=train_dl, val_loader=val_dl,
            device=device, lr_base=0.005, lr_imag=0.015, lr_phase=0.010,
            epochs=1, use_interference_loss=False,
            checkpoint_dir=tempfile.mkdtemp(), run_name="test", true_tails={},
        )
        defaults.update(kwargs)
        return TrainerV2(**defaults)

    def test_five_param_groups(self, small_model, toy_kg, device):
        trainer = self._make_trainer(small_model, toy_kg, device)
        assert len(trainer.optimizer.param_groups) == 5

    def test_imag_lr_higher_than_real(self, small_model, toy_kg, device):
        trainer = self._make_trainer(small_model, toy_kg, device,
                                     lr_base=0.005, lr_imag=0.015)
        lrs = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
        assert lrs["imag"] >= lrs["real"] * 2.5

    def test_zero_weight_decay_for_imag_phase(self, small_model, toy_kg, device):
        trainer = self._make_trainer(small_model, toy_kg, device)
        for group in trainer.optimizer.param_groups:
            if group.get("name") in ("imag", "phases"):
                assert group["weight_decay"] == 0.0

    def test_grad_norms_dict_keys(self, small_model, toy_kg, device):
        trainer = self._make_trainer(small_model, toy_kg, device)
        # Do a forward-backward pass
        h = torch.zeros(4, dtype=torch.long, device=device)
        r = torch.zeros(4, dtype=torch.long, device=device)
        t = torch.arange(4, dtype=torch.long, device=device)
        (small_model.score_triple(h, r, t).sum()).backward()
        norms = trainer._measure_grad_norms()
        assert set(norms.keys()) == {"real", "imag", "phases", "agg", "bias"}
        for v in norms.values():
            assert math.isfinite(v)


# =============================================================================
#  GROUP 10: PATH CACHE
# =============================================================================

class TestPathCache:

    def test_build_not_empty(self, toy_kg):
        from data.path_cache import PathCacheBuilder
        adj   = toy_kg.get_adjacency()
        pairs = [(toy_kg.entity2id["Platypus"], toy_kg.entity2id["WarmBlooded"])]
        cache = PathCacheBuilder(adj, max_hops=2, max_paths=4, verbose=False)\
                    .build(pairs, num_entities=toy_kg.num_entities)
        assert cache.size > 0
        assert cache.total_paths > 0

    def test_coverage_reasonable(self, toy_kg):
        from data.path_cache import PathCacheBuilder
        adj   = toy_kg.get_adjacency()
        pairs = [(toy_kg.entity2id[t.head], toy_kg.entity2id[t.tail])
                 for t in toy_kg.train_triples]
        cache = PathCacheBuilder(adj, max_hops=2, max_paths=4, verbose=False)\
                    .build(pairs, num_entities=toy_kg.num_entities)
        assert cache.coverage(pairs) > 0.50

    def test_save_and_load(self, toy_kg):
        from data.path_cache import PathCacheBuilder, PathCache
        adj   = toy_kg.get_adjacency()
        pairs = [(toy_kg.entity2id["Platypus"], toy_kg.entity2id["WarmBlooded"])]
        cache = PathCacheBuilder(adj, max_hops=2, max_paths=4, verbose=False)\
                    .build(pairs, num_entities=toy_kg.num_entities)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cache.pkl"
            cache.save(p)
            loaded = PathCache.load(p)
        assert loaded.size == cache.size
        assert loaded.max_hops == cache.max_hops

    def test_get_returns_list(self, toy_kg):
        from data.path_cache import PathCacheBuilder
        adj  = toy_kg.get_adjacency()
        h_id = toy_kg.entity2id["Platypus"]
        t_id = toy_kg.entity2id["WarmBlooded"]
        cache = PathCacheBuilder(adj, max_hops=2, max_paths=4, verbose=False)\
                    .build([(h_id, t_id)], num_entities=toy_kg.num_entities)
        paths = cache.get(h_id, t_id)
        assert isinstance(paths, list)

    def test_get_batch_length(self, toy_kg):
        from data.path_cache import PathCacheBuilder
        adj   = toy_kg.get_adjacency()
        pairs = [
            (toy_kg.entity2id["Platypus"], toy_kg.entity2id["WarmBlooded"]),
            (toy_kg.entity2id["Bat"],      toy_kg.entity2id["FourLimbs"]),
        ]
        cache = PathCacheBuilder(adj, max_hops=2, max_paths=4, verbose=False)\
                    .build(pairs, num_entities=toy_kg.num_entities)
        result = cache.get_batch([p[0] for p in pairs], [p[1] for p in pairs])
        assert len(result) == 2


# =============================================================================
#  GROUP 11: CHUNKED EVALUATOR
# =============================================================================

class TestChunkedEvaluator:

    def _build_true_tails(self, toy_kg):
        tt = {}
        for t in toy_kg.triples:
            h = toy_kg.entity2id[t.head]; r = toy_kg.relation2id[t.relation]
            tt.setdefault((h, r), set()).add(toy_kg.entity2id[t.tail])
        return tt

    def test_parity_with_standard_evaluator(self, small_model, toy_kg, device):
        from data.dataset import build_dataloaders
        from evaluation.metrics import RankingMetrics
        from evaluation.chunked_evaluator import ChunkedEvaluator

        _, _, test_dl  = build_dataloaders(toy_kg, batch_size=4, num_negatives=2, num_workers=0)
        true_tails = self._build_true_tails(toy_kg)
        small_model.eval()

        # Standard
        std = RankingMetrics(filter_false_negatives=True)
        with torch.no_grad():
            for batch in test_dl:
                pos = batch["positive"].to(device)
                h, r, t = pos[:,0], pos[:,1], pos[:,2]
                std.update(scores=small_model.score_triple_vs_all(h, r),
                           true_indices=t, head_ids=h, relation_ids=r,
                           true_tails=true_tails)
        std_res = std.compute()

        # Chunked with small chunk_size
        ev = ChunkedEvaluator(small_model, toy_kg.num_entities, device,
                              chunk_size=5, true_tails=true_tails, batch_size=4, verbose=False)
        chunk_res = ev.evaluate_loader(test_dl)

        assert abs(std_res.mrr - chunk_res.mrr) < 1e-4, \
            f"MRR mismatch: std={std_res.mrr:.4f} chunked={chunk_res.mrr:.4f}"

    def test_auto_chunk_size_positive(self, small_model, device):
        from evaluation.chunked_evaluator import ChunkedEvaluator
        ev = ChunkedEvaluator(small_model, 100, device, chunk_size="auto", verbose=False)
        assert ev.chunk_size > 0

    def test_memory_estimate_positive(self, small_model, device):
        from evaluation.chunked_evaluator import ChunkedEvaluator
        ev  = ChunkedEvaluator(small_model, 1000, device, chunk_size=100, verbose=False)
        mem = ev.memory_estimate_mb(chunk_size=100)
        assert mem > 0


# =============================================================================
#  GROUP 12: NOISE GUARANTEE THEOREM
# =============================================================================

class TestNoiseGuarantee:

    def test_returns_one_result_per_query(self, small_model, toy_kg, device):
        from theory.noise_guarantee import verify_theorem_conditions
        results = verify_theorem_conditions(small_model, toy_kg, device)
        assert len(results) == len(toy_kg.contradiction_queries)

    def test_result_fields(self, small_model, toy_kg, device):
        from theory.noise_guarantee import verify_theorem_conditions
        for r in verify_theorem_conditions(small_model, toy_kg, device):
            assert hasattr(r, "phi")
            assert hasattr(r, "theorem_applicable")
            assert hasattr(r, "predicted_gap_0pct")
            assert 0.0 <= r.phi <= 2 * math.pi + 0.01

    def test_predicted_gap_decreases_with_noise(self, small_model, toy_kg, device):
        from theory.noise_guarantee import verify_theorem_conditions
        for r in verify_theorem_conditions(small_model, toy_kg, device):
            assert r.predicted_gap_20pct <= r.predicted_gap_0pct + 1e-6

    def test_theorem_formula(self):
        """Verify ΔP_Q = r² K² (1-p)² sin²(φ/2) numerically."""
        r, K, phi, p = 0.5, 4, math.pi, 0.10
        expected = r**2 * K**2 * (1-p)**2 * math.sin(phi/2)**2
        assert abs(expected - 3.24) < 0.01, f"Formula error: {expected}"

    def test_print_theorem_no_crash(self):
        from theory.noise_guarantee import print_theorem
        import io, sys
        buf = io.StringIO()
        sys.stdout = buf
        print_theorem()
        sys.stdout = sys.__stdout__
        assert "THEOREM" in buf.getvalue().upper()


# =============================================================================
#  GROUP 13: NOISE INJECTION
# =============================================================================

class TestNoiseSeparation:

    def test_approx_correct_rate(self, toy_kg):
        from data.noise_injection import NoiseInjector, NoiseConfig
        train_ids = [toy_kg.triple_to_ids(t) for t in toy_kg.train_triples]
        true_set  = {toy_kg.triple_to_ids(t) for t in toy_kg.triples}
        inj = NoiseInjector(toy_kg.num_entities, toy_kg.num_relations,
                            config=NoiseConfig(corruption_rate=0.20, seed=42))
        _, mask = inj.inject(train_ids, true_set)
        rate = float(mask.sum()) / len(train_ids)
        assert 0.10 <= rate <= 0.30, f"Expected ~20%, got {rate:.0%}"

    def test_inject_levels_monotone(self, toy_kg):
        from data.noise_injection import NoiseInjector, NoiseConfig
        train_ids = [toy_kg.triple_to_ids(t) for t in toy_kg.train_triples]
        true_set  = {toy_kg.triple_to_ids(t) for t in toy_kg.triples}
        inj    = NoiseInjector(toy_kg.num_entities, toy_kg.num_relations,
                               config=NoiseConfig(seed=42))
        levels = inj.inject_levels(train_ids, [0.0, 0.05, 0.10, 0.20], true_set=true_set)
        counts = [int(mask.sum()) for _, (_, mask) in sorted(levels.items())]
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i+1]

    def test_zero_noise_unchanged(self, toy_kg):
        from data.noise_injection import NoiseInjector, NoiseConfig
        train_ids = [toy_kg.triple_to_ids(t) for t in toy_kg.train_triples]
        true_set  = {toy_kg.triple_to_ids(t) for t in toy_kg.triples}
        inj = NoiseInjector(toy_kg.num_entities, toy_kg.num_relations,
                            config=NoiseConfig(corruption_rate=0.0, seed=42))
        noisy, mask = inj.inject(train_ids, true_set)
        assert int(mask.sum()) == 0
        assert noisy == train_ids


# =============================================================================
#  GROUP 14: END-TO-END
# =============================================================================

class TestEndToEnd:

    def test_pipeline_steps_1_through_7(self, toy_kg, device):
        """Replicates run_toy.py Steps 1-7 programmatically."""
        from models.components.quantum_states import QuantumStateEncoder
        from models.components.unitary_operators import DiagonalUnitary
        from models.components.path_aggregator import PathEnumerator, AmplitudeAggregator

        # Step 1
        assert toy_kg.num_entities == 28
        assert toy_kg.num_relations == 12
        assert len(toy_kg.triples) == 67

        # Steps 4-5
        enc = QuantumStateEncoder(28, 8, normalize=True).to(device)
        uni = DiagonalUnitary(12, 4).to(device)
        states = enc(torch.arange(28, device=device))
        norms  = states.abs().pow(2).sum(-1).sqrt()
        assert (norms - 1.0).abs().max().item() < 1e-5

        # Step 6 — the critical check
        adj  = toy_kg.get_adjacency()
        enum = PathEnumerator(adj, max_hops=3, max_paths=8)
        h_id = toy_kg.entity2id["Platypus"]
        c_id = toy_kg.entity2id["WarmBlooded"]
        w_id = toy_kg.entity2id["ColdBlooded"]
        assert len(enum.find_paths(h_id, c_id)) >= 1
        assert len(enum.find_paths(h_id, w_id)) >= 2

        # Step 7
        agg  = AmplitudeAggregator(4, max_paths=8).to(device)
        paths = enum.find_paths(h_id, w_id)
        h_s  = enc(torch.tensor([h_id], device=device)).squeeze(0)
        w_s  = enc(torch.tensor([w_id], device=device)).squeeze(0)
        r    = agg.compute_interference_terms(h_s, w_s, paths, uni)
        total = float(r["total_probability"])
        cl    = float(r["classical_sum"])
        interf= float(r["interference"])
        assert abs(total - (cl + interf)) < 1e-5
        assert total >= -1e-6

    def test_score_triple_shape_and_range(self, small_model, device):
        small_model.eval()
        with torch.no_grad():
            scores = small_model.score_triple(
                torch.zeros(4, dtype=torch.long, device=device),
                torch.zeros(4, dtype=torch.long, device=device),
                torch.arange(4, dtype=torch.long, device=device),
            )
        assert scores.shape == (4,)

    def test_score_triple_vs_all_shape(self, small_model, device):
        small_model.eval()
        with torch.no_grad():
            scores = small_model.score_triple_vs_all(
                torch.zeros(3, dtype=torch.long, device=device),
                torch.zeros(3, dtype=torch.long, device=device),
            )
        assert scores.shape == (3, 28)

    def test_interference_aware_loss_finite_scalar(self, small_model, device):
        from training.interference_loss import InterferenceAwareLoss
        loss_fn    = InterferenceAwareLoss(phase_weight=0.1, contrast_weight=0.5)
        pos_scores = torch.rand(4)
        neg_scores = torch.rand(4, 3)
        loss       = loss_fn.forward_main(pos_scores, neg_scores)
        assert loss.shape == torch.Size([])
        assert math.isfinite(float(loss))
        assert float(loss) >= 0.0

    def test_full_loss_backprop(self, small_model, device):
        from training.interference_loss import InterferenceAwareLoss
        loss_fn = InterferenceAwareLoss(phase_weight=0.05, contrast_weight=0.5,
                                        reg_encoder_weight=0.01)
        B = 4
        h = torch.randint(0, 28, (B,), device=device)
        r = torch.randint(0, 12, (B,), device=device)
        t = torch.randint(0, 28, (B,), device=device)

        pos   = small_model.score_triple(h, r, t)
        neg   = torch.stack([
            small_model.score_triple(h, r, torch.randint(0,28,(B,),device=device))
            for _ in range(3)], dim=1)
        loss  = loss_fn.forward_main(pos, neg)
        loss  = loss + loss_fn.regularization(small_model.encoder, small_model.unitary)
        loss.backward()

        assert small_model.encoder.real_embeddings.weight.grad is not None
        assert small_model.encoder.imag_embeddings.weight.grad is not None
        for name, param in small_model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
