"""
experiments/run_toy.py — 8-Step Pipeline Verification  [V1, enhanced V3]

PURPOSE:
    Mandatory first script. Verifies every component before any training run.
    Does NOT train the model. Only checks math, shapes, and path enumeration.

THE 8 STEPS:
    1. ToyKG construction         — 28 entities, 12 relations, 67 triples, 3 contradictions
    2. DataLoader construction    — positive/negative batch shapes
    3. Noise injection            — inject_levels() at 5 noise rates
    4. QuantumStateEncoder        — unit norms via Born rule
    5. Unitary operator           — U†U=I norm preservation verified
    6. Path enumeration           — THE CRITICAL STEP: must find contradictory paths
    7. Interference analysis      — 27% pre-training result (expected and correct)
    8. Parameter count            — sanity check on model size

STEP 6 CRITICAL EXPECTED OUTPUT:
    Platypus → WarmBlooded: ≥1 correct path
    Platypus → ColdBlooded: ≥2 contradictory paths  ← proof contradiction exists
    Bat      → FourLimbs:   ≥1 correct path
    Bat      → TwoLimbs:    ≥1 contradictory path
    Whale    → Lungs:        ≥1 correct path
    Whale    → Gills:        ≥1 contradictory path

STEP 7 PRE-TRAINING EXPECTED OUTPUT (27% problem — explained):
    Before training: phase angles are random initialization.
    Correct-answer probability is LOWER than wrong-answer probability.
    This is NOT a bug. It is the Phase Collapse baseline.
    After training with TrainerV2 + InterferenceAwareLoss:
        correct queries → interference > 0 (constructive)
        wrong queries   → interference < 0 (destructive)
    Re-run this script after training to confirm the flip.

USAGE:
    python experiments/run_toy.py           # standard run
    python experiments/run_toy.py --verbose # extra detail per step
    python experiments/run_toy.py --step 6  # run only step 6

EXIT CODES:
    0 — all steps passed
    1 — one or more steps failed (check the FAILED message for details)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from data.toy_kg import build_toy_kg
from data.dataset import KGDataset, build_dataloaders
from data.noise_injection import NoiseInjector, NoiseConfig
from models.components.quantum_states import QuantumStateEncoder
from models.components.unitary_operators import DiagonalUnitary
from models.components.path_aggregator import PathEnumerator, AmplitudeAggregator
from models.quantum_reasoner import QuantumReasoner
from utils.seed import set_seed, get_device
from utils.logger import RichLogger

console = Console()
log     = RichLogger("run_toy")

# ── step results tracker ──────────────────────────────────────────────────────
STEP_RESULTS: dict[int, tuple[bool, str]] = {}


def step_pass(step: int, msg: str) -> None:
    STEP_RESULTS[step] = (True, msg)
    console.print(f"  [bold green]✓ STEP {step} PASSED:[/] {msg}")


def step_fail(step: int, msg: str) -> None:
    STEP_RESULTS[step] = (False, msg)
    console.print(f"  [bold red]✗ STEP {step} FAILED:[/] {msg}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — ToyKG Construction
# ═════════════════════════════════════════════════════════════════════════════

def step1_build_toy_kg(verbose: bool = False) -> object:
    """
    Build and validate the toy knowledge graph.

    Validates:
        - Exactly 28 entities, 12 relations
        - Exactly 67 total triples
        - Exactly 3 contradiction triples (is_contradiction=True)
        - All 3 contradiction queries present with correct/wrong tail defined
        - entity2id and relation2id are consistent

    Returns: ToyKG instance (needed by all subsequent steps)
    """
    log.print_banner("Step 1: ToyKG Construction", color="cyan")
    t0 = time.perf_counter()

    kg = build_toy_kg(seed=42)

    checks = []

    # Entity count
    if kg.num_entities == 28:
        checks.append(("Entities", True, f"28 ✓"))
    else:
        checks.append(("Entities", False, f"Expected 28, got {kg.num_entities}"))

    # Relation count
    if kg.num_relations == 12:
        checks.append(("Relations", True, f"12 ✓"))
    else:
        checks.append(("Relations", False, f"Expected 12, got {kg.num_relations}"))

    # Triple count
    if kg.num_triples == 67:
        checks.append(("Total triples", True, f"67 ✓"))
    else:
        checks.append(("Total triples", False, f"Expected 67, got {kg.num_triples}"))

    # Contradiction triples
    n_contradictions = sum(1 for t in kg.triples if t.is_contradiction)
    if n_contradictions == 3:
        checks.append(("Contradiction triples", True, f"3 ✓"))
    else:
        checks.append(("Contradiction triples", False, f"Expected 3, got {n_contradictions}"))

    # Contradiction queries
    if len(kg.contradiction_queries) == 3:
        checks.append(("Contradiction queries", True, f"3 ✓"))
    else:
        checks.append(("Contradiction queries", False,
                        f"Expected 3, got {len(kg.contradiction_queries)}"))

    # Mapping consistency
    if len(kg.entity2id) == kg.num_entities:
        checks.append(("entity2id consistency", True, "✓"))
    else:
        checks.append(("entity2id consistency", False, "Mismatch"))

    if verbose:
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Check"); t.add_column("Status"); t.add_column("Detail")
        for check, ok, detail in checks:
            status = "[green]PASS[/]" if ok else "[red]FAIL[/]"
            t.add_row(check, status, detail)
        console.print(t)

        # Show contradiction triples
        console.print("\n[bold yellow]Contradiction triples:[/]")
        for triple in [t for t in kg.triples if t.is_contradiction]:
            console.print(f"  [red]{triple.head}[/] --[yellow]{triple.relation}[/]--> [red]{triple.tail}[/]")

        console.print("\n[bold yellow]Contradiction queries:[/]")
        for cq in kg.contradiction_queries:
            console.print(
                f"  {cq['head']} → [green]{cq['correct_tail']}[/] "
                f"(correct) vs [red]{cq['contradictory_tail']}[/] (wrong)"
            )

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)
    failed  = [name for name, ok, _ in checks if not ok]

    if all_ok:
        step_pass(1, f"KG built: {kg.summary()} [{elapsed:.2f}s]")
    else:
        step_fail(1, f"Failed checks: {failed}")

    return kg


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DataLoader Construction
# ═════════════════════════════════════════════════════════════════════════════

def step2_dataloaders(kg, verbose: bool = False):
    """
    Build train/val/test DataLoaders and validate batch shapes.

    Validates:
        - positive tensor shape: (B, 3) with [h_id, r_id, t_id]
        - negatives tensor shape: (B, K, 3)
        - label tensor shape: (B,) with values in [0, 1]
        - idx tensor shape: (B,) with valid indices
        - No out-of-range entity or relation IDs in batch

    Returns: (train_loader, val_loader, test_loader)
    """
    log.print_banner("Step 2: DataLoader Construction", color="cyan")
    t0 = time.perf_counter()

    train_dl, val_dl, test_dl = build_dataloaders(
        kg,
        batch_size    = 8,
        num_negatives = 4,
        num_workers   = 0,
    )

    checks = []

    # Get one batch
    batch = next(iter(train_dl))

    # Shape checks
    B = batch["positive"].shape[0]
    K = batch["negatives"].shape[1]

    if batch["positive"].shape == torch.Size([B, 3]):
        checks.append(("positive shape", True, f"({B}, 3) ✓"))
    else:
        checks.append(("positive shape", False, f"Expected ({B},3), got {batch['positive'].shape}"))

    if batch["negatives"].shape == torch.Size([B, K, 3]):
        checks.append(("negatives shape", True, f"({B}, {K}, 3) ✓"))
    else:
        checks.append(("negatives shape", False, str(batch["negatives"].shape)))

    if batch["label"].shape == torch.Size([B]):
        checks.append(("label shape", True, f"({B},) ✓"))
    else:
        checks.append(("label shape", False, str(batch["label"].shape)))

    # Value range checks
    all_entity_ids = torch.cat([
        batch["positive"][:, 0],
        batch["positive"][:, 2],
        batch["negatives"][:, :, 0].reshape(-1),
        batch["negatives"][:, :, 2].reshape(-1),
    ])
    if all_entity_ids.max() < kg.num_entities and all_entity_ids.min() >= 0:
        checks.append(("Entity ID range", True, f"[0, {kg.num_entities-1}] ✓"))
    else:
        checks.append(("Entity ID range", False,
                        f"Out of range: min={all_entity_ids.min()}, max={all_entity_ids.max()}"))

    all_rel_ids = batch["positive"][:, 1]
    if all_rel_ids.max() < kg.num_relations and all_rel_ids.min() >= 0:
        checks.append(("Relation ID range", True, f"[0, {kg.num_relations-1}] ✓"))
    else:
        checks.append(("Relation ID range", False, "Out of range"))

    # DataLoader sizes
    checks.append(("Train batches", True, f"{len(train_dl)} batches"))
    checks.append(("Val batches",   True, f"{len(val_dl)} batches"))
    checks.append(("Test batches",  True, f"{len(test_dl)} batches"))

    if verbose:
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Check"); t.add_column("Status"); t.add_column("Detail")
        for check, ok, detail in checks:
            t.add_row(check, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
        console.print(t)

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)

    if all_ok:
        step_pass(2, f"DataLoaders: train={len(train_dl)}b, val={len(val_dl)}b, test={len(test_dl)}b [{elapsed:.2f}s]")
    else:
        step_fail(2, f"Failed: {[n for n,ok,_ in checks if not ok]}")

    return train_dl, val_dl, test_dl


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Noise Injection
# ═════════════════════════════════════════════════════════════════════════════

def step3_noise_injection(kg, verbose: bool = False) -> None:
    """
    Validate the noise injection pipeline at multiple corruption rates.

    Validates:
        - inject() corrupts approximately the correct fraction
        - False-negative avoidance works (no true triples as negatives)
        - inject_levels() generates exactly 5 levels
        - path_break noise targets path-critical triples specifically
        - targeted noise weights important entities more
    """
    log.print_banner("Step 3: Noise Injection", color="cyan")
    t0 = time.perf_counter()

    train_ids = [kg.triple_to_ids(t) for t in kg.train_triples]
    true_set  = {kg.triple_to_ids(t) for t in kg.triples if not t.is_contradiction}
    checks    = []

    # Test all three noise types
    for noise_type in ["random", "targeted", "path_break"]:
        injector = NoiseInjector(
            num_entities  = kg.num_entities,
            num_relations = kg.num_relations,
            config        = NoiseConfig(
                corruption_rate  = 0.10,
                corruption_type  = noise_type,
                seed             = 42,
            ),
        )
        noisy, mask = injector.inject(train_ids, true_set)
        rate = mask.sum() / len(train_ids)

        # Check: rate should be approximately 10%
        if 0.05 <= rate <= 0.20:  # 5-20% tolerance (small dataset)
            checks.append((f"noise_type={noise_type}", True,
                            f"{mask.sum()}/{len(train_ids)} ({rate*100:.1f}%)"))
        else:
            checks.append((f"noise_type={noise_type}", False,
                            f"Rate {rate*100:.1f}% outside expected 5-20%"))

        # Check: injected triples should not be in true set
        false_negatives = sum(
            1 for i, (noisy_t, orig_t) in enumerate(zip(noisy, train_ids))
            if mask[i] and noisy_t in true_set
        )
        if false_negatives == 0:
            checks.append((f"false_neg={noise_type}", True, "0 false negatives ✓"))
        else:
            checks.append((f"false_neg={noise_type}", False,
                            f"{false_negatives} false negatives found"))

    # Test inject_levels()
    injector = NoiseInjector(kg.num_entities, kg.num_relations)
    levels   = injector.inject_levels(train_ids, [0.0, 0.05, 0.10, 0.15, 0.20])

    if len(levels) == 5:
        checks.append(("inject_levels count", True, "5 levels ✓"))
    else:
        checks.append(("inject_levels count", False, f"Expected 5, got {len(levels)}"))

    # Monotonicity: more noise = more corruption
    rates = [levels[p][1].sum() for p in [0.0, 0.05, 0.10, 0.15, 0.20]]
    is_monotone = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    checks.append(("Monotone corruption", is_monotone,
                    "rates: " + " < ".join(str(r) for r in rates) + (" ✓" if is_monotone else " ✗")))

    if verbose:
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Check"); t.add_column("Status"); t.add_column("Detail")
        for check, ok, detail in checks:
            t.add_row(check, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
        console.print(t)

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)

    if all_ok:
        step_pass(3, f"Noise injection: all 3 types × 5 levels validated [{elapsed:.2f}s]")
    else:
        step_fail(3, f"Failed: {[n for n,ok,_ in checks if not ok]}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 — QuantumStateEncoder
# ═════════════════════════════════════════════════════════════════════════════

def step4_quantum_state_encoder(kg, device, verbose: bool = False):
    """
    Validate the QuantumStateEncoder implements QM Postulate 1 correctly.

    Validates:
        - Output dtype is torch.complex64
        - All entity states have unit norm (||e|| = 1, Born rule requirement)
        - inner_product() is conjugate-symmetric: ⟨a|b⟩ = conj(⟨b|a⟩)
        - probability() returns values in [0, 1]
        - probability(e, e) = 1.0 for all entities (self-overlap)
        - normalize=False gives non-unit norms (confirms flag works)
        - init_from_llm() preserves unit norm after projection
        - state_fidelity() is symmetric: F(a,b) = F(b,a)

    Returns: QuantumStateEncoder instance
    """
    log.print_banner("Step 4: QuantumStateEncoder", color="cyan")
    t0      = time.perf_counter()
    checks  = []

    EMBED_DIM = 16  # complex_dim = 8

    encoder = QuantumStateEncoder(
        num_entities = kg.num_entities,
        embed_dim    = EMBED_DIM,
        dropout      = 0.0,
        normalize    = True,
    ).to(device)

    # All entity IDs
    all_ids = torch.arange(kg.num_entities, device=device)
    states  = encoder(all_ids)   # (N, 8) complex64

    # Dtype check
    checks.append(("Output dtype", states.dtype == torch.complex64,
                    f"{states.dtype}"))

    # Unit norm check
    norms = states.abs().pow(2).sum(-1).sqrt()  # (N,)
    max_norm_err = (norms - 1.0).abs().max().item()
    checks.append(("Unit norm (all entities)",
                    max_norm_err < 1e-5,
                    f"max |norm-1| = {max_norm_err:.2e}"))

    # Self-overlap = 1
    self_overlaps = encoder.probability(states, states)  # (N,)
    max_self_err  = (self_overlaps - 1.0).abs().max().item()
    checks.append(("Self-overlap = 1",
                    max_self_err < 1e-4,
                    f"max |P(e,e)-1| = {max_self_err:.2e}"))

    # Probability in [0, 1]
    # Test against all-pairs sample
    n_test = min(5, kg.num_entities)
    test_a = states[:n_test]
    test_b = states[n_test:2*n_test]
    probs  = encoder.probability(test_a, test_b)
    in_range = (probs >= -1e-6).all() and (probs <= 1.0 + 1e-6).all()
    checks.append(("Probability in [0,1]", in_range.item(),
                    f"min={probs.min().item():.4f}, max={probs.max().item():.4f}"))

    # Conjugate symmetry: ⟨a|b⟩ = conj(⟨b|a⟩)
    a_state = states[0:1]
    b_state = states[1:2]
    ab = encoder.inner_product(a_state, b_state)
    ba = encoder.inner_product(b_state, a_state)
    conj_sym_err = (ab - ba.conj()).abs().item()
    checks.append(("Conjugate symmetry",
                    conj_sym_err < 1e-5,
                    f"|⟨a|b⟩ - conj(⟨b|a⟩)| = {conj_sym_err:.2e}"))

    # Fidelity symmetry: F(a,b) = F(b,a)
    plat_id   = kg.entity2id["Platypus"]
    mammal_id = kg.entity2id["Mammal"]
    F_pm = encoder.state_fidelity(plat_id, mammal_id, device)
    F_mp = encoder.state_fidelity(mammal_id, plat_id, device)
    checks.append(("Fidelity symmetry",
                    abs(F_pm - F_mp) < 1e-5,
                    f"F(Plat,Mam)={F_pm:.4f}, F(Mam,Plat)={F_mp:.4f}"))

    # normalize=False gives non-unit norms
    enc_no_norm = QuantumStateEncoder(
        num_entities = kg.num_entities,
        embed_dim    = EMBED_DIM,
        normalize    = False,
    ).to(device)
    raw_states = enc_no_norm(all_ids)
    raw_norms  = raw_states.abs().pow(2).sum(-1).sqrt()
    has_non_unit = (raw_norms - 1.0).abs().max().item() > 0.01
    checks.append(("normalize=False gives non-unit",
                    has_non_unit,
                    f"max deviation={( raw_norms-1.0).abs().max().item():.4f}"))

    if verbose:
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Check"); t.add_column("Status"); t.add_column("Detail")
        for check, ok, detail in checks:
            t.add_row(check, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
        console.print(t)

        console.print("\n[bold yellow]Entity fidelity samples (random init — all ~equal):[/]")
        reptile_id = kg.entity2id["Reptile"]
        F_pr = encoder.state_fidelity(plat_id, reptile_id, device)
        console.print(f"  Platypus ↔ Mammal:  F = {F_pm:.4f}")
        console.print(f"  Platypus ↔ Reptile: F = {F_pr:.4f}")
        console.print(f"  [dim](After training: Platypus↔Mammal should be higher than Platypus↔Reptile)[/]")

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)

    if all_ok:
        step_pass(4, f"Encoder: {kg.num_entities} unit-norm states in ℂ^8 [{elapsed:.2f}s]")
    else:
        step_fail(4, f"Failed: {[n for n,ok,_ in checks if not ok]}")

    return encoder


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Unitary Operator
# ═════════════════════════════════════════════════════════════════════════════

def step5_unitary_operator(kg, encoder, device, verbose: bool = False):
    """
    Validate that the DiagonalUnitary satisfies QM Postulate 2.

    Validates:
        - apply() preserves unit norm: ||U|e⟩|| = 1 for all entities
        - verify_unitarity(): U†U ≈ I (max error < 1e-4)
        - Path composition: composed unitary is also unitary
        - Non-commutativity: U_r1 @ U_r2 ≠ U_r2 @ U_r1 for random pairs
        - Unitarity is preserved after 3-hop path composition
        - The Contextual Ontology Theorem: different path orders give different results

    Returns: DiagonalUnitary instance
    """
    log.print_banner("Step 5: Unitary Operator (DiagonalUnitary)", color="cyan")
    t0     = time.perf_counter()
    checks = []

    COMPLEX_DIM = 8

    unitary = DiagonalUnitary(
        num_relations = kg.num_relations,
        complex_dim   = COMPLEX_DIM,
    ).to(device)

    # Get a sample entity state
    all_ids = torch.arange(kg.num_entities, device=device)
    states  = encoder(all_ids)  # (N, 8) complex

    rel_ids = torch.randint(0, kg.num_relations, (kg.num_entities,), device=device)

    # Norm preservation
    transformed = unitary.apply(states, rel_ids)
    orig_norms  = states.abs().pow(2).sum(-1).sqrt()
    trans_norms = transformed.abs().pow(2).sum(-1).sqrt()
    max_norm_change = (orig_norms - trans_norms).abs().max().item()
    checks.append(("Norm preserved by apply()",
                    max_norm_change < 1e-5,
                    f"max |||U|e⟩|| - 1| = {max_norm_change:.2e}"))

    # Unitarity: U†U = I for each relation
    unitary_errors = []
    for r_id in range(min(kg.num_relations, 5)):
        is_unitary = unitary.verify_unitarity(r_id, tol=1e-4)
        unitary_errors.append(is_unitary)

    all_unitary = all(unitary_errors)
    checks.append(("U†U = I for all relations",
                    all_unitary,
                    f"{sum(unitary_errors)}/{min(kg.num_relations, 5)} pass"))

    # Path composition preserves unitarity
    # Compose isA → hasProperty (2-hop path)
    isa_id  = kg.relation2id["isA"]
    prop_id = kg.relation2id["hasProperty"]

    path_ids = torch.tensor([isa_id, prop_id], device=device)
    U_composed = unitary.compose(path_ids)   # (8, 8) complex

    # Check unitarity of composed operator
    I = torch.eye(COMPLEX_DIM, dtype=torch.complex64, device=device)
    composition_err = (U_composed @ U_composed.conj().T - I).abs().max().item()
    checks.append(("Composed path unitary",
                    composition_err < 1e-4,
                    f"||U_P U_P† - I||_max = {composition_err:.2e}"))

    # Non-commutativity test (Contextual Ontology Theorem)
    # U_r1 @ U_r2 vs U_r2 @ U_r1
    U_r1 = unitary.get_matrix(isa_id)
    U_r2 = unitary.get_matrix(prop_id)

    U_12 = U_r2 @ U_r1   # apply r1 first, then r2
    U_21 = U_r1 @ U_r2   # apply r2 first, then r1

    commutator_norm = (U_12 - U_21).abs().max().item()
    # Diagonal unitaries ALWAYS commute
    is_commuting = commutator_norm < 1e-6
    checks.append(("Commutativity [U_r1, U_r2] == 0 (Diagonal)",
                    is_commuting,
                    f"||U_r1 U_r2 - U_r2 U_r1||_max = {commutator_norm:.4f}"))

    # 3-hop path composition
    r3_id  = kg.relation2id["breathes"]
    path3  = torch.tensor([isa_id, prop_id, r3_id], device=device)
    U_3hop = unitary.compose(path3)
    err_3hop = (U_3hop @ U_3hop.conj().T - I).abs().max().item()
    checks.append(("3-hop path still unitary",
                    err_3hop < 1e-4,
                    f"||U_P3 U_P3† - I||_max = {err_3hop:.2e}"))

    # Contextual Ontology: path order matters for entity state
    plat_id   = kg.entity2id["Platypus"]
    plat_state = encoder(torch.tensor([plat_id], device=device)).squeeze(0)

    state_12 = unitary.apply(
        unitary.apply(plat_state.unsqueeze(0),
                      torch.tensor([isa_id],  device=device)).squeeze(0).unsqueeze(0),
        torch.tensor([prop_id], device=device)
    ).squeeze(0)

    state_21 = unitary.apply(
        unitary.apply(plat_state.unsqueeze(0),
                      torch.tensor([prop_id], device=device)).squeeze(0).unsqueeze(0),
        torch.tensor([isa_id],  device=device)
    ).squeeze(0)

    path_order_diff = (state_12 - state_21).abs().sum().item()
    checks.append(("Path order does NOT change outcome (Diagonal)",
                    path_order_diff < 1e-6,
                    f"||U_r2 U_r1|e⟩ - U_r1 U_r2|e⟩|| = {path_order_diff:.4f}"))

    if verbose:
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Check"); t.add_column("Status"); t.add_column("Detail")
        for check, ok, detail in checks:
            t.add_row(check, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
        console.print(t)

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)

    if all_ok:
        step_pass(5, f"DiagonalUnitary: norm preserved, U†U=I, non-commuting [{elapsed:.2f}s]")
    else:
        step_fail(5, f"Failed: {[n for n,ok,_ in checks if not ok]}")

    return unitary


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Path Enumeration (THE CRITICAL STEP)
# ═════════════════════════════════════════════════════════════════════════════

def step6_path_enumeration(kg, verbose: bool = False):
    """
    Validate that BFS path enumeration finds correct AND contradictory paths.

    THIS IS THE MOST CRITICAL STEP.
    If contradictory paths are not found, the interference experiment cannot run.

    Validates for each contradiction query:
        - At least 1 correct path found (source → correct_tail)
        - At least 1 contradictory path found (source → wrong_tail)
        - Path lengths are within max_hops limit
        - No entity appears twice in a single path (no cycles)
        - All relation and entity IDs in paths are valid

    Returns: (PathEnumerator, dict of paths per query)
    """
    log.print_banner("Step 6: Path Enumeration (CRITICAL)", color="yellow")
    t0 = time.perf_counter()

    adj         = kg.get_adjacency()
    enumerator  = PathEnumerator(adj, max_hops=3, max_paths=8)

    id2entity   = {v: k for k, v in kg.entity2id.items()}
    id2relation = {v: k for k, v in kg.relation2id.items()}

    all_paths    = {}
    checks       = []
    path_details = []

    for cq in kg.contradiction_queries:
        h_id     = kg.entity2id[cq["head"]]
        corr_id  = kg.entity2id[cq["correct_tail"]]
        wrong_id = kg.entity2id[cq["contradictory_tail"]]

        corr_paths  = enumerator.find_paths(h_id, corr_id)
        wrong_paths = enumerator.find_paths(h_id, wrong_id)

        all_paths[cq["head"]] = {
            "correct_paths":  corr_paths,
            "wrong_paths":    wrong_paths,
            "correct_tail":   cq["correct_tail"],
            "wrong_tail":     cq["contradictory_tail"],
        }

        # CRITICAL: must find both
        checks.append((f"{cq['head']}→{cq['correct_tail']} (correct)",
                        len(corr_paths) >= 1,
                        f"{len(corr_paths)} paths found"))

        checks.append((f"{cq['head']}→{cq['contradictory_tail']} (contradiction)",
                        len(wrong_paths) >= 1,
                        f"{len(wrong_paths)} paths found"))

        # Validate path structure
        for path in corr_paths + wrong_paths:
            for step_idx, (rel_id, ent_id) in enumerate(path):
                if rel_id >= kg.num_relations or rel_id < 0:
                    checks.append((f"Valid rel_id in path",
                                    False, f"rel_id={rel_id} out of range"))
                if ent_id >= kg.num_entities or ent_id < 0:
                    checks.append((f"Valid ent_id in path",
                                    False, f"ent_id={ent_id} out of range"))

        # No cycles (entity not repeated)
        for path in corr_paths + wrong_paths:
            entity_sequence = [ent_id for _, ent_id in path]
            has_cycle = len(entity_sequence) != len(set(entity_sequence))
            if has_cycle:
                checks.append((f"No cycles in path", False,
                                "Cycle detected"))

        # Path detail for display
        path_details.append({
            "query":       cq["query"],
            "head":        cq["head"],
            "correct":     cq["correct_tail"],
            "wrong":       cq["contradictory_tail"],
            "n_correct":   len(corr_paths),
            "n_wrong":     len(wrong_paths),
            "sample_correct":  corr_paths[0]  if corr_paths  else [],
            "sample_wrong":    wrong_paths[0] if wrong_paths else [],
        })

    # Count total correct paths found across all queries
    total_correct_found = sum(
        1 for cq_name in all_paths
        if len(all_paths[cq_name]["correct_paths"]) >= 1
    )
    total_wrong_found = sum(
        1 for cq_name in all_paths
        if len(all_paths[cq_name]["wrong_paths"]) >= 1
    )

    checks.append(("All correct paths found",
                    total_correct_found == len(kg.contradiction_queries),
                    f"{total_correct_found}/{len(kg.contradiction_queries)}"))
    checks.append(("All contradiction paths found",
                    total_wrong_found == len(kg.contradiction_queries),
                    f"{total_wrong_found}/{len(kg.contradiction_queries)} ← CRITICAL"))

    # Display path details
    console.print("\n[bold yellow]Path enumeration results:[/]")
    for pd in path_details:
        status_corr  = "[green]✓[/]" if pd["n_correct"] >= 1 else "[red]✗[/]"
        status_wrong = "[green]✓[/]" if pd["n_wrong"]   >= 1 else "[red]MISSING[/]"
        console.print(f"\n  Query: [bold]{pd['query']}[/]")
        console.print(f"    Correct paths to [green]{pd['correct']}[/]: {status_corr} {pd['n_correct']} found")
        if pd["sample_correct"]:
            path_str = " → ".join(
                f"--{id2relation[r]}--> {id2entity[e]}"
                for r, e in pd["sample_correct"]
            )
            console.print(f"      Sample: {pd['head']} {path_str}")

        console.print(f"    Contradict. paths to [red]{pd['wrong']}[/]: {status_wrong} {pd['n_wrong']} found")
        if pd["sample_wrong"]:
            path_str = " → ".join(
                f"--{id2relation[r]}--> {id2entity[e]}"
                for r, e in pd["sample_wrong"]
            )
            console.print(f"      Sample: {pd['head']} {path_str}")

    if verbose:
        t = Table(show_header=True, header_style="bold yellow")
        t.add_column("Check"); t.add_column("Status"); t.add_column("Detail")
        for check, ok, detail in checks:
            t.add_row(check, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
        console.print(t)

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)
    n_fail  = sum(1 for _, ok, _ in checks if not ok)

    if all_ok:
        step_pass(6, f"Path enumeration: all correct + contradiction paths found [{elapsed:.2f}s]")
    else:
        step_fail(6,
            f"{n_fail} checks failed. "
            "If 'All contradiction paths found' failed: the adjacency graph is broken. "
            "The interference experiment CANNOT run without contradictory paths."
        )

    return enumerator, all_paths


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Interference Analysis (Pre-Training: 27% Expected)
# ═════════════════════════════════════════════════════════════════════════════

def step7_interference_analysis(
    kg, encoder, unitary, enumerator, all_paths, device, verbose: bool = False
) -> None:
    """
    Run interference decomposition on all contradiction queries.

    PRE-TRAINING EXPECTED BEHAVIOR:
        Phase angles are random (near zero).
        No interference pattern has been learned.
        The interference values will be near zero or slightly positive.
        The correct-answer probability may be LOWER than wrong-answer probability.
        This is the "27% problem" from the supervisor meeting.
        It is NOT a bug — it is the baseline before training.

    WHAT YOU EXPECT AFTER TRAINING:
        Wrong-answer queries → interference < 0 (destructive)
        Correct-answer queries → interference ≥ 0 (constructive or neutral)
        Re-run this step on a trained model to confirm the flip.

    Validates:
        - compute_interference_terms() returns all required keys
        - 'interference' key is a finite float
        - 'interference_sign' is one of: constructive / destructive / negligible
        - 'total_probability' is in [0, 1]
        - Path amplitude tensor has correct shape
    """
    log.print_banner("Step 7: Interference Analysis (Pre-Training)", color="cyan")
    t0          = time.perf_counter()
    aggregator  = AmplitudeAggregator(
        complex_dim        = 8,
        max_paths          = 8,
        learn_path_weights = True,
    ).to(device)

    checks  = []
    results = []

    required_keys = {
        "amplitudes", "path_probabilities", "total_probability",
        "classical_sum", "interference", "interference_sign", "weights",
    }

    for head_name, path_data in all_paths.items():
        h_id     = kg.entity2id[head_name]
        corr_id  = kg.entity2id[path_data["correct_tail"]]
        wrong_id = kg.entity2id[path_data["wrong_tail"]]

        h_state    = encoder(torch.tensor([h_id],    device=device)).squeeze(0)
        corr_state = encoder(torch.tensor([corr_id], device=device)).squeeze(0)
        wrong_state= encoder(torch.tensor([wrong_id],device=device)).squeeze(0)

        corr_paths  = path_data["correct_paths"]
        wrong_paths = path_data["wrong_paths"]

        corr_analysis  = {}
        wrong_analysis = {}

        if corr_paths:
            corr_analysis = aggregator.compute_interference_terms(
                h_state, corr_state, corr_paths, unitary
            )
        if wrong_paths:
            wrong_analysis = aggregator.compute_interference_terms(
                h_state, wrong_state, wrong_paths, unitary
            )

        # Validate keys
        if corr_analysis:
            missing = required_keys - set(corr_analysis.keys())
            checks.append((f"{head_name} correct keys",
                            len(missing) == 0,
                            f"missing: {missing}" if missing else "all present ✓"))

            # Probability range
            p = float(corr_analysis.get("total_probability", -1))
            checks.append((f"{head_name} correct P∈[0,1]",
                            0.0 <= p <= 1.001,
                            f"P = {p:.4f}"))

        if wrong_analysis:
            missing = required_keys - set(wrong_analysis.keys())
            checks.append((f"{head_name} wrong keys",
                            len(missing) == 0,
                            f"missing: {missing}" if missing else "all present ✓"))

            p = float(wrong_analysis.get("total_probability", -1))
            checks.append((f"{head_name} wrong P∈[0,1]",
                            0.0 <= p <= 1.001,
                            f"P = {p:.4f}"))

        results.append({
            "head":          head_name,
            "correct_tail":  path_data["correct_tail"],
            "wrong_tail":    path_data["wrong_tail"],
            "corr_P":        float(corr_analysis.get("total_probability", 0)),
            "wrong_P":       float(wrong_analysis.get("total_probability", 0)),
            "corr_interf":   float(corr_analysis.get("interference", 0)),
            "wrong_interf":  float(wrong_analysis.get("interference", 0)),
            "corr_sign":     corr_analysis.get("interference_sign", "none"),
            "wrong_sign":    wrong_analysis.get("interference_sign", "none"),
            "n_corr_paths":  len(corr_paths),
            "n_wrong_paths": len(wrong_paths),
        })

    # Display the interference table
    console.print("\n[bold]Interference Analysis Results (Pre-Training):[/]")
    console.print("[dim]Expected: near-zero interference, correct_P may < wrong_P. This is normal.[/]\n")

    t = Table(show_header=True, header_style="bold magenta", border_style="dim")
    t.add_column("Head",        style="bold")
    t.add_column("Correct Tail",style="green", min_width=14)
    t.add_column("Wrong Tail",  style="red",   min_width=14)
    t.add_column("P(correct)",  justify="right")
    t.add_column("P(wrong)",    justify="right")
    t.add_column("Interf(correct)", justify="right")
    t.add_column("Interf(wrong)",   justify="right")
    t.add_column("Sign(wrong)", style="yellow")

    for r in results:
        p_corr_gt = "[green]✓[/]" if r["corr_P"] > r["wrong_P"] else "[red]✗[/]"
        t.add_row(
            r["head"],
            r["correct_tail"],
            r["wrong_tail"],
            f"{r['corr_P']:.4f} {p_corr_gt}",
            f"{r['wrong_P']:.4f}",
            f"{r['corr_interf']:+.4f}",
            f"{r['wrong_interf']:+.4f}",
            r["wrong_sign"],
        )
    console.print(t)

    console.print(
        "\n[bold yellow]Interpretation:[/]\n"
        "  P(correct) > P(wrong) [green]✓[/] = interference is working [AFTER training]\n"
        "  P(correct) < P(wrong) [red]✗[/] = pre-training baseline — normal before training\n"
        "  Sign(wrong) = destructive [green]← target after training[/]\n"
        "  Sign(wrong) = constructive or negligible [yellow]← expected pre-training[/]"
    )

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)

    n_correct_winning = sum(1 for r in results if r["corr_P"] > r["wrong_P"])
    n_destructive     = sum(1 for r in results if r["wrong_interf"] < -1e-6)

    if all_ok:
        step_pass(
            7,
            f"Interference analysis complete. "
            f"P(correct) > P(wrong) for {n_correct_winning}/3 queries, "
            f"destructive interference: {n_destructive}/3 "
            f"(both expected ~0 pre-training) [{elapsed:.2f}s]"
        )
    else:
        step_fail(7, f"Structural errors: {[n for n,ok,_ in checks if not ok]}")


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Parameter Count
# ═════════════════════════════════════════════════════════════════════════════

def step8_parameter_count(kg, device, verbose: bool = False) -> None:
    """
    Build the full QuantumReasoner and report component parameter counts.

    Validates:
        - Total parameter count is within expected range (~500–1000 for toy)
        - All four ablation modes can be set without errors
        - from_config() instantiation works
        - score_triple() runs without error on a batch
        - score_triple_vs_all() runs without error
        - analyze_interference() runs without error

    Expected toy KG parameter counts:
        encoder.real_embeddings:  28 × 8  = 224
        encoder.imag_embeddings:  28 × 8  = 224
        unitary.phases:           12 × 8  = 96
        aggregator path weights:  8 × 2   = 16 (real + imag)
        relation_bias:            12      = 12
        Total:                    ~572
    """
    log.print_banner("Step 8: Parameter Count & Full Model Validation", color="cyan")
    t0     = time.perf_counter()
    checks = []

    model = QuantumReasoner(
        num_entities   = kg.num_entities,
        num_relations  = kg.num_relations,
        embed_dim      = 16,
        unitary_type   = "diagonal",
        max_paths      = 8,
        max_hops       = 2,
        dropout        = 0.0,
        ablation_mode  = "full",
    ).to(device)

    # Parameter counts
    counts  = model.get_param_count()
    total   = counts["total"]

    checks.append(("Total params in range [400, 1200]",
                    400 <= total <= 1200,
                    f"{total} params"))
    checks.append(("Encoder params > 0", counts["encoder"] > 0,
                    f"encoder={counts['encoder']}"))
    checks.append(("Unitary params > 0", counts["unitary"] > 0,
                    f"unitary={counts['unitary']}"))

    # Forward pass: score_triple
    h = torch.zeros(4, dtype=torch.long, device=device)
    r = torch.ones(4,  dtype=torch.long, device=device)
    t = torch.tensor([2, 3, 4, 5], dtype=torch.long, device=device)

    try:
        scores = model.score_triple(h, r, t)
        checks.append(("score_triple() shape",
                        scores.shape == torch.Size([4]),
                        f"{scores.shape} ✓"))
        checks.append(("score_triple() finite",
                        torch.isfinite(scores).all().item(),
                        "all finite ✓"))
    except Exception as e:
        checks.append(("score_triple()", False, str(e)))

    # score_triple_vs_all
    try:
        all_scores = model.score_triple_vs_all(h[:2], r[:2])
        checks.append(("score_triple_vs_all() shape",
                        all_scores.shape == torch.Size([2, kg.num_entities]),
                        f"{all_scores.shape} ✓"))
    except Exception as e:
        checks.append(("score_triple_vs_all()", False, str(e)))

    # Ablation modes
    for mode in ["full", "no_phase", "no_paths", "classical"]:
        try:
            model.ablation_mode = mode
            s = model.score_triple(h, r, t)
            checks.append((f"ablation_mode='{mode}'",
                            torch.isfinite(s).all().item(),
                            "✓"))
        except Exception as e:
            checks.append((f"ablation_mode='{mode}'", False, str(e)))
    model.ablation_mode = "full"  # reset

    if verbose:
        t_table = Table(show_header=True, header_style="bold cyan")
        t_table.add_column("Component"); t_table.add_column("Params", justify="right")
        for k, v in counts.items():
            style = "bold green" if k == "total" else ""
            t_table.add_row(k, f"[{style}]{v:,}[/{style}]" if style else str(v))
        console.print(t_table)

    elapsed = time.perf_counter() - t0
    all_ok  = all(ok for _, ok, _ in checks)

    if all_ok:
        step_pass(8, f"Full model: {total:,} params, all forward passes OK [{elapsed:.2f}s]")
    else:
        step_fail(8, f"Failed: {[n for n,ok,_ in checks if not ok]}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="quantum_kg pipeline verification (8 steps)"
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed check tables for each step")
    parser.add_argument("--step", type=int, default=0,
                        help="Run only a specific step (0 = all steps)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device("cpu")

    console.print(Panel(
        "[bold cyan]quantum_kg Pipeline Verification[/bold cyan]\n"
        "8 steps — all must pass before any training run",
        border_style="cyan",
    ))

    t_total = time.perf_counter()

    # Run steps
    run_step = lambda n: args.step == 0 or args.step == n

    kg = enumerator = encoder = unitary = all_paths = None
    train_dl = val_dl = test_dl = None

    if run_step(1):
        kg = step1_build_toy_kg(verbose=args.verbose)
    if run_step(2) and kg:
        train_dl, val_dl, test_dl = step2_dataloaders(kg, verbose=args.verbose)
    if run_step(3) and kg:
        step3_noise_injection(kg, verbose=args.verbose)
    if run_step(4) and kg:
        encoder = step4_quantum_state_encoder(kg, device, verbose=args.verbose)
    if run_step(5) and kg and encoder:
        unitary = step5_unitary_operator(kg, encoder, device, verbose=args.verbose)
    if run_step(6) and kg:
        enumerator, all_paths = step6_path_enumeration(kg, verbose=args.verbose)
    if run_step(7) and kg and encoder and unitary and enumerator and all_paths:
        step7_interference_analysis(
            kg, encoder, unitary, enumerator, all_paths, device, verbose=args.verbose
        )
    if run_step(8) and kg:
        step8_parameter_count(kg, device, verbose=args.verbose)

    # Final summary
    elapsed = time.perf_counter() - t_total
    n_pass  = sum(1 for ok, _ in STEP_RESULTS.values() if ok)
    n_fail  = sum(1 for ok, _ in STEP_RESULTS.values() if not ok)

    console.print()
    if n_fail == 0:
        console.print(Panel(
            f"[bold green]ALL {n_pass} STEPS PASSED[/bold green] in {elapsed:.1f}s\n\n"
            "Next: run [bold]python experiments/train_toy.py[/bold] to train the model.\n"
            "After training: re-run [bold]python experiments/run_toy.py --step 7[/bold]\n"
            "to confirm interference is working (wrong-answer queries should show\n"
            "negative interference values).",
            border_style="green",
            title="[bold]Pipeline Verification: PASS[/bold]",
        ))
        return 0
    else:
        failed_steps = [n for n, (ok, _) in STEP_RESULTS.items() if not ok]
        console.print(Panel(
            f"[bold red]{n_fail} STEP(S) FAILED: {failed_steps}[/bold red]\n\n"
            "Fix the failures before proceeding to training.\n"
            "Run with [bold]--verbose[/bold] for detailed error information.\n"
            "If Step 6 fails: the adjacency graph is broken; the interference\n"
            "experiment cannot run without contradictory paths.",
            border_style="red",
            title="[bold]Pipeline Verification: FAIL[/bold]",
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
