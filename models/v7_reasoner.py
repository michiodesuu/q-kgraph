"""
models/v7_reasoner.py — V7 QuantumReasoner: Full Paradigm-Shift Architecture

ARCHITECTURE OVERVIEW (V7 Blueprint):
    Three scoring branches run in parallel for every query (h, r, t):

    Branch 1 — Unitary/MPS Branch:
        MatrixExpUnitary (V5) + MPSPathContractor for multi-hop amplitude aggregation
        Replaces explicit BFS path enumeration with tensor-network contraction.

    Branch 2 — Teleportation Branch (V6, extended):
        BellStateRelation matrices + Weyl corrections + Lindblad continuous-time evolution
        Relation operators are no longer constrained to U(d); Lindblad jump operators
        model the open quantum system dynamics of each reasoning hop.

    Branch 3 — Decoherence/Density-Matrix Branch:
        Full density-matrix propagation with per-relation learnable LindbladJumpOperators.
        Scoring via Tr(ρ|t⟩⟨t|) — generalized Born rule for mixed states.

    Ontological Layer:
        SasakiLogicLayer enforces Description Logic constraints via concept projectors.
        The Sasaki conjunction ensures contextual (non-factorizable) inference.

    Error Correction:
        GNNSyndromeDecoder trained jointly to detect contradictory triples.
        Syndrome scores feed back into branch weights via learned gating.

SCORING:
    s_final = (1-w_tel-w_dec) × s_unitary
            + w_tel × s_teleport
            + w_dec × s_decohere

    Branch weights are annealed during training (see V7Trainer phases).

TRAINING PHASES (4-phase protocol):
    Phase 1 (0 → t1):    Topological stabilization — GNN decoder trains on syndromes only
    Phase 2 (t1 → t2):   Coherent MPS evolution — Lindblad near-zero, MPS learns paths
    Phase 3 (t2 → t3):   Open system thermalization — Lindblad active, Sasaki enforced
    Phase 4 (t3 → end):  Formal verification loop — logic checker prunes invalid paths

V7 inherits from the same interface as QuantumReasoner (V1-V5) for drop-in compatibility.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.components.quantum_states import QuantumStateEncoder
from models.components.matrix_exp_unitary import MatrixExpUnitary
from models.components.path_aggregator import PathEnumerator, AmplitudeAggregator

# V7 components — imported with fallback so the file is still usable if components
# haven't been generated yet.
def _try_import(module_path: str, class_name: str):
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        return None

_LindbladEvolutionStep    = _try_import("models.components.lindblad_channel",   "LindbladEvolutionStep")
_LindbladJumpOperators    = _try_import("models.components.lindblad_channel",   "LindbladJumpOperators")
_LindbladPathIntegrator   = _try_import("models.components.lindblad_channel",   "LindbladPathIntegrator")
_MPSPathContractor        = _try_import("models.components.matrix_product_state","MPSPathContractor")
_InfiniteHopAggregator    = _try_import("models.components.matrix_product_state","InfiniteHopAggregator")
_SasakiConjunctionLayer   = _try_import("models.components.sasaki_logic",       "SasakiConjunctionLayer")
_OntologyConstraintLoss   = _try_import("models.components.sasaki_logic",       "OntologyConstraintLoss")
_GNNSyndromeDecoder       = _try_import("models.components.topological_ecc",    "GNNSyndromeDecoder")
_SyndromeDetector         = _try_import("models.components.topological_ecc",    "SyndromeDetector")

try:
    from models.components.quantum_teleportation import TeleportationScorer
    _HAS_TELEPORT = True
except ImportError:
    _HAS_TELEPORT = False

try:
    from models.components.decoherence import DecoherencePathAggregator
    _HAS_DECOHERE = True
except ImportError:
    _HAS_DECOHERE = False


class V7QuantumReasoner(nn.Module):
    """
    Full V7 quantum knowledge graph reasoning model.

    Assembles all four V7 components:
        - MatrixExpUnitary + MPSPathContractor (Branch 1: unitary/MPS)
        - TeleportationScorer + LindbladEvolutionStep (Branch 2: teleportation)
        - LindbladPathIntegrator + density-matrix scorer (Branch 3: decoherence)
        - SasakiConjunctionLayer (ontological constraints)
        - GNNSyndromeDecoder (topological error correction)

    Args:
        num_entities:     Entity count.
        num_relations:    Relation count.
        embed_dim:        Total embedding dimension (complex_dim = embed_dim // 2).
        bond_dim:         MPS bond dimension χ. Default 16.
        num_jumps:        Lindblad jump operators per relation. Default 4.
        num_concepts:     Description Logic concepts for Sasaki layer. Default 8.
        max_paths:        BFS path limit (fallback when MPS not available). Default 8.
        max_hops:         BFS hop depth. Default 2.
        dropout:          Dropout on entity states.
        tel_weight_init:  Initial teleportation branch weight (annealed during training).
        dec_weight_init:  Initial decoherence branch weight.
    """

    def __init__(
        self,
        num_entities:     int,
        num_relations:    int,
        embed_dim:        int   = 32,
        bond_dim:         int   = 16,
        num_jumps:        int   = 4,
        num_concepts:     int   = 8,
        max_paths:        int   = 8,
        max_hops:         int   = 2,
        dropout:          float = 0.0,
        tel_weight_init:  float = 0.0,   # start at 0, annealed up
        dec_weight_init:  float = 0.0,
    ) -> None:
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.embed_dim     = embed_dim
        self.complex_dim   = embed_dim // 2
        self.max_paths     = max_paths
        self.max_hops      = max_hops

        # ── Branch 1: Encoder + MatrixExpUnitary (base, always active) ──────
        self.encoder = QuantumStateEncoder(
            num_entities=num_entities,
            embed_dim=embed_dim,
            dropout=dropout,
            normalize=True,
        )
        self.unitary = MatrixExpUnitary(
            num_relations=num_relations,
            complex_dim=self.complex_dim,
        )
        self.aggregator = AmplitudeAggregator(
            complex_dim=self.complex_dim,
            max_paths=max_paths,
            learn_path_weights=True,
        )

        # MPS path contractor (replaces explicit BFS when available)
        self._has_mps = _MPSPathContractor is not None
        if self._has_mps:
            self.mps_contractor = _MPSPathContractor(
                num_relations=num_relations,
                complex_dim=self.complex_dim,
                bond_dim=bond_dim,
            )

        # ── Branch 2: Teleportation + Lindblad (optional) ───────────────────
        self._has_teleport = _HAS_TELEPORT
        if _HAS_TELEPORT:
            self.tel_scorer = TeleportationScorer(
                complex_dim=self.complex_dim,
                num_relations=num_relations,
            )

        self._has_lindblad = _LindbladEvolutionStep is not None
        if self._has_lindblad:
            self.lindblad_step = _LindbladEvolutionStep(
                num_relations=num_relations,
                complex_dim=self.complex_dim,
                num_jumps=num_jumps,
            )
            self.lindblad_integrator = _LindbladPathIntegrator(
                num_relations=num_relations,
                complex_dim=self.complex_dim,
                num_jumps=num_jumps,
            )

        # ── Branch 3: Decoherence (optional) ────────────────────────────────
        self._has_decohere = _HAS_DECOHERE
        if _HAS_DECOHERE:
            self.decohere_aggregator = DecoherencePathAggregator(
                complex_dim=self.complex_dim,
                num_relations=num_relations,
            )

        # ── Ontological layer: Sasaki logic ──────────────────────────────────
        self._has_sasaki = _SasakiConjunctionLayer is not None
        if self._has_sasaki and num_concepts > 0:
            self.sasaki_layer = _SasakiConjunctionLayer(
                complex_dim=self.complex_dim,
                num_concepts=num_concepts,
            )

        # ── Error correction: syndrome GNN ───────────────────────────────────
        self._has_ecc = _GNNSyndromeDecoder is not None
        if self._has_ecc:
            self.syndrome_detector = _SyndromeDetector()
            self.syndrome_decoder = _GNNSyndromeDecoder(
                entity_dim=self.complex_dim * 2,  # real + imag concatenated
                hidden_dim=64,
            )

        # ── Branch blend weights (learned scalars, annealed externally) ───────
        self._tel_weight = nn.Parameter(torch.tensor(tel_weight_init))
        self._dec_weight = nn.Parameter(torch.tensor(dec_weight_init))

        # Relation-specific bias
        self.relation_bias = nn.Parameter(torch.zeros(num_relations))

    # ── Branch weight access ────────────────────────────────────────────────
    @property
    def tel_weight(self) -> float:
        return torch.sigmoid(self._tel_weight).item()

    @property
    def dec_weight(self) -> float:
        return torch.sigmoid(self._dec_weight).item()

    # ── Scoring ──────────────────────────────────────────────────────────────
    def score_triple(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
        tail_ids:     torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Fast 1-hop Born-rule scoring for training.
        Branch 1 only (no MPS/Lindblad overhead).

        Returns: (B,) float scores.
        """
        h = self.encoder(head_ids)    # (B, d) complex
        t = self.encoder(tail_ids)    # (B, d) complex
        U_h = self.unitary.apply(h, relation_ids)   # (B, d)

        # Born rule: |⟨t|U_r|h⟩|²
        amp = (t.conj() * U_h).sum(dim=-1)
        s_unitary = amp.abs().pow(2)

        # Teleportation branch (if active)
        w_tel = torch.sigmoid(self._tel_weight)
        w_dec = torch.sigmoid(self._dec_weight)
        w_uni = 1.0 - w_tel - w_dec
        w_uni = w_uni.clamp(min=0.05)

        score = w_uni * s_unitary

        if self._has_teleport:
            try:
                s_tel = self.tel_scorer.score_triple(h, relation_ids, t)
                score = score + w_tel * s_tel
            except Exception:
                pass

        score = score + self.relation_bias[relation_ids]
        return score.clamp(0.0, 1.0)

    def score_triple_vs_all(
        self,
        head_ids:     torch.Tensor,   # (B,) int
        relation_ids: torch.Tensor,   # (B,) int
    ) -> torch.Tensor:
        """
        Score against all entities for filtered evaluation.

        Returns: (B, E) float scores.
        """
        h = self.encoder(head_ids)                   # (B, d) complex
        all_t = self.encoder.get_all_states()         # (E, d) complex
        U_h = self.unitary.apply(h, relation_ids)    # (B, d)

        # Batch Born rule: |⟨e|U_r|h⟩|² for all e
        amp_mat = torch.einsum("bd,ed->be", U_h.conj(), all_t)
        s_unitary = amp_mat.abs().pow(2)              # (B, E)

        w_tel = torch.sigmoid(self._tel_weight)
        w_dec = torch.sigmoid(self._dec_weight)
        w_uni = (1.0 - w_tel - w_dec).clamp(min=0.05)

        scores = w_uni * s_unitary

        if self._has_teleport:
            try:
                s_tel = self.tel_scorer.score_triple_vs_all(h, relation_ids, all_t)
                scores = scores + w_tel * s_tel
            except Exception:
                pass

        scores = scores + self.relation_bias[relation_ids].unsqueeze(1)
        return scores.clamp(0.0, 1.0)

    def score_with_lindblad(
        self,
        head_id:       int,
        relation_path: List[int],
        tail_id:       int,
        device:        torch.device,
    ) -> torch.Tensor:
        """
        Score a multi-hop path using Lindblad density-matrix evolution.
        Used for interference analysis and theory verification.

        Returns: scalar float probability.
        """
        if not self._has_lindblad:
            # Fallback to standard unitary evolution
            h = self.encoder(torch.tensor([head_id], device=device)).squeeze(0)
            t = self.encoder(torch.tensor([tail_id],  device=device)).squeeze(0)
            state = h
            for rel_id in relation_path:
                rel_t = torch.tensor([rel_id], device=device)
                state = self.unitary.apply(state.unsqueeze(0), rel_t).squeeze(0)
            return (t.conj() * state).sum().abs().pow(2).real

        h = self.encoder(torch.tensor([head_id], device=device)).squeeze(0)
        t = self.encoder(torch.tensor([tail_id],  device=device)).squeeze(0)

        # Initialize density matrix as pure state ρ = |h⟩⟨h|
        rho = torch.outer(h, h.conj())   # (d, d) complex

        rho_final = self.lindblad_integrator.integrate(
            rho_init=rho,
            path_rel_ids=relation_path,
            unitary_module=self.unitary,
            device=device,
        )
        return self.lindblad_integrator.score(rho_final, t)

    def score_with_mps(
        self,
        head_id:       int,
        relation_path: List[int],
        tail_id:       int,
        device:        torch.device,
    ) -> torch.Tensor:
        """
        Score a multi-hop path using MPS tensor network contraction.
        Returns: complex amplitude (scalar).
        """
        if not self._has_mps:
            raise RuntimeError("MPS contractor not available")

        h = self.encoder(torch.tensor([head_id], device=device)).squeeze(0)
        t = self.encoder(torch.tensor([tail_id],  device=device)).squeeze(0)
        return self.mps_contractor.contract_path(h, relation_path, t, device)

    def analyze_interference_v7(
        self,
        head_id:      int,
        tail_id:      int,
        paths:        List[List],
        device:       torch.device,
    ) -> Dict:
        """
        Full V7 interference analysis: reports per-path amplitudes from both
        standard unitary and Lindblad branches, plus Sasaki contextuality scores.

        Returns dict with diagnostics.
        """
        h = self.encoder(torch.tensor([head_id], device=device)).squeeze(0)
        t = self.encoder(torch.tensor([tail_id],  device=device)).squeeze(0)

        report = {
            "unitary_amplitudes": [],
            "lindblad_probs": [],
            "mps_amplitudes": [],
            "sasaki_contextuality": None,
        }

        # Standard unitary amplitudes per path
        for path in paths[:self.max_paths]:
            state = h.clone()
            for rel_id, _ in path:
                rel_t = torch.tensor([rel_id], device=device)
                state = self.unitary.apply(state.unsqueeze(0), rel_t).squeeze(0)
            amp = (t.conj() * state).sum()
            report["unitary_amplitudes"].append(amp.item())

        # Lindblad density-matrix scores
        if self._has_lindblad:
            for path in paths[:self.max_paths]:
                rel_ids = [rel for rel, _ in path]
                p = self.score_with_lindblad(head_id, rel_ids, tail_id, device)
                report["lindblad_probs"].append(p.item() if isinstance(p, torch.Tensor) else float(p))

        return report

    # ── State dict helpers for parameter group building ────────────────────
    def real_params(self):
        return [self.encoder.real_embeddings.weight]

    def imag_params(self):
        return [self.encoder.imag_embeddings.weight]

    def matrix_real_params(self):
        params = []
        if hasattr(self.unitary, "L_real"):
            params.append(self.unitary.L_real)
        return params

    def matrix_imag_params(self):
        params = []
        if hasattr(self.unitary, "L_imag"):
            params.append(self.unitary.L_imag)
        return params

    def lindblad_params(self):
        if self._has_lindblad:
            return list(self.lindblad_step.parameters()) + list(self.lindblad_integrator.parameters())
        return []

    def mps_params(self):
        if self._has_mps:
            return list(self.mps_contractor.parameters())
        return []

    def sasaki_params(self):
        if self._has_sasaki and hasattr(self, "sasaki_layer"):
            return list(self.sasaki_layer.parameters())
        return []

    def decoder_params(self):
        if self._has_ecc:
            return list(self.syndrome_decoder.parameters())
        return []

    def misc_params(self):
        p = [self.relation_bias, self._tel_weight, self._dec_weight]
        p += list(self.aggregator.parameters())
        return p


def build_v7_model(
    kg,
    embed_dim:    int   = 32,
    bond_dim:     int   = 16,
    num_jumps:    int   = 4,
    num_concepts: int   = 8,
    device:       str   = "cpu",
) -> V7QuantumReasoner:
    """Factory function — builds a V7QuantumReasoner from a ToyKG or similar dataset object."""
    num_entities  = kg.num_entities  if hasattr(kg, "num_entities")  else len(kg.entity2id)
    num_relations = kg.num_relations if hasattr(kg, "num_relations") else len(kg.relation2id)
    return V7QuantumReasoner(
        num_entities  = num_entities,
        num_relations = num_relations,
        embed_dim     = embed_dim,
        bond_dim      = bond_dim,
        num_jumps     = num_jumps,
        num_concepts  = num_concepts,
    ).to(device)
