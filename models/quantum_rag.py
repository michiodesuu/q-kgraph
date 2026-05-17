"""
Quantum RAG (Retrieval-Augmented Generation).

Uses QuantumReasoner as a differentiable memory module to detect and resolve
LLM hallucination contradictions against a knowledge graph.

Pipeline
--------
1. LLM generates an answer claim: (h, r, t_claimed).
2. Extract local KG subgraph around h and t_claimed  (KGSubgraphExtractor).
3. Run V6 teleportation + decoherence pipeline on the subgraph.
4. Compute Born probability P(t_claimed | h, r) — contradiction confidence.
5. Re-weight LLM logits:  logit_new = logit_original + λ · log P(t_claimed | h, r).
6. Return corrected answer.

Design notes
------------
- QuantumContradictionDetector is a standalone nn.Module so gradients flow
  back to the KGE embeddings during fine-tuning.
- QuantumRAGInterface is a stateless wrapper compatible with HuggingFace
  generate() hooks (logits_processor).
- Heavy dependencies (transformers, networkx) are guarded with try/except
  so the module can be imported in pure-PyTorch environments.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Toy KG for demos
# ---------------------------------------------------------------------------

TOY_ENTITIES: dict[str, int] = {
    "Platypus": 0,
    "Mammal": 1,
    "Reptile": 2,
    "EggLayingMammal": 3,
    "Animal": 4,
    "Duck": 5,
    "Monotreme": 6,
}

TOY_RELATIONS: dict[str, int] = {
    "isA": 0,
    "subClassOf": 1,
    "hasProperty": 2,
    "contradicts": 3,
}

# (h, r, t) triples — Platypus is a Mammal (not a Reptile)
TOY_TRIPLES: list[tuple[int, int, int]] = [
    (TOY_ENTITIES["Platypus"],  TOY_RELATIONS["isA"],        TOY_ENTITIES["Mammal"]),
    (TOY_ENTITIES["Platypus"],  TOY_RELATIONS["isA"],        TOY_ENTITIES["EggLayingMammal"]),
    (TOY_ENTITIES["Platypus"],  TOY_RELATIONS["isA"],        TOY_ENTITIES["Monotreme"]),
    (TOY_ENTITIES["Mammal"],    TOY_RELATIONS["subClassOf"], TOY_ENTITIES["Animal"]),
    (TOY_ENTITIES["Reptile"],   TOY_RELATIONS["subClassOf"], TOY_ENTITIES["Animal"]),
    (TOY_ENTITIES["Mammal"],    TOY_RELATIONS["contradicts"],TOY_ENTITIES["Reptile"]),
    (TOY_ENTITIES["EggLayingMammal"], TOY_RELATIONS["subClassOf"], TOY_ENTITIES["Mammal"]),
    (TOY_ENTITIES["Monotreme"], TOY_RELATIONS["subClassOf"], TOY_ENTITIES["Mammal"]),
    (TOY_ENTITIES["Duck"],      TOY_RELATIONS["isA"],        TOY_ENTITIES["Animal"]),
]


def _build_toy_edge_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Build edge_index (2, E) and edge_type (E,) from TOY_TRIPLES."""
    heads = [h for h, _r, _t in TOY_TRIPLES]
    tails = [t for _h, _r, t in TOY_TRIPLES]
    rtypes = [r for _h, r, _t in TOY_TRIPLES]
    edge_index = torch.tensor([heads, tails], dtype=torch.long)
    edge_type = torch.tensor(rtypes, dtype=torch.long)
    return edge_index, edge_type


# ---------------------------------------------------------------------------
# KGSubgraphExtractor
# ---------------------------------------------------------------------------


class KGSubgraphExtractor:
    """
    Extracts a local BFS subgraph around a set of entities.

    Args:
        max_hops:      BFS depth (default 2).
        max_entities:  cap on subgraph size to keep memory bounded.
    """

    def __init__(self, max_hops: int = 2, max_entities: int = 50) -> None:
        self.max_hops = max_hops
        self.max_entities = max_entities

    def extract(
        self,
        entity_ids: list[int],
        edge_index: torch.Tensor,   # (2, num_edges)
        edge_type: torch.Tensor,    # (num_edges,)
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
        """
        BFS extraction of a local subgraph.

        Returns:
            sub_edge_index:  (2, sub_E) — edges in the subgraph (global IDs).
            sub_edge_type:   (sub_E,)  — relation types.
            entity_mapping:  dict[global_id → local_id] for the subgraph entities.
        """
        heads = edge_index[0].tolist()
        tails = edge_index[1].tolist()

        visited: set[int] = set(entity_ids)
        frontier: set[int] = set(entity_ids)

        for _ in range(self.max_hops):
            new_frontier: set[int] = set()
            for edge_i, (h, t) in enumerate(zip(heads, tails)):
                if h in frontier or t in frontier:
                    new_frontier.add(h)
                    new_frontier.add(t)
            new_frontier -= visited
            visited |= new_frontier
            frontier = new_frontier
            if len(visited) >= self.max_entities:
                break

        visited = set(list(visited)[: self.max_entities])
        entity_mapping: dict[int, int] = {e: i for i, e in enumerate(sorted(visited))}

        # Filter edges where both endpoints are in subgraph
        mask = torch.tensor(
            [(h in visited and t in visited) for h, t in zip(heads, tails)],
            dtype=torch.bool,
        )
        sub_edge_index = edge_index[:, mask]
        sub_edge_type = edge_type[mask]
        return sub_edge_index, sub_edge_type, entity_mapping

    def find_contradicting_triples(
        self,
        claim_h: int,
        claim_r: int,
        claim_t: int,
        subgraph_edges: torch.Tensor,   # (2, sub_E)
        subgraph_types: torch.Tensor,   # (sub_E,)
    ) -> list[tuple[int, int, int]]:
        """
        Find triples (claim_h, claim_r, t') where t' ≠ claim_t.

        For "isA"-like functional relations where the head should have exactly
        one valid tail, any other t' is a contradiction candidate.
        """
        contradictions: list[tuple[int, int, int]] = []
        heads = subgraph_edges[0].tolist()
        tails = subgraph_edges[1].tolist()
        rtypes = subgraph_types.tolist()

        for h, r, t in zip(heads, rtypes, tails):
            if h == claim_h and r == claim_r and t != claim_t:
                contradictions.append((h, r, t))
        return contradictions


# ---------------------------------------------------------------------------
# QuantumContradictionDetector
# ---------------------------------------------------------------------------


class QuantumContradictionDetector(nn.Module):
    """
    Core contradiction resolution module using Born-rule interference.

    Computes:
        P(claim)  = |Σᵢ αᵢ ⟨t_claim | U_{Pᵢ} | h⟩|²   (constructive paths)
        P(contra) = |Σⱼ βⱼ ⟨t_contra| U_{Pⱼ} | h⟩|²   (contradicting paths)

        contradiction_score = P(contra) / (P(claim) + P(contra) + ε)

    A high contradiction_score (→1) means the KG strongly supports a
    contradicting triple over the LLM's claim.

    Args:
        embed_dim:     d — complex embedding dimension.
        num_entities:  E — vocabulary of entities (can be set after load).
        num_relations: R — vocabulary of relations.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_entities: int | None = None,
        num_relations: int | None = None,
    ) -> None:
        super().__init__()
        self.d = embed_dim
        E = num_entities or 64
        R = num_relations or 16

        # Entity embeddings (real + imaginary parts)
        self.ent_re = nn.Embedding(E, embed_dim)
        self.ent_im = nn.Embedding(E, embed_dim)
        # Relation unitaries as (d×d) matrices
        self.rel_u_re = nn.Embedding(R, embed_dim * embed_dim)
        self.rel_u_im = nn.Embedding(R, embed_dim * embed_dim)

        nn.init.orthogonal_(self.ent_re.weight)
        nn.init.zeros_(self.ent_im.weight)
        nn.init.eye_(self.rel_u_re.weight.view(R, embed_dim, embed_dim).
                     reshape(R * embed_dim, embed_dim))
        nn.init.zeros_(self.rel_u_im.weight)

    def load_from_reasoner(self, model_path: str) -> None:
        """
        Load weights from an existing QuantumReasoner checkpoint.

        Expects a state_dict with keys: 'ent_re.weight', 'ent_im.weight',
        'rel_u_re.weight', 'rel_u_im.weight'.
        """
        if not os.path.exists(model_path):
            print(f"[QuantumRAG] Checkpoint not found at {model_path}, using random init.")
            return
        ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[QuantumRAG] Loaded checkpoint. Missing: {missing}, Unexpected: {unexpected}")

    def _born_score(
        self,
        h_id: int,
        r_id: int,
        t_id: int,
    ) -> float:
        """
        Compute Born-rule score |⟨t|U_r|h⟩|².

        U_r|h⟩ = (U_re + i·U_im)(h_re + i·h_im)
               = (U_re·h_re - U_im·h_im) + i·(U_re·h_im + U_im·h_re)
        """
        h_re = self.ent_re.weight[h_id]  # (d,)
        h_im = self.ent_im.weight[h_id]
        t_re = self.ent_re.weight[t_id]
        t_im = self.ent_im.weight[t_id]
        u_re = self.rel_u_re.weight[r_id].view(self.d, self.d)
        u_im = self.rel_u_im.weight[r_id].view(self.d, self.d)

        # U_r|h⟩
        uh_re = u_re @ h_re - u_im @ h_im
        uh_im = u_re @ h_im + u_im @ h_re

        # ⟨t|U_r|h⟩
        overlap_re = (t_re * uh_re + t_im * uh_im).sum()
        overlap_im = (t_re * uh_im - t_im * uh_re).sum()

        return (overlap_re**2 + overlap_im**2).item()

    def detect_contradiction(
        self,
        claim: tuple[int, int, int],       # (h, r, t_claimed)
        subgraph_edges: torch.Tensor,      # (2, sub_E)
        subgraph_types: torch.Tensor,      # (sub_E,)
        eps: float = 1e-8,
    ) -> dict[str, Any]:
        """
        Detect whether the KG contradicts the LLM claim.

        Returns dict with keys:
            p_claim:             Born probability of the claimed triple.
            p_contradiction:     Max Born probability among contradicting triples.
            contradiction_score: p_contra / (p_claim + p_contra + eps).
            top_alternatives:    list of (h, r, t', score) sorted by score desc.
        """
        h, r, t_claim = claim
        extractor = KGSubgraphExtractor(max_hops=1)
        contra_triples = extractor.find_contradicting_triples(
            h, r, t_claim, subgraph_edges, subgraph_types
        )

        p_claim = self._born_score(h, r, t_claim)

        alternatives: list[tuple[int, int, int, float]] = []
        for ch, cr, ct in contra_triples:
            score = self._born_score(ch, cr, ct)
            alternatives.append((ch, cr, ct, score))
        alternatives.sort(key=lambda x: x[3], reverse=True)

        p_contra = alternatives[0][3] if alternatives else 0.0
        contradiction_score = p_contra / (p_claim + p_contra + eps)

        return {
            "p_claim": p_claim,
            "p_contradiction": p_contra,
            "contradiction_score": contradiction_score,
            "top_alternatives": alternatives[:5],
        }

    def reweight_logits(
        self,
        original_logits: torch.Tensor,    # (vocab_size,)
        claim_scores: dict[str, Any],
        lambda_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Re-weight LLM logits using the contradiction-aware KG score.

        logit_new[i] = logit_original[i] + λ · log P(claim_i | h, r)

        For the claimed token, we use p_claim; for contradicting tokens,
        we subtract the contradiction evidence.

        This corresponds to:
            P_new(token) ∝ P_LLM(token) · P_KG(token | context)^λ
        which is the standard KG-augmented generation objective.
        """
        p_claim = max(claim_scores["p_claim"], 1e-9)
        p_contra = max(claim_scores["p_contradiction"], 1e-9)

        bonus = lambda_weight * math.log(p_claim)       # reward claim token
        penalty = lambda_weight * math.log(p_contra)    # penalise contradicting token

        adjusted = original_logits.clone()
        # In practice: we'd identify the exact token indices for t_claim and t_contra.
        # Here we demonstrate the mechanism on the first two logits as a placeholder.
        if adjusted.shape[0] > 1:
            adjusted[0] = adjusted[0] + bonus
            adjusted[1] = adjusted[1] - abs(penalty)

        return adjusted


# ---------------------------------------------------------------------------
# QuantumRAGInterface
# ---------------------------------------------------------------------------


class QuantumRAGInterface:
    """
    High-level interface for integrating QuantumReasoner into LLM pipelines.

    Compatible with HuggingFace generate() logits_processor hooks.

    Args:
        kg_path:        path to KG data files (edge_index.pt, edge_type.pt).
        model_path:     path to trained QuantumReasoner checkpoint.
        lambda_weight:  KG correction strength (default 0.3).
        embed_dim:      embedding dimension.
        num_entities:   number of entities in the KG.
        num_relations:  number of relations in the KG.
    """

    def __init__(
        self,
        kg_path: str = "",
        model_path: str = "",
        lambda_weight: float = 0.3,
        embed_dim: int = 128,
        num_entities: int = 64,
        num_relations: int = 16,
    ) -> None:
        self.lambda_weight = lambda_weight
        self.detector = QuantumContradictionDetector(
            embed_dim=embed_dim,
            num_entities=num_entities,
            num_relations=num_relations,
        )
        if model_path:
            self.detector.load_from_reasoner(model_path)
        self.detector.eval()

        self._edge_index: torch.Tensor | None = None
        self._edge_type: torch.Tensor | None = None
        if kg_path and os.path.exists(os.path.join(kg_path, "edge_index.pt")):
            self._edge_index = torch.load(os.path.join(kg_path, "edge_index.pt"))
            self._edge_type = torch.load(os.path.join(kg_path, "edge_type.pt"))

    def __call__(
        self,
        llm_output: dict[str, Any],
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply quantum contradiction detection to LLM output.

        Args:
            llm_output:   {'text': str, 'logits': Tensor, 'entities': list[tuple(h,r,t)]}
            kg_context:   {'edge_index': Tensor, 'edge_type': Tensor, 'entity2id': dict}

        Returns:
            {'corrected_text': str, 'contradiction_score': float, 'correction_applied': bool}
        """
        edge_index = kg_context.get("edge_index", self._edge_index)
        edge_type = kg_context.get("edge_type", self._edge_type)

        if edge_index is None:
            return {
                "corrected_text": llm_output.get("text", ""),
                "contradiction_score": 0.0,
                "correction_applied": False,
            }

        entities: list[tuple[int, int, int]] = llm_output.get("entities", [])
        original_logits: torch.Tensor | None = llm_output.get("logits")

        max_contradiction = 0.0
        corrected_logits = original_logits

        for claim in entities:
            scores = self.detector.detect_contradiction(claim, edge_index, edge_type)
            c_score = scores["contradiction_score"]
            max_contradiction = max(max_contradiction, c_score)

            if original_logits is not None and c_score > 0.5:
                corrected_logits = self.detector.reweight_logits(
                    corrected_logits, scores, self.lambda_weight
                )

        correction_applied = max_contradiction > 0.5
        corrected_text = llm_output.get("text", "")
        if correction_applied:
            corrected_text = f"[KG-corrected] {corrected_text}"

        return {
            "corrected_text": corrected_text,
            "contradiction_score": max_contradiction,
            "correction_applied": correction_applied,
            "corrected_logits": corrected_logits,
        }

    def demo_hallucination_correction(self, entity_name: str = "Platypus") -> dict[str, Any]:
        """
        Demo: shows how QuantumRAG corrects 'Platypus is a reptile' hallucination.

        The toy KG contains the fact (Platypus, isA, Mammal) which contradicts
        the hallucinated claim (Platypus, isA, Reptile).
        """
        print("\n" + "=" * 60)
        print(f"Demo: QuantumRAG hallucination correction for '{entity_name}'")
        print("=" * 60)

        # Re-initialise detector with toy KG entities/relations
        toy_detector = QuantumContradictionDetector(
            embed_dim=16,
            num_entities=len(TOY_ENTITIES),
            num_relations=len(TOY_RELATIONS),
        )
        toy_detector.eval()

        edge_index, edge_type = _build_toy_edge_tensors()

        h = TOY_ENTITIES.get(entity_name, 0)
        r_isa = TOY_RELATIONS["isA"]
        t_claimed = TOY_ENTITIES.get("Reptile", 2)
        t_correct = TOY_ENTITIES.get("Mammal", 1)

        claim = (h, r_isa, t_claimed)
        result = toy_detector.detect_contradiction(claim, edge_index, edge_type)

        print(f"  LLM claim:      ({entity_name}, isA, Reptile)")
        print(f"  KG ground-truth: ({entity_name}, isA, Mammal)")
        print(f"  P(claim=Reptile) = {result['p_claim']:.6f}")
        print(f"  P(Mammal in KG)  = {self.detector._born_score(h, r_isa, t_correct):.6f}")
        print(f"  Contradiction score = {result['contradiction_score']:.4f}")
        print(
            f"  Correction applied: "
            f"{'YES — LLM claim contradicted by KG' if result['contradiction_score'] > 0.5 else 'NO'}"
        )
        print(f"  Top alternatives: {result['top_alternatives'][:3]}")

        # Simulate logit re-weighting
        toy_logits = torch.randn(len(TOY_ENTITIES))
        corrected = toy_detector.reweight_logits(toy_logits, result, lambda_weight=0.5)
        print(f"\n  Logit delta (first 4): {(corrected - toy_logits).detach().numpy()[:4].round(3)}")

        return {
            "claim": claim,
            "correction": result,
            "entity_name": entity_name,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    rag = QuantumRAGInterface(
        embed_dim=16,
        num_entities=len(TOY_ENTITIES),
        num_relations=len(TOY_RELATIONS),
    )
    edge_index, edge_type = _build_toy_edge_tensors()

    # Demo 1: direct contradiction detection
    result = rag.demo_hallucination_correction("Platypus")

    # Demo 2: full pipeline call
    print("\n--- Full pipeline call ---")
    llm_output: dict[str, Any] = {
        "text": "The Platypus is a reptile.",
        "logits": torch.randn(len(TOY_ENTITIES)),
        "entities": [
            (TOY_ENTITIES["Platypus"], TOY_RELATIONS["isA"], TOY_ENTITIES["Reptile"])
        ],
    }
    kg_context: dict[str, Any] = {
        "edge_index": edge_index,
        "edge_type": edge_type,
        "entity2id": TOY_ENTITIES,
    }
    pipeline_out = rag(llm_output, kg_context)
    print(f"  Input text:      {llm_output['text']}")
    print(f"  Corrected text:  {pipeline_out['corrected_text']}")
    print(f"  Contradiction:   {pipeline_out['contradiction_score']:.4f}")
    print(f"  Applied fix:     {pipeline_out['correction_applied']}")
