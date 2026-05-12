"""
models/components/sasaki_logic.py — Description Logic as Quantum Observables  [V6]

PURPOSE:
    Implements ontological concepts (Description Logic classes) as orthogonal
    projectors in the entity Hilbert space, and replaces the classical Boolean
    meet (∧) with the Sasaki conjunction — the correct quantum-logical "AND"
    that respects measurement contextuality.

QUANTUM MECHANICS CONNECTION:
    In quantum logic, a proposition "entity e belongs to concept C" is
    represented by a projector P_C : H → H, where H = ℂ^d is the entity
    Hilbert space.  The truth value of C for entity |e⟩ is the expectation
    value ⟨e|P_C|e⟩ ∈ [0, 1] — a probability, not a Boolean.

    Classical Boolean AND:   P_A ∧ P_B  (commutative, order-independent)
    Sasaki conjunction:      P_A ∧_S P_B = P_A P_B P_A   (non-commutative!)

    The Sasaki form is the "correct" quantum AND because it computes:
      "Probability that e is in B, given that we first project/filter by A"
    This respects Gleason's theorem and the non-distributive orthomodular
    lattice structure of quantum logic (Birkhoff & von Neumann, 1936).

CONTEXTUALITY:
    If P_A P_B ≠ P_B P_A (the projectors do NOT commute), then the order of
    measurement matters — this is quantum contextuality.  The Sasaki conjunction
    exposes this: s_joint = ⟨ψ|(P_A P_B P_A)|ψ⟩ will differ from
    s_factored = ⟨ψ|P_A|ψ⟩ × ⟨ψ|P_B|ψ⟩ when the concepts are entangled in
    the entity's state space.

PARAMETERIZATION:
    A rank-r projector P_C = V_C V_C†  where V_C ∈ ℂ^{d×r}, V_C†V_C = I_r.
    We store raw parameters (V_real, V_imag) and orthogonalize via QR in
    every forward pass, so gradients always flow through a valid projector.

CLASSES:
    ConceptProjector          — Single DL concept as a learnable projector
    SasakiConjunctionLayer    — Batch Sasaki ∧_S over a vocabulary of concepts
    OntologyConstraintLoss    — Membership / hierarchy / disjointness losses
    SasakiContextualityLoss   — Penalizes classically-factorizable reasoning
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  ConceptProjector
# ──────────────────────────────────────────────────────────────────────────────

class ConceptProjector(nn.Module):
    """
    Orthogonal projector for an ontological concept, parameterized via a
    semi-orthogonal matrix V ∈ ℂ^{d × r} with V†V = I_r.

    The projector is  P_C = V V†  (a rank-r Hermitian positive semi-definite
    idempotent:  P² = P,  P† = P).

    Parameters are stored as real (V_real) and imaginary (V_imag) parts of V.
    At every forward call, V is orthogonalized via thin QR decomposition so
    that gradients propagate through a *valid* projector at each step.

    Args:
        complex_dim:   Hilbert space dimension d.
        subspace_dim:  Rank r of the projector (default: complex_dim // 4).
        concept_name:  Optional label for logging / debugging.
    """

    def __init__(
        self,
        complex_dim: int,
        subspace_dim: Optional[int] = None,
        concept_name: str = "",
    ) -> None:
        super().__init__()

        self.complex_dim  = complex_dim
        self.subspace_dim = subspace_dim if subspace_dim is not None else max(1, complex_dim // 4)
        self.concept_name = concept_name

        if self.subspace_dim > complex_dim:
            raise ValueError(
                f"subspace_dim={self.subspace_dim} cannot exceed "
                f"complex_dim={complex_dim}."
            )

        # Raw (un-orthogonalized) parameters for V ∈ ℂ^{d × r}
        # Stored as two real float tensors to stay compatible with standard
        # optimizers and enable independent learning rates if desired.
        self.V_real = nn.Parameter(
            torch.empty(complex_dim, self.subspace_dim)
        )
        self.V_imag = nn.Parameter(
            torch.empty(complex_dim, self.subspace_dim)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize with a random semi-orthogonal matrix.

        We draw a Gaussian matrix and take its QR decomposition to obtain
        a proper initial V†V = I_r.  The imaginary part is initialized smaller
        to keep early projectors approximately real (matching the convention
        used by QuantumStateEncoder).
        """
        with torch.no_grad():
            # Real part: orthogonal initialization via QR
            G = torch.randn(self.complex_dim, self.subspace_dim)
            Q, _ = torch.linalg.qr(G, mode="reduced")  # (d, r), Q†Q = I_r
            self.V_real.data.copy_(Q)

            # Imaginary part: small Gaussian perturbation
            nn.init.normal_(self.V_imag, mean=0.0, std=0.02)

    # ── Core QR orthogonalization ──────────────────────────────────────────────

    def _orthogonalized_V(self) -> torch.Tensor:
        """
        Return V ∈ ℂ^{d × r} with V†V = I_r.

        We form the raw complex matrix  V_raw = V_real + i·V_imag  and apply
        thin QR decomposition to its columns.  Because torch.linalg.qr works
        on complex tensors, this gives a unitary Q with Q†Q = I_r exactly.

        The QR gradient (via implicit differentiation) is numerically stable
        as long as V_raw has full column rank, which is maintained by the
        small-imaginary initialization and standard SGD/Adam updates.

        Returns:
            Complex tensor of shape (complex_dim, subspace_dim).
        """
        V_raw = torch.complex(self.V_real, self.V_imag)   # (d, r) complex
        Q, _ = torch.linalg.qr(V_raw, mode="reduced")     # (d, r), Q†Q = I_r
        return Q

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_projector(self) -> torch.Tensor:
        """
        Compute the rank-r Hermitian projector P_C = V V† ∈ ℂ^{d × d}.

        Properties guaranteed by construction:
            P² = (VV†)(VV†) = V(V†V)V† = VIV† = VV† = P    (idempotent)
            P† = (VV†)† = VV† = P                             (Hermitian)
            tr(P) = tr(V†V) = tr(I_r) = r                    (rank r)

        Returns:
            (complex_dim, complex_dim) complex64 tensor.
        """
        V = self._orthogonalized_V()           # (d, r)
        return V @ V.conj().T                   # (d, d) = V V†

    def project(self, states: torch.Tensor) -> torch.Tensor:
        """
        Apply projector P_C to a batch of entity states: |out⟩ = P_C|state⟩.

        This is the "collapse" operation: after projection, all states lie in
        the r-dimensional subspace associated with concept C.

        Args:
            states: (B, complex_dim) complex tensor of entity states.

        Returns:
            (B, complex_dim) complex tensor — projected states.
        """
        P = self.get_projector()     # (d, d)
        # states: (B, d) → (B, d, 1) for batched matmul → (B, d)
        return (states.unsqueeze(-1) * P.conj().T).sum(dim=-2)
        # Equivalent to:  states @ P.conj().T  but explicit for clarity.
        # Note: P = P†, so P† = P, and (states @ P) works too; we use
        #       the conj().T form to be explicit about the bra-ket convention.

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Alias for project() — enables use as a standard nn.Module layer."""
        return self.project(states)

    def membership_scores(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute ⟨state|P_C|state⟩ ∈ [0, 1] for each state in the batch.

        This is the Born-rule probability that entity 'state' belongs to C.

        Args:
            states: (B, d) complex entity states (should be unit-normed).

        Returns:
            (B,) real tensor of membership probabilities.
        """
        projected = self.project(states)                    # (B, d)
        # ⟨ψ|P|ψ⟩ = ||P|ψ⟩||² when |ψ⟩ is unit-normed and P is a projector
        return (states.conj() * projected).sum(dim=-1).real  # (B,) real

    def extra_repr(self) -> str:
        return (
            f"concept='{self.concept_name}', "
            f"complex_dim={self.complex_dim}, "
            f"subspace_dim={self.subspace_dim}"
        )


# ──────────────────────────────────────────────────────────────────────────────
#  SasakiConjunctionLayer
# ──────────────────────────────────────────────────────────────────────────────

class SasakiConjunctionLayer(nn.Module):
    """
    Applies the Sasaki conjunction  P_A ∧_S P_B = P_A P_B P_A  to entity states.

    The Sasaki conjunction models "entity satisfies A, AND THEN B in A's context".
    It is the unique quantum-logical AND compatible with the orthomodular lattice
    of closed subspaces of a Hilbert space (Sasaki, 1954; Jauch & Piron, 1963).

    Unlike the classical meet (P_A ∧ P_B = projector onto ran(P_A) ∩ ran(P_B)),
    the Sasaki form is:
        1. NOT symmetric in general: P_A ∧_S P_B ≠ P_B ∧_S P_A
        2. NOT idempotent (it is a Hermitian positive operator, not a projector)
        3. Sensitive to the order of contextual measurement

    This asymmetry is exactly what we want for directional inference:
    "satisfies A AND (given A) satisfies B" ≠ "satisfies B AND (given B) satisfies A".

    Args:
        complex_dim:   Hilbert space dimension.
        num_concepts:  Number of ontological concepts to maintain.
        subspace_dim:  Rank of each concept projector (default: complex_dim // 4).
    """

    def __init__(
        self,
        complex_dim: int,
        num_concepts: int,
        subspace_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.complex_dim  = complex_dim
        self.num_concepts = num_concepts

        # One learnable projector per concept
        self.projectors = nn.ModuleList([
            ConceptProjector(
                complex_dim=complex_dim,
                subspace_dim=subspace_dim,
                concept_name=f"concept_{i}",
            )
            for i in range(num_concepts)
        ])

    # ── Sasaki operations ──────────────────────────────────────────────────────

    def sasaki_and(
        self,
        proj_a: torch.Tensor,
        proj_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Sasaki conjunction  P_A ∧_S P_B = P_A @ P_B @ P_A.

        Note: the result is a Hermitian positive semi-definite operator with
        eigenvalues in [0, 1], but it is NOT itself an orthogonal projector
        (unless P_A and P_B happen to commute, in which case it equals their
        classical meet).

        Args:
            proj_a: (d, d) complex Hermitian projector for concept A.
            proj_b: (d, d) complex Hermitian projector for concept B.

        Returns:
            (d, d) complex tensor — the Sasaki conjunction operator.
        """
        return proj_a @ proj_b @ proj_a

    def apply_sasaki(
        self,
        states: torch.Tensor,
        concept_a_id: int,
        concept_b_id: int,
    ) -> torch.Tensor:
        """
        Apply Sasaki conjunction to entity states:  (P_A P_B P_A)|state⟩.

        Args:
            states:        (B, d) complex entity states.
            concept_a_id:  Index of concept A in self.projectors.
            concept_b_id:  Index of concept B in self.projectors.

        Returns:
            (B, d) complex tensor — states after Sasaki filtering.
        """
        P_A = self.projectors[concept_a_id].get_projector()   # (d, d)
        P_B = self.projectors[concept_b_id].get_projector()   # (d, d)
        S   = self.sasaki_and(P_A, P_B)                       # (d, d)
        # Apply: (B, d) @ (d, d).T → (B, d)
        return states @ S.conj().T

    def contextuality_score(
        self,
        state: torch.Tensor,
        concept_a_id: int,
        concept_b_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Measure quantum vs. classical contextuality for a single state.

        Computes two quantities:
            s_joint   = ⟨ψ|(P_A P_B P_A)|ψ⟩    (Sasaki joint probability)
            s_factored = ⟨ψ|P_A|ψ⟩ × ⟨ψ|P_B|ψ⟩  (classical product rule)

        If s_joint ≈ s_factored, the state is "classically factorisable" for
        this concept pair — the projectors effectively commute on this state.
        If |s_joint - s_factored| >> 0, the reasoning is genuinely contextual.

        Args:
            state:         (d,) complex tensor — a single entity state.
            concept_a_id:  Index of concept A.
            concept_b_id:  Index of concept B.

        Returns:
            Tuple (s_joint, s_factored), each a scalar real tensor.
        """
        P_A = self.projectors[concept_a_id].get_projector()  # (d, d)
        P_B = self.projectors[concept_b_id].get_projector()  # (d, d)

        # Sasaki joint:  ⟨ψ|P_A P_B P_A|ψ⟩ = ||P_B P_A |ψ⟩||²
        # (Use ||P_A|ψ⟩||² form for numerical stability)
        psi       = state                            # (d,)
        P_A_psi   = P_A @ psi                        # (d,)
        P_B_PA_psi = P_B @ P_A_psi                   # (d,)
        s_joint   = (psi.conj() @ (P_A @ P_B_PA_psi)).real  # scalar

        # Classical factored:  ⟨ψ|P_A|ψ⟩ × ⟨ψ|P_B|ψ⟩
        s_a       = (psi.conj() @ (P_A @ psi)).real  # scalar
        s_b       = (psi.conj() @ (P_B @ psi)).real  # scalar
        s_factored = s_a * s_b                        # scalar

        return s_joint, s_factored

    def forward(
        self,
        states: torch.Tensor,
        concept_a_id: int,
        concept_b_id: int,
    ) -> torch.Tensor:
        """
        Default forward: apply Sasaki conjunction and return filtered states.

        Args:
            states:        (B, d) complex entity states.
            concept_a_id:  Index of concept A.
            concept_b_id:  Index of concept B.

        Returns:
            (B, d) complex tensor.
        """
        return self.apply_sasaki(states, concept_a_id, concept_b_id)

    def extra_repr(self) -> str:
        return (
            f"complex_dim={self.complex_dim}, "
            f"num_concepts={self.num_concepts}"
        )


# ──────────────────────────────────────────────────────────────────────────────
#  OntologyConstraintLoss
# ──────────────────────────────────────────────────────────────────────────────

class OntologyConstraintLoss(nn.Module):
    """
    Regularisation losses that enforce Description Logic axioms.

    Three types of DL axiom are supported:

    (a) Membership:  e ∈ C
        L_mem = 1 - ⟨e|P_C|e⟩
        Minimised when the entity state lies entirely in concept C's subspace.

    (b) Subclass hierarchy:  C1 ⊑ C2  (every C1-instance is also a C2-instance)
        L_hier = ||P_{C2} P_{C1} - P_{C1}||_F²
        Minimised when ran(P_{C1}) ⊆ ran(P_{C2}), i.e. P_{C2} acts as identity
        on the smaller subspace.  Equivalent to P_{C2} P_{C1} = P_{C1}.

    (c) Disjointness:  C1 ⊓ C2 = ⊥  (C1 and C2 share no instances)
        L_disj = ||P_{C1} P_{C2}||_F²
        Minimised when ran(P_{C1}) ⊥ ran(P_{C2}), i.e. the subspaces are
        orthogonal complements.  Equivalent to P_{C1} P_{C2} = 0.

    Args:
        weight_membership: Loss weight for membership assertions.
        weight_hierarchy:  Loss weight for subclass rules.
        weight_disjoint:   Loss weight for disjointness rules.
    """

    def __init__(
        self,
        weight_membership: float = 1.0,
        weight_hierarchy:  float = 0.5,
        weight_disjoint:   float = 0.5,
    ) -> None:
        super().__init__()
        self.weight_membership = weight_membership
        self.weight_hierarchy  = weight_hierarchy
        self.weight_disjoint   = weight_disjoint

    def _membership_loss(
        self,
        concept_projectors: nn.ModuleList,
        entity_states: torch.Tensor,
        membership_assertions: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        L_membership = mean over (e, C) of  1 - ⟨e|P_C|e⟩.

        Args:
            concept_projectors:    nn.ModuleList of ConceptProjector.
            entity_states:         (N, d) complex entity states.
            membership_assertions: list of (entity_idx, concept_id) pairs.

        Returns:
            Scalar loss (0 if no assertions).
        """
        if not membership_assertions:
            return entity_states.new_zeros(1).squeeze()

        losses = []
        for entity_idx, concept_id in membership_assertions:
            state = entity_states[entity_idx]                     # (d,)
            score = concept_projectors[concept_id].membership_scores(
                state.unsqueeze(0)
            ).squeeze(0)                                          # scalar real
            losses.append(1.0 - score)

        return torch.stack(losses).mean()

    def _hierarchy_loss(
        self,
        concept_projectors: nn.ModuleList,
        hierarchy_rules: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        L_hierarchy = mean over (C1⊑C2) of  ||P_{C2} P_{C1} - P_{C1}||_F².

        Args:
            hierarchy_rules: list of (child_concept_id, parent_concept_id).

        Returns:
            Scalar loss (0 if no rules).
        """
        if not hierarchy_rules:
            return concept_projectors[0].V_real.new_zeros(1).squeeze()

        losses = []
        for child_id, parent_id in hierarchy_rules:
            P_child  = concept_projectors[child_id].get_projector()   # (d, d)
            P_parent = concept_projectors[parent_id].get_projector()  # (d, d)
            # Want: P_parent @ P_child ≈ P_child
            residual = P_parent @ P_child - P_child                   # (d, d)
            losses.append(_frobenius_sq(residual))

        return torch.stack(losses).mean()

    def _disjointness_loss(
        self,
        concept_projectors: nn.ModuleList,
        disjointness_rules: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        L_disjoint = mean over (C1, C2) of  ||P_{C1} P_{C2}||_F².

        Args:
            disjointness_rules: list of (concept_id_a, concept_id_b).

        Returns:
            Scalar loss (0 if no rules).
        """
        if not disjointness_rules:
            return concept_projectors[0].V_real.new_zeros(1).squeeze()

        losses = []
        for concept_a_id, concept_b_id in disjointness_rules:
            P_a = concept_projectors[concept_a_id].get_projector()  # (d, d)
            P_b = concept_projectors[concept_b_id].get_projector()  # (d, d)
            losses.append(_frobenius_sq(P_a @ P_b))

        return torch.stack(losses).mean()

    def forward(
        self,
        concept_projectors: nn.ModuleList,
        entity_states: torch.Tensor,
        membership_assertions: List[Tuple[int, int]],
        hierarchy_rules: List[Tuple[int, int]],
        disjointness_rules: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute the combined ontology constraint loss.

        Args:
            concept_projectors:    nn.ModuleList of ConceptProjector modules.
            entity_states:         (N, complex_dim) complex tensor of entity states.
            membership_assertions: List of (entity_idx, concept_id) pairs.
            hierarchy_rules:       List of (child_concept_id, parent_concept_id).
            disjointness_rules:    List of (concept_id_a, concept_id_b).

        Returns:
            Scalar tensor — the weighted sum of all constraint losses.
        """
        loss = entity_states.new_zeros(1).squeeze()

        if membership_assertions:
            loss = loss + self.weight_membership * self._membership_loss(
                concept_projectors, entity_states, membership_assertions
            )

        if hierarchy_rules:
            loss = loss + self.weight_hierarchy * self._hierarchy_loss(
                concept_projectors, hierarchy_rules
            )

        if disjointness_rules:
            loss = loss + self.weight_disjoint * self._disjointness_loss(
                concept_projectors, disjointness_rules
            )

        return loss


# ──────────────────────────────────────────────────────────────────────────────
#  SasakiContextualityLoss
# ──────────────────────────────────────────────────────────────────────────────

class SasakiContextualityLoss(nn.Module):
    """
    Encourages genuinely non-classical (contextual) quantum reasoning.

    Motivation:
        If s_joint ≈ s_factored for every concept pair, the model degenerates
        to classical Boolean logic — the projectors effectively commute and
        the quantum structure adds no representational power.  This loss
        pushes the model towards a regime where  |s_joint - s_factored| > delta_ctx,
        signifying that the Sasaki AND is doing genuinely non-classical work.

    Loss:
        L_ctx = mean over (ψ, A, B) triples of
                    ReLU(delta_ctx - |s_joint(ψ, A, B) - s_factored(ψ, A, B)|)

        This is a margin loss: if the gap already exceeds delta_ctx, the
        contribution is zero (no gradient); only "too classical" triples
        receive a non-zero gradient signal.

    IMPORTANT — gradient flow:
        Both s_joint and s_factored are differentiable w.r.t. entity_states
        (through the matrix-vector products) and w.r.t. concept projector
        parameters (through get_projector() → QR autograd).  The loss
        therefore trains BOTH the entity states AND the concept projectors
        to become more contextual.

    Args:
        delta_ctx: Minimum required gap between s_joint and s_factored.
                   Typical values: 0.01 – 0.1.
        weight:    Loss weight when combined with other losses.
    """

    def __init__(self, delta_ctx: float = 0.05, weight: float = 0.1) -> None:
        super().__init__()
        self.delta_ctx = delta_ctx
        self.weight    = weight

    def forward(
        self,
        sasaki_layer: SasakiConjunctionLayer,
        entity_states: torch.Tensor,
        concept_pair_ids: List[Tuple[int, int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute contextuality margin loss.

        For each (concept_a_id, concept_b_id) pair, we evaluate the gap
        |s_joint - s_factored| for every entity in the batch and penalize
        triples where the gap falls below delta_ctx.

        The computation is fully differentiable:
            - s_joint    = ⟨ψ|P_A P_B P_A|ψ⟩  → grad w.r.t. ψ and P_A, P_B
            - s_factored = ⟨ψ|P_A|ψ⟩ · ⟨ψ|P_B|ψ⟩ → grad w.r.t. ψ and P_A, P_B

        Args:
            sasaki_layer:     SasakiConjunctionLayer with learnable projectors.
            entity_states:    (B, d) complex entity states (should be unit-normed).
            concept_pair_ids: List of (concept_a_id, concept_b_id) pairs to evaluate.
            device:           Compute device.

        Returns:
            Scalar loss tensor (differentiable).
        """
        if not concept_pair_ids:
            return entity_states.new_zeros(1).squeeze()

        B = entity_states.shape[0]
        pair_losses = []

        for concept_a_id, concept_b_id in concept_pair_ids:
            P_A = sasaki_layer.projectors[concept_a_id].get_projector()  # (d, d)
            P_B = sasaki_layer.projectors[concept_b_id].get_projector()  # (d, d)

            # ── Sasaki joint:  ⟨ψ|P_A P_B P_A|ψ⟩  (B,)
            # We avoid building the full (d,d) Sasaki matrix and instead
            # chain three matrix-vector products for memory efficiency.
            #   step1 = P_A |ψ⟩  →  (B, d)
            #   step2 = P_B (step1)  →  (B, d)
            #   step3 = P_A (step2)  →  (B, d)
            #   s_joint = Re(⟨ψ|step3⟩)  →  (B,)
            step1   = entity_states @ P_A.conj().T                    # (B, d)
            step2   = step1 @ P_B.conj().T                            # (B, d)
            step3   = step2 @ P_A.conj().T                            # (B, d)
            s_joint = (entity_states.conj() * step3).sum(dim=-1).real # (B,)

            # ── Factored:  ⟨ψ|P_A|ψ⟩ × ⟨ψ|P_B|ψ⟩  (B,)
            s_a       = (entity_states.conj() * (entity_states @ P_A.conj().T)).sum(dim=-1).real
            s_b       = (entity_states.conj() * (entity_states @ P_B.conj().T)).sum(dim=-1).real
            s_factored = s_a * s_b                                    # (B,)

            # ── Margin loss:  ReLU(delta_ctx - |s_joint - s_factored|)
            gap       = (s_joint - s_factored).abs()                  # (B,) ≥ 0
            pair_loss = F.relu(self.delta_ctx - gap).mean()           # scalar
            pair_losses.append(pair_loss)

        return self.weight * torch.stack(pair_losses).mean()


# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _frobenius_sq(M: torch.Tensor) -> torch.Tensor:
    """
    Compute ||M||_F² = tr(M† M) = sum of squared absolute values of entries.

    For complex M ∈ ℂ^{m×n}: ||M||_F² = Σ_{ij} |M_{ij}|².

    Args:
        M: Complex or real tensor of shape (m, n).

    Returns:
        Scalar real tensor.
    """
    return M.abs().pow(2).sum()
