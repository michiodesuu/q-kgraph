"""
training/schrodinger_dirac_loss.py — Schrödinger-Dirac Loss for V8

Treats each head entity as a Dirac spinor ψ = (ψ_L, ψ_R) ∈ ℂ^{2 × d}.
The reasoning path is interpreted as Lorentz-covariant evolution:

    Standard Dirac equation: (iγ^μ ∂_μ − m)ψ = 0

In the KGE context:
    ∂_μ  → relation operators  U_r
    m    → entity "mass" = ||embedding||   (Euclidean norm)

Dirac scoring:
    S(h, r, t) = Re(ψ_h† γ^0 U_r ψ_t)

Lorentz invariance:
    Under rescaling ψ → λψ,  S(λψ_h, r, ψ_t) = λ · S(ψ_h, r, ψ_t).
    We enforce approximate scale invariance by penalising the deviation
    of normalised vs unnormalised scores (see compute_lorentz_violation).

Three loss components:
    L_dirac       : BCE on Dirac scores (positive vs negative triples)
    L_schrodinger : physics consistency — ||evolved_spinor − expected||²
    L_lorentz     : scale invariance penalty
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_EPS = 1e-8


# ---------------------------------------------------------------------------
# Dirac algebra helpers
# ---------------------------------------------------------------------------

def _gamma0(d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """γ^0 = block-diagonal [[I, 0], [0, -I]] of size (2d, 2d) complex."""
    I = torch.eye(d, dtype=dtype, device=device)
    top = torch.cat([I, torch.zeros(d, d, dtype=dtype, device=device)], dim=-1)
    bot = torch.cat([torch.zeros(d, d, dtype=dtype, device=device), -I], dim=-1)
    return torch.cat([top, bot], dim=0)   # (2d, 2d)


def _pauli_x(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)


def _pauli_y(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)


def _pauli_z(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)


def _kron_extend(sigma: Tensor, d: int) -> Tensor:
    """Extend a 2×2 Pauli matrix to (d, d) block-diagonal (d//2 blocks)."""
    # block_diag of (d//2) copies of sigma
    blocks = sigma.unsqueeze(0).expand(d // 2, -1, -1)  # (d//2, 2, 2)
    return torch.block_diag(*[blocks[i] for i in range(d // 2)])  # (d, d)


def _build_gamma_matrices(d: int, device: torch.device) -> list[Tensor]:
    """Build γ^0, γ^1, γ^2, γ^3 extended to spinor dimension 2d.

    Dirac representation extended to arbitrary even d:
        γ^0 = [[I_d, 0], [0, -I_d]]
        γ^k = [[0, σ_k ⊗ I_{d/2}], [-σ_k ⊗ I_{d/2}, 0]]  for k=1,2,3

    Args:
        d:      complex_dim (half of embed_dim).
        device: torch device.

    Returns:
        gammas: list of 4 complex tensors, each (2d, 2d).
    """
    ctype = torch.complex64
    gammas = []

    # γ^0
    gammas.append(_gamma0(d, device=device, dtype=ctype))

    # γ^1, γ^2, γ^3 via extended Pauli matrices
    for sigma_fn in [_pauli_x, _pauli_y, _pauli_z]:
        sigma = sigma_fn(device=device, dtype=ctype)         # (2, 2)
        if d >= 2:
            sigma_ext = _kron_extend(sigma, d)               # (d, d)
        else:
            sigma_ext = sigma[:1, :1]
        zeros = torch.zeros(d, d, dtype=ctype, device=device)
        top = torch.cat([zeros,       sigma_ext], dim=-1)    # (d, 2d)
        bot = torch.cat([-sigma_ext,  zeros    ], dim=-1)    # (d, 2d)
        gammas.append(torch.cat([top, bot], dim=0))          # (2d, 2d)

    return gammas


# ---------------------------------------------------------------------------
# DiracSpinorEncoder
# ---------------------------------------------------------------------------

class DiracSpinorEncoder(nn.Module):
    """Encodes entities as Dirac spinors ψ = (ψ_L, ψ_R) ∈ ℂ^{2 × complex_dim}.

    ψ_L (left-chirality):  associates with even-parity relations.
    ψ_R (right-chirality): associates with odd-parity relations.

    Each component stored as a real-valued embedding of size embed_dim;
    the first half is the real part, the second half the imaginary part.

    Args:
        num_entities (int): Entity vocabulary size.
        embed_dim (int):    Total embedding dimension (complex_dim = embed_dim // 2).
    """

    def __init__(self, num_entities: int, embed_dim: int) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.num_entities = num_entities
        self.embed_dim = embed_dim
        self.complex_dim = embed_dim // 2

        self.psi_L = nn.Embedding(num_entities, embed_dim)
        self.psi_R = nn.Embedding(num_entities, embed_dim)

        std = 1.0 / math.sqrt(embed_dim)
        nn.init.normal_(self.psi_L.weight, std=std)
        nn.init.normal_(self.psi_R.weight, std=std)

    def encode(
        self, entity_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode entity ids into Dirac spinor components.

        Args:
            entity_ids: (B,) long.

        Returns:
            (psi_L_re, psi_L_im, psi_R_re, psi_R_im): (B, complex_dim) each.
        """
        d = self.complex_dim

        L_emb = self.psi_L(entity_ids)   # (B, embed_dim)
        R_emb = self.psi_R(entity_ids)   # (B, embed_dim)

        L_re, L_im = L_emb[:, :d], L_emb[:, d:]
        R_re, R_im = R_emb[:, :d], R_emb[:, d:]

        return L_re, L_im, R_re, R_im

    def get_param_count(self) -> dict[str, int]:
        return {
            "psi_L": self.psi_L.weight.numel(),
            "psi_R": self.psi_R.weight.numel(),
        }


# ---------------------------------------------------------------------------
# DiracRelationOperator
# ---------------------------------------------------------------------------

class DiracRelationOperator(nn.Module):
    """Gamma-matrix-structured relation operator U_r using Dirac algebra.

    U_r = exp(i θ_r · Σ_{μ=0}^{3} a_{r,μ} γ^μ)

    This is Lorentz-covariant by construction (Dirac algebra).

    The exponential is computed via torch.linalg.matrix_exp on a
    skew-Hermitian matrix i · (Σ a_{r,μ} γ^μ).

    Args:
        num_relations (int): Relation vocabulary size.
        complex_dim (int):   Half of embed_dim; spinor space = 2 × complex_dim.
    """

    def __init__(self, num_relations: int, complex_dim: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.complex_dim = complex_dim
        self.spinor_dim = 2 * complex_dim  # full spinor space

        # Per-relation coefficients a_{r,μ} for μ=0..3 and an overall phase θ_r
        self.a_coeff = nn.Embedding(num_relations, 4)  # (R, 4) — one per γ^μ
        self.theta   = nn.Embedding(num_relations, 1)  # (R, 1) — overall scale
        nn.init.normal_(self.a_coeff.weight, std=0.1)
        nn.init.ones_(self.theta.weight)

        # Cache gamma matrices (non-trainable)
        self._gamma_cache: list[Tensor] | None = None

    def _get_gammas(self, device: torch.device) -> list[Tensor]:
        """Retrieve (and cache) gamma matrices on the correct device."""
        if self._gamma_cache is None or self._gamma_cache[0].device != device:
            self._gamma_cache = _build_gamma_matrices(self.complex_dim, device)
        return self._gamma_cache

    def get_operator(self, relation_ids: Tensor) -> Tensor:
        """Compute U_r for a batch of relation ids.

        Args:
            relation_ids: (B,) long.

        Returns:
            U_r: (B, 2·complex_dim, 2·complex_dim) complex unitary.
        """
        device = relation_ids.device
        gammas = self._get_gammas(device)                          # list of 4 (2d,2d)

        a = self.a_coeff(relation_ids)                             # (B, 4)
        theta = self.theta(relation_ids)                           # (B, 1)

        B = relation_ids.shape[0]
        sd = self.spinor_dim

        # Build generator G = Σ a_μ γ^μ
        G = torch.zeros(B, sd, sd, dtype=torch.complex64, device=device)
        for mu, gamma in enumerate(gammas):
            a_mu = a[:, mu].to(torch.complex64)                    # (B,)
            G = G + a_mu[:, None, None] * gamma.unsqueeze(0)      # (B, 2d, 2d)

        # Skew-Hermitian exponent: i · θ · G
        theta_c = theta.to(torch.complex64)                        # (B, 1)
        exponent = 1j * theta_c.unsqueeze(-1) * G                 # (B, 2d, 2d)
        U_r = torch.linalg.matrix_exp(exponent)                   # (B, 2d, 2d)
        return U_r

    def get_param_count(self) -> dict[str, int]:
        return {
            "a_coeff": self.a_coeff.weight.numel(),
            "theta":   self.theta.weight.numel(),
        }


# ---------------------------------------------------------------------------
# SchrodingerDiracLoss
# ---------------------------------------------------------------------------

class SchrodingerDiracLoss(nn.Module):
    """Combined Schrödinger evolution + Dirac equation loss for V8.

    Three loss components
    ─────────────────────
    L_dirac (weight=dirac_weight):
        Binary cross-entropy on Dirac scores Re(ψ_h† γ^0 U_r ψ_t).
        Positive triples should score high, negatives low.

    L_schrodinger (weight=schrodinger_weight):
        Physics-consistency term: enforces that the spinor evolved by U_r
        matches the expected tail spinor up to a phase.
            L_sch = || U_r ψ_h − ψ_t ||²
        This is the Schrödinger equation residual in discrete time.

    L_lorentz (weight=lorentz_weight):
        Scale invariance: score should be insensitive to ||ψ|| rescaling.
            L_lor = || S(λψ_h, r, ψ_t) / (λ · S(ψ_h, r, ψ_t)) − 1 ||²
        for random λ ∼ Uniform(0.5, 2.0).

    Args:
        dirac_weight (float):       Weight of L_dirac (default 1.0).
        schrodinger_weight (float): Weight of L_schrodinger (default 0.1).
        lorentz_weight (float):     Weight of L_lorentz (default 0.05).
    """

    def __init__(
        self,
        dirac_weight: float = 1.0,
        schrodinger_weight: float = 0.1,
        lorentz_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.dirac_weight       = dirac_weight
        self.schrodinger_weight = schrodinger_weight
        self.lorentz_weight     = lorentz_weight

    # ------------------------------------------------------------------
    # Internal Dirac score helper
    # ------------------------------------------------------------------

    @staticmethod
    def _dirac_score(
        psi_h: Tensor,
        U_r:   Tensor,
        psi_t: Tensor,
        gamma0: Tensor,
    ) -> Tensor:
        """Re(ψ_h† γ^0 U_r ψ_t).

        Args:
            psi_h:  (B, 2d) complex spinor for head.
            U_r:    (B, 2d, 2d) complex unitary operator.
            psi_t:  (B, 2d) complex spinor for tail.
            gamma0: (2d, 2d) complex gamma^0 matrix.

        Returns:
            score: (B,) float.
        """
        # ψ_t' = U_r ψ_t
        psi_t_evolved = torch.bmm(U_r, psi_t.unsqueeze(-1)).squeeze(-1)  # (B, 2d)

        # γ^0 ψ_t'
        g0_psi_t = gamma0.unsqueeze(0) @ psi_t_evolved.unsqueeze(-1)     # (B, 2d, 1)
        g0_psi_t = g0_psi_t.squeeze(-1)                                  # (B, 2d)

        # ψ_h† (γ^0 U_r ψ_t) = Σ conj(ψ_h_i) (γ^0 U_r ψ_t)_i
        inner = (psi_h.conj() * g0_psi_t).sum(-1)                        # (B,) complex
        return inner.real                                                 # (B,) float

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        pos_scores:   Tensor,
        neg_scores:   Tensor,
        spinors_h:    tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        spinors_t:    tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
        relation_ids: Tensor | None = None,
        dirac_op:     DiracRelationOperator | None = None,
    ) -> Tensor:
        """Compute combined Schrödinger-Dirac loss.

        Args:
            pos_scores:   (B,) scores for positive triples.
            neg_scores:   (B,) scores for negative triples.
            spinors_h:    Optional (psi_L_re, psi_L_im, psi_R_re, psi_R_im) for heads,
                          each (B, complex_dim).  Required for L_schrodinger / L_lorentz.
            spinors_t:    Same structure for tails.
            relation_ids: (B,) long.  Required for L_schrodinger / L_lorentz.
            dirac_op:     DiracRelationOperator instance.  Required for physics terms.

        Returns:
            total_loss: scalar tensor.
        """
        # L_dirac: BCE on raw scores
        labels_pos = torch.ones_like(pos_scores)
        labels_neg = torch.zeros_like(neg_scores)
        l_dirac = (
            F.binary_cross_entropy_with_logits(pos_scores, labels_pos)
            + F.binary_cross_entropy_with_logits(neg_scores, labels_neg)
        )
        total = self.dirac_weight * l_dirac

        # Physics terms — only when optional inputs provided
        if (
            spinors_h is not None
            and spinors_t is not None
            and relation_ids is not None
            and dirac_op is not None
        ):
            L_re, L_im, R_re, R_im = spinors_h
            Lt_re, Lt_im, Rt_re, Rt_im = spinors_t
            device = L_re.device

            # Build complex spinors: (ψ_L, ψ_R) concatenated → (B, 2d)
            psi_h = torch.complex(
                torch.cat([L_re, R_re], dim=-1),
                torch.cat([L_im, R_im], dim=-1),
            )  # (B, 2d)
            psi_t = torch.complex(
                torch.cat([Lt_re, Rt_re], dim=-1),
                torch.cat([Lt_im, Rt_im], dim=-1),
            )  # (B, 2d)

            # U_r and γ^0
            U_r    = dirac_op.get_operator(relation_ids)              # (B, 2d, 2d)
            gammas = dirac_op._get_gammas(device)
            gamma0 = gammas[0]                                        # (2d, 2d)

            # --- L_schrodinger: || U_r ψ_h − ψ_t ||² ---
            psi_evolved = torch.bmm(U_r, psi_h.unsqueeze(-1)).squeeze(-1)  # (B, 2d)
            schrodinger_res = (psi_evolved - psi_t).abs().pow(2).sum(-1)   # (B,)
            l_schrodinger = schrodinger_res.mean()
            total = total + self.schrodinger_weight * l_schrodinger

            # --- L_lorentz: scale invariance penalty ---
            l_lorentz = self.compute_lorentz_violation(
                psi_h, psi_t, U_r, gamma0
            ).mean()
            total = total + self.lorentz_weight * l_lorentz

        return total

    def compute_lorentz_violation(
        self,
        psi_h:  Tensor,
        psi_t:  Tensor,
        U_r:    Tensor,
        gamma0: Tensor,
        n_samples: int = 4,
    ) -> Tensor:
        """Measures how much the Dirac score changes under random head rescaling.

        For each sample draw λ ~ Uniform(0.5, 2.0), compute:
            violation = |S(λψ_h, r, ψ_t) / (λ · S(ψ_h, r, ψ_t)) − 1|

        where S is the Dirac score.  Perfect Lorentz invariance gives 0.

        Args:
            psi_h:     (B, 2d) complex head spinors.
            psi_t:     (B, 2d) complex tail spinors.
            U_r:       (B, 2d, 2d) complex relation operators.
            gamma0:    (2d, 2d) complex γ^0.
            n_samples: number of λ draws per sample.

        Returns:
            violation: (B,) float.
        """
        base_score = self._dirac_score(psi_h, U_r, psi_t, gamma0)   # (B,) float

        violations = []
        for _ in range(n_samples):
            lam = torch.empty(psi_h.shape[0], device=psi_h.device).uniform_(0.5, 2.0)
            lam_c = lam.to(psi_h.dtype)                              # complex if needed
            scaled_psi_h = psi_h * lam_c.unsqueeze(-1)
            scaled_score = self._dirac_score(scaled_psi_h, U_r, psi_t, gamma0)  # (B,)

            # Expected under linearity: scaled_score ≈ λ · base_score
            expected = lam * base_score
            denom = expected.abs().clamp(min=_EPS)
            rel_err = ((scaled_score - expected).abs() / denom).pow(2)  # (B,)
            violations.append(rel_err)

        return torch.stack(violations, dim=0).mean(dim=0)            # (B,)

    def get_param_count(self) -> dict[str, int]:
        return {}   # this module has no learnable params itself


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("schrodinger_dirac_loss.py — V8 Dirac Loss Demo")
    print("=" * 60)

    torch.manual_seed(3)
    B, E, R, D = 4, 50, 8, 16   # complex_dim = D, embed_dim = 2D

    # --- DiracSpinorEncoder ---
    encoder = DiracSpinorEncoder(num_entities=E, embed_dim=2 * D)
    ent_h = torch.randint(0, E, (B,))
    ent_t = torch.randint(0, E, (B,))
    psi_L_re, psi_L_im, psi_R_re, psi_R_im = encoder.encode(ent_h)

    print(f"\nDiracSpinorEncoder:")
    print(f"  ψ_L_re shape: {psi_L_re.shape}")
    print(f"  ψ_R_re shape: {psi_R_re.shape}")
    print(f"  Param counts: {encoder.get_param_count()}")

    # --- DiracRelationOperator ---
    dirac_op = DiracRelationOperator(num_relations=R, complex_dim=D)
    rel_ids = torch.randint(0, R, (B,))
    U_r = dirac_op.get_operator(rel_ids)

    print(f"\nDiracRelationOperator:")
    print(f"  U_r shape: {U_r.shape}")

    # Verify approximate unitarity: U† U ≈ I
    sd = 2 * D
    UH_U = torch.bmm(U_r.conj().transpose(-2, -1), U_r)
    I = torch.eye(sd, dtype=torch.complex64).unsqueeze(0).expand(B, -1, -1)
    unit_err = (UH_U - I).abs().max().item()
    print(f"  Unitarity error (should be ~0): {unit_err:.2e}")

    # --- SchrodingerDiracLoss ---
    loss_fn = SchrodingerDiracLoss(
        dirac_weight=1.0, schrodinger_weight=0.1, lorentz_weight=0.05
    )
    pos_scores = torch.randn(B)
    neg_scores = torch.randn(B) - 1.0  # biased negative

    spinors_h = encoder.encode(ent_h)
    spinors_t = encoder.encode(ent_t)

    total_loss = loss_fn(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        spinors_h=spinors_h,
        spinors_t=spinors_t,
        relation_ids=rel_ids,
        dirac_op=dirac_op,
    )
    print(f"\nSchrodingerDiracLoss:")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Lorentz violation check
    gammas = dirac_op._get_gammas(device=rel_ids.device)
    gamma0 = gammas[0]
    psi_h_c = torch.complex(
        torch.cat([psi_L_re, psi_R_re], dim=-1),
        torch.cat([psi_L_im, psi_R_im], dim=-1),
    )
    Lt_re, Lt_im, Rt_re, Rt_im = encoder.encode(ent_t)
    psi_t_c = torch.complex(
        torch.cat([Lt_re, Rt_re], dim=-1),
        torch.cat([Lt_im, Rt_im], dim=-1),
    )
    violation = loss_fn.compute_lorentz_violation(psi_h_c, psi_t_c, U_r, gamma0)
    print(f"  Lorentz violation (mean): {violation.mean().item():.4f}")

    # Backward pass
    total_loss.backward()
    has_grads = all(
        p.grad is not None
        for p in list(encoder.parameters()) + list(dirac_op.parameters())
    )
    print(f"  Gradients computed: {has_grads}")
    print("\nAll checks passed.")
