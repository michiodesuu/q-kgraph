"""
models/components/quaternion_states.py — Quaternion Entity Embeddings [V4]

SOURCE: QIQE-KGC (Information Sciences, 2023)
TECHNIQUE: Quaternion Hypercomplex Space H^d

WHY UPGRADE FROM COMPLEX TO QUATERNION:
    V1-V3 use ℂ^d (2D complex: real + imaginary).
    V4 upgrades to H^d (4D quaternion: real, i, j, k components).

    Quaternion advantage over complex:
    - Encodes 3D spatial rotation WITHOUT gimbal lock (3 rotation axes: pitch, yaw, roll)
    - Models 4 distinct relation patterns in one Hamilton product
    - Directly addresses the "DiagonalUnitary ≈ RotatE" reviewer objection

    Mathematical structure:
        q = r + a·i + b·j + c·k   where i²=j²=k²=ijk=−1
        Hamilton product: q₁⊙q₂ preserves associativity but NOT commutativity

    In QIQE-KGC: scoring function s(h,r,t) = Re(e_h ⊙ r ⊙ conj(e_t))
    In V4:        quaternion amplitude = ⟨t_q|U_r_q|s_q⟩ (quaternion inner product)

COMPATIBILITY:
    - Uses 4 separate nn.Embedding tables (compatible with TrainerV2 param groups)
    - All existing path_aggregator.py logic works via get_complex_projection()
    - Backward compatible: can fall back to ℂ^d by projecting (r+ai) component
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def hamilton_product(
    q1_r: torch.Tensor,   # (..., d) real component
    q1_i: torch.Tensor,   # (..., d) imaginary i
    q1_j: torch.Tensor,   # (..., d) imaginary j
    q1_k: torch.Tensor,   # (..., d) imaginary k
    q2_r: torch.Tensor,
    q2_i: torch.Tensor,
    q2_j: torch.Tensor,
    q2_k: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Hamilton product of two quaternions: q_out = q1 ⊙ q2.

    From QIQE-KGC equation for relation-entity composition.
    Non-commutative: q1⊙q2 ≠ q2⊙q1 (this is WHY quaternions model asymmetric relations).

    Standard Hamilton product rules:
        i·i = j·j = k·k = i·j·k = −1
        i·j = k,  j·i = −k
        j·k = i,  k·j = −i
        k·i = j,  i·k = −j

    Args:
        q1_*, q2_*: The four float32 components of each quaternion batch.

    Returns:
        (r, i, j, k) components of the product quaternion.
    """
    # Real component: r1r2 − a1a2 − b1b2 − c1c2
    out_r = (q1_r * q2_r) - (q1_i * q2_i) - (q1_j * q2_j) - (q1_k * q2_k)
    # i component: r1a2 + a1r2 + b1c2 − c1b2
    out_i = (q1_r * q2_i) + (q1_i * q2_r) + (q1_j * q2_k) - (q1_k * q2_j)
    # j component: r1b2 − a1c2 + b1r2 + c1a2
    out_j = (q1_r * q2_j) - (q1_i * q2_k) + (q1_j * q2_r) + (q1_k * q2_i)
    # k component: r1c2 + a1b2 − b1a2 + c1r2
    out_k = (q1_r * q2_k) + (q1_i * q2_j) - (q1_j * q2_i) + (q1_k * q2_r)

    return out_r, out_i, out_j, out_k


def quaternion_norm(
    r: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute quaternion norm: ||q|| = sqrt(r²+i²+j²+k²)."""
    return (r.pow(2) + i.pow(2) + j.pow(2) + k.pow(2)).sum(dim=-1, keepdim=True).sqrt().clamp(min=eps)


def quaternion_normalize(
    r: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Normalize to unit quaternion: ||q|| = 1."""
    norm = quaternion_norm(r, i, j, k)
    return r / norm, i / norm, j / norm, k / norm


def quaternion_conjugate(
    r: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Quaternion conjugate: conj(r + ai + bj + ck) = r − ai − bj − ck."""
    return r, -i, -j, -k


def quaternion_inner_product(
    q1_r: torch.Tensor, q1_i: torch.Tensor, q1_j: torch.Tensor, q1_k: torch.Tensor,
    q2_r: torch.Tensor, q2_i: torch.Tensor, q2_j: torch.Tensor, q2_k: torch.Tensor,
) -> torch.Tensor:
    """
    Quaternion inner product: ⟨q1|q2⟩ = sum of component-wise products.

    Returns a real scalar (used in scoring function from QIQE-KGC):
        s(h,r,t) = Re(e_h ⊙ r ⊙ conj(e_t))
    """
    return (q1_r * q2_r + q1_i * q2_i + q1_j * q2_j + q1_k * q2_k).sum(dim=-1)


class QuaternionStateEncoder(nn.Module):
    """
    Entity states as unit quaternions in H^d (4D hypercomplex space).

    Replaces V1-V3's QuantumStateEncoder (ℂ^d) with H^d quaternion embeddings.

    Four components per embedding dimension:
        q_e = r_e + a_e·i + b_e·j + c_e·k

    Why quaternions over complex for KGE:
    - Complex ℂ handles 2D rotation (one angle per dimension)
    - Quaternion H handles 3D rotation (three angles: roll, pitch, yaw per dimension)
    - Quaternion NATURALLY handles antisymmetry, inversion, composition
      via non-commutativity of Hamilton product (q1⊙q2 ≠ q2⊙q1)
    - Proven in QIQE-KGC to outperform complex by +2.82% MRR on FB15k-237

    Stored as 4 separate nn.Embedding tables to allow TrainerV2's independent LR per group.
    TrainerV2 integration:
        real_emb: 1× lr_base
        i_emb:    3× lr_base (same "anti-collapse" multiplier as V3 imaginary)
        j_emb:    2× lr_base
        k_emb:    2× lr_base

    Args:
        num_entities:  Total entity count.
        quaterion_dim: Dimension d of the quaternion space. Total params: 4×d per entity.
        normalize:     Enforce unit quaternion ||q|| = 1 at every forward.
        init_std:      Initialization std (small to start near identity quaternion).
    """

    def __init__(
        self,
        num_entities:  int,
        quaternion_dim: int  = 32,
        normalize:     bool  = True,
        init_std:      float = 0.1,
    ) -> None:
        super().__init__()
        self.num_entities   = num_entities
        self.quaternion_dim = quaternion_dim
        self.normalize      = normalize
        # embed_dim for backward compatibility with existing code
        self.embed_dim      = quaternion_dim * 4
        self.complex_dim    = quaternion_dim  # interface compatibility

        # Four embedding tables — separate for TrainerV2 param groups
        self.emb_r = nn.Embedding(num_entities, quaternion_dim)  # real
        self.emb_i = nn.Embedding(num_entities, quaternion_dim)  # imaginary i
        self.emb_j = nn.Embedding(num_entities, quaternion_dim)  # imaginary j
        self.emb_k = nn.Embedding(num_entities, quaternion_dim)  # imaginary k

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        """
        Initialize near identity quaternion (1, 0, 0, 0) per QIQE-KGC convention.
        Real component initialized larger than imaginary parts.
        """
        nn.init.normal_(self.emb_r.weight, mean=0.0, std=std)
        nn.init.normal_(self.emb_i.weight, mean=0.0, std=std * 0.3)
        nn.init.normal_(self.emb_j.weight, mean=0.0, std=std * 0.3)
        nn.init.normal_(self.emb_k.weight, mean=0.0, std=std * 0.3)
        with torch.no_grad():
            self._normalize_inplace()

    def _normalize_inplace(self) -> None:
        """Enforce unit quaternion norm in-place."""
        r = self.emb_r.weight.data
        i = self.emb_i.weight.data
        j = self.emb_j.weight.data
        k = self.emb_k.weight.data
        norms = (r.pow(2) + i.pow(2) + j.pow(2) + k.pow(2)).sum(-1, keepdim=True).sqrt().clamp(1e-10)
        self.emb_r.weight.data = r / norms
        self.emb_i.weight.data = i / norms
        self.emb_j.weight.data = j / norms
        self.emb_k.weight.data = k / norms

    def forward(
        self,
        entity_ids: torch.Tensor,  # (...) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Look up entity quaternion states.

        Returns:
            (r, i, j, k) — four float32 tensors of shape (..., quaternion_dim).
            Each is unit-normalized if self.normalize=True.
        """
        r = self.emb_r(entity_ids)
        i = self.emb_i(entity_ids)
        j = self.emb_j(entity_ids)
        k = self.emb_k(entity_ids)
        if self.normalize:
            r, i, j, k = quaternion_normalize(r, i, j, k)
        return r, i, j, k

    def get_complex_projection(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        Project quaternion to complex tensor for backward compatibility.

        Returns complex64 tensor of shape (..., quaternion_dim).
        Uses (r + i·sqrt(-1)) — dropping j and k components.
        Used as interface to V1-V3 path_aggregator.py components.
        """
        r, i, j, k = self.forward(entity_ids)
        return torch.complex(r, i)

    def quat_score_triple(
        self,
        h_ids: torch.Tensor,   # (B,) int64
        r_comp: tuple,         # (r,i,j,k) relation quaternion components from QuaternionUnitary
        t_ids: torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:
        """
        QIQE-KGC scoring function: s(h,r,t) = Re(e_h ⊙ r ⊙ conj(e_t)).

        The real part of the Hamilton product e_h ⊙ r_rel ⊙ conj(e_t) gives the score.
        This is equivalent to the dot product in the rotated quaternion space.

        Returns:
            (B,) float32 scores.
        """
        hr, hi, hj, hk = self.forward(h_ids)
        tr, ti, tj, tk = self.forward(t_ids)
        rr, ri, rj, rk = r_comp   # relation quaternion

        # Step 1: e_h ⊙ r_rel → intermediate quaternion
        ir, ii, ij, ik = hamilton_product(hr, hi, hj, hk, rr, ri, rj, rk)

        # Step 2: conj(e_t) = (tr, -ti, -tj, -tk)
        tcr, tci, tcj, tck = quaternion_conjugate(tr, ti, tj, tk)

        # Step 3: (e_h ⊙ r_rel) ⊙ conj(e_t)
        fr, fi, fj, fk = hamilton_product(ir, ii, ij, ik, tcr, tci, tcj, tck)

        # Step 4: Re(result) — sum over dimensions
        return fr.sum(dim=-1)   # (B,) float32

    def state_fidelity(
        self,
        entity_a: int,
        entity_b: int,
        device:   Optional[torch.device] = None,
    ) -> float:
        """Quaternion fidelity: |⟨q_a|q_b⟩|² (real inner product squared)."""
        if device is None:
            device = self.emb_r.weight.device
        with torch.no_grad():
            ra, ia, ja, ka = self.forward(torch.tensor([entity_a], device=device))
            rb, ib, jb, kb = self.forward(torch.tensor([entity_b], device=device))
            inner = quaternion_inner_product(
                ra.squeeze(0), ia.squeeze(0), ja.squeeze(0), ka.squeeze(0),
                rb.squeeze(0), ib.squeeze(0), jb.squeeze(0), kb.squeeze(0),
            )
            return float(inner.pow(2).item())

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_entities={self.num_entities}, "
            f"quaternion_dim={self.quaternion_dim}, "
            f"params={self.num_params():,}"
        )
