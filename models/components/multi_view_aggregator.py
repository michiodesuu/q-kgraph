"""
models/components/multi_view_aggregator.py — Multi-View Interference Aggregator [V4]

SOURCE: AAAI-2024 Quantum Interference Model for WSD
TECHNIQUE: Multi-View Superposition + Explicit Interference Term

MOTIVATION FROM THE WSD PAPER:
    Classical WSD fuses multiple gloss definitions into ONE vector → loses semantic bias.
    The AAAI-2024 model treats each gloss as a SEPARATE quantum state, preserving bias.
    Disambiguation uses the INTERFERENCE TERM between gloss pairs:

        P_sense(+) = Σk ||φk P⁺ |ψk⟩||² + Σ_{k≠k'} Int_{k,k'}(+)

    Where the interference term is:
        Int(k,k')(+) = 2·αk·αk'·cos(θkk')·|⟨ψk|P⁺|ψk'⟩|

V4 ADAPTATION FOR PATH REASONING:
    In our KG context:
    - "Glosses" → "Paths": each reasoning path Pi is a "view" of the query
    - Each path provides an amplitude: Ai = ⟨t|U_Pi|s⟩
    - The total probability is NOT just |Σαᵢ Aᵢ|² (V1-V3)
    - V4 EXPLICITLY separates self-terms and cross-terms:

        P(t|s) = Σᵢ |αᵢ|²|Aᵢ|²   (self-terms = per-path Born rule)
               + Σᵢ≠ⱼ Intᵢⱼ        (cross-terms = explicit interference)

    WHERE the V4 interference term is:
        Intᵢⱼ = 2·Re(αᵢ·αⱼ*)·cos(θᵢⱼ)·|⟨Aᵢ|Aⱼ⟩|·λᵢⱼ

    λᵢⱼ is a LOGIC WEIGHT from the QuantumLogicLattice:
        High λᵢⱼ → paths i and j are globally consistent → interference amplified
        Low λᵢⱼ  → one or both paths violate logic → interference dampened

WHAT THIS FIXES vs V1-V3:
    V1-V3: Implicit interference (cross-terms from squaring the sum — correct math but no control)
    V4:   EXPLICIT interference with logic weighting + semantic similarity weighting

    This directly addresses the QIQE-KGC finding that combining quantum logic (global)
    with amplitude interference (local) yields performance > sum of parts.
"""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewInterferenceAggregator(nn.Module):
    """
    Extends V1-V3 AmplitudeAggregator with explicit WSD-inspired interference term.

    The AAAI-2024 paper proved:
        Full model (F1=80.1%) > No superposition (F1=79.0%) > No interference (F1=77.1%)

    The 3% gap from interference term to full model proves the cross-path interaction
    is a measurable, non-negligible contribution beyond just multi-view preservation.

    V4 implements this explicitly for path reasoning:
    1. Compute per-path amplitudes Aᵢ (quaternion inner products)
    2. Compute self-terms: |αᵢ|²|Aᵢ|² (classical per-path probabilities)
    3. Compute pairwise interference: Intᵢⱼ with logic weights
    4. Total probability = self-terms + logic-weighted cross-terms

    Args:
        quaternion_dim:      Dimension of quaternion space.
        max_paths:           Maximum reasoning paths per query.
        learn_path_weights:  Learn complex weights αᵢ per path position.
        use_logic_weights:   Use lattice consistency scores to weight interference.
        interference_temp:   Temperature for cosine angle computation (θᵢⱼ/τ).
    """

    def __init__(
        self,
        quaternion_dim:     int   = 32,
        max_paths:          int   = 8,
        learn_path_weights: bool  = True,
        use_logic_weights:  bool  = True,
        interference_temp:  float = 1.0,
        eps:                float = 1e-10,
    ) -> None:
        super().__init__()
        self.quaternion_dim     = quaternion_dim
        self.complex_dim        = quaternion_dim  # backward compat
        self.max_paths          = max_paths
        self.use_logic_weights  = use_logic_weights
        self.interference_temp  = interference_temp
        self.eps                = eps

        if learn_path_weights:
            # Learned complex path weights αᵢ (stored as real + imaginary)
            self.weight_r = nn.Parameter(torch.randn(max_paths) * 0.01)
            self.weight_i = nn.Parameter(torch.randn(max_paths) * 0.01)
        else:
            self.register_buffer("weight_r", torch.ones(max_paths) / math.sqrt(max_paths))
            self.register_buffer("weight_i", torch.zeros(max_paths))

        # Learned interference scaling (separate from path weights)
        # Allows the model to calibrate how much interference contributes
        self.interference_scale = nn.Parameter(torch.tensor(1.0))

    def _get_weights(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized (real, imag) path weights for k paths."""
        wr = self.weight_r[:k]
        wi = self.weight_i[:k]
        # Normalize: ||α|| = 1
        norm = (wr.pow(2) + wi.pow(2)).sum().sqrt().clamp(min=self.eps)
        return wr / norm, wi / norm

    def _compute_quaternion_amplitude(
        self,
        source_quat: tuple[torch.Tensor, ...],   # (r,i,j,k) source state
        target_quat: tuple[torch.Tensor, ...],   # (r,i,j,k) target state
        path:        list,                        # [(rel_id, ent_id), ...]
        quat_unitary,                             # QuaternionUnitary module
    ) -> tuple[float, float]:
        """
        Compute quaternion path amplitude: A_i = ⟨t_q|U_Pi_q|s_q⟩.

        1. Evolve source through path: |s'_q⟩ = U_Pi_q|s_q⟩
        2. Quaternion inner product: A_i = ⟨t_q|s'_q⟩ = Σ t*_comp · s'_comp
        3. Return as (real_part, imag_proxy) for complex arithmetic

        The quaternion inner product uses all 4 components, capturing
        more geometric information than the complex inner product in V1-V3.

        Returns:
            (real_amplitude, imag_amplitude) as floats.
        """
        sr, si, sj, sk = [x.clone() for x in source_quat]

        for rel_id, _ in path:
            rel_tensor = torch.tensor([rel_id], dtype=torch.long,
                                       device=sr.device)
            sr, si, sj, sk = quat_unitary.apply(sr.unsqueeze(0), si.unsqueeze(0),
                                                  sj.unsqueeze(0), sk.unsqueeze(0),
                                                  rel_tensor)
            sr, si, sj, sk = sr.squeeze(0), si.squeeze(0), sj.squeeze(0), sk.squeeze(0)

        tr, ti, tj, tk = target_quat

        # Quaternion inner product: ⟨t|s'⟩ = Σ(t_r·s'_r + t_i·s'_i + t_j·s'_j + t_k·s'_k)
        # We separate into "real" and "imaginary proxy" for interference computation
        real_amp = (tr * sr + ti * si + tj * sj + tk * sk).sum().item()
        imag_amp = (tr * si - ti * sr + tj * sk - tk * sj).sum().item()  # quaternion cross term

        return real_amp, imag_amp

    def compute_interference_terms(
        self,
        source_quat:   tuple[torch.Tensor, ...],
        target_quat:   tuple[torch.Tensor, ...],
        paths:         list,
        quat_unitary,
        logic_scores:  Optional[list[float]] = None,
    ) -> dict:
        """
        Full interference decomposition with explicit WSD-inspired cross-terms.

        From AAAI-2024:
            P_sense(+) = Σk |φk|²|Ak|² + Σ_{k≠k'} Int_{k,k'}(+)
            Int(k,k') = 2·αk·αk'·cos(θkk')·|⟨Ak|Ak'⟩|

        V4 adaptation:
            Int(i,j) = 2·Re(αi·αj*)·cos(θij)·|⟨Ai|Aj⟩|·λij
            where λij = geometric mean of logic scores for paths i and j

        Args:
            source_quat:  Source entity quaternion components (r,i,j,k).
            target_quat:  Target entity quaternion components (r,i,j,k).
            paths:        List of path objects [(rel_id, ent_id), ...].
            quat_unitary: QuaternionUnitary module for path evolution.
            logic_scores: Optional per-path logic consistency scores from lattice.

        Returns:
            Dict with all decomposition terms (compatible with V3 analysis pipeline).
        """
        paths = paths[:self.max_paths]
        if not paths:
            return {
                "amplitudes": [], "path_probabilities": [], "weights": [],
                "total_probability": 0.0, "classical_sum": 0.0,
                "interference": 0.0, "interference_sign": "none",
                "explicit_interference_terms": [],
                "logic_weighted_interference": 0.0,
                "path_entropy": 0.0, "path_entropy_norm": 0.0,
                "n_paths": 0,
            }

        k       = len(paths)
        wr, wi  = self._get_weights(k)
        device  = source_quat[0].device

        # Step 1: Compute per-path quaternion amplitudes
        amplitudes_r = []   # real parts of amplitudes
        amplitudes_i = []   # imaginary proxies

        with torch.no_grad():
            for path in paths:
                ar, ai = self._compute_quaternion_amplitude(
                    source_quat, target_quat, path, quat_unitary
                )
                amplitudes_r.append(ar)
                amplitudes_i.append(ai)

        amplitudes_r = torch.tensor(amplitudes_r, device=device)  # (k,)
        amplitudes_i = torch.tensor(amplitudes_i, device=device)  # (k,)

        # Per-path probability: |Ai|² = ar² + ai²
        path_probs = amplitudes_r.pow(2) + amplitudes_i.pow(2)    # (k,)

        # Step 2: Self-terms (classical sum) = Σᵢ |αᵢ|²|Aᵢ|²
        alpha_sq    = wr.pow(2) + wi.pow(2)                       # (k,) real
        classical_sum = (alpha_sq * path_probs).sum().item()

        # Step 3: EXPLICIT CROSS-TERMS (WSD-inspired)
        # Int(i,j) = 2·Re(αi·αj*)·cos(θij)·|⟨Ai|Aj⟩|·λij

        explicit_terms = []
        total_cross    = 0.0

        for i_idx in range(k):
            for j_idx in range(i_idx + 1, k):
                # Re(αi·αj*) = αi_r·αj_r + αi_i·αj_i (real part of product)
                re_alpha = wr[i_idx] * wr[j_idx] + wi[i_idx] * wi[j_idx]

                # Phase angle between amplitudes θij
                # θij = angle between (Ai_r, Ai_i) and (Aj_r, Aj_i) in complex plane
                norm_i = math.sqrt(path_probs[i_idx].item() + self.eps)
                norm_j = math.sqrt(path_probs[j_idx].item() + self.eps)

                # cos(θij) via dot product of normalized amplitude vectors
                dot_rr = amplitudes_r[i_idx] * amplitudes_r[j_idx]
                dot_ii = amplitudes_i[i_idx] * amplitudes_i[j_idx]
                cos_theta = float((dot_rr + dot_ii) / (norm_i * norm_j + self.eps))
                cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp for stability

                # |⟨Ai|Aj⟩| = |Ai_r·Aj_r + Ai_i·Aj_i|
                inner_prod = abs(float(dot_rr + dot_ii))

                # Logic weight λij (from QIQE-KGC: global consistency)
                if logic_scores and i_idx < len(logic_scores) and j_idx < len(logic_scores):
                    lam = math.sqrt(logic_scores[i_idx] * logic_scores[j_idx] + self.eps)
                else:
                    lam = 1.0

                # Interference term (AAAI-2024 Equation):
                # Int(i,j) = 2·Re(αi·αj*)·cos(θij/τ)·|⟨Ai|Aj⟩|·λij
                cos_scaled = math.cos(math.acos(cos_theta) / self.interference_temp)
                int_term   = 2.0 * float(re_alpha) * cos_scaled * inner_prod * lam

                explicit_terms.append({
                    "i": i_idx, "j": j_idx,
                    "interference": int_term,
                    "cos_theta":    cos_theta,
                    "logic_weight": lam,
                })
                total_cross += int_term

        # Step 4: Total probability
        scale    = float(self.interference_scale.item())
        total_P  = max(0.0, classical_sum + scale * total_cross)

        # Interference = total - classical (can be negative → destructive)
        interference = total_P - classical_sum

        if interference < -1e-6:
            sign = "destructive"
        elif interference > 1e-6:
            sign = "constructive"
        else:
            sign = "negligible"

        # Path entropy (from custom metrics Section 6.3 of README)
        prob_sum = path_probs.sum().item() + self.eps
        probs_norm = path_probs / prob_sum
        entropy = float(-(probs_norm * (probs_norm + self.eps).log()).sum().item())
        max_entropy = math.log(k) if k > 1 else 1.0
        entropy_norm = entropy / max_entropy

        return {
            "amplitudes":                  [(ar, ai) for ar, ai in zip(amplitudes_r.tolist(), amplitudes_i.tolist())],
            "path_probabilities":          path_probs.tolist(),
            "weights":                     list(zip(wr.tolist(), wi.tolist())),
            "total_probability":           total_P,
            "classical_sum":               classical_sum,
            "interference":                interference,
            "interference_sign":           sign,
            "explicit_interference_terms": explicit_terms,
            "logic_weighted_interference": scale * total_cross,
            "interference_scale":          scale,
            "path_entropy":                entropy,
            "path_entropy_norm":           entropy_norm,
            "n_paths":                     k,
        }

    def forward(
        self,
        source_quats:  list[tuple[torch.Tensor, ...]],   # B source quaternions
        target_quats:  list[tuple[torch.Tensor, ...]],   # B target quaternions
        paths_batch:   list[list],                        # B lists of paths
        quat_unitary,
        logic_scores_batch: Optional[list[list[float]]] = None,
    ) -> torch.Tensor:
        """
        Compute V4 interference probability for a batch.

        Returns:
            (B,) float32 probabilities in [0, 1].
        """
        B     = len(source_quats)
        probs = torch.zeros(B, device=source_quats[0][0].device)

        for b in range(B):
            paths = paths_batch[b][:self.max_paths]
            logic_scores = logic_scores_batch[b] if logic_scores_batch else None

            if not paths:
                continue

            result = self.compute_interference_terms(
                source_quats[b], target_quats[b],
                paths, quat_unitary, logic_scores,
            )
            probs[b] = float(result["total_probability"])

        return probs.clamp(0.0, 1.0)
