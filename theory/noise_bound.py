"""
theory/noise_bound.py — Supporting Mathematical Utilities for Theorem 8.3  [V2]

Provides numerical tools for working with the noise-robustness bound:
    - Computing and plotting ΔP_Q(p) across noise levels
    - Finding the crossover point where quantum beats classical
    - Sensitivity analysis: how does the gap change with φ?
    - Analytical verification helpers

USAGE:
    from theory.noise_bound import (
        quantum_gap_curve, classical_gap_curve, compute_crossover,
        phase_sensitivity, NoiseBoundAnalyzer,
    )

    analyzer = NoiseBoundAnalyzer(r=0.3, K=4, phi=2.5)
    analyzer.print_summary()
    crossover = compute_crossover(r=0.3, K=4, phi=2.5)
    print(f"Quantum > Classical crossover: p = {crossover:.2f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Core formulas ─────────────────────────────────────────────────────────────

def quantum_gap_curve(
    noise_levels: list[float],
    r:            float,
    K:            float,
    phi:          float,
    K_prime:      Optional[float] = None,
) -> list[float]:
    """
    Compute ΔP_Q(p) = r²K²(1-p)²[1 - cos²(φ/2)] at multiple noise levels.

    This is the quantum model's probability gap: P_Q(correct) - P_Q(wrong).
    Always non-negative when φ > 0.

    Args:
        noise_levels: List of noise rates p ∈ [0, 1].
        r:            Mean path amplitude magnitude.
        K:            Number of correct-answer paths.
        phi:          Phase separation in radians [0, π].
        K_prime:      Number of wrong-answer paths. Default: K (equal counts).

    Returns:
        List of gap values, one per noise level.
    """
    Kp = K_prime if K_prime is not None else K
    gaps = []
    for p in noise_levels:
        p_corr  = (r ** 2) * (K  ** 2) * ((1 - p) ** 2)
        p_wrong = (r ** 2) * (Kp ** 2) * ((1 - p) ** 2) * (math.cos(phi / 2) ** 2)
        gaps.append(max(0.0, p_corr - p_wrong))
    return gaps


def classical_gap_curve(
    noise_levels: list[float],
    r:            float,
    K:            float,
    K_prime:      Optional[float] = None,
) -> list[float]:
    """
    Compute ΔP_C(p) = r²(1-p)(K - K') for a classical additive model.

    When K = K': ΔP_C = 0 at all noise levels (no discrimination).
    When K > K': gap decreases linearly with p.

    Args:
        noise_levels: List of noise rates.
        r:            Mean embedding magnitude.
        K:            Correct-answer path count.
        K_prime:      Wrong-answer path count. Default: K (equal counts).

    Returns:
        List of classical gap values.
    """
    Kp   = K_prime if K_prime is not None else K
    delta = K - Kp
    return [(r ** 2) * (1 - p) * delta for p in noise_levels]


def quantum_advantage_ratio(
    noise_levels: list[float],
    r:            float,
    K:            float,
    phi:          float,
    K_prime:      Optional[float] = None,
) -> list[float]:
    """
    Compute ΔP_Q(p) / ΔP_C(p) — the quantum advantage ratio.

    > 1 means quantum is better. → ∞ as ΔP_C → 0.
    Only meaningful when K ≠ K' (classical gap is non-zero).

    Args:
        See quantum_gap_curve.

    Returns:
        List of advantage ratios. Returns inf where classical gap = 0.
    """
    q_gaps = quantum_gap_curve(noise_levels, r, K, phi, K_prime)
    c_gaps = classical_gap_curve(noise_levels, r, K, K_prime)
    ratios = []
    for q, c in zip(q_gaps, c_gaps):
        if abs(c) < 1e-10:
            ratios.append(float("inf") if q > 1e-10 else 1.0)
        else:
            ratios.append(q / c)
    return ratios


def compute_crossover(
    r:       float,
    K:       float,
    phi:     float,
    K_prime: Optional[float] = None,
    n_steps: int = 1000,
) -> Optional[float]:
    """
    Find the noise rate p where ΔP_Q(p) = ΔP_C(p).

    Below this crossover: classical model may be competitive.
    Above this crossover: quantum advantage is guaranteed.

    Args:
        r, K, phi, K_prime: Theorem parameters.
        n_steps: Resolution for binary search.

    Returns:
        Crossover noise rate, or None if no crossover exists.
    """
    ps  = np.linspace(0.0, 0.99, n_steps)
    q   = np.array(quantum_gap_curve(ps.tolist(), r, K, phi, K_prime))
    c   = np.array(classical_gap_curve(ps.tolist(), r, K, K_prime))

    # Find first point where q > c
    for i in range(len(ps) - 1):
        if (q[i] <= c[i]) and (q[i+1] > c[i+1]):
            return float(ps[i])

    # If q > c everywhere, crossover is at p=0
    if q[0] > c[0]:
        return 0.0

    return None  # No crossover found


def phase_sensitivity(
    phi_values: list[float],
    r:          float = 0.3,
    K:          float = 4.0,
    p:          float = 0.10,
) -> list[float]:
    """
    Compute the noise gap ΔP_Q as a function of phase separation φ.

    Used to understand how important phase separation is.
    At φ = 0: gap = 0 (no advantage).
    At φ = π: gap = r²K²(1-p)² (maximum advantage).

    Args:
        phi_values: List of phase separations to evaluate.
        r, K:       Fixed amplitude and path count.
        p:          Fixed noise rate.

    Returns:
        List of gap values, one per φ value.
    """
    return [
        (r ** 2) * (K ** 2) * ((1 - p) ** 2) * (math.sin(phi / 2) ** 2)
        for phi in phi_values
    ]


# ── NoiseBoundAnalyzer ────────────────────────────────────────────────────────

@dataclass
class NoiseBoundAnalyzer:
    """
    Convenience wrapper for computing all noise-bound related quantities.

    Initialized with theorem parameters (r, K, phi) from
    verify_theorem_conditions() output.

    Args:
        r:       Mean amplitude magnitude.
        K:       Correct-answer path count.
        phi:     Phase separation in radians.
        K_prime: Wrong-answer path count (default: K).

    Example:
        >>> from theory.noise_guarantee import verify_theorem_conditions
        >>> results = verify_theorem_conditions(trained_model, toy_kg, device)
        >>> r = results[0]
        >>> if r.theorem_applicable:
        ...     analyzer = NoiseBoundAnalyzer(r=r.r_correct, K=r.K_correct, phi=r.phi)
        ...     analyzer.print_summary()
    """
    r:       float
    K:       float
    phi:     float
    K_prime: Optional[float] = None

    # Default noise levels for analysis
    NOISE_LEVELS: list = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20])

    def __post_init__(self):
        if self.K_prime is None:
            self.K_prime = self.K

    def quantum_gaps(self, noise_levels: Optional[list] = None) -> list[float]:
        """Quantum gap ΔP_Q at each noise level."""
        levels = noise_levels or self.NOISE_LEVELS
        return quantum_gap_curve(levels, self.r, self.K, self.phi, self.K_prime)

    def classical_gaps(self, noise_levels: Optional[list] = None) -> list[float]:
        """Classical gap ΔP_C at each noise level."""
        levels = noise_levels or self.NOISE_LEVELS
        return classical_gap_curve(levels, self.r, self.K, self.K_prime)

    def advantage_ratios(self, noise_levels: Optional[list] = None) -> list[float]:
        """ΔP_Q / ΔP_C at each noise level."""
        levels = noise_levels or self.NOISE_LEVELS
        return quantum_advantage_ratio(levels, self.r, self.K, self.phi, self.K_prime)

    def crossover_point(self) -> Optional[float]:
        """Noise rate where quantum advantage begins."""
        return compute_crossover(self.r, self.K, self.phi, self.K_prime)

    def min_phi_required(self, p: float = 0.10, min_gap: float = 0.01) -> float:
        """
        Minimum phase separation needed to achieve min_gap at noise rate p.

        Solving: r²K²(1-p)²sin²(φ/2) ≥ min_gap
        → sin(φ/2) ≥ sqrt(min_gap / (r²K²(1-p)²))
        → φ ≥ 2 arcsin(sqrt(min_gap / (r²K²(1-p)²)))
        """
        denom = (self.r ** 2) * (self.K ** 2) * ((1 - p) ** 2)
        if denom < 1e-10:
            return math.pi  # Can't achieve any gap
        ratio = min_gap / denom
        if ratio > 1.0:
            return math.pi
        return 2 * math.asin(math.sqrt(ratio))

    def print_summary(self) -> None:
        """Print a formatted summary of the noise bound analysis."""
        print(f"\n{'='*60}")
        print(f"NOISE BOUND ANALYSIS")
        print(f"  r={self.r:.3f}, K={self.K:.0f}, K'={self.K_prime:.0f}, φ={self.phi:.3f} rad")
        print(f"{'='*60}")
        print(f"{'Noise%':8s} {'ΔP_Q':10s} {'ΔP_C':10s} {'Ratio Q/C':12s}")
        print(f"{'-'*44}")

        q_gaps = self.quantum_gaps()
        c_gaps = self.classical_gaps()
        ratios = self.advantage_ratios()

        for p, q, c, r in zip(self.NOISE_LEVELS, q_gaps, c_gaps, ratios):
            ratio_str = f"{r:.2f}" if r != float("inf") else "∞"
            print(f"{p*100:6.0f}%   {q:10.4f}  {c:10.4f}  {ratio_str:12s}")

        crossover = self.crossover_point()
        if crossover is not None:
            print(f"\nQuantum > Classical crossover: p = {crossover:.2f} ({crossover*100:.0f}% noise)")
        else:
            print(f"\nQuantum ≥ Classical at all noise levels.")

        phi_half = math.pi / 2
        req_phi = self.min_phi_required(p=0.10)
        print(f"\nMin φ for Δ>0.01 at 10% noise: {req_phi:.2f} rad")
        print(f"Current φ={self.phi:.2f} rad {'(satisfies condition)' if self.phi > phi_half else '(φ < π/2: theorem not guaranteed)'}")
        print(f"{'='*60}\n")
