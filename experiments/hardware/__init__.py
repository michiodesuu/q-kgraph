"""
experiments/hardware/ — Real Quantum Hardware Integration  [V3]

This package bridges the classical simulation (quantum_kg) with real
quantum processing units (QPUs) via PennyLane.

Modules:
    quantum_circuit.py      PennyLane SWAP test + Zero-Noise Extrapolation
    ibm_integration.py      IBM Quantum via Qiskit Runtime Primitives
    braket_integration.py   AWS Braket Hybrid Jobs
    subspace_projection.py  Formal Subspace Projection Strategy

Hardware Requirements:
    IBM free tier:  5-7 qubits → supports complex_dim=2 (demo only)
    IBM Eagle:      127 qubits → supports complex_dim=32
    Full model:     complex_dim=128 → 256+ qubits (not yet public)

Software Requirements:
    pip install pennylane pennylane-qiskit qiskit-ibm-runtime amazon-braket-sdk

The recommended strategy for the paper:
    1. Train classically at complex_dim=128
    2. Project to complex_dim=4 via subspace_projection.py
    3. Run 3 contradiction queries on IBM hardware at complex_dim=4
    4. Show hardware results match classical simulation ± shot noise
    5. Cite as "hardware validation" in paper Section 6
"""

HARDWARE_REQUIREMENTS = {
    "complex_dim_2":  {"qubits_per_entity": 2, "total_qubits": 5,  "backend": "ibm_free_tier"},
    "complex_dim_4":  {"qubits_per_entity": 2, "total_qubits": 9,  "backend": "ibm_free_tier"},
    "complex_dim_8":  {"qubits_per_entity": 3, "total_qubits": 7,  "backend": "ibm_free_tier"},
    "complex_dim_32": {"qubits_per_entity": 5, "total_qubits": 11, "backend": "ibm_eagle"},
    "complex_dim_128":{"qubits_per_entity": 7, "total_qubits": 15, "backend": "future"},
}

def get_hardware_requirements(complex_dim: int) -> dict:
    """Return hardware requirements for a given complex_dim."""
    import math
    n_qubits  = math.ceil(math.log2(complex_dim))
    total     = 2 * n_qubits + 1  # 2 registers + 1 ancilla
    power_of2 = 2 ** n_qubits == complex_dim

    return {
        "complex_dim":      complex_dim,
        "qubits_per_entity": n_qubits,
        "total_qubits":     total,
        "is_power_of_2":    power_of2,
        "feasible_ibm_free": total <= 7,
        "feasible_ibm_eagle": total <= 127,
        "warning": None if power_of2 else
                   f"complex_dim={complex_dim} is not a power of 2. "
                   f"Use amplitude encoding requires exact power of 2. "
                   f"Nearest valid: {2**n_qubits} or {2**(n_qubits+1)}",
    }

__all__ = [
    "HARDWARE_REQUIREMENTS",
    "get_hardware_requirements",
]
