"""
experiments/hardware/braket_integration.py — AWS Braket Hybrid Jobs  [V3]

PURPOSE:
    Implements the AWS Braket Hybrid Jobs workflow for quantum-classical
    co-processing. This is the "Quantum-centric Supercomputing" paradigm:
    classical CPU/GPU handles training, QPU handles SWAP test inference.

ARCHITECTURE:
    Classical EC2 instance (ml.m5.2xlarge or ml.g4dn.xlarge):
        - TrainerV2 training loop
        - PathCache BFS
        - InterferenceAwareLoss computation
        - Model weight updates

    Quantum QPU (dispatched from EC2 for specific operations):
        - AmplitudeAggregator SWAP test
        - One circuit per (entity, path) pair
        - Results returned to EC2 for classical post-processing

WHY BRAKET HYBRID JOBS OVER IBM:
    1. Priority queue: Hybrid Jobs secure dedicated QPU time,
       avoiding the public queue (can be hours of wait).
    2. Tighter EC2-QPU integration: lower latency than API calls.
    3. GPU-accelerated simulation: PennyLane lightning.gpu on EC2
       for pre-QPU hyperparameter tuning.
    4. IonQ access: trapped-ion processors have higher gate fidelity
       than IBM superconducting, important for deep SWAP test circuits.

BACKENDS AVAILABLE ON BRAKET:
    "local:pennylane/lightning.gpu"   — GPU-accelerated simulator on EC2
    "braket:aws:sv1"                  — AWS StateVector1 simulator (free)
    "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"    — IonQ 11q
    "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"     — IonQ 25q
    "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"  — Rigetti 80q

USAGE:
    runner = BraketHybridRunner(
        device_arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
        complex_dim = 4,
        shots = 4096,
    )
    results = runner.run_demonstration(trained_model, kg)

REQUIREMENTS:
    pip install amazon-braket-sdk amazon-braket-pennylane-plugin pennylane
    AWS credentials configured (aws configure or IAM role)
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Optional Braket imports ────────────────────────────────────────────────────
try:
    import pennylane as qml
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    from braket.aws import AwsDevice
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    warnings.warn(
        "amazon-braket-sdk not installed. AWS Braket integration unavailable. "
        "Install with: pip install amazon-braket-sdk amazon-braket-pennylane-plugin",
        ImportWarning,
        stacklevel=2,
    )


@dataclass
class BraketJobResult:
    """
    Result from one Braket job.

    Attributes:
        task_arn:       AWS task ARN for retrieval and debugging.
        device_arn:     QPU device used.
        circuit_name:   Human-readable circuit label.
        p_ancilla_0:    P(ancilla=0) from measurement statistics.
        born_rule_prob: |⟨a|b⟩|² = 2*P(0) - 1 (clamped).
        measurement_counts: Raw counts dict.
        total_shots:    Shots executed.
        execution_time_s: Wall clock time.
        gate_fidelity:  Reported gate fidelity (from device calibration).
    """
    task_arn:          str
    device_arn:        str
    circuit_name:      str
    p_ancilla_0:       float
    born_rule_prob:    float
    measurement_counts: dict
    total_shots:       int
    execution_time_s:  float = 0.0
    gate_fidelity:     float = 0.0

    def __post_init__(self):
        self.born_rule_prob = max(0.0, min(1.0, self.born_rule_prob))

    def is_valid(self) -> bool:
        return abs(self.p_ancilla_0 - 0.5) > 1.0 / np.sqrt(max(self.total_shots, 1)) * 2


class BraketHybridRunner:
    """
    Runs SWAP test circuits on AWS Braket QPUs as part of a Hybrid Job.

    The key advantage over direct IBM access: Hybrid Jobs spin up a
    dedicated EC2 instance alongside the QPU, enabling tight classical-
    quantum co-processing without queue latency between iterations.

    For the paper's purposes, we use Braket in two modes:
        1. Local simulator (lightning.gpu): fast, exact, for tuning
        2. IonQ Aria: real hardware, higher fidelity than IBM SC qubits

    Why IonQ over IBM for small circuits:
        IonQ trapped-ion qubits have all-to-all connectivity
        (no topology routing overhead) and better T1/T2 for shallow circuits.
        For our 5-qubit SWAP test, IonQ Harmony is available on Braket
        and requires zero topology routing insertions.

    Args:
        device_arn:   Braket device ARN or local simulator string.
        complex_dim:  Entity state complex dimension (must be power of 2).
        shots:        Circuit executions per task.
        s3_bucket:    S3 bucket for task result storage (required for QPU).
        s3_prefix:    S3 key prefix for results.
        log_dir:      Local directory for result logging.
        verbose:      Print job status and timing.

    Example:
        >>> # Use local GPU simulator for pre-QPU testing
        >>> runner = BraketHybridRunner(
        ...     device_arn = "local:pennylane/lightning.gpu",
        ...     complex_dim = 4,
        ...     shots = 4096,
        ... )
        >>> results = runner.run_demonstration(trained_model, kg)
    """

    # Default device ARNs
    IONQ_HARMONY  = "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"
    IONQ_ARIA     = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
    RIGETTI_M3    = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"
    SV1_SIMULATOR = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
    LOCAL_SIM     = "local:pennylane/lightning.gpu"

    def __init__(
        self,
        device_arn:  str   = "local:pennylane/lightning.gpu",
        complex_dim: int   = 4,
        shots:       int   = 4096,
        s3_bucket:   str   = "",
        s3_prefix:   str   = "quantum_kg_results",
        log_dir:     Optional[str] = None,
        verbose:     bool  = True,
    ) -> None:
        if not BRAKET_AVAILABLE:
            raise ImportError(
                "amazon-braket-sdk is required. "
                "Install: pip install amazon-braket-sdk amazon-braket-pennylane-plugin"
            )

        import math
        n = math.log2(complex_dim)
        if not n.is_integer():
            raise ValueError(f"complex_dim={complex_dim} must be power of 2")

        self.device_arn  = device_arn
        self.complex_dim = complex_dim
        self.n_qubits    = int(n)
        self.total_qubits = 2 * self.n_qubits + 1
        self.shots       = shots
        self.s3_bucket   = s3_bucket
        self.s3_prefix   = s3_prefix
        self.log_dir     = Path(log_dir) if log_dir else None
        self.verbose     = verbose

        self.ancilla = 0
        self.a_wires = list(range(1, self.n_qubits + 1))
        self.b_wires = list(range(self.n_qubits + 1, 2 * self.n_qubits + 1))

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build PennyLane device
        self.dev = self._build_device()

        if verbose:
            print(
                f"[BraketHybridRunner] device={device_arn}\n"
                f"  complex_dim={complex_dim}, n_qubits={self.n_qubits}, "
                f"total_qubits={self.total_qubits}, shots={shots}"
            )

    def _build_device(self):
        """Build PennyLane device for the specified Braket backend."""
        if self.device_arn.startswith("local:"):
            # Local simulator (no AWS credentials needed)
            sim_name = self.device_arn.split("local:")[-1]
            try:
                dev = qml.device(
                    sim_name.replace("/", "."),
                    wires = self.total_qubits,
                    shots = self.shots,
                )
                return dev
            except Exception:
                # Fallback to default.qubit if lightning.gpu not available
                if self.verbose:
                    print("  lightning.gpu unavailable, falling back to default.qubit")
                return qml.device("default.qubit", wires=self.total_qubits, shots=self.shots)

        elif "braket" in self.device_arn.lower():
            # AWS Braket QPU
            if not self.s3_bucket:
                raise ValueError(
                    "s3_bucket is required for QPU execution. "
                    "Set s3_bucket='your-s3-bucket-name'"
                )
            return qml.device(
                "braket.aws.qubit",
                device_arn = self.device_arn,
                wires      = self.total_qubits,
                shots      = self.shots,
                s3_destination_folder = (self.s3_bucket, self.s3_prefix),
            )
        else:
            # SV1 simulator
            return qml.device(
                "braket.aws.qubit",
                device_arn = self.device_arn,
                wires      = self.total_qubits,
                shots      = self.shots,
            )

    def _build_swap_test_qnode(self, a_state: np.ndarray, b_state: np.ndarray):
        """Build and execute one SWAP test QNode."""

        @qml.qnode(self.dev, interface="numpy")
        def circuit():
            # Prepare |a⟩ and |b⟩
            try:
                qml.StatePrep(a_state / np.linalg.norm(a_state), wires=self.a_wires, pad_with=0)
                qml.StatePrep(b_state / np.linalg.norm(b_state), wires=self.b_wires, pad_with=0)
            except AttributeError:
                qml.QubitStateVector(a_state / np.linalg.norm(a_state), wires=self.a_wires)
                qml.QubitStateVector(b_state / np.linalg.norm(b_state), wires=self.b_wires)

            qml.Hadamard(wires=self.ancilla)

            for i in range(self.n_qubits):
                qml.CSWAP(wires=[self.ancilla, self.a_wires[i], self.b_wires[i]])

            qml.Hadamard(wires=self.ancilla)

            return qml.probs(wires=self.ancilla)

        return circuit()

    def run_swap_test(
        self,
        a_state:      np.ndarray,
        b_state:      np.ndarray,
        circuit_name: str = "swap_test",
    ) -> BraketJobResult:
        """
        Run one SWAP test on Braket (simulator or QPU).

        Args:
            a_state:      Path-evolved source state (complex_dim,).
            b_state:      Target state (complex_dim,).
            circuit_name: Label for logging.

        Returns:
            BraketJobResult with Born rule probability.
        """
        t0    = time.perf_counter()
        probs = self._build_swap_test_qnode(a_state, b_state)
        elapsed = time.perf_counter() - t0

        p0        = float(probs[0])
        born_prob = max(0.0, min(1.0, 2.0 * p0 - 1.0))

        result = BraketJobResult(
            task_arn          = f"local_{circuit_name}_{int(t0)}",
            device_arn        = self.device_arn,
            circuit_name      = circuit_name,
            p_ancilla_0       = p0,
            born_rule_prob    = born_prob,
            measurement_counts= {"0": int(p0 * self.shots), "1": int((1-p0)*self.shots)},
            total_shots       = self.shots,
            execution_time_s  = elapsed,
        )

        if self.verbose:
            print(
                f"  [{circuit_name}] P(0)={p0:.4f}, "
                f"|⟨a|b⟩|²={born_prob:.4f} "
                f"[{elapsed:.2f}s, {'QPU' if 'ionq' in self.device_arn else 'sim'}]"
            )

        if self.log_dir:
            fname = self.log_dir / f"braket_{circuit_name}.json"
            with open(fname, "w") as f:
                json.dump({
                    "task_arn":   result.task_arn,
                    "device":     self.device_arn,
                    "circuit":    circuit_name,
                    "p0":         p0,
                    "born_prob":  born_prob,
                    "shots":      self.shots,
                    "time_s":     elapsed,
                }, f, indent=2)

        return result

    def run_demonstration(
        self,
        trained_model,
        toy_kg,
        device: Optional[torch.device] = None,
        max_paths_per_query: int = 4,
    ) -> list[dict]:
        """
        Run the 3 Platypus/Bat/Whale contradiction queries.

        For each contradiction query:
            - Enumerate up to max_paths_per_query wrong-answer paths
            - Run SWAP test for each path
            - Compare to classical simulation

        Returns:
            List of result dicts per query.
        """
        from models.components.path_aggregator import PathEnumerator

        if device is None:
            device = next(trained_model.parameters()).device

        trained_model.eval()
        adj        = toy_kg.get_adjacency()
        enumerator = PathEnumerator(adj, max_hops=3, max_paths=max_paths_per_query)

        results    = []
        total_time = 0.0

        if self.verbose:
            print(f"\n[BraketHybridRunner] Starting contradiction query demonstration")
            print(f"  device={self.device_arn}")
            print(f"  complex_dim={self.complex_dim}, shots={self.shots}")

        for cq in toy_kg.contradiction_queries:
            h_id     = toy_kg.entity2id[cq["head"]]
            wrong_id = toy_kg.entity2id[cq["contradictory_tail"]]

            with torch.no_grad():
                h_state    = trained_model.encoder(
                    torch.tensor([h_id], device=device)).squeeze(0)
                t_wrong    = trained_model.encoder(
                    torch.tensor([wrong_id], device=device)).squeeze(0)

            wrong_paths = enumerator.find_paths(h_id, wrong_id)[:max_paths_per_query]

            if self.verbose:
                print(f"\n  Query: {cq['query']}")
                print(f"  Testing wrong answer: {cq['contradictory_tail']} ({len(wrong_paths)} paths)")

            query_hw    = []
            query_class = []

            for i, path in enumerate(wrong_paths):
                # Evolve through path
                evolved = h_state.clone()
                with torch.no_grad():
                    for rel_id, _ in path:
                        rel_t   = torch.tensor([rel_id], device=device)
                        evolved = trained_model.unitary.apply(
                            evolved.unsqueeze(0), rel_t
                        ).squeeze(0)

                evolved_np = evolved.cpu().numpy()
                t_np       = t_wrong.cpu().numpy()

                # Hardware
                hw = self.run_swap_test(
                    evolved_np, t_np,
                    circuit_name=f"{cq['head']}_wrong_p{i}"
                )
                total_time += hw.execution_time_s

                # Classical
                en  = evolved_np / np.linalg.norm(evolved_np)
                tn  = t_np / np.linalg.norm(t_np)
                cls = float(np.abs(np.dot(np.conj(en), tn)) ** 2)

                query_hw.append(hw.born_rule_prob)
                query_class.append(cls)

            mean_hw    = float(np.mean(query_hw)) if query_hw else 0.0
            mean_cls   = float(np.mean(query_class)) if query_class else 0.0
            agreement  = mean_hw / max(mean_cls, 1e-8)

            results.append({
                "query":            cq["query"],
                "head":             cq["head"],
                "wrong_tail":       cq["contradictory_tail"],
                "hw_probs":         query_hw,
                "classical_probs":  query_class,
                "mean_hw":          mean_hw,
                "mean_classical":   mean_cls,
                "agreement_ratio":  agreement,
                "validated":        0.85 <= agreement <= 1.15,
            })

        n_validated = sum(1 for r in results if r["validated"])
        if self.verbose:
            print(f"\n[BraketHybridRunner] Complete in {total_time:.1f}s")
            print(f"Validated: {n_validated}/3 queries (agreement 0.85-1.15)")

        return results
