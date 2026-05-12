"""
experiments/hardware/ibm_integration.py — IBM Quantum via Qiskit Runtime  [V3]

PURPOSE:
    Executes SWAP test circuits on IBM quantum hardware using Qiskit Runtime
    Primitives (Sampler and Estimator).

QISKIT PRIMITIVES EXPLAINED:
    Sampler:   Returns quasi-probability distribution of measurement outcomes.
               Input:  quantum circuit
               Output: {'0': 0.85, '1': 0.15}
               Use:    SWAP test Born rule measurement. Always use Sampler for this.

    Estimator: Returns expectation value ⟨ψ|H|ψ⟩ for a given observable H.
               Input:  circuit + observable (Pauli string)
               Output: scalar float
               Use:    When relations are modelled as Pauli Hamiltonians
                       (advanced QCRM-style integration). Not needed for baseline.

IBM TOPOLOGY — HEAVY-HEX LATTICE:
    IBM hardware has restricted qubit connectivity (heavy-hex lattice).
    Adjacent qubits can interact via CNOT/CSWAP directly.
    Non-adjacent qubits require SWAP routing (adds circuit depth).

    For the SWAP test with n_qubits=2 (complex_dim=4):
        Total qubits = 5 (1 ancilla + 2 + 2)
        Heavy-hex connectivity: 5-qubit connected chain is always available
        No routing overhead for our circuit.

    For n_qubits=3 (complex_dim=8):
        Total qubits = 7
        May require 1-2 additional SWAP gates for routing.
        Increases circuit depth by ~10%.

SHOT BUDGET FOR THE PAPER:
    3 contradiction queries × K=4 paths × 4096 shots = 49,152 circuits
    At ~1ms per shot on IBM:  ~49 seconds
    At ~5ms per shot on IBM:  ~245 seconds (~4 minutes)

USAGE:
    runner = IBMQuantumRunner(
        ibm_token    = "YOUR_TOKEN_HERE",
        backend_name = "ibm_nairobi",
        complex_dim  = 4,
        shots        = 4096,
    )
    results = runner.run_contradiction_queries(model, kg)

REQUIREMENTS:
    pip install qiskit-ibm-runtime pennylane-qiskit
    IBM Quantum account: https://quantum.ibm.com/
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


# ── Optional IBM imports ───────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        SamplerV2 as Sampler,
        Session,
    )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    warnings.warn(
        "qiskit-ibm-runtime not installed. IBM hardware integration unavailable. "
        "Install with: pip install qiskit-ibm-runtime",
        ImportWarning,
        stacklevel=2,
    )


@dataclass
class IBMJobResult:
    """
    Result from one IBM Quantum job execution.

    Attributes:
        job_id:           IBM Quantum job identifier (for retrieval/debugging).
        backend_name:     Hardware backend used.
        circuit_name:     Human-readable circuit identifier.
        p_ancilla_0:      Empirical P(ancilla=0) from measurement counts.
        born_rule_prob:   |⟨a|b⟩|² = 2*P(0) - 1 (clamped).
        counts:           Raw measurement count dictionary.
        total_shots:      Total shots executed.
        execution_time_s: Wall-clock time for job execution.
        circuit_depth:    Compiled circuit depth after transpilation.
        error_margin:     Statistical bound ±1/√shots.
        transpile_warnings: Any warnings from topology routing.
    """
    job_id:            str
    backend_name:      str
    circuit_name:      str
    p_ancilla_0:       float
    born_rule_prob:    float
    counts:            dict
    total_shots:       int
    execution_time_s:  float = 0.0
    circuit_depth:     int   = 0
    error_margin:      float = 0.0
    transpile_warnings:list  = field(default_factory=list)

    def __post_init__(self):
        self.born_rule_prob = max(0.0, min(1.0, self.born_rule_prob))
        if self.total_shots > 0:
            self.error_margin = 1.0 / np.sqrt(self.total_shots)

    def is_valid(self) -> bool:
        """True if result is not pure noise (P(0) meaningfully different from 0.5)."""
        return abs(self.p_ancilla_0 - 0.5) > 2 * self.error_margin


class IBMQuantumRunner:
    """
    Runs SWAP test circuits on IBM Quantum hardware via Qiskit Runtime Primitives.

    Uses the Sampler primitive for Born rule measurement.

    Args:
        ibm_token:    IBM Quantum API token from quantum.ibm.com.
        backend_name: IBM hardware backend (e.g., "ibm_nairobi", "ibm_perth").
                      Free tier options (as of 2025): ibm_nairobi, ibm_perth (5q).
                      Eagle processor (127q): ibm_eagle_r3 (premium access).
        complex_dim:  Entity state dimension (must be power of 2).
                      complex_dim=4 → 2 qubits per entity → 5 total qubits.
                      complex_dim=8 → 3 qubits per entity → 7 total qubits.
        shots:        Circuit executions per job.
        optimization_level: Transpilation optimization (0=none, 3=maximum).
                            Use 3 for hardware (minimizes circuit depth).
        log_dir:      Directory to save job results for reproducibility.
        verbose:      Print job status and timing.

    Example:
        >>> runner = IBMQuantumRunner(
        ...     ibm_token    = "MY_TOKEN",
        ...     backend_name = "ibm_nairobi",
        ...     complex_dim  = 4,
        ...     shots        = 4096,
        ... )
        >>> # Run demonstration on 3 contradiction queries
        >>> results = runner.run_contradiction_queries(trained_model, toy_kg)
        >>> for r in results:
        ...     print(r.summary() if hasattr(r, 'summary') else r)
    """

    def __init__(
        self,
        ibm_token:          str,
        backend_name:       str   = "ibm_nairobi",
        complex_dim:        int   = 4,
        shots:              int   = 4096,
        optimization_level: int   = 3,
        log_dir:            Optional[str] = None,
        verbose:            bool  = True,
    ) -> None:
        if not IBM_AVAILABLE:
            raise ImportError(
                "qiskit-ibm-runtime is required. Install: pip install qiskit-ibm-runtime"
            )

        import math
        n_qubits = math.log2(complex_dim)
        if not n_qubits.is_integer():
            raise ValueError(f"complex_dim={complex_dim} must be a power of 2")

        self.ibm_token          = ibm_token
        self.backend_name       = backend_name
        self.complex_dim        = complex_dim
        self.n_qubits           = int(n_qubits)
        self.total_qubits       = 2 * self.n_qubits + 1
        self.shots              = shots
        self.optimization_level = optimization_level
        self.log_dir            = Path(log_dir) if log_dir else None
        self.verbose            = verbose

        self.ancilla_qubit = 0
        self.a_qubits      = list(range(1, self.n_qubits + 1))
        self.b_qubits      = list(range(self.n_qubits + 1, 2 * self.n_qubits + 1))

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(
                f"[IBMQuantumRunner] backend={backend_name}, "
                f"complex_dim={complex_dim}, n_qubits={self.n_qubits}, "
                f"total_qubits={self.total_qubits}, shots={shots}"
            )

    def _connect_service(self):
        """Establish connection to IBM Quantum service."""
        if not IBM_AVAILABLE:
            raise RuntimeError("qiskit-ibm-runtime not installed")

        service = QiskitRuntimeService(
            channel = "ibm_quantum",
            token   = self.ibm_token,
        )
        backend = service.backend(self.backend_name)

        if self.verbose:
            print(f"[IBMQuantumRunner] Connected to {self.backend_name}")
            print(f"  Status: {backend.status()}")
            print(f"  Qubits: {backend.num_qubits}")

        return service, backend

    def _build_swap_test_circuit(
        self,
        a_state: np.ndarray,
        b_state: np.ndarray,
        circuit_name: str = "swap_test",
    ) -> QuantumCircuit:
        """
        Build a Qiskit QuantumCircuit implementing the SWAP test.

        Circuit structure (5 qubits for complex_dim=4):
            q0: ancilla
            q1, q2: register A = |a⟩
            q3, q4: register B = |b⟩

        Args:
            a_state: Complex array of shape (complex_dim,) — evolved path state.
            b_state: Complex array of shape (complex_dim,) — target state.
            circuit_name: Label for the circuit (for IBM job metadata).

        Returns:
            Qiskit QuantumCircuit with measurement on ancilla.
        """
        qc = QuantumCircuit(self.total_qubits, 1, name=circuit_name)

        # Normalize states
        a = a_state.astype(np.complex128)
        b = b_state.astype(np.complex128)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        # Step 1: Initialize register A with state |a⟩
        qc.initialize(a, self.a_qubits)

        # Step 2: Initialize register B with state |b⟩
        qc.initialize(b, self.b_qubits)

        # Step 3: Hadamard on ancilla
        qc.h(self.ancilla_qubit)

        # Step 4: Controlled-SWAP (Fredkin gates)
        for i in range(self.n_qubits):
            qc.cswap(self.ancilla_qubit, self.a_qubits[i], self.b_qubits[i])

        # Step 5: Second Hadamard on ancilla
        qc.h(self.ancilla_qubit)

        # Step 6: Measure ancilla → classical bit 0
        qc.measure(self.ancilla_qubit, 0)

        return qc

    def _transpile_circuit(self, qc: QuantumCircuit, backend):
        """
        Transpile circuit for IBM hardware topology.

        IBM hardware has restricted connectivity (heavy-hex lattice).
        Transpilation inserts SWAP gates to route non-adjacent qubits.
        optimization_level=3 minimizes gate count and depth.
        """
        pm = generate_preset_pass_manager(
            optimization_level = self.optimization_level,
            backend            = backend,
        )
        transpiled = pm.run(qc)

        if self.verbose:
            print(
                f"  Transpiled: depth={transpiled.depth()}, "
                f"gates={transpiled.count_ops()}"
            )

        return transpiled

    def _execute_sampler(
        self,
        circuits:  list,
        backend,
        session:   Optional[Session] = None,
    ) -> list[dict]:
        """
        Execute circuits using Qiskit Sampler primitive.

        The Sampler returns quasi-probability distributions:
            {'0': 0.85, '1': 0.15}
        for measurement outcomes of the classical register.

        For our SWAP test circuit: register has 1 bit (ancilla measurement).
        Parse P('0') as P(ancilla=0).

        Args:
            circuits: List of transpiled Qiskit QuantumCircuit objects.
            backend:  IBM hardware backend.
            session:  Optional active Session for batched execution.

        Returns:
            List of count dictionaries, one per circuit.
        """
        if session is not None:
            sampler = Sampler(session=session)
        else:
            sampler = Sampler(backend=backend)

        # Package circuits into PUBs (Primitive Unified Blocs)
        # PUB = (circuit,) for Sampler V2
        pubs = [(qc,) for qc in circuits]

        t0   = time.perf_counter()
        job  = sampler.run(pubs, shots=self.shots)

        if self.verbose:
            print(f"  Job submitted: {job.job_id()}")
            print(f"  Waiting for results...")

        results   = job.result()
        elapsed   = time.perf_counter() - t0

        if self.verbose:
            print(f"  Execution time: {elapsed:.2f}s")

        # Extract counts from each PUB result
        count_dicts = []
        for pub_result in results:
            counts_obj = pub_result.data.c.get_counts()  # SamplerPubResult
            count_dicts.append(counts_obj)

        return count_dicts, elapsed, job.job_id()

    def run_single_query(
        self,
        a_state:      np.ndarray,
        b_state:      np.ndarray,
        circuit_name: str = "swap_test",
        session:      Optional[Session] = None,
    ) -> IBMJobResult:
        """
        Run a single SWAP test query on IBM hardware.

        Args:
            a_state:      Path-evolved source state |s'⟩ = U_P|s⟩.
            b_state:      Target entity state |t⟩.
            circuit_name: Human-readable label for this circuit.
            session:      Active IBM Session (reuse for multiple queries).

        Returns:
            IBMJobResult with P(ancilla=0) and derived Born rule probability.
        """
        service, backend = self._connect_service()

        # Build and transpile circuit
        qc         = self._build_swap_test_circuit(a_state, b_state, circuit_name)
        qc_trans   = self._transpile_circuit(qc, backend)
        depth      = qc_trans.depth()

        # Execute
        count_dicts, elapsed, job_id = self._execute_sampler(
            [qc_trans], backend, session
        )
        counts = count_dicts[0]

        # Parse P(ancilla=0)
        total_shots = sum(counts.values())
        count_0     = counts.get("0", counts.get(0, 0))
        p0          = count_0 / max(total_shots, 1)

        # Born rule extraction
        born_prob = max(0.0, min(1.0, 2.0 * p0 - 1.0))

        result = IBMJobResult(
            job_id           = str(job_id),
            backend_name     = self.backend_name,
            circuit_name     = circuit_name,
            p_ancilla_0      = p0,
            born_rule_prob   = born_prob,
            counts           = dict(counts),
            total_shots      = total_shots,
            execution_time_s = elapsed,
            circuit_depth    = depth,
        )

        if self.log_dir:
            self._save_result(result)

        return result

    def run_batch(
        self,
        queries: list[tuple[np.ndarray, np.ndarray, str]],
        use_session: bool = True,
    ) -> list[IBMJobResult]:
        """
        Run multiple SWAP test circuits in a single IBM Session.

        Using a Session groups all circuits into one priority queue slot,
        dramatically reducing total wait time compared to separate submissions.

        Args:
            queries: List of (a_state, b_state, circuit_name) tuples.
                     For K=4 paths × 3 queries = 12 circuits.
            use_session: Use IBM Session for priority access (recommended).

        Returns:
            List of IBMJobResult, one per query.

        Usage for paper experiments:
            queries = []
            for cq in kg.contradiction_queries:
                for i, path in enumerate(paths):
                    queries.append((evolved_state, t_state, f"{cq['head']}_path_{i}"))
            results = runner.run_batch(queries)
        """
        service, backend = self._connect_service()

        # Build all circuits
        circuits = []
        for a_state, b_state, name in queries:
            qc      = self._build_swap_test_circuit(a_state, b_state, name)
            qc_trans= self._transpile_circuit(qc, backend)
            circuits.append((qc_trans, name))

        if self.verbose:
            total_shots = len(circuits) * self.shots
            print(f"\n[IBMQuantumRunner] Submitting {len(circuits)} circuits, "
                  f"{total_shots:,} total shots")

        # Execute in session (priority access)
        results = []
        if use_session:
            with Session(backend=backend) as session:
                for i, (qc_trans, name) in enumerate(circuits):
                    if self.verbose:
                        print(f"  [{i+1}/{len(circuits)}] Running {name}...")

                    count_dicts, elapsed, job_id = self._execute_sampler(
                        [qc_trans], backend, session=session
                    )
                    counts = count_dicts[0]
                    total  = sum(counts.values())
                    c0     = counts.get("0", counts.get(0, 0))
                    p0     = c0 / max(total, 1)

                    results.append(IBMJobResult(
                        job_id           = str(job_id),
                        backend_name     = self.backend_name,
                        circuit_name     = name,
                        p_ancilla_0      = p0,
                        born_rule_prob   = max(0.0, min(1.0, 2*p0 - 1.0)),
                        counts           = dict(counts),
                        total_shots      = total,
                        execution_time_s = elapsed,
                    ))
        else:
            for qc_trans, name in circuits:
                result = self.run_single_query(
                    np.zeros(self.complex_dim, dtype=complex),
                    np.zeros(self.complex_dim, dtype=complex),
                    circuit_name=name,
                )
                results.append(result)

        return results

    def run_contradiction_queries(
        self,
        trained_model,
        toy_kg,
        device: Optional[torch.device] = None,
    ) -> list[dict]:
        """
        Run the 3 Platypus/Bat/Whale contradiction queries on IBM hardware.

        This is the paper's Section 6 hardware validation experiment.

        For each query:
            1. Enumerate paths (correct and wrong)
            2. For each wrong-answer path: run SWAP test on hardware
            3. Compare hardware result to classical simulation
            4. Report agreement_ratio and whether within shot noise

        Returns:
            List of result dicts with hardware + classical comparison.
        """
        from models.components.path_aggregator import PathEnumerator

        if device is None:
            device = next(trained_model.parameters()).device

        trained_model.eval()
        adj        = toy_kg.get_adjacency()
        enumerator = PathEnumerator(adj, max_hops=3, max_paths=4)  # K=4 for hardware

        all_results = []
        total_start = time.perf_counter()

        if self.verbose:
            print(f"\n[IBMQuantumRunner] Running 3 contradiction queries on {self.backend_name}")
            print(f"  complex_dim={self.complex_dim}, shots={self.shots}")
            print(f"  Expected time: ~{3 * 4 * self.shots / 1000:.0f} seconds\n")

        for cq in toy_kg.contradiction_queries:
            h_id     = toy_kg.entity2id[cq["head"]]
            corr_id  = toy_kg.entity2id[cq["correct_tail"]]
            wrong_id = toy_kg.entity2id[cq["contradictory_tail"]]

            with torch.no_grad():
                h_state    = trained_model.encoder(
                    torch.tensor([h_id], device=device)).squeeze(0)
                t_wrong    = trained_model.encoder(
                    torch.tensor([wrong_id], device=device)).squeeze(0)

            wrong_paths = enumerator.find_paths(h_id, wrong_id)[:4]  # max 4 paths

            query_result = {
                "query":          cq["query"],
                "head":           cq["head"],
                "wrong_tail":     cq["contradictory_tail"],
                "hardware_probs": [],
                "classical_probs": [],
                "agreement_ratios": [],
            }

            for i, path in enumerate(wrong_paths):
                # Evolve source through path
                evolved = h_state.clone()
                with torch.no_grad():
                    for rel_id, _ in path:
                        rel_t   = torch.tensor([rel_id], device=device)
                        evolved = trained_model.unitary.apply(
                            evolved.unsqueeze(0), rel_t
                        ).squeeze(0)

                evolved_np    = evolved.cpu().numpy()
                t_wrong_np    = t_wrong.cpu().numpy()

                # Hardware
                hw_result = self.run_single_query(
                    evolved_np, t_wrong_np,
                    circuit_name=f"{cq['head']}_wrong_path_{i}"
                )

                # Classical comparison
                inner     = np.sum(np.conj(evolved_np / np.linalg.norm(evolved_np)) *
                                   (t_wrong_np / np.linalg.norm(t_wrong_np)))
                classical_p = float(np.abs(inner) ** 2)

                agreement   = hw_result.born_rule_prob / max(classical_p, 1e-8)

                query_result["hardware_probs"].append(hw_result.born_rule_prob)
                query_result["classical_probs"].append(classical_p)
                query_result["agreement_ratios"].append(agreement)

                if self.verbose:
                    print(
                        f"  {cq['head']} wrong path {i}: "
                        f"hardware={hw_result.born_rule_prob:.4f}, "
                        f"classical={classical_p:.4f}, "
                        f"agreement={agreement:.3f} "
                        f"({'✓' if hw_result.is_valid() else 'noise?'})"
                    )

            # Summary stats
            if query_result["hardware_probs"]:
                query_result["mean_hardware"]  = float(np.mean(query_result["hardware_probs"]))
                query_result["mean_classical"] = float(np.mean(query_result["classical_probs"]))
                query_result["mean_agreement"] = float(np.mean(query_result["agreement_ratios"]))
                query_result["hardware_validated"] = (
                    0.85 <= query_result["mean_agreement"] <= 1.15
                )

            all_results.append(query_result)

        total_time = time.perf_counter() - total_start
        if self.verbose:
            n_validated = sum(1 for r in all_results if r.get("hardware_validated", False))
            print(f"\n[IBMQuantumRunner] Complete in {total_time:.1f}s")
            print(f"Hardware validated: {n_validated}/3 queries (agreement ratio 0.85-1.15)")

        return all_results

    def _save_result(self, result: IBMJobResult) -> None:
        """Save result to log directory for reproducibility."""
        if not self.log_dir:
            return
        fname = self.log_dir / f"ibm_{result.job_id}_{result.circuit_name}.json"
        with open(fname, "w") as f:
            json.dump({
                "job_id":         result.job_id,
                "backend":        result.backend_name,
                "p_ancilla_0":    result.p_ancilla_0,
                "born_rule_prob": result.born_rule_prob,
                "counts":         result.counts,
                "shots":          result.total_shots,
                "depth":          result.circuit_depth,
                "time_s":         result.execution_time_s,
            }, f, indent=2)
