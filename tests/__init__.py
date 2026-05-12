"""
tests/ — Complete test suite for quantum_kg

Run all tests:    pytest tests/ -v
Run one group:    pytest tests/ -v -k "TestQuantumStates"
Run fast only:    pytest tests/ -v -m "not slow"
Show coverage:    pytest tests/ -v --cov=models --cov=training --cov=evaluation

Test groups:
    TestQuantumStates         — unit norms, Born rule, fidelity
    TestUnitaryOperators      — norm preservation, U†U=I, path composition
    TestPathAggregator        — BFS paths, interference decomposition
    TestKGUnitaries           — relational decomp, hierarchy, contextual
    TestPhaseSeparationLoss   — gradient direction, separation metric
    TestContrastiveLoss       — hinge behavior, contradiction batches
    TestInterferenceReg       — Phase Collapse detection and prevention
    TestInterferenceMonitor   — health checks, auto-correction
    TestTrainerV2             — parameter groups, differential LRs
    TestPathCache             — BFS caching, coverage, load/save
    TestChunkedEvaluator      — matches standard evaluator, OOM-safe
    TestNoiseGuarantee        — Theorem 8.3 conditions, formula
    TestNoiseSeparation       — noise injection correctness
    TestEndToEnd              — full pipeline regression test
"""
