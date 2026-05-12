"""
experiments/ — Entry Point Scripts for All Experiment Phases.

Package structure:
    run_toy.py          V1  8-step pipeline verification. Run first always.
    run_fb15k237.py     V2  Full FB15k-237 benchmark run.
    train_toy.py        V2  Toy KG training to fix the 27% problem.
    hardware/           V3  Real quantum hardware integration.

Import shortcuts for programmatic use:
    from experiments import run_pipeline_check
    from experiments import run_full_benchmark
"""
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT    = EXPERIMENTS_DIR.parent
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"


def run_pipeline_check() -> bool:
    """Programmatic entry point for run_toy.py Step verification."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(EXPERIMENTS_DIR / "run_toy.py")],
        capture_output=False,
    )
    return result.returncode == 0


def run_full_benchmark(dataset: str = "fb15k237", quick: bool = False) -> bool:
    """Programmatic entry point for run_fb15k237.py."""
    import subprocess, sys
    args = [sys.executable, str(EXPERIMENTS_DIR / "run_fb15k237.py"),
            f"--dataset={dataset}"]
    if quick:
        args.append("--quick_mode")
    result = subprocess.run(args, capture_output=False)
    return result.returncode == 0


__all__ = ["run_pipeline_check", "run_full_benchmark",
           "EXPERIMENTS_DIR", "PROJECT_ROOT", "OUTPUTS_DIR"]
