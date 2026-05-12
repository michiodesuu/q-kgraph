"""setup.py — Package installer. Run: pip install -e ."""
from setuptools import setup, find_packages

setup(
    name             = "quantum_kg",
    version          = "0.3.0",
    description      = "Interferential Multi-Hop Reasoning via Quantum Amplitude Interference",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "rich>=13.0.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
    ],
    extras_require = {
        "quantum": [
            "pennylane>=0.35.0",
            "pennylane-qiskit>=0.35.0",
            "qiskit>=1.0.0",
            "qiskit-ibm-runtime>=0.20.0",
        ],
        "nlp":    ["sentence-transformers>=2.2.0"],
        "dev":    ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "wandb":  ["wandb>=0.16.0"],
    },
)
