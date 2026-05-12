"""
utils/seed.py — Full Reproducibility Seeding  [V1]
"""
from __future__ import annotations
import os, random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed all RNG sources for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(preferred: str = "cuda") -> torch.device:
    """CUDA → MPS → CPU automatic fallback."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeedContext:
    """Context manager for temporary seed override."""
    def __init__(self, seed: int):
        self.seed = seed
        self._state_python = None
        self._state_numpy  = None
        self._state_torch  = None

    def __enter__(self):
        self._state_python = random.getstate()
        self._state_numpy  = np.random.get_state()
        self._state_torch  = torch.get_rng_state()
        set_seed(self.seed)
        return self

    def __exit__(self, *args):
        random.setstate(self._state_python)
        np.random.set_state(self._state_numpy)
        torch.set_rng_state(self._state_torch)
