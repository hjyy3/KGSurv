"""Determinism utilities.

Calling lock_determinism(seed, device) installs the constraints used for
deterministic experiment replay:
  1. CUBLAS_WORKSPACE_CONFIG=:4096:8 (set at module import time)
  2. random.seed
  3. numpy.random.seed
  4. torch.manual_seed
  5. torch.cuda.manual_seed_all (if CUDA)
  6. cudnn deterministic + benchmark off
  7. torch.use_deterministic_algorithms(True, warn_only=True)
"""
from __future__ import annotations

import hashlib
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def lock_determinism(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_snapshot() -> dict:
    """Capture current RNG state across libraries for later replay."""
    snap = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def state_dict_hash(sd: dict) -> str:
    """SHA256 over sorted state_dict keys + tensor bytes."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k].detach().cpu().contiguous().numpy()
        h.update(k.encode())
        h.update(v.tobytes())
    return h.hexdigest()
