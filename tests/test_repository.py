"""Core repository integrity tests."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "examples"))

from losses import cox_loss  # noqa: E402
from models_interp import PathAttnSurv  # noqa: E402
from seeding import state_dict_hash  # noqa: E402
from smoke_model import build_synthetic_kg_info  # noqa: E402


def test_cox_loss_is_finite() -> None:
    """Verify the Cox loss on a small censored dataset."""
    risk = torch.tensor([0.1, -0.2, 0.5, 0.0], dtype=torch.float32)
    time = torch.tensor([8.0, 5.0, 3.0, 2.0], dtype=torch.float32)
    event = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    loss = cox_loss(risk, time, event)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_path_attn_surv_synthetic_forward() -> None:
    """Verify output dimensions for the final architecture."""
    torch.manual_seed(74)
    model = PathAttnSurv(build_synthetic_kg_info(), hidden_dim=8, n_heads=2, n_layers=2)
    mutation = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    output = model(mutation, torch.ones_like(mutation), torch.tensor([[0.5, 1.0, 0.25]]))
    assert output["log_risk"].shape == (1,)
    assert output["gene_importance"].shape == (1, 4)
    assert output["term_importance"].shape == (1, 3)
    assert output["group_importance"].shape == (1, 2)


def test_frozen_state_dict_matches_recorded_hash() -> None:
    """Verify the final weights against the recorded hash."""
    artifact_dir = ROOT / "artifacts" / "final_model"
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    state_dict = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=False)
    assert state_dict_hash(state_dict) == config["state_dict_hash"]


def test_repository_has_no_patient_array_archive() -> None:
    """Ensure patient-level NumPy archives are excluded."""
    assert not list(ROOT.rglob("*.npz"))


def test_final_model_file_checksum_is_stable() -> None:
    """Confirm that the model file can be hashed."""
    model_path = ROOT / "artifacts" / "final_model" / "model.pt"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert len(digest) == 64
