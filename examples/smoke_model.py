"""Run a synthetic forward pass through the manuscript model architecture."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_interp import KGGroupInfo  # noqa: E402
from models_interp import PathAttnSurv  # noqa: E402


def build_synthetic_kg_info() -> KGGroupInfo:
    """Build a two-group KG structure for a dependency-light smoke test.

    Returns:
        A small, internally consistent KG metadata object.
    """
    group_a = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    group_b = torch.tensor(
        [[0.0], [1.0], [0.0], [1.0]],
        dtype=torch.float32,
    )
    return KGGroupInfo(
        kg_name="synthetic",
        group_names=["pathway", "ppi"],
        term_names=[["term_a", "term_b"], ["term_c"]],
        gene_term_mask=[group_a, group_b],
        fmb_slices=[(0, 2), (2, 3)],
        n_genes=4,
        n_total_terms=3,
    )


def main() -> int:
    """Run the synthetic model and print a compact verification summary.

    Returns:
        Zero when all output-shape and finiteness checks pass.
    """
    torch.manual_seed(74)
    model = PathAttnSurv(build_synthetic_kg_info(), hidden_dim=8, n_heads=2, n_layers=2)
    mutation = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    mask = torch.ones_like(mutation)
    fmb = torch.tensor([[0.5, 1.0, 0.25], [1.5, 0.25, 0.75]], dtype=torch.float32)

    with torch.no_grad():
        output = model(mutation, mask, fmb)

    assert output["log_risk"].shape == (2,)
    assert output["gene_importance"].shape == (2, 4)
    assert output["term_importance"].shape == (2, 3)
    assert output["group_importance"].shape == (2, 2)
    assert torch.isfinite(output["log_risk"]).all()

    print("KGSurv4ICI synthetic smoke test passed")
    print(f"log_risk={output['log_risk'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
