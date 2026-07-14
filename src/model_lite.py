"""KGLiteMLP: Lightweight MLP with pre-computed KG features.

Replaces the 4.5M-parameter HGT model with a ~150K-parameter MLP that
consumes frozen KG-derived features (FMB + Node2Vec embeddings).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class KGLiteMLP(nn.Module):
    """Lightweight MLP: [mut, mask, fmb, kg_emb, tmb] → Cox log-risk.

    Args:
        input_dim:   total feature dimension (varies by KG).
        hidden_dims: MLP hidden layer sizes.
        dropout:     dropout rate between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev, hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hd
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        features_batch: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features_batch: [batch, input_dim] — pre-concatenated features.
        Returns:
            dict with ``"log_risk"`` tensor of shape [batch].
        """
        log_risk = self.mlp(features_batch).squeeze(-1)
        return {"log_risk": log_risk}
