"""Ablation model variants for MutaPathSurv.

Three ablation variants to quantify contribution of each component:
  1. BaselineMLP: Pure [mut, mask] -> MLP -> Cox (no KG, no GNN, no pathway)
  2. MutaPathSurvNoMask: Full model but without panel mask input
  3. MutaPathSurvNoPathway: Full model but skip pathway pooling
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Pure MLP baseline: [mut, mask, tmb] -> Cox log-risk.

    No knowledge graph, no GNN, no pathway structure.
    Quantifies: total contribution of KG prior.
    """

    def __init__(self, n_candidate: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.n_candidate = n_candidate
        self.mlp = nn.Sequential(
            nn.Linear(n_candidate * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        mut_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        tmb_batch: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat([mut_batch, mask_batch, tmb_batch.unsqueeze(-1)], dim=-1)
        log_risk = self.mlp(x).squeeze(-1)
        return {"log_risk": log_risk}


class MutaPathSurvNoMask(nn.Module):
    """Full MutaPathSurv but without panel mask input.

    Mutation features only (no panel coverage awareness).
    Quantifies: contribution of panel-aware modeling.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        # Override gene_fusion to accept hidden_dim + 1 (mut only, no mask)
        hidden_dim = base_model.hidden_dim
        self.gene_fusion_nomask = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(
        self,
        mut_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        tmb_batch: torch.Tensor,
        edge_index_dict: dict,
        pathway_gene_edge: torch.Tensor,
        gene_indices: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        batch_size = mut_batch.size(0)
        device = mut_batch.device

        h_dict = self.base.encode_kg(edge_index_dict)
        gene_kg_emb = h_dict["gene/protein"]

        cand_kg_emb = gene_kg_emb[gene_indices].unsqueeze(0).expand(
            batch_size, -1, -1)

        # Only mutation status, no mask
        patient_feat = mut_batch.unsqueeze(-1)  # [batch, n_cand, 1]
        fused = torch.cat([cand_kg_emb, patient_feat], dim=-1)
        fused = self.gene_fusion_nomask(fused)

        patient_gene_scores = torch.sigmoid(
            self.base.patient_gate(fused)).squeeze(-1)

        fused_mean = fused.mean(dim=0)
        gene_emb_full = torch.zeros(
            self.base.n_genes, self.base.hidden_dim, device=device)
        gene_emb_full[gene_indices] = fused_mean

        pathway_emb, _ = self.base.pathway_pool(
            gene_emb_full, pathway_gene_edge, self.base.n_pathways)

        patient_gene_weighted = fused * patient_gene_scores.unsqueeze(-1)
        patient_emb = patient_gene_weighted.mean(dim=1)

        tmb_feat = tmb_batch.unsqueeze(-1)
        patient_feat_final = torch.cat([patient_emb, tmb_feat], dim=-1)
        log_risk = self.base.cox_head(patient_feat_final).squeeze(-1)

        return {"log_risk": log_risk}


class MutaPathSurvNoPathway(nn.Module):
    """Full MutaPathSurv but without pathway pooling.

    KG embeddings are used but pathway structure is skipped.
    Quantifies: contribution of pathway-level structure.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(
        self,
        mut_batch: torch.Tensor,
        mask_batch: torch.Tensor,
        tmb_batch: torch.Tensor,
        edge_index_dict: dict,
        pathway_gene_edge: torch.Tensor,
        gene_indices: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        batch_size = mut_batch.size(0)
        device = mut_batch.device

        h_dict = self.base.encode_kg(edge_index_dict)
        gene_kg_emb = h_dict["gene/protein"]

        cand_kg_emb = gene_kg_emb[gene_indices].unsqueeze(0).expand(
            batch_size, -1, -1)

        patient_feat = torch.stack([mut_batch, mask_batch], dim=-1)
        fused = torch.cat([cand_kg_emb, patient_feat], dim=-1)
        fused = self.base.gene_fusion(fused)

        patient_gene_scores = torch.sigmoid(
            self.base.patient_gate(fused)).squeeze(-1)

        # Skip pathway pooling — directly aggregate gene embeddings
        patient_gene_weighted = fused * patient_gene_scores.unsqueeze(-1)
        patient_emb = patient_gene_weighted.mean(dim=1)

        tmb_feat = tmb_batch.unsqueeze(-1)
        patient_feat_final = torch.cat([patient_emb, tmb_feat], dim=-1)
        log_risk = self.base.cox_head(patient_feat_final).squeeze(-1)

        return {"log_risk": log_risk}
