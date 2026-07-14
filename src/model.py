"""
MutaPath-Surv: Interpretable KG-guided survival prediction (GPU-batched).

Architecture (redesigned for efficient GPU batch training):
  Phase 1 — KG Encoding (once per forward, shared across batch):
    Learnable gene/node embeddings → HGT → gene_kg_emb [n_genes, hidden]

  Phase 2 — Patient Scoring (batched, GPU-parallel):
    Per-patient gene features [mut, mask] → combine with KG embeddings
    → gated gene importance → pathway pooling → patient embedding
    → Cox head → log-risk

Interpretability outputs:
  - gene_scores: patient-specific gene importance [batch, n_candidate]
  - pathway_scores: patient-specific pathway activation [batch, n_pathways]
  - global_gene_scores: global gene gate scores [n_genes] (reference)
  - pathway_emb: pathway embeddings [n_pathways, hidden_dim] (analysis)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv


class PathwayPooling(nn.Module):
    """Soft-aggregate gene embeddings into pathway embeddings via KG edges."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        gene_emb: torch.Tensor,           # [n_genes, hidden_dim]
        pathway_gene_edge: torch.Tensor,   # [2, n_edges] (pathway_idx, gene_idx)
        n_pathways: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pathway_emb   [n_pathways, hidden_dim]
            gene_scores   [n_genes] — gate score per gene (interpretability)
        """
        pathway_idx, gene_idx = pathway_gene_edge

        gene_scores = torch.sigmoid(self.gate(gene_emb)).squeeze(-1)  # [n_genes]
        weighted = gene_emb * gene_scores.unsqueeze(-1)               # [n_genes, d]

        pathway_emb = torch.zeros(n_pathways, gene_emb.size(-1),
                                  device=gene_emb.device)
        counts = torch.zeros(n_pathways, device=gene_emb.device)
        pathway_emb.index_add_(0, pathway_idx, weighted[gene_idx])
        counts.index_add_(0, pathway_idx,
                          torch.ones(len(gene_idx), device=gene_emb.device))
        counts = counts.clamp(min=1)
        pathway_emb = pathway_emb / counts.unsqueeze(-1)

        return pathway_emb, gene_scores


class MutaPathSurv(nn.Module):
    """
    KG-guided interpretable survival model — GPU-batched.

    Key design change from v1:
      - HGT runs ONCE per batch using learnable node embeddings (not per-patient)
      - Patient-specific [mut, mask] features are combined with KG embeddings
        via a lightweight fusion MLP → enables true batch parallelism

    Args:
        node_types      list of node type strings in the graph
        edge_types      list of (src, rel, dst) tuples
        hidden_dim      embedding dimension
        num_heads       HGT attention heads
        num_layers      number of HGT layers
        n_genes         number of gene nodes in KG subgraph
        n_pathways      number of pathway nodes
        n_candidate     number of candidate genes (input features)
        dropout         dropout rate
        node_counts     optional dict {node_type: count} for unique embeddings
    """

    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        n_genes: int = 11710,
        n_pathways: int = 975,
        n_candidate: int = 463,
        dropout: float = 0.3,
        node_counts: dict[str, int] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        self.n_candidate = n_candidate

        # Learnable node embeddings for KG encoding (not patient-specific)
        self.gene_emb = nn.Embedding(n_genes, hidden_dim)
        # Bidirectional mapping: ModuleDict keys cannot contain "/"
        self._nt_to_key = {}   # original node type → safe key
        self._key_to_nt = {}   # safe key → original node type
        emb_dict = {}
        for nt in node_types:
            if nt == "gene/protein":
                continue
            safe = nt.replace("/", "__SLASH__")
            self._nt_to_key[nt] = safe
            self._key_to_nt[safe] = nt
            # Use unique per-node embeddings if node_counts provided
            n_emb = node_counts.get(nt, 1) if node_counts else 1
            emb_dict[safe] = nn.Embedding(max(n_emb, 1), hidden_dim)
        self.node_emb = nn.ModuleDict(emb_dict)

        # HGT layers for structural KG encoding
        self.hgt_layers = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, (node_types, edge_types),
                    heads=num_heads)
            for _ in range(num_layers)
        ])
        self.hgt_dropout = nn.Dropout(dropout)

        # Patient-level fusion: combine per-gene [mut, mask] with KG embeddings
        # Input: [hidden_dim (KG emb) + 2 (mut, mask)] → hidden_dim
        self.gene_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pathway_pool = PathwayPooling(hidden_dim)

        # Patient-specific gene importance gate
        self.patient_gate = nn.Linear(hidden_dim, 1)

        # Cox head: pathway_emb_mean + TMB → log-risk
        self.cox_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_kg(
        self,
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        """Run HGT on KG structure — called ONCE per batch.

        Returns:
            h_dict: {node_type: embeddings [n_nodes, hidden_dim]}
        """
        device = self.gene_emb.weight.device

        h_dict: dict[str, torch.Tensor] = {
            "gene/protein": self.gene_emb.weight,  # [n_genes, hidden_dim]
        }
        for safe_key, emb in self.node_emb.items():
            original_nt = self._key_to_nt[safe_key]
            n = 0
            for et_key in edge_index_dict:
                ei = edge_index_dict[et_key]
                if ei.numel() == 0:
                    continue
                if et_key[0] == original_nt:
                    n = max(n, ei[0].max().item() + 1)
                if et_key[2] == original_nt:
                    n = max(n, ei[1].max().item() + 1)
            if n == 0:
                n = 1
            # Use unique per-node embeddings if available, else broadcast
            if emb.num_embeddings >= n:
                h_dict[original_nt] = emb.weight[:n]
            else:
                h_dict[original_nt] = emb.weight.expand(n, -1).contiguous()

        for layer in self.hgt_layers:
            new_h = layer(h_dict, edge_index_dict)
            for nt in h_dict:
                if new_h.get(nt) is None:
                    new_h[nt] = h_dict[nt]
            h_dict = {k: self.hgt_dropout(torch.relu(v))
                      for k, v in new_h.items()}

        return h_dict

    def forward(
        self,
        mut_batch: torch.Tensor,            # [batch, n_candidate]
        mask_batch: torch.Tensor,            # [batch, n_candidate]
        tmb_batch: torch.Tensor,             # [batch]
        edge_index_dict: dict,               # shared graph edges
        pathway_gene_edge: torch.Tensor,     # [2, E] pathway→gene edges
        gene_indices: torch.Tensor,          # [n_candidate] indices into KG gene nodes
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Batched forward pass.

        Args:
            mut_batch:  mutation status [batch, n_candidate]
            mask_batch: panel coverage [batch, n_candidate]
            tmb_batch:  TMB values [batch]
            edge_index_dict: KG edge indices (shared)
            pathway_gene_edge: pathway→gene edges [2, E]
            gene_indices: mapping from candidate gene position → KG gene node index
        """
        batch_size = mut_batch.size(0)
        device = mut_batch.device

        # Phase 1: KG encoding (ONCE per batch)
        h_dict = self.encode_kg(edge_index_dict)
        gene_kg_emb = h_dict["gene/protein"]  # [n_genes, hidden_dim]

        # Phase 2: Per-patient scoring (BATCHED)
        # Extract KG embeddings for candidate genes only
        cand_kg_emb = gene_kg_emb[gene_indices]  # [n_candidate, hidden_dim]
        cand_kg_emb = cand_kg_emb.unsqueeze(0).expand(
            batch_size, -1, -1)                   # [batch, n_candidate, hidden_dim]

        # Stack patient features
        patient_feat = torch.stack([
            mut_batch, mask_batch
        ], dim=-1)  # [batch, n_candidate, 2]

        # Fuse KG embeddings with patient-specific features
        fused = torch.cat([cand_kg_emb, patient_feat], dim=-1)  # [batch, n_cand, hidden+2]
        fused = self.gene_fusion(fused)  # [batch, n_candidate, hidden_dim]

        # Patient-specific gene importance scores
        patient_gene_scores = torch.sigmoid(
            self.patient_gate(fused)
        ).squeeze(-1)  # [batch, n_candidate]

        # Global gene scores (via PathwayPooling for pathway embedding)
        fused_mean = fused.mean(dim=0)  # [n_candidate, hidden_dim]
        gene_emb_full = torch.zeros(self.n_genes, self.hidden_dim, device=device)
        gene_emb_full[gene_indices] = fused_mean

        pathway_emb, global_gene_scores = self.pathway_pool(
            gene_emb_full, pathway_gene_edge, self.n_pathways
        )  # pathway_emb: [n_pathways, hidden_dim], global_gene_scores: [n_genes]

        # Per-patient embedding via patient-specific gene importance
        patient_gene_weighted = fused * patient_gene_scores.unsqueeze(-1)
        # [batch, n_candidate, hidden_dim]
        patient_emb = patient_gene_weighted.mean(dim=1)  # [batch, hidden_dim]

        # Patient-specific pathway scores: dot product with pathway embeddings
        # pathway_emb: [n_pathways, hidden_dim], patient_emb: [batch, hidden_dim]
        patient_pathway_scores = torch.matmul(
            patient_emb, pathway_emb.T
        )  # [batch, n_pathways]

        tmb_feat = tmb_batch.unsqueeze(-1)  # [batch, 1]
        patient_feat_final = torch.cat([patient_emb, tmb_feat], dim=-1)

        log_risk = self.cox_head(patient_feat_final).squeeze(-1)  # [batch]

        return {
            "log_risk": log_risk,
            "gene_scores": patient_gene_scores,       # [batch, n_candidate]
            "pathway_scores": patient_pathway_scores,  # [batch, n_pathways]
            "global_gene_scores": global_gene_scores,  # [n_genes]
            "pathway_emb": pathway_emb,                # [n_pathways, hidden_dim]
        }
