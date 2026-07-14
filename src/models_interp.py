"""Interpretable survival models with heterogeneous KG structure.

Three architectures that preserve KG edge-type heterogeneity and provide
multi-level importance attribution (gene / term / group).

All models:
  - Pure PyTorch (no torch-scatter / torch-cluster)
  - Target < 50K effective parameters
  - Return dict with: log_risk, gene_importance, term_importance, group_importance
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_interp import KGGroupInfo


def _make_risk_head(d_in: int, head_type: str = "linear",
                    head_hidden: int = 16, head_dropout: float = 0.2) -> nn.Module:
    """Build risk head.

    head_type='linear': Linear(d_in, 1) — original behavior.
    head_type='mlp':    Linear(d_in, h) → ReLU → Dropout → Linear(h, 1).
    """
    if head_type == "linear":
        return nn.Linear(d_in, 1)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(d_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )
    raise ValueError(f"Unknown head_type: {head_type}")


# ---------------------------------------------------------------------------
# Model A: SparsePathNet  (P-NET / BINN inspired)
# ---------------------------------------------------------------------------

class SparsePathNet(nn.Module):
    """Sparse biologically-structured network following KG hierarchy.

    gene(463) -> term(n_terms, sparse) -> group(n_groups, sparse) -> risk(1)
    Sparsity masks come from KG subgraph connectivity.
    """

    def __init__(self, kg_info: KGGroupInfo, dropout: float = 0.1,
                 head_type: str = "linear", head_hidden: int = 16,
                 head_dropout: float = 0.2, **_kw):
        super().__init__()
        n_genes = kg_info.n_genes
        n_total_terms = kg_info.n_total_terms
        n_groups = len(kg_info.group_names)

        # Layer 1: gene -> term (sparse via KG connectivity)
        # Input: [mut*mask, mask] = 2*n_genes to distinguish wild-type from unknown
        full_mask = torch.cat(kg_info.gene_term_mask, dim=1)     # [463, n_total_terms]
        # Duplicate mask rows for the concatenated input [mut*mask | mask]
        full_mask_2x = torch.cat([full_mask, full_mask], dim=0)  # [2*463, n_total_terms]
        self.register_buffer("gene_term_mask", full_mask_2x)
        self.gene_term_weight = nn.Parameter(torch.randn(2 * n_genes, n_total_terms) * 0.01)
        self.gene_term_bias = nn.Parameter(torch.zeros(n_total_terms))

        # Layer 2: term -> group (each term belongs to exactly one group)
        tg_mask = torch.zeros(n_total_terms, n_groups)
        for gi, terms in enumerate(kg_info.term_names):
            s, e = kg_info.fmb_slices[gi]
            tg_mask[s:e, gi] = 1.0
        self.register_buffer("term_group_mask", tg_mask)
        self.term_group_weight = nn.Parameter(torch.randn(n_total_terms, n_groups) * 0.01)
        self.term_group_bias = nn.Parameter(torch.zeros(n_groups))

        # Layer 3: group -> risk
        self.risk_head = _make_risk_head(n_groups, head_type, head_hidden, head_dropout)
        self.dropout = nn.Dropout(dropout)

        self._n_groups = n_groups
        # Store original n_genes for importance attribution
        self._n_genes = n_genes

    def forward(self, mut: torch.Tensor, mask: torch.Tensor,
                fmb: torch.Tensor | None = None,
                **_kw) -> dict:
        # Concatenate [mut*mask, mask] to distinguish wild-type(0,1) from unknown(0,0)
        x = torch.cat([mut * mask, mask], dim=1)                 # [B, 2*463]

        # Gene -> term (masked sparse)
        w1 = self.gene_term_weight * self.gene_term_mask
        term_h = F.relu(x @ w1 + self.gene_term_bias)            # [B, T]
        term_h = self.dropout(term_h)

        # Term -> group (masked sparse)
        w2 = self.term_group_weight * self.term_group_mask
        group_h = F.relu(term_h @ w2 + self.term_group_bias)     # [B, G]

        # Cox head
        log_risk = self.risk_head(group_h).squeeze(-1)

        # --- Importance attribution ---
        # Use only the first n_genes rows (mut*mask part) for gene importance
        w1_abs = (self.gene_term_weight[:self._n_genes] * self.gene_term_mask[:self._n_genes]).abs()
        gene_imp = (mut * mask) * w1_abs.sum(dim=1)              # [B, 463]
        term_imp = term_h                                         # [B, T]
        group_imp = group_h                                       # [B, G]

        return {
            "log_risk": log_risk,
            "gene_importance": gene_imp,
            "term_importance": term_imp,
            "group_importance": group_imp,
        }


# ---------------------------------------------------------------------------
# Model B: PathAttnSurv  (SurvPath inspired)
# ---------------------------------------------------------------------------

class _AttnBlock(nn.Module):
    """Pre-norm transformer block with exposed attention weights."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d),
        )

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=True)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


class PathAttnSurv(nn.Module):
    """Pathway-tokenized transformer survival model.

    FMB features grouped by edge_type -> per-group tokens + CLS
    -> self-attention -> CLS output -> risk.
    """

    def __init__(self, kg_info: KGGroupInfo, hidden_dim: int = 32,
                 n_heads: int = 2, n_layers: int = 2, dropout: float = 0.1,
                 head_type: str = "linear", head_hidden: int = 16,
                 head_dropout: float = 0.2, n_extra_risk: int = 0,
                 use_tmb: bool = True, **_kw):
        super().__init__()
        n_groups = len(kg_info.group_names)

        # Auto-reduce hidden_dim to stay under per-group projection budget.
        # Panel mode (n_total_terms ~600) keeps hidden_dim=32 fully.
        # WES mode (n_total_terms ~4590) needs a budget that keeps hidden_dim
        # fully usable (~150K params per group projection, total ~250K params).
        cap = max(35_000, hidden_dim * kg_info.n_total_terms)  # never reduce hidden_dim
        max_proj = kg_info.n_total_terms * hidden_dim
        if max_proj > cap:
            hidden_dim = max(8, cap // max(kg_info.n_total_terms, 1))
        hidden_dim = max(n_heads, (hidden_dim // n_heads) * n_heads)
        d = hidden_dim

        # Per-group linear projection: n_terms_in_group -> d
        self.group_proj = nn.ModuleList()
        for gi in range(n_groups):
            n_t = len(kg_info.term_names[gi])
            self.group_proj.append(nn.Linear(n_t, d))

        self.fmb_slices = kg_info.fmb_slices
        self.n_groups = n_groups
        self.d = d
        self.n_extra_risk = n_extra_risk
        self.use_tmb = use_tmb

        # CLS token + type embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.type_emb = nn.Embedding(n_groups + 1, d)

        # Transformer blocks
        self.blocks = nn.ModuleList([_AttnBlock(d, n_heads, dropout) for _ in range(n_layers)])

        # Risk head: CLS + (optional TMB) + (optional) extra side features -> 1
        self.risk_head = _make_risk_head(
            d + (1 if use_tmb else 0) + n_extra_risk, head_type, head_hidden, head_dropout)

        # Buffer for gene->term backprop of importance
        full_mask = torch.cat(kg_info.gene_term_mask, dim=1)     # [463, T]
        self.register_buffer("gene_term_full", full_mask)

    def forward(self, mut: torch.Tensor, mask: torch.Tensor,
                fmb: torch.Tensor,
                extra_risk: torch.Tensor | None = None,
                **_kw) -> dict:
        B = mut.shape[0]
        dev = mut.device
        tmb = (mut * mask).sum(dim=1, keepdim=True)              # [B, 1]

        # Build group tokens
        tokens = []
        for gi in range(self.n_groups):
            s, e = self.fmb_slices[gi]
            tok = self.group_proj[gi](fmb[:, s:e])               # [B, d]
            tok = tok + self.type_emb(torch.tensor(gi, device=dev))
            tokens.append(tok)
        tokens = torch.stack(tokens, dim=1)                       # [B, G, d]

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1) + self.type_emb(
            torch.tensor(self.n_groups, device=dev))
        seq = torch.cat([cls, tokens], dim=1)                     # [B, G+1, d]

        # Transformer
        last_attn = None
        for blk in self.blocks:
            seq, last_attn = blk(seq)

        cls_out = seq[:, 0, :]                                    # [B, d]
        head_in = [cls_out]
        if self.use_tmb:
            head_in.append(tmb)
        if self.n_extra_risk > 0:
            if extra_risk is None:
                extra_risk = torch.zeros(B, self.n_extra_risk, device=dev)
            head_in.append(extra_risk)
        log_risk = self.risk_head(torch.cat(head_in, dim=1)).squeeze(-1)

        # --- Importance attribution ---
        # Group: CLS->group attention from last layer
        group_imp = last_attn[:, 0, 1:]                           # [B, G]

        # Term: FMB activation weighted by group importance
        term_parts = []
        for gi in range(self.n_groups):
            s, e = self.fmb_slices[gi]
            term_parts.append(fmb[:, s:e] * group_imp[:, gi:gi + 1])
        term_imp = torch.cat(term_parts, dim=1)                   # [B, T]

        # Gene: backpropagate term importance through KG mask
        gene_imp = term_imp @ self.gene_term_full.t()             # [B, 463]
        gene_imp = gene_imp * (mut * mask)

        return {
            "log_risk": log_risk,
            "gene_importance": gene_imp,
            "term_importance": term_imp,
            "group_importance": group_imp,
        }


# ---------------------------------------------------------------------------
# Model C: BipartiteAttnSurv  (Heterogeneous bipartite attention)
# ---------------------------------------------------------------------------

class BipartiteAttnSurv(nn.Module):
    """Two-level attention: per-group gene->term bipartite + cross-group.

    Gene features attend to connected terms per edge-type group,
    then groups are fused via cross-group attention.
    """

    def __init__(self, kg_info: KGGroupInfo, hidden_dim: int = 32,
                 dropout: float = 0.1, head_type: str = "linear",
                 head_hidden: int = 16, head_dropout: float = 0.2, **_kw):
        super().__init__()
        n_groups = len(kg_info.group_names)
        d = hidden_dim

        # Gene projection: [mut, mask] -> d
        self.gene_proj = nn.Linear(2, d)

        # Per-group: learnable group key + value projection
        self.group_keys = nn.ParameterList()
        self.group_val_proj = nn.ModuleList()
        for gi in range(n_groups):
            self.group_keys.append(nn.Parameter(torch.randn(d) * 0.02))
            self.group_val_proj.append(nn.Linear(d, d, bias=False))
            self.register_buffer(f"gtm_{gi}", kg_info.gene_term_mask[gi])

        self.n_groups = n_groups
        self.d = d
        self._n_genes = kg_info.n_genes

        # Cross-group attention (query-based readout)
        self.cross_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cross_k = nn.Linear(d, d, bias=False)
        self.cross_v = nn.Linear(d, d, bias=False)

        # Risk head
        self.risk_head = _make_risk_head(d + 1, head_type, head_hidden, head_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fmb_slices = kg_info.fmb_slices

    def forward(self, mut: torch.Tensor, mask: torch.Tensor,
                fmb: torch.Tensor,
                **_kw) -> dict:
        B = mut.shape[0]
        dev = mut.device
        tmb = (mut * mask).sum(dim=1, keepdim=True)

        gene_in = torch.stack([mut, mask], dim=-1)                # [B, 463, 2]
        gene_feat = self.gene_proj(gene_in)                       # [B, 463, d]

        group_embs, gene_attns, term_imps = [], [], []
        for gi in range(self.n_groups):
            gtm = getattr(self, f"gtm_{gi}")                     # [463, n_t]
            key = self.group_keys[gi]                             # [d]

            # Gene attention scores
            scores = (gene_feat * key).sum(-1)                    # [B, 463]
            has_terms = gtm.sum(dim=1) > 0                        # [463]
            scores = scores.masked_fill(~has_terms.unsqueeze(0), -1e9)
            gene_attn = F.softmax(scores, dim=1)                  # [B, 463]
            gene_attns.append(gene_attn)

            # Weighted aggregation
            val = self.group_val_proj[gi](gene_feat)              # [B, 463, d]
            g_emb = (gene_attn.unsqueeze(-1) * val).sum(dim=1)   # [B, d]
            group_embs.append(g_emb)

            # Term importance: propagate gene attention through mask
            t_imp = gene_attn @ gtm                               # [B, n_t]
            term_imps.append(t_imp)

        group_stack = torch.stack(group_embs, dim=1)              # [B, G, d]
        group_stack = self.dropout(group_stack)

        # Cross-group attention
        Q = self.cross_query.expand(B, -1, -1)
        K = self.cross_k(group_stack)
        V = self.cross_v(group_stack)
        cross_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d)
        cross_attn = F.softmax(cross_scores, dim=-1)             # [B, 1, G]
        patient_emb = (cross_attn @ V).squeeze(1)                # [B, d]

        risk_in = [patient_emb, tmb]
        log_risk = self.risk_head(torch.cat(risk_in, dim=1)).squeeze(-1)

        # --- Importance attribution ---
        group_imp = cross_attn.squeeze(1)                         # [B, G]

        # Gene: weighted sum of per-group gene attention
        gene_imp = torch.zeros(B, self._n_genes, device=dev)
        for gi in range(self.n_groups):
            gene_imp = gene_imp + gene_attns[gi] * group_imp[:, gi:gi + 1]
        gene_imp = gene_imp * (mut * mask)

        # Term: concatenate per-group, weighted by group importance
        term_imp = torch.cat(
            [t * group_imp[:, gi:gi + 1] for gi, t in enumerate(term_imps)],
            dim=1,
        )

        return {
            "log_risk": log_risk,
            "gene_importance": gene_imp,
            "term_importance": term_imp,
            "group_importance": group_imp,
        }


# ---------------------------------------------------------------------------
# Factory + utilities
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "sparse_path": SparsePathNet,
    "path_attn": PathAttnSurv,
    "bipartite_attn": BipartiteAttnSurv,
}

ALL_MODELS = list(MODEL_REGISTRY.keys())


def create_model(name: str, kg_info: KGGroupInfo, **kwargs) -> nn.Module:
    """Instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {ALL_MODELS}")
    return MODEL_REGISTRY[name](kg_info, **kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, effective_params).

    For SparsePathNet, effective = only non-zero masked weights.
    """
    total = sum(p.numel() for p in model.parameters())
    effective = total

    if isinstance(model, SparsePathNet):
        # Subtract zeroed-out weights
        gt_full = model.gene_term_weight.numel()
        gt_eff = int(model.gene_term_mask.sum().item())
        effective -= (gt_full - gt_eff)

        tg_full = model.term_group_weight.numel()
        tg_eff = int(model.term_group_mask.sum().item())
        effective -= (tg_full - tg_eff)

    return total, effective


if __name__ == "__main__":
    from data_interp import build_kg_group_info, ALL_KGS

    for kg in ALL_KGS:
        info = build_kg_group_info(kg)
        print(f"\n{'='*50}")
        print(f"KG: {kg} ({info.n_total_terms} terms, {len(info.group_names)} groups)")
        for name in ALL_MODELS:
            m = create_model(name, info)
            tot, eff = count_parameters(m)
            # Quick forward test
            B = 4
            out = m(torch.randn(B, info.n_genes), torch.ones(B, info.n_genes),
                    torch.randn(B, info.n_total_terms))
            ok = all(k in out for k in ["log_risk", "gene_importance",
                                        "term_importance", "group_importance"])
            print(f"  {name:20s}: {tot:>8,} total, {eff:>8,} effective | forward OK={ok}")
