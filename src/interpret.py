"""Patient-level interpretability extraction for MutaPathSurv.

Extracts per-patient gene importance and pathway activation scores
from a trained model, enabling downstream visualization and analysis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "output" / "processed"


def extract_patient_attributions(
    model,
    mut: np.ndarray,
    mask: np.ndarray,
    tmb: np.ndarray,
    edge_index_dict: dict,
    pw_gene: torch.Tensor,
    gene_indices: torch.Tensor,
    candidate_genes: list[str],
    pathway_names: list[str] | None = None,
    patient_ids: list[str] | None = None,
    device: torch.device | None = None,
    batch_size: int = 128,
) -> dict[str, pd.DataFrame]:
    """Extract patient-level gene and pathway attributions.

    Returns:
        dict with keys:
          'gene_scores': DataFrame [n_patients, n_candidate_genes]
          'pathway_scores': DataFrame [n_patients, n_pathways]
          'risks': Series of log-risk scores
          'global_gene_scores': Series of global gene importance
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n = len(mut)

    all_gene_scores = []
    all_pathway_scores = []
    all_risks = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mut_b = torch.tensor(mut[start:end], device=device)
            mask_b = torch.tensor(mask[start:end], device=device)
            tmb_b = torch.tensor(tmb[start:end], device=device)

            out = model(mut_b, mask_b, tmb_b, edge_index_dict,
                        pw_gene, gene_indices)

            all_gene_scores.append(out["gene_scores"].cpu().numpy())
            all_pathway_scores.append(out["pathway_scores"].cpu().numpy())
            all_risks.append(out["log_risk"].cpu().numpy())

    gene_scores = np.concatenate(all_gene_scores, axis=0)
    pathway_scores = np.concatenate(all_pathway_scores, axis=0)
    risks = np.concatenate(all_risks)

    # Extract global gene scores from last batch
    with torch.no_grad():
        mut_b = torch.tensor(mut[:min(batch_size, n)], device=device)
        mask_b = torch.tensor(mask[:min(batch_size, n)], device=device)
        tmb_b = torch.tensor(tmb[:min(batch_size, n)], device=device)
        out = model(mut_b, mask_b, tmb_b, edge_index_dict,
                    pw_gene, gene_indices)
        global_scores = out["global_gene_scores"].cpu().numpy()

    idx = patient_ids if patient_ids else [f"P{i}" for i in range(n)]

    gene_df = pd.DataFrame(gene_scores, index=idx, columns=candidate_genes)

    pw_cols = pathway_names if pathway_names else [
        f"PW{i}" for i in range(pathway_scores.shape[1])]
    pathway_df = pd.DataFrame(pathway_scores, index=idx, columns=pw_cols)

    risk_series = pd.Series(risks, index=idx, name="log_risk")
    global_series = pd.Series(global_scores, name="global_gene_score")

    return {
        "gene_scores": gene_df,
        "pathway_scores": pathway_df,
        "risks": risk_series,
        "global_gene_scores": global_series,
    }


def top_genes_by_risk_group(
    gene_scores: pd.DataFrame,
    risks: pd.Series,
    top_n: int = 30,
) -> dict[str, pd.DataFrame]:
    """Compare gene importance between high/low risk groups.

    Returns:
        dict with 'high_risk', 'low_risk', 'delta' DataFrames
    """
    median_risk = risks.median()
    high_mask = risks >= median_risk
    low_mask = ~high_mask

    high_mean = gene_scores.loc[high_mask].mean()
    low_mean = gene_scores.loc[low_mask].mean()
    delta = high_mean - low_mean

    # Statistical test per gene (Welch's t-test)
    from scipy import stats
    p_values = []
    for gene in gene_scores.columns:
        h = gene_scores.loc[high_mask, gene].values
        l = gene_scores.loc[low_mask, gene].values
        if len(h) < 2 or len(l) < 2:
            p_values.append(1.0)
        else:
            _, p = stats.ttest_ind(h, l, equal_var=False)
            p_values.append(p)

    result = pd.DataFrame({
        "gene": gene_scores.columns,
        "high_risk_mean": high_mean.values,
        "low_risk_mean": low_mean.values,
        "delta": delta.values,
        "abs_delta": np.abs(delta.values),
        "p_value": p_values,
        "neg_log10_p": -np.log10(np.clip(p_values, 1e-300, 1.0)),
    }).sort_values("abs_delta", ascending=False)

    return {
        "ranking": result,
        "top_high": result.head(top_n),
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
    }


def top_pathways_by_risk_group(
    pathway_scores: pd.DataFrame,
    risks: pd.Series,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """Compare pathway activation between high/low risk groups.

    Returns:
        dict with 'ranking' DataFrame sorted by absolute delta
    """
    median_risk = risks.median()
    high_mask = risks >= median_risk
    low_mask = ~high_mask

    high_mean = pathway_scores.loc[high_mask].mean()
    low_mean = pathway_scores.loc[low_mask].mean()
    delta = high_mean - low_mean

    from scipy import stats
    p_values = []
    for pw in pathway_scores.columns:
        h = pathway_scores.loc[high_mask, pw].values
        l = pathway_scores.loc[low_mask, pw].values
        if len(h) < 2 or len(l) < 2:
            p_values.append(1.0)
        else:
            _, p = stats.ttest_ind(h, l, equal_var=False)
            p_values.append(p)

    result = pd.DataFrame({
        "pathway": pathway_scores.columns,
        "high_risk_mean": high_mean.values,
        "low_risk_mean": low_mean.values,
        "delta": delta.values,
        "abs_delta": np.abs(delta.values),
        "p_value": p_values,
        "neg_log10_p": -np.log10(np.clip(p_values, 1e-300, 1.0)),
    }).sort_values("abs_delta", ascending=False)

    return {
        "ranking": result,
        "top": result.head(top_n),
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
    }
