"""Data loading and KG structure extraction for interpretable models.

Reads pre-computed FMB features from output/kg_features/ and builds per-group
gene-term connectivity masks from output/subkg/ for interpretable architectures.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from preprocess import COV_COLS  # kept for potential future use

ROOT = Path(__file__).resolve().parent.parent
KG_DIR = ROOT / "output" / "kg_features"
SUBKG_DIR = ROOT / "output" / "subkg"
PROC_DIR = ROOT / "output" / "processed"
PROC_WES_DIR = ROOT / "output" / "processed_wes"
GENE_FILE = ROOT / "source" / "input_data" / "train" / "gene_candidate.csv"
WES_GENE_FILE = ROOT / "output" / "processed_wes" / "wes_candidate_genes.csv"

ALL_KGS = ["primekg", "hetionet", "drkg", "ibkh", "monarch", "ogb_biokg", "openbiolink"]
VALID_COHORTS = [
    "Braun", "CM214_JV101", "Gandara", "Hugo", "Liu", "Mariathasan", "Miao",
    "PUSH", "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae",
]
# WES holdout cohorts used for external evaluation.
WES_HOLDOUT_COHORTS = ["Whijae", "Hugo", "SnyderUC", "Pleasance"]


@dataclass
class KGGroupInfo:
    """Per-KG group structure extracted from subgraph and FMB columns."""
    kg_name: str
    group_names: list[str]                    # edge_type names
    term_names: list[list[str]]               # per-group term names
    gene_term_mask: list[torch.Tensor]        # per-group [n_genes, n_terms_in_group]
    fmb_slices: list[tuple[int, int]]         # per-group [start, end) in FMB columns
    n_genes: int = 463
    n_total_terms: int = 0


def _load_gene_list(path: Path | None = None) -> list[str]:
    """Load candidate gene list. Defaults to 463-panel; pass WES_GENE_FILE for WES."""
    p = Path(path) if path else GENE_FILE
    df = pd.read_csv(p)
    return df.iloc[:, 0].tolist()


def _norm_ws(s: str) -> str:
    """Normalize whitespace (handles multi-line CSV fields)."""
    return " ".join(str(s).split())


def build_kg_group_info(
    kg_name: str,
    feat_dir: Path | None = None,
    wes: bool = False,
    gene_list_path: Path | None = None,
) -> KGGroupInfo:
    """Parse subkg CSV + metadata -> KGGroupInfo with per-group connectivity.

    When wes=True, loads features from KG_DIR/{kg_name}_wes and the WES gene list.
    """
    if feat_dir is None:
        feat_dir = KG_DIR / (f"{kg_name}_wes" if wes else kg_name)
    kg_feat = feat_dir
    with open(kg_feat / "metadata.json") as f:
        meta = json.load(f)

    if wes:
        gene_path = gene_list_path or WES_GENE_FILE
    else:
        gene_path = gene_list_path
    genes = _load_gene_list(gene_path)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    # Parse FMB column headers: "{edge_type}::{term_name}"
    fmb_cols = pd.read_csv(kg_feat / "train_fmb.csv", index_col=0, nrows=1).columns.tolist()
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for col in fmb_cols:
        sep = col.find("::")
        if sep < 0:
            continue
        et = col[:sep]
        tn = _norm_ws(col[sep + 2:])
        groups.setdefault(et, []).append(tn)

    group_names, term_names_list, fmb_slices = [], [], []
    offset = 0
    for et, terms in groups.items():
        group_names.append(et)
        term_names_list.append(terms)
        fmb_slices.append((offset, offset + len(terms)))
        offset += len(terms)

    # Build gene->term connectivity from subgraph edges
    subkg = pd.read_csv(SUBKG_DIR / f"subkg_{kg_name}.csv", low_memory=False)
    subkg["y_name_norm"] = subkg["y_name"].apply(_norm_ws)

    gene_term_masks = []
    for gi, et in enumerate(group_names):
        terms = term_names_list[gi]
        n_terms = len(terms)
        term_to_idx = {t: i for i, t in enumerate(terms)}

        edges = subkg[subkg["relation"] == et]
        mask = torch.zeros(n_genes, n_terms, dtype=torch.float32)

        if len(edges) > 0:
            g_idx = edges["x_name"].map(gene_to_idx)
            t_idx = edges["y_name_norm"].map(term_to_idx)
            valid = g_idx.notna() & t_idx.notna()
            if valid.any():
                mask[g_idx[valid].astype(int).values, t_idx[valid].astype(int).values] = 1.0

        gene_term_masks.append(mask)

    return KGGroupInfo(
        kg_name=kg_name,
        group_names=group_names,
        term_names=term_names_list,
        gene_term_mask=gene_term_masks,
        fmb_slices=fmb_slices,
        n_genes=n_genes,
        n_total_terms=sum(len(t) for t in term_names_list),
    )


def load_split_data(
    kg_name: str,
    split: str = "train",
    feat_dir: Path | None = None,
    wes: bool = False,
) -> dict:
    """Load mut, mask, FMB, clin for one split (panel or WES).

    Args:
        kg_name: one of ALL_KGS
        split: "train" or a validation cohort name
        feat_dir: override KG feature directory
        wes: when True, reads from output/processed_wes/ with *_wes_ suffix
            and KG_DIR/{kg}_wes.
    """
    prefix = "train" if split == "train" else f"valid_{split}"
    if feat_dir is None:
        feat_dir = KG_DIR / (f"{kg_name}_wes" if wes else kg_name)
    kg_feat = feat_dir

    if wes:
        proc = PROC_WES_DIR
        suffix = "_wes_"
    else:
        proc = PROC_DIR
        suffix = "_"

    mut = pd.read_csv(proc / f"{prefix}{suffix}mut.csv", index_col=0)
    mask_df = pd.read_csv(proc / f"{prefix}{suffix}mask.csv", index_col=0)
    clin = pd.read_csv(proc / f"{prefix}{suffix}clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / f"{prefix}_fmb.csv", index_col=0)

    common = mut.index.intersection(clin.index).intersection(fmb.index)

    return {
        "mut": torch.tensor(mut.loc[common].values, dtype=torch.float32),
        "mask": torch.tensor(mask_df.loc[common].values, dtype=torch.float32),
        "time": torch.tensor(clin.loc[common, "OS_MONTHS"].values, dtype=torch.float32),
        "event": torch.tensor(clin.loc[common, "event"].values, dtype=torch.float32),
        "fmb": torch.tensor(fmb.loc[common].values, dtype=torch.float32),
        "sample_ids": common.tolist(),
    }


def load_all_data(
    kg_name: str,
    feat_dir: Path | None = None,
    wes: bool = False,
    holdout_cohorts: list[str] | None = None,
) -> dict:
    """Load training + validation cohorts for a KG.

    Args:
        feat_dir: override directory for FMB/metadata
        wes: when True, reads WES splits/features.
        holdout_cohorts: which cohorts to load as validation. Defaults to
            VALID_COHORTS (panel mode) or WES_HOLDOUT_COHORTS (wes mode).
    """
    kg_info = build_kg_group_info(kg_name, feat_dir=feat_dir, wes=wes)
    train_data = load_split_data(kg_name, "train", feat_dir=feat_dir, wes=wes)
    if holdout_cohorts is None:
        holdout_cohorts = WES_HOLDOUT_COHORTS if wes else VALID_COHORTS
    valid_data = {}
    for cohort in holdout_cohorts:
        try:
            valid_data[cohort] = load_split_data(kg_name, cohort, feat_dir=feat_dir, wes=wes)
        except FileNotFoundError:
            pass
    return {"train": train_data, "valid": valid_data, "kg_info": kg_info}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KG group info inspector")
    parser.add_argument("--kg", default="primekg")
    parser.add_argument("--wes", action="store_true", help="use WES gene set + features")
    args = parser.parse_args()

    info = build_kg_group_info(args.kg, wes=args.wes)
    print(f"KG: {info.kg_name}")
    print(f"Groups: {len(info.group_names)}, Total terms: {info.n_total_terms}")
    for i, name in enumerate(info.group_names):
        n_terms = len(info.term_names[i])
        n_edges = int(info.gene_term_mask[i].sum().item())
        s, e = info.fmb_slices[i]
        print(f"  [{i}] {name}: {n_terms} terms, {n_edges} gene-term edges, FMB[{s}:{e})")
    print(f"Genes: {info.n_genes}")
