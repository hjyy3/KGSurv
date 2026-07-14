"""
Targeted PPI feature regeneration after gene-alias fix.

Only rebuilds {prefix}_ppi.csv files for KGs that use current HGNC symbols
(DRKG, OpenBioLink, Monarch, ibkh, ogb_biokg).  FMB/kgemb files are
unchanged — run this script, then retrain experiments that used PPI features.

Usage:
    python src/regen_ppi_features.py
    python src/regen_ppi_features.py --kg drkg openbiolink
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_features import (
    load_candidate_genes, build_ppi_adjacency, compute_ppi_burden,
    PPI_EDGE_TYPES, GENE_ALIAS, GENE_ALIAS_REV,
    SUBKG_DIR, FEAT_DIR, PROCESSED,
)

# Hetionet uses old symbols → already correct; PrimeKG uses index-based IDs
# → handled separately.  These 5 KGs need rebuilding.
AFFECTED_KGS = ["drkg", "openbiolink", "monarch", "ibkh", "ogb_biokg"]


def iter_splits():
    train_path = PROCESSED / "train_mut.csv"
    if train_path.exists():
        yield "train", train_path
    for p in sorted(PROCESSED.glob("valid_*_mut.csv")):
        yield p.stem.replace("_mut", ""), p


def regen_ppi_for_kg(kg: str):
    subkg_path = SUBKG_DIR / f"subkg_{kg}.csv"
    feat_dir = FEAT_DIR / kg
    if not feat_dir.exists():
        print(f"  {kg}: feat dir missing, skipping")
        return

    ppi_types = PPI_EDGE_TYPES.get(kg, [])
    if not ppi_types:
        print(f"  {kg}: no PPI edge types defined, skipping")
        return

    candidate_genes = load_candidate_genes()
    print(f"  Building PPI adjacency for {kg}...")
    adj = build_ppi_adjacency(subkg_path, candidate_genes, ppi_types)
    n_edges = int(adj.sum() / 2)
    n_genes_with_ppi = int((adj.sum(axis=1) > 0).sum())
    print(f"  {kg}: {n_edges} edges, {n_genes_with_ppi}/{len(candidate_genes)} genes")

    # Check alias impact
    cand_set = set(candidate_genes)
    for old, new in GENE_ALIAS.items():
        if old in cand_set:
            idx = candidate_genes.index(old)
            deg = int(adj[idx].sum())
            print(f"    {old}({new}): degree={deg}")

    for prefix, mut_path in iter_splits():
        mask_path = PROCESSED / f"{prefix}_mask.csv"
        mut_df = pd.read_csv(mut_path, index_col=0)
        mask_df = pd.read_csv(mask_path, index_col=0)
        mut = mut_df.values.astype(np.float32)
        mask = mask_df.values.astype(np.float32)

        ppi_burd = compute_ppi_burden(adj, mut, mask)
        ppi_df = pd.DataFrame(
            ppi_burd, index=mut_df.index,
            columns=[f"ppi_{g}" for g in candidate_genes],
        )
        out_path = feat_dir / f"{prefix}_ppi.csv"
        ppi_df.to_csv(out_path)

    print(f"  {kg}: PPI files written ({len(list(feat_dir.glob('*_ppi.csv')))} splits)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", nargs="+", default=AFFECTED_KGS)
    args = parser.parse_args()

    for kg in args.kg:
        print(f"\n=== Regenerating PPI: {kg} ===")
        regen_ppi_for_kg(kg)

    print("\nDone. Next: retrain experiments that use PPI features.")
    print("Affected configs: any experiment with node_types including 'ppi'")
