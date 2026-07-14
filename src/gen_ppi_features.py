"""Generate PPI burden features for top KGs (without regenerating FMB/Node2Vec)."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from pathlib import Path
from kg_features import (
    load_candidate_genes, build_ppi_adjacency, compute_ppi_burden,
    compute_disease_gene_weight, PPI_EDGE_TYPES, DISEASE_EDGE_TYPES,
)

ROOT = Path(__file__).resolve().parent.parent
SUBKG = ROOT / "output" / "subkg"
PROC = ROOT / "output" / "processed"
FEAT = ROOT / "output" / "kg_features"

KGS = ["primekg", "ogb_biokg"]
genes = load_candidate_genes()


def iter_splits():
    yield "train", PROC / "train_mut.csv"
    for p in sorted(PROC.glob("valid_*_mut.csv")):
        yield p.stem.replace("_mut", ""), p


for kg in KGS:
    print(f"\n{'='*60}")
    print(f"  {kg}: generating PPI + Disease features")
    print(f"{'='*60}")

    subkg_path = SUBKG / f"subkg_{kg}.csv"
    out_dir = FEAT / kg

    # PPI adjacency
    ppi_types = PPI_EDGE_TYPES.get(kg, [])
    adj = build_ppi_adjacency(subkg_path, genes, ppi_types)
    n_ppi = int(adj.sum() / 2)
    print(f"  PPI: {n_ppi} edges, {int((adj.sum(axis=1)>0).sum())} genes connected")

    # Disease weights
    dis_types = DISEASE_EDGE_TYPES.get(kg, [])
    dw = compute_disease_gene_weight(subkg_path, genes, dis_types)
    print(f"  Disease: {int((dw>0).sum())} genes, max={dw.max():.0f}")
    np.save(out_dir / "disease_gene_weights.npy", dw)

    # Generate PPI burden per split
    for prefix, mut_path in iter_splits():
        mut = pd.read_csv(mut_path, index_col=0).values.astype(np.float32)
        mask = pd.read_csv(PROC / f"{prefix}_mask.csv", index_col=0).values.astype(np.float32)

        ppi = compute_ppi_burden(adj, mut, mask)
        ppi_df = pd.DataFrame(
            ppi,
            index=pd.read_csv(mut_path, index_col=0).index,
            columns=[f"ppi_{g}" for g in genes],
        )
        ppi_df.to_csv(out_dir / f"{prefix}_ppi.csv")
        print(f"    {prefix}: PPI shape={ppi.shape}")

    print(f"  Saved to {out_dir}")

print("\nDone.")
