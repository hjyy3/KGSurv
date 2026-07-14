"""Generate PPI features for ALL 7 KGs."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from pathlib import Path
from kg_features import (
    load_candidate_genes, build_ppi_adjacency, compute_ppi_burden,
    compute_disease_gene_weight, PPI_EDGE_TYPES, DISEASE_EDGE_TYPES,
    AVAILABLE_KGS,
)

ROOT = Path(__file__).resolve().parent.parent
SUBKG = ROOT / "output" / "subkg"
PROC = ROOT / "output" / "processed"
FEAT = ROOT / "output" / "kg_features"

genes = load_candidate_genes()


def iter_splits():
    yield "train", PROC / "train_mut.csv"
    for p in sorted(PROC.glob("valid_*_mut.csv")):
        yield p.stem.replace("_mut", ""), p


for kg in AVAILABLE_KGS:
    out_dir = FEAT / kg
    if (out_dir / "train_ppi.csv").exists():
        print(f"  [skip] {kg} PPI already exists")
        continue

    print(f"\n{'='*50}")
    print(f"  {kg}")
    print(f"{'='*50}")

    subkg_path = SUBKG / f"subkg_{kg}.csv"
    ppi_types = PPI_EDGE_TYPES.get(kg, [])
    dis_types = DISEASE_EDGE_TYPES.get(kg, [])

    adj = build_ppi_adjacency(subkg_path, genes, ppi_types)
    n_ppi = int(adj.sum() / 2)
    print(f"  PPI: {n_ppi} edges, {int((adj.sum(axis=1)>0).sum())} genes")

    if dis_types:
        dw = compute_disease_gene_weight(subkg_path, genes, dis_types)
        np.save(out_dir / "disease_gene_weights.npy", dw)
        print(f"  Disease: {int((dw>0).sum())} genes")

    for prefix, mut_path in iter_splits():
        mut = pd.read_csv(mut_path, index_col=0)
        mask = pd.read_csv(PROC / f"{prefix}_mask.csv", index_col=0)
        ppi = compute_ppi_burden(adj, mut.values.astype(np.float32),
                                 mask.values.astype(np.float32))
        pd.DataFrame(ppi, index=mut.index,
                     columns=[f"ppi_{g}" for g in genes]
                     ).to_csv(out_dir / f"{prefix}_ppi.csv")
    print(f"  Done: {kg}")

print("\nAll KGs done.")
