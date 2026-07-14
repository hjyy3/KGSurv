"""One-off: build base FMB CSVs for newly added holdout cohorts (Hellmann, Jung).

Only builds the 4-group base FMB that exp_wes_cancer_specific loads; PPI/Disease/
Drug multi-node features are computed on-the-fly at experiment time, so they are
not needed here. Verifies column alignment against an existing cohort's FMB so
the model's per-group projection stays consistent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kg_features import (
    FEAT_DIR,
    FMB_EDGE_TYPES,
    MIN_GENES_PER_TERM,
    SUBKG_DIR,
    compute_functional_burden,
    extract_function_gene_map,
    load_candidate_genes,
)

ROOT = Path(__file__).resolve().parents[1]
NEW_COHORTS = ["Hellmann", "Jung"]
KGS = ["drkg", "openbiolink"]


def main():
    wes_genes_path = ROOT / "output" / "processed_wes" / "wes_candidate_genes.csv"
    candidate_genes = load_candidate_genes(wes_genes_path)
    print(f"candidate genes: {len(candidate_genes)}")

    for kg in KGS:
        save_dir = FEAT_DIR / f"{kg}_wes"
        subkg = SUBKG_DIR / f"subkg_{kg}.csv"
        edge_types = FMB_EDGE_TYPES.get(kg, [])
        func_genes = extract_function_gene_map(
            subkg, candidate_genes, edge_types, min_genes=MIN_GENES_PER_TERM
        )
        print(f"\n{kg}: {len(func_genes)} FMB terms")
        existing = pd.read_csv(save_dir / "valid_Hugo_fmb.csv", index_col=0, nrows=1)
        for c in NEW_COHORTS:
            mut_df = pd.read_csv(ROOT / f"output/processed_wes/valid_{c}_wes_mut.csv", index_col=0)
            mask_df = pd.read_csv(ROOT / f"output/processed_wes/valid_{c}_wes_mask.csv", index_col=0)
            mut = mut_df.values.astype(np.float32)
            mask = mask_df.values.astype(np.float32)
            fmb, names = compute_functional_burden(func_genes, candidate_genes, mut, mask)
            if list(names) != list(existing.columns):
                print(f"  [!] {c}: column MISMATCH ({len(names)} vs {len(existing.columns)}) — NOT written")
                continue
            fmb_df = pd.DataFrame(fmb, index=mut_df.index, columns=names)
            fmb_df.to_csv(save_dir / f"valid_{c}_fmb.csv")
            print(f"  [wrote] {kg}/valid_{c}_fmb.csv shape={fmb_df.shape} (cols match Hugo)")


if __name__ == "__main__":
    main()
