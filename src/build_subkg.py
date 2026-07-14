"""
Build a gene-centric subgraph from PrimeKG.

Strategy:
  - Seed nodes: candidate genes (gene_candidate.csv)
  - Edge types: gene-involving biological relations only
  - Expansion: 1-hop from seed genes
  - Output: subkg.csv (same schema as PrimeKG)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMEKG = ROOT / "ref_KG" / "PrimeKG.csv"
GENE_CANDIDATES = ROOT / "source" / "input_data" / "train" / "gene_candidate.csv"
OUT_DIR = ROOT / "output" / "subkg"

# Edge types that involve gene/protein nodes (biologically meaningful for survival)
GENE_EDGE_TYPES = {
    "protein_protein",
    "bioprocess_protein",
    "pathway_protein",
    "disease_protein",
    "molfunc_protein",
    "cellcomp_protein",
    "phenotype_protein",
}


def load_candidates() -> set[str]:
    df = pd.read_csv(GENE_CANDIDATES)
    col = df.columns[0]
    return set(df[col].dropna().str.strip())


def build_subkg(
    primekg_path: Path = PRIMEKG,
    candidates: set[str] | None = None,
    hops: int = 1,
    out_path: Path | None = None,
) -> pd.DataFrame:
    if candidates is None:
        candidates = load_candidates()

    print(f"Seed genes: {len(candidates)}")
    print(f"Loading PrimeKG from {primekg_path} ...")

    kg = pd.read_csv(primekg_path, low_memory=False)

    # Keep only gene-involving edge types
    kg = kg[kg["relation"].isin(GENE_EDGE_TYPES)].copy()
    print(f"Gene-related edges: {len(kg)}")

    # 1-hop expansion: any edge where x_name OR y_name is a seed gene
    seed = set(candidates)
    for _ in range(hops):
        mask = kg["x_name"].isin(seed) | kg["y_name"].isin(seed)
        subkg = kg[mask].copy()
        # Expand seed with newly discovered gene/protein neighbors
        new_genes = set(subkg[subkg["x_type"] == "gene/protein"]["x_name"]) | \
                    set(subkg[subkg["y_type"] == "gene/protein"]["y_name"])
        seed = seed | new_genes

    print(f"Subgraph edges: {len(subkg)}")
    print(f"Unique relations: {subkg['relation'].value_counts().to_dict()}")

    out_path = out_path or (OUT_DIR / "subkg_primekg.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subkg.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return subkg


if __name__ == "__main__":
    build_subkg()
