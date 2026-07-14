"""
Build PyG HeteroData graph from PrimeKG subgraph — vectorized.

Node types: gene/protein, pathway, biological_process, disease,
            molecular_function, cellular_component, effect/phenotype
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
SUBKG = ROOT / "output" / "subkg" / "subkg_primekg.csv"

# Map PrimeKG relation → canonical edge type tuple (src_type, rel, dst_type)
EDGE_TYPE_MAP = {
    "protein_protein":    ("gene/protein", "ppi",      "gene/protein"),
    "disease_protein":    ("disease",      "assoc",     "gene/protein"),
    "bioprocess_protein": ("biological_process", "annot", "gene/protein"),
    "pathway_protein":    ("pathway",      "contains",  "gene/protein"),
    "molfunc_protein":    ("molecular_function", "annot", "gene/protein"),
    "cellcomp_protein":   ("cellular_component", "annot", "gene/protein"),
    "phenotype_protein":  ("effect/phenotype", "assoc",  "gene/protein"),
}


def _build_node_maps(kg: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build per-type node index maps using vectorized ops (no iterrows)."""
    node_maps: dict[str, dict[str, int]] = {}

    for name_col, type_col in [("x_name", "x_type"), ("y_name", "y_type")]:
        pairs = kg[[type_col, name_col]].drop_duplicates()
        for ntype, group in pairs.groupby(type_col):
            if ntype not in node_maps:
                node_maps[ntype] = {}
            for name in group[name_col]:
                if name not in node_maps[ntype]:
                    node_maps[ntype][name] = len(node_maps[ntype])

    return node_maps


def build_hetero_graph(subkg_path: Path = SUBKG) -> tuple[HeteroData, dict, dict, dict]:
    """
    Returns:
        data        - PyG HeteroData (no node features, only edge_index)
        node_maps   - {node_type: {name -> local_idx}}
        gene_map    - {gene_name -> local_idx}
        node_counts - {node_type: int} — number of nodes per type
    """
    kg = pd.read_csv(subkg_path, low_memory=False)

    # Build per-type node index maps (vectorized)
    node_maps = _build_node_maps(kg)

    data = HeteroData()
    for ntype, nmap in node_maps.items():
        data[ntype].num_nodes = len(nmap)

    # Build edge indices — vectorized per relation type
    for rel, (src_type, rel_name, dst_type) in EDGE_TYPE_MAP.items():
        rel_edges = kg[kg["relation"] == rel]
        if len(rel_edges) == 0:
            continue

        # Determine src/dst columns based on type matching
        # PrimeKG: x_type/y_type columns tell us which side is which
        x_is_src = rel_edges["x_type"] == src_type
        y_is_src = ~x_is_src

        src_names = pd.Series("", index=rel_edges.index)
        dst_names = pd.Series("", index=rel_edges.index)
        src_names[x_is_src] = rel_edges.loc[x_is_src, "x_name"]
        dst_names[x_is_src] = rel_edges.loc[x_is_src, "y_name"]
        src_names[y_is_src] = rel_edges.loc[y_is_src, "y_name"]
        dst_names[y_is_src] = rel_edges.loc[y_is_src, "x_name"]

        src_map = node_maps[src_type]
        dst_map = node_maps[dst_type]

        # Filter to only nodes present in maps
        valid = src_names.isin(src_map) & dst_names.isin(dst_map)
        src_names = src_names[valid]
        dst_names = dst_names[valid]

        if len(src_names) == 0:
            continue

        src_idx = src_names.map(src_map).values.astype(int)
        dst_idx = dst_names.map(dst_map).values.astype(int)

        data[src_type, rel_name, dst_type].edge_index = torch.tensor(
            [src_idx, dst_idx], dtype=torch.long
        )

    gene_map = node_maps.get("gene/protein", {})
    node_counts = {nt: len(nmap) for nt, nmap in node_maps.items()}
    return data, node_maps, gene_map, node_counts


if __name__ == "__main__":
    data, node_maps, gene_map, node_counts = build_hetero_graph()
    print("Node types and counts:")
    for ntype, nmap in node_maps.items():
        print(f"  {ntype}: {len(nmap)}")
    print("Edge types:")
    for et in data.edge_types:
        print(f"  {et}: {data[et].edge_index.shape[1]} edges")
    print(f"Gene nodes: {len(gene_map)}")
