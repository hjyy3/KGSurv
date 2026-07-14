"""
Generic PyG HeteroData builder for any KG subgraph in unified CSV format.

Input CSV columns: relation, x_type, x_name, y_type, y_name

Unlike graph_builder.py (PrimeKG-specific with hardcoded EDGE_TYPE_MAP),
this builder dynamically infers node types and edge types from the data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]


def _build_node_maps(kg: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build per-type node index maps."""
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


def build_hetero_graph(
    subkg_path: Path,
    gene_type: str = "auto",
) -> tuple[HeteroData, dict, dict, dict]:
    """Build PyG HeteroData from unified subgraph CSV.

    Args:
        subkg_path: Path to CSV with columns: relation, x_type, x_name, y_type, y_name
        gene_type: Node type name for gene nodes. "auto" detects from data.

    Returns:
        data        - PyG HeteroData
        node_maps   - {node_type: {name -> local_idx}}
        gene_map    - {gene_name -> local_idx}
        node_counts - {node_type: int} — number of nodes per type
    """
    kg = pd.read_csv(subkg_path, low_memory=False)

    # Auto-detect gene node type
    if gene_type == "auto":
        all_types = set(kg["x_type"]) | set(kg["y_type"])
        for candidate in ["gene/protein", "gene", "protein"]:
            if candidate in all_types:
                gene_type = candidate
                break
        else:
            gene_type = "gene"

    node_maps = _build_node_maps(kg)

    data = HeteroData()
    for ntype, nmap in node_maps.items():
        data[ntype].num_nodes = len(nmap)

    # Build edge indices grouped by (src_type, relation, dst_type)
    kg["edge_key"] = list(zip(kg["x_type"], kg["relation"], kg["y_type"]))
    for edge_key, group in kg.groupby("edge_key"):
        src_type, rel, dst_type = edge_key

        src_map = node_maps.get(src_type, {})
        dst_map = node_maps.get(dst_type, {})

        valid = group["x_name"].isin(src_map) & group["y_name"].isin(dst_map)
        group = group[valid]
        if len(group) == 0:
            continue

        src_idx = group["x_name"].map(src_map).values.astype(int)
        dst_idx = group["y_name"].map(dst_map).values.astype(int)

        # Sanitize relation name for PyG (no spaces, special chars)
        safe_rel = rel.lower().replace(" ", "_").replace("-", "_")[:30]
        edge_type = (src_type, safe_rel, dst_type)

        if edge_type in data.edge_types:
            # Merge with existing edges of same type
            existing = data[edge_type].edge_index
            new = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            data[edge_type].edge_index = torch.cat([existing, new], dim=1)
        else:
            data[edge_type].edge_index = torch.tensor(
                [src_idx, dst_idx], dtype=torch.long
            )

    gene_map = node_maps.get(gene_type, {})

    # Ensure pathway type exists (needed for PathwayPooling)
    # Detect pathway-like node type
    pathway_type = None
    for t in node_maps:
        if "pathway" in t.lower():
            pathway_type = t
            break

    node_counts = {nt: len(nmap) for nt, nmap in node_maps.items()}
    return data, node_maps, gene_map, node_counts


def get_pathway_gene_edges(
    data: HeteroData,
    node_maps: dict[str, dict[str, int]],
) -> tuple[torch.Tensor, int, str]:
    """Find pathway->gene edges in the HeteroData.

    Returns:
        pw_gene_edge  - [2, E] tensor (pathway_idx, gene_idx)
        n_pathways    - number of pathway nodes
        pathway_type  - name of pathway node type
    """
    # Find gene-like and pathway-like types
    gene_type = None
    pathway_type = None
    for t in node_maps:
        tl = t.lower()
        if "gene" in tl or "protein" in tl:
            gene_type = t
        if "pathway" in tl:
            pathway_type = t

    if gene_type is None or pathway_type is None:
        # No pathway structure found; create dummy
        n_genes = max(len(m) for m in node_maps.values()) if node_maps else 1
        dummy = torch.zeros(2, 0, dtype=torch.long)
        return dummy, 1, "pathway"

    # Find edge type connecting pathway to gene
    for et in data.edge_types:
        src, rel, dst = et
        if src == pathway_type and dst == gene_type:
            return data[et].edge_index, len(node_maps[pathway_type]), pathway_type
        if src == gene_type and dst == pathway_type:
            # Reverse the edge direction
            ei = data[et].edge_index
            return torch.stack([ei[1], ei[0]]), len(node_maps[pathway_type]), pathway_type

    # No pathway-gene edges found; create dummy
    dummy = torch.zeros(2, 0, dtype=torch.long)
    return dummy, 1, "pathway"


if __name__ == "__main__":
    import sys
    subkg_dir = ROOT / "output" / "subkg"
    for csv_path in sorted(subkg_dir.glob("subkg_*.csv")):
        kg_name = csv_path.stem.replace("subkg_", "")
        print(f"\n{'=' * 50}")
        print(f"Building graph: {kg_name}")
        print("=" * 50)
        try:
            data, node_maps, gene_map, node_counts = build_hetero_graph(csv_path)
            print(f"  Node types: {list(node_maps.keys())}")
            for nt, nm in node_maps.items():
                print(f"    {nt}: {len(nm)} nodes")
            print(f"  Edge types: {len(data.edge_types)}")
            for et in data.edge_types:
                print(f"    {et}: {data[et].edge_index.shape[1]} edges")
            print(f"  Gene nodes: {len(gene_map)}")

            pw_edge, n_pw, pw_type = get_pathway_gene_edges(data, node_maps)
            print(f"  Pathway type: {pw_type}, nodes: {n_pw}, "
                  f"pathway-gene edges: {pw_edge.shape[1]}")
        except Exception as e:
            print(f"  [ERROR] {e}")
