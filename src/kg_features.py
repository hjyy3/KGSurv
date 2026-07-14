"""Pre-compute KG-derived features for lightweight survival model.

Two feature families:
  1. Functional Mutation Burden (FMB): pathway/GO-level mutation density
  2. Node2Vec gene embeddings: frozen KG structural embeddings

Usage:
    python src/kg_features.py --kg primekg
    python src/kg_features.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "output" / "processed"
SUBKG_DIR = ROOT / "output" / "subkg"
FEAT_DIR = ROOT / "output" / "kg_features"

AVAILABLE_KGS = [
    "primekg", "hetionet", "drkg", "ibkh",
    "monarch", "ogb_biokg", "openbiolink",
]

# Edge types representing function/pathway → gene annotations per KG.
FMB_EDGE_TYPES: dict[str, list[str]] = {
    "primekg": [
        "pathway_protein", "bioprocess_protein",
        "molfunc_protein", "cellcomp_protein",
    ],
    "drkg": [
        "Gene:Pathway", "Gene:Biological Process",
        "Gene:Molecular Function", "Gene:Cellular Component",
    ],
    "hetionet": ["participates"],
    "ibkh": ["G_Pw"],
    "monarch": [
        "actively_involved_in", "participates_in", "enables",
        "is_active_in", "acts_upstream_of_or_within", "located_in",
    ],
    "ogb_biokg": ["protein-function"],
    "openbiolink": ["gene_go", "gene_pathway"],
}

# Edge types for PPI (gene-gene interactions)
PPI_EDGE_TYPES: dict[str, list[str]] = {
    "primekg": ["protein_protein"],
    "hetionet": ["interacts"],
    "drkg": ["Gene:Gene", "HumGenHumGen:Gene:Gene"],
    "ibkh": ["G_G"],
    "monarch": ["interacts_with"],
    "ogb_biokg": ["protein-protein_binding", "protein-protein_catalysis",
                  "protein-protein_activation", "protein-protein_inhibition"],
    "openbiolink": ["gene_gene", "gene_binding_gene"],
}

# Edge types for Disease-Gene associations
DISEASE_EDGE_TYPES: dict[str, list[str]] = {
    "primekg": ["disease_protein"],
    "hetionet": ["associates"],
    "drkg": ["Gene:Disease", "Disease:Gene"],
    "ibkh": ["Di_G"],
    "monarch": ["gene_associated_with_condition"],
    "ogb_biokg": ["disease-protein"],
    "openbiolink": ["gene_dis"],
}

MIN_GENES_PER_TERM = 3  # default; overridable via CLI --min_genes

# ---------------------------------------------------------------------------
# Gene alias mapping  (deprecated HGNC symbol → current official symbol)
# ---------------------------------------------------------------------------
# Our mutation data and gene_candidate.csv use MSK panel gene names, which may
# include deprecated symbols.  Several KGs (DRKG, OpenBioLink, ibkh) have
# already migrated to current HGNC symbols, so without this mapping those
# genes silently drop out of every FMB pathway term.
#
# Key: symbol used in our mutation data (old / deprecated)
# Val: symbol used in KG databases (current HGNC)
GENE_ALIAS: dict[str, str] = {
    "FAM46C":  "TENT5C",   # polyA RNA polymerase
    "PAK7":    "PAK5",     # p21-activated kinase 5
    "SETD8":   "KMT5A",    # histone methyltransferase
    "FAM175A": "ABRAXAS1", # BRCA1-BRCT domain scaffold
    "FAM175B": "ABRAXAS2", # BRISC complex subunit
    "PARK2":   "PRKN",     # Parkin RBR E3 ubiquitin ligase
    "WHSC1":   "NSD2",     # nuclear receptor-binding SET domain 2
    "WHSC1L1": "NSD3",     # nuclear receptor-binding SET domain 3
    "MRE11A":  "MRE11",    # meiotic recombination 11 homolog
}
# Reverse map: new symbol → old symbol (for normalising KG hits back to
# our mutation-data column names)
GENE_ALIAS_REV: dict[str, str] = {v: k for k, v in GENE_ALIAS.items()}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_candidate_genes(path: Path | None = None) -> list[str]:
    """Load candidate gene list. Defaults to the 463 MSK-IMPACT panel; pass
    ``path`` (e.g. output/processed_wes/wes_candidate_genes.csv) for WES mode.
    """
    p = Path(path) if path else (
        ROOT / "source" / "input_data" / "train" / "gene_candidate.csv"
    )
    df = pd.read_csv(p)
    return df.iloc[:, 0].dropna().str.strip().tolist()


PROCESSED_WES = ROOT / "output" / "processed_wes"


def _iter_splits(wes: bool = False):
    """Yield (split_prefix, mut_csv_path).

    panel mode reads from output/processed/{train,valid_*}_mut.csv.
    WES mode reads from output/processed_wes/{train,valid_*}_wes_mut.csv.
    """
    if wes:
        base = PROCESSED_WES
        train_path = base / "train_wes_mut.csv"
        if train_path.exists():
            yield "train", train_path
        for p in sorted(base.glob("valid_*_wes_mut.csv")):
            prefix = p.stem.replace("_wes_mut", "")
            yield prefix, p
        return
    train_path = PROCESSED / "train_mut.csv"
    if train_path.exists():
        yield "train", train_path
    for p in sorted(PROCESSED.glob("valid_*_mut.csv")):
        prefix = p.stem.replace("_mut", "")
        yield prefix, p


# ---------------------------------------------------------------------------
# FMB: Functional Mutation Burden
# ---------------------------------------------------------------------------

# HGNC re-naming applied symmetrically by _normalise_gene so that whether the
# KG uses the old or the new symbol it matches the candidate set (panel uses
# pre-2020 names, WES uses current HGNC).
HGNC_FORWARD: dict[str, str] = {
    **{f"MARCH{i}": f"MARCHF{i}" for i in range(1, 13)},
    **{f"SEPT{i}": f"SEPTIN{i}" for i in range(1, 16)},
    "DEC1": "BHLHE40",
}
HGNC_REVERSE: dict[str, str] = {v: k for k, v in HGNC_FORWARD.items()}


def _normalise_gene(name: str, cand_set: set[str]) -> str | None:
    """Return the candidate-gene symbol for *name*, handling aliases bidirectionally.

    Tries (in order): raw, panel alias forward (FAM46C->TENT5C), panel alias
    reverse, HGNC migration forward (MARCH1->MARCHF1), HGNC migration reverse.
    Returns None if no path matches a candidate.
    """
    if name in cand_set:
        return name
    for cand in (
        GENE_ALIAS.get(name),
        GENE_ALIAS_REV.get(name),
        HGNC_FORWARD.get(name),
        HGNC_REVERSE.get(name),
    ):
        if cand and cand in cand_set:
            return cand
    return None


def extract_function_gene_map(
    subkg_path: Path,
    candidate_genes: list[str],
    edge_types: list[str],
    min_genes: int | None = None,
) -> dict[str, set[str]]:
    """Extract function-term → gene-set mappings from KG subgraph.

    For each FMB-eligible edge, identify which end is a candidate gene and
    which is the functional annotation term.  Returns a dict keyed by
    ``"<relation>::<term_name>"`` → set of gene names.

    Gene aliases are resolved so that KGs using current HGNC symbols
    (e.g. TENT5C) are matched to our mutation-data columns which may still
    use deprecated names (e.g. FAM46C).
    """
    cand_set = set(candidate_genes)
    # Expanded set includes both old and new symbols for alias lookup
    cand_set_expanded = cand_set | set(GENE_ALIAS.values())
    df = pd.read_csv(subkg_path)
    df = df[df["relation"].isin(edge_types)]

    func_genes: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        x_name, y_name = str(row["x_name"]), str(row["y_name"])
        x_gene = _normalise_gene(x_name, cand_set)
        y_gene = _normalise_gene(y_name, cand_set)
        x_is_cand = x_gene is not None
        y_is_cand = y_gene is not None

        if x_is_cand and not y_is_cand:
            gene, func = x_gene, y_name
        elif y_is_cand and not x_is_cand:
            gene, func = y_gene, x_name
        else:
            continue  # both genes or neither — skip

        key = f"{row['relation']}::{func}"
        func_genes.setdefault(key, set()).add(gene)

    # Drop overly sparse terms
    threshold = min_genes if min_genes is not None else MIN_GENES_PER_TERM
    func_genes = {k: v for k, v in func_genes.items()
                  if len(v) >= threshold}
    return func_genes


def compute_functional_burden(
    func_genes: dict[str, set[str]],
    candidate_genes: list[str],
    mut: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Compute FMB matrix.

    fmb[p, f] = sum(mut[p,g]*mask[p,g] for g in genes_of[f])
              / max(sum(mask[p,g] for g in genes_of[f]), 1)

    Returns (fmb_matrix [n_samples, n_functions], function_names).
    """
    gene_to_idx = {g: i for i, g in enumerate(candidate_genes)}
    func_names = sorted(func_genes.keys())
    n_funcs = len(func_names)
    n_samples = mut.shape[0]

    fmb = np.zeros((n_samples, n_funcs), dtype=np.float32)
    for fi, fn in enumerate(func_names):
        gi = np.array([gene_to_idx[g] for g in func_genes[fn]
                       if g in gene_to_idx])
        if len(gi) == 0:
            continue
        numerator = np.sum(mut[:, gi] * mask[:, gi], axis=1)
        denominator = np.maximum(np.sum(mask[:, gi], axis=1), 1.0)
        fmb[:, fi] = numerator / denominator

    return fmb, func_names


# ---------------------------------------------------------------------------
# PPI Neighborhood Mutation Burden
# ---------------------------------------------------------------------------

def build_ppi_adjacency(
    subkg_path: Path,
    candidate_genes: list[str],
    edge_types: list[str],
) -> np.ndarray:
    """Build gene-gene adjacency matrix from PPI edges.

    Returns adj [n_genes, n_genes] binary matrix (symmetric).
    """
    cand_set = set(candidate_genes)
    gene_to_idx = {g: i for i, g in enumerate(candidate_genes)}
    n = len(candidate_genes)

    df = pd.read_csv(subkg_path, low_memory=False)
    df = df[df["relation"].isin(edge_types)]

    adj = np.zeros((n, n), dtype=np.float32)
    x_vals = df["x_name"].astype(str)
    y_vals = df["y_name"].astype(str)
    for x, y in zip(x_vals, y_vals):
        xn = _normalise_gene(x, cand_set)
        yn = _normalise_gene(y, cand_set)
        if xn and yn and xn != yn:
            i, j = gene_to_idx[xn], gene_to_idx[yn]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return adj


def compute_ppi_burden(
    adj: np.ndarray,
    mut: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """PPI neighborhood mutation burden per gene per patient.

    ppi_burden[p, g] = sum(mut[p, neighbor] * mask[p, neighbor])
                     / max(sum(mask[p, neighbor]), 1)
    for each neighbor of gene g in the PPI network.

    Returns ppi_burden [n_samples, n_genes].
    """
    # adj[g, neighbor] = 1 if connected
    # For each gene g, neighbors = adj[g, :] > 0
    degree = adj.sum(axis=1)  # [n_genes]

    # Vectorized: mut_masked = mut * mask  [n_samples, n_genes]
    mut_masked = mut * mask

    # ppi_burden = (mut_masked @ adj) / max(mask @ adj, 1)
    numerator = mut_masked @ adj       # [n_samples, n_genes]
    denominator = mask @ adj           # [n_samples, n_genes]
    denominator = np.maximum(denominator, 1.0)

    return (numerator / denominator).astype(np.float32)


# ---------------------------------------------------------------------------
# Disease-Gene Feature: disease association count per gene
# ---------------------------------------------------------------------------

def compute_disease_gene_weight(
    subkg_path: Path,
    candidate_genes: list[str],
    edge_types: list[str],
) -> np.ndarray:
    """Count number of diseases associated with each candidate gene.

    Returns disease_count [n_genes] — can be used as gene-level weight.
    """
    cand_set = set(candidate_genes)
    gene_to_idx = {g: i for i, g in enumerate(candidate_genes)}
    n = len(candidate_genes)

    df = pd.read_csv(subkg_path, low_memory=False)
    df = df[df["relation"].isin(edge_types)]

    gene_diseases: dict[int, set[str]] = {}
    for _, row in df.iterrows():
        x, y = str(row["x_name"]), str(row["y_name"])
        xn = _normalise_gene(x, cand_set)
        yn = _normalise_gene(y, cand_set)
        if xn:
            gene_diseases.setdefault(gene_to_idx[xn], set()).add(y)
        if yn:
            gene_diseases.setdefault(gene_to_idx[yn], set()).add(x)

    counts = np.zeros(n, dtype=np.float32)
    for gi, diseases in gene_diseases.items():
        counts[gi] = len(diseases)

    return counts

def train_node2vec(
    subkg_path: Path,
    embedding_dim: int = 64,
    walk_length: int = 20,
    context_size: int = 10,
    walks_per_node: int = 10,
    num_epochs: int = 5,
    lr: float = 0.01,
) -> dict[str, np.ndarray]:
    """Train Node2Vec on KG subgraph (homogeneous projection).

    Returns dict mapping node_name → embedding vector.
    """
    import torch

    df = pd.read_csv(subkg_path)
    all_nodes = sorted(set(df["x_name"].astype(str).tolist()
                           + df["y_name"].astype(str).tolist()))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(all_nodes)

    src = df["x_name"].astype(str).map(node_to_idx).values
    dst = df["y_name"].astype(str).map(node_to_idx).values
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]),
                  np.concatenate([dst, src])]),
        dtype=torch.long,
    )
    # Remove self-loops
    keep = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, keep]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from torch_geometric.nn import Node2Vec as PyGNode2Vec

        model = PyGNode2Vec(
            edge_index, embedding_dim=embedding_dim,
            walk_length=walk_length, context_size=context_size,
            walks_per_node=walks_per_node, num_negative_samples=1,
            sparse=True,
        ).to(device)

        loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)

        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(len(loader), 1)
            if epoch % 2 == 0 or epoch == 1:
                print(f"    Node2Vec epoch {epoch}/{num_epochs}: "
                      f"loss={avg:.4f}")

        embs = model.embedding.weight.detach().cpu().numpy()

    except Exception as e:
        print(f"    WARNING: PyG Node2Vec failed ({e}), using SVD fallback")
        # Fallback: truncated SVD on adjacency matrix
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import svds

        row = edge_index[0].numpy()
        col = edge_index[1].numpy()
        adj = coo_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(n_nodes, n_nodes),
        ).tocsr()
        k = min(embedding_dim, n_nodes - 2)
        u, s, _ = svds(adj.astype(np.float64), k=k)
        embs = (u * np.sqrt(s)[None, :]).astype(np.float32)
        # Pad if k < embedding_dim
        if embs.shape[1] < embedding_dim:
            pad = np.zeros((n_nodes, embedding_dim - embs.shape[1]),
                           dtype=np.float32)
            embs = np.concatenate([embs, pad], axis=1)

    return {name: embs[idx] for name, idx in node_to_idx.items()}


def compute_patient_kg_embedding(
    emb_dict: dict[str, np.ndarray],
    candidate_genes: list[str],
    mut: np.ndarray,
    mask: np.ndarray,
    dim: int = 64,
) -> np.ndarray:
    """Aggregate frozen KG embeddings per patient.

    Returns kg_emb_agg [n_samples, dim]:
        mean embedding of mutated + panel-covered genes.
    """
    gene_embs = np.zeros((len(candidate_genes), dim), dtype=np.float32)
    has_emb = np.zeros(len(candidate_genes), dtype=np.float32)
    for i, g in enumerate(candidate_genes):
        if g in emb_dict:
            gene_embs[i] = emb_dict[g]
            has_emb[i] = 1.0

    n_with = int(has_emb.sum())
    print(f"    {n_with}/{len(candidate_genes)} candidate genes have "
          f"KG embeddings")

    # weight[p, g] = mut[p,g] * mask[p,g] * has_emb[g]
    weight = mut * mask * has_emb[None, :]          # [n_samples, n_genes]
    total_w = weight.sum(axis=1, keepdims=True)     # [n_samples, 1]
    total_w = np.maximum(total_w, 1.0)
    kg_emb = (weight @ gene_embs) / total_w         # [n_samples, dim]

    return kg_emb.astype(np.float32)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def precompute_all_features(kg_name: str, save_dir: Path | None = None,
                            min_genes: int | None = None,
                            fmb_only: bool = False,
                            wes: bool = False,
                            gene_list_path: Path | None = None):
    """Load data + KG subgraph → compute FMB + Node2Vec → save CSVs.

    When wes=True, reads splits from output/processed_wes/ and the WES candidate
    gene list (intersection of WES union and KG union).
    """
    if save_dir is None:
        save_dir = FEAT_DIR / (f"{kg_name}_wes" if wes else kg_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    effective_min = min_genes if min_genes is not None else MIN_GENES_PER_TERM

    if wes:
        wes_path = gene_list_path or (
            ROOT / "output" / "processed_wes" / "wes_candidate_genes.csv"
        )
        candidate_genes = load_candidate_genes(wes_path)
        print(f"  WES mode: {len(candidate_genes)} genes")
    else:
        candidate_genes = load_candidate_genes()
    subkg_path = SUBKG_DIR / f"subkg_{kg_name}.csv"
    if not subkg_path.exists():
        print(f"  ERROR: Subgraph not found: {subkg_path}")
        return

    # --- FMB ---
    edge_types = FMB_EDGE_TYPES.get(kg_name, [])
    func_genes: dict[str, set[str]] = {}
    if edge_types:
        print(f"  Extracting FMB edges: {edge_types}")
        func_genes = extract_function_gene_map(
            subkg_path, candidate_genes, edge_types, min_genes=effective_min)
        print(f"  Found {len(func_genes)} functional terms "
              f"(>= {effective_min} genes each)")
    else:
        print(f"  No FMB edge types defined for {kg_name}")

    # --- PPI adjacency ---
    ppi_types = PPI_EDGE_TYPES.get(kg_name, [])
    ppi_adj = None
    if ppi_types:
        ppi_adj = build_ppi_adjacency(subkg_path, candidate_genes, ppi_types)
        n_ppi = int(ppi_adj.sum() / 2)
        genes_with_ppi = int((ppi_adj.sum(axis=1) > 0).sum())
        print(f"  PPI: {n_ppi} edges among {genes_with_ppi}/{len(candidate_genes)} genes")

    # --- Disease-Gene weights ---
    dis_types = DISEASE_EDGE_TYPES.get(kg_name, [])
    dis_weights = None
    if dis_types:
        dis_weights = compute_disease_gene_weight(
            subkg_path, candidate_genes, dis_types)
        n_with_dis = int((dis_weights > 0).sum())
        print(f"  Disease-Gene: {n_with_dis}/{len(candidate_genes)} genes "
              f"have disease associations (max={dis_weights.max():.0f})")

    # --- Node2Vec ---
    emb_dict = {}
    if not fmb_only:
        print(f"  Training Node2Vec embeddings (dim=64)...")
        emb_dict = train_node2vec(subkg_path, embedding_dim=64)

    # --- Process each split ---
    for prefix, mut_path in _iter_splits(wes=wes):
        if wes:
            mask_path = mut_path.parent / mut_path.name.replace("_mut.csv", "_mask.csv")
        else:
            mask_path = PROCESSED / f"{prefix}_mask.csv"
        mut_df = pd.read_csv(mut_path, index_col=0)
        mask_df = pd.read_csv(mask_path, index_col=0)
        mut = mut_df.values.astype(np.float32)
        mask = mask_df.values.astype(np.float32)

        # FMB
        if func_genes:
            fmb, func_names = compute_functional_burden(
                func_genes, candidate_genes, mut, mask)
            fmb_df = pd.DataFrame(fmb, index=mut_df.index,
                                  columns=func_names)
            fmb_df.to_csv(save_dir / f"{prefix}_fmb.csv")

        # PPI burden
        if ppi_adj is not None:
            ppi_burd = compute_ppi_burden(ppi_adj, mut, mask)
            ppi_df = pd.DataFrame(ppi_burd, index=mut_df.index,
                                  columns=[f"ppi_{g}" for g in candidate_genes])
            ppi_df.to_csv(save_dir / f"{prefix}_ppi.csv")

        # KG embedding
        if emb_dict:
            kg_emb = compute_patient_kg_embedding(
                emb_dict, candidate_genes, mut, mask)
            kg_df = pd.DataFrame(
                kg_emb, index=mut_df.index,
                columns=[f"kgemb_{i}" for i in range(kg_emb.shape[1])])
            kg_df.to_csv(save_dir / f"{prefix}_kgemb.csv")

        parts = [f"FMB={fmb.shape[1] if func_genes else 0}"]
        if ppi_adj is not None:
            parts.append(f"PPI={ppi_burd.shape[1]}")
        if emb_dict:
            parts.append(f"kgemb={kg_emb.shape[1]}")
        print(f"    {prefix}: {', '.join(parts)}")

    # Save metadata
    meta = {
        "kg_name": kg_name,
        "n_fmb_features": len(func_genes),
        "fmb_edge_types": edge_types,
        "n_candidate_genes": len(candidate_genes),
        "embedding_dim": 64,
        "n_ppi_edges": int(ppi_adj.sum() / 2) if ppi_adj is not None else 0,
        "n_disease_genes": int((dis_weights > 0).sum()) if dis_weights is not None else 0,
    }
    # Save disease weights once (same for all splits)
    if dis_weights is not None:
        np.save(save_dir / "disease_gene_weights.npy", dis_weights)
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Done — saved to {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute KG features (FMB + Node2Vec)")
    parser.add_argument("--kg", type=str, default="primekg",
                        help="KG name (or 'all')")
    parser.add_argument("--all", action="store_true",
                        help="Process all 7 KGs")
    parser.add_argument("--min_genes", type=int, default=None,
                        help="MIN_GENES_PER_TERM override (default=3)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output dir (e.g. '_mg5')")
    parser.add_argument("--wes", action="store_true",
                        help="Use WES gene set + output/processed_wes/ splits")
    args = parser.parse_args()

    kgs = AVAILABLE_KGS if args.all else [args.kg]
    for kg in kgs:
        print(f"\n{'=' * 60}")
        mg_label = args.min_genes if args.min_genes else MIN_GENES_PER_TERM
        print(f"Pre-computing features: {kg} (min_genes={mg_label})")
        print(f"{'=' * 60}")
        out_dir = FEAT_DIR / f"{kg}{args.output_suffix}" if args.output_suffix else None
        precompute_all_features(
            kg, save_dir=out_dir, min_genes=args.min_genes, wes=args.wes
        )
