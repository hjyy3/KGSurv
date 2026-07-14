"""Experiment: test each heterogeneous node type individually.

Node type categories:
  - PPI (gene-gene): already implemented as adjacency burden
  - Disease-Gene: FMB-style burden using disease edges
  - Drug-Gene: FMB-style burden using drug/compound edges
  - Phenotype-Gene: FMB-style burden using phenotype edges
  - Regulatory (gene-gene): adjacency burden using regulatory edges (Hetionet only)
  - Anatomy-Gene: FMB-style burden using expression/anatomy edges

For each: generate features → train SparsePathNet × top-3 KGs → compare
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import (
    KG_DIR, PROC_DIR, ALL_KGS, KGGroupInfo,
    _load_gene_list, build_kg_group_info,
)
from kg_features import (
    load_candidate_genes, extract_function_gene_map,
    compute_functional_burden, build_ppi_adjacency, compute_ppi_burden,
)
from losses import compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci

ROOT = Path(__file__).resolve().parent.parent
SUBKG = ROOT / "output" / "subkg"
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_COHORTS = [
    "Gandara", "Hugo", "Liu", "Mariathasan", "Miao",
    "PUSH", "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae",
]

# =====================================================================
# Define edge type categories per KG
# =====================================================================

# Gene-Term types (FMB-style burden)
DISEASE_EDGES = {
    "primekg": ["disease_protein"],
    "hetionet": ["associates"],
    "drkg": ["Gene:Disease", "Disease:Gene"],
    "ibkh": ["Di_G"],
    "monarch": ["gene_associated_with_condition"],
    "ogb_biokg": ["disease-protein"],
    "openbiolink": ["gene_dis"],
}

DRUG_EDGES = {
    "hetionet": ["binds"],
    "drkg": ["Compound:Gene", "Gene:Compound", "DrugHumGen:Compound:Gene"],
    "ibkh": ["D_G"],
    "ogb_biokg": ["drug-protein"],
    "openbiolink": ["gene_drug", "drug_binding_gene", "drug_inhibition_gene",
                    "drug_activation_gene"],
}

PHENOTYPE_EDGES = {
    "primekg": ["phenotype_protein"],
    "monarch": ["has_phenotype"],
    "openbiolink": ["gene_phenotype"],
}

ANATOMY_EDGES = {
    "hetionet": ["expresses"],
    "drkg": ["Anatomy:Gene"],
    "monarch": ["expressed_in"],
    "openbiolink": ["gene_expressed_anatomy"],
}

# Gene-Gene types (adjacency burden, like PPI)
REGULATORY_EDGES = {
    "hetionet": ["regulates", "upregulates", "downregulates"],
    "openbiolink": ["gene_activation_gene", "gene_inhibition_gene",
                    "gene_expression_gene"],
}

# Already defined in kg_features.py but repeat for completeness
PPI_EDGES = {
    "primekg": ["protein_protein"],
    "hetionet": ["interacts"],
    "drkg": ["Gene:Gene", "HumGenHumGen:Gene:Gene"],
    "ibkh": ["G_G"],
    "monarch": ["interacts_with"],
    "ogb_biokg": ["protein-protein_binding", "protein-protein_catalysis",
                  "protein-protein_activation", "protein-protein_inhibition"],
    "openbiolink": ["gene_gene", "gene_binding_gene"],
}

ALL_NODE_TYPES = {
    "ppi": ("adj", PPI_EDGES),
    "disease": ("fmb", DISEASE_EDGES),
    "drug": ("fmb", DRUG_EDGES),
    "phenotype": ("fmb", PHENOTYPE_EDGES),
    "anatomy": ("fmb", ANATOMY_EDGES),
    "regulatory": ("adj", REGULATORY_EDGES),
}


# =====================================================================
# Feature generation
# =====================================================================

def compute_node_features(kg_name, node_type, mode, edge_dict, genes, split_data):
    """Compute features for a node type. Returns (features, term_names) or None."""
    edges = edge_dict.get(kg_name, [])
    if not edges:
        return None

    subkg_path = SUBKG / f"subkg_{kg_name}.csv"

    if mode == "adj":
        # Gene-gene adjacency burden
        adj = build_ppi_adjacency(subkg_path, genes, edges)
        n_edges = int(adj.sum() / 2)
        if n_edges == 0:
            return None
        features = {}
        for prefix, (mut, mask) in split_data.items():
            features[prefix] = compute_ppi_burden(adj, mut, mask)
        term_names = [f"{node_type}_{g}" for g in genes]
        return features, term_names, n_edges

    elif mode == "fmb":
        # FMB-style term burden
        func_genes = extract_function_gene_map(
            subkg_path, genes, edges, min_genes=2)
        if len(func_genes) < 5:
            return None
        features = {}
        for prefix, (mut, mask) in split_data.items():
            fmb_mat, fmb_names = compute_functional_burden(func_genes, genes, mut, mask)
            features[prefix] = fmb_mat
        return features, fmb_names, len(func_genes)


# =====================================================================
# Training
# =====================================================================

def load_base_data(kg_name):
    """Load base FMB + mut/mask data."""
    kg_feat = KG_DIR / kg_name
    splits = {}
    raw_splits = {}

    prefix = "train"
    mut = pd.read_csv(PROC / f"{prefix}_mut.csv", index_col=0)
    mask = pd.read_csv(PROC / f"{prefix}_mask.csv", index_col=0)
    clin = pd.read_csv(PROC / f"{prefix}_clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / f"{prefix}_fmb.csv", index_col=0)
    common = mut.index.intersection(clin.index).intersection(fmb.index)
    splits["train"] = {
        "mut": torch.tensor(mut.loc[common].values, dtype=torch.float32),
        "mask": torch.tensor(mask.loc[common].values, dtype=torch.float32),
        "time": torch.tensor(clin.loc[common, "OS_MONTHS"].values, dtype=torch.float32),
        "event": torch.tensor(clin.loc[common, "event"].values, dtype=torch.float32),
        "fmb": torch.tensor(fmb.loc[common].values, dtype=torch.float32),
        "sample_ids": common.tolist(),
    }
    raw_splits["train"] = (mut.loc[common].values.astype(np.float32),
                           mask.loc[common].values.astype(np.float32))

    for c in EVAL_COHORTS:
        try:
            prefix = f"valid_{c}"
            m = pd.read_csv(PROC / f"{prefix}_mut.csv", index_col=0)
            mk = pd.read_csv(PROC / f"{prefix}_mask.csv", index_col=0)
            cl = pd.read_csv(PROC / f"{prefix}_clin.csv", index_col=0)
            fb = pd.read_csv(kg_feat / f"{prefix}_fmb.csv", index_col=0)
            cm = m.index.intersection(cl.index).intersection(fb.index)
            splits[c] = {
                "mut": torch.tensor(m.loc[cm].values, dtype=torch.float32),
                "mask": torch.tensor(mk.loc[cm].values, dtype=torch.float32),
                "time": torch.tensor(cl.loc[cm, "OS_MONTHS"].values, dtype=torch.float32),
                "event": torch.tensor(cl.loc[cm, "event"].values, dtype=torch.float32),
                "fmb": torch.tensor(fb.loc[cm].values, dtype=torch.float32),
                "sample_ids": cm.tolist(),
            }
            raw_splits[c] = (m.loc[cm].values.astype(np.float32),
                             mk.loc[cm].values.astype(np.float32))
        except FileNotFoundError:
            pass

    return splits, raw_splits


def add_features_to_splits(splits, node_features):
    """Concatenate node features to FMB in all splits."""
    out = {}
    for key, data in splits.items():
        if key not in node_features:
            continue
        new_data = dict(data)
        extra = torch.tensor(node_features[key], dtype=torch.float32)
        new_data["fmb"] = torch.cat([data["fmb"], extra], dim=1)
        out[key] = new_data
    return out


def build_info_with_extra(kg_name, term_names, n_genes, mode):
    """Build KGGroupInfo with extra node type as additional group."""
    base = build_kg_group_info(kg_name)
    if mode == "adj":
        extra_mask = torch.eye(n_genes, dtype=torch.float32)
    else:
        # FMB-style: need gene-term connectivity from subgraph
        # Simplified: use identity-like mask (each term maps to itself)
        extra_mask = torch.zeros(n_genes, len(term_names), dtype=torch.float32)
        # We don't have exact connectivity here, just use ones
        # (the actual burden is precomputed, mask is for interpretability only)
        extra_mask[:, :] = 1.0 / max(n_genes, 1)

    old_end = base.fmb_slices[-1][1] if base.fmb_slices else 0
    return KGGroupInfo(
        kg_name=kg_name,
        group_names=base.group_names + [f"extra_{term_names[0].split('_')[0]}"],
        term_names=base.term_names + [term_names],
        gene_term_mask=base.gene_term_mask + [extra_mask],
        fmb_slices=base.fmb_slices + [(old_end, old_end + len(term_names))],
        n_genes=base.n_genes,
        n_total_terms=base.n_total_terms + len(term_names),
    )


@torch.no_grad()
def get_risk(model, data):
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def train_and_eval(model_name, kg_info, train_data, valid_data, seed=42):
    _seed_everything(seed)
    tr, va = _split_data(train_data, 0.8, seed)
    model = create_model(model_name, kg_info, hidden_dim=32, dropout=0.1)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    best, pat, st = 0.0, 0, None
    for ep in range(1, 81):
        train_epoch(model, tr, opt, 64, device)
        ci = evaluate_ci(model, va, device)
        sch.step(ci)
        if ci > best:
            best, pat = ci, 0
            st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= 15:
            break
    if st:
        model.load_state_dict(st)

    n_sig = 0
    ext_cis = []
    sigs = []
    for c in EVAL_COHORTS:
        if c not in valid_data:
            continue
        risk = get_risk(model, valid_data[c])
        m = compute_all_metrics(risk, valid_data[c]["time"].numpy(), valid_data[c]["event"].numpy())
        ext_cis.append(m["c_index"])
        if m["p_value"] < 0.05:
            n_sig += 1
            sigs.append(c)
    return best, np.mean(ext_cis), n_sig, sigs


# =====================================================================
# Main
# =====================================================================

def main():
    TOP_KGS = ["primekg", "ogb_biokg", "hetionet"]
    MODEL = "sparse_path"
    genes = load_candidate_genes()
    n_genes = len(genes)

    all_results = []

    for kg in TOP_KGS:
        print(f"\n{'='*70}")
        print(f"  KG = {kg}")
        print(f"{'='*70}")

        splits, raw_splits = load_base_data(kg)
        base_info = build_kg_group_info(kg)

        # Baseline: FMB only
        print(f"\n  [baseline] FMB only")
        val_ci, ext_ci, n_sig, sigs = train_and_eval(MODEL, base_info, splits["train"],
                                                      {c: splits[c] for c in EVAL_COHORTS if c in splits})
        print(f"    val={val_ci:.4f} ext={ext_ci:.4f} sig={n_sig}/11: {sigs}")
        all_results.append({"kg": kg, "node_type": "baseline", "n_features": 0,
                           "val_ci": round(val_ci, 4), "ext_ci": round(ext_ci, 4),
                           "n_sig": n_sig, "sigs": sigs})

        # Test each node type
        for nt_name, (mode, edge_dict) in ALL_NODE_TYPES.items():
            print(f"\n  [{nt_name}] mode={mode}")

            result = compute_node_features(kg, nt_name, mode, edge_dict, genes, raw_splits)
            if result is None:
                print(f"    No edges for {kg}, skipping")
                all_results.append({"kg": kg, "node_type": nt_name, "n_features": 0,
                                   "val_ci": 0, "ext_ci": 0, "n_sig": -1, "sigs": ["N/A"]})
                continue

            node_feats, term_names, n_info = result
            print(f"    {n_info} {'edges' if mode == 'adj' else 'terms'}, "
                  f"{len(term_names)} features")

            # Build augmented data
            aug_splits = add_features_to_splits(splits, node_feats)
            if "train" not in aug_splits:
                continue
            aug_info = build_info_with_extra(kg, term_names, n_genes, mode)

            val_ci, ext_ci, n_sig, sigs = train_and_eval(
                MODEL, aug_info, aug_splits["train"],
                {c: aug_splits[c] for c in EVAL_COHORTS if c in aug_splits})
            print(f"    val={val_ci:.4f} ext={ext_ci:.4f} sig={n_sig}/11: {sigs}")

            all_results.append({"kg": kg, "node_type": nt_name, "n_features": len(term_names),
                               "val_ci": round(val_ci, 4), "ext_ci": round(ext_ci, 4),
                               "n_sig": n_sig, "sigs": sigs})

    # Summary
    print(f"\n{'='*90}")
    print("NODE TYPE ABLATION RESULTS")
    print(f"{'='*90}")
    print(f"{'KG':<12} {'Node Type':<14} {'N_feat':>7} {'ValCI':>7} {'ExtCI':>7} {'Sig':>5}  Cohorts")
    print("-" * 90)
    for r in all_results:
        sig_str = ", ".join(r["sigs"]) if r["n_sig"] >= 0 else "N/A"
        print(f"{r['kg']:<12} {r['node_type']:<14} {r['n_features']:>7} "
              f"{r['val_ci']:>7.4f} {r['ext_ci']:>7.4f} {r['n_sig']:>3}/11  {sig_str}")

    out = EXP_DIR / "node_type_ablation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
