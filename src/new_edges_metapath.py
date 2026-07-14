"""Add new edge types (reaction/catalysis/ptmod/covaries) + metapath gene-gene burden.

New adjacency types:
  - reaction: gene-gene metabolic reactions (OGB + OpenBioLink)
  - catalysis: gene-gene catalytic relations (OpenBioLink)
  - ptmod: post-translational modification (OGB + OpenBioLink)
  - covaries: gene covariation (Hetionet)
  - overexpr/underexpr: tissue over/under-expression (OpenBioLink, FMB-style)

Metapath burden (computed from existing edges, no subgraph change):
  - disease_cooccur: genes sharing disease associations → virtual adjacency
  - drug_cooccur: genes sharing drug targets → virtual adjacency
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import KG_DIR, PROC_DIR, KGGroupInfo, _load_gene_list, build_kg_group_info
from kg_features import (
    load_candidate_genes, build_ppi_adjacency, compute_ppi_burden,
    extract_function_gene_map, compute_functional_burden,
)
from losses import compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci
from node_type_ablation import EVAL_COHORTS

ROOT = Path(__file__).resolve().parent.parent
SUBKG = ROOT / "output" / "subkg"
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# New edge type definitions
NEW_ADJ_EDGES = {
    "reaction": {
        "ogb_biokg": ["protein-protein_reaction"],
        "openbiolink": ["gene_reaction_gene"],
    },
    "catalysis": {
        "openbiolink": ["gene_catalysis_gene"],
    },
    "ptmod": {
        "ogb_biokg": ["protein-protein_ptmod"],
        "openbiolink": ["gene_ptmod_gene"],
    },
    "covaries": {
        "hetionet": ["covaries"],
    },
}

NEW_FMB_EDGES = {
    "overexpr": {
        "openbiolink": ["gene_overexpressed_anatomy", "gene_underexpressed_anatomy"],
    },
}

# Metapath: build gene-gene adjacency from shared disease/drug associations
DISEASE_EDGES_FOR_META = {
    "primekg": ["disease_protein"],
    "hetionet": ["associates"],
    "drkg": ["Gene:Disease", "Disease:Gene"],
    "ogb_biokg": ["disease-protein"],
    "openbiolink": ["gene_dis"],
}
DRUG_EDGES_FOR_META = {
    "hetionet": ["binds"],
    "drkg": ["Compound:Gene", "Gene:Compound"],
    "ogb_biokg": ["drug-protein"],
    "openbiolink": ["gene_drug"],
}


def build_metapath_adjacency(subkg_path, candidate_genes, edge_types, min_shared=2):
    """Build gene-gene adjacency from shared non-gene neighbors.

    Two genes are connected if they share >= min_shared neighbors
    (e.g., both associated with the same disease).
    """
    cand_set = set(candidate_genes)
    gene_to_idx = {g: i for i, g in enumerate(candidate_genes)}
    n = len(candidate_genes)

    df = pd.read_csv(subkg_path, low_memory=False)
    df = df[df["relation"].isin(edge_types)]

    # gene → set of non-gene neighbors
    gene_neighbors = {}
    for _, row in df.iterrows():
        x, y = str(row["x_name"]), str(row["y_name"])
        if x in cand_set and y not in cand_set:
            gene_neighbors.setdefault(x, set()).add(y)
        elif y in cand_set and x not in cand_set:
            gene_neighbors.setdefault(y, set()).add(x)

    # Build adjacency: genes sharing >= min_shared neighbors
    adj = np.zeros((n, n), dtype=np.float32)
    gene_list = [g for g in candidate_genes if g in gene_neighbors]
    for i, g1 in enumerate(gene_list):
        idx1 = gene_to_idx[g1]
        n1 = gene_neighbors[g1]
        for g2 in gene_list[i+1:]:
            idx2 = gene_to_idx[g2]
            shared = len(n1 & gene_neighbors[g2])
            if shared >= min_shared:
                adj[idx1, idx2] = 1.0
                adj[idx2, idx1] = 1.0

    return adj


def load_base_splits(kg_name):
    kg_feat = KG_DIR / kg_name
    splits, raw = {}, {}
    for sn in ["train"] + EVAL_COHORTS:
        prefix = "train" if sn == "train" else f"valid_{sn}"
        try:
            mut = pd.read_csv(PROC / f"{prefix}_mut.csv", index_col=0)
            mask = pd.read_csv(PROC / f"{prefix}_mask.csv", index_col=0)
            clin = pd.read_csv(PROC / f"{prefix}_clin.csv", index_col=0)
            fmb = pd.read_csv(kg_feat / f"{prefix}_fmb.csv", index_col=0)
        except FileNotFoundError:
            continue
        common = mut.index.intersection(clin.index).intersection(fmb.index)
        splits[sn] = {
            "mut": torch.tensor(mut.loc[common].values, dtype=torch.float32),
            "mask": torch.tensor(mask.loc[common].values, dtype=torch.float32),
            "time": torch.tensor(clin.loc[common, "OS_MONTHS"].values, dtype=torch.float32),
            "event": torch.tensor(clin.loc[common, "event"].values, dtype=torch.float32),
            "fmb": torch.tensor(fmb.loc[common].values, dtype=torch.float32),
            "sample_ids": common.tolist(),
        }
        raw[sn] = (mut.loc[common].values.astype(np.float32),
                    mask.loc[common].values.astype(np.float32))
    return splits, raw


def augment_splits(splits, extra_features_list):
    out = {}
    for key, data in splits.items():
        extras = [torch.tensor(f[key], dtype=torch.float32)
                  for f in extra_features_list if key in f]
        if not extras:
            continue
        new = dict(data)
        new["fmb"] = torch.cat([data["fmb"]] + extras, dim=1)
        out[key] = new
    return out


def build_combo_info(kg_name, extra_term_lists, n_genes):
    base = build_kg_group_info(kg_name)
    groups = list(base.group_names)
    terms = list(base.term_names)
    masks = list(base.gene_term_mask)
    slices = list(base.fmb_slices)
    offset = slices[-1][1] if slices else 0
    total = base.n_total_terms
    for tnames, mode, label in extra_term_lists:
        groups.append(label)
        terms.append(tnames)
        n_t = len(tnames)
        m = torch.eye(n_genes, dtype=torch.float32) if mode == "adj" else \
            torch.ones(n_genes, n_t, dtype=torch.float32) / max(n_genes, 1)
        masks.append(m)
        slices.append((offset, offset + n_t))
        offset += n_t
        total += n_t
    return KGGroupInfo(kg_name=kg_name, group_names=groups, term_names=terms,
                       gene_term_mask=masks, fmb_slices=slices,
                       n_genes=n_genes, n_total_terms=total)


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
    r = {"val_ci": round(best, 4)}
    ext_cis, sigs = [], []
    for c in EVAL_COHORTS:
        if c not in valid_data:
            continue
        risk = get_risk(model, valid_data[c])
        m = compute_all_metrics(risk, valid_data[c]["time"].numpy(), valid_data[c]["event"].numpy())
        r[f"{c}_ci"] = round(m["c_index"], 4)
        r[f"{c}_p"] = round(m["p_value"], 4)
        ext_cis.append(m["c_index"])
        if m["p_value"] < 0.05:
            sigs.append(c)
    r["ext_ci"] = round(np.mean(ext_cis), 4)
    r["n_sig"] = len(sigs)
    r["sigs"] = sigs
    return r


def main():
    genes = load_candidate_genes()
    n_genes = len(genes)
    all_results = []

    # Test KGs with significant unused edges
    TEST_KGS = ["ogb_biokg", "openbiolink", "hetionet", "drkg"]
    MODEL = "path_attn"
    kg_cache = {}

    for kg in TEST_KGS:
        print(f"\n{'='*60}")
        print(f"  {kg}")
        print(f"{'='*60}")

        splits, raw = load_base_splits(kg)
        subkg_path = SUBKG / f"subkg_{kg}.csv"

        # Previously used PPI + best combo nodes
        from node_type_ablation import ALL_NODE_TYPES
        existing_nf = {}
        for nt, (mode, edict) in ALL_NODE_TYPES.items():
            from node_type_ablation import compute_node_features
            res = compute_node_features(kg, nt, mode, edict, genes, raw)
            if res:
                feats, tnames, n_info = res
                existing_nf[nt] = (feats, tnames, mode)

        # Compute NEW adj-type features
        new_features = {}
        for feat_name, kg_edges in NEW_ADJ_EDGES.items():
            edges = kg_edges.get(kg, [])
            if not edges:
                continue
            adj = build_ppi_adjacency(subkg_path, genes, edges)
            n_e = int(adj.sum() / 2)
            if n_e == 0:
                continue
            print(f"  NEW {feat_name}: {n_e} edges")
            feats = {}
            for prefix, (mut, mask) in raw.items():
                feats[prefix] = compute_ppi_burden(adj, mut, mask)
            new_features[feat_name] = (feats, [f"{feat_name}_{g}" for g in genes], "adj")

        # Compute NEW FMB-type features
        for feat_name, kg_edges in NEW_FMB_EDGES.items():
            edges = kg_edges.get(kg, [])
            if not edges:
                continue
            func_genes = extract_function_gene_map(subkg_path, genes, edges, min_genes=2)
            if len(func_genes) < 5:
                continue
            print(f"  NEW {feat_name}: {len(func_genes)} terms")
            feats = {}
            for prefix, (mut, mask) in raw.items():
                fmb_mat, fmb_names = compute_functional_burden(func_genes, genes, mut, mask)
                feats[prefix] = fmb_mat
            new_features[feat_name] = (feats, fmb_names, "fmb")

        # Compute metapath adjacencies
        for meta_name, meta_edges in [("meta_disease", DISEASE_EDGES_FOR_META),
                                       ("meta_drug", DRUG_EDGES_FOR_META)]:
            edges = meta_edges.get(kg, [])
            if not edges:
                continue
            adj = build_metapath_adjacency(subkg_path, genes, edges, min_shared=2)
            n_e = int(adj.sum() / 2)
            if n_e == 0:
                continue
            print(f"  METAPATH {meta_name}: {n_e} gene-gene edges (shared>=2)")
            feats = {}
            for prefix, (mut, mask) in raw.items():
                feats[prefix] = compute_ppi_burden(adj, mut, mask)
            new_features[meta_name] = (feats, [f"{meta_name}_{g}" for g in genes], "adj")

        if not new_features:
            print(f"  No new features for {kg}")
            continue

        # Test: best_existing_combo + each new feature individually
        # First get best existing combo per KG
        best_combos = {
            "drkg": ["ppi", "disease", "drug"],
            "ogb_biokg": ["ppi", "drug"],
            "openbiolink": ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"],
            "hetionet": ["ppi", "anatomy", "regulatory"],
        }
        base_nodes = best_combos.get(kg, ["ppi"])

        for new_name, (new_feats, new_tnames, new_mode) in new_features.items():
            # Combine existing best + new feature
            all_extra_f = []
            all_extra_i = []
            for nt in base_nodes:
                if nt in existing_nf:
                    all_extra_f.append(existing_nf[nt][0])
                    all_extra_i.append((existing_nf[nt][1], existing_nf[nt][2], f"x_{nt}"))
            all_extra_f.append(new_feats)
            all_extra_i.append((new_tnames, new_mode, f"x_{new_name}"))

            aug = augment_splits(splits, all_extra_f)
            info = build_combo_info(kg, all_extra_i, n_genes)

            tag = f"{MODEL}_{kg}_best+{new_name}"
            print(f"\n  {tag} ({info.n_total_terms} features)")

            r = train_and_eval(MODEL, info, aug["train"],
                               {c: aug[c] for c in EVAL_COHORTS if c in aug})
            r["tag"] = tag
            r["kg"] = kg
            r["new_feature"] = new_name
            print(f"    val={r['val_ci']:.4f} ext={r['ext_ci']:.4f} sig={r['n_sig']}/11: {r['sigs']}")
            all_results.append(r)

    print(f"\n{'='*80}")
    print("NEW EDGE TYPES + METAPATH RESULTS")
    print(f"{'='*80}")
    print(f"{'Tag':<50} {'ValCI':>6} {'ExtCI':>6} {'Sig':>5}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: (-x["n_sig"], -x["ext_ci"])):
        print(f"{r['tag']:<50} {r['val_ci']:>6.4f} {r['ext_ci']:>6.4f} {r['n_sig']:>3}/11")

    out = EXP_DIR / "new_edges_metapath.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
