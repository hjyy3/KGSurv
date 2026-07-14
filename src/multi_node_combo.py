"""Evaluate multi-node-type knowledge-graph combinations.

Based on ablation results, combine the best node types per KG:
  - primekg: FMB + PPI + Phenotype
  - ogb_biokg: FMB + PPI + Drug
  - hetionet: FMB + PPI + Anatomy, FMB + PPI + Regulatory, FMB + PPI + Anatomy + Regulatory
Also test FMB + ALL available node types per KG.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import (
    KG_DIR, PROC_DIR, KGGroupInfo, _load_gene_list, build_kg_group_info,
)
from kg_features import (
    load_candidate_genes, extract_function_gene_map,
    compute_functional_burden, build_ppi_adjacency, compute_ppi_burden,
)
from losses import compute_all_metrics
from models_interp import ALL_MODELS, create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci
from node_type_ablation import (
    ALL_NODE_TYPES, compute_node_features, EVAL_COHORTS,
)

ROOT = Path(__file__).resolve().parent.parent
SUBKG = ROOT / "output" / "subkg"
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define combinations to test
COMBOS = [
    # (kg, combo_name, node_types_to_add)
    ("primekg",  "ppi+pheno",         ["ppi", "phenotype"]),
    ("primekg",  "ppi+disease",       ["ppi", "disease"]),
    ("primekg",  "ppi+pheno+disease", ["ppi", "phenotype", "disease"]),
    ("ogb_biokg","ppi+drug",          ["ppi", "drug"]),
    ("ogb_biokg","ppi+drug+disease",  ["ppi", "drug", "disease"]),
    ("hetionet", "ppi+anatomy",       ["ppi", "anatomy"]),
    ("hetionet", "ppi+regulatory",    ["ppi", "regulatory"]),
    ("hetionet", "ppi+anat+reg",      ["ppi", "anatomy", "regulatory"]),
    ("hetionet", "ppi+anat+reg+disease", ["ppi", "anatomy", "regulatory", "disease"]),
]


def load_base_splits(kg_name):
    """Load FMB + mut/mask for train + validation cohorts."""
    kg_feat = KG_DIR / kg_name
    splits, raw = {}, {}
    for split_name in ["train"] + EVAL_COHORTS:
        prefix = "train" if split_name == "train" else f"valid_{split_name}"
        try:
            mut = pd.read_csv(PROC / f"{prefix}_mut.csv", index_col=0)
            mask = pd.read_csv(PROC / f"{prefix}_mask.csv", index_col=0)
            clin = pd.read_csv(PROC / f"{prefix}_clin.csv", index_col=0)
            fmb = pd.read_csv(kg_feat / f"{prefix}_fmb.csv", index_col=0)
        except FileNotFoundError:
            continue
        common = mut.index.intersection(clin.index).intersection(fmb.index)
        splits[split_name] = {
            "mut": torch.tensor(mut.loc[common].values, dtype=torch.float32),
            "mask": torch.tensor(mask.loc[common].values, dtype=torch.float32),
            "time": torch.tensor(clin.loc[common, "OS_MONTHS"].values, dtype=torch.float32),
            "event": torch.tensor(clin.loc[common, "event"].values, dtype=torch.float32),
            "fmb": torch.tensor(fmb.loc[common].values, dtype=torch.float32),
            "sample_ids": common.tolist(),
        }
        raw[split_name] = (mut.loc[common].values.astype(np.float32),
                           mask.loc[common].values.astype(np.float32))
    return splits, raw


def augment_splits(splits, extra_features_list):
    """Concatenate multiple extra feature sets to FMB."""
    out = {}
    for key, data in splits.items():
        extras = []
        for feats in extra_features_list:
            if key in feats:
                extras.append(torch.tensor(feats[key], dtype=torch.float32))
        if not extras:
            continue
        new = dict(data)
        new["fmb"] = torch.cat([data["fmb"]] + extras, dim=1)
        out[key] = new
    return out


def build_combo_info(kg_name, extra_term_lists, extra_modes, n_genes):
    """Build KGGroupInfo with multiple extra groups."""
    base = build_kg_group_info(kg_name)
    groups = list(base.group_names)
    terms = list(base.term_names)
    masks = list(base.gene_term_mask)
    slices = list(base.fmb_slices)
    offset = slices[-1][1] if slices else 0
    total = base.n_total_terms

    for (tnames, mode, label) in extra_term_lists:
        groups.append(label)
        terms.append(tnames)
        n_t = len(tnames)
        if mode == "adj":
            m = torch.eye(n_genes, dtype=torch.float32)
        else:
            m = torch.ones(n_genes, n_t, dtype=torch.float32) / max(n_genes, 1)
        masks.append(m)
        slices.append((offset, offset + n_t))
        offset += n_t
        total += n_t

    return KGGroupInfo(
        kg_name=kg_name, group_names=groups, term_names=terms,
        gene_term_mask=masks, fmb_slices=slices,
        n_genes=n_genes, n_total_terms=total,
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
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
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

    result = {"val_ci": round(best, 4)}
    ext_cis, sigs = [], []
    for c in EVAL_COHORTS:
        if c not in valid_data:
            continue
        risk = get_risk(model, valid_data[c])
        m = compute_all_metrics(risk, valid_data[c]["time"].numpy(),
                                valid_data[c]["event"].numpy())
        result[f"{c}_ci"] = round(m["c_index"], 4)
        result[f"{c}_p"] = round(m["p_value"], 4)
        ext_cis.append(m["c_index"])
        if m["p_value"] < 0.05:
            sigs.append(c)
    result["ext_ci"] = round(np.mean(ext_cis), 4)
    result["n_sig"] = len(sigs)
    result["sigs"] = sigs
    return model, result


def main():
    genes = load_candidate_genes()
    n_genes = len(genes)
    all_results = []
    total = len(COMBOS) * len(ALL_MODELS)
    done = 0

    # Cache per-KG data
    kg_cache = {}

    for kg, combo_name, node_types in COMBOS:
        if kg not in kg_cache:
            print(f"\nLoading {kg}...")
            splits, raw = load_base_splits(kg)
            # Compute all node type features for this KG
            node_feats = {}
            for nt_name, (mode, edge_dict) in ALL_NODE_TYPES.items():
                result = compute_node_features(kg, nt_name, mode, edge_dict, genes, raw)
                if result is not None:
                    feats, tnames, n_info = result
                    node_feats[nt_name] = (feats, tnames, mode)
            kg_cache[kg] = (splits, raw, node_feats)

        splits, raw, node_feats = kg_cache[kg]

        # Check all required node types are available
        missing = [nt for nt in node_types if nt not in node_feats]
        if missing:
            print(f"  [skip] {kg}/{combo_name}: missing {missing}")
            continue

        # Build augmented data
        extra_feats = [node_feats[nt][0] for nt in node_types]
        extra_info = [(node_feats[nt][1], node_feats[nt][2], f"extra_{nt}")
                      for nt in node_types]
        aug_splits = augment_splits(splits, extra_feats)
        aug_info = build_combo_info(kg, extra_info, None, n_genes)

        n_extra = sum(len(node_feats[nt][1]) for nt in node_types)
        print(f"\n  {kg} / FMB+{combo_name} ({aug_info.n_total_terms} total features, "
              f"+{n_extra} from {node_types})")

        for model_name in ALL_MODELS:
            done += 1
            tag = f"{model_name}_{kg}_fmb+{combo_name}"
            print(f"  [{done}/{total}] {tag}")

            model, result = train_and_eval(
                model_name, aug_info, aug_splits["train"],
                {c: aug_splits[c] for c in EVAL_COHORTS if c in aug_splits})
            result["tag"] = tag
            result["model"] = model_name
            result["kg"] = kg
            result["combo"] = combo_name
            result["node_types"] = node_types
            print(f"    val={result['val_ci']:.4f} ext={result['ext_ci']:.4f} "
                  f"sig={result['n_sig']}/11: {result['sigs']}")
            all_results.append(result)

    # Summary
    print(f"\n{'='*90}")
    print("MULTI-NODE COMBINATION RESULTS")
    print(f"{'='*90}")
    print(f"{'Tag':<50} {'ValCI':>6} {'ExtCI':>6} {'Sig':>5}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: (-x["n_sig"], -x["ext_ci"])):
        print(f"{r['tag']:<50} {r['val_ci']:>6.4f} {r['ext_ci']:>6.4f} {r['n_sig']:>3}/11")

    out = EXP_DIR / "multi_node_combo.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
