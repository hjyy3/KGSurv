"""Extended multi-node combos: 4 untested KGs + openbiolink full-type experiment.

Focus on PathAttnSurv (best for multi-node fusion).
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
    load_candidate_genes, extract_function_gene_map,
    compute_functional_burden, build_ppi_adjacency, compute_ppi_burden,
)
from losses import compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci
from node_type_ablation import ALL_NODE_TYPES, compute_node_features, EVAL_COHORTS

ROOT = Path(__file__).resolve().parent.parent
SUBKG = ROOT / "output" / "subkg"
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Focus on combos with most potential
COMBOS = [
    # openbiolink: 6 extra types available
    ("openbiolink", "ppi+drug+pheno+anat+reg",    ["ppi","drug","phenotype","anatomy","regulatory"]),
    ("openbiolink", "ppi+drug+anat+reg",           ["ppi","drug","anatomy","regulatory"]),
    ("openbiolink", "ppi+pheno+anat",              ["ppi","phenotype","anatomy"]),
    ("openbiolink", "ppi+disease+drug+pheno+anat+reg", ["ppi","disease","drug","phenotype","anatomy","regulatory"]),
    # drkg: 4 extra types
    ("drkg",        "ppi+drug",                    ["ppi","drug"]),
    ("drkg",        "ppi+disease+drug",            ["ppi","disease","drug"]),
    ("drkg",        "ppi+disease+drug+anat",       ["ppi","disease","drug","anatomy"]),
    # monarch: 4 extra types
    ("monarch",     "ppi+pheno",                   ["ppi","phenotype"]),
    ("monarch",     "ppi+disease+pheno+anat",      ["ppi","disease","phenotype","anatomy"]),
    ("monarch",     "ppi+pheno+anat",              ["ppi","phenotype","anatomy"]),
    # ibkh: 3 extra types
    ("ibkh",        "ppi+drug",                    ["ppi","drug"]),
    ("ibkh",        "ppi+disease+drug",            ["ppi","disease","drug"]),
    # hetionet: try ALL 5 extra
    ("hetionet",    "ALL5",                        ["ppi","disease","drug","anatomy","regulatory"]),
]

MODEL = "path_attn"


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


def train_and_eval(kg_info, train_data, valid_data, seed=42):
    _seed_everything(seed)
    tr, va = _split_data(train_data, 0.8, seed)
    model = create_model(MODEL, kg_info, hidden_dim=32, dropout=0.1)
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
    kg_cache = {}

    for kg, combo_name, node_types in COMBOS:
        if kg not in kg_cache:
            print(f"\nLoading {kg}...")
            splits, raw = load_base_splits(kg)
            nf = {}
            for nt, (mode, edict) in ALL_NODE_TYPES.items():
                res = compute_node_features(kg, nt, mode, edict, genes, raw)
                if res:
                    feats, tnames, n_info = res
                    nf[nt] = (feats, tnames, mode)
                    print(f"  {nt}: {n_info} {'edges' if mode=='adj' else 'terms'}")
            kg_cache[kg] = (splits, raw, nf)

        splits, raw, node_feats = kg_cache[kg]
        missing = [nt for nt in node_types if nt not in node_feats]
        if missing:
            print(f"  [skip] {kg}/{combo_name}: missing {missing}")
            all_results.append({"tag": f"{MODEL}_{kg}_fmb+{combo_name}", "kg": kg,
                               "combo": combo_name, "n_sig": -1, "sigs": ["N/A"]})
            continue

        extra_f = [node_feats[nt][0] for nt in node_types]
        extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in node_types]
        aug = augment_splits(splits, extra_f)
        info = build_combo_info(kg, extra_i, n_genes)

        tag = f"{MODEL}_{kg}_fmb+{combo_name}"
        print(f"\n  {tag} ({info.n_total_terms} features)")

        r = train_and_eval(info, aug["train"], {c: aug[c] for c in EVAL_COHORTS if c in aug})
        r["tag"] = tag
        r["kg"] = kg
        r["combo"] = combo_name
        r["node_types"] = node_types
        print(f"    val={r['val_ci']:.4f} ext={r['ext_ci']:.4f} sig={r['n_sig']}/11: {r['sigs']}")
        all_results.append(r)

    print(f"\n{'='*80}")
    print("EXTENDED MULTI-NODE RESULTS (PathAttnSurv)")
    print(f"{'='*80}")
    print(f"{'Tag':<55} {'ValCI':>6} {'ExtCI':>6} {'Sig':>5}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: (-x.get("n_sig",0), -x.get("ext_ci",0))):
        if r["n_sig"] < 0:
            continue
        print(f"{r['tag']:<55} {r['val_ci']:>6.4f} {r['ext_ci']:>6.4f} {r['n_sig']:>3}/11")

    out = EXP_DIR / "multi_node_extended.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
