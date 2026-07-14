"""Supplement: run missing model × combo combinations.

Round 2 only tested PathAttnSurv. Need SparsePathNet + BipartiteAttnSurv
for drkg, monarch, ibkh, openbiolink best combos + hetionet ALL5.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import KG_DIR, PROC_DIR, KGGroupInfo, _load_gene_list, build_kg_group_info
from kg_features import load_candidate_genes
from losses import compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci
from node_type_ablation import ALL_NODE_TYPES, compute_node_features, EVAL_COHORTS

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"
PROC = ROOT / "output" / "processed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only test the BEST combo per KG from round 2, with the 2 missing models
SUPPLEMENT = [
    ("drkg",        "ppi+disease+drug",            ["ppi", "disease", "drug"]),
    ("drkg",        "ppi+disease+drug+anat",       ["ppi", "disease", "drug", "anatomy"]),
    ("monarch",     "ppi+disease+pheno+anat",      ["ppi", "disease", "phenotype", "anatomy"]),
    ("monarch",     "ppi+pheno",                   ["ppi", "phenotype"]),
    ("ibkh",        "ppi+disease+drug",            ["ppi", "disease", "drug"]),
    ("openbiolink", "ppi+pheno+anat",              ["ppi", "phenotype", "anatomy"]),
    ("openbiolink", "ppi+disease+drug+pheno+anat+reg", ["ppi","disease","drug","phenotype","anatomy","regulatory"]),
    ("hetionet",    "ALL5",                        ["ppi", "disease", "drug", "anatomy", "regulatory"]),
]

MISSING_MODELS = ["sparse_path", "bipartite_attn"]


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
    r = {"val_ci": round(best, 4)}
    ext_cis, sigs = [], []
    for c in EVAL_COHORTS:
        if c not in valid_data:
            continue
        risk = get_risk(model, valid_data[c])
        m = compute_all_metrics(risk, valid_data[c]["time"].numpy(),
                                valid_data[c]["event"].numpy())
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
    total = len(SUPPLEMENT) * len(MISSING_MODELS)
    done = 0

    for kg, combo_name, node_types in SUPPLEMENT:
        if kg not in kg_cache:
            print(f"\nLoading {kg}...")
            splits, raw = load_base_splits(kg)
            nf = {}
            for nt, (mode, edict) in ALL_NODE_TYPES.items():
                res = compute_node_features(kg, nt, mode, edict, genes, raw)
                if res:
                    feats, tnames, n_info = res
                    nf[nt] = (feats, tnames, mode)
            kg_cache[kg] = (splits, raw, nf)

        splits, raw, node_feats = kg_cache[kg]
        missing = [nt for nt in node_types if nt not in node_feats]
        if missing:
            print(f"  [skip] {kg}/{combo_name}: missing {missing}")
            continue

        extra_f = [node_feats[nt][0] for nt in node_types]
        extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in node_types]
        aug = augment_splits(splits, extra_f)
        info = build_combo_info(kg, extra_i, n_genes)

        for model_name in MISSING_MODELS:
            done += 1
            tag = f"{model_name}_{kg}_fmb+{combo_name}"
            print(f"\n  [{done}/{total}] {tag} ({info.n_total_terms} features)")
            r = train_and_eval(model_name, info, aug["train"],
                               {c: aug[c] for c in EVAL_COHORTS if c in aug})
            r["tag"] = tag
            r["model"] = model_name
            r["kg"] = kg
            r["combo"] = combo_name
            r["node_types"] = node_types
            print(f"    val={r['val_ci']:.4f} ext={r['ext_ci']:.4f} "
                  f"sig={r['n_sig']}/11: {r['sigs']}")
            all_results.append(r)

    # Merge with existing results for full comparison
    existing_files = [
        EXP_DIR / "multi_node_combo.json",
        EXP_DIR / "multi_node_extended.json",
    ]
    all_existing = []
    for f in existing_files:
        if f.exists():
            with open(f) as fh:
                all_existing.extend(json.load(fh))

    combined = all_existing + all_results

    # Print full leaderboard
    print(f"\n{'='*90}")
    print("COMPLETE MULTI-NODE LEADERBOARD (all models × all KGs × all combos)")
    print(f"{'='*90}")
    print(f"{'Tag':<55} {'ValCI':>6} {'ExtCI':>6} {'Sig':>5}")
    print("-" * 75)
    for r in sorted(combined, key=lambda x: (-x.get("n_sig", 0), -x.get("ext_ci", 0))):
        if r.get("n_sig", 0) < 0:
            continue
        print(f"{r['tag']:<55} {r.get('val_ci',0):>6.4f} "
              f"{r.get('ext_ci',0):>6.4f} {r.get('n_sig',0):>3}/11")

    # Save supplement
    out = EXP_DIR / "multi_node_supplement.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved supplement to {out}")

    # Save combined leaderboard
    out2 = EXP_DIR / "multi_node_all.json"
    with open(out2, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Saved combined to {out2}")


if __name__ == "__main__":
    main()
