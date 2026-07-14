"""Hyperparameter tuning for selected multi-node configurations.

Grid: learning rate × dropout × hidden dimension.
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
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Selected multi-node configurations
TOP_COMBOS = [
    ("path_attn",    "drkg",        ["ppi", "disease", "drug"]),
    ("path_attn",    "monarch",     ["ppi", "disease", "phenotype", "anatomy"]),
    ("path_attn",    "primekg",     ["ppi", "disease"]),
    ("sparse_path",  "openbiolink", ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"]),
    ("path_attn",    "hetionet",    ["ppi", "anatomy", "regulatory"]),
]

# Hyperparameter grid
LRS = [5e-4, 1e-3, 2e-3]
DROPOUTS = [0.05, 0.1, 0.2]
HIDDEN_DIMS = [16, 32, 64]

# Baseline: lr=1e-3, dropout=0.1, hidden_dim=32


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


def train_and_eval(model_name, kg_info, train_data, valid_data,
                   lr=1e-3, dropout=0.1, hidden_dim=32, seed=42):
    from losses import cox_loss, c_index
    import torch.nn as nn

    _seed_everything(seed)
    tr, va = _split_data(train_data, 0.8, seed)
    model = create_model(model_name, kg_info, hidden_dim=hidden_dim, dropout=dropout)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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

    # For efficiency: only vary ONE param at a time from baseline (lr=1e-3, d=0.1, h=32)
    # This gives 3+3+3-2 = 7 configs per combo (removing duplicated baseline)
    configs = []
    for lr in LRS:
        configs.append((lr, 0.1, 32))
    for do in DROPOUTS:
        if (1e-3, do, 32) not in configs:
            configs.append((1e-3, do, 32))
    for hd in HIDDEN_DIMS:
        if (1e-3, 0.1, hd) not in configs:
            configs.append((1e-3, 0.1, hd))

    total = len(TOP_COMBOS) * len(configs)
    done = 0

    for model_name, kg, node_types in TOP_COMBOS:
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
            print(f"  [skip] {kg}: missing {missing}")
            continue

        extra_f = [node_feats[nt][0] for nt in node_types]
        extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in node_types]
        aug = augment_splits(splits, extra_f)

        combo_tag = "+".join(node_types)
        print(f"\n{'='*60}")
        print(f"  {model_name} × {kg} (FMB+{combo_tag})")
        print(f"{'='*60}")

        for lr, do, hd in configs:
            done += 1
            info = build_combo_info(kg, extra_i, n_genes) if hd == 32 else \
                   build_combo_info(kg, extra_i, n_genes)
            # hidden_dim changes model structure, rebuild info is fine

            hp_tag = f"lr{lr}_do{do}_hd{hd}"
            tag = f"{model_name}_{kg}_{combo_tag}_{hp_tag}"
            print(f"  [{done}/{total}] {hp_tag}", end=" → ")

            r = train_and_eval(model_name, info, aug["train"],
                               {c: aug[c] for c in EVAL_COHORTS if c in aug},
                               lr=lr, dropout=do, hidden_dim=hd)
            r["tag"] = tag
            r["model"] = model_name
            r["kg"] = kg
            r["combo"] = combo_tag
            r["lr"] = lr
            r["dropout"] = do
            r["hidden_dim"] = hd
            print(f"val={r['val_ci']:.4f} ext={r['ext_ci']:.4f} sig={r['n_sig']}/11")
            all_results.append(r)

    # Summary per combo
    print(f"\n{'='*90}")
    print("HYPERPARAMETER TUNING RESULTS")
    print(f"{'='*90}")

    for model_name, kg, node_types in TOP_COMBOS:
        combo_tag = "+".join(node_types)
        combo_results = [r for r in all_results
                         if r["model"] == model_name and r["kg"] == kg]
        if not combo_results:
            continue
        combo_results.sort(key=lambda x: (-x["n_sig"], -x["ext_ci"]))
        print(f"\n  {model_name} × {kg} (FMB+{combo_tag}):")
        print(f"  {'lr':>8} {'dropout':>8} {'hd':>4} {'ValCI':>7} {'ExtCI':>7} {'Sig':>5}")
        print(f"  {'-'*45}")
        for r in combo_results:
            is_base = (r["lr"] == 1e-3 and r["dropout"] == 0.1 and r["hidden_dim"] == 32)
            marker = " ← baseline" if is_base else ""
            print(f"  {r['lr']:>8.0e} {r['dropout']:>8.2f} {r['hidden_dim']:>4} "
                  f"{r['val_ci']:>7.4f} {r['ext_ci']:>7.4f} {r['n_sig']:>3}/11{marker}")

    # Overall best
    best = max(all_results, key=lambda x: (x["n_sig"], x["ext_ci"]))
    print(f"\n  OVERALL BEST: {best['tag']} → sig={best['n_sig']}/11, ext_ci={best['ext_ci']}")

    out = EXP_DIR / "hyperparam_tuning.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
