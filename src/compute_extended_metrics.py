"""Compute extended metrics for the overall best model configuration.

Best: PathAttnSurv x DRKG (FMB+PPI+Disease+Drug, dropout=0.05)
Metrics: Bootstrap CI, td-AUC@12/24/36m, ARR/NNT@24m, DCA
Also compute for top-5 combos for comparison.
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
from losses import (
    compute_all_metrics, bootstrap_c_index, compute_arr_nnt, compute_dca,
)
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci
from node_type_ablation import ALL_NODE_TYPES, compute_node_features, EVAL_COHORTS

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_CONFIGS = [
    {"model": "path_attn", "kg": "drkg",
     "nodes": ["ppi", "disease", "drug"], "lr": 1e-3, "do": 0.05, "hd": 32,
     "label": "DRKG best (do=0.05)"},
    {"model": "path_attn", "kg": "drkg",
     "nodes": ["ppi", "disease", "drug"], "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "DRKG baseline"},
    {"model": "path_attn", "kg": "monarch",
     "nodes": ["ppi", "disease", "phenotype", "anatomy"], "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "Monarch 4-node"},
    {"model": "path_attn", "kg": "primekg",
     "nodes": ["ppi", "disease"], "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "PrimeKG 2-node"},
    {"model": "sparse_path", "kg": "openbiolink",
     "nodes": ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"],
     "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "OpenBioLink ALL6"},
]


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


def train_model(model_name, kg_info, train_data, lr, dropout, hidden_dim, seed=42):
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
    return model, best


def main():
    genes = load_candidate_genes()
    n_genes = len(genes)
    kg_cache = {}
    all_results = []

    for cfg in TOP_CONFIGS:
        kg = cfg["kg"]
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
        extra_f = [node_feats[nt][0] for nt in cfg["nodes"]]
        extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in cfg["nodes"]]
        aug = augment_splits(splits, extra_f)
        info = build_combo_info(kg, extra_i, n_genes)

        print(f"\n{'='*60}")
        print(f"  {cfg['label']}")
        print(f"  {cfg['model']} x {kg} ({'+'.join(cfg['nodes'])})")
        print(f"  lr={cfg['lr']}, dropout={cfg['do']}, hd={cfg['hd']}")
        print(f"{'='*60}")

        model, val_ci = train_model(
            cfg["model"], info, aug["train"],
            cfg["lr"], cfg["do"], cfg["hd"])
        print(f"  val_ci = {val_ci:.4f}")

        # All-merged validation
        all_r, all_t, all_e = [], [], []
        cohort_metrics = {}
        for c in EVAL_COHORTS:
            if c not in aug:
                continue
            risk = get_risk(model, aug[c])
            t_np = aug[c]["time"].numpy()
            e_np = aug[c]["event"].numpy()
            all_r.append(risk)
            all_t.append(t_np)
            all_e.append(e_np)

            m = compute_all_metrics(risk, t_np, e_np)
            arr = compute_arr_nnt(risk, t_np, e_np, timepoints=(24.0,))
            cohort_metrics[c] = {**m, **arr}

        all_risk = np.concatenate(all_r)
        all_time = np.concatenate(all_t)
        all_event = np.concatenate(all_e)

        # All-merged metrics
        merged = compute_all_metrics(all_risk, all_time, all_event)
        boot = bootstrap_c_index(all_risk, all_time, all_event, n_boot=200)
        arr_merged = compute_arr_nnt(all_risk, all_time, all_event)
        dca = compute_dca(all_risk, all_time, all_event, t=24.0)
        mask_dca = (dca["thresholds"] >= 0.1) & (dca["thresholds"] <= 0.5)
        inb = float(np.trapz(dca["nb_model"][mask_dca], dca["thresholds"][mask_dca])) \
            if mask_dca.sum() > 1 else float("nan")

        n_sig = sum(1 for c in EVAL_COHORTS if c in cohort_metrics
                    and cohort_metrics[c]["p_value"] < 0.05)
        sigs = [c for c in EVAL_COHORTS if c in cohort_metrics
                and cohort_metrics[c]["p_value"] < 0.05]

        result = {
            "label": cfg["label"],
            "model": cfg["model"], "kg": kg,
            "nodes": cfg["nodes"],
            "lr": cfg["lr"], "dropout": cfg["do"], "hidden_dim": cfg["hd"],
            "val_ci": round(val_ci, 4),
            "n_sig": n_sig, "sigs": sigs,
            # All-merged
            "all_ci": round(merged["c_index"], 4),
            "all_hr": round(merged["hr"], 2),
            "all_p": merged["p_value"],
            "all_auc_12m": round(merged.get("auc_12m", 0), 4),
            "all_auc_24m": round(merged.get("auc_24m", 0), 4),
            "all_auc_36m": round(merged.get("auc_36m", 0), 4),
            "boot_ci": round(boot["boot_ci_mean"], 4),
            "boot_lo": round(boot["boot_ci_lo"], 4),
            "boot_hi": round(boot["boot_ci_hi"], 4),
            "arr_24m": round(arr_merged["arr_24m"], 4),
            "nnt_24m": round(arr_merged["nnt_24m"], 1)
                if not np.isnan(arr_merged["nnt_24m"]) and arr_merged["nnt_24m"] < 1000 else "N/A",
            "dca_inb": round(inb, 4),
            # Per-cohort
            "cohort_metrics": {
                c: {k: round(v, 4) if isinstance(v, float) and not np.isnan(v) else v
                     for k, v in m.items()}
                for c, m in cohort_metrics.items()
            },
        }
        all_results.append(result)

        print(f"  sig={n_sig}/11: {sigs}")
        print(f"  All-merged: CI={merged['c_index']:.4f}, HR={merged['hr']:.2f}, "
              f"p={merged['p_value']:.1e}")
        print(f"  Boot CI: {boot['boot_ci_mean']:.4f} [{boot['boot_ci_lo']:.4f}-{boot['boot_ci_hi']:.4f}]")
        print(f"  AUC: 12m={merged.get('auc_12m',0):.3f}, 24m={merged.get('auc_24m',0):.3f}, "
              f"36m={merged.get('auc_36m',0):.3f}")
        print(f"  ARR@24m={arr_merged['arr_24m']:+.3f}, NNT@24m={arr_merged['nnt_24m']:.1f}")
        print(f"  DCA INB={inb:.4f}")

    # Summary table
    print(f"\n{'='*120}")
    print("EXTENDED METRICS COMPARISON — TOP-5 CONFIGURATIONS")
    print(f"{'='*120}")
    print(f"{'Label':<25} {'Sig':>5} {'CI':>6} {'Boot [95%]':>22} "
          f"{'AUC12':>6} {'AUC24':>6} {'AUC36':>6} "
          f"{'HR':>5} {'ARR24':>7} {'NNT':>5} {'INB':>6}")
    print("-" * 120)
    for r in sorted(all_results, key=lambda x: (-x["n_sig"], -x["all_auc_24m"])):
        boot_s = f"{r['boot_ci']:.3f} [{r['boot_lo']:.3f}-{r['boot_hi']:.3f}]"
        nnt_s = str(r["nnt_24m"]) if r["nnt_24m"] != "N/A" else "N/A"
        print(f"{r['label']:<25} {r['n_sig']:>3}/11 {r['all_ci']:.3f} {boot_s:>22} "
              f"{r['all_auc_12m']:.3f} {r['all_auc_24m']:.3f} {r['all_auc_36m']:.3f} "
              f"{r['all_hr']:5.2f} {r['arr_24m']:+7.3f} {nnt_s:>5} {r['dca_inb']:.4f}")

    out = EXP_DIR / "top5_extended_metrics.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
