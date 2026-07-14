"""Compute extended metrics for selected model configurations.

Metrics: C-index, Bootstrap CI, HR, p, td-AUC@12/24/36m, ARR/NNT,
         DCA net benefit, Brier Score, IBS, Calibration slope.
Compute for: single seed (seed=42) + 5-seed ensemble.
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
    brier_score_at, integrated_brier_score, calibration_stats,
)
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci
from node_type_ablation import ALL_NODE_TYPES, compute_node_features, EVAL_COHORTS

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456, 789, 2024]

TOP_CONFIGS = [
    {"model": "path_attn", "kg": "drkg",
     "nodes": ["ppi", "disease", "drug"], "lr": 1e-3, "do": 0.05, "hd": 32,
     "label": "DRKG best"},
    {"model": "path_attn", "kg": "monarch",
     "nodes": ["ppi", "disease", "phenotype", "anatomy"], "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "Monarch 4-node"},
    {"model": "sparse_path", "kg": "openbiolink",
     "nodes": ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"],
     "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "OpenBioLink ALL6"},
    {"model": "path_attn", "kg": "primekg",
     "nodes": ["ppi", "disease"], "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "PrimeKG 2-node"},
    {"model": "path_attn", "kg": "hetionet",
     "nodes": ["ppi", "anatomy", "regulatory"], "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "Hetionet 3-node"},
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


def train_single(model_name, kg_info, train_data, lr, dropout, hidden_dim, seed):
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


def full_metrics(risk, time, event, label=""):
    """Compute ALL metrics on merged data."""
    m = compute_all_metrics(risk, time, event)
    boot = bootstrap_c_index(risk, time, event, n_boot=200)
    arr = compute_arr_nnt(risk, time, event)
    ibs = integrated_brier_score(risk, time, event)
    cal = calibration_stats(risk, time, event, n_groups=5, t=24.0)
    dca = compute_dca(risk, time, event, t=24.0)
    mask_dca = (dca["thresholds"] >= 0.1) & (dca["thresholds"] <= 0.5)
    inb = float(np.trapezoid(dca["nb_model"][mask_dca], dca["thresholds"][mask_dca])) \
        if mask_dca.sum() > 1 else float("nan")

    # Per-cohort sig count
    return {
        "ci": round(m["c_index"], 4),
        "boot_ci": round(boot["boot_ci_mean"], 4),
        "boot_lo": round(boot["boot_ci_lo"], 4),
        "boot_hi": round(boot["boot_ci_hi"], 4),
        "hr": round(m["hr"], 2),
        "p": m["p_value"],
        "auc_12m": round(m.get("auc_12m", 0), 3),
        "auc_24m": round(m.get("auc_24m", 0), 3),
        "auc_36m": round(m.get("auc_36m", 0), 3),
        "arr_24m": round(arr["arr_24m"], 3),
        "nnt_24m": round(arr["nnt_24m"], 1) if arr["nnt_24m"] < 100 else "N/A",
        "ibs": round(ibs, 4) if not np.isnan(ibs) else "N/A",
        "cal_slope": cal["slope"],
        "cal_intercept": cal["intercept"],
        "dca_inb": round(inb, 4) if not np.isnan(inb) else "N/A",
    }


def main():
    genes = load_candidate_genes()
    n_genes = len(genes)
    kg_cache = {}
    all_results = []

    for cfg in TOP_CONFIGS:
        kg = cfg["kg"]
        label = cfg["label"]

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
        vd = {c: aug[c] for c in EVAL_COHORTS if c in aug}

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        # --- Single seed (seed=42) ---
        model, val_ci = train_single(cfg["model"], info, aug["train"],
                                      cfg["lr"], cfg["do"], cfg["hd"], 42)

        all_r, all_t, all_e = [], [], []
        cohort_results = {}
        for c in EVAL_COHORTS:
            if c not in vd:
                continue
            risk = get_risk(model, vd[c])
            t_np = vd[c]["time"].numpy()
            e_np = vd[c]["event"].numpy()
            all_r.append(risk)
            all_t.append(t_np)
            all_e.append(e_np)
            cm = compute_all_metrics(risk, t_np, e_np)
            cohort_results[c] = {"ci": round(cm["c_index"], 4), "p": cm["p_value"],
                                  "sig": cm["p_value"] < 0.05}

        merged_risk = np.concatenate(all_r)
        merged_time = np.concatenate(all_t)
        merged_event = np.concatenate(all_e)

        single_m = full_metrics(merged_risk, merged_time, merged_event)
        single_sigs = [c for c, v in cohort_results.items() if v["sig"]]
        print(f"  Single seed: {len(single_sigs)}/11 sig, CI={single_m['ci']}, "
              f"IBS={single_m['ibs']}, Cal slope={single_m['cal_slope']}")

        # --- 5-seed ensemble ---
        seed_risks = {c: [] for c in EVAL_COHORTS if c in vd}
        for seed in SEEDS:
            mdl, _ = train_single(cfg["model"], info, aug["train"],
                                   cfg["lr"], cfg["do"], cfg["hd"], seed)
            for c in seed_risks:
                seed_risks[c].append(get_risk(mdl, vd[c]))

        ens_all_r, ens_all_t, ens_all_e = [], [], []
        ens_cohort = {}
        ens_sigs = []
        for c in EVAL_COHORTS:
            if c not in seed_risks or not seed_risks[c]:
                continue
            ens_risk = np.mean(seed_risks[c], axis=0)
            t_np = vd[c]["time"].numpy()
            e_np = vd[c]["event"].numpy()
            ens_all_r.append(ens_risk)
            ens_all_t.append(t_np)
            ens_all_e.append(e_np)
            cm = compute_all_metrics(ens_risk, t_np, e_np)
            ens_cohort[c] = {"ci": round(cm["c_index"], 4), "p": cm["p_value"]}
            if cm["p_value"] < 0.05:
                ens_sigs.append(c)

        ens_merged_risk = np.concatenate(ens_all_r)
        ens_merged_time = np.concatenate(ens_all_t)
        ens_merged_event = np.concatenate(ens_all_e)
        ens_m = full_metrics(ens_merged_risk, ens_merged_time, ens_merged_event)
        print(f"  Ensemble:    {len(ens_sigs)}/11 sig, CI={ens_m['ci']}, "
              f"IBS={ens_m['ibs']}, Cal slope={ens_m['cal_slope']}")

        all_results.append({
            "label": label, "model": cfg["model"], "kg": kg,
            "nodes": cfg["nodes"],
            "single_seed": {
                "n_sig": len(single_sigs), "sigs": single_sigs,
                "val_ci": round(val_ci, 4),
                **single_m, "cohorts": cohort_results,
            },
            "ensemble": {
                "n_sig": len(ens_sigs), "sigs": ens_sigs,
                **ens_m, "cohorts": ens_cohort,
            },
        })

    # Summary table
    print(f"\n{'='*130}")
    print("FULL METRICS COMPARISON (All-Merged Validation)")
    print(f"{'='*130}")
    print(f"{'Label':<20} {'Mode':<8} {'Sig':>5} {'CI':>6} {'Boot [95%]':>22} "
          f"{'AUC24':>6} {'HR':>5} {'IBS':>6} {'Cal.S':>6} {'ARR24':>7} {'NNT':>5} {'INB':>6}")
    print("-" * 130)
    for r in all_results:
        for mode, m in [("single", r["single_seed"]), ("ens(5)", r["ensemble"])]:
            boot_s = f"{m['boot_ci']:.3f} [{m['boot_lo']:.3f}-{m['boot_hi']:.3f}]"
            ibs_s = f"{m['ibs']:.4f}" if m['ibs'] != "N/A" else "N/A"
            nnt_s = str(m['nnt_24m']) if m['nnt_24m'] != "N/A" else "N/A"
            inb_s = f"{m['dca_inb']:.4f}" if m['dca_inb'] != "N/A" else "N/A"
            print(f"{r['label']:<20} {mode:<8} {m['n_sig']:>3}/11 {m['ci']:.3f} {boot_s:>22} "
                  f"{m['auc_24m']:.3f} {m['hr']:5.2f} {ibs_s:>6} {m['cal_slope']:>6.2f} "
                  f"{m['arr_24m']:+7.3f} {nnt_s:>5} {inb_s:>6}")

    out = EXP_DIR / "full_extended_metrics.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
