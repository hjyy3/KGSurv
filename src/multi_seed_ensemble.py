"""Multi-seed ensemble evaluation for selected configurations.

For each config, train with 5 seeds → average risk scores → evaluate.
Seeds: [42, 123, 456, 789, 2024]
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
from losses import compute_all_metrics, bootstrap_c_index, compute_arr_nnt
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
    {"model": "path_attn", "kg": "primekg",
     "nodes": ["ppi", "disease"], "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "PrimeKG 2-node"},
    {"model": "sparse_path", "kg": "openbiolink",
     "nodes": ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"],
     "lr": 1e-3, "do": 0.1, "hd": 32,
     "label": "OpenBioLink ALL6"},
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


def evaluate_risk_scores(risk_dict, valid_data):
    """Evaluate pre-computed risk scores on all cohorts."""
    r = {}
    ext_cis, sigs = [], []
    for c in EVAL_COHORTS:
        if c not in risk_dict or c not in valid_data:
            continue
        risk = risk_dict[c]
        t = valid_data[c]["time"].numpy()
        e = valid_data[c]["event"].numpy()
        m = compute_all_metrics(risk, t, e)
        r[f"{c}_ci"] = round(m["c_index"], 4)
        r[f"{c}_p"] = m["p_value"]
        ext_cis.append(m["c_index"])
        if m["p_value"] < 0.05:
            sigs.append(c)
    r["ext_ci"] = round(np.mean(ext_cis), 4) if ext_cis else 0
    r["n_sig"] = len(sigs)
    r["sigs"] = sigs
    return r


def main():
    genes = load_candidate_genes()
    n_genes = len(genes)
    kg_cache = {}
    all_results = []

    total_runs = len(TOP_CONFIGS) * len(SEEDS)
    done = 0

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
        print(f"  {label} — 5-seed ensemble")
        print(f"{'='*60}")

        # Train 5 seeds, collect per-cohort risk scores
        seed_risks = {c: [] for c in EVAL_COHORTS if c in vd}
        seed_val_cis = []
        seed_results = []

        for seed in SEEDS:
            done += 1
            print(f"  [{done}/{total_runs}] seed={seed}", end=" → ")
            model, val_ci = train_single(
                cfg["model"], info, aug["train"],
                cfg["lr"], cfg["do"], cfg["hd"], seed)
            seed_val_cis.append(val_ci)

            # Collect risk scores per cohort
            cohort_risks = {}
            for c in EVAL_COHORTS:
                if c not in vd:
                    continue
                risk = get_risk(model, vd[c])
                seed_risks[c].append(risk)
                cohort_risks[c] = risk

            single_r = evaluate_risk_scores(cohort_risks, vd)
            print(f"val={val_ci:.4f} ext={single_r['ext_ci']:.4f} sig={single_r['n_sig']}/11")
            seed_results.append({
                "seed": seed, "val_ci": round(val_ci, 4), **single_r
            })

        # Ensemble: average risk scores across seeds
        ensemble_risks = {}
        for c in seed_risks:
            if seed_risks[c]:
                ensemble_risks[c] = np.mean(seed_risks[c], axis=0)

        ensemble_r = evaluate_risk_scores(ensemble_risks, vd)

        # All-merged ensemble metrics
        all_r = np.concatenate([ensemble_risks[c] for c in EVAL_COHORTS if c in ensemble_risks])
        all_t = np.concatenate([vd[c]["time"].numpy() for c in EVAL_COHORTS if c in vd])
        all_e = np.concatenate([vd[c]["event"].numpy() for c in EVAL_COHORTS if c in vd])
        merged = compute_all_metrics(all_r, all_t, all_e)
        boot = bootstrap_c_index(all_r, all_t, all_e, n_boot=200)
        arr = compute_arr_nnt(all_r, all_t, all_e)

        print(f"\n  --- {label} ENSEMBLE (5 seeds) ---")
        print(f"  Val CI: {np.mean(seed_val_cis):.4f} ± {np.std(seed_val_cis):.4f}")
        print(f"  Ext CI: {ensemble_r['ext_ci']:.4f}")
        print(f"  Sig: {ensemble_r['n_sig']}/11: {ensemble_r['sigs']}")
        print(f"  All-merged: CI={merged['c_index']:.4f}, HR={merged['hr']:.2f}, p={merged['p_value']:.1e}")
        print(f"  Boot CI: {boot['boot_ci_mean']:.4f} [{boot['boot_ci_lo']:.4f}-{boot['boot_ci_hi']:.4f}]")
        print(f"  AUC@24m={merged.get('auc_24m',0):.3f}, ARR@24m={arr['arr_24m']:+.3f}")

        # Compare individual seeds vs ensemble
        single_sigs = [r["n_sig"] for r in seed_results]
        print(f"\n  Individual seeds sig: {single_sigs} → Ensemble: {ensemble_r['n_sig']}")

        result = {
            "label": label, "model": cfg["model"], "kg": kg,
            "nodes": cfg["nodes"],
            "lr": cfg["lr"], "dropout": cfg["do"], "hidden_dim": cfg["hd"],
            "seeds": SEEDS,
            "val_ci_mean": round(np.mean(seed_val_cis), 4),
            "val_ci_std": round(np.std(seed_val_cis), 4),
            "individual_sigs": single_sigs,
            "ensemble_ext_ci": ensemble_r["ext_ci"],
            "ensemble_n_sig": ensemble_r["n_sig"],
            "ensemble_sigs": ensemble_r["sigs"],
            "all_ci": round(merged["c_index"], 4),
            "all_hr": round(merged["hr"], 2),
            "all_p": merged["p_value"],
            "all_auc_24m": round(merged.get("auc_24m", 0), 4),
            "boot_ci": round(boot["boot_ci_mean"], 4),
            "boot_lo": round(boot["boot_ci_lo"], 4),
            "boot_hi": round(boot["boot_ci_hi"], 4),
            "arr_24m": round(arr["arr_24m"], 4),
            "seed_results": seed_results,
            "per_cohort_ensemble": {c: round(ensemble_r.get(f"{c}_ci", 0), 4)
                                    for c in EVAL_COHORTS if f"{c}_ci" in ensemble_r},
        }
        all_results.append(result)

    # Final summary
    print(f"\n{'='*90}")
    print("MULTI-SEED ENSEMBLE SUMMARY")
    print(f"{'='*90}")
    print(f"{'Label':<22} {'Seeds sig':<20} {'Ens sig':>8} {'Ens CI':>7} "
          f"{'Boot [95%]':>22} {'AUC24':>6}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: (-x["ensemble_n_sig"], -x["all_ci"])):
        seeds_s = str(r["individual_sigs"])
        boot_s = f"{r['boot_ci']:.3f} [{r['boot_lo']:.3f}-{r['boot_hi']:.3f}]"
        print(f"{r['label']:<22} {seeds_s:<20} {r['ensemble_n_sig']:>6}/11 "
              f"{r['ensemble_ext_ci']:>7.4f} {boot_s:>22} {r['all_auc_24m']:.3f}")

    out = EXP_DIR / "multi_seed_ensemble.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
