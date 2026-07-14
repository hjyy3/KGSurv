"""Compare our model with Long et al. 11-gene mutation signature.

Long signature:
  Risk = exp[sum(coef_i * mut_i) - intercept]
  11 genes: BRAF, PAK7, PTPRD, PTPRT, ROS1, SETD2, TET1, VHL, FAM46C, RNF43, ZFHX3
  Cutoff = 1.07 (high risk if risk_score >= 1.07)
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
from losses import compute_all_metrics, bootstrap_c_index, compute_arr_nnt, integrated_brier_score
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci
from node_type_ablation import ALL_NODE_TYPES, compute_node_features, EVAL_COHORTS
from lifelines.statistics import logrank_test

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Long et al. 11-gene signature
# =====================================================================

LONG_GENES = ["BRAF", "PAK7", "PTPRD", "PTPRT", "ROS1", "SETD2",
              "TET1", "VHL", "FAM46C", "RNF43", "ZFHX3"]

LONG_COEFS = {
    "BRAF":   -0.4885006,
    "PAK7":   -0.2618274,
    "PTPRD":  -0.2610592,
    "PTPRT":  -0.2404202,   # negative (all 11 genes have negative coefficients)
    "ROS1":   -0.2321493,
    "SETD2":  -0.2759073,
    "TET1":   -0.8026092,
    "VHL":    -1.0449158,
    "FAM46C": -1.7929573,
    "RNF43":  -0.7964559,   # negative
    "ZFHX3":  -0.3821696,
}
LONG_INTERCEPT = -0.3283004   # stored as negative; formula applies −(−0.3283) = +0.3283
LONG_CUTOFF = 1.07


def compute_long_risk(mut_df, mask_df):
    """Compute Long signature risk score for each patient.

    Paper formula: exp[sum(coef_i * mut_i) - (-0.3283004)]
                 = exp[linear + 0.3283004]
                 = exp[linear - LONG_INTERCEPT]  (LONG_INTERCEPT stored as -0.3283004)

    Unmutated baseline: exp(+0.3283) ≈ 1.39 > 1.07 → HIGH risk by default?
    All coefs negative, so any mutation REDUCES risk score below baseline.
    Uses mut * mask to handle panel coverage.
    """
    all_genes = list(mut_df.columns)
    linear = np.zeros(len(mut_df))

    for gene, coef in LONG_COEFS.items():
        if gene in all_genes:
            idx = all_genes.index(gene)
            mut_val = mut_df.iloc[:, idx].values * mask_df.iloc[:, idx].values
            linear += coef * mut_val

    risk_scores = np.exp(linear - LONG_INTERCEPT)   # correct: -(-0.3283004) = +0.3283004
    return risk_scores


def compute_long_binary(risk_scores):
    """Classify as high/low risk using Long cutoff."""
    return risk_scores >= LONG_CUTOFF


# =====================================================================
# Our model helper
# =====================================================================

def load_splits_with_nodes(kg_name, node_types):
    kg_feat = KG_DIR / kg_name
    genes = load_candidate_genes()
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
            "mut_df": mut.loc[common],
            "mask_df": mask.loc[common],
        }
        raw[sn] = (mut.loc[common].values.astype(np.float32),
                    mask.loc[common].values.astype(np.float32))
    # Compute node features
    nf = {}
    for nt, (mode, edict) in ALL_NODE_TYPES.items():
        res = compute_node_features(kg_name, nt, mode, edict, genes, raw)
        if res:
            feats, tnames, n_info = res
            nf[nt] = (feats, tnames, mode)
    # Augment
    for sn in list(splits.keys()):
        extras = []
        for nt in node_types:
            if nt in nf and sn in nf[nt][0]:
                extras.append(torch.tensor(nf[nt][0][sn], dtype=torch.float32))
        if extras:
            splits[sn]["fmb"] = torch.cat([splits[sn]["fmb"]] + extras, dim=1)
    return splits, nf


def build_combo_info(kg_name, node_types, nf, n_genes):
    base = build_kg_group_info(kg_name)
    groups = list(base.group_names)
    terms = list(base.term_names)
    masks = list(base.gene_term_mask)
    slices = list(base.fmb_slices)
    offset = slices[-1][1] if slices else 0
    total = base.n_total_terms
    for nt in node_types:
        if nt not in nf:
            continue
        tnames, mode = nf[nt][1], nf[nt][2]
        groups.append(f"x_{nt}")
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


def eval_on_cohorts(risk_fn, splits, label=""):
    """Evaluate a risk function across all cohorts."""
    results = {}
    all_r, all_t, all_e = [], [], []
    for c in EVAL_COHORTS:
        if c not in splits:
            continue
        risk = risk_fn(splits[c])
        time = splits[c]["time"].numpy()
        event = splits[c]["event"].numpy()
        all_r.append(risk)
        all_t.append(time)
        all_e.append(event)
        m = compute_all_metrics(risk, time, event)
        results[c] = {"ci": round(m["c_index"], 4), "p": m["p_value"],
                       "hr": round(m["hr"], 2)}

    merged_r = np.concatenate(all_r)
    merged_t = np.concatenate(all_t)
    merged_e = np.concatenate(all_e)
    merged_m = compute_all_metrics(merged_r, merged_t, merged_e)
    boot = bootstrap_c_index(merged_r, merged_t, merged_e, n_boot=200)
    ibs = integrated_brier_score(merged_r, merged_t, merged_e)

    n_sig = sum(1 for v in results.values() if v["p"] < 0.05)
    sigs = [c for c, v in results.items() if v["p"] < 0.05]

    return {
        "label": label,
        "n_sig": n_sig, "sigs": sigs,
        "merged_ci": round(merged_m["c_index"], 4),
        "merged_hr": round(merged_m["hr"], 2),
        "merged_p": merged_m["p_value"],
        "boot_ci": round(boot["boot_ci_mean"], 4),
        "boot_lo": round(boot["boot_ci_lo"], 4),
        "boot_hi": round(boot["boot_ci_hi"], 4),
        "auc_12m": round(merged_m.get("auc_12m", 0), 3),
        "auc_24m": round(merged_m.get("auc_24m", 0), 3),
        "auc_36m": round(merged_m.get("auc_36m", 0), 3),
        "ibs": round(ibs, 4) if not np.isnan(ibs) else "N/A",
        "per_cohort": results,
    }


def main():
    genes = load_candidate_genes()
    n_genes = len(genes)
    all_results = []

    # Our best configs
    OUR_CONFIGS = [
        ("path_attn", "drkg", ["ppi", "disease", "drug"], 0.05, "Our: DRKG (best single)"),
        ("sparse_path", "openbiolink",
         ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"],
         0.1, "Our: OpenBioLink ALL6"),
    ]

    # Load data for Long signature (use any KG's splits for mut/mask)
    print("Loading data...")
    splits_drkg, nf_drkg = load_splits_with_nodes("drkg", ["ppi", "disease", "drug"])

    # --- Long Signature ---
    print("\n" + "=" * 60)
    print("  Long et al. 11-gene signature")
    print("=" * 60)

    def long_risk_fn(data):
        return compute_long_risk(data["mut_df"], data["mask_df"])

    long_result = eval_on_cohorts(long_risk_fn, splits_drkg, "Long 11-gene")
    print(f"  Sig: {long_result['n_sig']}/11: {long_result['sigs']}")
    print(f"  Merged: CI={long_result['merged_ci']}, HR={long_result['merged_hr']}, "
          f"p={long_result['merged_p']:.1e}")
    print(f"  Boot: {long_result['boot_ci']} [{long_result['boot_lo']}-{long_result['boot_hi']}]")
    print(f"  AUC@24m={long_result['auc_24m']}, IBS={long_result['ibs']}")
    all_results.append(long_result)

    # --- Our models ---
    for model_name, kg, node_types, dropout, label in OUR_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        if kg == "drkg":
            splits, nf = splits_drkg, nf_drkg
        else:
            splits, nf = load_splits_with_nodes(kg, node_types)

        info = build_combo_info(kg, node_types, nf, n_genes)
        _seed_everything(42)
        tr, va = _split_data(splits["train"], 0.8, 42)
        model = create_model(model_name, info, hidden_dim=32, dropout=dropout)
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

        def our_risk_fn(data, mdl=model):
            return get_risk(mdl, data)

        our_result = eval_on_cohorts(our_risk_fn, splits, label)
        print(f"  Sig: {our_result['n_sig']}/11: {our_result['sigs']}")
        print(f"  Merged: CI={our_result['merged_ci']}, HR={our_result['merged_hr']}, "
              f"p={our_result['merged_p']:.1e}")
        print(f"  Boot: {our_result['boot_ci']} [{our_result['boot_lo']}-{our_result['boot_hi']}]")
        print(f"  AUC@24m={our_result['auc_24m']}, IBS={our_result['ibs']}")
        all_results.append(our_result)

    # Comparison table
    print(f"\n{'='*100}")
    print("HEAD-TO-HEAD COMPARISON: Our Model vs Long 11-Gene Signature")
    print(f"{'='*100}")
    print(f"{'Model':<30} {'Sig':>5} {'CI':>6} {'Boot [95%]':>22} "
          f"{'AUC24':>6} {'HR':>5} {'IBS':>6}")
    print("-" * 100)
    for r in all_results:
        boot_s = f"{r['boot_ci']:.3f} [{r['boot_lo']:.3f}-{r['boot_hi']:.3f}]"
        ibs_s = f"{r['ibs']}" if r['ibs'] != "N/A" else "N/A"
        print(f"{r['label']:<30} {r['n_sig']:>3}/11 {r['merged_ci']:.3f} {boot_s:>22} "
              f"{r['auc_24m']:.3f} {r['merged_hr']:5.2f} {ibs_s:>6}")

    # Per-cohort comparison
    print(f"\n{'='*100}")
    print("PER-COHORT COMPARISON")
    print(f"{'='*100}")
    print(f"{'Cohort':<15}", end="")
    for r in all_results:
        print(f" {r['label'][:20]:>22}", end="")
    print()
    print("-" * 100)
    for c in EVAL_COHORTS:
        print(f"{c:<15}", end="")
        for r in all_results:
            if c in r["per_cohort"]:
                pc = r["per_cohort"][c]
                sig = "*" if pc["p"] < 0.05 else " "
                print(f" CI={pc['ci']:.3f} p={pc['p']:.3f}{sig}", end="")
            else:
                print(f" {'N/A':>22}", end="")
        print()

    out = EXP_DIR / "long_signature_comparison.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
