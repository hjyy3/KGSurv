"""Filtered evaluation with cohort, cancer-type, and treatment stratification."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from data_interp import ALL_KGS, VALID_COHORTS, build_kg_group_info, load_split_data
from losses import (
    bootstrap_c_index,
    compute_all_metrics,
    compute_arr_nnt,
    compute_dca,
)
from models_interp import ALL_MODELS, create_model

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"
VSRC = ROOT / "source" / "input_data" / "valid"

# --- Configuration ---
EXCLUDED_COHORTS = {"Snyder_UC", "Whijae"}  # too small (n=25, n=19)
FILTERED_COHORTS = [c for c in VALID_COHORTS if c not in EXCLUDED_COHORTS]

# Per-sample cancer type normalization (NOT cohort-level — Miao is multi-cancer)
CANCER_NORM = {
    "RCC": "RCC", "Melanoma": "Melanoma",
    "Lung": "NSCLC", "Lung Adenocarcinoma": "NSCLC",
    "Non-Small Cell Lung Cancer": "NSCLC", "Non-small cell lung cancer": "NSCLC",
    "NSCLC": "NSCLC", "Lung Squamous Cell Carcinoma": "NSCLC",
    "Sarcomatoid Carcinoma of the Lung": "NSCLC",
    "Bladder": "Bladder/UC", "UC": "Bladder/UC", "Urothelial cancer": "Bladder/UC",
    "HNSCC": "HNSCC",
    "Oropharynx Squamous Cell Carcinoma": "HNSCC",
    "Head and Neck Mucosal Melanoma": "HNSCC",
    "GC": "GI", "EC": "GI", "CRC": "GI",
    "Colorectal Adenocarcinoma": "GI", "Stomach Adenocarcinoma": "GI",
    "Adenocarcinoma of the Gastroesophageal Junction": "GI",
}
CANCER_ORDER = ["NSCLC", "RCC", "Melanoma", "Bladder/UC", "GI"]

# Drug classification
PD1_DRUGS = {
    "Anti-PD-1/PD-L1", "anti-PD-1/anti-PD-L1", "PD(L)1",
}
COMBO_DRUGS = {
    "Nivo+Ipi", "Avelumab + Axitinib", "Anti-PD-1/PD-L1+CTLA-4",
    "anti-CTLA-4 + anti-PD-1/PD-L1", "PD(L)1 + CTLA4", "PD(L)1 + Other",
}

# Load per-sample drug type for drug stratification
sample_drug = {}
sample_cancer = {}
for cohort in FILTERED_COHORTS:
    clin_path = VSRC / f"clin_{cohort}.csv"
    if not clin_path.exists():
        continue
    clin = pd.read_csv(clin_path)
    id_col = clin.columns[0]
    for _, row in clin.iterrows():
        sid = str(row[id_col])
        drug = str(row.get("Drug_type", "")).strip()
        if drug in PD1_DRUGS:
            sample_drug[sid] = "PD1"
        elif drug in COMBO_DRUGS:
            sample_drug[sid] = "Combo"
        else:
            sample_drug[sid] = "Other"
        # Cancer type from per-sample Cancer_type column (Miao is multi-cancer)
        ct_raw = str(row.get("Cancer_type", "")).strip()
        sample_cancer[sid] = CANCER_NORM.get(ct_raw, "Other")


def _load_preds(model_name, kg_name):
    """Load model and compute predictions for filtered cohorts."""
    exp_dir = EXP_DIR / f"interp_{model_name}_{kg_name}"
    model_path = exp_dir / "model.pt"
    if not model_path.exists():
        return None

    kg_info = build_kg_group_info(kg_name)
    model = create_model(model_name, kg_info)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    preds = {}
    for cohort in FILTERED_COHORTS:
        try:
            data = load_split_data(kg_name, cohort)
        except FileNotFoundError:
            continue
        with torch.no_grad():
            out = model(data["mut"], data["mask"], data["fmb"])
        preds[cohort] = {
            "risk": out["log_risk"].numpy(),
            "time": data["time"].numpy(),
            "event": data["event"].numpy(),
            "sample_ids": data["sample_ids"],
        }
    return preds


def _eval_group(risk, time, event):
    """Compute all metrics for a risk/time/event group."""
    if len(risk) < 10 or np.std(risk) < 1e-8:
        return None
    basic = compute_all_metrics(risk, time, event)
    boot = bootstrap_c_index(risk, time, event, n_boot=200)
    arr = compute_arr_nnt(risk, time, event)
    dca = compute_dca(risk, time, event, t=24.0)
    valid_nb = dca["nb_model"][np.isfinite(dca["nb_model"])]
    mask_dca = (dca["thresholds"] >= 0.1) & (dca["thresholds"] <= 0.5)
    inb = (
        float(np.trapz(dca["nb_model"][mask_dca], dca["thresholds"][mask_dca]))
        if mask_dca.sum() > 1
        else float("nan")
    )
    return {
        **basic,
        **boot,
        "arr_24m": arr["arr_24m"],
        "nnt_24m": arr["nnt_24m"],
        "surv_high_24m": arr["surv_high_24m"],
        "surv_low_24m": arr["surv_low_24m"],
        "dca_inb": inb,
    }


def main():
    all_rows = []

    for model_name in ALL_MODELS:
        for kg_name in ALL_KGS:
            preds = _load_preds(model_name, kg_name)
            if preds is None:
                continue

            row = {"model": model_name, "kg": kg_name}

            # --- Per-cohort metrics ---
            n_sig = 0
            for cohort in FILTERED_COHORTS:
                if cohort not in preds:
                    continue
                d = preds[cohort]
                m = compute_all_metrics(d["risk"], d["time"], d["event"])
                row[f"{cohort}_ci"] = m["c_index"]
                row[f"{cohort}_p"] = m["p_value"]
                if m["p_value"] is not None and m["p_value"] < 0.05:
                    n_sig += 1
            row["n_sig_cohorts"] = n_sig

            # --- Cancer-type merged (per-sample mapping, NOT cohort-level) ---
            n_sig_ct = 0
            # Collect per-sample predictions with cancer type
            ct_risk = {ct: [] for ct in CANCER_ORDER}
            ct_time = {ct: [] for ct in CANCER_ORDER}
            ct_event = {ct: [] for ct in CANCER_ORDER}
            for cohort in FILTERED_COHORTS:
                if cohort not in preds:
                    continue
                d = preds[cohort]
                for i, sid in enumerate(d["sample_ids"]):
                    ct = sample_cancer.get(str(sid), "Other")
                    if ct in ct_risk:
                        ct_risk[ct].append(d["risk"][i])
                        ct_time[ct].append(d["time"][i])
                        ct_event[ct].append(d["event"][i])
            for ct in CANCER_ORDER:
                if len(ct_risk[ct]) < 10:
                    continue
                cr = np.array(ct_risk[ct])
                ct_t = np.array(ct_time[ct])
                ce = np.array(ct_event[ct])
                cm = compute_all_metrics(cr, ct_t, ce)
                row[f"{ct}_merged_ci"] = cm["c_index"]
                row[f"{ct}_merged_p"] = cm["p_value"]
                row[f"{ct}_merged_n"] = len(cr)
                if cm["p_value"] is not None and cm["p_value"] < 0.05:
                    n_sig_ct += 1
            row["n_sig_cancers"] = n_sig_ct

            # --- All-merged (filtered) ---
            all_r = np.concatenate([preds[c]["risk"] for c in FILTERED_COHORTS if c in preds])
            all_t = np.concatenate([preds[c]["time"] for c in FILTERED_COHORTS if c in preds])
            all_e = np.concatenate([preds[c]["event"] for c in FILTERED_COHORTS if c in preds])

            if np.std(all_r) < 1e-8:
                continue

            am = _eval_group(all_r, all_t, all_e)
            if am:
                for k, v in am.items():
                    row[f"all_{k}"] = v

            # --- Drug stratification: PD1-only ---
            pd1_r, pd1_t, pd1_e = [], [], []
            for cohort in FILTERED_COHORTS:
                if cohort not in preds:
                    continue
                d = preds[cohort]
                for i, sid in enumerate(d["sample_ids"]):
                    if sample_drug.get(str(sid)) == "PD1":
                        pd1_r.append(d["risk"][i])
                        pd1_t.append(d["time"][i])
                        pd1_e.append(d["event"][i])
            if len(pd1_r) > 50:
                pd1_m = _eval_group(
                    np.array(pd1_r), np.array(pd1_t), np.array(pd1_e)
                )
                if pd1_m:
                    for k, v in pd1_m.items():
                        row[f"pd1_{k}"] = v

            all_rows.append(row)
            print(f"  {model_name:20s} x {kg_name:15s} done")

    # Sort
    all_rows.sort(
        key=lambda x: (
            x.get("n_sig_cohorts", 0),
            x.get("n_sig_cancers", 0),
            x.get("all_auc_24m", 0) or 0,
        ),
        reverse=True,
    )

    # --- Print leaderboard ---
    print()
    print("=" * 180)
    print("FILTERED EVALUATION (excl Snyder_UC/Whijae, CM214=RCC, 11 cohorts)")
    print("=" * 180)
    hdr = (
        f"{'Rk':>3s} {'Model':20s} {'KG':15s} "
        f"{'Sig':>5s} {'CT':>4s} "
        f"{'CI':>6s} {'Boot [95%]':>22s} "
        f"{'AUC12':>6s} {'AUC24':>6s} {'AUC36':>6s} "
        f"{'ARR24':>7s} {'NNT24':>6s} {'INB':>6s} "
        f"| {'PD1_CI':>7s} {'PD1_AUC24':>9s} {'PD1_ARR24':>9s}"
    )
    print(hdr)
    print("-" * 180)

    for rank, r in enumerate(all_rows, 1):
        boot_s = f"{r.get('all_boot_ci_mean',0):.3f} [{r.get('all_boot_ci_lo',0):.3f}-{r.get('all_boot_ci_hi',0):.3f}]"
        nnt = r.get("all_nnt_24m", float("nan"))
        nnt_s = f"{nnt:.1f}" if not np.isnan(nnt) and nnt < 1000 else "N/A"
        pd1_ci = r.get("pd1_c_index", float("nan"))
        pd1_auc = r.get("pd1_auc_24m", float("nan"))
        pd1_arr = r.get("pd1_arr_24m", float("nan"))
        print(
            f"{rank:3d} {r['model']:20s} {r['kg']:15s} "
            f"{r['n_sig_cohorts']:>3d}/11 {r['n_sig_cancers']:>2d}/5 "
            f"{r.get('all_c_index',0):.3f} {boot_s:>22s} "
            f"{r.get('all_auc_12m',0):.3f} {r.get('all_auc_24m',0):.3f} {r.get('all_auc_36m',0):.3f} "
            f"{r.get('all_arr_24m',0):+7.3f} {nnt_s:>6s} {r.get('all_dca_inb',0):6.4f} "
            f"| {pd1_ci:7.3f} {pd1_auc:9.3f} {pd1_arr:+9.3f}"
        )

    # --- Cancer-type merged detail for top 5 ---
    print()
    print("CANCER-TYPE MERGED (corrected: CM214=RCC)")
    print("=" * 120)
    for rank, r in enumerate(all_rows[:5], 1):
        parts = []
        for ct in CANCER_ORDER:
            ci = r.get(f"{ct}_merged_ci", float("nan"))
            p = r.get(f"{ct}_merged_p", float("nan"))
            sig = "*" if (not np.isnan(p) and p < 0.05) else " "
            parts.append(f"{ct}={ci:.3f}(p={p:.1e}){sig}")
        print(f"#{rank} {r['model']:20s} x {r['kg']:15s} sig={r['n_sig_cancers']}/5 | {', '.join(parts)}")

    # Save JSON
    out_path = str(ROOT / "output" / "experiments" / "filtered_eval_phase1.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
