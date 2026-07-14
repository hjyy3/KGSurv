"""Comprehensive evaluation of all 21 experiments with new metrics."""
import sys
import warnings

sys.path.insert(0, ".")
import numpy as np
import torch
import json
from pathlib import Path

warnings.filterwarnings("ignore")

from data_interp import ALL_KGS, VALID_COHORTS, load_all_data
from models_interp import ALL_MODELS, create_model
from losses import (
    compute_all_metrics,
    bootstrap_c_index,
    compute_arr_nnt,
    compute_dca,
)

ROOT = Path(".").resolve().parent
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cpu")

cancer_map = {
    "Gandara": "NSCLC", "Ravi": "NSCLC",
    "Braun": "RCC", "CM214_JV101": "RCC",
    "Hugo": "Melanoma", "Liu": "Melanoma",
    "Riaz": "Melanoma", "Whijae": "Melanoma",
    "Mariathasan": "Bladder/UC", "PUSH": "Bladder/UC", "Snyder_UC": "Bladder/UC",
    "Pleasance": "Pan",
    # Miao is multi-cancer — NOT mapped at cohort level
    # Cancer-type merging uses cohort_preds per-sample, so Miao samples
    # that don't match any cancer_map cohort are simply excluded from
    # the cancer-type merged analysis (acceptable for this script)
}
cancer_order = ["NSCLC", "RCC", "Melanoma", "Bladder/UC", "GI"]

all_results = []

for model_name in ALL_MODELS:
    for kg_name in ALL_KGS:
        exp_dir = EXP_DIR / f"interp_{model_name}_{kg_name}"
        model_path = exp_dir / "model.pt"
        if not model_path.exists():
            continue

        all_data = load_all_data(kg_name)
        kg_info = all_data["kg_info"]
        model = create_model(model_name, kg_info)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        # Collect all validation predictions
        all_risk, all_time, all_event = [], [], []
        cohort_preds = {}
        for cohort in VALID_COHORTS:
            if cohort not in all_data["valid"]:
                continue
            cdata = all_data["valid"][cohort]
            with torch.no_grad():
                out = model(cdata["mut"], cdata["mask"], cdata["fmb"])
            r = out["log_risk"].numpy()
            t = cdata["time"].numpy()
            e = cdata["event"].numpy()
            cohort_preds[cohort] = (r, t, e)
            all_risk.append(r)
            all_time.append(t)
            all_event.append(e)

        all_risk = np.concatenate(all_risk)
        all_time = np.concatenate(all_time)
        all_event = np.concatenate(all_event)

        if np.std(all_risk) < 1e-8:
            continue

        # 1) Basic metrics
        basic = compute_all_metrics(all_risk, all_time, all_event)

        # 2) Bootstrap C-index
        boot = bootstrap_c_index(all_risk, all_time, all_event, n_boot=200)

        # 3) ARR / NNT
        arr_nnt = compute_arr_nnt(all_risk, all_time, all_event)

        # 4) DCA
        dca = compute_dca(all_risk, all_time, all_event, t=24.0)
        valid_nb = dca["nb_model"][np.isfinite(dca["nb_model"])]
        max_nb = float(np.max(valid_nb)) if len(valid_nb) > 0 else float("nan")
        mask_dca = (dca["thresholds"] >= 0.1) & (dca["thresholds"] <= 0.5)
        if mask_dca.sum() > 1:
            inb = float(np.trapz(dca["nb_model"][mask_dca], dca["thresholds"][mask_dca]))
        else:
            inb = float("nan")

        # 5) Per-cohort sig count
        n_sig = 0
        for cohort in VALID_COHORTS:
            if cohort not in cohort_preds:
                continue
            r, t, e = cohort_preds[cohort]
            m = compute_all_metrics(r, t, e)
            if m["p_value"] is not None and m["p_value"] < 0.05:
                n_sig += 1

        # 6) Cancer-type merged sig count
        n_sig_ct = 0
        for ct in cancer_order:
            ct_cohorts = [
                c for c in VALID_COHORTS
                if cancer_map.get(c) == ct and c in cohort_preds
            ]
            if not ct_cohorts:
                continue
            cr = np.concatenate([cohort_preds[c][0] for c in ct_cohorts])
            ct_ = np.concatenate([cohort_preds[c][1] for c in ct_cohorts])
            ce = np.concatenate([cohort_preds[c][2] for c in ct_cohorts])
            cm = compute_all_metrics(cr, ct_, ce)
            if cm["p_value"] is not None and cm["p_value"] < 0.05:
                n_sig_ct += 1

        row = {
            "model": model_name, "kg": kg_name,
            "n_sig_cohorts": n_sig, "n_sig_cancers": n_sig_ct,
            "all_ci": basic["c_index"], "all_hr": basic["hr"],
            "all_p": basic["p_value"],
            "auc_12m": basic["auc_12m"], "auc_24m": basic["auc_24m"],
            "auc_36m": basic["auc_36m"],
            "boot_ci": boot["boot_ci_mean"], "boot_lo": boot["boot_ci_lo"],
            "boot_hi": boot["boot_ci_hi"],
            "surv_high_24m": arr_nnt["surv_high_24m"],
            "surv_low_24m": arr_nnt["surv_low_24m"],
            "arr_24m": arr_nnt["arr_24m"], "nnt_24m": arr_nnt["nnt_24m"],
            "dca_max_nb": max_nb, "dca_inb": inb,
        }
        all_results.append(row)
        print(f"  {model_name:20s} x {kg_name:15s} done")

# Sort
all_results.sort(
    key=lambda x: (x["n_sig_cohorts"], x["n_sig_cancers"], x.get("dca_inb", 0) or 0),
    reverse=True,
)

# Print
print()
print("=" * 170)
print("COMPREHENSIVE EVALUATION (No Clinical Covariates, All-Validation-Merged)")
print("=" * 170)
hdr = (
    f"{'Rk':>3s} {'Model':20s} {'KG':15s} "
    f"{'Sig':>5s} {'CT':>4s} "
    f"{'CI':>6s} {'Boot CI [95%]':>22s} "
    f"{'AUC12':>6s} {'AUC24':>6s} {'AUC36':>6s} "
    f"{'HR':>5s} {'p-val':>10s} "
    f"{'S_hi24':>7s} {'S_lo24':>7s} {'ARR24':>7s} {'NNT24':>6s} "
    f"{'MaxNB':>7s} {'INB':>7s}"
)
print(hdr)
print("-" * 170)

for rank, r in enumerate(all_results, 1):
    boot_s = f"{r['boot_ci']:.3f} [{r['boot_lo']:.3f}-{r['boot_hi']:.3f}]"
    nnt_s = f"{r['nnt_24m']:.1f}" if not np.isnan(r["nnt_24m"]) and r["nnt_24m"] < 1000 else "N/A"
    p_s = f"{r['all_p']:.1e}" if r["all_p"] < 0.001 else f"{r['all_p']:.3f}"
    print(
        f"{rank:3d} {r['model']:20s} {r['kg']:15s} "
        f"{r['n_sig_cohorts']:>3d}/13 {r['n_sig_cancers']:>2d}/5 "
        f"{r['all_ci']:.3f} {boot_s:>22s} "
        f"{r['auc_12m']:.3f} {r['auc_24m']:.3f} {r['auc_36m']:.3f} "
        f"{r['all_hr']:5.2f} {p_s:>10s} "
        f"{r['surv_high_24m']:7.3f} {r['surv_low_24m']:7.3f} {r['arr_24m']:+7.3f} {nnt_s:>6s} "
        f"{r['dca_max_nb']:7.4f} {r['dca_inb']:7.4f}"
    )

# Save
out_path = str(ROOT / "output" / "experiments" / "comprehensive_eval_noclin.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {out_path}")
