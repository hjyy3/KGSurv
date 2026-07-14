"""Cancer-type and drug heterogeneity interaction analysis.

Uses DRKG best model (PathAttnSurv FMB+PPI+Disease+Drug, seed=42) to predict
risk scores on all external cohorts. Then fits Cox models with interaction
terms:

  1. Surv(time, event) ~ risk + cancer_type + risk:cancer_type
  2. Surv(time, event) ~ risk + drug_class + risk:drug_class

If interaction p < 0.05, the biomarker effect is heterogeneous across
subgroups — explains why Mariathasan (Bladder/Atezolizumab) and Snyder_UC
(Bladder/CTLA-4) fail when NSCLC-trained signal transfers poorly.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from lifelines.statistics import logrank_test

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kg_features import load_candidate_genes
from models_interp import create_model
from multi_node_extended import (
    augment_splits, build_combo_info, load_base_splits, train_and_eval,
    EVAL_COHORTS, MODEL,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANCER_COLS = {
    "NSCLC":    "cov_cancer_Non-Small Cell Lung Cancer",
    "Melanoma": "cov_cancer_Melanoma",
    "Bladder":  "cov_cancer_Bladder Cancer",
    "RCC":      "cov_cancer_Renal Cell Carcinoma",
    "HNSCC":    "cov_cancer_Head and Neck Cancer",
    "GI":       "cov_cancer_Esophagogastric Cancer",
    "CRC":      "cov_cancer_Colorectal Cancer",
    "Glioma":   "cov_cancer_Glioma",
    "CUP":      "cov_cancer_Cancer of Unknown Primary",
    "HCC":      "cov_cancer_Hepatobiliary Cancer",
}

DRUG_COLS = {
    "PD1/PDL1": "cov_drug_PD-1/PDL-1",
    "Combo":    "cov_drug_Combo",
    "CTLA4":    "cov_drug_CTLA4",
}


def label_from_onehot(clin: pd.DataFrame, col_map: dict[str, str],
                       other: str = "Other") -> pd.Series:
    """Convert one-hot columns to categorical label."""
    out = pd.Series(other, index=clin.index, dtype=object)
    for label, col in col_map.items():
        if col in clin.columns:
            out[clin[col] > 0.5] = label
    return out


@torch.no_grad()
def get_risk(model, data):
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def train_drkg_best():
    """Train DRKG best config (PathAttnSurv FMB+PPI+Disease+Drug)."""
    print("Training DRKG best config...")
    genes = load_candidate_genes()
    node_types = ["ppi", "disease", "drug"]
    splits, raw = load_base_splits("drkg")
    nf = {}
    for nt in node_types:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features("drkg", nt, mode, edict, genes, raw)
        if res:
            feats, tnames, m = res
            nf[nt] = (feats, tnames, mode)
    extra_f = [nf[nt][0] for nt in node_types]
    extra_i = [(nf[nt][1], nf[nt][2], f"x_{nt}") for nt in node_types]
    aug = augment_splits(splits, extra_f)
    info = build_combo_info("drkg", extra_i, len(genes))

    _seed_everything(42)
    tr, va = _split_data(aug["train"], 0.8, 42)
    model = create_model(MODEL, info, hidden_dim=32, dropout=0.05)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                     patience=7, factor=0.5, min_lr=1e-6)
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
    print(f"  val_ci={best:.4f}")
    return model, aug


def collect_predictions(model, aug):
    """Predict risk on all valid cohorts and return a pooled DataFrame."""
    rows = []
    for c in EVAL_COHORTS:
        if c not in aug:
            continue
        risk = get_risk(model, aug[c])
        clin = pd.read_csv(PROC / f"valid_{c}_clin.csv", index_col=0)
        ids = aug[c]["sample_ids"]
        clin = clin.loc[ids]
        cancer = label_from_onehot(clin, CANCER_COLS)
        drug = label_from_onehot(clin, DRUG_COLS)
        for i, sid in enumerate(ids):
            rows.append({
                "sample_id": sid, "cohort": c,
                "risk": float(risk[i]),
                "time": float(aug[c]["time"][i].item()),
                "event": int(aug[c]["event"][i].item()),
                "cancer_type": cancer.iloc[i],
                "drug_class": drug.iloc[i],
            })
    return pd.DataFrame(rows)


def fit_interaction_cox(df: pd.DataFrame, covariate: str,
                         min_samples: int = 25) -> dict:
    """Fit Cox with risk * covariate interaction.

    Returns p-values for interaction terms and subgroup-specific HRs.
    """
    # Subset to categories with enough samples
    vc = df[covariate].value_counts()
    keep_cats = vc[vc >= min_samples].index.tolist()
    df = df[df[covariate].isin(keep_cats)].copy()
    if len(keep_cats) < 2:
        return {"status": "skip", "reason": "need ≥2 cats with ≥25 samples"}

    # Median split for risk (binary "high risk")
    df["risk_high"] = (df["risk"] >= df["risk"].median()).astype(int)

    # One-hot encode covariate (reference = largest category)
    ref = vc.idxmax()
    cats = [c for c in keep_cats if c != ref]
    for c in cats:
        df[f"{covariate}_{c}"] = (df[covariate] == c).astype(int)
        df[f"risk_x_{c}"] = df["risk_high"] * df[f"{covariate}_{c}"]

    cols = (["risk_high", "time", "event"]
            + [f"{covariate}_{c}" for c in cats]
            + [f"risk_x_{c}" for c in cats])
    data = df[cols].copy()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    try:
        cph = CoxPHFitter(penalizer=1e-3)
        cph.fit(data, duration_col="time", event_col="event", show_progress=False)
    except ConvergenceError as e:
        return {"status": "error", "error": str(e)}

    summary = cph.summary
    interaction_ps = {}
    main_hr = float(summary.loc["risk_high", "exp(coef)"])
    main_p = float(summary.loc["risk_high", "p"])
    for c in cats:
        row = summary.loc[f"risk_x_{c}"]
        interaction_ps[c] = {
            "hr_ratio": float(row["exp(coef)"]),  # e^beta_int: how HR shifts vs ref
            "p": float(row["p"]),
            "ci_lo": float(row["exp(coef) lower 95%"]),
            "ci_hi": float(row["exp(coef) upper 95%"]),
        }

    # Per-subgroup log-rank test
    per_subgroup = {}
    for cat in keep_cats:
        sub = df[df[covariate] == cat]
        if len(sub) < min_samples or sub["event"].sum() < 5:
            per_subgroup[cat] = {"n": len(sub), "events": int(sub["event"].sum()),
                                  "p": None, "hr": None}
            continue
        hi = sub[sub["risk_high"] == 1]
        lo = sub[sub["risk_high"] == 0]
        if len(hi) < 5 or len(lo) < 5:
            per_subgroup[cat] = {"n": len(sub), "events": int(sub["event"].sum()),
                                  "p": None, "hr": None}
            continue
        lr = logrank_test(lo["time"], hi["time"], lo["event"], hi["event"])
        # HR per-subgroup via univariate Cox
        try:
            sub_cph = CoxPHFitter(penalizer=1e-3)
            sub_cph.fit(sub[["risk_high", "time", "event"]],
                        duration_col="time", event_col="event", show_progress=False)
            hr = float(sub_cph.summary.loc["risk_high", "exp(coef)"])
            hr_p = float(sub_cph.summary.loc["risk_high", "p"])
        except (ConvergenceError, ValueError, np.linalg.LinAlgError):
            hr, hr_p = None, None
        per_subgroup[cat] = {
            "n": len(sub), "events": int(sub["event"].sum()),
            "logrank_p": float(lr.p_value), "hr": hr, "hr_p": hr_p,
        }

    return {
        "status": "ok", "n_total": len(df), "reference": ref,
        "main_effect": {"hr": main_hr, "p": main_p},
        "interactions": interaction_ps,
        "per_subgroup": per_subgroup,
    }


def main():
    model, aug = train_drkg_best()
    pred = collect_predictions(model, aug)
    pred_path = EXP_DIR / "heterogeneity_predictions.csv"
    pred.to_csv(pred_path, index=False)
    print(f"Saved {len(pred)} predictions → {pred_path}")

    print("\n=== Overall (pooled) ===")
    print(f"  n={len(pred)}, events={pred['event'].sum()}")
    print(f"  Cancer types: {pred['cancer_type'].value_counts().to_dict()}")
    print(f"  Drug classes: {pred['drug_class'].value_counts().to_dict()}")

    print("\n" + "=" * 60)
    print("  HETEROGENEITY: Cancer Type")
    print("=" * 60)
    cancer_result = fit_interaction_cox(pred, "cancer_type", min_samples=25)
    print(json.dumps(cancer_result, indent=2, default=str))

    print("\n" + "=" * 60)
    print("  HETEROGENEITY: Drug Class")
    print("=" * 60)
    drug_result = fit_interaction_cox(pred, "drug_class", min_samples=25)
    print(json.dumps(drug_result, indent=2, default=str))

    out = {
        "model": "DRKG PathAttnSurv FMB+PPI+Disease+Drug seed=42",
        "n_predictions": len(pred),
        "n_events": int(pred["event"].sum()),
        "cancer_interaction": cancer_result,
        "drug_interaction": drug_result,
    }
    with open(EXP_DIR / "heterogeneity.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {EXP_DIR / 'heterogeneity.json'}")


if __name__ == "__main__":
    main()
