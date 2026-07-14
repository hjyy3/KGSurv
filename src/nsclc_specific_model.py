"""Direction A: NSCLC cancer-specific model (plan phase 3).

Train DRKG best architecture (PathAttnSurv FMB+PPI+Disease+Drug) on NSCLC-only
subset of training data (n≈344), evaluate on NSCLC-only subsets of external
cohorts. Compare with pan-cancer DRKG model on same NSCLC test sets.

Test cohorts (NSCLC):
    Gandara (n=427, all NSCLC)  - primary
    Ravi    (n=306, all NSCLC)  - primary
    Miao    (n=56,  NSCLC subset)
    Pleasance (n=18, NSCLC subset)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kg_features import load_candidate_genes
from losses import compute_all_metrics
from models_interp import create_model
from multi_node_extended import (
    augment_splits, build_combo_info, load_base_splits, EVAL_COHORTS, MODEL,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NSCLC_COL = "cov_cancer_Non-Small Cell Lung Cancer"

# Eval cohorts with NSCLC samples
NSCLC_COHORTS = ["Gandara", "Ravi", "Miao", "Pleasance"]


@torch.no_grad()
def _get_risk(model, data):
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def _select(data: dict, idx: torch.Tensor) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx.tolist()]
        else:
            out[k] = v
    return out


def filter_nsclc(splits: dict) -> dict:
    """Subset each split to NSCLC samples only (by cov_cancer col in clin)."""
    out = {}
    for sn, data in splits.items():
        prefix = "train" if sn == "train" else f"valid_{sn}"
        clin_path = PROC / f"{prefix}_clin.csv"
        if not clin_path.exists():
            continue
        clin = pd.read_csv(clin_path, index_col=0)
        if NSCLC_COL not in clin.columns:
            continue
        sample_ids = data["sample_ids"]
        is_nsclc = [int(clin.loc[sid, NSCLC_COL] > 0.5) if sid in clin.index else 0
                    for sid in sample_ids]
        keep_idx = torch.tensor([i for i, v in enumerate(is_nsclc) if v],
                                dtype=torch.long)
        if len(keep_idx) == 0:
            continue
        out[sn] = _select(data, keep_idx)
    return out


def train_and_eval(info, train_data, valid_data, seed: int, dropout: float = 0.05):
    _seed_everything(seed)
    tr, va = _split_data(train_data, 0.8, seed)
    model = create_model(MODEL, info, hidden_dim=32, dropout=dropout)
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
    r = {"val_ci": round(best, 4)}
    per_cohort = {}
    ext_cis, sigs = [], []
    for c in NSCLC_COHORTS:
        if c not in valid_data:
            continue
        risk = _get_risk(model, valid_data[c])
        m = compute_all_metrics(risk, valid_data[c]["time"].numpy(),
                                 valid_data[c]["event"].numpy())
        pc = {
            "n": len(risk),
            "c_index": round(m["c_index"], 4),
            "hr": round(m["hr"], 3),
            "p": round(m["p_value"], 4),
            "auc_12m": round(m.get("auc_12m", 0), 3),
            "auc_24m": round(m.get("auc_24m", 0), 3),
            "events": int(valid_data[c]["event"].sum().item()),
        }
        per_cohort[c] = pc
        ext_cis.append(m["c_index"])
        if m["p_value"] < 0.05:
            sigs.append(c)
    r["per_cohort"] = per_cohort
    r["ext_ci"] = round(float(np.mean(ext_cis)) if ext_cis else 0.0, 4)
    r["n_sig"] = len(sigs)
    r["sigs"] = sigs
    return r, model


def main():
    genes = load_candidate_genes()
    node_types = ["ppi", "disease", "drug"]

    print("Loading DRKG full splits...")
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
    aug_full = augment_splits(splits, extra_f)
    info = build_combo_info("drkg", extra_i, len(genes))
    print(f"  info total terms: {info.n_total_terms}")

    # Filter to NSCLC
    print("\nFiltering to NSCLC...")
    aug_nsclc = filter_nsclc(aug_full)
    for sn, data in aug_nsclc.items():
        print(f"  {sn}: n={data['mut'].shape[0]}, "
              f"events={int(data['event'].sum().item())}")

    # === Train pan-cancer baseline (for comparison) ===
    print("\n=== Pan-cancer (baseline) — train on all 1467, eval on NSCLC subsets ===")
    pan_result, pan_model = train_and_eval(
        info, aug_full["train"],
        {c: aug_nsclc[c] for c in NSCLC_COHORTS if c in aug_nsclc},
        seed=42)
    print(f"  val_ci (pan)={pan_result['val_ci']}, ext_ci={pan_result['ext_ci']}, "
          f"n_sig={pan_result['n_sig']}/{len([c for c in NSCLC_COHORTS if c in aug_nsclc])}")
    for c, pc in pan_result["per_cohort"].items():
        star = "*" if pc["p"] < 0.05 else " "
        print(f"    {c:<12} n={pc['n']:>3} ev={pc['events']:>3}  "
              f"CI={pc['c_index']:.4f} HR={pc['hr']:.2f} p={pc['p']:.4f}{star} "
              f"AUC24={pc['auc_24m']:.3f}")

    # === Train NSCLC-specific ===
    print("\n=== NSCLC-specific — train on 344, eval on NSCLC subsets ===")
    n_train = aug_nsclc["train"]["mut"].shape[0]
    print(f"  Train size: {n_train}")
    nsclc_result, nsclc_model = train_and_eval(
        info, aug_nsclc["train"],
        {c: aug_nsclc[c] for c in NSCLC_COHORTS if c in aug_nsclc},
        seed=42, dropout=0.1)   # slightly higher dropout for smaller sample
    print(f"  val_ci (nsclc)={nsclc_result['val_ci']}, ext_ci={nsclc_result['ext_ci']}, "
          f"n_sig={nsclc_result['n_sig']}")
    for c, pc in nsclc_result["per_cohort"].items():
        star = "*" if pc["p"] < 0.05 else " "
        print(f"    {c:<12} n={pc['n']:>3} ev={pc['events']:>3}  "
              f"CI={pc['c_index']:.4f} HR={pc['hr']:.2f} p={pc['p']:.4f}{star} "
              f"AUC24={pc['auc_24m']:.3f}")

    # Comparison summary
    print(f"\n{'=' * 70}")
    print("COMPARISON (NSCLC eval set)")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'val_ci':<8} {'ext_ci':<8} {'n_sig':<8} "
          f"{'Gandara_CI':<12} {'Ravi_CI':<10}")
    print("-" * 70)
    def row(label, r):
        g = r["per_cohort"].get("Gandara", {})
        ra = r["per_cohort"].get("Ravi", {})
        print(f"{label:<20} {r['val_ci']:<8} {r['ext_ci']:<8} "
              f"{r['n_sig']:<8} {g.get('c_index', '—'):<12} "
              f"{ra.get('c_index', '—'):<10}")
    row("Pan-cancer", pan_result)
    row("NSCLC-specific", nsclc_result)

    out = {
        "pan_cancer": pan_result,
        "nsclc_specific": nsclc_result,
    }
    with open(EXP_DIR / "nsclc_specific_model.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {EXP_DIR / 'nsclc_specific_model.json'}")


if __name__ == "__main__":
    main()
