"""Replay the 7/11-peak fold (DRKG seed=42 fold=0, PathAttnSurv on FMB+PPI+Disease+Drug)
and compare three risk-stratification cutoff strategies:

  C1) per-cohort median  (current baseline, gives 7/11 on this fold)
  C2) MSK train median   (fixed; derived from the model's training fold)
  C3) MSK train ROC      (fixed; threshold minimising distance to (0,1.0) on
                          time-dependent ROC at t = 24 months on train risks)

Output: output/experiments/fixed_cutoff_drkg_fold0.json + ..._table.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kg_features import load_candidate_genes  # noqa: E402
from kfold_cv import (  # noqa: E402
    _select, kfold_indices, BEST_COMBOS, K, EPOCHS, PATIENCE,
    HIDDEN, DROPOUT, LR, WD, BATCH,
)
from models_interp import create_model  # noqa: E402
from multi_node_extended import (  # noqa: E402
    augment_splits, build_combo_info, load_base_splits, MODEL, EVAL_COHORTS,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features  # noqa: E402
from train_interp import _seed_everything, train_epoch, evaluate_ci  # noqa: E402

OUT = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KG = "drkg"
NODE_TYPES = BEST_COMBOS[KG]   # ["ppi", "disease", "drug"]
FOLD_INDEX = 0                 # the fold that produced 7/11
SEED_BASE = 42                 # same as kfold_cv default
ROC_TIME = 24.0                # months — Plan-A primary time-point


@torch.no_grad()
def _get_risk(model, data) -> np.ndarray:
    model.eval()
    out = model(data["mut"].to(device),
                data["mask"].to(device),
                data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def _train_one_fold(kg_info, train_data, val_data, seed: int):
    _seed_everything(seed)
    model = create_model(MODEL, kg_info, hidden_dim=HIDDEN, dropout=DROPOUT)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=7, factor=0.5, min_lr=1e-6,
    )
    best_ci, patience_left, best_state = 0.0, 0, None
    for _ in range(EPOCHS):
        train_epoch(model, train_data, opt, BATCH, device)
        ci = evaluate_ci(model, val_data, device)
        sch.step(ci)
        if ci > best_ci:
            best_ci = ci
            patience_left = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left += 1
        if patience_left >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ci


def _logrank(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
             threshold: float) -> dict:
    """Apply fixed threshold (high = risk >= threshold) and run log-rank."""
    high = risk >= threshold
    low = ~high
    n_high, n_low = int(high.sum()), int(low.sum())
    if n_high < 2 or n_low < 2:
        return {"n_high": n_high, "n_low": n_low,
                "hr": float("nan"), "p": float("nan"), "sig": False}
    res = logrank_test(time[high], time[low], event[high], event[low])
    ev_hi = event[high].sum() / max(n_high, 1)
    ev_lo = event[low].sum() / max(n_low, 1)
    hr = ev_hi / max(ev_lo, 1e-8)
    return {"n_high": n_high, "n_low": n_low,
            "hr": float(hr), "p": float(res.p_value),
            "sig": bool(res.p_value < 0.05)}


def _train_roc_threshold(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                          t: float = ROC_TIME) -> dict:
    """Pick threshold minimising distance to (0,1.0) on the time-t binary ROC.

    Binary label at time t:
        positive: time <= t and event == 1     (died by t)
        negative: time > t                     (alive at t)
        excluded: event == 0 and time <= t     (censored before t — unknown)
    """
    pos = (time <= t) & (event == 1)
    neg = time > t
    keep = pos | neg
    y_true = pos[keep].astype(int)
    y_score = risk[keep]
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos < 5 or n_neg < 5:
        return {"threshold": float(np.median(risk)),
                "n_pos": n_pos, "n_neg": n_neg,
                "fpr": float("nan"), "tpr": float("nan"),
                "distance": float("nan"),
                "fallback": "median_due_to_low_n"}
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    distances = np.sqrt(fpr ** 2 + (1.0 - tpr) ** 2)
    best = int(np.argmin(distances))
    return {
        "threshold": float(thresh[best]),
        "n_pos": n_pos, "n_neg": n_neg,
        "fpr": float(fpr[best]),
        "tpr": float(tpr[best]),
        "distance": float(distances[best]),
        "fallback": None,
    }


def main():
    print("Replaying DRKG fold 0 (PathAttnSurv, FMB+PPI+Disease+Drug)…")
    genes = load_candidate_genes()
    splits, raw = load_base_splits(KG)

    node_feats = {}
    for nt in NODE_TYPES:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features(KG, nt, mode, edict, genes, raw)
        assert res is not None, f"node feat failed for {nt}"
        feats, tnames, mat = res
        node_feats[nt] = (feats, tnames, mat)

    extra_f = [node_feats[nt][0] for nt in NODE_TYPES]
    extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}")
               for nt in NODE_TYPES]
    aug = augment_splits(splits, extra_f)
    info = build_combo_info(KG, extra_i, len(genes))

    train_full = aug["train"]
    valid_cohorts = {c: aug[c] for c in EVAL_COHORTS if c in aug}
    n = train_full["mut"].shape[0]
    folds = kfold_indices(n, K, seed=SEED_BASE)
    tr_idx, va_idx = folds[FOLD_INDEX]
    print(f"  Train pool n={n}, fold {FOLD_INDEX}: "
          f"train={len(tr_idx)}, val={len(va_idx)}")

    tr_data = _select(train_full, tr_idx)
    va_data = _select(train_full, va_idx)

    model, best_val_ci = _train_one_fold(info, tr_data, va_data,
                                         seed=SEED_BASE + FOLD_INDEX + 1)
    print(f"  best val_ci = {best_val_ci:.4f}")

    # Risks for train pool (the fold's training subset)
    train_risks = _get_risk(model, tr_data)
    train_time = tr_data["time"].numpy()
    train_event = tr_data["event"].numpy()

    # Derive fixed cutoffs from train
    train_median = float(np.median(train_risks))
    roc_info = _train_roc_threshold(train_risks, train_time, train_event)
    train_roc = roc_info["threshold"]
    print(f"  train median = {train_median:.5f}")
    print(f"  train ROC@24m closest to (0,1) = {train_roc:.5f}  "
          f"(fpr={roc_info['fpr']:.3f}, tpr={roc_info['tpr']:.3f}, "
          f"dist={roc_info['distance']:.3f})")

    # Evaluate the three cutoff strategies on each external cohort
    rows = []
    for c in EVAL_COHORTS:
        if c not in valid_cohorts:
            continue
        cd = valid_cohorts[c]
        risk = _get_risk(model, cd)
        t_arr = cd["time"].numpy()
        e_arr = cd["event"].numpy()
        cohort_median = float(np.median(risk))

        r1 = _logrank(risk, t_arr, e_arr, cohort_median)
        r2 = _logrank(risk, t_arr, e_arr, train_median)
        r3 = _logrank(risk, t_arr, e_arr, train_roc)

        rows.append({
            "cohort": c,
            "n": len(risk),
            "events": int(e_arr.sum()),
            "risk_min": float(risk.min()),
            "risk_med": cohort_median,
            "risk_max": float(risk.max()),
            # cutoff 1 — per-cohort median
            "c1_thr": cohort_median,
            "c1_n_high": r1["n_high"], "c1_n_low": r1["n_low"],
            "c1_hr": r1["hr"], "c1_p": r1["p"], "c1_sig": r1["sig"],
            # cutoff 2 — MSK train median
            "c2_thr": train_median,
            "c2_n_high": r2["n_high"], "c2_n_low": r2["n_low"],
            "c2_hr": r2["hr"], "c2_p": r2["p"], "c2_sig": r2["sig"],
            # cutoff 3 — train ROC closest-to-(0,1) at t=24m
            "c3_thr": train_roc,
            "c3_n_high": r3["n_high"], "c3_n_low": r3["n_low"],
            "c3_hr": r3["hr"], "c3_p": r3["p"], "c3_sig": r3["sig"],
        })
        print(f"  {c:<14}  c1: p={r1['p']:.4f} HR={r1['hr']:.2f}  "
              f"c2: p={r2['p']:.4f} HR={r2['hr']:.2f} "
              f"(high={r2['n_high']:>2}/{r2['n_high']+r2['n_low']})  "
              f"c3: p={r3['p']:.4f} HR={r3['hr']:.2f} "
              f"(high={r3['n_high']:>2}/{r3['n_high']+r3['n_low']})")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "fixed_cutoff_drkg_fold0_table.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "config": {
            "kg": KG, "node_types": NODE_TYPES,
            "model": MODEL, "fold": FOLD_INDEX,
            "seed_base": SEED_BASE, "seed_train": SEED_BASE + FOLD_INDEX + 1,
            "roc_time_months": ROC_TIME,
        },
        "best_val_ci": best_val_ci,
        "train_n": int(len(train_risks)),
        "train_median": train_median,
        "train_roc": roc_info,
        "n_sig": {
            "c1_per_cohort_median": int(df["c1_sig"].sum()),
            "c2_train_median":       int(df["c2_sig"].sum()),
            "c3_train_roc":          int(df["c3_sig"].sum()),
        },
        "sig_cohorts": {
            "c1": df.loc[df["c1_sig"], "cohort"].tolist(),
            "c2": df.loc[df["c2_sig"], "cohort"].tolist(),
            "c3": df.loc[df["c3_sig"], "cohort"].tolist(),
        },
        "rows": rows,
    }
    json_path = OUT / "fixed_cutoff_drkg_fold0.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: bool(x)
                  if isinstance(x, np.bool_) else float(x))

    print("\n=== n_sig (p<0.05) ===")
    print(f"  C1 per-cohort median : {summary['n_sig']['c1_per_cohort_median']}/11"
          f"  →  {summary['sig_cohorts']['c1']}")
    print(f"  C2 MSK train median  : {summary['n_sig']['c2_train_median']}/11"
          f"  →  {summary['sig_cohorts']['c2']}")
    print(f"  C3 MSK train ROC@24m : {summary['n_sig']['c3_train_roc']}/11"
          f"  →  {summary['sig_cohorts']['c3']}")
    print(f"\nSaved → {json_path}\n        {csv_path}")


if __name__ == "__main__":
    main()
