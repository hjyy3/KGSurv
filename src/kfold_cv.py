"""K-fold cross-validation for selected knowledge-graph configurations.

For each KG (with its best multi-node combo), do 5-fold CV on the training
set. For each fold: train model, pick best checkpoint by val_ci on held-out
fold, then evaluate on all 11 external cohorts. Aggregate per-fold metrics
(mean ± std) to quantify single-split variance.

No ensembling — external prediction is per-fold, reported as fold variance only.
"""
from __future__ import annotations

import argparse
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
    augment_splits, build_combo_info, load_base_splits, MODEL, EVAL_COHORTS,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features
from train_interp import _seed_everything, train_epoch, evaluate_ci

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best multi-node combo per KG (same as retrain_fs.py)
BEST_COMBOS = {
    "drkg":        ["ppi", "disease", "drug"],
    "openbiolink": ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"],
    "monarch":     ["ppi", "disease", "phenotype", "anatomy"],
    "hetionet":    ["ppi", "disease", "drug", "anatomy", "regulatory"],
    "primekg":     ["ppi"],
    "ibkh":        ["ppi", "disease", "drug"],
    "ogb_biokg":   ["ppi", "drug"],
}

BASELINES = {  # single 80/20 best_n_sig from PROGRESS.md
    "drkg":        {"n_sig": 7, "ext_ci": 0.598},
    "openbiolink": {"n_sig": 6, "ext_ci": 0.594},
    "monarch":     {"n_sig": 6, "ext_ci": 0.596},
    "hetionet":    {"n_sig": 5, "ext_ci": 0.592},
    "primekg":     {"n_sig": 6, "ext_ci": 0.554},
    "ibkh":        {"n_sig": 4, "ext_ci": 0.571},
    "ogb_biokg":   {"n_sig": 4, "ext_ci": 0.566},
}

K = 5
EPOCHS = 80
PATIENCE = 15
HIDDEN = 32
DROPOUT = 0.1
LR = 1e-3
WD = 1e-4
BATCH = 64


def _select(data: dict, idx_tensor: torch.Tensor) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx_tensor]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx_tensor.tolist()]
        else:
            out[k] = v
    return out


def kfold_indices(n: int, k: int, seed: int = 42) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return list of (train_idx, val_idx) tensors for k-fold CV."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = n if i == k - 1 else (i + 1) * fold_size
        val_idx = perm[start:end]
        train_idx = torch.cat([perm[:start], perm[end:]])
        folds.append((train_idx, val_idx))
    return folds


@torch.no_grad()
def _get_risk(model, data):
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def train_fold(kg_info, train_data, val_data, valid_cohorts, seed: int) -> dict:
    """Train on train_data, select best checkpoint by val_data val_ci, eval externally."""
    _seed_everything(seed)
    model = create_model(MODEL, kg_info, hidden_dim=HIDDEN, dropout=DROPOUT)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                     patience=7, factor=0.5, min_lr=1e-6)
    best, pat, st = 0.0, 0, None
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, train_data, opt, BATCH, device)
        ci = evaluate_ci(model, val_data, device)
        sch.step(ci)
        if ci > best:
            best, pat = ci, 0
            st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= PATIENCE:
            break
    if st:
        model.load_state_dict(st)

    r = {"val_ci": round(best, 4)}
    ext_cis, sigs = [], []
    per_cohort = {}
    for c in EVAL_COHORTS:
        if c not in valid_cohorts:
            continue
        risk = _get_risk(model, valid_cohorts[c])
        m = compute_all_metrics(risk, valid_cohorts[c]["time"].numpy(),
                                 valid_cohorts[c]["event"].numpy())
        per_cohort[c] = {"ci": round(m["c_index"], 4), "p": round(m["p_value"], 4)}
        ext_cis.append(m["c_index"])
        if m["p_value"] < 0.05:
            sigs.append(c)
    r["ext_ci"] = round(float(np.mean(ext_cis)), 4)
    r["n_sig"] = len(sigs)
    r["sigs"] = sigs
    r["per_cohort"] = per_cohort
    return r


def run_kfold_for_kg(kg: str, node_types: list[str], genes: list[str], k: int,
                     seed: int = 42) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {kg}  nodes={node_types}  K={k}")
    print(f"{'=' * 60}")

    splits, raw = load_base_splits(kg)
    # Compute node features
    node_feats = {}
    for nt in node_types:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features(kg, nt, mode, edict, genes, raw)
        if res:
            feats, tnames, m = res
            node_feats[nt] = (feats, tnames, m)
    missing = [nt for nt in node_types if nt not in node_feats]
    if missing:
        return {"status": "skip", "missing": missing}

    extra_f = [node_feats[nt][0] for nt in node_types]
    extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in node_types]
    aug = augment_splits(splits, extra_f)
    info = build_combo_info(kg, extra_i, len(genes))

    train_full = aug["train"]
    valid_cohorts = {c: aug[c] for c in EVAL_COHORTS if c in aug}

    n = train_full["mut"].shape[0]
    folds = kfold_indices(n, k, seed=seed)
    print(f"  Training set n={n}; fold sizes: {[len(v[1]) for v in folds]}")

    fold_results = []
    for i, (tr_idx, va_idx) in enumerate(folds, 1):
        tr = _select(train_full, tr_idx)
        va = _select(train_full, va_idx)
        r = train_fold(info, tr, va, valid_cohorts, seed=seed + i)
        fold_results.append(r)
        print(f"  fold {i}/{k}: val={r['val_ci']:.4f}  ext={r['ext_ci']:.4f}  "
              f"sig={r['n_sig']}/11  sigs={r['sigs']}")

    # Aggregate
    val_cis = [r["val_ci"] for r in fold_results]
    ext_cis = [r["ext_ci"] for r in fold_results]
    n_sigs = [r["n_sig"] for r in fold_results]
    summary = {
        "status": "ok", "kg": kg, "node_types": node_types, "k": k,
        "val_ci_mean": float(np.mean(val_cis)), "val_ci_std": float(np.std(val_cis)),
        "val_ci_min": float(np.min(val_cis)), "val_ci_max": float(np.max(val_cis)),
        "ext_ci_mean": float(np.mean(ext_cis)), "ext_ci_std": float(np.std(ext_cis)),
        "ext_ci_min": float(np.min(ext_cis)), "ext_ci_max": float(np.max(ext_cis)),
        "n_sig_mean": float(np.mean(n_sigs)), "n_sig_std": float(np.std(n_sigs)),
        "n_sig_min": int(np.min(n_sigs)), "n_sig_max": int(np.max(n_sigs)),
        "fold_results": fold_results,
    }
    print(f"  SUMMARY: val {summary['val_ci_mean']:.4f}±{summary['val_ci_std']:.4f}  "
          f"ext {summary['ext_ci_mean']:.4f}±{summary['ext_ci_std']:.4f}  "
          f"sig {summary['n_sig_mean']:.1f}±{summary['n_sig_std']:.1f} "
          f"(range {summary['n_sig_min']}-{summary['n_sig_max']})")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgs", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="kfold_cv.json")
    args = parser.parse_args()

    genes = load_candidate_genes()
    kgs = args.kgs if args.kgs else list(BEST_COMBOS.keys())

    all_results = []
    for kg in kgs:
        try:
            res = run_kfold_for_kg(kg, BEST_COMBOS[kg], genes, args.k, args.seed)
        except Exception as e:
            print(f"  ERROR {kg}: {e}")
            res = {"status": "error", "kg": kg, "error": str(e)}
        base = BASELINES.get(kg, {})
        res["baseline_n_sig"] = base.get("n_sig")
        res["baseline_ext_ci"] = base.get("ext_ci")
        all_results.append(res)

        out_path = EXP_DIR / args.out
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved → {EXP_DIR / args.out}")


if __name__ == "__main__":
    main()
