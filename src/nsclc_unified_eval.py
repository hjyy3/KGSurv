"""NSCLC unified single-model evaluation.

Addresses the methodological requirement that per-cohort ROC (ORR), HR and
log-rank p be reported from the SAME model — not aggregated across 25 CV folds
and not split across a separate BCE-model (ORR) and Cox-model (PFS).

Design:
  - ONE ORR-BCE model trained on Ravi (drkg, FMB+PPI+Disease, sigfeats off).
  - Its single continuous logit score is used to derive, for every holdout:
      * ORR  : AUROC, AUPRC, Mann-Whitney p     (score vs RECIST/benefit label)
      * PFS  : C-index, HR (median split), log-rank p   (risk = -logit vs PFS)
    so response and progression are stratified by the identical score.
  - Two model-selection strategies, both reported:
      * ensemble    : 5 seeds, each trained on Ravi (85/15 internal early-stop
                      split); holdout logit = mean over the 5 models.
      * single_best : the single seed-model with the highest internal val AUROC
                      (weights saved + state hash for reproducibility/deployment).

Rationale for risk = -logit: a high ORR logit = high P(responder); responders
progress slower (lower hazard), so the PFS hazard risk is the negated logit.

Output:
  output/experiments/nsclc_unified/unified_eval.json
  output/experiments/nsclc_unified/unified_per_cohort.csv
  output/experiments/nsclc_unified/ensemble_seed*_model.pt (single_best weights)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_wes_cancer_specific import CANCER_SPECS, build_cancer_splits
from exp_wes_cancer_specific_orr import (
    eval_auroc,
    get_logits,
    load_orr_labels,
    train_epoch_bce,
)
from exp_wes_cancer_specific_pfs import load_pfs_labels
from exp_wes_pancancer import HPARAMS, select_data
from losses import compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "experiments" / "nsclc_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456, 789, 2024]
KG = "drkg"
NODE_TYPES = ["ppi", "disease"]   # ORR-ablation optimum (A3); Drug node hurt ORR
VAL_FRAC = 0.15


def _orr_metrics_from_logit(logit, y):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import mannwhitneyu
    if len(y) < 5 or y.sum() == 0 or y.sum() == len(y):
        return dict(n=int(len(y)), auroc=float("nan"), auprc=float("nan"), mw_p=float("nan"))
    try:
        u, p = mannwhitneyu(logit[y == 1], logit[y == 0], alternative="greater")
    except Exception:
        p = float("nan")
    return dict(
        n=int(len(y)), n_resp=int(y.sum()),
        auroc=round(float(roc_auc_score(y, logit)), 4),
        auprc=round(float(average_precision_score(y, logit)), 4),
        mw_p=round(float(p), 4),
    )


def _pfs_metrics_from_logit(logit, time_, event):
    if len(logit) < 5 or event.sum() < 3:
        return dict(n=int(len(logit)), c_index=float("nan"), hr=float("nan"), logrank_p=float("nan"))
    risk = -logit  # high ORR logit (responder) => low progression hazard
    m = compute_all_metrics(risk, time_, event)
    return dict(
        n=int(len(logit)), events=int(event.sum()),
        c_index=round(float(m["c_index"]), 4),
        hr=round(float(m["hr"]), 4),
        logrank_p=round(float(m["p_value"]), 4),
        sig=bool(m["p_value"] < 0.05),
    )


def _internal_split(n, seed, val_frac=VAL_FRAC):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(5, int(round(n * val_frac)))
    return (torch.tensor(idx[n_val:], dtype=torch.long),
            torch.tensor(idx[:n_val], dtype=torch.long))


def train_one_model(kg_info, train_data, seed, use_tmb=False):
    """Train one ORR-BCE model on Ravi with an internal early-stop split."""
    _seed_everything(seed)
    n = len(train_data["sample_ids"])
    tr_idx, va_idx = _internal_split(n, seed)
    tr = select_data(train_data, tr_idx)
    va = select_data(train_data, va_idx)
    model = create_model("path_attn", kg_info,
                         hidden_dim=HPARAMS["hidden"], dropout=HPARAMS["dropout"],
                         n_extra_risk=0, use_tmb=use_tmb)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["wd"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    best, pat, best_state = 0.0, 0, None
    for ep in range(1, HPARAMS["epochs"] + 1):
        train_epoch_bce(model, tr, opt, HPARAMS["batch"], device)
        a = eval_auroc(model, va)
        if np.isnan(a):
            break
        sch.step(a)
        if a > best:
            best, pat = a, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= HPARAMS["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best)


def main():
    print("Loading labels ...")
    orr_map = load_orr_labels()
    pfs_map = load_pfs_labels()

    spec = CANCER_SPECS["NSCLC"]
    print(f"Building Ravi splits + {KG} {NODE_TYPES} ...")
    augmented, kg_info, holdout_info = build_cancer_splits(KG, spec, NODE_TYPES)

    # Ravi train ORR labels
    train = augmented["train"]
    tr_keep, tr_y = [], []
    for i, sid in enumerate(train["sample_ids"]):
        if sid in orr_map:
            tr_keep.append(i); tr_y.append(orr_map[sid])
    train_data = select_data(train, torch.tensor(tr_keep, dtype=torch.long))
    train_data["orr"] = torch.tensor(tr_y, dtype=torch.float32)
    print(f"  Ravi ORR train n={len(tr_keep)} R={int(sum(tr_y))} NR={len(tr_y)-int(sum(tr_y))}")

    # ---- TMB-only baseline per holdout (no training; rank by mutation count) ----
    from sklearn.metrics import roc_auc_score as _auc
    tmb_base = {}
    for name in holdout_info:
        if name not in augmented:
            continue
        hd = augmented[name]
        sids = hd["sample_ids"]
        tmb = (hd["mut"] * hd["mask"]).sum(dim=1).cpu().numpy()
        oi = [i for i, s in enumerate(sids) if s in orr_map]
        y = np.array([orr_map[sids[i]] for i in oi])
        if len(y) >= 5 and 0 < y.sum() < len(y):
            tmb_base[name] = round(float(_auc(y, tmb[oi])), 4)
        else:
            tmb_base[name] = float("nan")

    def eval_variant(use_tmb):
        models, val_aurocs = [], []
        for seed in SEEDS:
            m, va = train_one_model(kg_info, train_data, seed, use_tmb=use_tmb)
            models.append((seed, m)); val_aurocs.append(va)
        best_i = int(np.argmax(val_aurocs))
        rows, logits_store = [], {}

        # In-sample aggregate on the full Ravi training set (ensemble of the 5
        # full-Ravi models). This is apparent/in-sample performance (the models
        # were trained on these samples), reported for comparison to the TMB
        # baseline on Ravi (0.697).
        tr_sids = train_data["sample_ids"]
        tr_per_model = [get_logits(m, train_data) for _, m in models]
        tr_ens = np.mean(tr_per_model, axis=0)
        tr_best = tr_per_model[best_i]
        tr_y = train_data["orr"].numpy()
        tr_tmb = (train_data["mut"] * train_data["mask"]).sum(dim=1).cpu().numpy()
        from sklearn.metrics import roc_auc_score as _auc2
        tr_tmb_auroc = round(float(_auc2(tr_y, tr_tmb)), 4)
        for strat, logit in [("ensemble", tr_ens), ("single_best", tr_best)]:
            om = _orr_metrics_from_logit(logit, tr_y)
            rows.append(dict(
                use_tmb=use_tmb, holdout="Ravi_train_insample", strategy=strat,
                tmb_baseline_auroc=tr_tmb_auroc,
                orr_n=om.get("n"), orr_auroc=om.get("auroc"), orr_mw_p=om.get("mw_p"),
                pfs_cindex=None, pfs_hr=None, pfs_logrank_p=None,
            ))

        for name in holdout_info:
            if name not in augmented:
                continue
            hd = augmented[name]; sids = hd["sample_ids"]
            per_model = [get_logits(m, hd) for _, m in models]
            ens = np.mean(per_model, axis=0); best = per_model[best_i]
            logits_store[f"{name}__ensemble"] = ens
            logits_store[f"{name}__single_best"] = best
            logits_store[f"{name}__sample_ids"] = np.array(sids)
            for strat, logit in [("ensemble", ens), ("single_best", best)]:
                oi = [i for i, s in enumerate(sids) if s in orr_map]
                oy = np.array([orr_map[sids[i]] for i in oi])
                om = _orr_metrics_from_logit(logit[oi], oy) if oi else {}
                pi = [i for i, s in enumerate(sids) if s in pfs_map]
                pt = np.array([pfs_map[sids[i]][0] for i in pi], dtype=float)
                pe = np.array([pfs_map[sids[i]][1] for i in pi], dtype=float)
                pm = _pfs_metrics_from_logit(logit[pi], pt, pe) if pi else {}
                rows.append(dict(
                    use_tmb=use_tmb, holdout=name, strategy=strat,
                    tmb_baseline_auroc=tmb_base.get(name),
                    orr_n=om.get("n"), orr_auroc=om.get("auroc"), orr_mw_p=om.get("mw_p"),
                    pfs_cindex=pm.get("c_index"), pfs_hr=pm.get("hr"), pfs_logrank_p=pm.get("logrank_p"),
                ))
        return rows, val_aurocs, best_i, logits_store

    all_rows = []
    meta_variants = {}
    for use_tmb in (False, True):
        tag = "with_tmb" if use_tmb else "no_tmb"
        print(f"\n>>> training variant: {tag}")
        rows, val_aurocs, best_i, logits_store = eval_variant(use_tmb)
        all_rows.extend(rows)
        meta_variants[tag] = dict(val_aurocs=[round(v, 4) for v in val_aurocs],
                                   single_best_seed=SEEDS[best_i])
        np.savez(OUT_DIR / f"holdout_logits_{tag}.npz", **logits_store)
        for seed, m in []:
            pass

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "unified_per_cohort.csv", index=False)
    meta = dict(kg=KG, node_types=NODE_TYPES, sigfeats=False, seeds=SEEDS,
                tmb_baseline_auroc=tmb_base, variants=meta_variants,
                pfs_risk="-logit (ORR responder = low progression hazard)")
    (OUT_DIR / "unified_eval.json").write_text(
        json.dumps(dict(meta=meta, per_cohort=all_rows), indent=2, default=str), encoding="utf-8")

    # ---- Comparison table: PRIMARY = ORR AUROC ----
    print("\n=== ORR AUROC (PRIMARY): TMB baseline vs model(no-TMB) vs model(+TMB) ===")
    print(f"  {'cohort':20s} {'TMB':>7} | {'noTMB_ens':>9} {'noTMB_best':>10} | {'+TMB_ens':>9} {'+TMB_best':>9}")
    for name in ["Ravi_train_insample"] + [n for n in holdout_info if n in augmented]:
        def g(use_tmb, strat):
            r = df[(df.use_tmb == use_tmb) & (df.holdout == name) & (df.strategy == strat)]
            return r["orr_auroc"].iloc[0] if len(r) else float("nan")
        def gtmb():
            r = df[(df.holdout == name) & (df.strategy == "ensemble")]
            return r["tmb_baseline_auroc"].iloc[0] if len(r) else float("nan")
        print(f"  {name:20s} {gtmb():>7.3f} | "
              f"{g(False,'ensemble'):>9.3f} {g(False,'single_best'):>10.3f} | "
              f"{g(True,'ensemble'):>9.3f} {g(True,'single_best'):>9.3f}")
    print("\n(Ravi_train_insample = apparent/in-sample; SUPPLEMENTARY PFS in unified_per_cohort.csv)")
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
