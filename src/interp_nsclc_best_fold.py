"""Interpretation extractor for best NSCLC fold (seed=42 fold=4 from N1).

Retrains the best-performing N1 fold, then runs a forward pass on the full
Ravi train + Miao_Lung holdout to extract:
  - group_importance (per pathway group, CLS attention weights)
  - term_importance  (per FMB / PPI-burden / Disease-burden / Drug-burden term)
  - gene_importance  (per WES gene, backprop through KG mask, masked by mut)

Aggregates separately for responders (ORR=1) vs non-responders (ORR=0) to
surface response-associated features. Outputs:

  output/experiments/wes_nsclc_interp/best_fold_top_groups.csv
  output/experiments/wes_nsclc_interp/best_fold_top_genes.csv
  output/experiments/wes_nsclc_interp/best_fold_top_disease_terms.csv
  output/experiments/wes_nsclc_interp/best_fold_top_drug_terms.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_wes_cancer_specific import CANCER_SPECS, MULTI_NODE_RECIPE, build_cancer_splits
from exp_wes_cancer_specific_orr import (
    attach_orr_filter,
    attach_sigfeats_orr,
    attach_train_sigfeats,
    load_orr_labels,
    stratified_kfold_by_event,
    train_one_fold_orr,
)
from exp_wes_pancancer import HPARAMS, load_sigfeats, normalise_sigfeats, select_data
from models_interp import create_model
from train_interp import _seed_everything, train_epoch

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "experiments"
OUT_DIR = EXP_DIR / "wes_nsclc_interp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best fold from N1 sweep (seed=42 fold=4: ext_auroc=0.863, Miao_Lung 0.882)
BEST_SEED = 42
BEST_FOLD = 4
CONFIG = dict(kg="drkg", sigfeats=True, node_types=["ppi", "disease", "drug"])
TOP_K = 30


def get_wes_genes():
    return pd.read_csv(ROOT / "output" / "processed_wes" / "wes_candidate_genes.csv")["gene"].tolist()


def main():
    print("Loading ORR + sigfeats ...")
    orr_map = load_orr_labels()
    sigfeats_norm = normalise_sigfeats(load_sigfeats())

    spec = CANCER_SPECS["NSCLC"]
    print(f"Building Ravi splits + DRKG multi-node {CONFIG['node_types']} ...")
    augmented, kg_info, holdout_info = build_cancer_splits("drkg", spec, CONFIG["node_types"])
    print(f"  kg_info groups={len(kg_info.group_names)}, total_terms={kg_info.n_total_terms}")
    print(f"  group_names={kg_info.group_names}")

    # ORR labels
    train_data = attach_orr_filter(augmented["train"], orr_map, "Ravi")
    holdouts = {n: attach_orr_filter(augmented[n], orr_map, holdout_info[n]["base"])
                for n in holdout_info if n in augmented}
    holdouts = {n: hd for n, hd in holdouts.items() if hd is not None}

    # Sigfeats
    train_data = attach_train_sigfeats(train_data, sigfeats_norm, "Ravi")
    for n in list(holdouts.keys()):
        holdouts[n] = attach_sigfeats_orr(holdouts[n], sigfeats_norm, holdout_info[n]["sigfeats_key"])
    n_extra_risk = train_data["extra_risk"].shape[1]

    # Recreate the same fold split
    folds = stratified_kfold_by_event(train_data["orr"].numpy(), 5, BEST_SEED)
    tr_idx, va_idx = folds[BEST_FOLD]
    tr_split = select_data(train_data, tr_idx)
    va_split = select_data(train_data, va_idx)

    # Retrain
    print(f"\nRetraining seed={BEST_SEED} fold={BEST_FOLD} ...")
    r = train_one_fold_orr(kg_info, tr_split, va_split, holdouts, BEST_SEED, n_extra_risk)
    print(f"  val_auroc={r['val']['auroc']}  ext_auroc={r['ext_auroc_primary']}  n_sig={r['n_sig_primary']}")
    for ch, m in r["per_cohort"].items():
        print(f"    {ch}: auroc={m['auroc']} mw_p={m['mw_p']} sig={m['sig']}")

    # Rebuild model with same seed + train to convergence, then forward pass with importance
    _seed_everything(BEST_SEED)
    model = create_model(
        "path_attn", kg_info,
        hidden_dim=HPARAMS["hidden"], dropout=HPARAMS["dropout"],
        n_extra_risk=n_extra_risk,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["wd"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    from exp_wes_cancer_specific_orr import eval_auroc, train_epoch_bce
    best, pat, best_state = 0.0, 0, None
    for ep in range(1, HPARAMS["epochs"] + 1):
        train_epoch_bce(model, tr_split, opt, HPARAMS["batch"], device)
        a = eval_auroc(model, va_split)
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
    print(f"  Retrained: best val_auroc={best:.4f}")

    # ------------------------------------------------------------------
    # Forward pass on train + Miao_Lung with importance returned
    # ------------------------------------------------------------------
    def forward_with_importance(data):
        model.eval()
        kw = {}
        if "extra_risk" in data:
            kw["extra_risk"] = data["extra_risk"].to(device)
        with torch.no_grad():
            out = model(data["mut"].to(device), data["mask"].to(device),
                        data["fmb"].to(device), **kw)
        return {
            "logit": out["log_risk"].cpu().numpy(),
            "gene_imp": out["gene_importance"].cpu().numpy(),
            "term_imp": out["term_importance"].cpu().numpy(),
            "group_imp": out["group_importance"].cpu().numpy(),
        }

    train_out = forward_with_importance(train_data)
    miao_out = forward_with_importance(holdouts["Miao_Lung"])

    # ------------------------------------------------------------------
    # Aggregate per group / term / gene; split by responder status
    # ------------------------------------------------------------------
    genes = get_wes_genes()
    group_names = list(kg_info.group_names)

    # Flatten term names with group prefix for disambiguation
    flat_term_names: list[str] = []
    flat_term_group: list[str] = []
    for gi, gname in enumerate(group_names):
        for tname in kg_info.term_names[gi]:
            flat_term_names.append(tname)
            flat_term_group.append(gname)

    def summarise(arrs, y_arr, label_suffix=""):
        """arrs is dict with gene_imp/term_imp/group_imp; y_arr is ORR labels.
        Returns 3 dataframes (groups, terms, genes) with mean overall + per class."""
        gi_all = arrs["group_imp"]    # [n, n_groups]
        ti_all = arrs["term_imp"]     # [n, n_terms]
        ge_all = arrs["gene_imp"]     # [n, n_genes]

        is_resp = y_arr == 1
        is_nonr = y_arr == 0

        def mean_safe(x, m):
            if m.sum() == 0:
                return np.full(x.shape[1], float("nan"))
            return x[m].mean(axis=0)

        df_groups = pd.DataFrame({
            "group": group_names,
            f"mean_all{label_suffix}":  gi_all.mean(axis=0),
            f"mean_resp{label_suffix}": mean_safe(gi_all, is_resp),
            f"mean_nonresp{label_suffix}": mean_safe(gi_all, is_nonr),
        })
        df_groups[f"delta_R_minus_NR{label_suffix}"] = (
            df_groups[f"mean_resp{label_suffix}"] - df_groups[f"mean_nonresp{label_suffix}"]
        )

        df_terms = pd.DataFrame({
            "term": flat_term_names,
            "group": flat_term_group,
            f"mean_all{label_suffix}":  ti_all.mean(axis=0),
            f"mean_resp{label_suffix}": mean_safe(ti_all, is_resp),
            f"mean_nonresp{label_suffix}": mean_safe(ti_all, is_nonr),
        })
        df_terms[f"delta_R_minus_NR{label_suffix}"] = (
            df_terms[f"mean_resp{label_suffix}"] - df_terms[f"mean_nonresp{label_suffix}"]
        )

        df_genes = pd.DataFrame({
            "gene": genes,
            f"mean_all{label_suffix}":  ge_all.mean(axis=0),
            f"mean_resp{label_suffix}": mean_safe(ge_all, is_resp),
            f"mean_nonresp{label_suffix}": mean_safe(ge_all, is_nonr),
        })
        df_genes[f"delta_R_minus_NR{label_suffix}"] = (
            df_genes[f"mean_resp{label_suffix}"] - df_genes[f"mean_nonresp{label_suffix}"]
        )
        return df_groups, df_terms, df_genes

    tr_y = train_data["orr"].numpy()
    miao_y = holdouts["Miao_Lung"]["orr"].numpy()
    tg, tt, tg_genes = summarise(train_out, tr_y, "_train")
    mg, mt, mg_genes = summarise(miao_out, miao_y, "_miao")

    # Merge train + miao views
    df_groups = tg.merge(mg, on="group", how="outer")
    df_terms = tt.merge(mt, on=["term", "group"], how="outer")
    df_genes = tg_genes.merge(mg_genes, on="gene", how="outer")

    # Save full tables
    df_groups.to_csv(OUT_DIR / "best_fold_group_importance.csv", index=False)
    df_terms.to_csv(OUT_DIR / "best_fold_term_importance.csv", index=False)
    df_genes.to_csv(OUT_DIR / "best_fold_gene_importance.csv", index=False)

    # Top-K by miao delta_R_minus_NR (response-associated)
    df_genes_top = df_genes.dropna(subset=["delta_R_minus_NR_miao"]) \
        .sort_values("delta_R_minus_NR_miao", ascending=False).head(TOP_K)
    df_genes_top.to_csv(OUT_DIR / "best_fold_top_genes_response.csv", index=False)

    df_disease_top = df_terms[df_terms["group"] == "x_disease"] \
        .dropna(subset=["delta_R_minus_NR_miao"]) \
        .sort_values("delta_R_minus_NR_miao", ascending=False).head(TOP_K)
    df_disease_top.to_csv(OUT_DIR / "best_fold_top_disease_terms.csv", index=False)

    df_drug_top = df_terms[df_terms["group"] == "x_drug"] \
        .dropna(subset=["delta_R_minus_NR_miao"]) \
        .sort_values("delta_R_minus_NR_miao", ascending=False).head(TOP_K)
    df_drug_top.to_csv(OUT_DIR / "best_fold_top_drug_terms.csv", index=False)

    # Print top-10 highlights
    print(f"\n{'='*60}\nTOP-10 GROUPS by Miao_Lung response-association (delta R-NR):\n")
    print(df_groups.sort_values("delta_R_minus_NR_miao", ascending=False).head(10).to_string(index=False))
    print(f"\n{'='*60}\nTOP-10 GENES on Miao_Lung (R high):\n")
    print(df_genes_top.head(10)[["gene", "mean_resp_miao", "mean_nonresp_miao",
                                  "delta_R_minus_NR_miao"]].to_string(index=False))
    print(f"\n{'='*60}\nTOP-10 DISEASE terms (R high) on Miao_Lung:\n")
    print(df_disease_top.head(10)[["term", "mean_resp_miao", "delta_R_minus_NR_miao"]].to_string(index=False))
    print(f"\n{'='*60}\nTOP-10 DRUG terms (R high) on Miao_Lung:\n")
    print(df_drug_top.head(10)[["term", "mean_resp_miao", "delta_R_minus_NR_miao"]].to_string(index=False))

    meta = {
        "seed": BEST_SEED, "fold": BEST_FOLD,
        "config": "N1_orr_nsclc_ravi_drkg_multinode",
        "val_auroc": r["val"]["auroc"],
        "ext_auroc_primary": r["ext_auroc_primary"],
        "per_cohort": r["per_cohort"],
        "group_names": group_names,
        "n_train": len(train_data["sample_ids"]),
        "n_train_resp": int(tr_y.sum()),
        "n_miao": len(holdouts["Miao_Lung"]["sample_ids"]),
        "n_miao_resp": int(miao_y.sum()),
    }
    (OUT_DIR / "interp_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved tables + meta to {OUT_DIR}")


if __name__ == "__main__":
    main()
