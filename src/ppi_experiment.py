"""Experiment: FMB + PPI burden features.

Strategy:
  - Load FMB (pathway-level) + PPI burden (gene-level, 463 dims)
  - For PathAttnSurv: add PPI as a new group in the FMB feature vector
  - For SparsePathNet: concatenate PPI to [mut*mask, mask] input
  - Compare with FMB-only baseline
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import (
    KG_DIR, PROC_DIR, SUBKG_DIR, VALID_COHORTS,
    KGGroupInfo, _load_gene_list, _norm_ws, build_kg_group_info,
)
from losses import c_index, compute_all_metrics, cox_loss
from models_interp import create_model, count_parameters
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exclude RCC
EVAL_COHORTS = [
    "Gandara", "Hugo", "Liu", "Mariathasan", "Miao",
    "PUSH", "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae",
]


def load_split_with_ppi(kg_name: str, split: str) -> dict:
    """Load mut, mask, FMB, PPI for one split."""
    prefix = "train" if split == "train" else f"valid_{split}"
    kg_feat = KG_DIR / kg_name

    mut = pd.read_csv(PROC_DIR / f"{prefix}_mut.csv", index_col=0)
    mask_df = pd.read_csv(PROC_DIR / f"{prefix}_mask.csv", index_col=0)
    clin = pd.read_csv(PROC_DIR / f"{prefix}_clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / f"{prefix}_fmb.csv", index_col=0)
    ppi = pd.read_csv(kg_feat / f"{prefix}_ppi.csv", index_col=0)

    common = mut.index.intersection(clin.index).intersection(fmb.index).intersection(ppi.index)

    return {
        "mut": torch.tensor(mut.loc[common].values, dtype=torch.float32),
        "mask": torch.tensor(mask_df.loc[common].values, dtype=torch.float32),
        "time": torch.tensor(clin.loc[common, "OS_MONTHS"].values, dtype=torch.float32),
        "event": torch.tensor(clin.loc[common, "event"].values, dtype=torch.float32),
        "fmb": torch.tensor(fmb.loc[common].values, dtype=torch.float32),
        "ppi": torch.tensor(ppi.loc[common].values, dtype=torch.float32),
        "sample_ids": common.tolist(),
    }


def build_kg_group_info_with_ppi(kg_name: str) -> KGGroupInfo:
    """Build KGGroupInfo with PPI as an additional group."""
    base_info = build_kg_group_info(kg_name)
    genes = _load_gene_list()

    # Add PPI as a new group: 463 "terms" (one per gene)
    ppi_terms = [f"ppi_{g}" for g in genes]
    ppi_mask = torch.eye(len(genes), dtype=torch.float32)  # diagonal: each gene maps to itself

    new_group_names = base_info.group_names + ["ppi_neighborhood"]
    new_term_names = base_info.term_names + [ppi_terms]
    new_gene_term_mask = base_info.gene_term_mask + [ppi_mask]

    old_end = base_info.fmb_slices[-1][1] if base_info.fmb_slices else 0
    new_slices = base_info.fmb_slices + [(old_end, old_end + len(ppi_terms))]

    return KGGroupInfo(
        kg_name=kg_name,
        group_names=new_group_names,
        term_names=new_term_names,
        gene_term_mask=new_gene_term_mask,
        fmb_slices=new_slices,
        n_genes=base_info.n_genes,
        n_total_terms=base_info.n_total_terms + len(ppi_terms),
    )


def concat_fmb_ppi(data: dict) -> dict:
    """Return new dict with fmb = [fmb | ppi] concatenated."""
    out = dict(data)
    out["fmb"] = torch.cat([data["fmb"], data["ppi"]], dim=1)
    return out


@torch.no_grad()
def get_risk(model, data):
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def train_model(model_name, kg_info, train_data, seed=42):
    _seed_everything(seed)
    train_split, val_split = _split_data(train_data, 0.8, seed)
    model = create_model(model_name, kg_info, hidden_dim=32, dropout=0.1)
    model.to(device)
    tot, eff = count_parameters(model)
    print(f"  {model_name}: {tot:,} params ({eff:,} effective)")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    best_ci, patience_ctr, best_state = 0.0, 0, None
    for ep in range(1, 81):
        train_epoch(model, train_split, opt, 64, device)
        val_ci = evaluate_ci(model, val_split, device)
        sched.step(val_ci)
        if val_ci > best_ci:
            best_ci, patience_ctr = val_ci, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= 15:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_ci


def main():
    KGS = ["primekg", "ogb_biokg"]
    MODELS = ["sparse_path", "path_attn"]
    all_results = []

    for kg in KGS:
        print(f"\n{'='*60}")
        print(f"  KG = {kg}")
        print(f"{'='*60}")

        # Load data
        train_data = load_split_with_ppi(kg, "train")
        valid_data = {}
        for c in EVAL_COHORTS:
            try:
                valid_data[c] = load_split_with_ppi(kg, c)
            except FileNotFoundError:
                pass

        # Build group info with PPI
        kg_info_ppi = build_kg_group_info_with_ppi(kg)
        kg_info_base = build_kg_group_info(kg)

        print(f"  FMB terms: {kg_info_base.n_total_terms}")
        print(f"  FMB+PPI terms: {kg_info_ppi.n_total_terms}")
        print(f"  PPI group adds {kg_info_ppi.n_total_terms - kg_info_base.n_total_terms} features")

        for model_name in MODELS:
            for mode in ["fmb_only", "fmb+ppi"]:
                tag = f"{model_name}_{kg}_{mode.replace('+','_')}"
                print(f"\n--- {tag} ---")

                if mode == "fmb_only":
                    ki = kg_info_base
                    td = train_data  # fmb only
                    vd = valid_data
                else:
                    ki = kg_info_ppi
                    td = concat_fmb_ppi(train_data)
                    vd = {c: concat_fmb_ppi(d) for c, d in valid_data.items()}

                model, val_ci = train_model(model_name, ki, td)
                print(f"  val_ci = {val_ci:.4f}")

                result = {
                    "tag": tag, "model": model_name, "kg": kg, "mode": mode,
                    "val_ci": round(val_ci, 4),
                }

                ext_cis = []
                n_sig = 0
                for c in EVAL_COHORTS:
                    if c not in vd:
                        continue
                    risk = get_risk(model, vd[c])
                    m = compute_all_metrics(
                        risk, vd[c]["time"].numpy(), vd[c]["event"].numpy())
                    result[f"{c}_ci"] = round(m["c_index"], 4)
                    result[f"{c}_p"] = round(m["p_value"], 4)
                    ext_cis.append(m["c_index"])
                    if m["p_value"] < 0.05:
                        n_sig += 1

                result["ext_avg_ci"] = round(np.mean(ext_cis), 4)
                result["n_sig"] = n_sig
                sigs = [c for c in EVAL_COHORTS if result.get(f"{c}_p", 1) < 0.05]
                print(f"  ext_avg_ci={result['ext_avg_ci']:.4f}, sig={n_sig}/11: {sigs}")
                all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("FMB vs FMB+PPI COMPARISON")
    print(f"{'='*80}")
    print(f"{'Tag':<35} {'ValCI':>6} {'ExtCI':>6} {'Sig':>5}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: -x["n_sig"]):
        print(f"{r['tag']:<35} {r['val_ci']:>6.4f} {r['ext_avg_ci']:>6.4f} {r['n_sig']:>3}/11")

    out = EXP_DIR / "ppi_experiment.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
