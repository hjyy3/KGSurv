"""RCC train/val split experiment.

Training: MSK_RCC + one RCC cohort (80/20 internal split)
Validation: remaining two RCC cohorts (fully held-out)
Try all 3 possible training partner choices.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import KG_DIR, PROC_DIR, build_kg_group_info
from losses import c_index, compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch

ROOT = Path(__file__).resolve().parent.parent
RAW_VALID = ROOT / "source" / "input_data" / "valid"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensors(mut, mask, clin, fmb, idx) -> dict:
    return {
        "mut": torch.tensor(mut.loc[idx].values, dtype=torch.float32),
        "mask": torch.tensor(mask.loc[idx].values, dtype=torch.float32),
        "time": torch.tensor(clin.loc[idx, "OS_MONTHS"].values, dtype=torch.float32),
        "event": torch.tensor(clin.loc[idx, "event"].values, dtype=torch.float32),
        "fmb": torch.tensor(fmb.loc[idx].values, dtype=torch.float32),
        "sample_ids": idx.tolist(),
    }


def _pool(*datasets: dict) -> dict:
    pooled = {}
    for key in ["mut", "mask", "time", "event", "fmb"]:
        pooled[key] = torch.cat([d[key] for d in datasets], dim=0)
    pooled["sample_ids"] = sum([d["sample_ids"] for d in datasets], [])
    return pooled


def load_rcc_cohorts(kg_name: str) -> dict[str, dict]:
    kg_feat = KG_DIR / kg_name
    cohorts = {}

    # MSK RCC
    mut = pd.read_csv(PROC_DIR / "train_mut.csv", index_col=0)
    mask_df = pd.read_csv(PROC_DIR / "train_mask.csv", index_col=0)
    clin = pd.read_csv(PROC_DIR / "train_clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / "train_fmb.csv", index_col=0)
    raw = pd.read_csv(ROOT / "source/input_data/train/clin.csv",
                       on_bad_lines="skip", index_col="SAMPLE_ID")
    rcc_ids = raw[raw["CANCER_TYPE"] == "Renal Cell Carcinoma"].index
    common = mut.index.intersection(clin.index).intersection(fmb.index).intersection(rcc_ids)
    cohorts["MSK_RCC"] = _to_tensors(mut, mask_df, clin, fmb, common)

    # CM214 + JV101
    mut2 = pd.read_csv(PROC_DIR / "valid_CM214_JV101_mut.csv", index_col=0)
    mask2 = pd.read_csv(PROC_DIR / "valid_CM214_JV101_mask.csv", index_col=0)
    clin2 = pd.read_csv(PROC_DIR / "valid_CM214_JV101_clin.csv", index_col=0)
    fmb2 = pd.read_csv(kg_feat / "valid_CM214_JV101_fmb.csv", index_col=0)
    raw2 = pd.read_csv(RAW_VALID / "clin_CM214_JV101.csv").set_index("Sample.ID")
    for sub, label in [("CM-214", "CM214"), ("JAVELIN-101", "JV101")]:
        ids = raw2[raw2["Cohort"] == sub].index
        c = mut2.index.intersection(clin2.index).intersection(fmb2.index).intersection(ids)
        cohorts[label] = _to_tensors(mut2, mask2, clin2, fmb2, c)

    # Braun
    mut3 = pd.read_csv(PROC_DIR / "valid_Braun_mut.csv", index_col=0)
    mask3 = pd.read_csv(PROC_DIR / "valid_Braun_mask.csv", index_col=0)
    clin3 = pd.read_csv(PROC_DIR / "valid_Braun_clin.csv", index_col=0)
    fmb3 = pd.read_csv(kg_feat / "valid_Braun_fmb.csv", index_col=0)
    c3 = mut3.index.intersection(clin3.index).intersection(fmb3.index)
    cohorts["Braun"] = _to_tensors(mut3, mask3, clin3, fmb3, c3)

    for name, d in cohorts.items():
        n = d["mut"].shape[0]
        ev = d["event"].sum().int().item()
        print(f"  {name:>10}: n={n}, events={ev}")
    return cohorts


@torch.no_grad()
def _evaluate(model, data) -> float:
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return c_index(out["log_risk"].cpu(), data["time"], data["event"])


@torch.no_grad()
def _get_risk(model, data) -> np.ndarray:
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def train_model(model_name, kg_name, train_data, seed=42):
    _seed_everything(seed)
    train_split, val_split = _split_data(train_data, 0.8, seed)
    n_tr = train_split["mut"].shape[0]
    n_val = val_split["mut"].shape[0]
    print(f"    internal split: train={n_tr}, val={n_val}")

    kg_info = build_kg_group_info(kg_name)
    model = create_model(model_name, kg_info, hidden_dim=32, dropout=0.1)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    best_ci, patience, best_state = 0.0, 0, None
    for ep in range(1, 81):
        train_epoch(model, train_split, opt, 64, device)
        val_ci = _evaluate(model, val_split)
        sched.step(val_ci)
        if val_ci > best_ci:
            best_ci, patience = val_ci, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep <= 5 or patience == 0:
            print(f"    Epoch {ep:3d}: val_ci={val_ci:.4f}  best={best_ci:.4f}")
        if patience >= 15:
            print(f"    Early stop at epoch {ep}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_ci


def main():
    MODELS = ["sparse_path", "path_attn"]
    KGS = ["primekg", "ogb_biokg"]
    PARTNERS = ["Braun", "CM214", "JV101"]  # which cohort joins MSK for training

    all_results = []

    for kg in KGS:
        print(f"\n{'='*60}")
        print(f"  KG = {kg}")
        print(f"{'='*60}")
        cohorts = load_rcc_cohorts(kg)

        for model_name in MODELS:
            for partner in PARTNERS:
                val_names = [c for c in PARTNERS if c != partner]
                train_data = _pool(cohorts["MSK_RCC"], cohorts[partner])
                n_train = train_data["mut"].shape[0]

                print(f"\n--- {model_name} x {kg} | "
                      f"Train: MSK+{partner} (n={n_train}) | "
                      f"Val: {val_names} ---")

                model, int_val_ci = train_model(model_name, kg, train_data)

                result = {
                    "model": model_name, "kg": kg,
                    "train_partner": partner,
                    "train_n": n_train,
                    "internal_val_ci": round(int_val_ci, 4),
                }

                for vname in val_names:
                    vdata = cohorts[vname]
                    risk = _get_risk(model, vdata)
                    m = compute_all_metrics(
                        risk, vdata["time"].numpy(), vdata["event"].numpy())
                    result[f"{vname}_ci"] = round(m["c_index"], 4)
                    result[f"{vname}_hr"] = round(m["hr"], 2)
                    result[f"{vname}_p"] = round(m["p_value"], 4)
                    sig = "*" if m["p_value"] < 0.05 else " "
                    print(f"    {vname:>6}: CI={m['c_index']:.4f}  "
                          f"HR={m['hr']:.2f}  p={m['p_value']:.4f} {sig}")

                all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("RCC TRAIN/VAL SPLIT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<14} {'KG':<12} {'Train (MSK+)':<14} {'N':>4} "
          f"{'IntCI':>6} {'Val1':>18} {'Val2':>18}")
    print("-" * 100)
    for r in all_results:
        partner = r["train_partner"]
        val_names = [c for c in PARTNERS if c != partner]
        v1, v2 = val_names

        def _fmt(name):
            ci = r.get(f"{name}_ci", 0)
            p = r.get(f"{name}_p", 1)
            sig = "**" if p < 0.01 else " *" if p < 0.05 else "  "
            return f"{name} {ci:.3f}{sig}"

        print(f"{r['model']:<14} {r['kg']:<12} {partner:<14} {r['train_n']:>4} "
              f"{r['internal_val_ci']:>6.4f} {_fmt(v1):>18} {_fmt(v2):>18}")

    out = EXP_DIR / "rcc_trainval_experiment.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
