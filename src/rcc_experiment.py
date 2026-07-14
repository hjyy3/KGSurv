"""RCC-specific training experiment.

Strategy:
1. Extract RCC samples from MSK train set (151 samples)
2. Split CM214_JV101 into CM-214 (182) and JAVELIN-101 (283)
3. Pool all RCC data (MSK_RCC + CM214 + JV101 + Braun = 877)
4. Train SparsePathNet × {ogb_biokg, primekg} with 80/20 split
5. Evaluate on held-out split
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
    KG_DIR,
    PROC_DIR,
    SUBKG_DIR,
    build_kg_group_info,
)
from losses import c_index, compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, _split_data, train_epoch

ROOT = Path(__file__).resolve().parent.parent
RAW_VALID = ROOT / "source" / "input_data" / "valid"
EXP_DIR = ROOT / "output" / "experiments"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading: build RCC-only tensors
# ---------------------------------------------------------------------------

def _load_rcc_from_msk(kg_name: str, feat_dir: Path | None = None) -> dict:
    """Extract RCC samples from MSK training set."""
    kg_feat = feat_dir if feat_dir is not None else KG_DIR / kg_name

    mut = pd.read_csv(PROC_DIR / "train_mut.csv", index_col=0)
    mask = pd.read_csv(PROC_DIR / "train_mask.csv", index_col=0)
    clin = pd.read_csv(PROC_DIR / "train_clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / "train_fmb.csv", index_col=0)

    # raw clinical has CANCER_TYPE
    raw_clin = pd.read_csv(
        ROOT / "source/input_data/train/clin.csv",
        on_bad_lines="skip",
        index_col="SAMPLE_ID",
    )
    rcc_ids = raw_clin[raw_clin["CANCER_TYPE"] == "Renal Cell Carcinoma"].index
    common = mut.index.intersection(clin.index).intersection(fmb.index).intersection(rcc_ids)
    print(f"  MSK RCC: {len(common)} samples")

    return _to_tensors(mut, mask, clin, fmb, common)


def _load_cohort_split(kg_name: str, cohort_tag: str,
                       feat_dir: Path | None = None) -> tuple[dict | None, dict | None]:
    """Load CM214_JV101 and split by Cohort column into CM-214 and JAVELIN-101."""
    kg_feat = feat_dir if feat_dir is not None else KG_DIR / kg_name

    mut = pd.read_csv(PROC_DIR / "valid_CM214_JV101_mut.csv", index_col=0)
    mask = pd.read_csv(PROC_DIR / "valid_CM214_JV101_mask.csv", index_col=0)
    clin = pd.read_csv(PROC_DIR / "valid_CM214_JV101_clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / "valid_CM214_JV101_fmb.csv", index_col=0)

    raw_clin = pd.read_csv(RAW_VALID / "clin_CM214_JV101.csv")
    raw_clin = raw_clin.set_index("Sample.ID")

    results = {}
    for sub_cohort in ["CM-214", "JAVELIN-101"]:
        sub_ids = raw_clin[raw_clin["Cohort"] == sub_cohort].index
        common = mut.index.intersection(clin.index).intersection(fmb.index).intersection(sub_ids)
        if len(common) == 0:
            print(f"  {sub_cohort}: 0 samples (skipped)")
            results[sub_cohort] = None
            continue
        print(f"  {sub_cohort}: {len(common)} samples")
        results[sub_cohort] = _to_tensors(mut, mask, clin, fmb, common)

    return results.get("CM-214"), results.get("JAVELIN-101")


def _load_braun(kg_name: str, feat_dir: Path | None = None) -> dict:
    """Load Braun (RCC) cohort."""
    kg_feat = feat_dir if feat_dir is not None else KG_DIR / kg_name
    mut = pd.read_csv(PROC_DIR / "valid_Braun_mut.csv", index_col=0)
    mask = pd.read_csv(PROC_DIR / "valid_Braun_mask.csv", index_col=0)
    clin = pd.read_csv(PROC_DIR / "valid_Braun_clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / "valid_Braun_fmb.csv", index_col=0)
    common = mut.index.intersection(clin.index).intersection(fmb.index)
    print(f"  Braun: {len(common)} samples")
    return _to_tensors(mut, mask, clin, fmb, common)


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
    """Concatenate multiple data dicts."""
    pooled = {}
    for key in ["mut", "mask", "time", "event", "fmb"]:
        pooled[key] = torch.cat([d[key] for d in datasets], dim=0)
    pooled["sample_ids"] = sum([d["sample_ids"] for d in datasets], [])
    return pooled


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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


def train_rcc_model(model_name: str, kg_name: str, train_data: dict,
                    val_data: dict, seed: int = 42):
    """Train model on RCC data, return (model, results_dict)."""
    _seed_everything(seed)

    train_split, val_split = _split_data(train_data, 0.8, seed)
    print(f"  Train: {train_split['mut'].shape[0]}, Val: {val_split['mut'].shape[0]}")

    kg_info = build_kg_group_info(kg_name)
    model = create_model(model_name, kg_info, hidden_dim=32, dropout=0.1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=7, factor=0.5, min_lr=1e-6
    )

    best_ci, patience, best_state = 0.0, 0, None
    for ep in range(1, 81):
        loss = train_epoch(model, train_split, opt, 64, device)
        val_ci = _evaluate(model, val_split)
        sched.step(val_ci)
        if val_ci > best_ci:
            best_ci = val_ci
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep <= 10 or ep % 10 == 0 or patience == 0:
            print(f"  Epoch {ep:3d}: loss={loss:.4f}  val_ci={val_ci:.4f}  best={best_ci:.4f}")
        if patience >= 15:
            print(f"  Early stop at epoch {ep}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_ci


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS = ["sparse_path", "path_attn"]
    KGS = ["ogb_biokg", "primekg"]

    all_results = []

    for kg in KGS:
        print(f"\n{'='*60}")
        print(f"  Loading RCC data with KG={kg}")
        print(f"{'='*60}")

        msk_rcc = _load_rcc_from_msk(kg)
        cm214, jv101 = _load_cohort_split(kg, "CM214_JV101")
        braun = _load_braun(kg)

        # Pool all RCC
        pool_parts = [msk_rcc]
        if cm214:
            pool_parts.append(cm214)
        if jv101:
            pool_parts.append(jv101)
        pool_parts.append(braun)
        pooled = _pool(*pool_parts)
        print(f"  Pooled RCC: {pooled['mut'].shape[0]} samples, "
              f"events={pooled['event'].sum().int().item()}")

        for model_name in MODELS:
            print(f"\n--- {model_name} x {kg} (RCC pooled) ---")
            t0 = time.time()

            model, best_val_ci = train_rcc_model(model_name, kg, pooled, pooled)
            elapsed = time.time() - t0

            result = {
                "model": model_name, "kg": kg,
                "rcc_pool_n": pooled["mut"].shape[0],
                "val_ci": round(best_val_ci, 4),
                "elapsed_s": round(elapsed, 1),
            }

            # Evaluate on each sub-cohort individually
            for name, data in [("MSK_RCC", msk_rcc), ("CM214", cm214),
                               ("JV101", jv101), ("Braun", braun)]:
                if data is None:
                    continue
                risk = _get_risk(model, data)
                m = compute_all_metrics(risk, data["time"].numpy(), data["event"].numpy())
                result[f"{name}_n"] = data["mut"].shape[0]
                result[f"{name}_ci"] = round(m["c_index"], 4)
                result[f"{name}_hr"] = round(m["hr"], 2)
                result[f"{name}_p"] = round(m["p_value"], 4)
                print(f"  {name:>10}: n={data['mut'].shape[0]:4d}  "
                      f"CI={m['c_index']:.4f}  HR={m['hr']:.2f}  p={m['p_value']:.4f}")

            all_results.append(result)

            # Save model
            out_dir = EXP_DIR / f"rcc_{model_name}_{kg}"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "model.pt")

    # Summary
    print(f"\n{'='*70}")
    print("RCC EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model+KG':<30} {'ValCI':>6} {'MSK':>6} {'CM214':>6} "
          f"{'JV101':>6} {'Braun':>6}")
    print("-" * 70)
    for r in all_results:
        tag = f"{r['model']}_{r['kg']}"
        print(f"{tag:<30} {r['val_ci']:>6.4f} "
              f"{r.get('MSK_RCC_ci',''):>6} {r.get('CM214_ci',''):>6} "
              f"{r.get('JV101_ci',''):>6} {r.get('Braun_ci',''):>6}")

    out_file = EXP_DIR / "rcc_experiment.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
