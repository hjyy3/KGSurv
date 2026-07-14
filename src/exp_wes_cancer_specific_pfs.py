"""Cancer-specific WES retraining for the PFS time-to-event endpoint.

Companion to:
  - exp_wes_cancer_specific.py      (OS Cox; smoke-only, OS reverse-direction evidence)
  - exp_wes_cancer_specific_orr.py  (ORR binary; primary)

This script validates the same single-cohort NSCLC model on the **PFS** endpoint,
which is the natural time-to-event counterpart to ORR and is fully available for
the two new OS-less cohorts (Hellmann, Jung) plus Ravi/Miao.

  NSCLC: train = Ravi (PFS), holdouts = Miao_Lung, Hellmann, Jung (all PFS).
         Pleasance_NSCLC has no PFS and is auto-dropped (<5 labelled).

Design:
  - Reuses build_cancer_splits (mut/mask/FMB + multi-node) + the OS Cox
    train_one_fold + full_metrics from exp_wes_cancer_specific.py.
  - Overrides every split's time/event with canonical PFS (months) + progression
    event from source/input_data/valid/clin_<cohort>.csv, so training AND
    evaluation use PFS consistently (Ravi processed clin stores OS, so the
    override is required, not optional).
  - Metrics follow the OS Fold-Result schema (ci/hr/p, AUC@t, IBS, calibration,
    ARR/NNT, DCA, bootstrap CI); n_sig_primary = log-rank p<0.05 holdout count.

Output: output/experiments/wes_cancer_pfs_nsclc.json + _summary.csv + _per_cohort.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_wes_cancer_specific import (
    CANCER_SPECS,
    MULTI_NODE_RECIPE,
    build_cancer_splits,
    full_metrics,
    stratified_kfold_by_event,
    train_one_fold,
)
from exp_wes_pancancer import (
    HPARAMS,
    SEEDS,
    attach_sigfeats_to_holdout,
    attach_sigfeats_to_train,
    load_sigfeats,
    normalise_sigfeats,
    select_data,
)

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "experiments"
RAW_CLIN_DIR = ROOT / "source" / "input_data" / "valid"
PROC_WES = ROOT / "output" / "processed_wes"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5

NSCLC_CONFIGS_PFS = {
    "P1_pfs_nsclc_ravi_drkg_multinode":       dict(kg="drkg",        sigfeats=True),
    "P2_pfs_nsclc_ravi_drkg_multinode_nosig": dict(kg="drkg",        sigfeats=False),
    "P3_pfs_nsclc_ravi_openbiolink_multinode": dict(kg="openbiolink", sigfeats=True),
}

# Best-config node-ablation winner from ORR track (FMB+PPI+Disease) for parity.
NSCLC_PFS_A3 = {
    "P4_pfs_nsclc_ravi_drkg_fmb_ppi_disease": dict(kg="drkg", sigfeats=True,
                                                   node_types=["ppi", "disease"]),
}


def load_pfs_labels() -> dict[str, tuple[float, int]]:
    """Global Sample.ID -> (PFS_months, progression_event). Drops NaN PFS."""
    out: dict[str, tuple[float, int]] = {}
    for f in sorted(RAW_CLIN_DIR.glob("clin_*.csv")):
        df = pd.read_csv(f)
        if "Sample.ID" not in df.columns or "Progression.free.survival" not in df.columns:
            continue
        pfs = pd.to_numeric(df["Progression.free.survival"], errors="coerce")
        if "Progression..status" in df.columns:
            evt = pd.to_numeric(df["Progression..status"], errors="coerce")
        else:
            evt = pd.Series([np.nan] * len(df))
        for sid, t, e in zip(df["Sample.ID"].astype(str), pfs, evt):
            if pd.isna(t) or pd.isna(e):
                continue
            out[sid] = (float(t), int(e))
    return out


def attach_pfs(data: dict, pfs_map: dict[str, tuple[float, int]]) -> dict | None:
    """Override time/event with PFS; drop samples lacking a PFS label.

    Returns None if <5 PFS-labelled samples remain (cohort un-evaluable)."""
    keep, times, events = [], [], []
    for i, sid in enumerate(data["sample_ids"]):
        if sid in pfs_map:
            keep.append(i)
            t, e = pfs_map[sid]
            times.append(t)
            events.append(e)
    if len(keep) < 5:
        return None
    t_idx = torch.tensor(keep, dtype=torch.long)
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[t_idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in keep]
        else:
            out[k] = v
    out["time"] = torch.tensor(times, dtype=torch.float32)
    out["event"] = torch.tensor(events, dtype=torch.float32)
    return out


def run_config_pfs(config_name, kg, sigfeats_on, sigfeats_norm, pfs_map,
                   seeds, n_folds, out_dir, node_types=None):
    print()
    print("=" * 70)
    print(f"  {config_name}: kg={kg}, sigfeats={sigfeats_on} (PFS)")
    print("=" * 70)

    spec = CANCER_SPECS["NSCLC"]
    if node_types is None:
        node_types = MULTI_NODE_RECIPE[kg]
    print(f"  node_types: {node_types}")
    augmented, kg_info, holdout_info = build_cancer_splits(kg, spec, node_types)

    # Override time/event with PFS for train + holdouts
    train_data = attach_pfs(augmented["train"], pfs_map)
    if train_data is None:
        raise RuntimeError("train cohort has <5 PFS-labelled samples")
    print(f"  train PFS: n={len(train_data['sample_ids'])}, "
          f"events={int(train_data['event'].sum().item())}, "
          f"median_PFS={float(train_data['time'].median()):.1f}")

    holdouts: dict[str, dict] = {}
    for name in holdout_info.keys():
        if name not in augmented:
            continue
        hd = attach_pfs(augmented[name], pfs_map)
        if hd is None:
            print(f"  [drop] holdout '{name}': <5 PFS-labelled samples")
            continue
        holdouts[name] = hd
        print(f"  holdout '{name}': n={len(hd['sample_ids'])}, "
              f"events={int(hd['event'].sum().item())}, "
              f"median_PFS={float(hd['time'].median()):.1f}")

    # Sigfeats riskside (Ravi has sbs; new cohorts get neutral zeros)
    if sigfeats_on and sigfeats_norm is not None:
        train_meta = pd.read_csv(PROC_WES / "train_wes_meta.csv", index_col=0)
        train_meta = train_meta.loc[train_data["sample_ids"]]
        train_data = attach_sigfeats_to_train(train_data, train_meta, sigfeats_norm)
        for name in list(holdouts.keys()):
            sig_key = holdout_info[name]["sigfeats_key"]
            holdouts[name] = attach_sigfeats_to_holdout(holdouts[name], sigfeats_norm, sig_key)
        n_extra_risk = train_data["extra_risk"].shape[1]
        print(f"  sigfeats riskside: extra_risk shape {train_data['extra_risk'].shape}")
    else:
        n_extra_risk = 0

    cfg_dir = out_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)

    events_np = train_data["event"].numpy()
    fold_results = []
    for seed in seeds:
        folds = stratified_kfold_by_event(events_np, n_folds, seed)
        for fi, (tr_idx, va_idx) in enumerate(folds):
            fold_json = cfg_dir / f"seed{seed}_fold{fi}.json"
            if fold_json.exists():
                fold_results.append(json.loads(fold_json.read_text(encoding="utf-8")))
                print(f"  seed={seed} fold={fi}: cached -> skip")
                continue
            t0 = time.time()
            tr_split = select_data(train_data, tr_idx)
            va_split = select_data(train_data, va_idx)
            r = train_one_fold(kg_info, tr_split, va_split, holdouts, seed, n_extra_risk)
            r["seed"] = int(seed)
            r["fold"] = int(fi)
            r["elapsed_s"] = round(time.time() - t0, 1)
            fold_json.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
            fold_results.append(r)
            print(f"  seed={seed} fold={fi}: val_ci={r['val']['ci']:.4f} "
                  f"ext_ci={r['ext_ci_primary']} n_sig={r['n_sig_primary']} "
                  f"HR={r['pooled_HR_primary_geo']} ({r['elapsed_s']:.0f}s)")

    def agg(chain):
        vals = []
        for fr in fold_results:
            v = fr
            for k in chain:
                v = v[k] if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        return (float(np.mean(vals)) if vals else float("nan"),
                float(np.std(vals)) if vals else float("nan"))

    return {
        "config": config_name, "cancer": "NSCLC", "train_cohort": "Ravi",
        "kg": kg, "sigfeats_on": sigfeats_on, "node_types": node_types,
        "endpoint": "PFS", "n_runs": len(fold_results),
        "holdouts": {n: {"base": info["base"], "n": info["n"]} for n, info in holdout_info.items()},
        "train_ci_mean": round(agg(["train", "ci"])[0], 4),
        "val_ci_mean":   round(agg(["val", "ci"])[0], 4),
        "val_ci_std":    round(agg(["val", "ci"])[1], 4),
        "ext_ci_primary_mean": round(agg(["ext_ci_primary"])[0], 4),
        "ext_ci_primary_std":  round(agg(["ext_ci_primary"])[1], 4),
        "n_sig_primary_mean":  round(agg(["n_sig_primary"])[0], 2),
        "n_sig_primary_max":   int(max((fr["n_sig_primary"] for fr in fold_results), default=0)),
        "pooled_HR_primary_geo_mean": round(agg(["pooled_HR_primary_geo"])[0], 4),
        "val_auc_24m_mean": round(agg(["val", "auc_24m"])[0], 4),
        "val_ibs_mean":     round(agg(["val", "ibs"])[0], 4),
        "fold_results": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Cancer-specific WES PFS validation")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--with-a3", action="store_true", help="also run P4 (FMB+PPI+Disease)")
    parser.add_argument("--out", default=str(EXP_DIR / "wes_cancer_pfs_nsclc.json"))
    args = parser.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.folds = 2

    pool = dict(NSCLC_CONFIGS_PFS)
    if args.with_a3:
        pool.update(NSCLC_PFS_A3)
    configs = args.configs or list(pool.keys())
    out_path = Path(args.out)
    out_dir = EXP_DIR / "wes_cancer_pfs_nsclc_folds"

    print("Loading PFS labels ...")
    pfs_map = load_pfs_labels()
    print(f"  PFS labels loaded: {len(pfs_map)} samples global")

    print("Loading sigfeats ...")
    sigfeats_norm = normalise_sigfeats(load_sigfeats())

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    for cname in configs:
        if cname not in pool:
            print(f"[skip] unknown config '{cname}'")
            continue
        cfg = pool[cname]
        summary = run_config_pfs(
            cname, cfg["kg"], cfg["sigfeats"],
            sigfeats_norm if cfg["sigfeats"] else None,
            pfs_map, args.seeds, args.folds, out_dir,
            node_types=cfg.get("node_types"),
        )
        all_results[cname] = summary
        out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"  -> saved partial to {out_path}")

    print()
    print("Done. PFS summary:")
    for c, s in all_results.items():
        print(f"  {c}: val_ci={s['val_ci_mean']:.4f}+/-{s['val_ci_std']:.4f}, "
              f"ext_ci={s['ext_ci_primary_mean']:.4f}+/-{s['ext_ci_primary_std']:.4f}, "
              f"n_sig={s['n_sig_primary_mean']:.2f} (max={s['n_sig_primary_max']}), "
              f"pool_HR={s['pooled_HR_primary_geo_mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
