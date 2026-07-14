"""Track B: Melanoma WES retraining experiment."""
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

from data_interp import build_kg_group_info, load_split_data
from exp_wes_pancancer import (
    HPARAMS, SEEDS, COHORT_TO_SBS, PROC_WES,
    load_sigfeats, normalise_sigfeats,
    attach_sigfeats_to_train, attach_sigfeats_to_holdout,
    select_data, get_risk, train_one_fold,
)
from losses import compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, evaluate_ci, train_epoch

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "experiments"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Track B holdouts: Mel-only (Whijae + Hugo)
HOLDOUT_COHORTS_MEL = ["Whijae", "Hugo"]
# Track B train: Liu + Riaz + Miao(Melanoma subset only)
TRAIN_COHORTS_MEL = ["Liu", "Riaz", "Miao"]
N_FOLDS = 5

CONFIGS_MEL = {
    "B1_wes_mel_drkg_sigriskside": dict(kg="drkg", sigfeats=True),
    "B2_wes_mel_drkg_nosig":      dict(kg="drkg", sigfeats=False),
}


def filter_mel(train_data, train_meta):
    """Subset train_data to Melanoma samples only (per-cohort cancer_type)."""
    is_mel = train_meta["cancer_type"].astype(str).str.lower().str.contains("melanoma")
    mel_ids = train_meta.index[is_mel].tolist()
    keep_mask = np.array([sid in set(mel_ids) for sid in train_data["sample_ids"]])
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        raise RuntimeError("No Melanoma samples found in train pool")
    out = {}
    for k, v in train_data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[torch.tensor(keep_idx, dtype=torch.long)]
        elif isinstance(v, list):
            out[k] = [v[i] for i in keep_idx.tolist()]
        else:
            out[k] = v
    new_meta = train_meta.loc[[train_data["sample_ids"][i] for i in keep_idx]]
    return out, new_meta


def stratified_kfold_mel(meta, k, seed):
    """Stratify Mel-only folds by cohort (since cancer_type==Melanoma all)."""
    from sklearn.model_selection import StratifiedKFold
    strata = meta["cohort"].astype(str).values
    counts = pd.Series(strata).value_counts()
    rare = counts[counts < k].index
    if len(rare) > 0:
        strata = np.where(np.isin(strata, rare), "OTHER", strata)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    indices = np.arange(len(meta))
    for tr, va in skf.split(indices, strata):
        folds.append((torch.tensor(tr, dtype=torch.long), torch.tensor(va, dtype=torch.long)))
    return folds


def run_config_mel(config_name, kg, sigfeats_on, sigfeats_norm, seeds, n_folds):
    print()
    print("=" * 60)
    print(f"  {config_name}: kg={kg}, sigfeats={sigfeats_on}")
    print("=" * 60)

    kg_info = build_kg_group_info(kg, wes=True)
    train_data = load_split_data(kg, "train", wes=True)
    train_meta = pd.read_csv(PROC_WES / "train_wes_meta.csv", index_col=0)
    train_meta = train_meta.loc[train_data["sample_ids"]]

    # Filter to Mel only
    train_data, train_meta = filter_mel(train_data, train_meta)
    print(f"  Mel-only train pool: n={len(train_data['sample_ids'])}")

    holdouts = {}
    for c in HOLDOUT_COHORTS_MEL:
        holdouts[c] = load_split_data(kg, c, wes=True)

    if sigfeats_on and sigfeats_norm is not None:
        train_data = attach_sigfeats_to_train(train_data, train_meta, sigfeats_norm)
        for c in list(holdouts.keys()):
            holdouts[c] = attach_sigfeats_to_holdout(holdouts[c], sigfeats_norm, c)
        n_extra_risk = train_data["extra_risk"].shape[1]
        print(f"  sigfeats riskside: train extra_risk shape {train_data['extra_risk'].shape}")
    else:
        n_extra_risk = 0

    fold_results = []
    for seed in seeds:
        folds = stratified_kfold_mel(train_meta, n_folds, seed=seed)
        for fi, (tr_idx, va_idx) in enumerate(folds):
            t0 = time.time()
            tr_split = select_data(train_data, tr_idx)
            va_split = select_data(train_data, va_idx)
            r = train_one_fold(kg_info, tr_split, va_split, holdouts, seed, n_extra_risk)
            r["seed"] = seed
            r["fold"] = fi
            r["elapsed_s"] = round(time.time() - t0, 1)
            fold_results.append(r)
            print(f"  seed={seed} fold={fi}: val_ci={r['val_ci']:.4f} ext_ci={r['ext_ci']:.4f} n_sig={r['n_sig']} ({r['elapsed_s']:.0f}s)")

    val_cis = [r["val_ci"] for r in fold_results]
    ext_cis = [r["ext_ci"] for r in fold_results]
    n_sigs = [r["n_sig"] for r in fold_results]
    return {
        "config": config_name,
        "kg": kg,
        "sigfeats_on": sigfeats_on,
        "n_runs": len(fold_results),
        "val_ci_mean": round(float(np.mean(val_cis)), 4),
        "val_ci_std":  round(float(np.std(val_cis)), 4),
        "ext_ci_mean": round(float(np.mean(ext_cis)), 4),
        "ext_ci_std":  round(float(np.std(ext_cis)), 4),
        "n_sig_mean":  round(float(np.mean(n_sigs)), 2),
        "n_sig_std":   round(float(np.std(n_sigs)), 2),
        "n_sig_max":   int(np.max(n_sigs)),
        "n_sig_min":   int(np.min(n_sigs)),
        "fold_results": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Track B WES melanoma experiment")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS_MEL.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=str(EXP_DIR / "wes_melanoma.json"))
    args = parser.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.folds = 2

    print("Loading sigfeats ...")
    sigfeats = load_sigfeats()
    sigfeats_norm = normalise_sigfeats(sigfeats)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for cname in args.configs:
        if cname not in CONFIGS_MEL:
            print(f"[skip] unknown config {cname}")
            continue
        cfg = CONFIGS_MEL[cname]
        summary = run_config_mel(cname, cfg["kg"], cfg["sigfeats"],
                                 sigfeats_norm if cfg["sigfeats"] else None,
                                 args.seeds, args.folds)
        all_results[cname] = summary
        Path(args.out).write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"  -> saved partial to {args.out}")

    print()
    print("Done. Summary:")
    for c, s in all_results.items():
        print(f"  {c}: ext_ci={s['ext_ci_mean']:.4f}+/-{s['ext_ci_std']:.4f}, n_sig={s['n_sig_mean']:.1f}+/-{s['n_sig_std']:.1f} (max={s['n_sig_max']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
