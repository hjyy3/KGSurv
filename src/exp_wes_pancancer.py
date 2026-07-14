"""Track A: Pan-cancer WES retraining experiment."""
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
from losses import compute_all_metrics
from models_interp import create_model
from train_interp import _seed_everything, evaluate_ci, train_epoch

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "experiments"
SBS_DIR = ROOT / "output" / "sbs_features"
PROC_WES = ROOT / "output" / "processed_wes"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cohort split used for pan-cancer WES evaluation.
HOLDOUT_PRIMARY = ["Whijae", "Hugo", "SnyderUC", "Pleasance"]
HOLDOUT_RCC = ["CM214", "Braun"]
HOLDOUT_COHORTS = HOLDOUT_PRIMARY + HOLDOUT_RCC
TRAIN_COHORTS = ["Liu", "Riaz", "Miao", "Ravi", "JV101", "PUSH"]

HPARAMS = dict(lr=1e-3, wd=1e-4, hidden=32, dropout=0.05, batch=64, epochs=80, patience=15)
SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5

COHORT_TO_SBS = {
    "Whijae": "Whijae", "Hugo": "Hugo", "Pleasance": "Pleasance",
    "SnyderUC": "Snyder_UC",
    "Liu": "Liu", "Riaz": "Riaz", "Miao": "Miao", "Ravi": "Ravi",
    "Braun": "Braun", "CM214": "CM214_JV101", "JV101": "CM214_JV101", "PUSH": "PUSH",
}

CONFIGS = {
    "A1_wes_drkg_sigriskside": dict(kg="drkg", sigfeats=True),
    "A2_wes_drkg_nosig":      dict(kg="drkg", sigfeats=False),
    "A3_wes_openbiolink_sigriskside": dict(kg="openbiolink", sigfeats=True),
}


def load_sigfeats():
    out = {}
    for wes_name, sbs_name in COHORT_TO_SBS.items():
        p = SBS_DIR / f"valid_{sbs_name}_sigfeats.csv"
        if not p.exists():
            print(f"  [warn] sigfeats missing for {wes_name} -> {p}")
            continue
        out[wes_name] = pd.read_csv(p, index_col=0)
    return out


def normalise_sigfeats(sigfeats):
    out = {}
    for label, df in sigfeats.items():
        f = df.copy().fillna(0.0)
        for c in ("APOBEC_count", "MMR_indel_burden", "DDR_burden"):
            if c in f.columns:
                f[c] = np.log1p(np.clip(f[c].values, 0, None))
        mu = f.mean(axis=0)
        sd = f.std(axis=0).replace(0.0, 1.0)
        out[label] = (f - mu) / sd
    return out


def attach_sigfeats_to_train(train_data, sample_meta, sigfeats_norm):
    n = len(sample_meta)
    n_feat = next(iter(sigfeats_norm.values())).shape[1]
    sf = np.zeros((n, n_feat), dtype=np.float32)
    for i, (sid, row) in enumerate(sample_meta.iterrows()):
        cohort = row["cohort"]
        if cohort in sigfeats_norm and sid in sigfeats_norm[cohort].index:
            sf[i] = sigfeats_norm[cohort].loc[sid].values.astype(np.float32)
    out = dict(train_data)
    out["extra_risk"] = torch.tensor(sf, dtype=torch.float32)
    return out


def attach_sigfeats_to_holdout(valid_data, sigfeats_norm, cohort):
    n = len(valid_data["sample_ids"])
    if cohort not in sigfeats_norm:
        return valid_data
    sf_df = sigfeats_norm[cohort]
    n_feat = sf_df.shape[1]
    sf = np.zeros((n, n_feat), dtype=np.float32)
    for i, sid in enumerate(valid_data["sample_ids"]):
        if sid in sf_df.index:
            sf[i] = sf_df.loc[sid].values.astype(np.float32)
    out = dict(valid_data)
    out["extra_risk"] = torch.tensor(sf, dtype=torch.float32)
    return out


def stratified_kfold_indices(meta, k, seed):
    from sklearn.model_selection import StratifiedKFold
    strata = (meta["cohort"].astype(str) + "::" + meta["cancer_type"].astype(str)).values
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


def select_data(data, idx):
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx.tolist()]
        else:
            out[k] = v
    return out


@torch.no_grad()
def get_risk(model, data):
    model.eval()
    kw = {}
    if "extra_risk" in data:
        kw["extra_risk"] = data["extra_risk"].to(device)
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device), **kw)
    return out["log_risk"].cpu().numpy()


def train_one_fold(kg_info, train_data, val_data, valid_cohorts, seed, n_extra_risk):
    _seed_everything(seed)
    model = create_model("path_attn", kg_info,
                         hidden_dim=HPARAMS["hidden"], dropout=HPARAMS["dropout"],
                         n_extra_risk=n_extra_risk)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["wd"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    best_ci, pat, best_state = 0.0, 0, None
    for ep in range(1, HPARAMS["epochs"] + 1):
        train_epoch(model, train_data, opt, HPARAMS["batch"], device)
        ci = evaluate_ci(model, val_data, device)
        sch.step(ci)
        if ci > best_ci:
            best_ci, pat = ci, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= HPARAMS["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)

    # Train + val fold full metrics (HR, p-value via compute_all_metrics)
    train_risk = get_risk(model, train_data)
    train_m = compute_all_metrics(
        train_risk, train_data["time"].numpy(), train_data["event"].numpy()
    )
    val_risk = get_risk(model, val_data)
    val_m = compute_all_metrics(
        val_risk, val_data["time"].numpy(), val_data["event"].numpy()
    )

    per_cohort = {}
    primary_cis = []
    primary_sigs = []
    rcc_cis = []
    rcc_sigs = []
    for cname, cdata in valid_cohorts.items():
        risk = get_risk(model, cdata)
        m = compute_all_metrics(risk, cdata["time"].numpy(), cdata["event"].numpy())
        per_cohort[cname] = {
            "ci": round(float(m["c_index"]), 4),
            "p":  round(float(m["p_value"]), 4),
            "hr": round(float(m["hr"]), 4),
        }
        if cname in HOLDOUT_RCC:
            rcc_cis.append(m["c_index"])
            if m["p_value"] < 0.05:
                rcc_sigs.append(cname)
        else:
            primary_cis.append(m["c_index"])
            if m["p_value"] < 0.05:
                primary_sigs.append(cname)

    return {
        "val_ci": round(float(best_ci), 4),
        "train_ci": round(float(train_m["c_index"]), 4),
        "train_hr": round(float(train_m["hr"]), 4),
        "train_p": round(float(train_m["p_value"]), 4),
        "val_hr": round(float(val_m["hr"]), 4),
        "val_p": round(float(val_m["p_value"]), 4),
        "ext_ci": round(float(np.mean(primary_cis)), 4) if primary_cis else 0.0,
        "n_sig": len(primary_sigs),
        "sigs": primary_sigs,
        "rcc_arm_ci": round(float(np.mean(rcc_cis)), 4) if rcc_cis else 0.0,
        "rcc_arm_n_sig": len(rcc_sigs),
        "rcc_arm_sigs": rcc_sigs,
        "per_cohort": per_cohort,
    }


def run_config(config_name, kg, sigfeats_on, sigfeats_norm, seeds, n_folds):
    print()
    print("=" * 60)
    print(f"  {config_name}: kg={kg}, sigfeats={sigfeats_on}")
    print("=" * 60)

    kg_info = build_kg_group_info(kg, wes=True)
    train_data = load_split_data(kg, "train", wes=True)
    train_meta = pd.read_csv(PROC_WES / "train_wes_meta.csv", index_col=0)
    train_meta = train_meta.loc[train_data["sample_ids"]]

    holdouts = {}
    for c in HOLDOUT_COHORTS:
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
        folds = stratified_kfold_indices(train_meta, n_folds, seed=seed)
        for fi, (tr_idx, va_idx) in enumerate(folds):
            t0 = time.time()
            tr_split = select_data(train_data, tr_idx)
            va_split = select_data(train_data, va_idx)
            r = train_one_fold(kg_info, tr_split, va_split, holdouts, seed, n_extra_risk)
            r["seed"] = seed
            r["fold"] = fi
            r["elapsed_s"] = round(time.time() - t0, 1)
            fold_results.append(r)
            print(f"  seed={seed} fold={fi}: val_ci={r['val_ci']:.4f} ext_ci={r['ext_ci']:.4f} n_sig={r['n_sig']} | RCC_ci={r['rcc_arm_ci']:.4f} ({r['elapsed_s']:.0f}s)")

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
        "rcc_arm_ci_mean": round(float(np.mean([r["rcc_arm_ci"] for r in fold_results])), 4),
        "rcc_arm_ci_std":  round(float(np.std([r["rcc_arm_ci"] for r in fold_results])), 4),
        "rcc_arm_n_sig_mean": round(float(np.mean([r["rcc_arm_n_sig"] for r in fold_results])), 2),
        "fold_results": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Track A WES pan-cancer experiment")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=str(EXP_DIR / "wes_pancancer.json"))
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
        if cname not in CONFIGS:
            print(f"[skip] unknown config {cname}")
            continue
        cfg = CONFIGS[cname]
        summary = run_config(cname, cfg["kg"], cfg["sigfeats"],
                             sigfeats_norm if cfg["sigfeats"] else None,
                             args.seeds, args.folds)
        all_results[cname] = summary
        Path(args.out).write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"  -> saved partial to {args.out}")

    print()
    print("Done. Summary:")
    for c, s in all_results.items():
        print(f"  {c}: primary ext_ci={s['ext_ci_mean']:.4f}+/-{s['ext_ci_std']:.4f}, n_sig={s['n_sig_mean']:.1f} (max={s['n_sig_max']}); RCC arm ci={s['rcc_arm_ci_mean']:.4f}+/-{s['rcc_arm_ci_std']:.4f}, n_sig={s['rcc_arm_n_sig_mean']:.1f}/2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
