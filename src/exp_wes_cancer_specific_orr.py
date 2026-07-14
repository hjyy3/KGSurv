"""Cancer-specific WES retraining for ORR binary classification.

Companion to exp_wes_cancer_specific.py (which uses OS Cox). This script keeps
the same single-cohort training design but swaps:
  - Endpoint:   OS Cox  ->  ORR binary (BCEWithLogits on log_risk as logit)
  - Eval CI:    val C-index  ->  val AUROC
  - Per-cohort: log-rank p   ->  Mann-Whitney U one-sided p on logits
  - Full metrics: ci/HR/IBS/cal/DCA -> AUROC/AUPRC/F1/Brier/calibration slope

ORR labels are loaded from source/input_data/valid/clin_{cohort}.csv (column
"ORR"; 1 = responder, 0 = non-responder). Samples with NaN ORR are dropped.

Output JSONs go to output/experiments/wes_cancer_orr_{cancer}.json (kept
separate from the OS results so both endpoints can coexist).
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import KGGroupInfo, PROC_WES_DIR
from exp_wes_cancer_specific import (
    CANCER_SPECS,
    MULTI_NODE_RECIPE,
    PLEASANCE_NSCLC_TYPES,
    build_cancer_splits,
    select_by_positions,
    stratified_kfold_by_event,
)
from exp_wes_pancancer import HPARAMS, SEEDS, select_data
from models_interp import create_model
from train_interp import _seed_everything

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "experiments"
RAW_CLIN_DIR = ROOT / "source" / "input_data" / "valid"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS = 5

# Map our cohort names to raw clin_{name}.csv files (handles Miao_Mel etc -> Miao)
COHORT_TO_RAW_CLIN = {
    "Liu": "Liu", "Riaz": "Riaz", "Ravi": "Ravi", "Miao": "Miao",
    "Miao_Mel": "Miao", "Miao_Lung": "Miao",
    "Hugo": "Hugo", "Whijae": "Whijae",
    "Pleasance": "Pleasance", "Pleasance_Mel": "Pleasance", "Pleasance_NSCLC": "Pleasance",
}

# Per-cancer config menu (separate from OS configs to keep output JSONs clean)
MELANOMA_CONFIGS_ORR = {
    "M1_orr_mel_liu_drkg_multinode":      dict(kg="drkg",        sigfeats=True),
    "M2_orr_mel_liu_drkg_multinode_nosig": dict(kg="drkg",       sigfeats=False),
    "M3_orr_mel_liu_openbiolink_multinode": dict(kg="openbiolink", sigfeats=True),
}
NSCLC_CONFIGS_ORR = {
    "N1_orr_nsclc_ravi_drkg_multinode":      dict(kg="drkg",        sigfeats=True),
    "N2_orr_nsclc_ravi_drkg_multinode_nosig": dict(kg="drkg",       sigfeats=False),
    "N3_orr_nsclc_ravi_openbiolink_multinode": dict(kg="openbiolink", sigfeats=True),
    # No-TMB canonical models (per directive: do not feed TMB as input).
    # Pure KG-pathway; node recipe = ablation-optimal FMB+PPI+Disease.
    "NT1_orr_nsclc_ravi_drkg_ppidisease_notmb": dict(kg="drkg", sigfeats=False,
                                                     node_types=["ppi", "disease"], use_tmb=False),
    "NT2_orr_nsclc_ravi_drkg_multinode_notmb":  dict(kg="drkg", sigfeats=False,
                                                     node_types=["ppi", "disease", "drug"], use_tmb=False),
}


# --------------------------------------------------------------------------
# ORR labels
# --------------------------------------------------------------------------


def load_orr_labels() -> dict[str, int]:
    """Build a global Sample.ID -> ORR (0/1) map from all raw clin files."""
    out: dict[str, int] = {}
    for cohort_file in sorted(RAW_CLIN_DIR.glob("clin_*.csv")):
        df = pd.read_csv(cohort_file)
        if "Sample.ID" not in df.columns or "ORR" not in df.columns:
            continue
        for _, row in df.iterrows():
            orr = row["ORR"]
            if pd.isna(orr):
                continue
            try:
                out[str(row["Sample.ID"])] = int(orr)
            except Exception:
                continue
    return out


def attach_orr_filter(data: dict, orr_map: dict[str, int], cohort_label: str) -> dict | None:
    """Attach `data['orr']` tensor; drop samples with no ORR label.

    Returns None if cohort has <5 labelled samples (too small to eval).
    """
    keep_pos: list[int] = []
    labels: list[int] = []
    for i, sid in enumerate(data["sample_ids"]):
        # sample_ids may carry cohort-prefixed format; raw clin uses raw Sample.ID
        # Strip common prefixes (e.g. "Riaz_Pt01" -> "Pt01") if needed; but
        # Miao_Mel reuses Miao IDs as-is. Try direct lookup first, then prefix-stripped.
        if sid in orr_map:
            keep_pos.append(i)
            labels.append(orr_map[sid])
            continue
        # try stripping "{cohort}_" prefix
        for pfx in (cohort_label + "_", "Miao_", "Ravi_", "Liu_", "Riaz_", "Hugo_",
                    "Whijae_", "Pleasance_"):
            if sid.startswith(pfx):
                trimmed = sid[len(pfx):]
                if trimmed in orr_map:
                    keep_pos.append(i)
                    labels.append(orr_map[trimmed])
                    break
    if len(keep_pos) < 5:
        return None
    new = select_by_positions(data, keep_pos)
    new["orr"] = torch.tensor(labels, dtype=torch.float32)
    return new


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------


def _safe(x: float) -> float:
    try:
        x = float(x)
        return round(x, 4) if not np.isnan(x) else float("nan")
    except Exception:
        return float("nan")


def orr_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """AUROC, AUPRC, F1@0.5, accuracy, Brier, calibration slope, MW p, bootstrap AUROC 95%."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score, accuracy_score, brier_score_loss,
    )
    from scipy.stats import mannwhitneyu
    nan_block = {k: float("nan") for k in (
        "auroc", "auprc", "f1", "acc", "brier", "cal_slope", "cal_intercept",
        "mw_p", "boot_auroc", "boot_auroc_lo", "boot_auroc_hi",
        "mean_p_responder", "mean_p_nonresponder",
    )}
    nan_block["n"] = int(len(y))
    nan_block["n_responder"] = int(np.sum(y == 1))
    nan_block["n_nonresponder"] = int(np.sum(y == 0))
    nan_block["sig"] = False
    if len(y) < 5 or nan_block["n_responder"] == 0 or nan_block["n_nonresponder"] == 0:
        return nan_block

    probs = 1.0 / (1.0 + np.exp(-logits))
    try:
        auroc = roc_auc_score(y, logits)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y, logits)
    except Exception:
        auprc = float("nan")
    preds = (probs >= 0.5).astype(int)
    try:
        f1 = f1_score(y, preds, zero_division=0)
        acc = accuracy_score(y, preds)
    except Exception:
        f1 = float("nan"); acc = float("nan")
    try:
        brier = brier_score_loss(y, probs)
    except Exception:
        brier = float("nan")

    # Calibration: 5-quantile observed vs mean predicted prob; slope of linear fit
    try:
        order = np.argsort(probs)
        n_groups = min(5, len(np.unique(probs)))
        if n_groups >= 2:
            chunks = np.array_split(order, n_groups)
            obs = np.array([y[c].mean() for c in chunks])
            pred = np.array([probs[c].mean() for c in chunks])
            if pred.std() > 0:
                slope = float(np.polyfit(pred, obs, 1)[0])
                intercept = float(np.polyfit(pred, obs, 1)[1])
            else:
                slope = float("nan"); intercept = float("nan")
        else:
            slope = float("nan"); intercept = float("nan")
    except Exception:
        slope = float("nan"); intercept = float("nan")

    # Mann-Whitney U one-sided: responders have higher logit?
    try:
        u, mw_p = mannwhitneyu(logits[y == 1], logits[y == 0], alternative="greater")
    except Exception:
        mw_p = float("nan")

    # Bootstrap AUROC 95% CI
    try:
        rng = np.random.RandomState(42)
        n = len(y)
        bs = []
        for _ in range(200):
            idx = rng.randint(0, n, size=n)
            yb = y[idx]
            if yb.sum() == 0 or yb.sum() == n:
                continue
            bs.append(roc_auc_score(yb, logits[idx]))
        if bs:
            boot_auroc = float(np.mean(bs))
            boot_lo = float(np.percentile(bs, 2.5))
            boot_hi = float(np.percentile(bs, 97.5))
        else:
            boot_auroc = boot_lo = boot_hi = float("nan")
    except Exception:
        boot_auroc = boot_lo = boot_hi = float("nan")

    return {
        "n": int(len(y)),
        "n_responder": int(y.sum()),
        "n_nonresponder": int(len(y) - y.sum()),
        "auroc": _safe(auroc),
        "auprc": _safe(auprc),
        "f1": _safe(f1),
        "acc": _safe(acc),
        "brier": _safe(brier),
        "cal_slope": _safe(slope),
        "cal_intercept": _safe(intercept),
        "mw_p": _safe(mw_p),
        "boot_auroc": _safe(boot_auroc),
        "boot_auroc_lo": _safe(boot_lo),
        "boot_auroc_hi": _safe(boot_hi),
        "mean_p_responder": _safe(probs[y == 1].mean()),
        "mean_p_nonresponder": _safe(probs[y == 0].mean()),
        "sig": bool(not np.isnan(mw_p) and mw_p < 0.05 and not np.isnan(auroc) and auroc > 0.5),
    }


# --------------------------------------------------------------------------
# ORR training / eval
# --------------------------------------------------------------------------


def train_epoch_bce(model, data, optimizer, batch_size, device):
    model.train()
    n = data["mut"].shape[0]
    perm = torch.randperm(n)
    has_extra = "extra_risk" in data
    loss_fn = nn.BCEWithLogitsLoss()
    total, nb = 0.0, 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        b_mut = data["mut"][idx].to(device)
        b_mask = data["mask"][idx].to(device)
        b_fmb = data["fmb"][idx].to(device)
        b_y = data["orr"][idx].to(device)
        kw = {}
        if has_extra:
            kw["extra_risk"] = data["extra_risk"][idx].to(device)
        optimizer.zero_grad()
        out = model(b_mut, b_mask, b_fmb, **kw)
        logit = out["log_risk"]
        loss = loss_fn(logit, b_y)
        if torch.isnan(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / max(nb, 1)


@torch.no_grad()
def get_logits(model, data):
    model.eval()
    kw = {}
    if "extra_risk" in data:
        kw["extra_risk"] = data["extra_risk"].to(device)
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device), **kw)
    return out["log_risk"].cpu().numpy()


def eval_auroc(model, data) -> float:
    from sklearn.metrics import roc_auc_score
    logits = get_logits(model, data)
    y = data["orr"].numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, logits))


def train_one_fold_orr(kg_info, train_data, val_data, holdouts, seed, n_extra_risk, use_tmb=True):
    _seed_everything(seed)
    model = create_model(
        "path_attn", kg_info,
        hidden_dim=HPARAMS["hidden"], dropout=HPARAMS["dropout"],
        n_extra_risk=n_extra_risk, use_tmb=use_tmb,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["wd"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    best, pat, best_state = 0.0, 0, None
    for ep in range(1, HPARAMS["epochs"] + 1):
        train_epoch_bce(model, train_data, opt, HPARAMS["batch"], device)
        auroc = eval_auroc(model, val_data)
        if np.isnan(auroc):
            break
        sch.step(auroc)
        if auroc > best:
            best, pat = auroc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= HPARAMS["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    tr_logits = get_logits(model, train_data)
    train_m = orr_metrics(tr_logits, train_data["orr"].numpy())
    va_logits = get_logits(model, val_data)
    val_m = orr_metrics(va_logits, val_data["orr"].numpy())

    per_cohort = {}
    aurocs, sigs = [], []
    for cname, cdata in holdouts.items():
        if cdata is None:
            continue
        lo = get_logits(model, cdata)
        m = orr_metrics(lo, cdata["orr"].numpy())
        per_cohort[cname] = m
        if not np.isnan(m["auroc"]):
            aurocs.append(m["auroc"])
            if m["sig"]:
                sigs.append(cname)

    ext_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    return {
        "train": train_m,
        "val": val_m,
        "per_cohort": per_cohort,
        "ext_auroc_primary": round(ext_auroc, 4) if not np.isnan(ext_auroc) else float("nan"),
        "n_sig_primary": len(sigs),
        "sigs_primary": sigs,
        "best_val_auroc": round(float(best), 4),
    }


# --------------------------------------------------------------------------
# Sigfeats helper (riskside attach, reusing OS pipeline)
# --------------------------------------------------------------------------


def attach_sigfeats_orr(data: dict, sigfeats_norm: dict, cohort_label: str) -> dict:
    """Risk-side sigfeats attachment for ORR splits. Uses sigfeats indexed by
    sample id; falls back to zero vector for missing ids."""
    if cohort_label not in sigfeats_norm:
        return data
    sf_df = sigfeats_norm[cohort_label]
    n_feat = sf_df.shape[1]
    arr = np.zeros((len(data["sample_ids"]), n_feat), dtype=np.float32)
    for i, sid in enumerate(data["sample_ids"]):
        if sid in sf_df.index:
            arr[i] = sf_df.loc[sid].values.astype(np.float32)
    new = dict(data)
    new["extra_risk"] = torch.tensor(arr, dtype=torch.float32)
    return new


def attach_train_sigfeats(data: dict, sigfeats_norm: dict, train_cohort: str) -> dict:
    return attach_sigfeats_orr(data, sigfeats_norm, train_cohort)


# --------------------------------------------------------------------------
# Per-config runner
# --------------------------------------------------------------------------


def run_config_orr(
    cancer: str,
    config_name: str,
    kg: str,
    sigfeats_on: bool,
    sigfeats_norm,
    orr_map: dict[str, int],
    seeds: list[int],
    n_folds: int,
    out_dir: Path,
    node_types: list[str] | None = None,
    use_tmb: bool = True,
) -> dict:
    print()
    print("=" * 70)
    print(f"  {config_name}: cancer={cancer}, kg={kg}, sigfeats={sigfeats_on} (ORR)")
    print("=" * 70)

    spec = CANCER_SPECS[cancer]
    if node_types is None:
        node_types = MULTI_NODE_RECIPE[kg]
    print(f"  node_types: {node_types}")
    augmented, kg_info, holdout_info = build_cancer_splits(kg, spec, node_types)
    print(f"  aug kg_info: groups={len(kg_info.group_names)}, total_terms={kg_info.n_total_terms}")

    # Attach ORR labels + drop NaN-ORR samples per cohort
    train_data = attach_orr_filter(augmented["train"], orr_map, spec["train_cohort"])
    if train_data is None:
        raise RuntimeError(f"train cohort '{spec['train_cohort']}' has <5 ORR-labelled samples")
    print(f"  train ORR: n={len(train_data['sample_ids'])}, "
          f"R={int(train_data['orr'].sum().item())}, "
          f"NR={int(len(train_data['sample_ids']) - train_data['orr'].sum().item())}")

    holdouts: dict[str, dict] = {}
    for name in holdout_info.keys():
        if name not in augmented:
            continue
        base = holdout_info[name]["base"]
        hd = attach_orr_filter(augmented[name], orr_map, base)
        if hd is None:
            print(f"  [warn] holdout '{name}' has <5 ORR-labelled -> drop")
            continue
        holdouts[name] = hd
        print(f"  holdout '{name}': n={len(hd['sample_ids'])}, "
              f"R={int(hd['orr'].sum().item())}, "
              f"NR={int(len(hd['sample_ids']) - hd['orr'].sum().item())}")

    # Sigfeats riskside
    if sigfeats_on and sigfeats_norm is not None:
        train_data = attach_train_sigfeats(train_data, sigfeats_norm, spec["train_cohort"])
        for name in list(holdouts.keys()):
            holdouts[name] = attach_sigfeats_orr(
                holdouts[name], sigfeats_norm, holdout_info[name]["sigfeats_key"])
        n_extra_risk = train_data["extra_risk"].shape[1]
        print(f"  sigfeats riskside: extra_risk shape={train_data['extra_risk'].shape}")
    else:
        n_extra_risk = 0

    cfg_dir = out_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Stratify CV by ORR label (binary balanced)
    y_np = train_data["orr"].numpy()
    fold_results = []
    for seed in seeds:
        folds = stratified_kfold_by_event(y_np, n_folds, seed)
        for fi, (tr_idx, va_idx) in enumerate(folds):
            fold_json = cfg_dir / f"seed{seed}_fold{fi}.json"
            if fold_json.exists():
                fold_results.append(json.loads(fold_json.read_text(encoding="utf-8")))
                print(f"  seed={seed} fold={fi}: cached -> skip")
                continue
            t0 = time.time()
            tr_split = select_data(train_data, tr_idx)
            va_split = select_data(train_data, va_idx)
            r = train_one_fold_orr(kg_info, tr_split, va_split, holdouts, seed, n_extra_risk, use_tmb=use_tmb)
            r["seed"] = int(seed); r["fold"] = int(fi)
            r["elapsed_s"] = round(time.time() - t0, 1)
            fold_json.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
            fold_results.append(r)
            print(f"  seed={seed} fold={fi}: val_auroc={r['val']['auroc']} "
                  f"ext_auroc={r['ext_auroc_primary']} n_sig={r['n_sig_primary']} ({r['elapsed_s']:.0f}s)")

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
        "config": config_name,
        "cancer": cancer,
        "train_cohort": spec["train_cohort"],
        "kg": kg,
        "sigfeats_on": sigfeats_on,
        "node_types": node_types,
        "endpoint": "ORR",
        "n_runs": len(fold_results),
        "holdouts": {n: {"base": info["base"], "n": info["n"]} for n, info in holdout_info.items()},
        "train_auroc_mean": round(agg(["train", "auroc"])[0], 4),
        "train_auroc_std":  round(agg(["train", "auroc"])[1], 4),
        "val_auroc_mean":   round(agg(["val", "auroc"])[0], 4),
        "val_auroc_std":    round(agg(["val", "auroc"])[1], 4),
        "val_auprc_mean":   round(agg(["val", "auprc"])[0], 4),
        "val_brier_mean":   round(agg(["val", "brier"])[0], 4),
        "val_cal_slope_mean": round(agg(["val", "cal_slope"])[0], 4),
        "val_f1_mean":      round(agg(["val", "f1"])[0], 4),
        "ext_auroc_primary_mean": round(agg(["ext_auroc_primary"])[0], 4),
        "ext_auroc_primary_std":  round(agg(["ext_auroc_primary"])[1], 4),
        "n_sig_primary_mean":  round(agg(["n_sig_primary"])[0], 2),
        "n_sig_primary_std":   round(agg(["n_sig_primary"])[1], 2),
        "n_sig_primary_max":   int(max((fr["n_sig_primary"] for fr in fold_results), default=0)),
        "fold_results": fold_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Cancer-specific WES ORR classification")
    parser.add_argument("--cancer", required=True, choices=list(CANCER_SPECS.keys()))
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.folds = 2

    pool = MELANOMA_CONFIGS_ORR if args.cancer == "Melanoma" else NSCLC_CONFIGS_ORR
    configs = args.configs or list(pool.keys())
    out_path = Path(args.out) if args.out else EXP_DIR / f"wes_cancer_orr_{args.cancer.lower()}.json"
    out_dir = EXP_DIR / f"wes_cancer_orr_{args.cancer.lower()}_folds"

    print("Loading ORR labels ...")
    orr_map = load_orr_labels()
    print(f"  ORR labels loaded: {len(orr_map)} samples global")

    print("Loading sigfeats ...")
    from exp_wes_pancancer import load_sigfeats, normalise_sigfeats
    sigfeats_norm = normalise_sigfeats(load_sigfeats())

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    for cname in configs:
        if cname not in pool:
            print(f"[skip] unknown config '{cname}' for {args.cancer}")
            continue
        cfg = pool[cname]
        summary = run_config_orr(
            args.cancer, cname, cfg["kg"], cfg["sigfeats"],
            sigfeats_norm if cfg["sigfeats"] else None,
            orr_map, args.seeds, args.folds, out_dir,
            node_types=cfg.get("node_types"),
            use_tmb=cfg.get("use_tmb", True),
        )
        all_results[cname] = summary
        out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"  -> saved partial to {out_path}")

    print()
    print("Done. Summary:")
    for c, s in all_results.items():
        print(f"  {c}: val_auroc={s['val_auroc_mean']:.4f}+/-{s['val_auroc_std']:.4f}, "
              f"ext_auroc={s['ext_auroc_primary_mean']:.4f}+/-{s['ext_auroc_primary_std']:.4f}, "
              f"n_sig={s['n_sig_primary_mean']:.2f} (max={s['n_sig_primary_max']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
