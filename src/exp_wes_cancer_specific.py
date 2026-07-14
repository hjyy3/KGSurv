"""Cancer-specific WES retraining with a single-cohort training design.

Strategy: pick ONE training cohort, all others (incl. external) go to validation.
This is a stricter cross-cohort generalization test than pooled training.

  Melanoma: train = Liu (n=144)
            holdouts = Riaz, Miao_Mel, Whijae, Hugo, Pleasance_Mel (5 cohorts)

  NSCLC+Lung: train = Ravi (n=306)
              holdouts = Miao_Lung, Pleasance_NSCLC (2 cohorts)

Notes:
  - Multi-node KG features (FMB + PPI + Disease + Drug + ...) computed
    on-the-fly via node_type_ablation helpers with the WES gene list + raw
    mut/mask from output/processed_wes/.
  - 5-fold x 5-seed stratified-by-event CV within the train cohort. Val fold
    metrics double as "internal Liu/Ravi" performance.
  - Fold-result JSON output includes the full metric set for training,
    validation, and individual cohorts.
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

from data_interp import (
    KG_DIR,
    PROC_WES_DIR,
    KGGroupInfo,
    build_kg_group_info,
    load_split_data,
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
from losses import (
    bootstrap_c_index,
    calibration_stats,
    compute_all_metrics,
    compute_arr_nnt,
    compute_dca,
    integrated_brier_score,
)
from models_interp import create_model
from node_type_ablation import ALL_NODE_TYPES, compute_node_features
from train_interp import _seed_everything, evaluate_ci, train_epoch

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "experiments"
SBS_DIR = ROOT / "output" / "sbs_features"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS = 5

# Pleasance cancer_type strings have trailing spaces — always .str.strip() first.
PLEASANCE_NSCLC_TYPES = {
    "Lung Adenocarcinoma",
    "Lung Squamous Cell Carcinoma",
    "Non-Small Cell Lung Cancer",
    "Sarcomatoid Carcinoma of the Lung",
}

# Holdout spec format:
#   name:   label used in per_cohort dict
#   source: "train_pool" (extract from merged train_wes_*) or "external" (valid_{base}_wes_*)
#   base:   original cohort name in metadata
#   filter: None or set of cancer_type strings (post .strip()) to keep
#   sigfeats_key: which key in COHORT_TO_SBS / sigfeats_norm to use for risk-side injection
CANCER_SPECS = {
    "Melanoma": {
        "train_cohort": "Liu",
        "holdouts": [
            {"name": "Riaz",          "source": "train_pool", "base": "Riaz",      "filter": None,                  "sigfeats_key": "Riaz"},
            {"name": "Miao_Mel",      "source": "train_pool", "base": "Miao",      "filter": {"Melanoma"},          "sigfeats_key": "Miao"},
            {"name": "Whijae",        "source": "external",   "base": "Whijae",    "filter": None,                  "sigfeats_key": "Whijae"},
            {"name": "Hugo",          "source": "external",   "base": "Hugo",      "filter": None,                  "sigfeats_key": "Hugo"},
            {"name": "Pleasance_Mel", "source": "external",   "base": "Pleasance", "filter": {"Melanoma"},          "sigfeats_key": "Pleasance"},
        ],
    },
    "NSCLC": {
        "train_cohort": "Ravi",
        "holdouts": [
            {"name": "Miao_Lung",       "source": "train_pool", "base": "Miao",      "filter": {"Lung"},             "sigfeats_key": "Miao"},
            {"name": "Pleasance_NSCLC", "source": "external",   "base": "Pleasance", "filter": PLEASANCE_NSCLC_TYPES, "sigfeats_key": "Pleasance"},
            {"name": "Hellmann",        "source": "external",   "base": "Hellmann",  "filter": None,                 "sigfeats_key": "Hellmann"},
            {"name": "Jung",            "source": "external",   "base": "Jung",      "filter": None,                 "sigfeats_key": "Jung"},
        ],
    },
}

# Multi-node feature recipe per KG (panel-mode best practice)
MULTI_NODE_RECIPE = {
    "drkg": ["ppi", "disease", "drug"],
    "openbiolink": ["ppi", "disease", "drug", "phenotype", "anatomy"],
    "monarch": ["ppi", "disease", "phenotype", "anatomy"],
}

MELANOMA_CONFIGS = {
    "M1_mel_liu_drkg_multinode_sigrisk":        dict(kg="drkg",        sigfeats=True),
    "M2_mel_liu_drkg_multinode_nosig":          dict(kg="drkg",        sigfeats=False),
    "M3_mel_liu_openbiolink_multinode_sigrisk": dict(kg="openbiolink", sigfeats=True),
}
NSCLC_CONFIGS = {
    "N1_nsclc_ravi_drkg_multinode_sigrisk":        dict(kg="drkg",        sigfeats=True),
    "N2_nsclc_ravi_drkg_multinode_nosig":          dict(kg="drkg",        sigfeats=False),
    "N3_nsclc_ravi_openbiolink_multinode_sigrisk": dict(kg="openbiolink", sigfeats=True),
}


# --------------------------------------------------------------------------
# WES I/O helpers
# --------------------------------------------------------------------------


def load_wes_gene_list() -> list[str]:
    return pd.read_csv(PROC_WES_DIR / "wes_candidate_genes.csv")["gene"].tolist()


def load_raw_wes_split(prefix: str) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Read processed_wes/{prefix}_mut.csv & _mask.csv."""
    mut = pd.read_csv(PROC_WES_DIR / f"{prefix}_mut.csv", index_col=0)
    mask = pd.read_csv(PROC_WES_DIR / f"{prefix}_mask.csv", index_col=0)
    return mut.index, mut.values.astype(np.float32), mask.values.astype(np.float32)


def select_by_positions(data: dict, positions: list[int]) -> dict:
    if not positions:
        raise RuntimeError("empty positions list")
    t_idx = torch.tensor(positions, dtype=torch.long)
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[t_idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in positions]
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# Build augmented splits per cancer spec
# --------------------------------------------------------------------------


def build_cancer_splits(
    kg_name: str,
    cancer_spec: dict,
    node_types: list[str],
) -> tuple[dict[str, dict], KGGroupInfo, dict[str, dict]]:
    """Load FMB splits + compute multi-node features for one cancer spec.

    Returns (augmented splits, augmented kg_info, holdout_info).
    holdout_info[name] = {"base": ..., "sigfeats_key": ..., "n": int}.
    """
    base_info = build_kg_group_info(kg_name, wes=True)
    genes = load_wes_gene_list()
    n_genes = len(genes)

    # 1) Load combined train pool (Liu+Riaz+Miao+Ravi+JV101+PUSH) and raw arrays
    train_combined = load_split_data(kg_name, "train", wes=True)
    train_meta = pd.read_csv(PROC_WES_DIR / "train_wes_meta.csv", index_col=0)
    train_meta = train_meta.loc[train_combined["sample_ids"]]

    sids_all, mut_all, mask_all = load_raw_wes_split("train_wes")
    sid_to_pos = {sid: i for i, sid in enumerate(sids_all)}
    aligned_pos = [sid_to_pos[s] for s in train_combined["sample_ids"]]
    mut_all = mut_all[aligned_pos]
    mask_all = mask_all[aligned_pos]

    splits: dict[str, dict] = {}
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    holdout_info: dict[str, dict] = {}

    # 2) Train cohort split
    tc = cancer_spec["train_cohort"]
    train_mask = (train_meta["cohort"].astype(str) == tc).values
    train_positions = list(np.where(train_mask)[0])
    if not train_positions:
        raise RuntimeError(f"train cohort '{tc}' has 0 samples in train pool")
    splits["train"] = select_by_positions(train_combined, train_positions)
    raw["train"] = (mut_all[train_positions], mask_all[train_positions])
    print(f"  train cohort '{tc}': n={len(train_positions)}")

    # 3) Holdouts
    for h in cancer_spec["holdouts"]:
        name, source, base, flt = h["name"], h["source"], h["base"], h["filter"]
        if source == "train_pool":
            cohort_mask = (train_meta["cohort"].astype(str) == base).values
            if flt is not None:
                ct = train_meta["cancer_type"].astype(str).str.strip()
                cohort_mask = cohort_mask & ct.isin(flt).values
            positions = list(np.where(cohort_mask)[0])
            if not positions:
                print(f"  [warn] holdout '{name}' (from train_pool {base}) has 0 samples")
                continue
            splits[name] = select_by_positions(train_combined, positions)
            raw[name] = (mut_all[positions], mask_all[positions])
        else:  # external
            try:
                cdata = load_split_data(kg_name, base, wes=True)
            except FileNotFoundError:
                print(f"  [warn] external cohort '{base}' files missing")
                continue
            ext_sids, ext_mut, ext_mask = load_raw_wes_split(f"valid_{base}_wes")
            ext_sid_to_pos = {sid: i for i, sid in enumerate(ext_sids)}
            ext_pos = [ext_sid_to_pos[s] for s in cdata["sample_ids"]]
            ext_mut = ext_mut[ext_pos]
            ext_mask = ext_mask[ext_pos]
            if flt is not None:
                cmeta = pd.read_csv(PROC_WES_DIR / f"valid_{base}_wes_meta.csv", index_col=0)
                cmeta = cmeta.loc[cdata["sample_ids"]]
                ctypes = cmeta["cancer_type"].astype(str).str.strip()
                keep_mask = ctypes.isin(flt).values
                if not keep_mask.any():
                    print(f"  [warn] holdout '{name}' filter yielded 0 samples in {base}")
                    continue
                keep_pos = list(np.where(keep_mask)[0])
                cdata = select_by_positions(cdata, keep_pos)
                ext_mut = ext_mut[keep_pos]
                ext_mask = ext_mask[keep_pos]
            splits[name] = cdata
            raw[name] = (ext_mut, ext_mask)
        holdout_info[name] = {
            "base": base,
            "sigfeats_key": h["sigfeats_key"],
            "n": len(splits[name]["sample_ids"]),
        }
        print(f"  holdout '{name}' (base={base}, filter={flt}): n={holdout_info[name]['n']}")

    # 4) Compute multi-node features across all splits at once
    extras: list[tuple[str, dict[str, np.ndarray], list[str], str]] = []
    for nt in node_types:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features(kg_name, nt, mode, edict, genes, raw)
        if res is None:
            print(f"  [skip] node_type '{nt}' has no edges for {kg_name}")
            continue
        feats_by_cohort, term_names, _info = res
        extras.append((nt, feats_by_cohort, term_names, mode))
        print(f"  +{nt}: {len(term_names)} terms ({mode})")

    # 5) Augment splits
    augmented: dict[str, dict] = {}
    for c, data in splits.items():
        extra_tensors = []
        for nt, feats_by_cohort, names, _mode in extras:
            if c in feats_by_cohort:
                extra_tensors.append(torch.tensor(feats_by_cohort[c], dtype=torch.float32))
            else:
                extra_tensors.append(
                    torch.zeros(len(data["sample_ids"]), len(names), dtype=torch.float32)
                )
        new = dict(data)
        new["fmb"] = torch.cat([data["fmb"]] + extra_tensors, dim=1)
        augmented[c] = new

    # 6) Augmented KGGroupInfo
    groups = list(base_info.group_names)
    terms = list(base_info.term_names)
    masks = list(base_info.gene_term_mask)
    slices = list(base_info.fmb_slices)
    offset = slices[-1][1] if slices else 0
    total = base_info.n_total_terms
    for nt, _f, term_names, mode in extras:
        groups.append(f"x_{nt}")
        terms.append(term_names)
        n_t = len(term_names)
        m = (
            torch.eye(n_genes, dtype=torch.float32)
            if mode == "adj"
            else torch.ones(n_genes, n_t, dtype=torch.float32) / max(n_genes, 1)
        )
        masks.append(m)
        slices.append((offset, offset + n_t))
        offset += n_t
        total += n_t

    aug_info = KGGroupInfo(
        kg_name=kg_name,
        group_names=groups,
        term_names=terms,
        gene_term_mask=masks,
        fmb_slices=slices,
        n_genes=n_genes,
        n_total_terms=total,
    )
    return augmented, aug_info, holdout_info


# --------------------------------------------------------------------------
# Full evaluation metrics
# --------------------------------------------------------------------------


def full_metrics(risk: np.ndarray, time_: np.ndarray, event: np.ndarray) -> dict:
    """ci/hr/p, AUC@12/24/36, IBS, cal slope/intercept, ARR/NNT@24m, DCA INB, boot CI 95%."""
    n = len(risk)
    nan_block = {k: float("nan") for k in (
        "ci","hr","p","auc_12m","auc_24m","auc_36m","ibs","cal_slope","cal_intercept",
        "arr_24m","nnt_24m","dca_inb","boot_ci","boot_ci_lo","boot_ci_hi",
    )}
    if n < 5:
        return nan_block
    m = compute_all_metrics(risk, time_, event)
    try:
        boot = bootstrap_c_index(risk, time_, event, n_boot=200)
    except Exception:
        boot = {"boot_ci_mean": float("nan"), "boot_ci_lo": float("nan"), "boot_ci_hi": float("nan")}
    try:
        arr = compute_arr_nnt(risk, time_, event)
    except Exception:
        arr = {"arr_24m": float("nan"), "nnt_24m": float("nan")}
    try:
        ibs = integrated_brier_score(risk, time_, event)
    except Exception:
        ibs = float("nan")
    try:
        cal = calibration_stats(risk, time_, event, n_groups=5, t=24.0)
    except Exception:
        cal = {"slope": float("nan"), "intercept": float("nan")}
    try:
        dca = compute_dca(risk, time_, event, t=24.0)
        mdca = (dca["thresholds"] >= 0.1) & (dca["thresholds"] <= 0.5)
        inb = float(np.trapezoid(dca["nb_model"][mdca], dca["thresholds"][mdca])) \
            if mdca.sum() > 1 else float("nan")
    except Exception:
        inb = float("nan")

    def r4(x):
        try:
            x = float(x)
            return round(x, 4) if not np.isnan(x) else float("nan")
        except Exception:
            return float("nan")

    return {
        "ci": r4(m["c_index"]),
        "hr": r4(m["hr"]),
        "p":  r4(m["p_value"]),
        "auc_12m": r4(m.get("auc_12m", float("nan"))),
        "auc_24m": r4(m.get("auc_24m", float("nan"))),
        "auc_36m": r4(m.get("auc_36m", float("nan"))),
        "ibs": r4(ibs),
        "cal_slope": r4(cal.get("slope", float("nan"))),
        "cal_intercept": r4(cal.get("intercept", float("nan"))),
        "arr_24m": r4(arr.get("arr_24m", float("nan"))),
        "nnt_24m": r4(arr.get("nnt_24m", float("nan"))),
        "dca_inb": r4(inb),
        "boot_ci": r4(boot["boot_ci_mean"]),
        "boot_ci_lo": r4(boot["boot_ci_lo"]),
        "boot_ci_hi": r4(boot["boot_ci_hi"]),
    }


# --------------------------------------------------------------------------
# CV utilities
# --------------------------------------------------------------------------


def stratified_kfold_by_event(events: np.ndarray, k: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Stratify k-fold by event status (single-cohort train, no cohort variation)."""
    from sklearn.model_selection import StratifiedKFold

    strata = events.astype(int)
    counts = pd.Series(strata).value_counts()
    if (counts < k).any():
        # fall back to plain shuffle KFold
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        return [
            (torch.tensor(tr, dtype=torch.long), torch.tensor(va, dtype=torch.long))
            for tr, va in kf.split(np.arange(len(strata)))
        ]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [
        (torch.tensor(tr, dtype=torch.long), torch.tensor(va, dtype=torch.long))
        for tr, va in skf.split(np.arange(len(strata)), strata)
    ]


@torch.no_grad()
def get_risk(model, data: dict) -> np.ndarray:
    model.eval()
    kw = {}
    if "extra_risk" in data:
        kw["extra_risk"] = data["extra_risk"].to(device)
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device), **kw)
    return out["log_risk"].cpu().numpy()


def train_one_fold(
    kg_info: KGGroupInfo,
    train_data: dict,
    val_data: dict,
    holdouts: dict[str, dict],
    seed: int,
    n_extra_risk: int,
) -> dict:
    _seed_everything(seed)
    model = create_model(
        "path_attn",
        kg_info,
        hidden_dim=HPARAMS["hidden"],
        dropout=HPARAMS["dropout"],
        n_extra_risk=n_extra_risk,
    )
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
    if best_state is not None:
        model.load_state_dict(best_state)

    # train + val full metrics
    tr_risk = get_risk(model, train_data)
    train_m = full_metrics(tr_risk, train_data["time"].numpy(), train_data["event"].numpy())
    val_risk = get_risk(model, val_data)
    val_m = full_metrics(val_risk, val_data["time"].numpy(), val_data["event"].numpy())

    per_cohort: dict[str, dict] = {}
    cis, sigs, hrs = [], [], []
    for cname, cdata in holdouts.items():
        n_c = len(cdata["sample_ids"])
        events_c = int(cdata["event"].sum().item())
        if n_c < 5:
            per_cohort[cname] = {"n": n_c, "events": events_c,
                                 **{k: float("nan") for k in ("ci","hr","p","auc_12m","auc_24m","auc_36m","ibs",
                                                              "cal_slope","cal_intercept","arr_24m","nnt_24m",
                                                              "dca_inb","boot_ci","boot_ci_lo","boot_ci_hi")},
                                 "sig": False}
            continue
        risk = get_risk(model, cdata)
        m = full_metrics(risk, cdata["time"].numpy(), cdata["event"].numpy())
        per_cohort[cname] = {"n": n_c, "events": events_c, **m,
                             "sig": bool(not np.isnan(m["p"]) and m["p"] < 0.05)}
        if not np.isnan(m["ci"]):
            cis.append(m["ci"])
            hrs.append(max(m["hr"], 1e-6))
            if m["p"] < 0.05:
                sigs.append(cname)

    ext_ci = float(np.mean(cis)) if cis else float("nan")
    pooled_HR_geo = float(np.exp(np.mean(np.log(hrs)))) if hrs else float("nan")
    return {
        "train": train_m,
        "val": val_m,
        "per_cohort": per_cohort,
        "ext_ci_primary": round(ext_ci, 4) if not np.isnan(ext_ci) else float("nan"),
        "n_sig_primary": len(sigs),
        "sigs_primary": sigs,
        "pooled_HR_primary_geo": round(pooled_HR_geo, 4) if not np.isnan(pooled_HR_geo) else float("nan"),
        "best_val_ci_during_training": round(float(best_ci), 4),
    }


# --------------------------------------------------------------------------
# Per-config runner
# --------------------------------------------------------------------------


def run_config(
    cancer: str,
    config_name: str,
    kg: str,
    sigfeats_on: bool,
    sigfeats_norm,
    seeds: list[int],
    n_folds: int,
    out_dir: Path,
) -> dict:
    print()
    print("=" * 70)
    print(f"  {config_name}: cancer={cancer}, kg={kg}, sigfeats={sigfeats_on}")
    print("=" * 70)

    spec = CANCER_SPECS[cancer]
    node_types = MULTI_NODE_RECIPE[kg]
    augmented, kg_info, holdout_info = build_cancer_splits(kg, spec, node_types)
    print(f"  aug kg_info: groups={len(kg_info.group_names)}, total_terms={kg_info.n_total_terms}")

    train_data = augmented["train"]
    holdouts = {name: augmented[name] for name in holdout_info.keys() if name in augmented}

    # sigfeats riskside
    if sigfeats_on and sigfeats_norm is not None:
        # train meta for sigfeats attach (single-cohort meta)
        train_meta = pd.read_csv(PROC_WES_DIR / "train_wes_meta.csv", index_col=0)
        train_meta = train_meta.loc[train_data["sample_ids"]]
        train_data = attach_sigfeats_to_train(train_data, train_meta, sigfeats_norm)
        for name in list(holdouts.keys()):
            sig_key = holdout_info[name]["sigfeats_key"]
            holdouts[name] = attach_sigfeats_to_holdout(holdouts[name], sigfeats_norm, sig_key)
        n_extra_risk = train_data["extra_risk"].shape[1]
        print(f"  sigfeats riskside: train extra_risk shape {train_data['extra_risk'].shape}")
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
            print(f"  seed={seed} fold={fi}: val_ci={r['val']['ci']:.4f} ext_ci={r['ext_ci_primary']} "
                  f"n_sig={r['n_sig_primary']} HR={r['pooled_HR_primary_geo']} ({r['elapsed_s']:.0f}s)")

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

    summary = {
        "config": config_name,
        "cancer": cancer,
        "train_cohort": spec["train_cohort"],
        "kg": kg,
        "sigfeats_on": sigfeats_on,
        "node_types": node_types,
        "n_runs": len(fold_results),
        "holdouts": {n: {"base": info["base"], "n": info["n"]} for n, info in holdout_info.items()},
        "train_ci_mean": round(agg(["train","ci"])[0], 4), "train_ci_std": round(agg(["train","ci"])[1], 4),
        "val_ci_mean":   round(agg(["val","ci"])[0], 4),   "val_ci_std":   round(agg(["val","ci"])[1], 4),
        "ext_ci_primary_mean": round(agg(["ext_ci_primary"])[0], 4),
        "ext_ci_primary_std":  round(agg(["ext_ci_primary"])[1], 4),
        "n_sig_primary_mean":  round(agg(["n_sig_primary"])[0], 2),
        "n_sig_primary_std":   round(agg(["n_sig_primary"])[1], 2),
        "n_sig_primary_max":   int(max((fr["n_sig_primary"] for fr in fold_results), default=0)),
        "pooled_HR_primary_geo_mean": round(agg(["pooled_HR_primary_geo"])[0], 4),
        "pooled_HR_primary_geo_std":  round(agg(["pooled_HR_primary_geo"])[1], 4),
        "val_auc_24m_mean":  round(agg(["val","auc_24m"])[0], 4),
        "val_ibs_mean":      round(agg(["val","ibs"])[0], 4),
        "val_cal_slope_mean":round(agg(["val","cal_slope"])[0], 4),
        "val_arr_24m_mean":  round(agg(["val","arr_24m"])[0], 4),
        "val_dca_inb_mean":  round(agg(["val","dca_inb"])[0], 4),
        "val_boot_ci_mean":  round(agg(["val","boot_ci"])[0], 4),
        "fold_results": fold_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cancer-specific WES retraining (single-cohort train)")
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

    pool = MELANOMA_CONFIGS if args.cancer == "Melanoma" else NSCLC_CONFIGS
    configs = args.configs or list(pool.keys())
    out_path = Path(args.out) if args.out else EXP_DIR / f"wes_cancer_{args.cancer.lower()}.json"
    out_dir = EXP_DIR / f"wes_cancer_{args.cancer.lower()}_folds"

    print("Loading sigfeats ...")
    sigfeats = load_sigfeats()
    sigfeats_norm = normalise_sigfeats(sigfeats)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    for cname in configs:
        if cname not in pool:
            print(f"[skip] unknown config '{cname}' for {args.cancer}")
            continue
        cfg = pool[cname]
        summary = run_config(
            args.cancer, cname, cfg["kg"], cfg["sigfeats"],
            sigfeats_norm if cfg["sigfeats"] else None,
            args.seeds, args.folds, out_dir,
        )
        all_results[cname] = summary
        out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"  -> saved partial to {out_path}")

    print()
    print("Done. Summary:")
    for c, s in all_results.items():
        print(f"  {c}: val_ci={s['val_ci_mean']:.4f}+/-{s['val_ci_std']:.4f}, "
              f"ext_ci={s['ext_ci_primary_mean']:.4f}+/-{s['ext_ci_primary_std']:.4f}, "
              f"n_sig={s['n_sig_primary_mean']:.2f} (max={s['n_sig_primary_max']}), "
              f"pool_HR={s['pooled_HR_primary_geo_mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
