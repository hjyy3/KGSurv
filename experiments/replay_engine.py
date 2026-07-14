"""Single-run engine: train → risks → fixed cutoffs → per-cohort metrics.

Idempotent: skips if runs/<run_id>/done.flag exists.
Artifacts written:
  config.json        — full spec + hyperparams + library versions + cutoffs
  rng_state.pt       — RNG snapshot taken after lock + before training
  risks.npz          — train + per-cohort risk arrays, time, event
  model.pt           — only saved if max(c2_n_sig, c3_n_sig) >= save_ckpt_min_sig
  per_cohort.json    — final summary (consumed by aggregate.py)
  done.flag          — completion marker
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(EXP_ROOT / "code"))

from kg_features import load_candidate_genes  # noqa: E402
from kfold_cv import (  # noqa: E402
    _select, kfold_indices,
    EPOCHS, PATIENCE, HIDDEN, DROPOUT, LR, WD, BATCH,
)
from data_interp import KGGroupInfo, KG_DIR, build_kg_group_info  # noqa: E402
from models_interp import create_model  # noqa: E402
from multi_node_extended import (  # noqa: E402
    augment_splits, load_base_splits, EVAL_COHORTS,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features  # noqa: E402
from train_interp import train_epoch, evaluate_ci  # noqa: E402

from spec import ExperimentSpec  # noqa: E402
from seeding import lock_determinism, rng_snapshot, state_dict_hash  # noqa: E402
from cutoffs import (  # noqa: E402
    per_cohort_median, compute_all_train_cutoffs, logrank_split, CUTOFF_SCHEMA,
)

RUNS_DIR = EXP_ROOT / "runs"

# Cache (aug, info) per (kg, effective_kg, node_types). KG features are
# pure functions of the KG; reusing them avoids re-running the buggy
# pandas iterrows path inside compute_node_features which has been
# observed to corrupt pandas internal state after repeated calls.
_KG_CACHE: dict[tuple[str, str, tuple[str, ...]], tuple] = {}


def _build_combo_info(kg_base: str, eff_kg: str, extra_term_lists, n_genes: int):
    """Variant-aware build_combo_info:
    - subkg + group connectivity comes from kg_base
    - FMB column parsing + metadata comes from eff_kg (via feat_dir override)

    Identical to multi_node_extended.build_combo_info when kg_base == eff_kg.
    """
    import torch as _t
    feat_dir = KG_DIR / eff_kg if eff_kg != kg_base else None
    base = build_kg_group_info(kg_base, feat_dir=feat_dir)
    groups = list(base.group_names)
    terms = list(base.term_names)
    masks = list(base.gene_term_mask)
    slices = list(base.fmb_slices)
    offset = slices[-1][1] if slices else 0
    total = base.n_total_terms

    for tnames, mode, label in extra_term_lists:
        groups.append(label)
        terms.append(tnames)
        n_t = len(tnames)
        m = (_t.eye(n_genes, dtype=_t.float32) if mode == "adj"
             else _t.ones(n_genes, n_t, dtype=_t.float32) / max(n_genes, 1))
        masks.append(m)
        slices.append((offset, offset + n_t))
        offset += n_t
        total += n_t

    return KGGroupInfo(kg_name=kg_base, group_names=groups, term_names=terms,
                       gene_term_mask=masks, fmb_slices=slices,
                       n_genes=n_genes, n_total_terms=total)


def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return None


def _load_kg_data(kg: str, node_types, effective_kg: str | None = None):
    """Load splits + KGGroupInfo, cached per (kg, effective_kg, node_types).

    kg           — base KG name used for subkg lookup (node-type features)
    effective_kg — KG variant name used for FMB + metadata (e.g. hetionet_mg2)
                   defaults to kg when no FMB variant is requested
    """
    eff = effective_kg or kg
    key = (kg, eff, tuple(node_types))
    if key in _KG_CACHE:
        return _KG_CACHE[key]

    genes = load_candidate_genes()
    splits, raw = load_base_splits(eff)
    extra_f, extra_i = [], []
    for nt in node_types:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features(kg, nt, mode, edict, genes, raw)
        if res is None:
            raise RuntimeError(f"compute_node_features → None for kg={kg} nt={nt}")
        feats, tnames, mat = res
        extra_f.append(feats)
        extra_i.append((tnames, mat, f"x_{nt}"))
    aug = augment_splits(splits, extra_f)
    info = _build_combo_info(kg, eff, extra_i, len(genes))
    _KG_CACHE[key] = (aug, info)
    return aug, info


@torch.no_grad()
def _get_risk(model, data, device) -> np.ndarray:
    model.eval()
    out = model(
        data["mut"].to(device),
        data["mask"].to(device),
        data["fmb"].to(device),
    )
    return out["log_risk"].cpu().numpy()


def _train(info, tr, va, device, model_name: str):
    model = create_model(model_name, info, hidden_dim=HIDDEN, dropout=DROPOUT)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=7, factor=0.5, min_lr=1e-6,
    )
    best_ci, patience_left, best_state = 0.0, 0, None
    ep = 0
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, tr, opt, BATCH, device)
        ci = evaluate_ci(model, va, device)
        sch.step(ci)
        if ci > best_ci:
            best_ci = ci
            patience_left = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left += 1
        if patience_left >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_ci), ep


def run_one(
    spec: ExperimentSpec,
    device: torch.device,
    save_ckpt_min_sig: int = 6,
    roc_time: float = 24.0,
) -> dict:
    run_dir = RUNS_DIR / spec.run_id
    done = run_dir / "done.flag"
    if done.exists():
        with open(run_dir / "per_cohort.json", encoding="utf-8") as f:
            return json.load(f)

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lock_determinism(spec.seed, device.type)
    aug, info = _load_kg_data(spec.kg, spec.node_types,
                               effective_kg=spec.effective_kg)
    n = aug["train"]["mut"].shape[0]
    folds = kfold_indices(n, 5, seed=spec.seed)
    tr_idx, va_idx = folds[spec.fold]
    tr = _select(aug["train"], tr_idx)
    va = _select(aug["train"], va_idx)

    # Re-lock right before training so model.init uses identical RNG stream
    lock_determinism(spec.seed, device.type)
    init_rng = rng_snapshot()
    model, best_val_ci, n_epochs = _train(info, tr, va, device, spec.model)

    train_risks = _get_risk(model, tr, device)
    train_time = tr["time"].numpy()
    train_event = tr["event"].numpy()

    cohort_risks, cohort_t, cohort_e = {}, {}, {}
    for c in EVAL_COHORTS:
        if c not in aug:
            continue
        cd = aug[c]
        cohort_risks[c] = _get_risk(model, cd, device)
        cohort_t[c] = cd["time"].numpy()
        cohort_e[c] = cd["event"].numpy()

    train_med = float(np.median(train_risks))
    cutoff_meta = compute_all_train_cutoffs(train_risks, train_time, train_event)

    per_cohort = {}
    for c in cohort_risks:
        risk = cohort_risks[c]
        coh_med = per_cohort_median(risk)
        cd = {
            "n": int(len(risk)),
            "events": int(cohort_e[c].sum()),
            "c1": {"thr": coh_med, "rule": "per_cohort_median",
                   **logrank_split(risk, cohort_t[c], cohort_e[c], coh_med)},
        }
        for cid, _label in CUTOFF_SCHEMA:
            thr = cutoff_meta[cid]["threshold"]
            cd[cid] = {"thr": float(thr),
                       "rule": cutoff_meta[cid].get("rule", cid),
                       **logrank_split(risk, cohort_t[c], cohort_e[c], thr)}
        per_cohort[c] = cd

    all_cutoff_ids = ["c1"] + [cid for cid, _ in CUTOFF_SCHEMA]
    n_sig = {cid: sum(1 for v in per_cohort.values() if v[cid]["sig"])
             for cid in all_cutoff_ids}
    sigs = {cid: [c for c, v in per_cohort.items() if v[cid]["sig"]]
            for cid in all_cutoff_ids}
    fixed_max = max(n_sig[cid] for cid in
                    ["c2","c3","c4","c5","c6","c7","c8","c9","c10"])

    sd = model.state_dict()
    sd_hash = state_dict_hash(sd)
    elapsed = time.time() - t0

    config = {
        "spec": spec.to_dict(),
        "device": device.type,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "best_val_ci": best_val_ci,
        "n_epochs_run": n_epochs,
        "epochs_max": EPOCHS,
        "hidden": HIDDEN, "dropout": DROPOUT,
        "lr": LR, "wd": WD, "batch": BATCH, "patience": PATIENCE,
        "roc_time": roc_time,
        "train_n": int(len(train_risks)),
        "train_median": train_med,
        "cutoff_meta": cutoff_meta,
        "fixed_max_n_sig": fixed_max,
        "state_dict_hash": sd_hash,
        "elapsed_s": elapsed,
    }

    # Risks always saved (small, needed for replay)
    np.savez(
        run_dir / "risks.npz",
        train=train_risks,
        train_time=train_time,
        train_event=train_event,
        **{f"risk_{c}": v for c, v in cohort_risks.items()},
        **{f"time_{c}": v for c, v in cohort_t.items()},
        **{f"event_{c}": v for c, v in cohort_e.items()},
    )
    torch.save(init_rng, run_dir / "rng_state.pt")

    if fixed_max >= save_ckpt_min_sig:
        torch.save(sd, run_dir / "model.pt")

    summary = {
        "run_id": spec.run_id,
        "spec": spec.to_dict(),
        "best_val_ci": best_val_ci,
        "n_sig": n_sig,
        "sigs": sigs,
        "per_cohort": per_cohort,
        "config": config,
    }
    with open(run_dir / "per_cohort.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=_jsonable)
    done.write_text("ok\n")
    return summary
