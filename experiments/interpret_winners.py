"""3-level interpretability extractor for the 3 deployment-winner PathAttnSurv models.

LOADS saved model.pt weights (NO retraining). For each winner:
  1. Rebuild (augmented_data, kg_info) exactly as replay_engine does for its spec.
  2. create_model("path_attn", kg_info, hidden_dim=32, dropout=0.1) + load_state_dict.
  3. CORRECTNESS GATE: state_dict_hash(model) must equal config.json state_dict_hash.
  4. SANITY: forward log_risk on each EVAL cohort must allclose saved risks.npz.
  5. forward_with_importance on MSK train (spec fold train split) + 11 EVAL cohorts.
  6. Aggregate group/term/gene importance: pooled-EVAL and NSCLC-subset views,
     split by deployment-threshold high/low risk groups.

Outputs (only under results/interp_winners/):
  <short>_group_importance.csv, <short>_top_terms.csv, <short>_top_genes.csv
  interp_winners.json (all detail + hash/risk sanity, numpy-safe)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(EXP_ROOT / "code"))

from kg_features import load_candidate_genes  # noqa: E402
from kfold_cv import _select, kfold_indices, HIDDEN, DROPOUT  # noqa: E402
from data_interp import KGGroupInfo, KG_DIR, build_kg_group_info  # noqa: E402
from models_interp import create_model  # noqa: E402
from multi_node_extended import augment_splits, load_base_splits, EVAL_COHORTS  # noqa: E402
from node_type_ablation import ALL_NODE_TYPES, compute_node_features  # noqa: E402
from seeding import state_dict_hash  # noqa: E402

RUNS_DIR = EXP_ROOT / "runs"
OUT_DIR = EXP_ROOT / "results" / "interp_winners"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_CLIN_DIR = PROJ_ROOT / "source" / "input_data" / "valid"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_TERMS = 20
TOP_GENES = 30

# (run_id, short, kg, node_types, fmb_variant, seed, fold, deploy_thr, deploy_rule)
WINNERS = [
    ("path_attn_ibkh_ppi+disease+drug_seed63_fold2", "ibkh",
     "ibkh", ["ppi", "disease", "drug"], None, 63, 2, 0.5123345851898193, "roc12m(c8)"),
    ("path_attn_primekg_mg2_ppi_seed74_fold2", "primekg_mg2",
     "primekg", ["ppi"], "mg2", 74, 2, -0.7085622549057007, "train_mean(c10)"),
    ("path_attn_hetionet_mg5_ppi+disease+drug+anatomy+regulatory_seed69_fold3", "hetionet_mg5",
     "hetionet", ["ppi", "disease", "drug", "anatomy", "regulatory"], "mg5", 69, 3,
     -0.8391652703285217, "train_mean(c10)"),
]


# ---------------------------------------------------------------------------
# Data + kg_info build (mirrors replay_engine._build_combo_info / _load_kg_data)
# ---------------------------------------------------------------------------
def _build_combo_info(kg_base, eff_kg, extra_term_lists, n_genes):
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
        m = (torch.eye(n_genes, dtype=torch.float32) if mode == "adj"
             else torch.ones(n_genes, n_t, dtype=torch.float32) / max(n_genes, 1))
        masks.append(m)
        slices.append((offset, offset + n_t))
        offset += n_t
        total += n_t
    return KGGroupInfo(kg_name=kg_base, group_names=groups, term_names=terms,
                       gene_term_mask=masks, fmb_slices=slices,
                       n_genes=n_genes, n_total_terms=total)


def load_kg_data(kg, node_types, effective_kg):
    genes = load_candidate_genes()
    splits, raw = load_base_splits(effective_kg)
    extra_f, extra_i = [], []
    for nt in node_types:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features(kg, nt, mode, edict, genes, raw)
        if res is None:
            raise RuntimeError(f"compute_node_features None for kg={kg} nt={nt}")
        feats, tnames, mat = res
        extra_f.append(feats)
        extra_i.append((tnames, mat, f"x_{nt}"))
    aug = augment_splits(splits, extra_f)
    info = _build_combo_info(kg, effective_kg, extra_i, len(genes))
    return aug, info, genes


@torch.no_grad()
def forward_with_importance(model, data):
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return {
        "log_risk": out["log_risk"].cpu().numpy(),
        "group_imp": out["group_importance"].cpu().numpy(),
        "term_imp": out["term_importance"].cpu().numpy(),
        "gene_imp": out["gene_importance"].cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# NSCLC mask via per-sample Cancer_type (source clin CSV, matched on Sample.ID)
# ---------------------------------------------------------------------------
def is_nsclc(label):
    s = str(label).lower()
    return ("lung" in s) or ("nsclc" in s)


def cancer_type_map(cohort):
    p = SRC_CLIN_DIR / f"clin_{cohort}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "Sample.ID" not in df.columns or "Cancer_type" not in df.columns:
        return {}
    return dict(zip(df["Sample.ID"].astype(str), df["Cancer_type"].astype(str)))


def _jsonable(x):
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return None


def mean_axis0(x):
    if x.shape[0] == 0:
        return np.full(x.shape[1], np.nan)
    return x.mean(axis=0)


def main():
    summary = {"winners": {}}

    for (run_id, short, kg, node_types, fmb_variant, seed, fold,
         deploy_thr, deploy_rule) in WINNERS:
        print(f"\n{'='*72}\nWINNER: {short}  ({run_id})\n{'='*72}")
        run_dir = RUNS_DIR / run_id
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        expected_hash = cfg["state_dict_hash"]
        effective_kg = f"{kg}_{fmb_variant}" if fmb_variant else kg

        # NOTE: deliberately NOT calling lock_determinism here. It enables
        # torch.use_deterministic_algorithms(True), which breaks numpy.copyto
        # inside pandas .iterrows() (used by extract_function_gene_map) under
        # this numpy build. We only LOAD weights and verify via state_dict_hash,
        # so RNG determinism is irrelevant to correctness; the hash gate proves
        # the architecture matches the trained model bit-for-bit.
        aug, info, genes = load_kg_data(kg, node_types, effective_kg)
        print(f"  groups={info.group_names}  n_total_terms={info.n_total_terms}  n_genes={len(genes)}")

        # Build model + load weights (NO training)
        model = create_model("path_attn", info, hidden_dim=HIDDEN, dropout=DROPOUT)
        model.to(device)
        state = torch.load(run_dir / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        # ---- CORRECTNESS GATE: state_dict_hash ----
        got_hash = state_dict_hash(model.state_dict())
        hash_pass = (got_hash == expected_hash)
        print(f"  HASH expected={expected_hash}")
        print(f"  HASH got     ={got_hash}")
        print(f"  HASH {'PASS' if hash_pass else 'FAIL'}")
        win_rec = {
            "run_id": run_id, "short": short, "kg": kg, "node_types": node_types,
            "fmb_variant": fmb_variant, "effective_kg": effective_kg,
            "seed": seed, "fold": fold,
            "deploy_threshold": deploy_thr, "deploy_rule": deploy_rule,
            "group_names": list(info.group_names),
            "hash_expected": expected_hash, "hash_got": got_hash,
            "hash_pass": bool(hash_pass),
        }
        if not hash_pass:
            win_rec["status"] = "STOPPED_HASH_MISMATCH"
            summary["winners"][short] = win_rec
            print("  !! HASH MISMATCH -> skipping importance for this winner.")
            continue

        # ---- SANITY: forward log_risk vs saved risks.npz ----
        rnpz = np.load(run_dir / "risks.npz")
        sanity = {}
        first_cohort_check = None
        eval_present = [c for c in EVAL_COHORTS if c in aug]
        for c in eval_present:
            fr = forward_with_importance(model, aug[c])["log_risk"]
            sr = rnpz[f"risk_{c}"]
            ok = bool(np.allclose(fr, sr, atol=1e-4))
            md = float(np.max(np.abs(fr - sr)))
            sanity[c] = {"allclose_1e-4": ok, "max_abs_diff": md, "n": int(len(fr))}
            if first_cohort_check is None:
                first_cohort_check = (c, ok, md)
        all_sanity_ok = all(v["allclose_1e-4"] for v in sanity.values())
        c0, ok0, md0 = first_cohort_check
        print(f"  SANITY {c0}: forward vs saved risk allclose={ok0} (max_abs_diff={md0:.2e})")
        print(f"  SANITY all {len(sanity)} cohorts allclose: {all_sanity_ok}")
        win_rec["sanity_forward_vs_saved"] = sanity
        win_rec["sanity_all_ok"] = bool(all_sanity_ok)

        # ---- MSK train split (spec fold) ----
        n_tr_full = aug["train"]["mut"].shape[0]
        folds = kfold_indices(n_tr_full, 5, seed=seed)
        tr_idx, _va_idx = folds[fold]
        tr_split = _select(aug["train"], tr_idx)
        train_out = forward_with_importance(model, tr_split)
        train_saved = rnpz["train"]
        train_sanity_ok = bool(
            train_out["log_risk"].shape == train_saved.shape
            and np.allclose(np.sort(train_out["log_risk"]), np.sort(train_saved), atol=1e-4)
        )
        win_rec["train_n_fold"] = int(len(tr_idx))
        win_rec["train_forward_vs_saved_sorted_allclose"] = train_sanity_ok
        print(f"  TRAIN fold n={len(tr_idx)}  forward-vs-saved(sorted) allclose={train_sanity_ok}")

        # ---- Collect EVAL importance (pooled across cohorts) + NSCLC mask ----
        gi_parts, ti_parts, ge_parts, lr_parts, nsclc_mask = [], [], [], [], []
        per_cohort_n = {}
        for c in eval_present:
            out = forward_with_importance(model, aug[c])
            gi_parts.append(out["group_imp"])
            ti_parts.append(out["term_imp"])
            ge_parts.append(out["gene_imp"])
            lr_parts.append(out["log_risk"])
            ctmap = cancer_type_map(c)
            sids = aug[c]["sample_ids"]
            m = np.array([is_nsclc(ctmap.get(str(sid), "")) for sid in sids], dtype=bool)
            nsclc_mask.append(m)
            per_cohort_n[c] = {"n": int(len(sids)), "n_nsclc": int(m.sum())}

        gi = np.concatenate(gi_parts, axis=0)   # [N, G]
        ti = np.concatenate(ti_parts, axis=0)   # [N, T]
        ge = np.concatenate(ge_parts, axis=0)   # [N, 463]
        lr = np.concatenate(lr_parts, axis=0)   # [N]
        nsclc = np.concatenate(nsclc_mask, axis=0)  # [N]
        win_rec["per_cohort_n"] = per_cohort_n
        win_rec["n_eval_pooled"] = int(len(lr))
        win_rec["n_eval_nsclc"] = int(nsclc.sum())

        # Risk groups by deployment threshold
        hi = lr >= deploy_thr
        lo = ~hi
        win_rec["n_high_pooled"] = int(hi.sum())
        win_rec["n_low_pooled"] = int(lo.sum())

        group_names = list(info.group_names)
        flat_terms, flat_groups = [], []
        for gidx, gname in enumerate(group_names):
            for tname in info.term_names[gidx]:
                flat_terms.append(str(tname))
                flat_groups.append(gname)

        # ===================== build tables for a given subset mask =========
        def build_tables(sub, suffix):
            sg = gi[sub]; st = ti[sub]; se = ge[sub]
            shi = hi[sub]; slo = lo[sub]
            # GROUP
            g_mean = mean_axis0(sg)
            g_hi = mean_axis0(sg[shi])
            g_lo = mean_axis0(sg[slo])
            g_delta = g_hi - g_lo
            denom = np.nansum(g_mean)
            g_frac = (g_mean / denom if denom and not np.isnan(denom)
                      else np.full_like(g_mean, np.nan))
            df_g = pd.DataFrame({
                "group": group_names,
                f"mean{suffix}": g_mean,
                f"frac{suffix}": g_frac,
                f"high_mean{suffix}": g_hi,
                f"low_mean{suffix}": g_lo,
                f"delta_high_minus_low{suffix}": g_delta,
            })
            # TERM
            t_mean = mean_axis0(st)
            t_hi = mean_axis0(st[shi])
            t_lo = mean_axis0(st[slo])
            df_t = pd.DataFrame({
                "term": flat_terms, "group": flat_groups,
                f"mean{suffix}": t_mean,
                f"high_mean{suffix}": t_hi,
                f"low_mean{suffix}": t_lo,
                f"delta_high_minus_low{suffix}": t_hi - t_lo,
            })
            # GENE
            e_mean = mean_axis0(se)
            e_hi = mean_axis0(se[shi])
            e_lo = mean_axis0(se[slo])
            df_e = pd.DataFrame({
                "gene": genes,
                f"mean{suffix}": e_mean,
                f"high_mean{suffix}": e_hi,
                f"low_mean{suffix}": e_lo,
                f"delta_high_minus_low{suffix}": e_hi - e_lo,
            })
            return df_g, df_t, df_e

        dg_p, dt_p, de_p = build_tables(np.ones(len(lr), dtype=bool), "_pooled")
        dg_n, dt_n, de_n = build_tables(nsclc, "_nsclc")

        df_groups = dg_p.merge(dg_n, on="group", how="outer")
        df_terms = dt_p.merge(dt_n, on=["term", "group"], how="outer")
        df_genes = de_p.merge(de_n, on="gene", how="outer")

        # ---- Save CSVs ----
        df_groups.to_csv(OUT_DIR / f"{short}_group_importance.csv", index=False)

        top_by_mean = dt_p.sort_values("mean_pooled", ascending=False).head(TOP_TERMS).copy()
        top_by_mean["rank_basis"] = "mean_pooled"
        top_by_delta = dt_p.sort_values(
            "delta_high_minus_low_pooled", ascending=False).head(TOP_TERMS).copy()
        top_by_delta["rank_basis"] = "delta_high_minus_low_pooled"
        terms_out = pd.concat([top_by_mean, top_by_delta], ignore_index=True)
        terms_out.to_csv(OUT_DIR / f"{short}_top_terms.csv", index=False)

        gtop_by_mean = de_p.sort_values("mean_pooled", ascending=False).head(TOP_GENES).copy()
        gtop_by_mean["rank_basis"] = "mean_pooled"
        gtop_by_delta = de_p.sort_values(
            "delta_high_minus_low_pooled", ascending=False).head(TOP_GENES).copy()
        gtop_by_delta["rank_basis"] = "delta_high_minus_low_pooled"
        if nsclc.sum() > 0:
            gtop_nsclc_mean = de_n.sort_values("mean_nsclc", ascending=False).head(TOP_GENES).copy()
            gtop_nsclc_mean["rank_basis"] = "mean_nsclc"
        else:
            gtop_nsclc_mean = pd.DataFrame()
        genes_out = pd.concat([gtop_by_mean, gtop_by_delta, gtop_nsclc_mean], ignore_index=True)
        genes_out.to_csv(OUT_DIR / f"{short}_top_genes.csv", index=False)

        # ---- JSON detail ----
        win_rec["group_importance_pooled"] = json.loads(
            df_groups[["group", "mean_pooled", "frac_pooled", "high_mean_pooled",
                       "low_mean_pooled", "delta_high_minus_low_pooled"]]
            .to_json(orient="records"))
        win_rec["group_importance_nsclc"] = json.loads(
            df_groups[["group", "mean_nsclc", "frac_nsclc", "high_mean_nsclc",
                       "low_mean_nsclc", "delta_high_minus_low_nsclc"]]
            .to_json(orient="records"))
        win_rec["top_terms_by_mean_pooled"] = json.loads(top_by_mean.to_json(orient="records"))
        win_rec["top_terms_by_delta_pooled"] = json.loads(top_by_delta.to_json(orient="records"))
        win_rec["top_genes_by_mean_pooled"] = json.loads(gtop_by_mean.to_json(orient="records"))
        win_rec["top_genes_by_delta_pooled"] = json.loads(gtop_by_delta.to_json(orient="records"))
        if not gtop_nsclc_mean.empty:
            win_rec["top_genes_by_mean_nsclc"] = json.loads(gtop_nsclc_mean.to_json(orient="records"))

        # ---- Console summary ----
        print(f"\n  GROUP importance (pooled EVAL, n_high={int(hi.sum())}, n_low={int(lo.sum())}):")
        print(df_groups[["group", "mean_pooled", "frac_pooled", "high_mean_pooled",
                         "low_mean_pooled", "delta_high_minus_low_pooled"]].to_string(index=False))
        dom = df_groups.loc[df_groups["mean_pooled"].idxmax(), "group"]
        print(f"  --> dominant group (pooled mean): {dom}")
        print(f"\n  TOP-12 GENES by mean (pooled):")
        print(gtop_by_mean.head(12)[["gene", "mean_pooled", "high_mean_pooled",
                                     "low_mean_pooled", "delta_high_minus_low_pooled"]].to_string(index=False))
        print(f"\n  TOP-12 GENES by delta high-low (pooled):")
        print(gtop_by_delta.head(12)[["gene", "mean_pooled", "high_mean_pooled",
                                      "low_mean_pooled", "delta_high_minus_low_pooled"]].to_string(index=False))
        print(f"\n  TOP-12 TERMS by mean (pooled):")
        print(top_by_mean.head(12)[["term", "group", "mean_pooled",
                                    "delta_high_minus_low_pooled"]].to_string(index=False))
        if nsclc.sum() > 0:
            print(f"\n  TOP-12 GENES by mean (NSCLC subset, n={int(nsclc.sum())}):")
            print(gtop_nsclc_mean.head(12)[["gene", "mean_nsclc", "high_mean_nsclc",
                                            "delta_high_minus_low_nsclc"]].to_string(index=False))

        win_rec["dominant_group_pooled"] = dom
        win_rec["status"] = "OK"
        summary["winners"][short] = win_rec

    (OUT_DIR / "interp_winners.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
    print(f"\nSaved all outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
