"""Retrain best config per KG with Cox-univariate pre-filtered FMB features.

Pipeline:
  1. Read selected feature names from output/experiments/fs_selected/{kg}_{strat}_{val}.txt
  2. Subset train/valid FMB CSVs to keep only those columns
  3. Rebuild KGGroupInfo: groups with zero retained terms are dropped; remaining
     groups get fresh gene_term_mask and fmb_slices aligned with filtered FMB.
  4. Add multi-node extras (PPI/Disease/Drug/etc.) matching the KG's best combo.
  5. Train PathAttnSurv + evaluate across 11 cohorts.

Usage:
    python src/retrain_fs.py                # run all configs
    python src/retrain_fs.py --kgs drkg     # single KG
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import (
    KGGroupInfo, SUBKG_DIR, _load_gene_list, _norm_ws, build_kg_group_info,
)
from kg_features import load_candidate_genes
from multi_node_extended import (
    augment_splits, build_combo_info, train_and_eval, EVAL_COHORTS, MODEL,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features

ROOT = Path(__file__).resolve().parent.parent
KG_DIR = ROOT / "output" / "kg_features"
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"
FS_SEL_DIR = EXP_DIR / "fs_selected"

# Best multi-node combo per KG (from previous experiments)
BEST_COMBOS = {
    "drkg":        ["ppi", "disease", "drug"],
    "openbiolink": ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"],
    "monarch":     ["ppi", "disease", "phenotype", "anatomy"],
    "hetionet":    ["ppi", "disease", "drug", "anatomy", "regulatory"],
    "primekg":     ["ppi"],
    "ibkh":        ["ppi", "disease", "drug"],
    "ogb_biokg":   ["ppi", "drug"],
}

# Baseline n_sig (unfiltered) — from PROGRESS.md / full_extended_metrics.json
BASELINES = {
    "drkg":        {"n_sig": 7, "ext_ci": 0.598, "auc24": 0.696, "ibs": 0.264},
    "openbiolink": {"n_sig": 6, "ext_ci": 0.594, "auc24": 0.689, "ibs": 0.268},
    "monarch":     {"n_sig": 6, "ext_ci": 0.596, "auc24": 0.703, "ibs": 0.264},
    "hetionet":    {"n_sig": 5, "ext_ci": 0.592, "auc24": 0.681, "ibs": 0.269},
    "primekg":     {"n_sig": 6, "ext_ci": 0.554, "auc24": 0.578, "ibs": 0.294},
    "ibkh":        {"n_sig": None, "ext_ci": None, "auc24": None, "ibs": None},
    "ogb_biokg":   {"n_sig": None, "ext_ci": None, "auc24": None, "ibs": None},
}

STRATEGIES = [
    ("top_pct", 0.25),
    ("top_pct", 0.50),
    ("top_pct", 0.75),
    ("pval_thresh", 0.05),
    ("pval_thresh", 0.10),
    ("pval_thresh", 0.20),
]

N_SEEDS = 1


def _strat_tag(strategy: str, value: float) -> str:
    return f"{strategy}_{value}".replace(".", "p")


def _load_selected(kg: str, strategy: str, value: float) -> set[str]:
    tag = _strat_tag(strategy, value)
    path = FS_SEL_DIR / f"{kg}_{tag}.txt"
    return set(path.read_text(encoding="utf-8").splitlines())


def _filter_fmb_csv(csv_path: Path, selected_cols: set[str]) -> pd.DataFrame:
    """Load FMB csv, keep only columns in `selected_cols` (preserves original order)."""
    df = pd.read_csv(csv_path, index_col=0)
    keep = [c for c in df.columns if c in selected_cols]
    return df.loc[:, keep]


def load_filtered_splits(kg: str, selected_cols: set[str]):
    """Load train + valid splits with FMB subset to selected_cols."""
    splits, raw = {}, {}
    for sn in ["train"] + EVAL_COHORTS:
        prefix = "train" if sn == "train" else f"valid_{sn}"
        try:
            mut = pd.read_csv(PROC / f"{prefix}_mut.csv", index_col=0)
            mask = pd.read_csv(PROC / f"{prefix}_mask.csv", index_col=0)
            clin = pd.read_csv(PROC / f"{prefix}_clin.csv", index_col=0)
            fmb = _filter_fmb_csv(KG_DIR / kg / f"{prefix}_fmb.csv", selected_cols)
        except FileNotFoundError:
            continue
        common = mut.index.intersection(clin.index).intersection(fmb.index)
        splits[sn] = {
            "mut": torch.tensor(mut.loc[common].values, dtype=torch.float32),
            "mask": torch.tensor(mask.loc[common].values, dtype=torch.float32),
            "time": torch.tensor(clin.loc[common, "OS_MONTHS"].values, dtype=torch.float32),
            "event": torch.tensor(clin.loc[common, "event"].values, dtype=torch.float32),
            "fmb": torch.tensor(fmb.loc[common].values, dtype=torch.float32),
            "sample_ids": common.tolist(),
        }
        raw[sn] = (mut.loc[common].values.astype(np.float32),
                   mask.loc[common].values.astype(np.float32))
    return splits, raw


def build_filtered_kg_info(kg: str, selected_cols: set[str]) -> KGGroupInfo:
    """Rebuild KGGroupInfo with groups/masks filtered to selected_cols.

    Keeps original edge_type grouping, drops empty groups, recomputes slices.
    """
    base = build_kg_group_info(kg)
    genes = _load_gene_list()
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    # Per-group filter using stored term_names
    subkg = pd.read_csv(SUBKG_DIR / f"subkg_{kg}.csv", low_memory=False)
    subkg["y_name_norm"] = subkg["y_name"].apply(_norm_ws)

    new_groups, new_terms, new_masks, new_slices = [], [], [], []
    offset = 0
    for gi, et in enumerate(base.group_names):
        terms = base.term_names[gi]
        # Recover full column name as it appears in FMB: "{et}::{term}"
        keep_idx = [i for i, t in enumerate(terms) if f"{et}::{t}" in selected_cols]
        if not keep_idx:
            continue
        kept_terms = [terms[i] for i in keep_idx]
        n_t = len(kept_terms)
        new_groups.append(et)
        new_terms.append(kept_terms)
        # Rebuild mask for this group from subgraph (only kept terms)
        term_to_idx = {t: i for i, t in enumerate(kept_terms)}
        edges = subkg[subkg["relation"] == et]
        m = torch.zeros(n_genes, n_t, dtype=torch.float32)
        if len(edges) > 0:
            g_idx = edges["x_name"].map(gene_to_idx)
            t_idx = edges["y_name_norm"].map(term_to_idx)
            valid = g_idx.notna() & t_idx.notna()
            if valid.any():
                m[g_idx[valid].astype(int).values, t_idx[valid].astype(int).values] = 1.0
        new_masks.append(m)
        new_slices.append((offset, offset + n_t))
        offset += n_t

    return KGGroupInfo(
        kg_name=kg, group_names=new_groups, term_names=new_terms,
        gene_term_mask=new_masks, fmb_slices=new_slices,
        n_genes=n_genes, n_total_terms=offset,
    )


def run_one_config(kg: str, strategy: str, value: float, node_types: list[str],
                   genes: list[str]) -> dict:
    n_genes = len(genes)
    selected = _load_selected(kg, strategy, value)
    splits, raw = load_filtered_splits(kg, selected)
    base_info = build_filtered_kg_info(kg, selected)

    # Compute node features (PPI, disease, etc.) — unaffected by FMB filtering
    node_feats = {}
    for nt in node_types:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features(kg, nt, mode, edict, genes, raw)
        if res:
            feats, tnames, n_info = res
            node_feats[nt] = (feats, tnames, mode)
    missing = [nt for nt in node_types if nt not in node_feats]
    if missing:
        return {"status": "skip", "missing": missing}

    extra_f = [node_feats[nt][0] for nt in node_types]
    extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in node_types]
    aug = augment_splits(splits, extra_f)

    # Combine FMB info + node features into final KGGroupInfo
    info = _combine_info(base_info, extra_i, n_genes)

    seed_results = []
    for seed in range(42, 42 + N_SEEDS):
        r = train_and_eval(
            info, aug["train"],
            {c: aug[c] for c in EVAL_COHORTS if c in aug},
            seed=seed,
        )
        seed_results.append(r)

    avg_sig = float(np.mean([r["n_sig"] for r in seed_results]))
    avg_ci = float(np.mean([r["ext_ci"] for r in seed_results]))
    best = max(seed_results, key=lambda x: x["n_sig"])
    return {
        "status": "ok",
        "n_selected": len(selected),
        "n_fmb_after": base_info.n_total_terms,
        "n_groups_after": len(base_info.group_names),
        "avg_n_sig": avg_sig,
        "avg_ext_ci": avg_ci,
        "best_n_sig": best["n_sig"],
        "best_ext_ci": best["ext_ci"],
        "best_sigs": best["sigs"],
        "best_per_cohort": {c: {"ci": best.get(f"{c}_ci"), "p": best.get(f"{c}_p")}
                            for c in EVAL_COHORTS if f"{c}_ci" in best},
    }


def _combine_info(base_info: KGGroupInfo, extra_term_lists, n_genes: int) -> KGGroupInfo:
    """Append node-type extras (PPI/disease/...) to filtered FMB info."""
    groups = list(base_info.group_names)
    terms = list(base_info.term_names)
    masks = list(base_info.gene_term_mask)
    slices = list(base_info.fmb_slices)
    offset = slices[-1][1] if slices else 0
    total = base_info.n_total_terms
    for tnames, mode, label in extra_term_lists:
        groups.append(label)
        terms.append(tnames)
        n_t = len(tnames)
        m = torch.eye(n_genes, dtype=torch.float32) if mode == "adj" else \
            torch.ones(n_genes, n_t, dtype=torch.float32) / max(n_genes, 1)
        masks.append(m)
        slices.append((offset, offset + n_t))
        offset += n_t
        total += n_t
    return KGGroupInfo(kg_name=base_info.kg_name, group_names=groups,
                       term_names=terms, gene_term_mask=masks,
                       fmb_slices=slices, n_genes=n_genes, n_total_terms=total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgs", nargs="+", default=None,
                        help="subset of KGs to run (default all)")
    parser.add_argument("--out", default="feature_selection.json")
    args = parser.parse_args()

    genes = load_candidate_genes()
    kgs = args.kgs if args.kgs else list(BEST_COMBOS.keys())

    all_results = []
    for kg in kgs:
        node_types = BEST_COMBOS[kg]
        base = BASELINES.get(kg, {})
        print(f"\n{'=' * 60}")
        print(f"  {kg}   nodes={node_types}   baseline n_sig={base.get('n_sig')}")
        print(f"{'=' * 60}")
        for strategy, value in STRATEGIES:
            print(f"\n[{kg}  {strategy}={value}]")
            try:
                res = run_one_config(kg, strategy, value, node_types, genes)
            except Exception as e:
                print(f"  ERROR: {e}")
                res = {"status": "error", "error": str(e)}
            res.update({"kg": kg, "strategy": strategy, "value": value,
                        "node_types": node_types, "baseline_n_sig": base.get("n_sig")})
            if res["status"] == "ok":
                delta = res["best_n_sig"] - (base.get("n_sig") or 0)
                sign = "+" if delta >= 0 else ""
                print(f"  n_fmb_after={res['n_fmb_after']}  "
                      f"best_sig={res['best_n_sig']}/11 ({sign}{delta})  "
                      f"ci={res['best_ext_ci']:.4f}  sigs={res['best_sigs']}")
            all_results.append(res)

        # Save incrementally
        out = EXP_DIR / args.out
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
