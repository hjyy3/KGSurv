"""
Retrain key configs after gene-alias fix to measure performance delta.

Target configs:
  1. DRKG PathAttnSurv FMB+PPI+Disease+Drug  (was 7/11 sig)
  2. OpenBioLink PathAttnSurv ALL 6 types     (was 5/11 sig, best ensemble)

Run:
    python src/retrain_alias_fixed.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_node_extended import (
    load_base_splits, augment_splits, build_combo_info,
    train_and_eval, MODEL,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features, EVAL_COHORTS
from kg_features import load_candidate_genes

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"

# Before/after comparison baselines (from PROGRESS.md)
BASELINES = {
    "path_attn_drkg_fmb+ppi+disease+drug": {"n_sig": 7, "ext_ci": 0.5794},
    "path_attn_openbiolink_fmb+ppi+disease+drug+phenotype+anatomy+regulatory": {
        "n_sig": 5, "ext_ci": None
    },
}

RETRAIN_COMBOS = [
    # (kg, combo_name, node_types)
    ("drkg", "ppi+disease+drug",
     ["ppi", "disease", "drug"]),
    ("openbiolink", "ppi+disease+drug+pheno+anat+reg",
     ["ppi", "disease", "drug", "phenotype", "anatomy", "regulatory"]),
]

N_SEEDS = 1  # Quick validation — use 3 seeds for more reliable estimate


def main():
    genes = load_candidate_genes()
    n_genes = len(genes)
    results = []

    kg_cache: dict = {}
    for kg, combo_name, node_types in RETRAIN_COMBOS:
        if kg not in kg_cache:
            print(f"\nLoading {kg} features (alias-fixed)...")
            splits, raw = load_base_splits(kg)
            nf = {}
            for nt, (mode, edict) in ALL_NODE_TYPES.items():
                res = compute_node_features(kg, nt, mode, edict, genes, raw)
                if res:
                    feats, tnames, n_info = res
                    nf[nt] = (feats, tnames, mode)
                    print(f"  {nt}: {n_info} {'edges' if mode=='adj' else 'terms'}")
            kg_cache[kg] = (splits, raw, nf)

        splits, raw, node_feats = kg_cache[kg]
        missing = [nt for nt in node_types if nt not in node_feats]
        if missing:
            print(f"  [skip] {kg}/{combo_name}: missing {missing}")
            continue

        extra_f = [node_feats[nt][0] for nt in node_types]
        extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in node_types]
        aug = augment_splits(splits, extra_f)
        info = build_combo_info(kg, extra_i, n_genes)

        tag = f"{MODEL}_{kg}_fmb+{combo_name}"
        base = BASELINES.get(tag, {})
        print(f"\n  {tag}")
        print(f"  Baseline: n_sig={base.get('n_sig','?')}, ext_ci={base.get('ext_ci','?')}")

        seed_results = []
        for seed in range(42, 42 + N_SEEDS):
            r = train_and_eval(
                info, aug["train"],
                {c: aug[c] for c in EVAL_COHORTS if c in aug},
                seed=seed,
            )
            seed_results.append(r)
            print(f"    seed={seed}: val={r['val_ci']:.4f} "
                  f"ext={r['ext_ci']:.4f} sig={r['n_sig']}/11 {r['sigs']}")

        # Average
        avg_sig = np.mean([r["n_sig"] for r in seed_results])
        avg_ci  = np.mean([r["ext_ci"] for r in seed_results])
        print(f"  >>> alias-fixed: avg_sig={avg_sig:.1f}/11, avg_ext_ci={avg_ci:.4f}")
        base_sig = base.get("n_sig", "?")
        delta = f"+{avg_sig - base_sig:.1f}" if isinstance(base_sig, (int, float)) else "?"
        print(f"  >>> delta vs baseline: sig {delta}")

        r_best = max(seed_results, key=lambda x: x["n_sig"])
        r_best.update({"tag": tag, "kg": kg, "combo": combo_name,
                       "node_types": node_types, "alias_fixed": True,
                       "baseline_n_sig": base.get("n_sig"),
                       "baseline_ext_ci": base.get("ext_ci")})
        results.append(r_best)

    # Save
    out = EXP_DIR / "alias_fixed_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
