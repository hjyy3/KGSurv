"""NSCLC depth-of-evidence: node-ablation sweep on Ravi cohort.

Tests progressive addition of KG node types on top of FMB to isolate each
node-type's contribution to ORR prediction. 4 configs:
  A1  FMB-only             (no multi-node)
  A2  FMB + PPI            (gene-gene adjacency)
  A3  FMB + PPI + Disease  (Gene-Disease FMB-style burden)
  A4  FMB + PPI + Disease + Drug  (== N1 from primary sweep)

5-fold × 5-seed × 4 configs = 100 runs. Reuses ORR pipeline; sigfeats=True
(matches N1 baseline). Output:
  output/experiments/wes_nsclc_ablation.json + _summary.csv
  output/experiments/wes_nsclc_ablation_folds/<config>/seed*_fold*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_wes_cancer_specific_orr import (
    SEEDS,
    load_orr_labels,
    run_config_orr,
)
from exp_wes_pancancer import load_sigfeats, normalise_sigfeats

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "experiments"

N_FOLDS = 5

# Each config: (config_name, kg, sigfeats, node_types_override)
ABLATION_CONFIGS = [
    ("A1_nsclc_drkg_fmb_only",             "drkg", True,  []),
    ("A2_nsclc_drkg_fmb_ppi",              "drkg", True,  ["ppi"]),
    ("A3_nsclc_drkg_fmb_ppi_disease",      "drkg", True,  ["ppi", "disease"]),
    ("A4_nsclc_drkg_fmb_ppi_disease_drug", "drkg", True,  ["ppi", "disease", "drug"]),
]


def main():
    parser = argparse.ArgumentParser(description="NSCLC node-ablation sweep")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="config names to run; default = all 4")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=str(EXP_DIR / "wes_nsclc_ablation.json"))
    args = parser.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.folds = 2

    out_path = Path(args.out)
    out_dir = EXP_DIR / "wes_nsclc_ablation_folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.configs) if args.configs else None

    print("Loading ORR labels ...")
    orr_map = load_orr_labels()
    print(f"  ORR labels loaded: {len(orr_map)} samples global")

    print("Loading sigfeats ...")
    sigfeats_norm = normalise_sigfeats(load_sigfeats())

    all_results: dict[str, dict] = {}
    for cname, kg, sigfeats_on, node_types in ABLATION_CONFIGS:
        if requested and cname not in requested:
            continue
        summary = run_config_orr(
            cancer="NSCLC",
            config_name=cname,
            kg=kg,
            sigfeats_on=sigfeats_on,
            sigfeats_norm=sigfeats_norm if sigfeats_on else None,
            orr_map=orr_map,
            seeds=args.seeds,
            n_folds=args.folds,
            out_dir=out_dir,
            node_types=node_types,
        )
        all_results[cname] = summary
        out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"  -> saved partial to {out_path}")

    print()
    print("Done. Ablation summary:")
    for c, s in all_results.items():
        print(f"  {c}: val={s['val_auroc_mean']:.4f}+/-{s['val_auroc_std']:.4f} "
              f"ext={s['ext_auroc_primary_mean']:.4f}+/-{s['ext_auroc_primary_std']:.4f} "
              f"n_sig={s['n_sig_primary_mean']:.2f} (max={s['n_sig_primary_max']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
