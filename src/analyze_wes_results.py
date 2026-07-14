"""Aggregate WES experiment JSON results into summary CSVs.

Inputs: one or more aggregated experiment JSON files (top-level dict by config).
Outputs:
  output/experiments/{stem}_summary.csv      - per (config) row, ~30 cols
  output/experiments/{stem}_per_cohort.csv   - per (config × cohort) row, ~15 cols
  console: top rows printed for quick scan

Usage:
  python src/analyze_wes_results.py                                # all wes_*.json
  python src/analyze_wes_results.py output/experiments/wes_pancancer_A_plus_full.json
  python src/analyze_wes_results.py --compare a.json b.json        # head-to-head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "output" / "experiments"


def _agg(values, op="mean"):
    """Safe aggregate; returns None if all-NaN or empty."""
    if not values:
        return None
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(arr) == 0:
        return None
    if op == "mean":
        return float(arr.mean())
    if op == "std":
        return float(arr.std())
    if op == "median":
        return float(np.median(arr))
    if op == "max":
        return float(arr.max())
    if op == "min":
        return float(arr.min())
    if op == "geomean":
        if (arr > 0).all():
            return float(np.exp(np.log(arr).mean()))
        return None
    raise ValueError(f"unknown op {op}")


def per_config_row(name: str, summary: dict) -> dict:
    """Aggregate one config across all fold_results -> single summary row."""
    fr = summary.get("fold_results", [])
    if not fr:
        return {"config": name, "n_runs": 0}

    row = {
        "config": name,
        "kg": summary.get("kg", ""),
        "sigfeats_on": summary.get("sigfeats_on", None),
        "n_runs": len(fr),
    }

    # Train + val (basic Cox)
    for prefix in ("train", "val"):
        for key in ("ci", "hr", "p"):
            full_key = f"{prefix}_{key}"
            vals = [r.get(full_key) for r in fr]
            row[f"{prefix}_{key}_mean"] = _agg(vals, "mean")
            row[f"{prefix}_{key}_std"] = _agg(vals, "std")
        # train+val significance rate (p < 0.05)
        ps = [r.get(f"{prefix}_p") for r in fr]
        ps = [p for p in ps if p is not None]
        row[f"{prefix}_sig_rate"] = (
            sum(1 for p in ps if p < 0.05) / len(ps) if ps else None
        )

    # External holdout aggregates already stored at fold level
    for key in ("ext_ci", "n_sig", "rcc_arm_ci", "rcc_arm_n_sig"):
        vals = [r.get(key) for r in fr]
        row[f"{key}_mean"] = _agg(vals, "mean")
        row[f"{key}_std"] = _agg(vals, "std")
    row["n_sig_max"] = _agg([r.get("n_sig") for r in fr], "max")

    # Pooled HR across all (fold × cohort) — primary + RCC arm
    primary_cohorts = ["Whijae", "Hugo", "SnyderUC", "Pleasance"]
    rcc_cohorts = ["CM214", "Braun"]
    primary_hrs, rcc_hrs = [], []
    for r in fr:
        for c, m in r.get("per_cohort", {}).items():
            hr = m.get("hr")
            if hr is None or hr <= 0:
                continue
            if c in rcc_cohorts:
                rcc_hrs.append(hr)
            else:
                primary_hrs.append(hr)
    row["pooled_HR_primary_geo"] = _agg(primary_hrs, "geomean")
    row["pooled_HR_rcc_geo"] = _agg(rcc_hrs, "geomean")

    # Extended metrics; tolerate their absence in older JSON files.
    for key in (
        "boot_ci", "auc_24m", "ibs", "cal_slope", "cal_intercept",
        "arr_24m", "dca_inb",
    ):
        for prefix in ("train", "val"):
            full = f"{prefix}_{key}"
            vals = [r.get(full) for r in fr]
            mean_v = _agg(vals, "mean")
            if mean_v is not None:
                row[f"{prefix}_{key}_mean"] = mean_v

    return row


def per_cohort_rows(name: str, summary: dict) -> list[dict]:
    """One row per (config × cohort) aggregating across folds."""
    fr = summary.get("fold_results", [])
    cohort_data: dict[str, dict[str, list]] = {}
    for r in fr:
        for c, m in r.get("per_cohort", {}).items():
            d = cohort_data.setdefault(c, {})
            for key in ("ci", "hr", "p", "auc_24m", "ibs", "cal_slope", "boot_ci"):
                v = m.get(key)
                if v is not None:
                    d.setdefault(key, []).append(v)

    rows = []
    for c in sorted(cohort_data):
        d = cohort_data[c]
        row = {
            "config": name,
            "cohort": c,
            "n_runs": len(d.get("ci", [])),
        }
        for key in ("ci", "hr", "auc_24m", "ibs", "cal_slope", "boot_ci"):
            if key in d:
                row[f"{key}_mean"] = _agg(d[key], "mean")
                row[f"{key}_std"] = _agg(d[key], "std")
        if "p" in d:
            row["p_mean"] = _agg(d["p"], "mean")
            row["p_median"] = _agg(d["p"], "median")
            row["sig_rate"] = sum(1 for p in d["p"] if p < 0.05) / len(d["p"])
        if "hr" in d:
            row["rev_hr_rate"] = sum(1 for h in d["hr"] if h < 1.0) / len(d["hr"])
        rows.append(row)
    return rows


def analyze_one(json_path: Path, out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = out_dir or json_path.parent
    stem = json_path.stem
    data = json.loads(json_path.read_text(encoding="utf-8"))

    summary_rows = []
    per_cohort_records = []
    for name, summary in data.items():
        summary_rows.append(per_config_row(name, summary))
        per_cohort_records.extend(per_cohort_rows(name, summary))

    summary_df = pd.DataFrame(summary_rows)
    per_cohort_df = pd.DataFrame(per_cohort_records)

    summary_csv = out_dir / f"{stem}_summary.csv"
    per_cohort_csv = out_dir / f"{stem}_per_cohort.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_cohort_df.to_csv(per_cohort_csv, index=False)

    return summary_df, per_cohort_df, summary_csv, per_cohort_csv


def print_summary(summary_df: pd.DataFrame, per_cohort_df: pd.DataFrame, source: str) -> None:
    print(f"\n## {source}")
    print(f"  configs: {len(summary_df)}, cohorts: {per_cohort_df['cohort'].nunique() if len(per_cohort_df) else 0}")

    # Compact summary table
    cols_key = ["config", "n_runs", "train_ci_mean", "val_ci_mean", "ext_ci_mean",
                "n_sig_mean", "n_sig_max", "pooled_HR_primary_geo", "pooled_HR_rcc_geo"]
    cols_show = [c for c in cols_key if c in summary_df.columns]
    print("\n  Summary (key cols):")
    print(summary_df[cols_show].to_string(index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)))

    # Per-cohort compact
    if len(per_cohort_df):
        cols_pc = ["config", "cohort", "n_runs", "ci_mean", "hr_mean", "p_median", "sig_rate", "rev_hr_rate"]
        cols_pc = [c for c in cols_pc if c in per_cohort_df.columns]
        print("\n  Per-cohort (key cols):")
        print(per_cohort_df[cols_pc].to_string(index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)))


def main():
    parser = argparse.ArgumentParser(description="Aggregate WES experiment JSON -> CSV")
    parser.add_argument("inputs", nargs="*", help="JSON file(s); default: all output/experiments/wes_*.json")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "NEW"),
                        help="head-to-head diff two JSON files")
    parser.add_argument("--out-dir", default=None, help="output dir for CSV (default: same as JSON)")
    args = parser.parse_args()

    if args.compare:
        ref_path = Path(args.compare[0])
        new_path = Path(args.compare[1])
        ref_sum, _, _, _ = analyze_one(ref_path)
        new_sum, _, _, _ = analyze_one(new_path)
        # join on config
        merged = ref_sum.merge(new_sum, on="config", how="outer",
                               suffixes=("_ref", "_new"))
        print(f"\n## Head-to-head: {ref_path.name} vs {new_path.name}\n")
        cols = ["config", "ext_ci_mean_ref", "ext_ci_mean_new",
                "n_sig_mean_ref", "n_sig_mean_new",
                "pooled_HR_primary_geo_ref", "pooled_HR_primary_geo_new"]
        cols = [c for c in cols if c in merged.columns]
        print(merged[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)))
        return 0

    inputs = [Path(p) for p in args.inputs] if args.inputs else sorted(EXP.glob("wes_*.json"))
    inputs = [p for p in inputs if p.is_file()]
    if not inputs:
        print("No input JSON found.")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else None
    for json_path in inputs:
        summary_df, per_cohort_df, summary_csv, per_cohort_csv = analyze_one(json_path, out_dir)
        print(f"\n>>> {json_path.name}")
        try:
            print(f"    wrote {summary_csv.resolve().relative_to(ROOT)}")
            print(f"    wrote {per_cohort_csv.resolve().relative_to(ROOT)}")
        except ValueError:
            print(f"    wrote {summary_csv}")
            print(f"    wrote {per_cohort_csv}")
        print_summary(summary_df, per_cohort_df, json_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
