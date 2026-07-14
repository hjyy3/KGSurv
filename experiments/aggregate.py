"""Aggregate per-run JSONs into CSV summary + leaderboard.

Usage:
  python code/aggregate.py --tier 0
  python code/aggregate.py --tier 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0)
    args = ap.parse_args()

    runs_dir = EXP_ROOT / "runs"
    results_dir = EXP_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows, cohort_rows = [], []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        pj = run_dir / "per_cohort.json"
        if not pj.exists():
            continue
        with open(pj, encoding="utf-8") as f:
            r = json.load(f)
        sp = r["spec"]
        rows.append({
            "run_id": r["run_id"],
            "model": sp["model"], "kg": sp["kg"], "combo": sp["combo_tag"],
            "seed": sp["seed"], "fold": sp["fold"],
            "val_ci": r["best_val_ci"],
            "c1_n_sig": r["n_sig"]["c1"],
            "c2_n_sig": r["n_sig"]["c2"],
            "c3_n_sig": r["n_sig"]["c3"],
            "fixed_max_n_sig": max(r["n_sig"]["c2"], r["n_sig"]["c3"]),
            "c2_sigs": ";".join(r["sigs"]["c2"]),
            "c3_sigs": ";".join(r["sigs"]["c3"]),
            "elapsed_s": r["config"].get("elapsed_s", float("nan")),
            "state_hash": r["config"]["state_dict_hash"][:16],
        })
        for cohort, v in r["per_cohort"].items():
            cohort_rows.append({
                "run_id": r["run_id"],
                "model": sp["model"], "kg": sp["kg"],
                "seed": sp["seed"], "fold": sp["fold"],
                "cohort": cohort, "n": v["n"], "events": v["events"],
                "c1_hr": v["c1"]["hr"], "c1_p": v["c1"]["p"], "c1_sig": v["c1"]["sig"],
                "c2_hr": v["c2"]["hr"], "c2_p": v["c2"]["p"], "c2_sig": v["c2"]["sig"],
                "c3_hr": v["c3"]["hr"], "c3_p": v["c3"]["p"], "c3_sig": v["c3"]["sig"],
            })

    if not rows:
        print(f"No runs found under {runs_dir}")
        return

    df = pd.DataFrame(rows).sort_values(
        ["fixed_max_n_sig", "c2_n_sig", "val_ci"], ascending=False,
    )
    df_cohort = pd.DataFrame(cohort_rows)

    summary_csv = results_dir / f"tier{args.tier}_summary.csv"
    cohort_csv = results_dir / f"tier{args.tier}_per_cohort.csv"
    df.to_csv(summary_csv, index=False)
    df_cohort.to_csv(cohort_csv, index=False)

    print(f"\n=== TIER {args.tier} LEADERBOARD (top 20 by fixed_max_n_sig) ===\n")
    show_cols = ["run_id", "val_ci", "c1_n_sig", "c2_n_sig", "c3_n_sig",
                 "fixed_max_n_sig", "c2_sigs"]
    print(df[show_cols].head(20).to_string(index=False))

    # Per (model, kg) aggregation (handy when seeds > 1)
    if df["seed"].nunique() > 1 or df["fold"].nunique() > 1:
        agg = df.groupby(["model", "kg"]).agg(
            n=("run_id", "size"),
            c1_mean=("c1_n_sig", "mean"), c1_max=("c1_n_sig", "max"),
            c2_mean=("c2_n_sig", "mean"), c2_max=("c2_n_sig", "max"),
            c3_mean=("c3_n_sig", "mean"), c3_max=("c3_n_sig", "max"),
            fixed_mean=("fixed_max_n_sig", "mean"),
            fixed_max=("fixed_max_n_sig", "max"),
            val_mean=("val_ci", "mean"),
        ).sort_values("fixed_max", ascending=False)
        agg.to_csv(results_dir / f"tier{args.tier}_by_model_kg.csv")
        print("\n=== BY (model, kg) ===\n")
        print(agg.to_string())

    print(f"\nSaved → {summary_csv}")
    print(f"        {cohort_csv}")


if __name__ == "__main__":
    main()
