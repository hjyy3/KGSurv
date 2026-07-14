"""Cox univariate feature pre-screening for FMB features.

For each KG's train_fmb.csv, fit single-variable Cox regression for every
feature column and rank by p-value. Two selection strategies:

  1. top_pct    — keep top percent by p-value (e.g., 25%, 50%, 75%)
  2. pval_thresh — keep features with p < threshold (e.g., 0.05, 0.1, 0.2)

Outputs:
  output/experiments/fs_pvalues/{kg}.csv        — per-feature p-value + HR + coef
  output/experiments/fs_selected/{kg}_{strat}_{val}.txt   — selected col names
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning

ROOT = Path(__file__).resolve().parent.parent
KG_DIR = ROOT / "output" / "kg_features"
PROC = ROOT / "output" / "processed"
EXP_DIR = ROOT / "output" / "experiments"

ALL_KGS = ["primekg", "hetionet", "drkg", "ibkh", "monarch", "ogb_biokg", "openbiolink"]

PVAL_DIR = EXP_DIR / "fs_pvalues"
SEL_DIR = EXP_DIR / "fs_selected"


def compute_cox_univariate(fmb: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    """Single-variable Cox per column. Returns DataFrame with p, hr, coef, n_nonzero."""
    common = fmb.index.intersection(clin.index)
    fmb = fmb.loc[common]
    time = clin.loc[common, "OS_MONTHS"].values
    event = clin.loc[common, "event"].values

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    results = []
    n_cols = fmb.shape[1]
    for i, col in enumerate(fmb.columns):
        x = fmb[col].values
        nonzero = int((x != 0).sum())
        # Skip zero-variance or near-constant features (cannot fit Cox)
        if nonzero < 10 or x.std() < 1e-10:
            results.append({"feature": col, "p": np.nan, "hr": np.nan,
                            "coef": np.nan, "n_nonzero": nonzero})
            continue
        try:
            df = pd.DataFrame({"feat": x, "time": time, "event": event})
            cph = CoxPHFitter(penalizer=1e-4)
            cph.fit(df, duration_col="time", event_col="event",
                    show_progress=False)
            s = cph.summary.loc["feat"]
            results.append({
                "feature": col,
                "p": float(s["p"]),
                "hr": float(s["exp(coef)"]),
                "coef": float(s["coef"]),
                "n_nonzero": nonzero,
            })
        except (ConvergenceError, ValueError, np.linalg.LinAlgError):
            results.append({"feature": col, "p": np.nan, "hr": np.nan,
                            "coef": np.nan, "n_nonzero": nonzero})
        if (i + 1) % 500 == 0:
            print(f"    progress {i + 1}/{n_cols}", flush=True)
    return pd.DataFrame(results)


def compute_all_kgs():
    PVAL_DIR.mkdir(parents=True, exist_ok=True)
    clin = pd.read_csv(PROC / "train_clin.csv", index_col=0)
    summary = {}
    for kg in ALL_KGS:
        fmb_path = KG_DIR / kg / "train_fmb.csv"
        if not fmb_path.exists():
            print(f"[skip] {kg}: no train_fmb.csv")
            continue
        print(f"\n=== {kg} ===")
        fmb = pd.read_csv(fmb_path, index_col=0)
        print(f"  shape={fmb.shape}, starting Cox univariate…")
        df = compute_cox_univariate(fmb, clin)
        out_path = PVAL_DIR / f"{kg}.csv"
        df.to_csv(out_path, index=False)
        n_valid = int(df["p"].notna().sum())
        n_sig05 = int((df["p"] < 0.05).sum())
        n_sig10 = int((df["p"] < 0.10).sum())
        print(f"  saved → {out_path}  ({n_valid} valid, {n_sig05} p<0.05, {n_sig10} p<0.10)")
        summary[kg] = {
            "n_total": len(df),
            "n_valid": n_valid,
            "n_sig05": n_sig05,
            "n_sig10": n_sig10,
        }
    with open(EXP_DIR / "fs_univariate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {EXP_DIR / 'fs_univariate_summary.json'}")
    return summary


def select_by_strategy(pvals: pd.DataFrame, strategy: str, value: float) -> list[str]:
    """Return list of selected feature names."""
    valid = pvals.dropna(subset=["p"]).sort_values("p")
    if strategy == "top_pct":
        k = max(1, int(len(valid) * value))
        return valid.head(k)["feature"].tolist()
    if strategy == "pval_thresh":
        return valid[valid["p"] < value]["feature"].tolist()
    raise ValueError(f"Unknown strategy {strategy}")


def write_selected(strategies: list[tuple[str, float]]):
    """Apply each strategy to each KG's p-values; save selected feature lists."""
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for kg in ALL_KGS:
        pv_path = PVAL_DIR / f"{kg}.csv"
        if not pv_path.exists():
            continue
        pvals = pd.read_csv(pv_path)
        for strat, val in strategies:
            sel = select_by_strategy(pvals, strat, val)
            tag = f"{strat}_{val}".replace(".", "p")
            out = SEL_DIR / f"{kg}_{tag}.txt"
            out.write_text("\n".join(sel), encoding="utf-8")
            summary.append({"kg": kg, "strategy": strat, "value": val,
                            "n_selected": len(sel), "n_total": len(pvals)})
            print(f"  {kg}  {strat}={val}  → {len(sel)}/{len(pvals)}")
    with open(EXP_DIR / "fs_selected_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true",
                        help="compute Cox univariate p-values (slow)")
    parser.add_argument("--select", action="store_true",
                        help="apply selection strategies using existing p-values")
    args = parser.parse_args()

    if args.compute:
        compute_all_kgs()

    if args.select or not args.compute:
        strategies = [
            ("top_pct", 0.25),
            ("top_pct", 0.50),
            ("top_pct", 0.75),
            ("pval_thresh", 0.05),
            ("pval_thresh", 0.10),
            ("pval_thresh", 0.20),
        ]
        write_selected(strategies)


if __name__ == "__main__":
    main()
