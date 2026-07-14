"""Direction E: Meta-analysis across cohorts (fixed + random effects).

Uses heterogeneity_predictions.csv from strategy 14 (DRKG best risk scores).
For each cohort, fit univariate Cox → log-HR + SE. Then combine across cohorts:
  - Fixed effects (inverse-variance weighted)
  - Random effects (DerSimonian-Laird, accounts for between-cohort variance)

Question: Can we boost borderline cohorts (p=0.05-0.1) into collective significance
when pooled correctly?
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

COHORTS = ["Gandara", "Hugo", "Liu", "Mariathasan", "Miao", "PUSH",
           "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae"]

CANCER_GROUPS = {
    "NSCLC":    ["Gandara", "Ravi"],           # + NSCLC subsets of Miao/Pleasance
    "Melanoma": ["Hugo", "Liu", "Riaz"],
    "Bladder":  ["Mariathasan", "Snyder_UC"],
    "GI":       ["PUSH"],
    "Mixed":    ["Miao", "Pleasance", "Whijae"],
}


def fit_cox_univariate(sub: pd.DataFrame, use_continuous: bool = False) -> dict:
    """Fit Cox on (risk_high or continuous risk). Return log-HR + SE + p."""
    if len(sub) < 10 or sub["event"].sum() < 3:
        return {"status": "skip", "n": len(sub), "events": int(sub["event"].sum())}
    if use_continuous:
        df = sub[["risk", "time", "event"]].rename(columns={"risk": "x"})
    else:
        med = sub["risk"].median()
        df = pd.DataFrame({"x": (sub["risk"] >= med).astype(int).values,
                            "time": sub["time"].values,
                            "event": sub["event"].values})
        if df["x"].sum() < 3 or df["x"].sum() > len(df) - 3:
            return {"status": "skip", "reason": "imbalanced split"}
    try:
        cph = CoxPHFitter(penalizer=1e-4)
        cph.fit(df, duration_col="time", event_col="event", show_progress=False)
        s = cph.summary.loc["x"]
        return {
            "status": "ok", "n": len(sub), "events": int(sub["event"].sum()),
            "log_hr": float(s["coef"]), "se": float(s["se(coef)"]),
            "hr": float(s["exp(coef)"]),
            "ci_lo": float(s["exp(coef) lower 95%"]),
            "ci_hi": float(s["exp(coef) upper 95%"]),
            "p": float(s["p"]),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "n": len(sub)}


def fixed_effects(log_hrs: list[float], ses: list[float]) -> dict:
    """Inverse-variance weighted pooled log-HR."""
    w = np.array([1.0 / (s ** 2) for s in ses])
    lh = np.array(log_hrs)
    pooled = (w * lh).sum() / w.sum()
    pooled_se = 1.0 / np.sqrt(w.sum())
    z = pooled / pooled_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"pooled_log_hr": float(pooled), "pooled_se": float(pooled_se),
            "pooled_hr": float(np.exp(pooled)), "z": float(z), "p": float(p),
            "ci_lo": float(np.exp(pooled - 1.96 * pooled_se)),
            "ci_hi": float(np.exp(pooled + 1.96 * pooled_se))}


def random_effects(log_hrs: list[float], ses: list[float]) -> dict:
    """DerSimonian-Laird random-effects pooled log-HR."""
    lh = np.array(log_hrs)
    w = np.array([1.0 / (s ** 2) for s in ses])
    fe = (w * lh).sum() / w.sum()
    q = (w * (lh - fe) ** 2).sum()
    df = len(lh) - 1
    c = w.sum() - (w * w).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    w_re = np.array([1.0 / (s ** 2 + tau2) for s in ses])
    pooled = (w_re * lh).sum() / w_re.sum()
    pooled_se = 1.0 / np.sqrt(w_re.sum())
    z = pooled / pooled_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    # I² heterogeneity statistic
    i2 = max(0.0, (q - df) / q * 100) if q > 0 else 0.0
    return {"pooled_log_hr": float(pooled), "pooled_se": float(pooled_se),
            "pooled_hr": float(np.exp(pooled)), "z": float(z), "p": float(p),
            "ci_lo": float(np.exp(pooled - 1.96 * pooled_se)),
            "ci_hi": float(np.exp(pooled + 1.96 * pooled_se)),
            "tau2": float(tau2), "Q": float(q), "df": int(df),
            "Q_p": float(1 - stats.chi2.cdf(q, df)) if df > 0 else None,
            "I2_pct": float(i2)}


def main():
    pred = pd.read_csv(EXP_DIR / "heterogeneity_predictions.csv")
    print(f"Loaded {len(pred)} predictions, {pred['cohort'].nunique()} cohorts")
    print()

    # === Per-cohort Cox (binary risk) ===
    print("=" * 92)
    print("Per-cohort Cox (risk_high median split)")
    print("=" * 92)
    print(f"{'Cohort':<14} {'n':<5} {'ev':<5} {'HR':<7} {'95% CI':<20} "
          f"{'log-HR':<9} {'SE':<7} {'p':<9}")
    print("-" * 92)

    cohort_fits = {}
    for c in COHORTS:
        sub = pred[pred["cohort"] == c]
        fit = fit_cox_univariate(sub, use_continuous=False)
        cohort_fits[c] = fit
        if fit.get("status") == "ok":
            ci = f"{fit['ci_lo']:.2f}-{fit['ci_hi']:.2f}"
            star = "*" if fit["p"] < 0.05 else " "
            print(f"{c:<14} {fit['n']:<5} {fit['events']:<5} {fit['hr']:<7.2f} "
                  f"{ci:<20} {fit['log_hr']:+.3f}    {fit['se']:.3f}   "
                  f"{fit['p']:.4f}{star}")
        else:
            print(f"{c:<14} skip/error ({fit.get('reason', '')})")

    valid = {c: f for c, f in cohort_fits.items() if f.get("status") == "ok"}

    # === Overall pooled (all cohorts) ===
    log_hrs = [f["log_hr"] for f in valid.values()]
    ses = [f["se"] for f in valid.values()]
    fe = fixed_effects(log_hrs, ses)
    re = random_effects(log_hrs, ses)

    print()
    print("=" * 92)
    print("  META-ANALYSIS: All cohorts pooled")
    print("=" * 92)
    print(f"  Fixed effects:  HR={fe['pooled_hr']:.3f} "
          f"[{fe['ci_lo']:.3f}-{fe['ci_hi']:.3f}], p={fe['p']:.3e}")
    print(f"  Random effects: HR={re['pooled_hr']:.3f} "
          f"[{re['ci_lo']:.3f}-{re['ci_hi']:.3f}], p={re['p']:.3e}")
    print(f"  Heterogeneity:  Q={re['Q']:.2f}, df={re['df']}, "
          f"Q_p={re['Q_p']:.4f}, I2={re['I2_pct']:.1f}%, tau2={re['tau2']:.4f}")

    # === Per cancer-group pooled ===
    print()
    print("=" * 92)
    print("  META-ANALYSIS: Per cancer type")
    print("=" * 92)
    group_results = {}
    for grp, members in CANCER_GROUPS.items():
        mem_fits = [valid[m] for m in members if m in valid]
        if len(mem_fits) < 1:
            continue
        log_hrs = [f["log_hr"] for f in mem_fits]
        ses = [f["se"] for f in mem_fits]
        if len(mem_fits) == 1:
            # Single cohort, just report as-is
            f0 = mem_fits[0]
            group_results[grp] = {"n_cohorts": 1, "fixed": {"pooled_hr": f0["hr"],
                                 "p": f0["p"], "ci_lo": f0["ci_lo"], "ci_hi": f0["ci_hi"]}}
            print(f"  {grp:<10} cohorts={len(members)} n_fit=1 (single cohort)   "
                  f"HR={f0['hr']:.3f} p={f0['p']:.4f}")
            continue
        fe_g = fixed_effects(log_hrs, ses)
        re_g = random_effects(log_hrs, ses)
        group_results[grp] = {
            "n_cohorts": len(mem_fits), "fixed": fe_g, "random": re_g,
            "members": [m for m in members if m in valid]
        }
        print(f"  {grp:<10} cohorts={len(mem_fits)}  "
              f"FE: HR={fe_g['pooled_hr']:.3f} p={fe_g['p']:.3e}  |  "
              f"RE: HR={re_g['pooled_hr']:.3f} p={re_g['p']:.3e}  |  I2={re_g['I2_pct']:.0f}%")

    # === Compare to single-cohort log-rank approach ===
    print()
    print("=" * 92)
    print("  COMPARISON: Single-cohort p<0.05 vs meta-analysis pooled")
    print("=" * 92)
    n_sig_single = sum(1 for f in valid.values() if f["p"] < 0.05)
    print(f"  Per-cohort single: {n_sig_single}/{len(valid)} cohorts p<0.05")
    print(f"  Pooled FE: HR={fe['pooled_hr']:.3f}, p={fe['p']:.3e}  "
          f"{'[SIGNIFICANT]' if fe['p'] < 0.05 else ''}")
    print(f"  Pooled RE: HR={re['pooled_hr']:.3f}, p={re['p']:.3e}  "
          f"{'[SIGNIFICANT]' if re['p'] < 0.05 else ''}")
    print(f"  Note: meta-pooled p << per-cohort p because n scales up")

    out = {
        "per_cohort": cohort_fits,
        "overall_fixed": fe,
        "overall_random": re,
        "per_cancer_group": group_results,
        "n_sig_single": n_sig_single,
        "n_cohorts": len(valid),
    }
    with open(EXP_DIR / "meta_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved → {EXP_DIR / 'meta_analysis.json'}")


if __name__ == "__main__":
    main()
