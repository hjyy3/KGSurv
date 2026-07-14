"""Strategies 11-12: STEPP analysis + RCS continuous Cox modeling.

STEPP (Subpopulation Treatment Effect Pattern Plot):
  Slide overlapping windows over sorted risk score → compute HR in each window.
  Shows whether biomarker effect is monotone (expected) or flat/reversed.

RCS (Restricted Cubic Splines):
  Model Surv(time, event) ~ rcs(risk, knots) without dichotomization.
  Tests for non-linear association and quantifies continuous effect.

Uses heterogeneity_predictions.csv (DRKG best risk scores for 1703 patients).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =====================================================================
# STEPP
# =====================================================================

def stepp_sliding(pred: pd.DataFrame, window_size: int = 300, step: int = 50,
                  subset: str | None = None) -> pd.DataFrame:
    """Slide overlapping windows of width `window_size` over sorted risk.

    For each window: compute HR (high-risk vs low-risk within window would be
    tautological; instead we compute HR of window vs complement cohort).
    """
    df = pred.copy() if subset is None else pred[pred["cohort"] == subset].copy()
    df = df.sort_values("risk").reset_index(drop=True)
    n = len(df)
    rows = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = df.iloc[start:end]
        # Mean risk in window
        mean_risk = window["risk"].mean()
        median_risk = window["risk"].median()
        # KM median survival in window (restricted)
        evs = window["event"].sum()
        # Empirical observation: 12-month survival rate in window
        surv_12m = ((window["time"] > 12) | ((window["time"] <= 12) & (window["event"] == 0))).mean()
        surv_24m = ((window["time"] > 24) | ((window["time"] <= 24) & (window["event"] == 0))).mean()
        # Fit cohort-adjusted Cox: window indicator, using full pred
        tmp = df.copy()
        tmp["in_window"] = 0
        tmp.loc[window.index, "in_window"] = 1
        tmp["in_window"] = tmp["in_window"].astype(int)
        try:
            cph = CoxPHFitter(penalizer=1e-4)
            cph.fit(tmp[["in_window", "time", "event"]],
                    duration_col="time", event_col="event", show_progress=False)
            s = cph.summary.loc["in_window"]
            hr = float(s["exp(coef)"])
            hr_lo = float(s["exp(coef) lower 95%"])
            hr_hi = float(s["exp(coef) upper 95%"])
            p = float(s["p"])
        except Exception:
            hr, hr_lo, hr_hi, p = np.nan, np.nan, np.nan, np.nan
        rows.append({"mid_rank": start + window_size // 2,
                      "mean_risk": mean_risk, "median_risk": median_risk,
                      "n": window_size, "events": int(evs),
                      "hr_vs_rest": hr, "hr_lo": hr_lo, "hr_hi": hr_hi, "p": p,
                      "surv_12m": surv_12m, "surv_24m": surv_24m})
    return pd.DataFrame(rows)


def plot_stepp(stepp_df: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1, ax2 = axes

    x = stepp_df["mid_rank"]
    # Top: HR curve with 95% CI ribbon
    ax1.semilogy(x, stepp_df["hr_vs_rest"], "b-", lw=2, label="HR (window vs rest)")
    ax1.fill_between(x, stepp_df["hr_lo"], stepp_df["hr_hi"],
                     alpha=0.3, color="blue", label="95% CI")
    ax1.axhline(1.0, color="red", ls="--", alpha=0.7)
    ax1.set_ylabel("Hazard Ratio (log scale)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom: 12m/24m survival
    ax2.plot(x, stepp_df["surv_12m"], "g-", lw=2, label="12m survival")
    ax2.plot(x, stepp_df["surv_24m"], "orange", lw=2, label="24m survival")
    ax2.set_xlabel("Mid-rank of sliding window (sorted by risk)")
    ax2.set_ylabel("Survival fraction")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# =====================================================================
# RCS (Restricted Cubic Splines)
# =====================================================================

def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell RCS basis expansion: n columns = len(knots) - 1 features.

    First column is linear; remaining are cubic spline components.
    """
    k = len(knots)
    out = [x]  # linear
    tk, t_last = knots[:-1], knots[-1]
    denom = (t_last - tk[0]) ** 2
    for j in range(k - 2):
        tj = tk[j]
        col = (np.maximum(0, x - tj) ** 3
               - np.maximum(0, x - tk[-1]) ** 3
               * (t_last - tj) / (t_last - tk[-1])
               + np.maximum(0, x - t_last) ** 3
               * (tk[-1] - tj) / (t_last - tk[-1]))
        out.append(col / denom)
    return np.array(out).T


def fit_rcs_cox(pred: pd.DataFrame, knot_quantiles=(0.1, 0.3, 0.5, 0.7, 0.9),
                 subset: str | None = None):
    df = pred.copy() if subset is None else pred[pred["cohort"] == subset].copy()
    x = df["risk"].values
    knots = np.quantile(x, knot_quantiles)
    basis = rcs_basis(x, knots)
    n_cols = basis.shape[1]
    col_names = [f"rcs_{i}" for i in range(n_cols)]
    data = pd.DataFrame(basis, columns=col_names)
    data["time"] = df["time"].values
    data["event"] = df["event"].values

    try:
        cph = CoxPHFitter(penalizer=1e-4)
        cph.fit(data, duration_col="time", event_col="event", show_progress=False)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Likelihood ratio test vs linear-only model
    try:
        linear_only = CoxPHFitter(penalizer=1e-4)
        linear_only.fit(data[["rcs_0", "time", "event"]],
                         duration_col="time", event_col="event", show_progress=False)
        llr_nonlin = 2 * (cph.log_likelihood_ - linear_only.log_likelihood_)
        df_diff = n_cols - 1
        p_nonlin = float(1 - stats.chi2.cdf(llr_nonlin, df_diff))
    except Exception:
        llr_nonlin, p_nonlin = None, None

    # Predicted log-HR curve on grid
    grid = np.linspace(x.min(), x.max(), 200)
    grid_basis = rcs_basis(grid, knots)
    betas = cph.params_.values
    log_hr_ref = (grid_basis * betas).sum(axis=1)
    # Center on median
    median_idx = np.argmin(np.abs(grid - np.median(x)))
    log_hr_centered = log_hr_ref - log_hr_ref[median_idx]

    return {
        "status": "ok",
        "knots": knots.tolist(),
        "coefs": dict(zip(col_names, betas.tolist())),
        "log_hr_p": float(cph.summary.loc["rcs_0", "p"]),
        "nonlinearity_llr": llr_nonlin,
        "nonlinearity_p": p_nonlin,
        "grid_risk": grid.tolist(),
        "grid_log_hr": log_hr_centered.tolist(),
    }


def plot_rcs(rcs_result: dict, out_path: Path, title: str):
    if rcs_result.get("status") != "ok":
        return
    grid = np.array(rcs_result["grid_risk"])
    log_hr = np.array(rcs_result["grid_log_hr"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, np.exp(log_hr), "b-", lw=2)
    ax.axhline(1.0, color="red", ls="--", alpha=0.7, label="HR=1 (ref=median)")
    for k in rcs_result["knots"]:
        ax.axvline(k, color="gray", ls=":", alpha=0.5)
    nonlin_p = rcs_result.get("nonlinearity_p")
    nonlin_str = f"non-lin p={nonlin_p:.3f}" if nonlin_p is not None else ""
    ax.set_title(f"{title}\nLinear p={rcs_result['log_hr_p']:.2e}, {nonlin_str}")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("HR (centered on median)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# =====================================================================
# Main
# =====================================================================

def main():
    pred = pd.read_csv(EXP_DIR / "heterogeneity_predictions.csv")
    print(f"Loaded {len(pred)} predictions")

    all_results = {}

    # --- Overall STEPP ---
    print("\n=== STEPP overall ===")
    stepp_all = stepp_sliding(pred, window_size=300, step=50)
    out_csv = EXP_DIR / "stepp_overall.csv"
    stepp_all.to_csv(out_csv, index=False)
    plot_stepp(stepp_all, FIG_DIR / "stepp_overall.png",
               "STEPP: HR vs risk window (overall, n=1703)")
    print(f"  {len(stepp_all)} windows; HR range "
          f"{stepp_all['hr_vs_rest'].min():.2f}-{stepp_all['hr_vs_rest'].max():.2f}")
    print(f"  saved → {FIG_DIR / 'stepp_overall.png'}")
    all_results["stepp_overall_n_windows"] = len(stepp_all)
    all_results["stepp_overall_hr_range"] = [
        float(stepp_all["hr_vs_rest"].min()),
        float(stepp_all["hr_vs_rest"].max())]

    # --- Per cancer-type STEPP ---
    CANCER_SUBSETS = {
        "NSCLC": pred[pred["cancer_type"] == "NSCLC"],
        "Melanoma": pred[pred["cancer_type"] == "Melanoma"],
        "Bladder": pred[pred["cancer_type"] == "Bladder"],
    }
    for name, sub in CANCER_SUBSETS.items():
        if len(sub) < 200:
            print(f"\nSkipping {name} STEPP (n={len(sub)} too small)")
            continue
        print(f"\n=== STEPP {name} (n={len(sub)}) ===")
        ws = max(100, int(len(sub) * 0.3))
        step_size = max(20, int(ws * 0.15))
        sub_sorted = sub.sort_values("risk").reset_index(drop=True)
        stepp_c = stepp_sliding(sub_sorted, window_size=ws, step=step_size)
        stepp_c.to_csv(EXP_DIR / f"stepp_{name}.csv", index=False)
        plot_stepp(stepp_c, FIG_DIR / f"stepp_{name}.png",
                   f"STEPP: {name} (n={len(sub)}, window={ws})")
        print(f"  {len(stepp_c)} windows; HR range "
              f"{stepp_c['hr_vs_rest'].min():.2f}-{stepp_c['hr_vs_rest'].max():.2f}")

    # --- RCS: overall + per cancer-type ---
    print("\n=== RCS Cox (5-knot spline) ===")
    rcs_overall = fit_rcs_cox(pred)
    plot_rcs(rcs_overall, FIG_DIR / "rcs_overall.png", "RCS: Overall (n=1703)")
    if rcs_overall.get("status") == "ok":
        nl_p = rcs_overall.get("nonlinearity_p")
        nl_str = f"{nl_p:.3f}" if nl_p is not None else "n/a"
        print(f"  Overall: linear p={rcs_overall['log_hr_p']:.2e}, "
              f"non-linearity p={nl_str}")
    all_results["rcs_overall"] = rcs_overall

    for name, sub in CANCER_SUBSETS.items():
        if len(sub) < 100:
            continue
        print(f"\nRCS {name}:")
        rcs_c = fit_rcs_cox(sub)
        plot_rcs(rcs_c, FIG_DIR / f"rcs_{name}.png",
                 f"RCS: {name} (n={len(sub)})")
        if rcs_c.get("status") == "ok":
            nl_p = rcs_c.get("nonlinearity_p")
            nl_str = f"{nl_p:.3f}" if nl_p is not None else "n/a"
            print(f"  {name}: linear p={rcs_c['log_hr_p']:.2e}, "
                  f"non-linearity p={nl_str}")
        all_results[f"rcs_{name}"] = rcs_c

    out = EXP_DIR / "stepp_rcs.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {out}")
    print(f"Figures → {FIG_DIR}")


if __name__ == "__main__":
    main()
