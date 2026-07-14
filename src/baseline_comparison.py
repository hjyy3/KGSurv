"""ML baseline comparison: Cox-TMB, Cox-Lasso, RSF vs KG models.

Trains on MSK, evaluates on 11 validation cohorts (RCC excluded).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from losses import c_index, compute_all_metrics

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "output" / "processed"
KG_DIR = ROOT / "output" / "kg_features"
EXP_DIR = ROOT / "output" / "experiments"

# Exclude RCC cohorts
COHORTS = [
    "Gandara", "Hugo", "Liu", "Mariathasan", "Miao",
    "PUSH", "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae",
]


def load_data(split: str) -> dict:
    """Load mutation + clinical for a split."""
    prefix = "train" if split == "train" else f"valid_{split}"
    mut = pd.read_csv(PROC / f"{prefix}_mut.csv", index_col=0)
    clin = pd.read_csv(PROC / f"{prefix}_clin.csv", index_col=0)
    common = mut.index.intersection(clin.index)
    return {
        "X": mut.loc[common].values.astype(np.float64),
        "time": clin.loc[common, "OS_MONTHS"].values.astype(np.float64),
        "event": clin.loc[common, "event"].values.astype(bool),
        "ids": common.tolist(),
    }


def make_y(time, event):
    """Create structured array for sksurv."""
    return np.array(
        [(e, t) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )


# =====================================================================
# Baseline 1: Cox-TMB (single feature = total mutation count)
# =====================================================================
def train_cox_tmb(train):
    from lifelines import CoxPHFitter
    tmb = train["X"].sum(axis=1)
    df = pd.DataFrame({
        "tmb": tmb,
        "time": train["time"],
        "event": train["event"].astype(int),
    })
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(df, duration_col="time", event_col="event")
    return cph


def predict_cox_tmb(cph, data):
    tmb = data["X"].sum(axis=1)
    df = pd.DataFrame({"tmb": tmb})
    return cph.predict_partial_hazard(df).values.ravel()


# =====================================================================
# Baseline 2: Cox-Lasso (L1-penalized Cox on 463 mutation features)
# =====================================================================
def train_cox_lasso(train):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    y = make_y(train["time"], train["event"])
    # Use cross-validated alpha
    model = CoxnetSurvivalAnalysis(
        l1_ratio=1.0, alpha_min_ratio=0.01, max_iter=1000,
        n_alphas=50, fit_baseline_model=True,
    )
    model.fit(train["X"], y)
    return model


def predict_cox_lasso(model, data):
    return model.predict(data["X"])


# =====================================================================
# Baseline 3: Random Survival Forest
# =====================================================================
def train_rsf(train):
    from sksurv.ensemble import RandomSurvivalForest
    y = make_y(train["time"], train["event"])
    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=5, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    )
    rsf.fit(train["X"], y)
    return rsf


def predict_rsf(model, data):
    return model.predict(data["X"])


# =====================================================================
# Load KG model results for comparison
# =====================================================================
def load_kg_results():
    """Load Phase 1 SparsePathNet x {ogb_biokg, primekg} results."""
    kg_results = {}
    for kg in ["ogb_biokg", "primekg"]:
        path = EXP_DIR / f"interp_sparse_path_{kg}" / "summary.json"
        if path.exists():
            with open(path) as f:
                kg_results[f"SparsePathNet_{kg}"] = json.load(f)
    return kg_results


# =====================================================================
# Main
# =====================================================================
def main():
    print("Loading training data...")
    train = load_data("train")
    print(f"  Train: n={len(train['ids'])}, genes={train['X'].shape[1]}, "
          f"events={train['event'].sum()}")

    # Train baselines
    print("\nTraining Cox-TMB...")
    cox_tmb = train_cox_tmb(train)

    print("Training Cox-Lasso...")
    cox_lasso = train_cox_lasso(train)
    n_nonzero = (np.abs(cox_lasso.coef_[-1]) > 1e-6).sum()
    print(f"  Cox-Lasso: {n_nonzero} / {train['X'].shape[1]} non-zero coefficients")

    print("Training RSF...")
    rsf = train_rsf(train)

    baselines = [
        ("Cox-TMB", cox_tmb, predict_cox_tmb),
        ("Cox-Lasso", cox_lasso, predict_cox_lasso),
        ("RSF", rsf, predict_rsf),
    ]

    # Evaluate on validation cohorts
    results = {name: {} for name, _, _ in baselines}

    for cohort in COHORTS:
        try:
            data = load_data(cohort)
        except FileNotFoundError:
            continue

        for name, model, predict_fn in baselines:
            risk = predict_fn(model, data)
            m = compute_all_metrics(risk, data["time"], data["event"])
            results[name][cohort] = m

    # Load KG results
    kg_results = load_kg_results()

    # --- Print comparison table ---
    all_models = [n for n, _, _ in baselines] + list(kg_results.keys())

    print(f"\n{'='*100}")
    print("C-INDEX COMPARISON (RCC excluded)")
    print(f"{'='*100}")
    print(f"{'Cohort':<14}", end="")
    for m in all_models:
        print(f" {m[:16]:>16}", end="")
    print()
    print("-" * (14 + 17 * len(all_models)))

    model_cis = {m: [] for m in all_models}
    model_sigs = {m: 0 for m in all_models}

    for c in COHORTS:
        print(f"{c:<14}", end="")
        for m in all_models:
            if m in results and c in results[m]:
                ci = results[m][c]["c_index"]
                p = results[m][c]["p_value"]
            elif m in kg_results:
                ci = kg_results[m].get(f"{c}_ci", float("nan"))
                p = kg_results[m].get(f"{c}_p", 1.0)
            else:
                ci, p = float("nan"), 1.0

            if not np.isnan(ci):
                model_cis[m].append(ci)
                if p < 0.05:
                    model_sigs[m] += 1
            sig = "**" if p < 0.01 else " *" if p < 0.05 else "  "
            print(f" {ci:>14.4f}{sig}", end="")
        print()

    # Averages
    print("-" * (14 + 17 * len(all_models)))
    print(f"{'Avg CI':<14}", end="")
    for m in all_models:
        avg = np.mean(model_cis[m]) if model_cis[m] else 0
        print(f" {avg:>16.4f}", end="")
    print()
    print(f"{'Sig (p<0.05)':<14}", end="")
    for m in all_models:
        print(f" {model_sigs[m]:>14}/11  ", end="")
    print()

    # --- P-value table ---
    print(f"\n{'='*100}")
    print("P-VALUE COMPARISON")
    print(f"{'='*100}")
    print(f"{'Cohort':<14}", end="")
    for m in all_models:
        print(f" {m[:16]:>16}", end="")
    print()
    print("-" * (14 + 17 * len(all_models)))

    for c in COHORTS:
        print(f"{c:<14}", end="")
        for m in all_models:
            if m in results and c in results[m]:
                p = results[m][c]["p_value"]
            elif m in kg_results:
                p = kg_results[m].get(f"{c}_p", float("nan"))
            else:
                p = float("nan")
            if np.isnan(p):
                print(f" {'N/A':>16}", end="")
            elif p < 0.001:
                print(f" {'<0.001':>14}**", end="")
            else:
                sig = "**" if p < 0.01 else " *" if p < 0.05 else "  "
                print(f" {p:>14.4f}{sig}", end="")
        print()

    # --- HR table ---
    print(f"\n{'='*100}")
    print("HAZARD RATIO COMPARISON")
    print(f"{'='*100}")
    print(f"{'Cohort':<14}", end="")
    for m in all_models:
        print(f" {m[:16]:>16}", end="")
    print()
    print("-" * (14 + 17 * len(all_models)))

    for c in COHORTS:
        print(f"{c:<14}", end="")
        for m in all_models:
            if m in results and c in results[m]:
                hr = results[m][c]["hr"]
            elif m in kg_results:
                hr = kg_results[m].get(f"{c}_hr", float("nan"))
            else:
                hr = float("nan")
            if np.isnan(hr):
                print(f" {'N/A':>16}", end="")
            else:
                print(f" {hr:>16.2f}", end="")
        print()

    # --- Summary ---
    print(f"\n{'='*100}")
    print("SUMMARY RANKING")
    print(f"{'='*100}")
    ranking = sorted(all_models, key=lambda m: (-model_sigs[m], -np.mean(model_cis[m])))
    print(f"{'Rank':>4} {'Model':<25} {'Sig':>5} {'Avg CI':>8}")
    print("-" * 45)
    for i, m in enumerate(ranking, 1):
        avg = np.mean(model_cis[m]) if model_cis[m] else 0
        print(f"{i:>4} {m:<25} {model_sigs[m]:>3}/11 {avg:>8.4f}")

    # Save
    out = {"baselines": {}, "kg_models": {}}
    for name in ["Cox-TMB", "Cox-Lasso", "RSF"]:
        out["baselines"][name] = {}
        for c in COHORTS:
            if c in results[name]:
                out["baselines"][name][c] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in results[name][c].items()
                }
    for kg_name, r in kg_results.items():
        out["kg_models"][kg_name] = {
            c: {"c_index": r.get(f"{c}_ci"), "hr": r.get(f"{c}_hr"),
                "p_value": r.get(f"{c}_p")}
            for c in COHORTS if r.get(f"{c}_ci") is not None
        }

    out_path = EXP_DIR / "baseline_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
