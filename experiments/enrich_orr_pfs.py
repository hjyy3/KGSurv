"""Enrich OS-based risk grouping with ORR / PFS / RECIST endpoints.

For each of the 3 deployment-candidate "8/11 winners", each patient gets a
continuous OS-risk score and a FIXED train-derived threshold splitting them
into HIGH-risk (risk >= tau) vs LOW-risk. The models were trained on OS only.
This script tests whether the SAME high/low grouping ALSO separates three
INDEPENDENT clinical endpoints that were never used in training:

  * ORR    -- objective response (1 responder / 0 non-responder)
  * PFS    -- progression-free survival (months + event)
  * RECIST -- best treatment response (CR/PR/SD/PD ...)

No circularity: ORR/PFS/RECIST are external endpoints, so this is a
validation of the grouping's clinical utility, not a refit.

Join (POSITIONAL, verified):
  risk_<cohort>[i]  <->  valid_<cohort>_clin.index[i]  (== Sample.ID)
  Endpoints come from source/input_data/valid/clin_<cohort>.csv keyed by
  the column "Sample.ID". The length equality len(risk)==len(processed) is
  ASSERTED per cohort; any mismatch raises (loud, never silent).

Outputs (results/enrich_orr_pfs/):
  <winner_short>_per_cohort.csv  -- per (winner x cohort) ORR/PFS/RECIST stats
  enrich_summary.csv             -- one row per winner: pooled stats + #sig
  enrich_by_cancer.csv           -- winner x per-sample Cancer_type rows
  enrich_full.json               -- all detail, numpy-safe

Usage:
  python -X utf8 experiments/enrich_orr_pfs.py

Pure post-processing. Idempotent (overwrites outputs each run).
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# --- Paths ------------------------------------------------------------------
EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[1]
RUNS_DIR = EXP_ROOT / "runs"
RESULTS_DIR = EXP_ROOT / "results" / "enrich_orr_pfs"
PROCESSED_DIR = REPO_ROOT / "output" / "processed"
CLIN_DIR = REPO_ROOT / "source" / "input_data" / "valid"

EVAL_COHORTS: tuple[str, ...] = (
    "Gandara", "Hugo", "Liu", "Mariathasan", "Miao",
    "PUSH", "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae",
)

# winner_short -> (run_dir, cutoff_id, expected_threshold)
WINNERS: dict[str, tuple[str, str, float]] = {
    "ibkh": (
        "path_attn_ibkh_ppi+disease+drug_seed63_fold2", "c8", 0.51233,
    ),
    "primekg_mg2": (
        "path_attn_primekg_mg2_ppi_seed74_fold2", "c10", -0.70856,
    ),
    "hetionet_mg5": (
        "path_attn_hetionet_mg5_ppi+disease+drug+anatomy+regulatory"
        "_seed69_fold3", "c10", -0.83917,
    ),
}

# Minimum labelled-n gates (per spec)
MIN_ORR_PER_GROUP = 5
MIN_PFS_PER_GROUP = 10
MIN_CANCER_N = 20  # per cancer-type aggregate gate

# RECIST label -> canonical class. Unmapped labels are dropped (recorded).
RECIST_MAP: dict[str, str] = {
    "CR": "CR", "PR": "PR", "SD": "SD", "PD": "PD",
    # already-binned durable-clinical-benefit labels
    "DCB": "DCB", "NDB": "NDB",
    "Durable clinical benefit": "DCB",
    "No durable clinical benefit": "NDB",
}
# Classes counted as DCB / non-DCB. SD counts as DCB (clinical-benefit defn).
DCB_CLASSES = {"CR", "PR", "SD", "DCB"}
NONDCB_CLASSES = {"PD", "NDB"}


def _norm_recist(raw: Any) -> str | None:
    """Map a raw RECIST string to a canonical class or None (dropped)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    return RECIST_MAP.get(str(raw).strip(), None)


# --- numpy-safe JSON --------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types / NaN to JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def confirm_cutoff(run_dir: str, cutoff_id: str, expected: float) -> float:
    """Confirm the max-n_sig cutoff & threshold against rescored.json.

    Raises RuntimeError on any disagreement (never guesses).
    """
    rj = json.loads((RUNS_DIR / run_dir / "rescored.json").read_text())
    n_sig = rj["n_sig"]
    # restrict to fixed cutoffs c2..c10 (c1 = per-cohort median, not fixed)
    fixed = {k: v for k, v in n_sig.items() if k != "c1"}
    best = max(fixed, key=lambda k: fixed[k])
    thr = float(rj["cutoff_thresholds"][cutoff_id])
    if best != cutoff_id:
        raise RuntimeError(
            f"{run_dir}: max-n_sig fixed cutoff is {best} "
            f"(n_sig={fixed[best]}), expected {cutoff_id} "
            f"(n_sig={fixed.get(cutoff_id)}). STOPPING."
        )
    if abs(thr - expected) > 1e-3:
        raise RuntimeError(
            f"{run_dir}: cutoff_thresholds[{cutoff_id}]={thr:.6f} "
            f"disagrees with expected {expected:.6f}. STOPPING."
        )
    return thr


# --- data loading + positional join ----------------------------------------
def _endpoints_from_clin(m: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Extract ORR/RECIST/PFS/Cancer_type columns from joined clin frame."""
    df = pd.DataFrame(index=index)
    df["ORR"] = pd.to_numeric(m.get("ORR"), errors="coerce")
    rec_raw = m.get("Best.treatment.response")
    df["recist"] = rec_raw.map(_norm_recist) if rec_raw is not None else None
    df["recist_raw"] = rec_raw if rec_raw is not None else None
    df["pfs_time"] = pd.to_numeric(
        m.get("Progression.free.survival"), errors="coerce"
    )
    df["pfs_event"] = pd.to_numeric(
        m.get("Progression..status"), errors="coerce"
    )
    ct = m.get("Cancer_type")
    df["cancer_type"] = (
        ct.astype("string").str.strip() if ct is not None else pd.NA
    )
    return df


def load_cohort_frame(run_dir: str, cohort: str) -> pd.DataFrame | None:
    """Build a per-patient frame for one (winner-run x cohort).

    Columns: risk, os_time, os_event, ORR, recist (canonical), pfs_time,
    pfs_event, cancer_type. Index = Sample.ID. Returns None if cohort
    artifacts are absent. ASSERTS positional length equality.
    """
    npz_path = RUNS_DIR / run_dir / "risks.npz"
    proc_path = PROCESSED_DIR / f"valid_{cohort}_clin.csv"
    clin_path = CLIN_DIR / f"clin_{cohort}.csv"
    if not (npz_path.exists() and proc_path.exists() and clin_path.exists()):
        return None

    z = np.load(npz_path)
    rk, tk, ek = f"risk_{cohort}", f"time_{cohort}", f"event_{cohort}"
    if rk not in z:
        return None
    risk = np.asarray(z[rk], dtype=float)

    proc = pd.read_csv(proc_path, index_col=0)
    if len(risk) != len(proc):
        raise RuntimeError(
            f"LENGTH MISMATCH {run_dir} {cohort}: "
            f"len(risk)={len(risk)} != len(processed)={len(proc)}. STOPPING."
        )

    clin = pd.read_csv(clin_path)
    if "Sample.ID" not in clin.columns:
        raise RuntimeError(f"{cohort}: clin has no Sample.ID column.")
    clin = clin.set_index("Sample.ID")
    m = clin.reindex(proc.index)  # positional: proc.index[i] == Sample.ID

    df = _endpoints_from_clin(m, proc.index)
    df.insert(0, "risk", risk)
    df["os_time"] = np.asarray(z[tk], dtype=float) if tk in z else np.nan
    df["os_event"] = np.asarray(z[ek], dtype=float) if ek in z else np.nan
    return df


# --- endpoint analyses ------------------------------------------------------
def analyze_orr(df: pd.DataFrame, high: pd.Series) -> dict:
    """ORR responder-rate low vs high + Fisher exact, OR, RD, AUROC.

    high-risk is expected to have a LOWER responder rate. AUROC uses
    y_true = non-responder(=1) vs risk score (>0.5 = risk flags non-resp).
    """
    sub = df[df["ORR"].notna()].copy()
    sub["high"] = high.reindex(sub.index).astype(bool)
    n = len(sub)
    out: dict[str, Any] = {"orr_n": n}
    if n == 0:
        return {**out, "orr_skip": "no labelled ORR"}

    lo, hi = sub[~sub["high"]], sub[sub["high"]]
    rl, nrl = int((lo["ORR"] == 1).sum()), int((lo["ORR"] == 0).sum())
    rh, nrh = int((hi["ORR"] == 1).sum()), int((hi["ORR"] == 0).sum())
    out.update(
        orr_low_R=rl, orr_low_NR=nrl, orr_high_R=rh, orr_high_NR=nrh,
        orr_low_n=rl + nrl, orr_high_n=rh + nrh,
    )
    if rl + nrl < MIN_ORR_PER_GROUP or rh + nrh < MIN_ORR_PER_GROUP:
        out["orr_skip"] = (
            f"group <{MIN_ORR_PER_GROUP} labelled "
            f"(low={rl + nrl}, high={rh + nrh})"
        )
        return out

    rate_lo = rl / (rl + nrl)
    rate_hi = rh / (rh + nrh)
    # 2x2: rows=[low, high], cols=[responder, non-responder]
    res = fisher_exact([[rl, nrl], [rh, nrh]])
    auroc = float("nan")
    y = (sub["ORR"] == 0).astype(int).to_numpy()  # non-responder = 1
    if y.min() != y.max():
        auroc = float(roc_auc_score(y, sub["risk"].to_numpy()))
    out.update(
        orr_low_rate=rate_lo, orr_high_rate=rate_hi,
        orr_risk_diff=rate_lo - rate_hi,  # >0 = low responds more (expected)
        orr_OR=float(res.statistic), orr_fisher_p=float(res.pvalue),
        orr_fisher_sig=bool(res.pvalue < 0.05),
        orr_auroc_nonresp=auroc,
        orr_direction_ok=bool(rate_lo > rate_hi),
    )
    return out


def _cox_hr(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> dict:
    """Cox HR for group=1 (high) vs group=0 (low). NaN on failure."""
    d = pd.DataFrame({"T": time, "E": event, "group": group.astype(float)})
    try:
        cph = CoxPHFitter().fit(d, "T", "E")
        s = cph.summary.loc["group"]
        return {
            "hr": float(s["exp(coef)"]),
            "hr_lo": float(s["exp(coef) lower 95%"]),
            "hr_hi": float(s["exp(coef) upper 95%"]),
            "cox_p": float(s["p"]),
        }
    except Exception as e:  # noqa: BLE001
        return {"hr": float("nan"), "hr_lo": float("nan"),
                "hr_hi": float("nan"), "cox_p": float("nan"),
                "cox_err": str(e)[:120]}


def analyze_pfs(df: pd.DataFrame, high: pd.Series) -> dict:
    """PFS Cox HR(high vs low) + log-rank + median PFS per group.

    Requires PFS time AND status present (NaN-aware drop; never imputes an
    event). Skips unless >=MIN_PFS_PER_GROUP labelled and >=1 event each.
    """
    sub = df[df["pfs_time"].notna() & df["pfs_event"].notna()].copy()
    sub["high"] = high.reindex(sub.index).astype(bool)
    n = len(sub)
    n_dropped_status = int(
        (df["pfs_time"].notna() & df["pfs_event"].isna()).sum()
    )
    out: dict[str, Any] = {"pfs_n": n, "pfs_dropped_nan_status": n_dropped_status}
    if n == 0:
        return {**out, "pfs_skip": "no PFS in cohort"}

    lo, hi = sub[~sub["high"]], sub[sub["high"]]
    ev_lo, ev_hi = int((lo["pfs_event"] == 1).sum()), int((hi["pfs_event"] == 1).sum())
    out.update(pfs_low_n=len(lo), pfs_high_n=len(hi),
               pfs_low_events=ev_lo, pfs_high_events=ev_hi)
    if (len(lo) < MIN_PFS_PER_GROUP or len(hi) < MIN_PFS_PER_GROUP
            or ev_lo < 1 or ev_hi < 1):
        out["pfs_skip"] = (
            f"group <{MIN_PFS_PER_GROUP} or 0 events "
            f"(low n={len(lo)} ev={ev_lo}, high n={len(hi)} ev={ev_hi})"
        )
        return out

    cox = _cox_hr(sub["pfs_time"].to_numpy(), sub["pfs_event"].to_numpy(),
                  sub["high"].to_numpy())
    lr = logrank_test(hi["pfs_time"], lo["pfs_time"],
                      hi["pfs_event"], lo["pfs_event"])
    km_lo = KaplanMeierFitter().fit(lo["pfs_time"], lo["pfs_event"])
    km_hi = KaplanMeierFitter().fit(hi["pfs_time"], hi["pfs_event"])
    out.update(
        pfs_hr=cox["hr"], pfs_hr_lo=cox["hr_lo"], pfs_hr_hi=cox["hr_hi"],
        pfs_cox_p=cox["cox_p"], pfs_logrank_p=float(lr.p_value),
        pfs_logrank_sig=bool(lr.p_value < 0.05),
        pfs_median_low=float(km_lo.median_survival_time_),
        pfs_median_high=float(km_hi.median_survival_time_),
        pfs_direction_ok=bool(cox["hr"] > 1) if not np.isnan(cox["hr"]) else None,
    )
    if "cox_err" in cox:
        out["pfs_cox_err"] = cox["cox_err"]
    return out


def analyze_recist(df: pd.DataFrame, high: pd.Series) -> dict:
    """CR/PR/SD/PD counts per risk group + DCB rate per group.

    DCB = best response in {CR,PR,SD,DCB-binned}; non-DCB = {PD,NDB}.
    Unmapped/unknown labels (NE, X, '.', 'Only scanned ...') are dropped.
    """
    sub = df[df["recist"].notna()].copy()
    sub["high"] = high.reindex(sub.index).astype(bool)
    n_drop = int(df["recist_raw"].notna().sum() - sub.shape[0])
    out: dict[str, Any] = {"recist_n": int(sub.shape[0]),
                           "recist_dropped_unknown": n_drop}
    if sub.shape[0] == 0:
        return {**out, "recist_skip": "no mappable RECIST"}

    for grp, mask in (("low", ~sub["high"]), ("high", sub["high"])):
        g = sub[mask]
        for cls in ("CR", "PR", "SD", "PD", "DCB", "NDB"):
            out[f"recist_{grp}_{cls}"] = int((g["recist"] == cls).sum())
        dcb = int(g["recist"].isin(DCB_CLASSES).sum())
        ndb = int(g["recist"].isin(NONDCB_CLASSES).sum())
        out[f"recist_{grp}_DCB_n"] = dcb
        out[f"recist_{grp}_nonDCB_n"] = ndb
        out[f"recist_{grp}_DCB_rate"] = (
            dcb / (dcb + ndb) if (dcb + ndb) > 0 else float("nan")
        )
    return out


# --- pooled aggregates ------------------------------------------------------
def pooled_orr(pool: pd.DataFrame) -> dict:
    """Pooled ORR 2x2 across all patients (one winner)."""
    sub = pool[pool["ORR"].notna()]
    rl = int(((sub["ORR"] == 1) & (~sub["high"])).sum())
    nrl = int(((sub["ORR"] == 0) & (~sub["high"])).sum())
    rh = int(((sub["ORR"] == 1) & (sub["high"])).sum())
    nrh = int(((sub["ORR"] == 0) & (sub["high"])).sum())
    out: dict[str, Any] = {
        "pooled_orr_n": int(len(sub)),
        "pooled_orr_low_R": rl, "pooled_orr_low_NR": nrl,
        "pooled_orr_high_R": rh, "pooled_orr_high_NR": nrh,
    }
    if min(rl + nrl, rh + nrh) < MIN_ORR_PER_GROUP:
        out["pooled_orr_skip"] = "a group <5 labelled"
        return out
    rate_lo, rate_hi = rl / (rl + nrl), rh / (rh + nrh)
    res = fisher_exact([[rl, nrl], [rh, nrh]])
    out.update(
        pooled_orr_low_rate=rate_lo, pooled_orr_high_rate=rate_hi,
        pooled_orr_risk_diff=rate_lo - rate_hi,
        pooled_orr_OR=float(res.statistic),
        pooled_orr_fisher_p=float(res.pvalue),
        pooled_orr_fisher_sig=bool(res.pvalue < 0.05),
        pooled_orr_direction_ok=bool(rate_lo > rate_hi),
    )
    return out


def pooled_pfs(pool: pd.DataFrame) -> dict:
    """Pooled PFS HR via Cox stratified by cohort."""
    sub = pool[pool["pfs_time"].notna() & pool["pfs_event"].notna()].copy()
    out: dict[str, Any] = {"pooled_pfs_n": int(len(sub))}
    n_cohorts = sub["cohort"].nunique()
    if len(sub) < 20 or sub["high"].nunique() < 2 or n_cohorts < 1:
        out["pooled_pfs_skip"] = f"insufficient (n={len(sub)})"
        return out
    d = pd.DataFrame({
        "T": sub["pfs_time"].to_numpy(),
        "E": sub["pfs_event"].to_numpy(),
        "group": sub["high"].astype(float).to_numpy(),
        "cohort": sub["cohort"].to_numpy(),
    })
    try:
        cph = CoxPHFitter().fit(d, "T", "E", strata=["cohort"])
        s = cph.summary.loc["group"]
        out.update(
            pooled_pfs_hr=float(s["exp(coef)"]),
            pooled_pfs_hr_lo=float(s["exp(coef) lower 95%"]),
            pooled_pfs_hr_hi=float(s["exp(coef) upper 95%"]),
            pooled_pfs_cox_p=float(s["p"]),
            pooled_pfs_strata_cohorts=int(n_cohorts),
            pooled_pfs_direction_ok=bool(float(s["exp(coef)"]) > 1),
        )
    except Exception as e:  # noqa: BLE001
        out["pooled_pfs_skip"] = f"cox failed: {str(e)[:100]}"
    return out


def by_cancer(pool: pd.DataFrame) -> list[dict]:
    """Per per-sample Cancer_type: ORR rate low/high (+Fisher) & PFS HR."""
    rows: list[dict] = []
    ctypes = (
        pool["cancer_type"].dropna().astype(str).value_counts()
    )
    for ct, n_total in ctypes.items():
        sub = pool[pool["cancer_type"].astype(str) == ct]
        row: dict[str, Any] = {"cancer_type": ct, "n_total": int(n_total)}
        if n_total < MIN_CANCER_N:
            row["skip"] = f"n<{MIN_CANCER_N}"
            rows.append(row)
            continue
        row.update(_cancer_orr(sub))
        row.update(_cancer_pfs(sub))
        rows.append(row)
    return rows


def _cancer_orr(sub: pd.DataFrame) -> dict:
    o = sub[sub["ORR"].notna()]
    rl = int(((o["ORR"] == 1) & (~o["high"])).sum())
    nrl = int(((o["ORR"] == 0) & (~o["high"])).sum())
    rh = int(((o["ORR"] == 1) & (o["high"])).sum())
    nrh = int(((o["ORR"] == 0) & (o["high"])).sum())
    out: dict[str, Any] = {"orr_n": int(len(o)),
                           "orr_low_n": rl + nrl, "orr_high_n": rh + nrh}
    if min(rl + nrl, rh + nrh) < MIN_ORR_PER_GROUP:
        out["orr_skip"] = "a group <5"
        return out
    rate_lo, rate_hi = rl / (rl + nrl), rh / (rh + nrh)
    res = fisher_exact([[rl, nrl], [rh, nrh]])
    out.update(orr_low_rate=rate_lo, orr_high_rate=rate_hi,
               orr_OR=float(res.statistic), orr_fisher_p=float(res.pvalue),
               orr_fisher_sig=bool(res.pvalue < 0.05),
               orr_direction_ok=bool(rate_lo > rate_hi))
    return out


def _cancer_pfs(sub: pd.DataFrame) -> dict:
    p = sub[sub["pfs_time"].notna() & sub["pfs_event"].notna()]
    out: dict[str, Any] = {"pfs_n": int(len(p))}
    lo, hi = p[~p["high"]], p[p["high"]]
    ev_lo, ev_hi = int((lo["pfs_event"] == 1).sum()), int((hi["pfs_event"] == 1).sum())
    if (len(lo) < MIN_PFS_PER_GROUP or len(hi) < MIN_PFS_PER_GROUP
            or ev_lo < 1 or ev_hi < 1):
        out["pfs_skip"] = f"group<{MIN_PFS_PER_GROUP} or 0 events"
        return out
    cox = _cox_hr(p["pfs_time"].to_numpy(), p["pfs_event"].to_numpy(),
                  p["high"].to_numpy())
    lr = logrank_test(hi["pfs_time"], lo["pfs_time"],
                      hi["pfs_event"], lo["pfs_event"])
    out.update(pfs_hr=cox["hr"], pfs_hr_lo=cox["hr_lo"],
               pfs_hr_hi=cox["hr_hi"], pfs_logrank_p=float(lr.p_value),
               pfs_logrank_sig=bool(lr.p_value < 0.05),
               pfs_direction_ok=bool(cox["hr"] > 1) if not np.isnan(cox["hr"]) else None)
    return out


# --- per-winner driver ------------------------------------------------------
@dataclass
class WinnerResult:
    short: str
    run_dir: str
    cutoff_id: str
    threshold: float
    per_cohort: list[dict]
    pooled: dict
    by_cancer: list[dict]
    skipped: list[str]


def run_winner(short: str, run_dir: str, cutoff_id: str,
               expected: float) -> WinnerResult:
    """Confirm cutoff, analyse every cohort, build pooled + by-cancer."""
    threshold = confirm_cutoff(run_dir, cutoff_id, expected)
    per_cohort: list[dict] = []
    pool_frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for cohort in EVAL_COHORTS:
        df = load_cohort_frame(run_dir, cohort)
        if df is None:
            skipped.append(f"{cohort}: artifacts missing")
            continue
        high = df["risk"] >= threshold
        rec: dict[str, Any] = {
            "winner": short, "cohort": cohort,
            "threshold": threshold, "cutoff_id": cutoff_id,
            "n": int(len(df)), "n_high": int(high.sum()),
            "n_low": int((~high).sum()),
        }
        rec.update(analyze_orr(df, high))
        rec.update(analyze_pfs(df, high))
        rec.update(analyze_recist(df, high))
        rec["direction_ok"] = _direction_flag(rec)
        per_cohort.append(rec)
        # accumulate for pooled / by-cancer
        keep = df.copy()
        keep["high"] = high
        keep["cohort"] = cohort
        pool_frames.append(keep)
        # record endpoint-level skips for the report
        if rec.get("orr_skip"):
            skipped.append(f"{cohort} ORR: {rec['orr_skip']}")
        if rec.get("pfs_skip"):
            skipped.append(f"{cohort} PFS: {rec['pfs_skip']}")
        if rec.get("recist_skip"):
            skipped.append(f"{cohort} RECIST: {rec['recist_skip']}")

    pool = (pd.concat(pool_frames) if pool_frames
            else pd.DataFrame(columns=["ORR", "high", "cohort"]))
    pooled = {**pooled_orr(pool), **pooled_pfs(pool),
              **_aggregate_counts(per_cohort)}
    cancer_rows = by_cancer(pool)
    return WinnerResult(short, run_dir, cutoff_id, threshold,
                        per_cohort, pooled, cancer_rows, skipped)


def _direction_flag(rec: dict) -> bool | None:
    """direction_ok = orr_low>orr_high AND pfs_HR>1 (where computable)."""
    orr_ok = rec.get("orr_direction_ok")
    pfs_ok = rec.get("pfs_direction_ok")
    parts = [v for v in (orr_ok, pfs_ok) if v is not None]
    if not parts:
        return None
    return all(parts)


def _aggregate_counts(per_cohort: list[dict]) -> dict:
    """Count cohorts hitting direction / significance / evaluable criteria."""
    def n_true(key: str) -> int:
        return sum(1 for r in per_cohort if r.get(key) is True)
    n_pfs_eval = sum(
        1 for r in per_cohort
        if isinstance(r.get("pfs_hr"), float) and not np.isnan(r["pfs_hr"])
    )
    n_orr_eval = sum(
        1 for r in per_cohort
        if isinstance(r.get("orr_fisher_p"), float)
        and not np.isnan(r["orr_fisher_p"])
    )
    return {
        "n_cohorts_orr_evaluable": n_orr_eval,
        "n_cohorts_orr_low_gt_high": n_true("orr_direction_ok"),
        "n_cohorts_orr_fisher_sig": n_true("orr_fisher_sig"),
        "n_cohorts_pfs_evaluable": n_pfs_eval,
        "n_cohorts_pfs_hr_gt1": n_true("pfs_direction_ok"),
        "n_cohorts_pfs_logrank_sig": n_true("pfs_logrank_sig"),
    }


# --- output writers ---------------------------------------------------------
def write_per_cohort_csv(res: WinnerResult) -> Path:
    """Write per (winner x cohort) detail CSV."""
    path = RESULTS_DIR / f"{res.short}_per_cohort.csv"
    df = pd.DataFrame(res.per_cohort)
    df.to_csv(path, index=False)
    return path


def build_summary_row(res: WinnerResult) -> dict:
    """One summary row per winner: cutoff + pooled stats + #sig counts."""
    p = res.pooled
    row: dict[str, Any] = {
        "winner": res.short, "run_dir": res.run_dir,
        "cutoff_id": res.cutoff_id, "threshold": res.threshold,
        "n_cohorts_evaluated": len(res.per_cohort),
    }
    # carry every pooled_* / n_cohorts_* scalar through verbatim
    for k, v in p.items():
        row[k] = v
    return row


def write_summary_csv(results: list[WinnerResult]) -> Path:
    """Write one-row-per-winner summary CSV."""
    path = RESULTS_DIR / "enrich_summary.csv"
    rows = [build_summary_row(r) for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def write_by_cancer_csv(results: list[WinnerResult]) -> Path:
    """Write winner x cancer_type CSV."""
    path = RESULTS_DIR / "enrich_by_cancer.csv"
    rows: list[dict] = []
    for r in results:
        for cr in r.by_cancer:
            rows.append({"winner": r.short, "threshold": r.threshold, **cr})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_full_json(results: list[WinnerResult]) -> Path:
    """Write full numpy-safe JSON with all detail."""
    path = RESULTS_DIR / "enrich_full.json"
    blob = {
        r.short: {
            "run_dir": r.run_dir,
            "cutoff_id": r.cutoff_id,
            "threshold": r.threshold,
            "per_cohort": r.per_cohort,
            "pooled": r.pooled,
            "by_cancer": r.by_cancer,
            "skipped": r.skipped,
        }
        for r in results
    }
    path.write_text(json.dumps(_json_safe(blob), indent=2))
    return path


# --- main -------------------------------------------------------------------
def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[WinnerResult] = []
    for short, (run_dir, cid, expected) in WINNERS.items():
        print(f"[enrich] {short}: confirming cutoff {cid} ...", flush=True)
        res = run_winner(short, run_dir, cid, expected)
        pc = write_per_cohort_csv(res)
        print(f"[enrich] {short}: threshold={res.threshold:.5f} "
              f"-> {pc.name} ({len(res.per_cohort)} cohorts)", flush=True)
        for s in res.skipped:
            print(f"    SKIP {s}", flush=True)
        results.append(res)

    sp = write_summary_csv(results)
    bp = write_by_cancer_csv(results)
    jp = write_full_json(results)
    print(f"[enrich] wrote {sp.name}, {bp.name}, {jp.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
