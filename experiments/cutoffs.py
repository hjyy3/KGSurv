"""Risk-stratification cutoff strategies (extended).

Convention: every function returns a scalar threshold `tau` (or a dict whose
`threshold` key is `tau`). High-risk group is defined as `risk >= tau`.

C1 per_cohort_median       — each external cohort uses its own median (legacy)
C2 train_median            — fixed cutoff = MSK fold-train median
C3 train_roc_at(t=24)      — fixed: ROC@24m, threshold minimising distance to (0,1)
C4 train_youden_at(t=24)   — fixed: argmax(TPR - FPR) on ROC@24m
C5 train_percentile(60)    — fixed: 60th percentile (high group ~ top 40%)
C6 train_percentile(67)    — fixed: 67th percentile (high group ~ top 1/3)
C7 train_percentile(75)    — fixed: 75th percentile (high group ~ top 1/4)
C8 train_roc_at(t=12)      — early-event cutoff
C9 train_roc_at(t=36)      — long-term cutoff
C10 train_mean             — simple replacement for median when distribution skewed

All "train_*" cutoffs are derived ONLY from training-fold risks/labels — no
holdout leakage. Each is single-patient deployable: threshold is a scalar
fixed before seeing any new patient.

Binary label for ROC/Youden at time t:
  positive : time <= t and event == 1     (died by t)
  negative : time >  t                    (alive at t)
  excluded : event == 0 and time <= t     (censored before t — unknown status)
"""
from __future__ import annotations

import numpy as np
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve


def per_cohort_median(risk_cohort: np.ndarray) -> float:
    return float(np.median(risk_cohort))


def train_median(train_risks: np.ndarray) -> float:
    return float(np.median(train_risks))


def train_mean(train_risks: np.ndarray) -> float:
    return float(np.mean(train_risks))


def train_percentile(train_risks: np.ndarray, q: float) -> float:
    return float(np.percentile(train_risks, q))


def _binary_labels(time: np.ndarray, event: np.ndarray, t: float):
    pos = (time <= t) & (event == 1)
    neg = time > t
    keep = pos | neg
    return pos, neg, keep


def train_roc_at(
    train_risks: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    t: float = 24.0,
) -> dict:
    """Threshold minimising sqrt(FPR^2 + (1-TPR)^2) on ROC at time t."""
    pos, neg, keep = _binary_labels(train_time, train_event, t)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos < 5 or n_neg < 5:
        return {
            "threshold": float(np.median(train_risks)),
            "n_pos": n_pos, "n_neg": n_neg,
            "fpr": float("nan"), "tpr": float("nan"),
            "distance": float("nan"),
            "fallback": "median_due_to_low_n",
        }
    fpr, tpr, thr = roc_curve(pos[keep].astype(int), train_risks[keep])
    dist = np.sqrt(fpr ** 2 + (1.0 - tpr) ** 2)
    best = int(np.argmin(dist))
    return {
        "threshold": float(thr[best]),
        "n_pos": n_pos, "n_neg": n_neg,
        "fpr": float(fpr[best]),
        "tpr": float(tpr[best]),
        "distance": float(dist[best]),
        "fallback": None,
    }


def train_youden_at(
    train_risks: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
    t: float = 24.0,
) -> dict:
    """Threshold maximising Youden's J = TPR - FPR on ROC at time t."""
    pos, neg, keep = _binary_labels(train_time, train_event, t)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos < 5 or n_neg < 5:
        return {
            "threshold": float(np.median(train_risks)),
            "n_pos": n_pos, "n_neg": n_neg,
            "fpr": float("nan"), "tpr": float("nan"),
            "youden": float("nan"),
            "fallback": "median_due_to_low_n",
        }
    fpr, tpr, thr = roc_curve(pos[keep].astype(int), train_risks[keep])
    youden = tpr - fpr
    best = int(np.argmax(youden))
    return {
        "threshold": float(thr[best]),
        "n_pos": n_pos, "n_neg": n_neg,
        "fpr": float(fpr[best]),
        "tpr": float(tpr[best]),
        "youden": float(youden[best]),
        "fallback": None,
    }


# ── Cutoff schema used by replay_engine + rescore ───────────────────────────
# Each entry: (id, label, callable returning threshold-scalar OR info-dict)
CUTOFF_SCHEMA: list[tuple[str, str]] = [
    ("c2", "train_median"),
    ("c3", "roc24m"),
    ("c4", "youden24m"),
    ("c5", "p60"),
    ("c6", "p67"),
    ("c7", "p75"),
    ("c8", "roc12m"),
    ("c9", "roc36m"),
    ("c10", "train_mean"),
]


def compute_all_train_cutoffs(
    train_risks: np.ndarray,
    train_time: np.ndarray,
    train_event: np.ndarray,
) -> dict:
    """Return dict of {cutoff_id: {threshold + meta}} for all C2-C10."""
    return {
        "c2":  {"threshold": train_median(train_risks),
                "rule": "train_median"},
        "c3":  {**train_roc_at(train_risks, train_time, train_event, 24.0),
                "rule": "roc24m"},
        "c4":  {**train_youden_at(train_risks, train_time, train_event, 24.0),
                "rule": "youden24m"},
        "c5":  {"threshold": train_percentile(train_risks, 60.0),
                "rule": "p60"},
        "c6":  {"threshold": train_percentile(train_risks, 67.0),
                "rule": "p67"},
        "c7":  {"threshold": train_percentile(train_risks, 75.0),
                "rule": "p75"},
        "c8":  {**train_roc_at(train_risks, train_time, train_event, 12.0),
                "rule": "roc12m"},
        "c9":  {**train_roc_at(train_risks, train_time, train_event, 36.0),
                "rule": "roc36m"},
        "c10": {"threshold": train_mean(train_risks),
                "rule": "train_mean"},
    }


def logrank_split(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    threshold: float,
) -> dict:
    high = risk >= threshold
    low = ~high
    n_high, n_low = int(high.sum()), int(low.sum())
    if n_high < 2 or n_low < 2:
        return {"n_high": n_high, "n_low": n_low,
                "hr": float("nan"), "p": float("nan"), "sig": False}
    res = logrank_test(time[high], time[low], event[high], event[low])
    ev_h = event[high].sum() / max(n_high, 1)
    ev_l = event[low].sum() / max(n_low, 1)
    hr = ev_h / max(ev_l, 1e-8)
    return {
        "n_high": n_high, "n_low": n_low,
        "hr": float(hr), "p": float(res.p_value),
        "sig": bool(res.p_value < 0.05),
    }
