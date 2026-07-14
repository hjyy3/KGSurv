"""Cox partial likelihood loss, C-index, and time-dependent AUC (vectorized)."""
from __future__ import annotations

import numpy as np
import torch


def cox_loss(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (logcumsumexp risk-set; no tie correction).

    Args:
        log_risk: predicted log-risk scores [N]
        time: survival times [N]
        event: event indicators (1=event, 0=censored) [N]
    """
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]

    log_cumsum = torch.logcumsumexp(log_risk, dim=0)
    loss = -torch.mean((log_risk - log_cumsum) * event)
    return loss


def cox_loss_stratified(log_risk: torch.Tensor, time: torch.Tensor,
                          event: torch.Tensor, stratum: torch.Tensor) -> torch.Tensor:
    """Stratified Cox PL: 每 stratum 独立排序与 log-cumsum 后求和除以总 events.

    消除跨 stratum 的 partial-likelihood 比较, 阻断跨癌种梯度污染.

    Args:
        log_risk: [N]
        time: [N]
        event: [N]
        stratum: [N] long, 每个 sample 的癌种 id
    """
    total_loss = log_risk.new_zeros(())
    total_events = log_risk.new_zeros(())
    for s in stratum.unique():
        sel = stratum == s
        if event[sel].sum() < 1:
            continue
        lr_s, t_s, e_s = log_risk[sel], time[sel], event[sel]
        order = torch.argsort(t_s, descending=True)
        log_cumsum = torch.logcumsumexp(lr_s[order], dim=0)
        stratum_nll = -((lr_s[order] - log_cumsum) * e_s[order]).sum()
        total_loss = total_loss + stratum_nll
        total_events = total_events + e_s.sum()
    return total_loss / total_events.clamp(min=1.0)


def c_index(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """Harrell's C-index — vectorized O(n²) via broadcasting, GPU-friendly.

    Computes all pairwise comparisons in a single tensor operation
    instead of Python nested loops.
    """
    with torch.no_grad():
        # Only consider pairs where subject i had an event
        event_mask = event == 1  # [N]
        n = len(time)

        if n < 2 or event_mask.sum() == 0:
            return 0.5

        # Pairwise comparison: time[j] > time[i] for event subjects i
        # Shape: [n_events, N]
        t_event = time[event_mask]       # [n_events]
        r_event = log_risk[event_mask]   # [n_events]

        # time[j] > time[i] → j survived longer than i
        permissible = time.unsqueeze(0) > t_event.unsqueeze(1)  # [n_events, N]

        # r_event[i] > log_risk[j] → model correctly assigns higher risk to i
        concordant = r_event.unsqueeze(1) > log_risk.unsqueeze(0)  # [n_events, N]
        tied = r_event.unsqueeze(1) == log_risk.unsqueeze(0)       # [n_events, N]

        n_permissible = permissible.sum().item()
        if n_permissible == 0:
            return 0.5

        n_concordant = (permissible & concordant).sum().item()
        n_tied = (permissible & tied).sum().item()

        return (n_concordant + 0.5 * n_tied) / n_permissible


def time_dependent_auc(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t: float,
) -> float:
    """Incident/dynamic AUC at time t (Heagerty & Zheng, 2005).

    Cases: event before t.  Controls: alive at t.
    AUC = P(risk_i > risk_j | i is case, j is control).
    """
    case = (time <= t) & (event == 1)
    ctrl = time > t
    n_case = case.sum()
    n_ctrl = ctrl.sum()
    if n_case == 0 or n_ctrl == 0:
        return float("nan")

    r_case = risk[case]
    r_ctrl = risk[ctrl]
    concordant = 0
    tied = 0
    for rc in r_case:
        concordant += (rc > r_ctrl).sum()
        tied += (rc == r_ctrl).sum()
    total = n_case * n_ctrl
    return float((concordant + 0.5 * tied) / total)


# ---------------------------------------------------------------------------
# New evaluation metrics: Bootstrap CI, ARR/NNT, DCA, td-AUC curve
# ---------------------------------------------------------------------------


def bootstrap_c_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap C-index with 95% CI."""
    import torch as _torch

    rng = np.random.RandomState(seed)
    n = len(risk)
    cis = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ci = c_index(
            _torch.tensor(risk[idx], dtype=_torch.float32),
            _torch.tensor(time[idx], dtype=_torch.float32),
            _torch.tensor(event[idx], dtype=_torch.float32),
        )
        cis.append(ci)
    cis = np.array(cis)
    return {
        "boot_ci_mean": float(np.mean(cis)),
        "boot_ci_lo": float(np.percentile(cis, 2.5)),
        "boot_ci_hi": float(np.percentile(cis, 97.5)),
    }


def km_survival_at(time: np.ndarray, event: np.ndarray, t: float) -> float:
    """Kaplan-Meier survival probability at time t."""
    order = np.argsort(time)
    time_s, event_s = time[order], event[order]
    surv = 1.0
    n_at_risk = len(time_s)
    for i, (ti, ei) in enumerate(zip(time_s, event_s)):
        if ti > t:
            break
        if ei == 1:
            surv *= 1.0 - 1.0 / max(n_at_risk, 1)
        n_at_risk -= 1
    return surv


def compute_arr_nnt(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    timepoints: tuple[float, ...] = (12.0, 24.0, 36.0),
) -> dict[str, float]:
    """ARR and NNT between high/low risk groups at given timepoints."""
    median_risk = np.median(risk)
    high = risk >= median_risk
    low = ~high
    results = {}
    for tp in timepoints:
        key = int(tp)
        if high.sum() < 2 or low.sum() < 2:
            results[f"surv_high_{key}m"] = float("nan")
            results[f"surv_low_{key}m"] = float("nan")
            results[f"arr_{key}m"] = float("nan")
            results[f"nnt_{key}m"] = float("nan")
            continue
        s_high = km_survival_at(time[high], event[high], tp)
        s_low = km_survival_at(time[low], event[low], tp)
        arr = s_low - s_high  # positive = low-risk survives more
        nnt = 1.0 / max(abs(arr), 1e-8) if abs(arr) > 0.001 else float("nan")
        results[f"surv_high_{key}m"] = s_high
        results[f"surv_low_{key}m"] = s_low
        results[f"arr_{key}m"] = arr
        results[f"nnt_{key}m"] = nnt
    return results


def compute_dca(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t: float = 24.0,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Decision Curve Analysis at time t.

    Returns threshold array and net benefit for: model, treat-all, treat-none.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    # Event rate at time t (1 - KM survival)
    event_rate = 1.0 - km_survival_at(time, event, t)
    n = len(risk)

    # Convert risk scores to predicted probabilities via ranking
    from scipy.stats import rankdata
    risk_rank = rankdata(risk) / len(risk)  # percentile-based "probability"

    nb_model = np.full_like(thresholds, np.nan)
    nb_all = np.full_like(thresholds, np.nan)

    for i, pt in enumerate(thresholds):
        # Treat all: net benefit
        nb_all[i] = event_rate - (1.0 - event_rate) * pt / max(1.0 - pt, 1e-8)

        # Model-based: predict positive if risk_rank > 1-pt (top pt fraction)
        pred_pos = risk_rank >= (1.0 - pt)
        n_pos = pred_pos.sum()
        if n_pos == 0:
            nb_model[i] = 0.0
            continue

        # True positive rate among predicted positives
        # Cases = event before t; controls = survived past t
        case = (time <= t) & (event == 1)
        ctrl = time > t
        tp = (pred_pos & case).sum()
        fp = (pred_pos & ctrl).sum()
        nb_model[i] = tp / n - fp / n * pt / max(1.0 - pt, 1e-8)

    return {
        "thresholds": thresholds,
        "nb_model": nb_model,
        "nb_all": nb_all,
        "nb_none": np.zeros_like(thresholds),
    }


# ---------------------------------------------------------------------------
# Brier Score & Integrated Brier Score (IBS)
# ---------------------------------------------------------------------------

def brier_score_at(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t: float,
) -> float:
    """Time-dependent Brier Score at time t using IPCW (inverse probability of censoring weighting).

    BS(t) = (1/n) * sum_i [ w_i * (S_hat_i(t) - I(T_i > t))^2 ]
    """
    from lifelines import KaplanMeierFitter

    n = len(risk)
    if n < 10:
        return float("nan")

    # Estimate censoring distribution G(t) = P(C > t) using KM on censoring times
    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(time, 1 - event)  # fit on censoring indicator

    def _g(t_val):
        """Get censoring survival probability at time t_val."""
        pred = kmf_censor.predict(t_val)
        v = pred.values[0] if hasattr(pred, 'values') else float(pred)
        return max(v, 1e-4)

    # Convert risk scores to survival probabilities via ranking
    # Higher risk → lower survival probability
    from scipy.stats import rankdata
    surv_pred = 1.0 - rankdata(risk) / n  # predicted S(t)

    bs = 0.0
    for i in range(n):
        if time[i] <= t and event[i] == 1:
            # Died before t: prediction should be 0 (dead)
            bs += (surv_pred[i] - 0.0) ** 2 / _g(time[i])
        elif time[i] > t:
            # Alive at t: prediction should be 1 (alive)
            bs += (surv_pred[i] - 1.0) ** 2 / _g(t)
        # Censored before t: excluded (IPCW handles this)

    return bs / n


def integrated_brier_score(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    t_max: float | None = None,
    n_points: int = 20,
) -> float:
    """Integrated Brier Score (IBS) over [0, t_max].

    IBS = (1/t_max) * integral_0^t_max BS(t) dt
    Lower is better. Random model ≈ 0.25.
    """
    if t_max is None:
        t_max = float(np.percentile(time[time > 0], 90))

    t_grid = np.linspace(1.0, t_max, n_points)
    bs_values = []
    for t in t_grid:
        bs = brier_score_at(risk, time, event, t)
        if not np.isnan(bs):
            bs_values.append(bs)

    if len(bs_values) < 3:
        return float("nan")

    return float(np.trapz(bs_values, t_grid[:len(bs_values)]) / t_max)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_stats(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_groups: int = 5,
    t: float = 24.0,
) -> dict:
    """Calibration: compare predicted risk quantiles with observed event rates at time t.

    Returns per-group predicted risk rank, observed event rate, and
    calibration slope/intercept from linear regression.
    """
    n = len(risk)
    if n < n_groups * 5:
        return {"slope": float("nan"), "intercept": float("nan"), "groups": []}

    # Sort by risk and divide into quantile groups
    order = np.argsort(risk)
    group_size = n // n_groups

    groups = []
    pred_means, obs_rates = [], []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n
        idx = order[start:end]

        pred_risk_mean = float(np.mean(risk[idx]))
        # Observed event rate at time t
        obs_rate = float(np.mean((time[idx] <= t) & (event[idx] == 1)))

        groups.append({
            "group": g + 1,
            "n": len(idx),
            "pred_risk_mean": round(pred_risk_mean, 4),
            "obs_event_rate": round(obs_rate, 4),
        })
        pred_means.append(pred_risk_mean)
        obs_rates.append(obs_rate)

    # Linear calibration: obs = intercept + slope * pred
    pred_means = np.array(pred_means)
    obs_rates = np.array(obs_rates)
    if np.std(pred_means) < 1e-8:
        slope, intercept = float("nan"), float("nan")
    else:
        slope, intercept = np.polyfit(pred_means, obs_rates, 1)

    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "groups": groups,
    }


def compute_all_metrics(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    timepoints: tuple[float, ...] = (12.0, 24.0, 36.0),
) -> dict[str, float]:
    """Compute comprehensive survival metrics.

    Returns dict with: c_index, hr, p_value, auc_12m, auc_24m, auc_36m.
    """
    from lifelines.statistics import logrank_test

    risk_t = torch.tensor(risk, dtype=torch.float32)
    time_t = torch.tensor(time, dtype=torch.float32)
    event_t = torch.tensor(event, dtype=torch.float32)

    ci = c_index(risk_t, time_t, event_t)

    # HR and log-rank test
    median_risk = np.median(risk)
    high = risk >= median_risk

    n_high = high.sum()
    n_low = (~high).sum()
    if n_high < 2 or n_low < 2:
        hr, p_val = float("nan"), float("nan")
    else:
        try:
            result = logrank_test(time[high], time[~high],
                                  event[high], event[~high])
            p_val = result.p_value
        except Exception:
            p_val = float("nan")

        ev_high = event[high].sum() / max(n_high, 1)
        ev_low = event[~high].sum() / max(n_low, 1)
        hr = ev_high / max(ev_low, 1e-8)

    metrics = {"c_index": ci, "hr": hr, "p_value": p_val}

    for tp in timepoints:
        auc = time_dependent_auc(risk, time, event, tp)
        metrics[f"auc_{int(tp)}m"] = auc

    return metrics
