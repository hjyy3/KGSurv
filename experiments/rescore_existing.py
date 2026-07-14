"""Re-score existing runs with extended cutoff suite (C1-C10).

Reads risks.npz from each run dir under runs/, applies all 10 cutoff
strategies to each cohort, writes rescored.json next to per_cohort.json.
No training. Run is idempotent: skips dirs with rescored.json + stale-check.

Usage:
  python code/rescore_existing.py            # all runs
  python code/rescore_existing.py --limit 5  # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_ROOT / "code"))

from cutoffs import (  # noqa: E402
    per_cohort_median,
    compute_all_train_cutoffs,
    logrank_split,
    CUTOFF_SCHEMA,
)

RUNS_DIR = EXP_ROOT / "runs"
EVAL_COHORTS = ("Gandara", "Hugo", "Liu", "Mariathasan", "Miao",
                "PUSH", "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae")


def _jsonable(x):
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.ndarray): return x.tolist()
    return None


def rescore_one(run_dir: Path) -> dict | None:
    npz = run_dir / "risks.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    train_risks = z["train"]
    train_time = z["train_time"]
    train_event = z["train_event"]

    thr_meta = compute_all_train_cutoffs(train_risks, train_time, train_event)
    cohort_results = {}
    for c in EVAL_COHORTS:
        rk_key = f"risk_{c}"
        if rk_key not in z.files:
            continue
        risk = z[rk_key]
        time_a = z[f"time_{c}"]
        event_a = z[f"event_{c}"]
        cd = {
            "n": int(len(risk)),
            "events": int(event_a.sum()),
            "c1": {"thr": per_cohort_median(risk),
                   "rule": "per_cohort_median",
                   **logrank_split(risk, time_a, event_a, per_cohort_median(risk))},
        }
        for cid, _label in CUTOFF_SCHEMA:
            thr = thr_meta[cid]["threshold"]
            cd[cid] = {"thr": float(thr),
                       "rule": thr_meta[cid].get("rule", cid),
                       **logrank_split(risk, time_a, event_a, thr)}
        cohort_results[c] = cd

    all_ids = ["c1"] + [cid for cid, _ in CUTOFF_SCHEMA]
    n_sig = {cid: sum(1 for v in cohort_results.values() if v[cid]["sig"])
             for cid in all_ids}
    sigs = {cid: [c for c, v in cohort_results.items() if v[cid]["sig"]]
            for cid in all_ids}
    fixed_max = max(n_sig[cid] for cid in ["c2","c3","c4","c5","c6","c7","c8","c9","c10"])

    return {
        "run_id": run_dir.name,
        "cutoff_thresholds": {cid: float(thr_meta[cid]["threshold"])
                              for cid, _ in CUTOFF_SCHEMA},
        "cutoff_meta": thr_meta,
        "per_cohort": cohort_results,
        "n_sig": n_sig,
        "sigs": sigs,
        "fixed_max_n_sig": fixed_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Re-score even if rescored.json exists")
    args = ap.parse_args()

    dirs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
    if args.limit:
        dirs = dirs[: args.limit]

    print(f"Re-scoring {len(dirs)} runs...")
    t0 = time.time()
    n_done = n_skip = n_err = 0
    for i, d in enumerate(dirs, 1):
        out = d / "rescored.json"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        try:
            r = rescore_one(d)
            if r is None:
                n_err += 1
                continue
            with open(out, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, default=_jsonable)
            n_done += 1
            if i % 50 == 0 or i == len(dirs):
                print(f"  [{i}/{len(dirs)}] last: {d.name} fixed_max={r['fixed_max_n_sig']}",
                      flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {d.name}: {e}", flush=True)
            n_err += 1

    print(f"\nDone: rescored={n_done}  skipped={n_skip}  err={n_err}  "
          f"elapsed={(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
