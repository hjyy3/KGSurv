"""Scan entry point.

Usage:
  python code/scan.py --tier 0 --device cuda
  python code/scan.py --tier 1 --device cuda
  python code/scan.py --tier 0 --limit 3  # smoke-of-smoke for debugging
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_ROOT / "code"))

import torch  # noqa: E402

from spec import (  # noqa: E402
    tier_0_specs, tier_1_specs, tier_2_specs, tier_3_specs,
    tier_4_specs, tier_5_specs, tier_6_specs,
)
from replay_engine import run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--save_ckpt_min_sig", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    specs = (tier_0_specs() if args.tier == 0
             else tier_1_specs() if args.tier == 1
             else tier_2_specs() if args.tier == 2
             else tier_3_specs() if args.tier == 3
             else tier_4_specs() if args.tier == 4
             else tier_5_specs() if args.tier == 5
             else tier_6_specs())
    if args.limit:
        specs = specs[: args.limit]

    print(f"Tier {args.tier}: {len(specs)} runs on {device}")
    print(f"Save ckpt threshold: >={args.save_ckpt_min_sig} fixed-cutoff sig\n")

    t_start = time.time()
    n_done = n_err = 0
    for i, sp in enumerate(specs, 1):
        t0 = time.time()
        try:
            r = run_one(sp, device, args.save_ckpt_min_sig)
            ns = r["n_sig"]
            # Tolerant to old runs (no c4-c10): only sum keys that exist
            fixed_max = max(ns.get(cid, 0) for cid in
                            ["c2","c3","c4","c5","c6","c7","c8","c9","c10"])
            val = r["best_val_ci"]
            extra = (f"c4={ns.get('c4','-')} c10={ns.get('c10','-')}"
                     if 'c4' in ns else 'OLD')
            print(f"[{i:3d}/{len(specs)}] {sp.run_id:62s}  "
                  f"val={val:.4f}  c1={ns['c1']} c2={ns['c2']} c3={ns['c3']} "
                  f"{extra} max={fixed_max}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            n_done += 1
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[{i:3d}/{len(specs)}] {sp.run_id:62s}  ERROR:",
                  flush=True)
            print(tb, flush=True)
            n_err += 1
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    total = time.time() - t_start
    print(f"\nTotal: {total/60:.1f} min  done={n_done}  err={n_err}")


if __name__ == "__main__":
    main()
