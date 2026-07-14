"""Train KGLiteMLP — lightweight KG-feature-injected survival model.

Loads pre-computed FMB + Node2Vec features (from kg_features.py),
trains a simple MLP with Cox partial likelihood, and evaluates on
the same 3-tier protocol (train-val-external) as run_ablation.py.

Usage:
    python src/train_lite.py --kg primekg --epochs 80
    python src/train_lite.py --kg none --epochs 80   # baseline (no KG)
    python src/train_lite.py --all --epochs 80        # all KGs + none
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "output" / "processed"
FEAT_DIR = ROOT / "output" / "kg_features"
EXP_DIR = ROOT / "output" / "experiments"

AVAILABLE_KGS = [
    "primekg", "hetionet", "drkg", "ibkh",
    "monarch", "ogb_biokg", "openbiolink",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_base_data(split: str = "train"):
    """Load mutation matrix, panel mask, and clinical outcomes."""
    mut = pd.read_csv(PROCESSED / f"{split}_mut.csv",
                      index_col=0).values.astype(np.float32)
    mask = pd.read_csv(PROCESSED / f"{split}_mask.csv",
                       index_col=0).values.astype(np.float32)
    clin = pd.read_csv(PROCESSED / f"{split}_clin.csv", index_col=0)
    t = clin["OS_MONTHS"].values.astype(np.float32)
    e = clin["event"].values.astype(np.float32)
    return mut, mask, t, e


def load_tmb() -> np.ndarray:
    """Load log1p-transformed TMB for the training cohort."""
    clin_proc = pd.read_csv(PROCESSED / "train_clin.csv", index_col=0)
    clin_raw = pd.read_csv(
        ROOT / "source" / "input_data" / "train" / "clin.csv",
        index_col=0, on_bad_lines="skip",
    )
    tmb = (clin_raw.reindex(clin_proc.index)["TMB_NONSYNONYMOUS"]
           .fillna(0).values.astype(np.float32))
    return np.log1p(tmb)


def load_kg_features(kg_name: str, split: str = "train"):
    """Load pre-computed FMB and KG-embedding CSVs.

    Returns (fmb | None, kgemb | None).
    """
    feat_dir = FEAT_DIR / kg_name
    fmb, kgemb = None, None

    fmb_path = feat_dir / f"{split}_fmb.csv"
    if fmb_path.exists():
        fmb = pd.read_csv(fmb_path, index_col=0).values.astype(np.float32)

    kgemb_path = feat_dir / f"{split}_kgemb.csv"
    if kgemb_path.exists():
        kgemb = pd.read_csv(kgemb_path, index_col=0).values.astype(np.float32)

    return fmb, kgemb


def build_feature_tensor(
    mut: np.ndarray,
    mask: np.ndarray,
    fmb: np.ndarray | None,
    kgemb: np.ndarray | None,
    tmb: np.ndarray,
) -> np.ndarray:
    """Concatenate [mut, mask, fmb?, kgemb?, tmb] → [n, D]."""
    parts = [mut, mask]
    if fmb is not None:
        parts.append(fmb)
    if kgemb is not None:
        parts.append(kgemb)
    parts.append(tmb[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


def get_valid_cohorts() -> list[str]:
    return [p.stem.replace("valid_", "").replace("_mut", "")
            for p in sorted(PROCESSED.glob("valid_*_mut.csv"))]


def load_valid_all(kg_name: str, cohort: str):
    """Load validation base data + KG features, build feature tensor."""
    prefix = f"valid_{cohort}"
    mut_path = PROCESSED / f"{prefix}_mut.csv"
    if not mut_path.exists():
        return None

    mut = pd.read_csv(mut_path, index_col=0).values.astype(np.float32)
    mask = pd.read_csv(PROCESSED / f"{prefix}_mask.csv",
                       index_col=0).values.astype(np.float32)
    clin = pd.read_csv(PROCESSED / f"{prefix}_clin.csv", index_col=0)
    t = clin["OS_MONTHS"].values.astype(np.float32)
    e = clin["event"].values.astype(np.float32)
    tmb_val = np.zeros(len(t), dtype=np.float32)

    fmb, kgemb = None, None
    if kg_name != "none":
        fmb, kgemb = load_kg_features(kg_name, prefix)

    features = build_feature_tensor(mut, mask, fmb, kgemb, tmb_val)
    return features, t, e


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_risks(model, features: np.ndarray, device, batch_size: int = 128):
    model.eval()
    parts: list[np.ndarray] = []
    n = len(features)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.tensor(features[start:end], device=device)
            out = model(batch)
            parts.append(out["log_risk"].cpu().numpy())
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(kg_name: str, args):
    """Train KGLiteMLP on one KG config and evaluate 3-tier."""
    sys.path.insert(0, str(ROOT / "src"))
    from losses import cox_loss, c_index as ci_fn, compute_all_metrics
    from model_lite import KGLiteMLP

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"  Device: {device}")

    # --- Load data ---
    mut, mask, time_arr, event_arr = load_base_data("train")
    tmb = load_tmb()

    fmb, kgemb = None, None
    if kg_name != "none":
        fmb, kgemb = load_kg_features(kg_name, "train")
        if fmb is not None:
            print(f"  FMB features: {fmb.shape[1]}")
        if kgemb is not None:
            print(f"  KG embedding dim: {kgemb.shape[1]}")

    features = build_feature_tensor(mut, mask, fmb, kgemb, tmb)
    input_dim = features.shape[1]
    print(f"  Input dim: {input_dim} "
          f"(mut={mut.shape[1]}, mask={mask.shape[1]}, "
          f"fmb={fmb.shape[1] if fmb is not None else 0}, "
          f"kgemb={kgemb.shape[1] if kgemb is not None else 0}, tmb=1)")

    # --- Train / val split ---
    n_samples = len(features)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_samples)
    n_train = int(n_samples * args.train_ratio)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = TensorDataset(
        torch.tensor(features[train_idx]),
        torch.tensor(time_arr[train_idx]),
        torch.tensor(event_arr[train_idx]),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # --- Model ---
    model = KGLiteMLP(
        input_dim, hidden_dims=(256, 128, 64), dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    # --- Training loop ---
    best_ci = 0.0
    patience_count = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feat_b, time_b, event_b = [x.to(device) for x in batch]
            if len(feat_b) < 2:
                continue

            out = model(feat_b)
            loss = cox_loss(out["log_risk"], time_b, event_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        risks_val = predict_risks(model, features[val_idx], device)
        val_ci = ci_fn(
            torch.tensor(risks_val),
            torch.tensor(time_arr[val_idx]),
            torch.tensor(event_arr[val_idx]),
        )

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(-val_ci)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Loss {avg_loss:.4f} "
                  f"| Val CI {val_ci:.4f}")

        if val_ci > best_ci:
            best_ci = val_ci
            patience_count = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best Val CI: {best_ci:.4f}")

    # --- Save model ---
    out_dir = EXP_DIR / f"lite_{kg_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")

    # --- External validation ---
    cohorts = get_valid_cohorts()
    ext_metrics: list[dict] = []

    for cohort in cohorts:
        vdata = load_valid_all(kg_name, cohort)
        if vdata is None or len(vdata[0]) < 5:
            continue
        v_features, v_time, v_event = vdata
        risks = predict_risks(model, v_features, device)
        m = compute_all_metrics(risks, v_time, v_event)
        m["cohort"] = cohort
        ext_metrics.append(m)

    result: dict = {
        "kg_name": kg_name,
        "val_ci": best_ci,
        "n_params": n_params,
        "input_dim": input_dim,
        "n_fmb": fmb.shape[1] if fmb is not None else 0,
    }

    if ext_metrics:
        avg_ci = np.mean([m["c_index"] for m in ext_metrics])
        avg_hr = np.mean([m["hr"] for m in ext_metrics
                          if not np.isnan(m["hr"])])
        n_sig = sum(1 for m in ext_metrics if m["p_value"] < 0.05)
        result["ext_avg_ci"] = avg_ci
        result["ext_avg_hr"] = avg_hr
        result["n_sig"] = n_sig
        result["n_cohorts"] = len(ext_metrics)

        print(f"  External: avg CI={avg_ci:.4f}, avg HR={avg_hr:.2f}, "
              f"sig={n_sig}/{len(ext_metrics)}")

    pd.DataFrame(ext_metrics).to_csv(
        out_dir / "cohort_metrics.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train KGLiteMLP")
    parser.add_argument("--kg", type=str, default="primekg",
                        help="KG name, or 'none' for baseline")
    parser.add_argument("--all", action="store_true",
                        help="Train all 7 KGs + none baseline")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    kgs = AVAILABLE_KGS + ["none"] if args.all else [args.kg]

    results: list[dict] = []
    for kg in kgs:
        print(f"\n{'=' * 60}")
        print(f"Training: lite_{kg}")
        print(f"{'=' * 60}")
        result = train_and_evaluate(kg, args)
        results.append(result)

    # --- Comparison summary ---
    if len(results) > 1:
        summary = pd.DataFrame(results)
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        print(summary.to_string(index=False))

        comp_dir = EXP_DIR / "lite_comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(comp_dir / "comparison_summary.csv", index=False)
        print(f"\nSaved to {comp_dir}/")


if __name__ == "__main__":
    main()
