"""Unified training and evaluation for interpretable KG survival models.

Usage:
    # Single experiment
    python src/train_interp.py --model sparse_path --kg primekg --epochs 80

    # All 21 experiments (3 models x 7 KGs)
    python src/train_interp.py --all --epochs 80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import ALL_KGS, VALID_COHORTS, WES_GENE_FILE, load_all_data, _load_gene_list
from losses import c_index, compute_all_metrics, cox_loss
from models_interp import ALL_MODELS, count_parameters, create_model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "output" / "experiments"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train interpretable KG survival models")
    p.add_argument("--model", choices=ALL_MODELS, default="sparse_path")
    p.add_argument("--kg", choices=ALL_KGS, default="primekg")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", action="store_true", help="Run all 3x7=21 experiments")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--feat_dir", type=str, default=None,
                   help="Override KG feature directory (for subgraph variants)")
    p.add_argument("--run_tag", type=str, default=None,
                   help="Custom tag for output dir (default: {model}_{kg})")
    p.add_argument("--wes", action="store_true",
                   help="WES retraining mode: read output/processed_wes/ + KG_DIR/{kg}_wes/")
    p.add_argument("--holdout_cohorts", type=str, default=None,
                   help="Comma-separated holdout cohort list (default: WES_HOLDOUT_COHORTS in wes mode, all VALID_COHORTS otherwise)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _split_data(data: dict, ratio: float, seed: int) -> tuple[dict, dict]:
    """80/20 split by patient index, preserving tensor structure."""
    n = data["mut"].shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = int(n * ratio)

    def _sel(d, idx):
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[idx]
            elif isinstance(v, list):
                idx_list = idx.tolist()
                out[k] = [v[i] for i in idx_list]
            else:
                out[k] = v
        return out

    return _sel(data, perm[:n_train]), _sel(data, perm[n_train:])


def train_epoch(model: nn.Module, data: dict, optimizer: torch.optim.Optimizer,
                batch_size: int, device: torch.device) -> float:
    model.train()
    n = data["mut"].shape[0]
    perm = torch.randperm(n)
    total_loss, n_batches = 0.0, 0

    has_extra = "extra_risk" in data and data["extra_risk"] is not None
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        b_mut = data["mut"][idx].to(device)
        b_mask = data["mask"][idx].to(device)
        b_fmb = data["fmb"][idx].to(device)
        b_time = data["time"][idx].to(device)
        b_event = data["event"][idx].to(device)
        kw = {}
        if has_extra:
            kw["extra_risk"] = data["extra_risk"][idx].to(device)

        optimizer.zero_grad()
        out = model(b_mut, b_mask, b_fmb, **kw)
        loss = cox_loss(out["log_risk"], b_time, b_event)
        if torch.isnan(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_ci(model: nn.Module, data: dict, device: torch.device) -> float:
    model.eval()
    kw = {}
    if "extra_risk" in data and data["extra_risk"] is not None:
        kw["extra_risk"] = data["extra_risk"].to(device)
    out = model(data["mut"].to(device), data["mask"].to(device),
                data["fmb"].to(device), **kw)
    return c_index(out["log_risk"].cpu(), data["time"], data["event"])


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Fix all random sources for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(model_name: str, kg_name: str, args: argparse.Namespace) -> dict:
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print(f"Loading {kg_name} data (wes={args.wes})...")
    feat_dir = Path(args.feat_dir) if args.feat_dir else None
    holdout = (
        [s.strip() for s in args.holdout_cohorts.split(",") if s.strip()]
        if args.holdout_cohorts else None
    )
    all_data = load_all_data(
        kg_name,
        feat_dir=feat_dir,
        wes=args.wes,
        holdout_cohorts=holdout,
    )
    kg_info = all_data["kg_info"]
    train_split, val_split = _split_data(all_data["train"], args.train_ratio, args.seed)

    model = create_model(model_name, kg_info, hidden_dim=args.hidden_dim,
                         dropout=args.dropout)
    model.to(device)
    tot_p, eff_p = count_parameters(model)
    print(f"  Model: {model_name} | Params: {tot_p:,} total, {eff_p:,} effective")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=7, factor=0.5, min_lr=1e-6,
    )

    best_ci, patience_ctr, best_state = 0.0, 0, None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_split, optimizer, args.batch_size, device)
        val_ci = evaluate_ci(model, val_split, device)

        if val_ci > best_ci:
            best_ci = val_ci
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        scheduler.step(val_ci)

        if epoch % 10 == 0 or patience_ctr == 0:
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}  val_ci={val_ci:.4f}  best={best_ci:.4f}")

        if patience_ctr >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # --- Evaluation on all cohorts ---
    results: dict = {"model": model_name, "kg": kg_name}
    results["msk_val_ci"] = evaluate_ci(model, val_split, device)

    ext_cis = []
    risk_scores: dict[str, np.ndarray] = {}
    for cohort, cdata in all_data["valid"].items():
        risk = _get_risk(model, cdata, device)
        risk_scores[cohort] = risk
        t_np = cdata["time"].numpy()
        e_np = cdata["event"].numpy()
        m = compute_all_metrics(risk, t_np, e_np)
        results[f"{cohort}_ci"] = m["c_index"]
        results[f"{cohort}_hr"] = m["hr"]
        results[f"{cohort}_p"] = m["p_value"]
        for k, v in m.items():
            if k.startswith("auc_"):
                results[f"{cohort}_{k}"] = v
        ext_cis.append(m["c_index"])

    results["ext_avg_ci"] = float(np.mean(ext_cis)) if ext_cis else 0.0
    results["total_params"] = tot_p
    results["effective_params"] = eff_p
    results["elapsed_s"] = round(time.time() - t0, 1)

    # --- Save artifacts ---
    run_tag = args.run_tag if args.run_tag else (
        f"{model_name}_{kg_name}_wes" if args.wes else f"{model_name}_{kg_name}"
    )
    out_dir = Path(args.output_dir) / f"interp_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    # Save risk scores for reproducible downstream analysis
    np.savez(out_dir / "risk_scores.npz", **risk_scores)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _save_interpretability(model, all_data, out_dir, device)
    print(f"  => val_ci={results['msk_val_ci']:.4f}  ext_avg={results['ext_avg_ci']:.4f}"
          f"  ({results['elapsed_s']}s)")
    return results


@torch.no_grad()
def _get_risk(model: nn.Module, data: dict, device: torch.device) -> np.ndarray:
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device),
                data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def _save_interpretability(model: nn.Module, all_data: dict, out_dir: Path,
                           device: torch.device) -> None:
    model.eval()
    idir = out_dir / "interpretability"
    idir.mkdir(exist_ok=True)
    train = all_data["train"]
    kg_info = all_data["kg_info"]

    with torch.no_grad():
        out = model(train["mut"].to(device), train["mask"].to(device),
                    train["fmb"].to(device))

    gene_imp = out["gene_importance"].cpu().numpy().mean(axis=0)
    term_imp = out["term_importance"].cpu().numpy().mean(axis=0)
    group_imp = out["group_importance"].cpu().numpy().mean(axis=0)

    # Gene importance CSV
    genes = (
        _load_gene_list(WES_GENE_FILE)
        if all_data.get("kg_info") and getattr(all_data["kg_info"], "n_genes", 0) > 1000
        else _load_gene_list()
    )
    gdf = pd.DataFrame({"gene": genes, "importance": gene_imp})
    gdf.sort_values("importance", ascending=False, inplace=True)
    gdf.to_csv(idir / "gene_importance.csv", index=False)

    # Term importance CSV
    rows = []
    for gi, terms in enumerate(kg_info.term_names):
        s = kg_info.fmb_slices[gi][0]
        for ti, t in enumerate(terms):
            rows.append({"group": kg_info.group_names[gi], "term": t,
                         "importance": float(term_imp[s + ti])})
    pd.DataFrame(rows).sort_values("importance", ascending=False).to_csv(
        idir / "term_importance.csv", index=False)

    # Group importance CSV
    pd.DataFrame({"group": kg_info.group_names, "importance": group_imp}).to_csv(
        idir / "group_importance.csv", index=False)

    # Numpy arrays for notebook use
    np.save(idir / "gene_importance.npy", gene_imp)
    np.save(idir / "term_importance.npy", term_imp)
    np.save(idir / "group_importance.npy", group_imp)


# ---------------------------------------------------------------------------
# All-experiments runner
# ---------------------------------------------------------------------------

def run_all(args: argparse.Namespace) -> None:
    all_results = []
    for model_name in ALL_MODELS:
        for kg_name in ALL_KGS:
            print(f"\n{'=' * 60}")
            print(f"  {model_name} x {kg_name}")
            print(f"{'=' * 60}")
            try:
                r = train_and_evaluate(model_name, kg_name, args)
                all_results.append(r)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                all_results.append({"model": model_name, "kg": kg_name, "error": str(exc)})

    # Summary table
    out_path = Path(args.output_dir) / "interp_summary.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(out_path, index=False)
    print(f"\nSummary -> {out_path}")

    # Leaderboard
    if "ext_avg_ci" in df.columns:
        df_ok = df.dropna(subset=["ext_avg_ci"]).sort_values("ext_avg_ci", ascending=False)
        print(f"\n{'=' * 60}")
        print("LEADERBOARD (by external avg C-index)")
        print(f"{'=' * 60}")
        for _, row in df_ok.iterrows():
            print(f"  {row['model']:20s} x {row['kg']:15s}  "
                  f"val={row['msk_val_ci']:.4f}  ext={row['ext_avg_ci']:.4f}")


def main() -> None:
    args = parse_args()
    if args.all:
        run_all(args)
    else:
        train_and_evaluate(args.model, args.kg, args)


if __name__ == "__main__":
    main()
