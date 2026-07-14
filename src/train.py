"""
Training script for MutaPath-Surv — GPU-batched with external validation.

Key improvements over v1:
  - True batch training via DataLoader (no per-patient forward pass)
  - HGT runs ONCE per batch (not once per patient)
  - Proper GPU utilization with CUDA tensors
  - External validation cohort evaluation
  - Comprehensive metrics: C-index, HR, log-rank p-value
"""
from __future__ import annotations

import argparse
import time as time_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "output" / "processed"
OUT_DIR = ROOT / "output" / "models"


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(split: str = "train") -> tuple[np.ndarray, ...]:
    """Returns mut[n,g], mask[n,g], time[n], event[n]."""
    mut = pd.read_csv(PROCESSED / f"{split}_mut.csv",
                      index_col=0).values.astype(np.float32)
    mask = pd.read_csv(PROCESSED / f"{split}_mask.csv",
                       index_col=0).values.astype(np.float32)
    clin = pd.read_csv(PROCESSED / f"{split}_clin.csv", index_col=0)
    t = clin["OS_MONTHS"].values.astype(np.float32)
    e = clin["event"].values.astype(np.float32)
    return mut, mask, t, e


def load_tmb(split: str = "train") -> np.ndarray:
    """Load TMB, aligned to processed samples."""
    clin_proc = pd.read_csv(PROCESSED / f"{split}_clin.csv", index_col=0)
    clin_raw = pd.read_csv(
        ROOT / "source" / "input_data" / "train" / "clin.csv",
        index_col=0, on_bad_lines="skip",
    )
    tmb = (clin_raw.reindex(clin_proc.index)["TMB_NONSYNONYMOUS"]
           .fillna(0).values.astype(np.float32))
    return np.log1p(tmb)


def load_valid_data(cohort: str) -> tuple[np.ndarray, ...] | None:
    """Load external validation cohort data."""
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
    tmb = np.zeros(len(t), dtype=np.float32)
    return mut, mask, t, e, tmb


def get_valid_cohorts() -> list[str]:
    """Discover available validation cohorts."""
    cohorts = []
    for p in sorted(PROCESSED.glob("valid_*_mut.csv")):
        cohort = p.stem.replace("valid_", "").replace("_mut", "")
        cohorts.append(cohort)
    return cohorts


# ── Model Building ───────────────────────────────────────────────────────────

def build_model_and_graph(
    hidden_dim: int, num_heads: int, num_layers: int,
    dropout: float, n_candidate: int, device: torch.device,
):
    """Build graph structure, gene index mapping, and model."""
    import warnings
    warnings.filterwarnings("ignore")
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from graph_builder import build_hetero_graph
    from model import MutaPathSurv

    data, node_maps, gene_map = build_hetero_graph()
    data = data.to(device)

    pw_gene = data["pathway", "contains", "gene/protein"].edge_index.to(device)

    model = MutaPathSurv(
        node_types=list(node_maps.keys()),
        edge_types=data.edge_types,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        n_genes=len(node_maps["gene/protein"]),
        n_pathways=len(node_maps["pathway"]),
        n_candidate=n_candidate,
        dropout=dropout,
    ).to(device)

    return model, data, node_maps, gene_map, pw_gene


def build_gene_indices(
    candidate_genes: list[str], gene_map: dict[str, int], device: torch.device,
) -> torch.Tensor:
    """Map candidate gene names → KG node indices.

    Returns:
        gene_indices [n_candidate] — index into KG gene nodes
            -1 for genes not in KG (will be masked later)
    """
    indices = []
    for g in candidate_genes:
        if g in gene_map:
            indices.append(gene_map[g])
        else:
            indices.append(0)  # fallback to node 0 (will have zero features)
    return torch.tensor(indices, dtype=torch.long, device=device)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_hr(risks: np.ndarray, time: np.ndarray, event: np.ndarray):
    """Compute hazard ratio (high vs low risk) and log-rank p-value."""
    median_risk = np.median(risks)
    high = risks >= median_risk

    try:
        from lifelines.statistics import logrank_test
        result = logrank_test(time[high], time[~high], event[high], event[~high])
        p_value = result.p_value
    except ImportError:
        p_value = float("nan")

    # Simple HR estimate: event rate ratio
    hr_high = event[high].sum() / max(high.sum(), 1)
    hr_low = event[~high].sum() / max((~high).sum(), 1)
    hr = hr_high / max(hr_low, 1e-8)

    return hr, p_value


# ── Training ─────────────────────────────────────────────────────────────────

def evaluate(
    model, mut, mask, tmb, time_arr, event_arr,
    edge_index_dict, pw_gene, gene_indices, device,
    batch_size: int = 128,
) -> dict[str, float]:
    """Evaluate model on a dataset, return metrics."""
    from losses import cox_loss, c_index

    model.eval()
    all_risks = []

    n = len(mut)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mut_b = torch.tensor(mut[start:end], device=device)
            mask_b = torch.tensor(mask[start:end], device=device)
            tmb_b = torch.tensor(tmb[start:end], device=device)

            out = model(mut_b, mask_b, tmb_b, edge_index_dict,
                        pw_gene, gene_indices)
            all_risks.append(out["log_risk"].cpu())

    all_risks_t = torch.cat(all_risks)
    time_t = torch.tensor(time_arr)
    event_t = torch.tensor(event_arr)

    ci = c_index(all_risks_t, time_t, event_t)
    loss = cox_loss(all_risks_t, time_t, event_t).item()
    hr, p_val = compute_hr(all_risks_t.numpy(), time_arr, event_arr)

    return {"c_index": ci, "loss": loss, "hr": hr, "p_value": p_val}


def train(args: argparse.Namespace) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from losses import cox_loss, c_index

    # Device selection
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    mut, mask, time_arr, event_arr = load_data("train")
    tmb = load_tmb("train")
    n_samples, n_genes_data = mut.shape
    print(f"Samples: {n_samples}, Genes: {n_genes_data}")

    # Train/val split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_samples)
    n_train = int(n_samples * args.train_ratio)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    # Candidate gene list
    cand_df = pd.read_csv(
        ROOT / "source" / "input_data" / "train" / "gene_candidate.csv")
    candidate_genes = cand_df.iloc[:, 0].dropna().str.strip().tolist()

    # Build model and graph
    model, data, node_maps, gene_map, pw_gene = build_model_and_graph(
        args.hidden_dim, args.num_heads, args.num_layers,
        args.dropout, len(candidate_genes), device,
    )
    gene_indices = build_gene_indices(candidate_genes, gene_map, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create DataLoader for training
    train_dataset = TensorDataset(
        torch.tensor(mut[train_idx]),
        torch.tensor(mask[train_idx]),
        torch.tensor(tmb[train_idx]),
        torch.tensor(time_arr[train_idx]),
        torch.tensor(event_arr[train_idx]),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=False, pin_memory=(device.type == "cuda"),
    )

    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_cindex = 0.0
    patience_count = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'Val CI':>8} | "
          f"{'Val HR':>8} | {'LR':>10} | {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time_module.time()

        for mut_b, mask_b, tmb_b, time_b, event_b in train_loader:
            if len(mut_b) < 2:
                continue

            mut_b = mut_b.to(device)
            mask_b = mask_b.to(device)
            tmb_b = tmb_b.to(device)
            time_b = time_b.to(device)
            event_b = event_b.to(device)

            out = model(mut_b, mask_b, tmb_b, data.edge_index_dict,
                        pw_gene, gene_indices)

            loss = cox_loss(out["log_risk"], time_b, event_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        val_metrics = evaluate(
            model, mut[val_idx], mask[val_idx], tmb[val_idx],
            time_arr[val_idx], event_arr[val_idx],
            data.edge_index_dict, pw_gene, gene_indices, device,
        )
        val_ci = val_metrics["c_index"]
        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time_module.time() - t_start

        scheduler.step(-val_ci)

        print(f"{epoch:5d} | {avg_loss:8.4f} | {val_ci:8.4f} | "
              f"{val_metrics['hr']:8.2f} | {lr_now:10.6f} | {elapsed:5.1f}s")

        if val_ci > best_cindex:
            best_cindex = val_ci
            patience_count = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"Best validation C-index: {best_cindex:.4f}")
    print(f"Model saved to {OUT_DIR / 'best_model.pt'}")

    # ── External validation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("External Validation")
    print(f"{'='*60}")

    # Load best model
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt",
                                     weights_only=True))

    cohorts = get_valid_cohorts()
    if not cohorts:
        print("No validation cohorts found.")
        return

    results = []
    print(f"\n{'Cohort':<15} | {'N':>5} | {'C-index':>8} | "
          f"{'HR':>8} | {'p-value':>10}")
    print("-" * 55)

    for cohort in cohorts:
        vdata = load_valid_data(cohort)
        if vdata is None:
            continue
        v_mut, v_mask, v_time, v_event, v_tmb = vdata

        if len(v_mut) < 5:
            continue

        metrics = evaluate(
            model, v_mut, v_mask, v_tmb, v_time, v_event,
            data.edge_index_dict, pw_gene, gene_indices, device,
        )

        p_str = (f"{metrics['p_value']:.4f}"
                 if not np.isnan(metrics["p_value"]) else "N/A")
        print(f"{cohort:<15} | {len(v_mut):5d} | {metrics['c_index']:8.4f} | "
              f"{metrics['hr']:8.2f} | {p_str:>10}")

        results.append({"cohort": cohort, "n": len(v_mut), **metrics})

    if results:
        avg_ci = np.mean([r["c_index"] for r in results])
        print(f"\n{'Average':<15} | {'':>5} | {avg_ci:8.4f}")

        # Save results
        pd.DataFrame(results).to_csv(
            OUT_DIR / "external_validation.csv", index=False)
        print(f"\nResults saved to {OUT_DIR / 'external_validation.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MutaPath-Surv Training")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
