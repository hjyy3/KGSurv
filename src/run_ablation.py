"""Ablation experiment runner.

Trains 4 model variants on the best KG and compares:
  1. Full MutaPathSurv (reference)
  2. BaselineMLP (no KG/GNN/pathway)
  3. MutaPathSurvNoMask (no panel mask)
  4. MutaPathSurvNoPathway (no pathway pooling)

Usage:
    python src/run_ablation.py --kg monarch --epochs 80
"""
from __future__ import annotations

import argparse
import json
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
SUBKG_DIR = ROOT / "output" / "subkg"
EXP_DIR = ROOT / "output" / "experiments"


def load_data(split="train"):
    mut = pd.read_csv(PROCESSED / f"{split}_mut.csv", index_col=0).values.astype(np.float32)
    mask = pd.read_csv(PROCESSED / f"{split}_mask.csv", index_col=0).values.astype(np.float32)
    clin = pd.read_csv(PROCESSED / f"{split}_clin.csv", index_col=0)
    t = clin["OS_MONTHS"].values.astype(np.float32)
    e = clin["event"].values.astype(np.float32)
    return mut, mask, t, e


def load_tmb():
    clin_proc = pd.read_csv(PROCESSED / "train_clin.csv", index_col=0)
    clin_raw = pd.read_csv(
        ROOT / "source" / "input_data" / "train" / "clin.csv",
        index_col=0, on_bad_lines="skip",
    )
    tmb = (clin_raw.reindex(clin_proc.index)["TMB_NONSYNONYMOUS"]
           .fillna(0).values.astype(np.float32))
    return np.log1p(tmb)


def load_valid_data(cohort):
    prefix = f"valid_{cohort}"
    mut_path = PROCESSED / f"{prefix}_mut.csv"
    if not mut_path.exists():
        return None
    mut = pd.read_csv(mut_path, index_col=0).values.astype(np.float32)
    mask = pd.read_csv(PROCESSED / f"{prefix}_mask.csv", index_col=0).values.astype(np.float32)
    clin = pd.read_csv(PROCESSED / f"{prefix}_clin.csv", index_col=0)
    t = clin["OS_MONTHS"].values.astype(np.float32)
    e = clin["event"].values.astype(np.float32)
    tmb = np.zeros(len(t), dtype=np.float32)
    return mut, mask, t, e, tmb


def get_valid_cohorts():
    return [p.stem.replace("valid_", "").replace("_mut", "")
            for p in sorted(PROCESSED.glob("valid_*_mut.csv"))]


def predict_risks(model, mut, mask, tmb, device, batch_size=128,
                  edge_index_dict=None, pw_gene=None, gene_indices=None):
    model.eval()
    all_risks = []
    n = len(mut)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            mut_b = torch.tensor(mut[start:end], device=device)
            mask_b = torch.tensor(mask[start:end], device=device)
            tmb_b = torch.tensor(tmb[start:end], device=device)

            kwargs = {}
            if edge_index_dict is not None:
                kwargs["edge_index_dict"] = edge_index_dict
            if pw_gene is not None:
                kwargs["pathway_gene_edge"] = pw_gene
            if gene_indices is not None:
                kwargs["gene_indices"] = gene_indices

            out = model(mut_b, mask_b, tmb_b, **kwargs)
            all_risks.append(out["log_risk"].cpu().numpy())
    return np.concatenate(all_risks)


def train_model(model, train_loader, optimizer, scheduler, device,
                mut_val, mask_val, tmb_val, time_val, event_val,
                epochs, patience, grad_clip,
                edge_index_dict=None, pw_gene=None, gene_indices=None):
    """Generic training loop for any model variant."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from losses import cox_loss, c_index as ci_fn

    best_ci = 0.0
    patience_count = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            mut_b, mask_b, tmb_b, time_b, event_b = [x.to(device) for x in batch]
            if len(mut_b) < 2:
                continue

            kwargs = {}
            if edge_index_dict is not None:
                kwargs["edge_index_dict"] = edge_index_dict
            if pw_gene is not None:
                kwargs["pathway_gene_edge"] = pw_gene
            if gene_indices is not None:
                kwargs["gene_indices"] = gene_indices

            out = model(mut_b, mask_b, tmb_b, **kwargs)
            loss = cox_loss(out["log_risk"], time_b, event_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        risks_val = predict_risks(
            model, mut_val, mask_val, tmb_val, device,
            edge_index_dict=edge_index_dict, pw_gene=pw_gene,
            gene_indices=gene_indices)
        val_ci = ci_fn(
            torch.tensor(risks_val), torch.tensor(time_val),
            torch.tensor(event_val))

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(-val_ci)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Loss {avg_loss:.4f} | Val CI {val_ci:.4f}")

        if val_ci > best_ci:
            best_ci = val_ci
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return best_ci


def run_ablation(args):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from losses import compute_all_metrics

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    print(f"Device: {device}")

    # Load data
    mut, mask, time_arr, event_arr = load_data("train")
    tmb = load_tmb()
    n_samples, n_candidate = mut.shape

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_samples)
    n_train = int(n_samples * args.train_ratio)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    cand_df = pd.read_csv(
        ROOT / "source" / "input_data" / "train" / "gene_candidate.csv")
    candidate_genes = cand_df.iloc[:, 0].dropna().str.strip().tolist()

    train_dataset = TensorDataset(
        torch.tensor(mut[train_idx]), torch.tensor(mask[train_idx]),
        torch.tensor(tmb[train_idx]), torch.tensor(time_arr[train_idx]),
        torch.tensor(event_arr[train_idx]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)

    # Build KG-informed model components
    from train_single_kg import build_model_and_graph, build_gene_indices
    full_model, data, node_maps, gene_map, pw_gene, node_counts = \
        build_model_and_graph(
            args.kg, args.hidden_dim, args.num_heads, args.num_layers,
            args.dropout, n_candidate, device)
    gene_indices = build_gene_indices(candidate_genes, gene_map, device)

    # Define ablation variants
    from ablation_models import BaselineMLP, MutaPathSurvNoMask, MutaPathSurvNoPathway

    variants = {
        "full_model": {
            "model": full_model,
            "needs_kg": True,
        },
        "baseline_mlp": {
            "model": BaselineMLP(n_candidate, args.hidden_dim, args.dropout).to(device),
            "needs_kg": False,
        },
        "no_mask": {
            "model": MutaPathSurvNoMask(full_model).to(device),
            "needs_kg": True,
        },
        "no_pathway": {
            "model": MutaPathSurvNoPathway(full_model).to(device),
            "needs_kg": True,
        },
    }

    ablation_dir = EXP_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    results = []
    cohorts = get_valid_cohorts()

    for name, cfg in variants.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {name}")
        print(f"{'=' * 60}")

        model = cfg["model"]
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # For full_model and no_pathway, rebuild fresh (shared params issue)
        if name == "no_mask":
            from model import MutaPathSurv
            base = MutaPathSurv(
                node_types=list(node_maps.keys()),
                edge_types=data.edge_types,
                hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                num_layers=args.num_layers, n_genes=full_model.n_genes,
                n_pathways=full_model.n_pathways, n_candidate=n_candidate,
                dropout=args.dropout, node_counts=node_counts,
            ).to(device)
            model = MutaPathSurvNoMask(base).to(device)
            n_params = sum(p.numel() for p in model.parameters())
        elif name == "no_pathway":
            from model import MutaPathSurv
            base = MutaPathSurv(
                node_types=list(node_maps.keys()),
                edge_types=data.edge_types,
                hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                num_layers=args.num_layers, n_genes=full_model.n_genes,
                n_pathways=full_model.n_pathways, n_candidate=n_candidate,
                dropout=args.dropout, node_counts=node_counts,
            ).to(device)
            model = MutaPathSurvNoPathway(base).to(device)
            n_params = sum(p.numel() for p in model.parameters())
        elif name == "full_model":
            from model import MutaPathSurv
            model = MutaPathSurv(
                node_types=list(node_maps.keys()),
                edge_types=data.edge_types,
                hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                num_layers=args.num_layers, n_genes=full_model.n_genes,
                n_pathways=full_model.n_pathways, n_candidate=n_candidate,
                dropout=args.dropout, node_counts=node_counts,
            ).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

        kg_args = {}
        if cfg["needs_kg"]:
            kg_args = {
                "edge_index_dict": data.edge_index_dict,
                "pw_gene": pw_gene,
                "gene_indices": gene_indices,
            }

        best_ci = train_model(
            model, train_loader, optimizer, scheduler, device,
            mut[val_idx], mask[val_idx], tmb[val_idx],
            time_arr[val_idx], event_arr[val_idx],
            args.epochs, args.patience, args.grad_clip,
            **kg_args)

        print(f"  Best Val CI: {best_ci:.4f}")

        # Save model
        torch.save(model.state_dict(), ablation_dir / f"{name}_model.pt")

        # External validation
        variant_results = {"variant": name, "val_ci": best_ci, "n_params": n_params}
        ext_metrics = []

        for cohort in cohorts:
            vdata = load_valid_data(cohort)
            if vdata is None or len(vdata[0]) < 5:
                continue
            v_mut, v_mask, v_time, v_event, v_tmb = vdata
            risks = predict_risks(model, v_mut, v_mask, v_tmb, device, **kg_args)
            m = compute_all_metrics(risks, v_time, v_event)
            m["cohort"] = cohort
            ext_metrics.append(m)

        if ext_metrics:
            avg_ci = np.mean([m["c_index"] for m in ext_metrics])
            avg_hr = np.mean([m["hr"] for m in ext_metrics
                              if not np.isnan(m["hr"])])
            n_sig = sum(1 for m in ext_metrics if m["p_value"] < 0.05)
            variant_results["ext_avg_ci"] = avg_ci
            variant_results["ext_avg_hr"] = avg_hr
            variant_results["n_sig"] = n_sig
            variant_results["n_cohorts"] = len(ext_metrics)

            print(f"  External: avg CI={avg_ci:.4f}, avg HR={avg_hr:.2f}, "
                  f"sig={n_sig}/{len(ext_metrics)}")

        results.append(variant_results)
        pd.DataFrame(ext_metrics).to_csv(
            ablation_dir / f"{name}_cohorts.csv", index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(ablation_dir / "ablation_summary.csv", index=False)
    print(f"\nSaved to {ablation_dir}/")

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Ablation experiments")
    p.add_argument("--kg", type=str, default="monarch")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
