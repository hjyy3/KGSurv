"""
Single-KG experiment runner — train MutaPathSurv with one KG prior at a time.

Three-tier validation:
  1. Per-cohort: individual external cohort evaluation
  2. Same-cancer merge: merge cohorts by cancer type
  3. All-cohort merge: merge all external cohorts

Primary metrics: HR + log-rank p-value (C-index as reference)

Usage:
    python src/train_single_kg.py --kg primekg
    python src/train_single_kg.py --kg hetionet --epochs 80
    python src/train_single_kg.py --all   # Run all available KGs
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

# Cancer type grouping for tier-2 validation
CANCER_TYPE_MAP = {
    "Hugo": "melanoma", "Riaz": "melanoma",
    "Liu": "melanoma", "Whijae": "melanoma",
    "Braun": "rcc", "Miao": "rcc",
    "CM214_JV101": "melanoma",
    "PUSH": "uro", "Snyder_UC": "uro",
    "Pleasance": "pan",
}


# -- Data Loading -------------------------------------------------------------

def load_data(split="train"):
    mut = pd.read_csv(PROCESSED / f"{split}_mut.csv", index_col=0).values.astype(np.float32)
    mask = pd.read_csv(PROCESSED / f"{split}_mask.csv", index_col=0).values.astype(np.float32)
    clin = pd.read_csv(PROCESSED / f"{split}_clin.csv", index_col=0)
    t = clin["OS_MONTHS"].values.astype(np.float32)
    e = clin["event"].values.astype(np.float32)
    return mut, mask, t, e


def load_tmb(split="train"):
    clin_proc = pd.read_csv(PROCESSED / f"{split}_clin.csv", index_col=0)
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
    cohorts = []
    for p in sorted(PROCESSED.glob("valid_*_mut.csv")):
        cohort = p.stem.replace("valid_", "").replace("_mut", "")
        cohorts.append(cohort)
    return cohorts


# -- Model Building -----------------------------------------------------------

def build_model_and_graph(kg_name, hidden_dim, num_heads, num_layers,
                          dropout, n_candidate, device):
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    if kg_name == "primekg":
        from graph_builder import build_hetero_graph
        data, node_maps, gene_map, node_counts = build_hetero_graph()
    else:
        from graph_builder_multi import build_hetero_graph, get_pathway_gene_edges
        subkg_path = SUBKG_DIR / f"subkg_{kg_name}.csv"
        data, node_maps, gene_map, node_counts = build_hetero_graph(subkg_path)

    data = data.to(device)
    from model import MutaPathSurv

    # Find pathway-gene edges
    pw_gene = None
    for et in data.edge_types:
        src, rel, dst = et
        if "pathway" in src.lower() and ("gene" in dst.lower() or "protein" in dst.lower()):
            pw_gene = data[et].edge_index.to(device)
            break
        if ("gene" in src.lower() or "protein" in src.lower()) and "pathway" in dst.lower():
            ei = data[et].edge_index
            pw_gene = torch.stack([ei[1], ei[0]]).to(device)
            break

    n_pathways = 1
    for nt in node_maps:
        if "pathway" in nt.lower():
            n_pathways = len(node_maps[nt])
            break

    if pw_gene is None:
        pw_gene = torch.zeros(2, 0, dtype=torch.long, device=device)

    # Find gene node type
    gene_type = None
    for nt in node_maps:
        if "gene" in nt.lower() or "protein" in nt.lower():
            gene_type = nt
            break
    n_genes = len(node_maps.get(gene_type, {})) if gene_type else 1

    model = MutaPathSurv(
        node_types=list(node_maps.keys()),
        edge_types=data.edge_types,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        n_genes=n_genes,
        n_pathways=n_pathways,
        n_candidate=n_candidate,
        dropout=dropout,
        node_counts=node_counts,
    ).to(device)

    return model, data, node_maps, gene_map, pw_gene, node_counts


def build_gene_indices(candidate_genes, gene_map, device):
    indices = []
    for g in candidate_genes:
        indices.append(gene_map.get(g, 0))
    return torch.tensor(indices, dtype=torch.long, device=device)


# -- Evaluation ---------------------------------------------------------------

def predict_risks(model, mut, mask, tmb, edge_index_dict, pw_gene,
                  gene_indices, device, batch_size=128):
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
            all_risks.append(out["log_risk"].cpu().numpy())
    return np.concatenate(all_risks)


def evaluate_cohort(model, mut, mask, tmb, time_arr, event_arr,
                    edge_index_dict, pw_gene, gene_indices, device):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from losses import compute_all_metrics

    risks = predict_risks(model, mut, mask, tmb, edge_index_dict,
                          pw_gene, gene_indices, device)
    return compute_all_metrics(risks, time_arr, event_arr)


# -- Training -----------------------------------------------------------------

def train_single_kg(kg_name, args):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from losses import cox_loss, c_index

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    print(f"\n{'#' * 70}")
    print(f"# KG: {kg_name}")
    print(f"# Device: {device}")
    print(f"{'#' * 70}")

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

    cand_df = pd.read_csv(
        ROOT / "source" / "input_data" / "train" / "gene_candidate.csv")
    candidate_genes = cand_df.iloc[:, 0].dropna().str.strip().tolist()

    # Build model
    try:
        model, data, node_maps, gene_map, pw_gene, node_counts = build_model_and_graph(
            kg_name, args.hidden_dim, args.num_heads, args.num_layers,
            args.dropout, len(candidate_genes), device,
        )
    except Exception as e:
        print(f"[ERROR] Failed to build model for {kg_name}: {e}")
        return None

    gene_indices = build_gene_indices(candidate_genes, gene_map, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"KG gene coverage: {sum(1 for g in candidate_genes if g in gene_map)}/{len(candidate_genes)}")

    # DataLoader
    train_dataset = TensorDataset(
        torch.tensor(mut[train_idx]), torch.tensor(mask[train_idx]),
        torch.tensor(tmb[train_idx]), torch.tensor(time_arr[train_idx]),
        torch.tensor(event_arr[train_idx]),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=False,
                              pin_memory=(device.type == "cuda"))

    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    exp_dir = EXP_DIR / kg_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_cindex = 0.0
    patience_count = 0
    training_log = []

    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'Val CI':>8} | {'LR':>10} | {'Time':>6}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time_module.time()

        for mut_b, mask_b, tmb_b, time_b, event_b in train_loader:
            if len(mut_b) < 2:
                continue
            mut_b, mask_b = mut_b.to(device), mask_b.to(device)
            tmb_b, time_b, event_b = (tmb_b.to(device), time_b.to(device),
                                       event_b.to(device))

            out = model(mut_b, mask_b, tmb_b, data.edge_index_dict,
                        pw_gene, gene_indices)
            loss = cox_loss(out["log_risk"], time_b, event_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Quick validation (C-index only for speed)
        risks_val = predict_risks(
            model, mut[val_idx], mask[val_idx], tmb[val_idx],
            data.edge_index_dict, pw_gene, gene_indices, device,
        )
        from losses import c_index as ci_fn
        val_ci = ci_fn(
            torch.tensor(risks_val), torch.tensor(time_arr[val_idx]),
            torch.tensor(event_arr[val_idx]),
        )

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time_module.time() - t_start
        scheduler.step(-val_ci)

        training_log.append({"epoch": epoch, "loss": avg_loss,
                             "val_ci": val_ci, "lr": lr_now})

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:5d} | {avg_loss:8.4f} | {val_ci:8.4f} | "
                  f"{lr_now:10.6f} | {elapsed:5.1f}s")

        if val_ci > best_cindex:
            best_cindex = val_ci
            patience_count = 0
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest validation C-index: {best_cindex:.4f}")

    # Save training log
    pd.DataFrame(training_log).to_csv(exp_dir / "training_log.csv", index=False)

    # Load best model for evaluation
    model.load_state_dict(torch.load(exp_dir / "best_model.pt", weights_only=True))

    # ── Three-tier external validation ───────────────────────────────────
    print(f"\n{'=' * 60}")
    print("External Validation (3-tier)")
    print(f"{'=' * 60}")

    cohorts = get_valid_cohorts()
    per_cohort_results = []
    all_risks, all_times, all_events = [], [], []

    # Tier 1: Per-cohort
    print(f"\n--- Tier 1: Per-cohort ---")
    print(f"{'Cohort':<15} | {'N':>5} | {'CI':>6} | {'HR':>6} | "
          f"{'p-value':>10} | {'AUC12':>6} | {'AUC24':>6}")
    print("-" * 70)

    for cohort in cohorts:
        vdata = load_valid_data(cohort)
        if vdata is None or len(vdata[0]) < 5:
            continue
        v_mut, v_mask, v_time, v_event, v_tmb = vdata

        metrics = evaluate_cohort(
            model, v_mut, v_mask, v_tmb, v_time, v_event,
            data.edge_index_dict, pw_gene, gene_indices, device,
        )
        metrics["cohort"] = cohort
        metrics["n"] = len(v_mut)
        metrics["cancer_type"] = CANCER_TYPE_MAP.get(cohort, "other")
        per_cohort_results.append(metrics)

        # Accumulate for tier-3
        risks = predict_risks(model, v_mut, v_mask, v_tmb,
                              data.edge_index_dict, pw_gene, gene_indices, device)
        all_risks.append(risks)
        all_times.append(v_time)
        all_events.append(v_event)

        p_str = f"{metrics['p_value']:.4f}" if not np.isnan(metrics["p_value"]) else "N/A"
        a12 = f"{metrics.get('auc_12m', float('nan')):.3f}"
        a24 = f"{metrics.get('auc_24m', float('nan')):.3f}"
        sig = "*" if metrics["p_value"] < 0.05 else ""
        print(f"{cohort:<15} | {len(v_mut):5d} | {metrics['c_index']:.4f} | "
              f"{metrics['hr']:6.2f} | {p_str:>10}{sig} | {a12:>6} | {a24:>6}")

    # Tier 2: Same-cancer merge
    print(f"\n--- Tier 2: Same-cancer-type merged ---")
    from losses import compute_all_metrics
    cancer_groups = {}
    for r in per_cohort_results:
        ct = r["cancer_type"]
        if ct not in cancer_groups:
            cancer_groups[ct] = {"cohorts": [], "idx": []}
        cancer_groups[ct]["cohorts"].append(r["cohort"])

    # Re-collect per cancer type
    merged_results = []
    for ct, info in cancer_groups.items():
        ct_risks, ct_times, ct_events = [], [], []
        for cohort_name in info["cohorts"]:
            vdata = load_valid_data(cohort_name)
            if vdata is None:
                continue
            v_mut, v_mask, v_time, v_event, v_tmb = vdata
            risks = predict_risks(model, v_mut, v_mask, v_tmb,
                                  data.edge_index_dict, pw_gene, gene_indices, device)
            ct_risks.append(risks)
            ct_times.append(v_time)
            ct_events.append(v_event)

        if not ct_risks:
            continue
        ct_r = np.concatenate(ct_risks)
        ct_t = np.concatenate(ct_times)
        ct_e = np.concatenate(ct_events)

        m = compute_all_metrics(ct_r, ct_t, ct_e)
        m["cancer_type"] = ct
        m["n"] = len(ct_r)
        m["cohorts"] = ",".join(info["cohorts"])
        merged_results.append(m)

        p_str = f"{m['p_value']:.4f}" if not np.isnan(m["p_value"]) else "N/A"
        sig = "*" if m["p_value"] < 0.05 else ""
        print(f"  {ct:<12} (n={m['n']:>4}) | CI={m['c_index']:.4f} | "
              f"HR={m['hr']:.2f} | p={p_str}{sig}")

    # Tier 3: All-cohort merge
    print(f"\n--- Tier 3: All-cohort merged ---")
    if all_risks:
        all_r = np.concatenate(all_risks)
        all_t = np.concatenate(all_times)
        all_e = np.concatenate(all_events)
        all_m = compute_all_metrics(all_r, all_t, all_e)

        p_str = f"{all_m['p_value']:.4f}" if not np.isnan(all_m["p_value"]) else "N/A"
        sig = "**" if all_m["p_value"] < 0.01 else ("*" if all_m["p_value"] < 0.05 else "")
        print(f"  ALL (n={len(all_r)}) | CI={all_m['c_index']:.4f} | "
              f"HR={all_m['hr']:.2f} | p={p_str}{sig}")
        print(f"  AUC: 12m={all_m.get('auc_12m', float('nan')):.3f}, "
              f"24m={all_m.get('auc_24m', float('nan')):.3f}, "
              f"36m={all_m.get('auc_36m', float('nan')):.3f}")

    # ── Save all results ─────────────────────────────────────────────────
    pd.DataFrame(per_cohort_results).to_csv(
        exp_dir / "tier1_per_cohort.csv", index=False)
    pd.DataFrame(merged_results).to_csv(
        exp_dir / "tier2_cancer_type.csv", index=False)
    if all_risks:
        pd.DataFrame([all_m]).to_csv(exp_dir / "tier3_all_merged.csv", index=False)

    # Summary dict
    n_sig = sum(1 for r in per_cohort_results if r["p_value"] < 0.05)
    summary = {
        "kg": kg_name,
        "best_val_ci": best_cindex,
        "n_params": n_params,
        "n_cohorts_sig": n_sig,
        "n_cohorts_total": len(per_cohort_results),
        "avg_ext_ci": np.mean([r["c_index"] for r in per_cohort_results])
                      if per_cohort_results else 0,
        "all_merged_hr": all_m.get("hr", float("nan")) if all_risks else float("nan"),
        "all_merged_p": all_m.get("p_value", float("nan")) if all_risks else float("nan"),
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {exp_dir}/")
    print(f"Significant cohorts (p<0.05): {n_sig}/{len(per_cohort_results)}")

    return summary


# -- Main ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Single-KG experiment")
    p.add_argument("--kg", type=str, default="primekg")
    p.add_argument("--all", action="store_true", help="Run all available KGs")
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

    if args.all:
        kg_names = [p.stem.replace("subkg_", "")
                    for p in sorted(SUBKG_DIR.glob("subkg_*.csv"))]
        print(f"Running {len(kg_names)} KGs: {kg_names}")
        all_summaries = []
        for kg in kg_names:
            try:
                s = train_single_kg(kg, args)
                if s:
                    all_summaries.append(s)
            except Exception as e:
                print(f"\n[ERROR] {kg}: {e}")
                import traceback
                traceback.print_exc()

        if all_summaries:
            print(f"\n{'=' * 70}")
            print("KG COMPARISON SUMMARY")
            print(f"{'=' * 70}")
            df = pd.DataFrame(all_summaries)
            df = df.sort_values("all_merged_p")
            print(df.to_string(index=False))
            df.to_csv(EXP_DIR / "kg_comparison.csv", index=False)
            print(f"\nSaved to {EXP_DIR / 'kg_comparison.csv'}")
    else:
        train_single_kg(args.kg, args)
