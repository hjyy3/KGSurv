"""Interpretability demo: PathAttnSurv × Hetionet.

Generates publication-quality figures showing the 3-level attribution chain:
  Gene → Pathway/GO Term → Group (edge type)

Also shows patient-level risk stratification by top gene mutations.

Usage:
    python src/plot_interpretability_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import VALID_COHORTS, build_kg_group_info, load_all_data
from losses import compute_all_metrics
from models_interp import create_model
from preprocess import N_CLIN

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"
FIG_DIR = ROOT / "output" / "figures" / "interpretability"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "path_attn"
KG_NAME = "hetionet"
INTERP_DIR = EXP_DIR / f"interp_{MODEL_NAME}_{KG_NAME}" / "interpretability"


def load_interp_data():
    gene_df = pd.read_csv(INTERP_DIR / "gene_importance.csv")
    term_df = pd.read_csv(INTERP_DIR / "term_importance.csv")
    group_df = pd.read_csv(INTERP_DIR / "group_importance.csv")
    return gene_df, term_df, group_df


def fig1_gene_importance(gene_df: pd.DataFrame):
    """Top 30 gene importance bar chart."""
    top = gene_df.head(30).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 9))
    colors = ["#D32F2F" if g in ("TP53", "KRAS", "BRAF", "PIK3CA", "PTEN", "EGFR")
              else "#1976D2" for g in top["gene"]]
    bars = ax.barh(range(len(top)), top["importance"], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["gene"], fontsize=9)
    ax.set_xlabel("Mean Importance Score (training cohort)", fontsize=11)
    ax.set_title("PathAttnSurv × Hetionet — Top 30 Gene Importance\n"
                 "(red = known oncogene/TSG)", fontsize=12, fontweight="bold")
    ax.axvline(x=0, color="grey", linewidth=0.5)

    # Annotate TP53
    tp53_val = gene_df[gene_df["gene"] == "TP53"]["importance"].values[0]
    ax.annotate(f"TP53: {tp53_val:.1f}", xy=(tp53_val, len(top) - 1),
                xytext=(tp53_val * 0.6, len(top) - 3),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_gene_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_gene_importance.png")


def fig2_term_importance(term_df: pd.DataFrame):
    """Top 25 pathway/GO term importance."""
    top = term_df.head(25).iloc[::-1]
    # Truncate long names
    labels = [t[:60] + "..." if len(t) > 60 else t for t in top["term"]]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(top)), top["importance"], color="#7B1FA2", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Term Importance", fontsize=11)
    ax.set_title("PathAttnSurv × Hetionet — Top 25 Pathway/GO Terms\n"
                 "(cross-pathway attention × FMB activation)", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_term_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_term_importance.png")


def fig3_gene_term_sankey(gene_df: pd.DataFrame, term_df: pd.DataFrame):
    """Gene → Term connection heatmap (top 10 genes × top 15 terms).

    Uses KG subgraph connectivity to show which genes drive which pathways.
    """
    subkg = pd.read_csv(ROOT / "output" / "subkg" / f"subkg_{KG_NAME}.csv",
                        low_memory=False)

    top_genes = gene_df.head(10)["gene"].tolist()
    top_terms = term_df.head(15)["term"].tolist()

    # Build connectivity matrix
    conn = np.zeros((len(top_genes), len(top_terms)))
    for i, g in enumerate(top_genes):
        for j, t in enumerate(top_terms):
            # Check if gene-term edge exists
            mask = ((subkg["x_name"] == g) & (subkg["y_name"].str.strip() == t)) | \
                   ((subkg["y_name"] == g) & (subkg["x_name"].str.strip() == t))
            if mask.any():
                conn[i, j] = 1.0

    # Weight by gene importance × term importance
    gene_imp = gene_df.set_index("gene").loc[top_genes, "importance"].values
    term_imp = term_df.set_index("term").loc[top_terms, "importance"].values
    weighted = conn * gene_imp[:, None] * term_imp[None, :]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(weighted, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top_terms)))
    ax.set_xticklabels([t[:45] + "..." if len(t) > 45 else t for t in top_terms],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=10)

    # Mark non-zero cells
    for i in range(len(top_genes)):
        for j in range(len(top_terms)):
            if conn[i, j] > 0:
                ax.text(j, i, "●", ha="center", va="center", fontsize=8,
                        color="white" if weighted[i, j] > weighted.max() * 0.6
                        else "black")

    plt.colorbar(im, ax=ax, label="Gene Imp × Term Imp", shrink=0.8)
    ax.set_title("Gene → Pathway Attribution Map\n"
                 "(● = KG edge exists, color = importance product)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Pathway / GO Term")
    ax.set_ylabel("Gene")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_gene_term_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_gene_term_map.png")


def fig4_patient_level_risk(gene_df: pd.DataFrame):
    """Show per-patient gene importance for high-risk vs low-risk groups."""
    # Load model and compute patient-level attributions
    kg_info = build_kg_group_info(KG_NAME)
    model = create_model(MODEL_NAME, kg_info, n_clin=N_CLIN)
    model_path = EXP_DIR / f"interp_{MODEL_NAME}_{KG_NAME}" / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu",
                                     weights_only=True))
    model.eval()

    all_data = load_all_data(KG_NAME)
    train = all_data["train"]

    with torch.no_grad():
        out = model(train["mut"], train["mask"], train["fmb"],
                    clin_cov=train["clin_cov"])

    risk = out["log_risk"].numpy()
    gene_imp = out["gene_importance"].numpy()  # [n_patients, 463]

    # Split into high/low risk
    median_risk = np.median(risk)
    high_mask = risk >= median_risk
    low_mask = ~high_mask

    # Top 15 genes
    top15 = gene_df.head(15)["gene"].tolist()
    from data_interp import _load_gene_list
    all_genes = _load_gene_list()
    gene_idx = [all_genes.index(g) for g in top15]

    high_imp = gene_imp[high_mask][:, gene_idx].mean(axis=0)
    low_imp = gene_imp[low_mask][:, gene_idx].mean(axis=0)

    x = np.arange(len(top15))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, high_imp, width, label="High-risk group",
           color="#D32F2F", alpha=0.85)
    ax.bar(x + width / 2, low_imp, width, label="Low-risk group",
           color="#1976D2", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top15, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Gene Importance", fontsize=11)
    ax.set_title("Patient-Level Gene Attribution: High-Risk vs Low-Risk\n"
                 "(PathAttnSurv × Hetionet, training cohort)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_patient_risk_genes.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_patient_risk_genes.png")


def fig5_tp53_survival():
    """TP53-mutated vs wild-type survival across validation cohorts."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    kg_info = build_kg_group_info(KG_NAME)
    model = create_model(MODEL_NAME, kg_info, n_clin=N_CLIN)
    model_path = EXP_DIR / f"interp_{MODEL_NAME}_{KG_NAME}" / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu",
                                     weights_only=True))
    model.eval()

    all_data = load_all_data(KG_NAME)

    # Pool all validation
    from data_interp import _load_gene_list
    all_genes = _load_gene_list()
    tp53_idx = all_genes.index("TP53")

    all_risk, all_time, all_event, all_tp53 = [], [], [], []
    for cohort in VALID_COHORTS:
        if cohort not in all_data["valid"]:
            continue
        d = all_data["valid"][cohort]
        with torch.no_grad():
            out = model(d["mut"], d["mask"], d["fmb"], clin_cov=d["clin_cov"])
        r = out["log_risk"].numpy()
        tp53_mut = (d["mut"][:, tp53_idx] * d["mask"][:, tp53_idx]).numpy()
        all_risk.append(r)
        all_time.append(d["time"].numpy())
        all_event.append(d["event"].numpy())
        all_tp53.append(tp53_mut)

    risk = np.concatenate(all_risk)
    time = np.concatenate(all_time).astype(np.float64)
    event = np.concatenate(all_event).astype(np.float64)
    tp53 = np.concatenate(all_tp53)

    # 2x2: TP53 status × model risk
    median_risk = np.median(risk)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: TP53 mut vs WT survival
    ax = axes[0]
    tp53_mut_mask = tp53 > 0.5
    tp53_wt_mask = ~tp53_mut_mask
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    valid_mut = np.isfinite(time) & (time > 0) & tp53_mut_mask
    valid_wt = np.isfinite(time) & (time > 0) & tp53_wt_mask
    if valid_mut.sum() > 5 and valid_wt.sum() > 5:
        kmf1.fit(time[valid_mut], event[valid_mut], label=f"TP53-mut (n={valid_mut.sum()})")
        kmf2.fit(time[valid_wt], event[valid_wt], label=f"TP53-WT (n={valid_wt.sum()})")
        kmf1.plot_survival_function(ax=ax, color="#D32F2F", ci_show=False)
        kmf2.plot_survival_function(ax=ax, color="#1976D2", ci_show=False)
        lr = logrank_test(time[valid_mut], time[valid_wt],
                          event[valid_mut], event[valid_wt])
        ax.text(0.98, 0.02, f"log-rank p={lr.p_value:.2e}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="wheat", alpha=0.8))
    ax.set_title("(A) TP53 Mutation Status", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.legend(fontsize=9)

    # Panel B: Model-predicted risk
    ax = axes[1]
    high = risk >= median_risk
    low = ~high
    valid_h = np.isfinite(time) & (time > 0) & high
    valid_l = np.isfinite(time) & (time > 0) & low
    kmf3 = KaplanMeierFitter()
    kmf4 = KaplanMeierFitter()
    if valid_h.sum() > 5 and valid_l.sum() > 5:
        kmf3.fit(time[valid_h], event[valid_h], label=f"High risk (n={valid_h.sum()})")
        kmf4.fit(time[valid_l], event[valid_l], label=f"Low risk (n={valid_l.sum()})")
        kmf3.plot_survival_function(ax=ax, color="#D32F2F", ci_show=False)
        kmf4.plot_survival_function(ax=ax, color="#1976D2", ci_show=False)
        m = compute_all_metrics(risk, time, event)
        ax.text(0.98, 0.02, f"CI={m['c_index']:.3f}  HR={m['hr']:.2f}\np={m['p_value']:.2e}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="wheat", alpha=0.8))
    ax.set_title("(B) Model-Predicted Risk", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (months)")
    ax.legend(fontsize=9)

    # Panel C: 4-group stratification (TP53 × risk)
    ax = axes[2]
    groups = {
        "TP53-mut + High": tp53_mut_mask & high,
        "TP53-mut + Low": tp53_mut_mask & low,
        "TP53-WT + High": tp53_wt_mask & high,
        "TP53-WT + Low": tp53_wt_mask & low,
    }
    colors_4 = ["#B71C1C", "#EF9A9A", "#0D47A1", "#90CAF9"]
    for (label, gmask), color in zip(groups.items(), colors_4):
        valid_g = np.isfinite(time) & (time > 0) & gmask
        if valid_g.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(time[valid_g], event[valid_g], label=f"{label} (n={valid_g.sum()})")
        kmf.plot_survival_function(ax=ax, color=color, ci_show=False, linewidth=1.5)
    ax.set_title("(C) TP53 × Model Risk (4-group)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (months)")
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("TP53 Mutation vs Model Risk — All Validation Cohorts Pooled",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_tp53_vs_model_risk.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_tp53_vs_model_risk.png")


def main():
    print("Loading interpretability data...")
    gene_df, term_df, group_df = load_interp_data()
    print(f"  Genes: {len(gene_df)}, Terms: {len(term_df)}, Groups: {len(group_df)}")

    print("\n--- Fig 1: Gene Importance ---")
    fig1_gene_importance(gene_df)

    print("\n--- Fig 2: Term Importance ---")
    fig2_term_importance(term_df)

    print("\n--- Fig 3: Gene → Term Map ---")
    fig3_gene_term_sankey(gene_df, term_df)

    print("\n--- Fig 4: Patient-Level Risk Attribution ---")
    fig4_patient_level_risk(gene_df)

    print("\n--- Fig 5: TP53 vs Model Risk ---")
    fig5_tp53_survival()

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
