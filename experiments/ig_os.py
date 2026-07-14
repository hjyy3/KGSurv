"""Integrated Gradients attribution (independent of attention) for OS winners.

For each winner: reload model.pt (hash gate), pool 11 EVAL cohorts, compute IG
of log_risk w.r.t. (a) fmb input -> term/group attribution, (b) mut input ->
gene attribution. Baseline = zeros, n_steps=32. Signed IG gives direction
(positive => pushes log_risk UP => higher OS risk).

Compares IG gene ranking against the attention-based ranking in
interp_os_deep.json (rank overlap of top-15). Writes results/interp_os_deep/ig_os.json.
"""
from __future__ import annotations
import argparse
import faulthandler
import json
import sys
from pathlib import Path

import numpy as np
import torch

faulthandler.enable()

EXP_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(EXP_ROOT / "code"))

from interpret_winners import load_kg_data, WINNERS  # noqa: E402
from kfold_cv import HIDDEN, DROPOUT  # noqa: E402
from models_interp import create_model  # noqa: E402
from seeding import state_dict_hash  # noqa: E402
from multi_node_extended import EVAL_COHORTS  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = EXP_ROOT / "runs"
OUT = EXP_ROOT / "results" / "interp_os_deep"
ATTN = OUT / "interp_os_deep.json"
N_STEPS = 32

HEADLINE = ("path_attn_hetionet_mg2_ppi+disease+drug+anatomy+regulatory_seed56_fold2",
            "hetionet_mg2", "hetionet", ["ppi", "disease", "drug", "anatomy", "regulatory"],
            "mg2", 56, 2, -0.3238859176635742, "roc24m(c3)")
ALL_WINNERS = [HEADLINE] + list(WINNERS)


def ig(model, mut, mask, fmb, target, n_steps=N_STEPS):
    mut = mut.to(dev); mask = mask.to(dev); fmb = fmb.to(dev)
    x = fmb if target == "fmb" else mut
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for k in range(1, n_steps + 1):
        a = float(k) / n_steps
        xi = (base + a * (x - base)).clone().requires_grad_(True)
        if target == "fmb":
            out = model(mut, mask, xi)
        else:
            out = model(xi, mask, fmb)
        g, = torch.autograd.grad(out["log_risk"].sum(), xi)
        total = total + g.detach()
    return ((x - base) * total / n_steps).cpu().numpy()


def _js(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    sel = None if args.only == "all" else set(args.only.split(","))
    attn = json.loads(ATTN.read_text(encoding="utf-8")) if ATTN.exists() else {}

    for (run_id, short, kg, node_types, fmb_variant, seed, fold,
         deploy_thr, deploy_rule) in ALL_WINNERS:
        if sel is not None and short not in sel:
            continue
        print(f"\n==== IG {short} ====", flush=True)
        run_dir = RUNS_DIR / run_id
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        eff = f"{kg}_{fmb_variant}" if fmb_variant else kg
        aug, info, genes = load_kg_data(kg, node_types, eff)
        model = create_model("path_attn", info, hidden_dim=HIDDEN, dropout=DROPOUT).to(dev)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev))
        model.eval()
        if state_dict_hash(model.state_dict()) != cfg["state_dict_hash"]:
            print("  HASH FAIL -> skip", flush=True)
            continue

        group_names = list(info.group_names)
        flat_terms, flat_groups = [], []
        for gi_, gn in enumerate(group_names):
            for tn in info.term_names[gi_]:
                flat_terms.append(str(tn)); flat_groups.append(gn)

        present = [c for c in EVAL_COHORTS if c in aug]
        T = info.n_total_terms
        Ng = len(genes)
        term_sum = np.zeros(T); gene_sum = np.zeros(Ng); n_tot = 0
        for c in present:
            term_sum += ig(model, aug[c]["mut"], aug[c]["mask"], aug[c]["fmb"], "fmb").sum(0)
            gene_sum += ig(model, aug[c]["mut"], aug[c]["mask"], aug[c]["fmb"], "mut").sum(0)
            n_tot += aug[c]["mut"].shape[0]
        term_ig = term_sum / n_tot
        gene_ig = gene_sum / n_tot

        # group-level signed IG (sum of term IG within group)
        grp_ig = {}
        for gi_, gn in enumerate(group_names):
            s, e = info.fmb_slices[gi_]
            grp_ig[gn] = float(term_ig[s:e].sum())

        def top(vals, names, k=20):
            order = np.argsort(-np.abs(vals))[:k]
            return [{"name": names[i], "ig": float(vals[i])} for i in order]

        top_genes = top(gene_ig, genes, 30)
        top_terms = top(term_ig, flat_terms, 20)

        # overlap vs attention top genes (by mean importance)
        overlap = None
        if short in attn and "top_genes_by_mean" in attn[short]:
            attn_top = [d["name"] for d in attn[short]["top_genes_by_mean"][:15]]
            ig_top = [d["name"] for d in top_genes[:15]]
            inter = set(attn_top) & set(ig_top)
            overlap = {"attn_top15": attn_top, "ig_top15": ig_top,
                       "n_overlap": len(inter), "overlap": sorted(inter),
                       "tp53_ig_rank": (ig_top.index("TP53") + 1) if "TP53" in ig_top else None}

        rec = {"run_id": run_id, "n_pooled": n_tot, "group_ig_signed": grp_ig,
               "top_genes_ig": top_genes, "top_terms_ig": top_terms,
               "gene_overlap_vs_attention": overlap}
        (OUT / f"_ig_{short}.json").write_text(json.dumps(rec, indent=2, default=_js), encoding="utf-8")
        print(f"  group_ig: " + ", ".join(f"{k[:10]}:{v:+.3f}" for k, v in
              sorted(grp_ig.items(), key=lambda kv: -abs(kv[1]))), flush=True)
        print(f"  top5 gene IG: " + ", ".join(f"{d['name']}:{d['ig']:+.3f}"
              for d in top_genes[:5]), flush=True)
        if overlap:
            print(f"  gene top15 overlap IG vs attn = {overlap['n_overlap']}/15", flush=True)

    merged = {}
    for p in sorted(OUT.glob("_ig_*.json")):
        merged[p.stem[4:]] = json.loads(p.read_text(encoding="utf-8"))
    (OUT / "ig_os.json").write_text(json.dumps(merged, indent=2, default=_js), encoding="utf-8")
    print(f"\nSaved -> {OUT}/ig_os.json ({len(merged)} winners)", flush=True)


if __name__ == "__main__":
    main()
