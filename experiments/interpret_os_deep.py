"""Deep OS-line interpretability: headline 8/11 winner + 3 deployment winners.

NO retraining. For each winner:
  1. Rebuild (aug, kg_info) exactly as replay_engine; load model.pt.
  2. GATE: state_dict_hash == config.json hash  AND  forward log_risk allclose risks.npz.
  3. Global 3-level importance (group/term/gene), pooled EVAL, split by deploy thr.
  4. Sample-level: representative samples + per-sample top features + leave-out delta-risk.
  5. Perturbation: mask top-K group/term/gene -> rescore 11 cohorts (C2-C10),
     report delta fixed_max_n_sig / delta C-index / mean delta-risk vs RANDOM control.

Writes only to results/interp_os_deep/. Frozen artifacts untouched.
"""
from __future__ import annotations
import argparse
import faulthandler
import json
import sys
from pathlib import Path

faulthandler.enable()

import numpy as np
import pandas as pd
import torch

EXP_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = EXP_ROOT.parents[1]
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(EXP_ROOT / "code"))

from interpret_winners import load_kg_data, cancer_type_map, WINNERS  # noqa: E402
from kfold_cv import _select, kfold_indices, HIDDEN, DROPOUT  # noqa: E402
from models_interp import create_model  # noqa: E402
from seeding import state_dict_hash  # noqa: E402
from cutoffs import compute_all_train_cutoffs, logrank_split, CUTOFF_SCHEMA  # noqa: E402
from multi_node_extended import EVAL_COHORTS  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = EXP_ROOT / "runs"
OUT = EXP_ROOT / "results" / "interp_os_deep"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.RandomState(20260611)
N_RANDOM = 20

HEADLINE = ("path_attn_hetionet_mg2_ppi+disease+drug+anatomy+regulatory_seed56_fold2",
            "hetionet_mg2", "hetionet", ["ppi", "disease", "drug", "anatomy", "regulatory"],
            "mg2", 56, 2, -0.3238859176635742, "roc24m(c3)")
ALL_WINNERS = [HEADLINE] + list(WINNERS)


def _js(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return None


@torch.no_grad()
def fwd(model, mut, mask, fmb):
    out = model(mut.to(device), mask.to(device), fmb.to(device))
    return {k: out[k].cpu().numpy() for k in
            ("log_risk", "group_importance", "term_importance", "gene_importance")}


def rescore(tr_risk, tr_t, tr_e, coh):
    meta = compute_all_train_cutoffs(tr_risk, tr_t, tr_e)
    per = {}
    for c, (r, t, e) in coh.items():
        per[c] = {cid: logrank_split(r, t, e, meta[cid]["threshold"])
                  for cid, _ in CUTOFF_SCHEMA}
    n_sig = {cid: int(sum(1 for v in per.values() if v[cid]["sig"]))
             for cid, _ in CUTOFF_SCHEMA}
    fixed_max = max(n_sig.values())
    best_cid = max(n_sig, key=n_sig.get)
    return fixed_max, n_sig, best_cid, per


def pooled_cindex(coh):
    # higher log_risk -> shorter OS -> pass -risk so c>0.5 == good
    r = np.concatenate([v[0] for v in coh.values()])
    t = np.concatenate([v[1] for v in coh.values()])
    e = np.concatenate([v[2] for v in coh.values()])
    return float(concordance_index(t, -r, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="comma-sep winner shorts or 'all'")
    args = ap.parse_args()
    sel = None if args.only == "all" else set(args.only.split(","))
    summary = {}
    for (run_id, short, kg, node_types, fmb_variant, seed, fold,
         deploy_thr, deploy_rule) in ALL_WINNERS:
        if sel is not None and short not in sel:
            continue
        print(f"\n{'='*72}\n{short}  ({run_id})\n{'='*72}", flush=True)
        run_dir = RUNS_DIR / run_id
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        eff = f"{kg}_{fmb_variant}" if fmb_variant else kg
        aug, info, genes = load_kg_data(kg, node_types, eff)
        group_names = list(info.group_names)
        flat_terms, flat_groups = [], []
        for gi_, gn in enumerate(group_names):
            for tn in info.term_names[gi_]:
                flat_terms.append(str(tn))
                flat_groups.append(gn)

        model = create_model("path_attn", info, hidden_dim=HIDDEN, dropout=DROPOUT).to(device)
        state = torch.load(run_dir / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        got = state_dict_hash(model.state_dict())
        hash_ok = (got == cfg["state_dict_hash"])
        rnpz = np.load(run_dir / "risks.npz")
        present = [c for c in EVAL_COHORTS if c in aug]
        sane = True
        for c in present:
            fr = fwd(model, aug[c]["mut"], aug[c]["mask"], aug[c]["fmb"])["log_risk"]
            if not np.allclose(fr, rnpz[f"risk_{c}"], atol=1e-4):
                sane = False
        print(f"  hash_ok={hash_ok}  risk_sane={sane}", flush=True)
        rec = {"run_id": run_id, "kg": kg, "node_types": node_types,
               "fmb_variant": fmb_variant, "deploy_thr": deploy_thr,
               "deploy_rule": deploy_rule, "group_names": group_names,
               "hash_ok": bool(hash_ok), "risk_sane": bool(sane)}
        if not (hash_ok and sane):
            rec["status"] = "GATE_FAIL"
            summary[short] = rec
            continue

        n = aug["train"]["mut"].shape[0]
        tr_idx, _ = kfold_indices(n, 5, seed=seed)[fold]
        tr = _select(aug["train"], tr_idx)
        tr_t, tr_e = tr["time"].numpy(), tr["event"].numpy()

        base_tr = fwd(model, tr["mut"], tr["mask"], tr["fmb"])["log_risk"]
        out_c = {c: fwd(model, aug[c]["mut"], aug[c]["mask"], aug[c]["fmb"]) for c in present}
        coh_base = {c: (out_c[c]["log_risk"], aug[c]["time"].numpy(), aug[c]["event"].numpy())
                    for c in present}
        b_fmax, b_nsig, b_cid, _ = rescore(base_tr, tr_t, tr_e, coh_base)
        b_ci = pooled_cindex(coh_base)
        rec["baseline_fixed_max"] = b_fmax
        rec["baseline_best_cutoff"] = b_cid
        rec["baseline_pooled_cindex"] = round(b_ci, 4)
        rec["config_fixed_max"] = cfg.get("fixed_max_n_sig")
        print(f"  baseline fixed_max={b_fmax} (cfg={cfg.get('fixed_max_n_sig')})  "
              f"pooled_C={b_ci:.4f}", flush=True)

        gi = np.concatenate([out_c[c]["group_importance"] for c in present])
        ti = np.concatenate([out_c[c]["term_importance"] for c in present])
        ge = np.concatenate([out_c[c]["gene_importance"] for c in present])
        lr = np.concatenate([out_c[c]["log_risk"] for c in present])
        cohort_of = np.concatenate([[c] * len(out_c[c]["log_risk"]) for c in present])
        ctmap_all = {c: cancer_type_map(c) for c in present}
        cancer_of = np.concatenate([
            [ctmap_all[c].get(str(s), "NA") for s in aug[c]["sample_ids"]] for c in present])
        time_all = np.concatenate([aug[c]["time"].numpy() for c in present])
        event_all = np.concatenate([aug[c]["event"].numpy() for c in present])
        sid_all = np.concatenate([np.array(aug[c]["sample_ids"], dtype=object) for c in present])
        hi = lr >= deploy_thr

        def imp_table(arr, names, groups=None):
            mean = arr.mean(0)
            if hi.any() and (~hi).any():
                d = arr[hi].mean(0) - arr[~hi].mean(0)
            else:
                d = np.full(arr.shape[1], np.nan)
            return [{"name": names[i], "group": (groups[i] if groups else ""),
                     "mean": float(mean[i]), "delta_hi_lo": float(d[i])}
                    for i in range(len(names))]

        grp_rows = imp_table(gi, group_names)
        term_rows = imp_table(ti, flat_terms, flat_groups)
        gene_rows = imp_table(ge, genes)
        rec["group_importance"] = sorted(grp_rows, key=lambda r: -r["mean"])
        rec["top_terms_by_mean"] = sorted(term_rows, key=lambda r: -r["mean"])[:20]
        rec["top_terms_by_delta"] = sorted(term_rows, key=lambda r: -r["delta_hi_lo"])[:20]
        rec["top_genes_by_mean"] = sorted(gene_rows, key=lambda r: -r["mean"])[:30]
        rec["top_genes_by_delta"] = sorted(gene_rows, key=lambda r: -r["delta_hi_lo"])[:30]
        rec["dominant_group"] = max(grp_rows, key=lambda r: r["mean"])["name"]

        med_t = np.median(time_all)
        correct_high = hi & (event_all == 1) & (time_all <= med_t)
        correct_low = (~hi) & (event_all == 0) & (time_all >= med_t)
        dist = np.abs(lr - deploy_thr)
        picks = {}
        if correct_high.any():
            picks["correct_high"] = int(np.where(correct_high)[0][np.argmax(lr[correct_high])])
        if correct_low.any():
            picks["correct_low"] = int(np.where(correct_low)[0][np.argmin(lr[correct_low])])
        picks["boundary"] = int(np.argmin(dist))
        picks["most_confident_high"] = int(np.argmax(lr))
        picks["most_confident_low"] = int(np.argmin(lr))

        samp_rows = []
        for tag, idx in picks.items():
            c = cohort_of[idx]
            local = list(aug[c]["sample_ids"]).index(sid_all[idx])
            m1 = aug[c]["mut"][local:local + 1].clone()
            mk = aug[c]["mask"][local:local + 1].clone()
            fb = aug[c]["fmb"][local:local + 1].clone()
            orig = float(fwd(model, m1, mk, fb)["log_risk"][0])
            top_t = np.argsort(-ti[idx])[:5]
            top_g = np.argsort(-ge[idx])[:5]
            fb2 = fb.clone()
            fb2[0, top_t] = 0.0
            m2 = m1.clone()
            m2[0, top_g] = 0.0
            d_term = float(fwd(model, m1, mk, fb2)["log_risk"][0]) - orig
            d_gene = float(fwd(model, m2, mk, fb)["log_risk"][0]) - orig
            samp_rows.append({
                "tag": tag, "sample_id": str(sid_all[idx]), "cohort": str(c),
                "cancer_type": str(cancer_of[idx]),
                "log_risk": round(orig, 4), "deploy_high": bool(hi[idx]),
                "OS_months": round(float(time_all[idx]), 2), "event": int(event_all[idx]),
                "top_terms": [flat_terms[i] for i in top_t],
                "top_term_groups": [flat_groups[i] for i in top_t],
                "top_genes": [genes[i] for i in top_g],
                "delta_risk_remove_top5_terms": round(d_term, 4),
                "delta_risk_remove_top5_genes": round(d_gene, 4),
            })
        rec["samples"] = samp_rows

        # ---- PERTURBATION ----
        def rescore_masked(mask_fn):
            tr_r = fwd(model, *mask_fn(tr["mut"], tr["mask"], tr["fmb"]))["log_risk"]
            coh = {}
            for c in present:
                r = fwd(model, *mask_fn(aug[c]["mut"], aug[c]["mask"], aug[c]["fmb"]))["log_risk"]
                coh[c] = (r, aug[c]["time"].numpy(), aug[c]["event"].numpy())
            fmax, _ns, _cid, _ = rescore(tr_r, tr_t, tr_e, coh)
            ci = pooled_cindex(coh)
            dr = float(np.mean(np.concatenate([coh[c][0] for c in present]) - lr))
            return fmax, ci, dr

        grp_order = np.argsort(-gi.mean(0))  # noqa: F841 (kept for parity)
        term_order = np.argsort(-ti.mean(0))
        gene_order = np.argsort(-ge.mean(0))
        T = ti.shape[1]
        Gn = len(genes)

        def mk_term(cols):
            cols = list(cols)
            def f(mut, mask, fmb):
                fb = fmb.clone()
                fb[:, cols] = 0.0
                return mut, mask, fb
            return f

        def mk_gene(cols):
            cols = list(cols)
            def f(mut, mask, fmb):
                m = mut.clone()
                m[:, cols] = 0.0
                return m, mask, fmb
            return f

        def mk_group(g):
            s, e = info.fmb_slices[g]
            def f(mut, mask, fmb):
                fb = fmb.clone()
                fb[:, s:e] = 0.0
                return mut, mask, fb
            return f

        pert = {"baseline": {"fixed_max": b_fmax, "pooled_cindex": round(b_ci, 4)}}
        grp_res = []
        for g in range(len(group_names)):
            fmax, ci, dr = rescore_masked(mk_group(g))
            grp_res.append({"group": group_names[g], "fixed_max": fmax,
                            "d_fixed_max": fmax - b_fmax, "d_cindex": round(ci - b_ci, 4),
                            "mean_d_risk": round(dr, 4)})
        fmax, ci, dr = rescore_masked(lambda mu, ma, fb: (mu, ma, torch.zeros_like(fb)))
        grp_res.append({"group": "ALL_KG_OFF(fmb=0,TMB-only)", "fixed_max": fmax,
                        "d_fixed_max": fmax - b_fmax, "d_cindex": round(ci - b_ci, 4),
                        "mean_d_risk": round(dr, 4)})
        pert["group_mask"] = grp_res

        def topk_vs_random(order, K, mk, dim):
            fmax, ci, dr = rescore_masked(mk(order[:K]))
            rnd = [rescore_masked(mk(RNG.choice(dim, K, replace=False))) for _ in range(N_RANDOM)]
            r_fmax = np.array([x[0] for x in rnd])
            r_ci = np.array([x[1] for x in rnd])
            return {"K": K, "top_fixed_max": fmax, "top_d_fixed_max": fmax - b_fmax,
                    "top_d_cindex": round(ci - b_ci, 4), "top_mean_d_risk": round(dr, 4),
                    "rand_fixed_max_mean": round(float(r_fmax.mean()), 2),
                    "rand_fixed_max_std": round(float(r_fmax.std()), 2),
                    "rand_d_cindex_mean": round(float((r_ci - b_ci).mean()), 4)}

        pert["term_mask"] = [topk_vs_random(term_order, K, mk_term, T) for K in (3, 5, 10)]
        pert["gene_mask"] = [topk_vs_random(gene_order, K, mk_gene, Gn) for K in (5, 10, 20)]
        rec["perturbation"] = pert
        rec["status"] = "OK"
        summary[short] = rec

        pd.DataFrame(rec["group_importance"]).to_csv(OUT / f"{short}_group.csv", index=False)
        pd.DataFrame(samp_rows).to_csv(OUT / f"{short}_samples.csv", index=False)
        print("  group_mask: " + ", ".join(
            f"{r['group'][:10]}:{r['d_fixed_max']:+d}" for r in grp_res), flush=True)
        (OUT / f"_win_{short}.json").write_text(
            json.dumps(rec, indent=2, default=_js), encoding="utf-8")

    merged = {}
    for p in sorted(OUT.glob("_win_*.json")):
        merged[p.stem[5:]] = json.loads(p.read_text(encoding="utf-8"))
    (OUT / "interp_os_deep.json").write_text(
        json.dumps(merged, indent=2, default=_js), encoding="utf-8")
    print(f"\nSaved -> {OUT}/interp_os_deep.json  ({len(merged)} winners)", flush=True)


if __name__ == "__main__":
    main()
