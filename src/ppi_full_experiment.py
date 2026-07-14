"""Full PPI experiment: 3 models x 7 KGs x 2 modes = 42 experiments."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_interp import (
    KG_DIR, PROC_DIR, ALL_KGS, VALID_COHORTS,
    KGGroupInfo, _load_gene_list, build_kg_group_info,
)
from losses import c_index, compute_all_metrics, cox_loss
from models_interp import ALL_MODELS, create_model, count_parameters
from train_interp import _seed_everything, _split_data, train_epoch, evaluate_ci

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_COHORTS = [
    "Gandara", "Hugo", "Liu", "Mariathasan", "Miao",
    "PUSH", "Pleasance", "Ravi", "Riaz", "Snyder_UC", "Whijae",
]


def load_split(kg_name, split):
    prefix = "train" if split == "train" else f"valid_{split}"
    kg_feat = KG_DIR / kg_name
    mut = pd.read_csv(PROC_DIR / f"{prefix}_mut.csv", index_col=0)
    mask = pd.read_csv(PROC_DIR / f"{prefix}_mask.csv", index_col=0)
    clin = pd.read_csv(PROC_DIR / f"{prefix}_clin.csv", index_col=0)
    fmb = pd.read_csv(kg_feat / f"{prefix}_fmb.csv", index_col=0)
    ppi_path = kg_feat / f"{prefix}_ppi.csv"
    ppi = pd.read_csv(ppi_path, index_col=0) if ppi_path.exists() else None
    common = mut.index.intersection(clin.index).intersection(fmb.index)
    if ppi is not None:
        common = common.intersection(ppi.index)
    d = {
        "mut": torch.tensor(mut.loc[common].values, dtype=torch.float32),
        "mask": torch.tensor(mask.loc[common].values, dtype=torch.float32),
        "time": torch.tensor(clin.loc[common, "OS_MONTHS"].values, dtype=torch.float32),
        "event": torch.tensor(clin.loc[common, "event"].values, dtype=torch.float32),
        "fmb": torch.tensor(fmb.loc[common].values, dtype=torch.float32),
        "sample_ids": common.tolist(),
    }
    if ppi is not None:
        d["ppi"] = torch.tensor(ppi.loc[common].values, dtype=torch.float32)
    return d


def build_info_with_ppi(kg_name):
    base = build_kg_group_info(kg_name)
    genes = _load_gene_list()
    ppi_terms = [f"ppi_{g}" for g in genes]
    ppi_mask = torch.eye(len(genes), dtype=torch.float32)
    old_end = base.fmb_slices[-1][1] if base.fmb_slices else 0
    return KGGroupInfo(
        kg_name=kg_name,
        group_names=base.group_names + ["ppi_neighborhood"],
        term_names=base.term_names + [ppi_terms],
        gene_term_mask=base.gene_term_mask + [ppi_mask],
        fmb_slices=base.fmb_slices + [(old_end, old_end + len(ppi_terms))],
        n_genes=base.n_genes,
        n_total_terms=base.n_total_terms + len(ppi_terms),
    )


def concat_ppi(data):
    out = dict(data)
    out["fmb"] = torch.cat([data["fmb"], data["ppi"]], dim=1)
    return out


@torch.no_grad()
def get_risk(model, data):
    model.eval()
    out = model(data["mut"].to(device), data["mask"].to(device), data["fmb"].to(device))
    return out["log_risk"].cpu().numpy()


def train_model(model_name, kg_info, train_data, seed=42):
    _seed_everything(seed)
    tr, va = _split_data(train_data, 0.8, seed)
    model = create_model(model_name, kg_info, hidden_dim=32, dropout=0.1)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=7, factor=0.5, min_lr=1e-6)
    best, pat, st = 0.0, 0, None
    for ep in range(1, 81):
        train_epoch(model, tr, opt, 64, device)
        ci = evaluate_ci(model, va, device)
        sch.step(ci)
        if ci > best:
            best, pat = ci, 0
            st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= 15:
            break
    if st:
        model.load_state_dict(st)
    return model, best


def main():
    all_results = []
    total = len(ALL_MODELS) * len(ALL_KGS) * 2
    done = 0

    for kg in ALL_KGS:
        print(f"\n{'='*60}")
        print(f"  Loading {kg}")
        print(f"{'='*60}")

        train_base = load_split(kg, "train")
        valid_base = {}
        for c in EVAL_COHORTS:
            try:
                valid_base[c] = load_split(kg, c)
            except FileNotFoundError:
                pass

        has_ppi = "ppi" in train_base
        ki_base = build_kg_group_info(kg)
        ki_ppi = build_info_with_ppi(kg) if has_ppi else None

        for model_name in ALL_MODELS:
            for mode in ["fmb_only", "fmb+ppi"]:
                if mode == "fmb+ppi" and not has_ppi:
                    continue
                done += 1
                tag = f"{model_name}_{kg}_{mode.replace('+','_')}"
                print(f"\n[{done}/{total}] {tag}")

                if mode == "fmb_only":
                    ki, td = ki_base, train_base
                    vd = valid_base
                else:
                    ki = ki_ppi
                    td = concat_ppi(train_base)
                    vd = {c: concat_ppi(d) for c, d in valid_base.items() if "ppi" in d}

                model, val_ci = train_model(model_name, ki, td)

                r = {"tag": tag, "model": model_name, "kg": kg, "mode": mode,
                     "val_ci": round(val_ci, 4)}
                ext_cis, n_sig = [], 0
                for c in EVAL_COHORTS:
                    if c not in vd:
                        continue
                    risk = get_risk(model, vd[c])
                    m = compute_all_metrics(risk, vd[c]["time"].numpy(), vd[c]["event"].numpy())
                    r[f"{c}_ci"] = round(m["c_index"], 4)
                    r[f"{c}_p"] = round(m["p_value"], 4)
                    ext_cis.append(m["c_index"])
                    if m["p_value"] < 0.05:
                        n_sig += 1
                r["ext_avg_ci"] = round(np.mean(ext_cis), 4)
                r["n_sig"] = n_sig
                sigs = [c for c in EVAL_COHORTS if r.get(f"{c}_p", 1) < 0.05]
                print(f"  val={val_ci:.4f} ext={r['ext_avg_ci']:.4f} "
                      f"sig={n_sig}/11: {sigs}")
                all_results.append(r)

    # Summary
    print(f"\n{'='*90}")
    print("FULL RESULTS: 3 models x 7 KGs x 2 modes")
    print(f"{'='*90}")
    print(f"{'Tag':<45} {'ValCI':>6} {'ExtCI':>6} {'Sig':>5}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: (-x["n_sig"], -x["ext_avg_ci"])):
        print(f"{r['tag']:<45} {r['val_ci']:>6.4f} {r['ext_avg_ci']:>6.4f} {r['n_sig']:>3}/11")

    out = EXP_DIR / "ppi_full_experiment.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
