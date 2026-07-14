"""Evaluate extended node types extracted from the source knowledge graphs.

Test whether fine-grained edge types (currently stripped in subkg/) add signal:
  - DRKG STRING PPI subtypes (REACTION/CATALYSIS/BINDING/ACTIVATION/INHIBITION)
  - DRKG INTACT PTM reactions (PHOSPHORYLATION/UBIQUITINATION/etc)
  - DRKG STRING::PTMOD
  - DRKG GNBR gene-disease directional subtypes (merged)
  - Monarch variant (is_sequence_variant_of)
  - Monarch orthology (orthologous_to)
  - Monarch causal disease (causes + contributes_to + associated_with_increased_likelihood_of)

Each new type is computed via the same FMB-burden (gene→term) or adj-burden
(gene→gene) pattern used in node_type_ablation, then appended to the KG's
best multi-node combo and benchmarked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kg_features import (
    load_candidate_genes, extract_function_gene_map,
    compute_functional_burden, build_ppi_adjacency, compute_ppi_burden,
)
from multi_node_extended import (
    augment_splits, build_combo_info, load_base_splits, train_and_eval,
    EVAL_COHORTS, MODEL,
)
from node_type_ablation import ALL_NODE_TYPES, compute_node_features

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "output" / "ext_edges"
EXP_DIR = ROOT / "output" / "experiments"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Extended node types: {tag: (kg, mode, [relations], description)}
EXT_TYPES = {
    # DRKG extensions
    "drkg_string_reaction":  ("drkg", "adj", ["STRING::REACTION::Gene:Gene"], "STRING reaction"),
    "drkg_string_catalysis": ("drkg", "adj", ["STRING::CATALYSIS::Gene:Gene"], "STRING catalysis"),
    "drkg_string_binding":   ("drkg", "adj", ["STRING::BINDING::Gene:Gene"], "STRING binding"),
    "drkg_string_activation":("drkg", "adj", ["STRING::ACTIVATION::Gene:Gene"], "STRING activation"),
    "drkg_string_inhibition":("drkg", "adj", ["STRING::INHIBITION::Gene:Gene"], "STRING inhibition"),
    "drkg_ptm_string":       ("drkg", "adj", ["STRING::PTMOD::Gene:Gene"], "STRING PTMOD"),
    "drkg_ptm_intact": (
        "drkg", "adj",
        ["INTACT::PHOSPHORYLATION REACTION::Gene:Gene",
         "INTACT::UBIQUITINATION REACTION::Gene:Gene",
         "INTACT::DEPHOSPHORYLATION REACTION::Gene:Gene",
         "INTACT::CLEAVAGE REACTION::Gene:Gene",
         "INTACT::ADP RIBOSYLATION REACTION::Gene:Gene",
         "INTACT::PROTEIN CLEAVAGE::Gene:Gene"],
        "INTACT PTM reactions",
    ),
    "drkg_gnbr_disease": (
        "drkg", "fmb",
        ["GNBR::L::Gene:Disease", "GNBR::J::Gene:Disease",
         "GNBR::U::Gene:Disease", "GNBR::D::Gene:Disease",
         "GNBR::Te::Gene:Disease", "GNBR::Y::Gene:Disease",
         "GNBR::Md::Gene:Disease", "GNBR::Ud::Gene:Disease",
         "GNBR::G::Gene:Disease", "GNBR::X::Gene:Disease"],
        "GNBR gene-disease (all 10 subtypes)",
    ),

    # Monarch extensions
    "monarch_variant": (
        "monarch", "fmb",
        ["biolink:is_sequence_variant_of", "biolink:has_sequence_variant"],
        "variant nodes",
    ),
    "monarch_orthology": (
        "monarch", "fmb",
        ["biolink:orthologous_to"],
        "orthologous genes",
    ),
    "monarch_causal_disease": (
        "monarch", "fmb",
        ["biolink:causes", "biolink:contributes_to",
         "biolink:associated_with_increased_likelihood_of",
         "biolink:genetically_associated_with"],
        "causal disease edges",
    ),
}

# KG best combos (baselines)
BEST_COMBOS = {
    "drkg":    ["ppi", "disease", "drug"],
    "monarch": ["ppi", "disease", "phenotype", "anatomy"],
}

BASELINES = {
    "drkg":    {"n_sig": 7, "ext_ci": 0.598},
    "monarch": {"n_sig": 6, "ext_ci": 0.596},
}


def compute_ext_node_features(kg: str, mode: str, relations: list[str],
                               genes: list[str], raw_splits: dict):
    """Compute burden features from extended edges.

    Args:
        kg: "drkg" or "monarch"
        mode: "fmb" or "adj"
        relations: full relation names in ext_edges CSV
        genes: candidate gene list
        raw_splits: {split_name: (mut, mask)} numpy arrays

    Returns:
        (features_dict, term_names, count_info) or None if no edges match.
    """
    ext_path = EXT / f"{kg}_ext.csv"
    if not ext_path.exists():
        return None

    if mode == "adj":
        adj = build_ppi_adjacency(ext_path, genes, relations)
        n_edges = int(adj.sum() / 2)
        if n_edges == 0:
            return None
        features = {}
        for prefix, (mut, mask) in raw_splits.items():
            features[prefix] = compute_ppi_burden(adj, mut, mask)
        term_names = [f"ext_{g}" for g in genes]
        return features, term_names, n_edges

    if mode == "fmb":
        func_genes = extract_function_gene_map(ext_path, genes, relations, min_genes=2)
        if len(func_genes) < 5:
            return None
        features = {}
        for prefix, (mut, mask) in raw_splits.items():
            fmb_mat, fmb_names = compute_functional_burden(func_genes, genes, mut, mask)
            features[prefix] = fmb_mat
        return features, fmb_names, len(func_genes)

    return None


def cache_base_and_node_feats(kg: str, node_types: list[str], genes: list[str]):
    """Load splits and compute base-combo node features once per KG."""
    splits, raw = load_base_splits(kg)
    nf = {}
    for nt in node_types:
        mode, edict = ALL_NODE_TYPES[nt]
        res = compute_node_features(kg, nt, mode, edict, genes, raw)
        if res:
            feats, tnames, m = res
            nf[nt] = (feats, tnames, mode)
    return splits, raw, nf


def run_config(kg: str, splits, raw, node_feats, node_types, ext_tags: list[str],
               genes: list[str], seed: int = 42, label: str = "") -> dict:
    """Train one config: baseline + specified extensions."""
    n_genes = len(genes)
    # Check all base types present
    missing = [nt for nt in node_types if nt not in node_feats]
    if missing:
        return {"status": "skip", "missing": missing, "label": label}

    extra_f = [node_feats[nt][0] for nt in node_types]
    extra_i = [(node_feats[nt][1], node_feats[nt][2], f"x_{nt}") for nt in node_types]

    # Add extensions
    ext_info = []
    for tag in ext_tags:
        kg_ext, mode, rels, desc = EXT_TYPES[tag]
        if kg_ext != kg:
            return {"status": "skip", "reason": f"{tag} belongs to {kg_ext}, not {kg}",
                    "label": label}
        res = compute_ext_node_features(kg, mode, rels, genes, raw)
        if res is None:
            return {"status": "skip", "reason": f"{tag} produced 0 features",
                    "label": label}
        feats, tnames, n_info = res
        extra_f.append(feats)
        extra_i.append((tnames, mode, f"x_{tag}"))
        ext_info.append({"tag": tag, "mode": mode, "n": n_info,
                         "n_features": len(tnames)})
        print(f"    + {tag}: {n_info} {'edges' if mode == 'adj' else 'terms'}")

    aug = augment_splits(splits, extra_f)
    info = build_combo_info(kg, extra_i, n_genes)

    r = train_and_eval(info, aug["train"],
                        {c: aug[c] for c in EVAL_COHORTS if c in aug},
                        seed=seed)
    r.update({
        "status": "ok", "kg": kg, "label": label,
        "base_types": node_types, "ext_tags": ext_tags,
        "ext_info": ext_info,
    })
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ext_node_types.json")
    args = parser.parse_args()

    genes = load_candidate_genes()
    all_results = []

    # === DRKG experiments ===
    print("\n=== DRKG: caching base splits + node features ===")
    drkg_splits, drkg_raw, drkg_nf = cache_base_and_node_feats(
        "drkg", BEST_COMBOS["drkg"], genes)
    drkg_base_combo = BEST_COMBOS["drkg"]

    drkg_configs = [
        ("drkg_baseline",       []),
        ("+ptm_string",         ["drkg_ptm_string"]),
        ("+ptm_intact",         ["drkg_ptm_intact"]),
        ("+ptm_both",           ["drkg_ptm_string", "drkg_ptm_intact"]),
        ("+gnbr_disease",       ["drkg_gnbr_disease"]),
        ("+string_reaction",    ["drkg_string_reaction"]),
        ("+string_catalysis",   ["drkg_string_catalysis"]),
        ("+string_binding",     ["drkg_string_binding"]),
        ("+string_activation",  ["drkg_string_activation"]),
        ("+string_inhibition",  ["drkg_string_inhibition"]),
        ("+string_all5",        ["drkg_string_reaction","drkg_string_catalysis",
                                  "drkg_string_binding","drkg_string_activation",
                                  "drkg_string_inhibition"]),
        ("+ptm+gnbr_disease",   ["drkg_ptm_string","drkg_ptm_intact","drkg_gnbr_disease"]),
        ("+all_ext",            ["drkg_ptm_string","drkg_ptm_intact","drkg_gnbr_disease",
                                  "drkg_string_reaction","drkg_string_catalysis",
                                  "drkg_string_binding","drkg_string_activation",
                                  "drkg_string_inhibition"]),
    ]

    for label, ext_tags in drkg_configs:
        print(f"\n[DRKG] {label}")
        r = run_config("drkg", drkg_splits, drkg_raw, drkg_nf,
                       drkg_base_combo, ext_tags, genes, seed=42, label=label)
        if r["status"] == "ok":
            base_sig = BASELINES["drkg"]["n_sig"]
            d = r["n_sig"] - base_sig
            ds = f"{'+' if d >= 0 else ''}{d}"
            print(f"  n_sig={r['n_sig']}/11 ({ds} vs 7)  ext_ci={r['ext_ci']:.4f}  "
                  f"sigs={r['sigs']}")
        else:
            print(f"  SKIP: {r.get('reason') or r.get('missing')}")
        all_results.append(r)
        # Save incrementally
        with open(EXP_DIR / args.out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # === Monarch experiments ===
    print("\n=== Monarch: caching base splits + node features ===")
    m_splits, m_raw, m_nf = cache_base_and_node_feats(
        "monarch", BEST_COMBOS["monarch"], genes)
    m_base_combo = BEST_COMBOS["monarch"]

    monarch_configs = [
        ("monarch_baseline",    []),
        ("+variant",            ["monarch_variant"]),
        ("+orthology",          ["monarch_orthology"]),
        ("+causal_disease",     ["monarch_causal_disease"]),
        ("+variant+orthology",  ["monarch_variant", "monarch_orthology"]),
        ("+all3",               ["monarch_variant","monarch_orthology",
                                  "monarch_causal_disease"]),
    ]

    for label, ext_tags in monarch_configs:
        print(f"\n[Monarch] {label}")
        r = run_config("monarch", m_splits, m_raw, m_nf,
                       m_base_combo, ext_tags, genes, seed=42, label=label)
        if r["status"] == "ok":
            base_sig = BASELINES["monarch"]["n_sig"]
            d = r["n_sig"] - base_sig
            ds = f"{'+' if d >= 0 else ''}{d}"
            print(f"  n_sig={r['n_sig']}/11 ({ds} vs 6)  ext_ci={r['ext_ci']:.4f}  "
                  f"sigs={r['sigs']}")
        else:
            print(f"  SKIP: {r.get('reason') or r.get('missing')}")
        all_results.append(r)
        with open(EXP_DIR / args.out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved → {EXP_DIR / args.out}")


if __name__ == "__main__":
    main()
