"""Extract extended edges from raw ref_KG with FULL relation names preserved.

Unlike build_subkg_multi.py (which strips relation prefixes like
`STRING::REACTION::Gene:Gene` → `Gene:Gene`), this keeps them intact so we can
access STRING-specific PPI subtypes, INTACT PTM reactions, GNBR gene-disease
directional subtypes, Monarch variant/orthology/causal edges, etc.

Outputs per-KG cached CSVs in output/ext_edges/{kg}_edges.csv with candidate-gene
filtered rows. Re-run only when source ref_KG changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "ref_KG"
OUT = ROOT / "output" / "ext_edges"
GENE_FILE = ROOT / "source" / "input_data" / "train" / "gene_candidate.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_features import GENE_ALIAS_REV  # reuse alias-reverse for normalization


def load_candidates() -> set[str]:
    return set(pd.read_csv(GENE_FILE).iloc[:, 0].dropna().str.strip())


def load_entrez_map() -> dict[int, str]:
    df = pd.read_csv(REF / "entrez_to_symbol.csv")
    return dict(zip(df["entrez_id"].astype(int), df["symbol"]))


def load_hgnc_map() -> dict[str, str]:
    df = pd.read_csv(REF / "ibkh" / "Entity" / "gene_vocab.csv")
    return dict(zip(df["primary"], df["symbol"]))


def _norm_symbol(sym: str, cands: set[str]) -> str | None:
    """Return symbol if it's in candidates (direct or via alias), else None."""
    if sym in cands:
        return sym
    old = GENE_ALIAS_REV.get(sym)
    if old and old in cands:
        return old
    return None


def extract_drkg_full(keep_prefixes: list[str] | None = None) -> pd.DataFrame:
    """Extract DRKG triples keeping full relation name.

    Args:
        keep_prefixes: list of relation-name substrings to keep (e.g.,
            ['STRING::', 'INTACT::', 'GNBR::']). None = keep all.
    """
    print("  [DRKG] scanning ref_KG/drkg.tsv ...")
    entrez_map = load_entrez_map()
    cands = load_candidates()
    rows = []
    with open(REF / "drkg.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if keep_prefixes and not any(p in r for p in keep_prefixes):
                continue
            hp = h.split("::")
            tp = t.split("::")
            ht, hi = hp[0], hp[1] if len(hp) > 1 else ""
            tt, ti = tp[0], tp[1] if len(tp) > 1 else ""
            hn = entrez_map.get(int(hi), hi) if ht == "Gene" and hi.isdigit() else hi
            tn = entrez_map.get(int(ti), ti) if tt == "Gene" and ti.isdigit() else ti
            # Keep only if at least one end is a candidate gene (symbol or alias)
            hnorm = _norm_symbol(str(hn), cands) if ht == "Gene" else None
            tnorm = _norm_symbol(str(tn), cands) if tt == "Gene" else None
            if not hnorm and not tnorm:
                continue
            rows.append({"relation": r,
                          "x_type": ht.lower(), "x_name": hnorm or str(hn),
                          "y_type": tt.lower(), "y_name": tnorm or str(tn)})
    df = pd.DataFrame(rows)
    print(f"  [DRKG] kept {len(df):,} rows with prefixes={keep_prefixes}")
    return df


def extract_monarch_full(keep_predicates: list[str] | None = None) -> pd.DataFrame:
    """Extract Monarch triples keeping `biolink:` predicate.

    Args:
        keep_predicates: set of predicates (e.g., ['biolink:is_sequence_variant_of',
            'biolink:orthologous_to']). None = keep all.
    """
    print("  [Monarch] scanning monarch-kg_edges.tsv ...")
    hgnc_map = load_hgnc_map()
    cands = load_candidates()
    # Build HGNC → symbol map for candidate filter
    # Node file has HGNC: IDs → names; we only need to recognize HGNC: prefix gene IDs
    sym_by_hgnc: dict[str, str] = {}
    with open(REF / "monarch_kg" / "monarch-kg_nodes.tsv", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_i = header.index("id")
        nm_i = header.index("name") if "name" in header else -1
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= id_i:
                continue
            nid = parts[id_i]
            if not nid.startswith("HGNC:"):
                continue
            sym = hgnc_map.get(nid)
            if not sym and nm_i >= 0 and len(parts) > nm_i:
                sym = parts[nm_i]
            if sym:
                sym_by_hgnc[nid] = sym

    rows = []
    with open(REF / "monarch_kg" / "monarch-kg_edges.tsv", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        si = header.index("subject")
        oi = header.index("object")
        pi = header.index("predicate")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(si, oi, pi):
                continue
            pred = parts[pi]
            if keep_predicates and pred not in keep_predicates:
                continue
            subj, obj = parts[si], parts[oi]
            # Resolve gene IDs via HGNC map (only for human genes)
            s_sym = sym_by_hgnc.get(subj)
            o_sym = sym_by_hgnc.get(obj)
            s_norm = _norm_symbol(s_sym, cands) if s_sym else None
            o_norm = _norm_symbol(o_sym, cands) if o_sym else None
            if not s_norm and not o_norm:
                continue
            rows.append({
                "relation": pred,
                "x_type": "gene" if s_sym else "other",
                "x_name": s_norm or subj,
                "y_type": "gene" if o_sym else "other",
                "y_name": o_norm or obj,
            })
    df = pd.DataFrame(rows)
    print(f"  [Monarch] kept {len(df):,} rows for predicates={keep_predicates}")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- DRKG: STRING PPI + INTACT PTM + GNBR disease ---
    drkg_df = extract_drkg_full(keep_prefixes=["STRING::", "INTACT::", "GNBR::"])
    drkg_path = OUT / "drkg_ext.csv"
    drkg_df.to_csv(drkg_path, index=False)
    print(f"  saved → {drkg_path}   (by relation: {drkg_df['relation'].nunique()})")
    print("  top relations:")
    for r, n in drkg_df["relation"].value_counts().head(15).items():
        print(f"    {r:<55} {n:>8}")

    # --- Monarch: variant + orthology + causal ---
    monarch_preds = [
        "biolink:is_sequence_variant_of",
        "biolink:has_sequence_variant",
        "biolink:orthologous_to",
        "biolink:homologous_to",
        "biolink:causes",
        "biolink:contributes_to",
        "biolink:associated_with_increased_likelihood_of",
        "biolink:genetically_associated_with",
    ]
    monarch_df = extract_monarch_full(keep_predicates=monarch_preds)
    monarch_path = OUT / "monarch_ext.csv"
    monarch_df.to_csv(monarch_path, index=False)
    print(f"  saved → {monarch_path}   (by predicate: {monarch_df['relation'].nunique()})")
    print("  top predicates:")
    for r, n in monarch_df["relation"].value_counts().items():
        print(f"    {r:<55} {n:>8}")


if __name__ == "__main__":
    main()
