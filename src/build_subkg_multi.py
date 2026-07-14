"""
Multi-KG subgraph extraction - unified adapters for 7 biomedical KGs.

Usage:
    python src/build_subkg_multi.py                 # Extract all KGs
    python src/build_subkg_multi.py --kg hetionet    # Extract single KG
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "ref_KG"
OUT = ROOT / "output" / "subkg"
GENE_CANDIDATES = ROOT / "source" / "input_data" / "train" / "gene_candidate.csv"


def load_candidates() -> set[str]:
    df = pd.read_csv(GENE_CANDIDATES)
    return set(df.iloc[:, 0].dropna().str.strip())


def load_entrez_map() -> dict[int, str]:
    path = REF / "entrez_to_symbol.csv"
    df = pd.read_csv(path)
    return dict(zip(df["entrez_id"].astype(int), df["symbol"]))


def load_hgnc_map() -> dict[str, str]:
    path = REF / "ibkh" / "Entity" / "gene_vocab.csv"
    df = pd.read_csv(path)
    return dict(zip(df["primary"], df["symbol"]))


def _one_hop_filter(edges, seed, x_col="x_name", y_col="y_name"):
    mask = edges[x_col].isin(seed) | edges[y_col].isin(seed)
    return edges[mask].copy()


def _to_unified(edges, rel_col, xt_col, xn_col, yt_col, yn_col):
    return pd.DataFrame({
        "relation": edges[rel_col], "x_type": edges[xt_col],
        "x_name": edges[xn_col], "y_type": edges[yt_col],
        "y_name": edges[yn_col],
    })


# -- PrimeKG -----------------------------------------------------------------

def extract_primekg(seed):
    print("  Loading PrimeKG...")
    kg = pd.read_csv(REF / "PrimeKG.csv", low_memory=False)
    gene_rels = {"protein_protein", "bioprocess_protein", "pathway_protein",
                 "disease_protein", "molfunc_protein", "cellcomp_protein",
                 "phenotype_protein"}
    kg = kg[kg["relation"].isin(gene_rels)]
    sub = _one_hop_filter(kg, seed, "x_name", "y_name")
    return _to_unified(sub, "relation", "x_type", "x_name", "y_type", "y_name")


# -- Hetionet -----------------------------------------------------------------

def extract_hetionet(seed):
    print("  Loading Hetionet...")
    with open(REF / "hetionet-v1.0.json", encoding="utf-8") as f:
        data = json.load(f)
    node_name = {}
    for n in data["nodes"]:
        node_name[(n["kind"], str(n["identifier"]))] = n["name"]
    rows = []
    for e in data["edges"]:
        sk, si = e["source_id"][0], str(e["source_id"][1])
        tk, ti = e["target_id"][0], str(e["target_id"][1])
        sn = node_name.get((sk, si), si)
        tn = node_name.get((tk, ti), ti)
        rows.append({"relation": e["kind"],
                      "x_type": sk.lower().replace(" ", "_"), "x_name": sn,
                      "y_type": tk.lower().replace(" ", "_"), "y_name": tn})
    df = pd.DataFrame(rows)
    return _one_hop_filter(df, seed)


# -- DRKG ---------------------------------------------------------------------

def extract_drkg(seed):
    print("  Loading DRKG...")
    entrez_map = load_entrez_map()
    rows = []
    with open(REF / "drkg.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            hp = parts[0].split("::")
            tp = parts[2].split("::")
            ht, hi = hp[0], hp[1] if len(hp) > 1 else ""
            tt, ti = tp[0], tp[1] if len(tp) > 1 else ""
            hn = entrez_map.get(int(hi), hi) if ht == "Gene" and hi.isdigit() else hi
            tn = entrez_map.get(int(ti), ti) if tt == "Gene" and ti.isdigit() else ti
            rel = parts[1].split("::")[-1] if "::" in parts[1] else parts[1]
            rows.append({"relation": rel, "x_type": ht.lower(), "x_name": str(hn),
                          "y_type": tt.lower(), "y_name": str(tn)})
    df = pd.DataFrame(rows)
    return _one_hop_filter(df, seed)


# -- OGB-biokg ----------------------------------------------------------------

def extract_ogb_biokg(seed):
    print("  Loading OGB-biokg...")
    import gzip
    import csv
    import numpy as np

    base = REF / "ogb_biokg" / "ogbl_biokg"
    entrez_map = load_entrez_map()
    entity_maps = {}
    for etype in ["protein", "disease", "drug", "function", "sideeffect"]:
        path = base / "mapping" / f"{etype}_entidx2name.csv.gz"
        if path.exists():
            m = {}
            with gzip.open(path, "rt") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    m[int(row[0])] = row[1]
            entity_maps[etype] = m
    rel_map = {}
    with gzip.open(base / "mapping" / "relidx2relname.csv.gz", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rel_map[int(row[0])] = row[1]

    etypes = ["disease", "drug", "function", "protein", "sideeffect"]

    # Load from raw/relations directories (each dir = one relation type)
    raw_rels = base / "raw" / "relations"
    rows = []
    for rel_dir in sorted(raw_rels.iterdir()):
        if not rel_dir.is_dir():
            continue
        edge_file = rel_dir / "edge.csv.gz"
        if not edge_file.exists():
            continue

        rel_name = rel_dir.name  # e.g. "disease___disease-protein___protein"
        parts = rel_name.split("___")
        if len(parts) != 3:
            continue
        h_type_raw = parts[0]
        t_type_raw = parts[2]

        # Only keep gene/protein-involving relations
        if "protein" not in h_type_raw and "protein" not in t_type_raw:
            continue

        import gzip as gzip2
        edges = []
        with gzip2.open(edge_file, "rt") as f:
            for line in f:
                vals = line.strip().split(",")
                if len(vals) >= 2:
                    edges.append((int(vals[0]), int(vals[1])))

        for h_idx, t_idx in edges:
            htn = h_type_raw
            ttn = t_type_raw
            hn = entity_maps.get(htn, {}).get(h_idx, str(h_idx))
            tn = entity_maps.get(ttn, {}).get(t_idx, str(t_idx))

            if htn == "protein":
                mapped = entrez_map.get(int(hn), None) if str(hn).isdigit() else None
                if mapped:
                    hn = mapped
                htn = "gene"
            if ttn == "protein":
                mapped = entrez_map.get(int(tn), None) if str(tn).isdigit() else None
                if mapped:
                    tn = mapped
                ttn = "gene"
            rows.append({"relation": parts[1], "x_type": htn, "x_name": str(hn),
                          "y_type": ttn, "y_name": str(tn)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _one_hop_filter(df, seed)


# -- Monarch ------------------------------------------------------------------

def extract_monarch(seed):
    print("  Loading Monarch KG...")
    hgnc_map = load_hgnc_map()
    node_name = {}
    node_cat = {}
    with open(REF / "monarch_kg" / "monarch-kg_nodes.tsv", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_i = header.index("id")
        nm_i = header.index("name") if "name" in header else -1
        ct_i = header.index("category") if "category" in header else -1
        for line in f:
            parts = line.strip().split("\t")
            nid = parts[id_i] if len(parts) > id_i else ""
            if not nid:
                continue
            if nm_i >= 0 and len(parts) > nm_i:
                node_name[nid] = parts[nm_i]
            if ct_i >= 0 and len(parts) > ct_i:
                node_cat[nid] = parts[ct_i]
            if nid.startswith("HGNC:"):
                sym = hgnc_map.get(nid)
                if sym:
                    node_name[nid] = sym
    rows = []
    with open(REF / "monarch_kg" / "monarch-kg_edges.tsv", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        si = header.index("subject")
        oi = header.index("object")
        pi = header.index("predicate")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(si, oi, pi):
                continue
            subj, obj_val, pred = parts[si], parts[oi], parts[pi]
            sn = node_name.get(subj, subj)
            on = node_name.get(obj_val, obj_val)
            sc = node_cat.get(subj, "unknown")
            oc = node_cat.get(obj_val, "unknown")
            sc = sc.split(":")[-1].lower() if ":" in sc else sc.lower()
            oc = oc.split(":")[-1].lower() if ":" in oc else oc.lower()
            rows.append({"relation": pred.split(":")[-1] if ":" in pred else pred,
                          "x_type": sc, "x_name": sn, "y_type": oc, "y_name": on})
    df = pd.DataFrame(rows)
    return _one_hop_filter(df, seed)


# -- OpenBioLink --------------------------------------------------------------

def extract_openbiolink(seed):
    print("  Loading OpenBioLink...")
    entrez_map = load_entrez_map()
    ntm = {}
    np_ = REF / "openbiolink" / "HQ_DIR" / "graph_files" / "nodes.csv"
    with open(np_) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ntm[parts[0]] = parts[1].lower()
    ep = REF / "openbiolink" / "HQ_DIR" / "graph_files" / "edges.csv"
    rows = []
    with open(ep) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            hid, rel, tid = parts[0], parts[1], parts[2]
            ht = ntm.get(hid, "unknown")
            tt = ntm.get(tid, "unknown")
            hn, tn = hid, tid
            if hid.startswith("NCBIGENE:"):
                hn = entrez_map.get(int(hid.split(":")[1]), hid)
                ht = "gene"
            if tid.startswith("NCBIGENE:"):
                tn = entrez_map.get(int(tid.split(":")[1]), tid)
                tt = "gene"
            rows.append({"relation": rel.lower(), "x_type": ht, "x_name": hn,
                          "y_type": tt, "y_name": tn})
    df = pd.DataFrame(rows)
    return _one_hop_filter(df, seed)


# -- iBKH --------------------------------------------------------------------

def extract_ibkh(seed):
    print("  Loading iBKH...")
    hgnc_map = load_hgnc_map()
    hgnc_series = pd.Series(hgnc_map)
    rel_dir = REF / "ibkh" / "Relation"
    tm = {"A_G": ("anatomy", "gene"), "D_G": ("drug", "gene"),
          "Di_G": ("disease", "gene"), "G_G": ("gene", "gene"),
          "G_Pw": ("gene", "pathway")}
    frames = []
    for fp in sorted(rel_dir.glob("*_res.csv")):
        prefix = fp.stem.replace("_res", "")
        if prefix not in tm:
            continue
        xt, yt = tm[prefix]
        print(f"    {prefix} ...", end=" ", flush=True)
        df = pd.read_csv(fp, low_memory=False)
        if len(df.columns) < 2:
            print("skip (too few cols)")
            continue
        xc, yc = df.columns[0], df.columns[1]

        # Map HGNC IDs to gene symbols (vectorized)
        x_names = df[xc].astype(str)
        y_names = df[yc].astype(str)
        if xt == "gene":
            x_names = x_names.map(hgnc_map).fillna(x_names)
        if yt == "gene":
            y_names = y_names.map(hgnc_map).fillna(y_names)

        # Pre-filter: keep only rows where x or y is a seed gene
        in_seed = x_names.isin(seed) | y_names.isin(seed)
        x_names = x_names[in_seed]
        y_names = y_names[in_seed]
        print(f"{in_seed.sum():,}/{len(df):,} seed-matched")

        if in_seed.sum() == 0:
            continue

        sub = pd.DataFrame({
            "relation": prefix,
            "x_type": xt,
            "x_name": x_names.values,
            "y_type": yt,
            "y_name": y_names.values,
        })
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["relation", "x_type", "x_name", "y_type", "y_name"])
    return pd.concat(frames, ignore_index=True)


# -- Registry & Main ----------------------------------------------------------

ADAPTERS = {
    "primekg": extract_primekg,
    "hetionet": extract_hetionet,
    "drkg": extract_drkg,
    "ogb_biokg": extract_ogb_biokg,
    "monarch": extract_monarch,
    "openbiolink": extract_openbiolink,
    "ibkh": extract_ibkh,
}


def extract_all(kg_names=None):
    seed = load_candidates()
    print(f"Seed genes: {len(seed)}")
    if kg_names is None:
        kg_names = list(ADAPTERS.keys())
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in kg_names:
        if name not in ADAPTERS:
            print(f"  [SKIP] Unknown KG: {name}")
            continue
        print(f"\n{'=' * 60}")
        print(f"Extracting: {name}")
        print("=" * 60)
        try:
            sub = ADAPTERS[name](seed)
            sub = sub.drop_duplicates()
            out_path = OUT / f"subkg_{name}.csv"
            sub.to_csv(out_path, index=False)
            gene_pat = "gene|protein"
            n_gx = sub[sub["x_type"].str.contains(gene_pat, case=False)]["x_name"].nunique()
            n_gy = sub[sub["y_type"].str.contains(gene_pat, case=False)]["y_name"].nunique()
            sc = len(seed & (set(sub["x_name"]) | set(sub["y_name"])))
            print(f"  Edges: {len(sub):,}")
            print(f"  Unique relations: {sub['relation'].nunique()}")
            print(f"  Node types: {sorted(set(sub['x_type']) | set(sub['y_type']))}")
            print(f"  Gene nodes (x): {n_gx}, (y): {n_gy}")
            print(f"  Seed gene coverage: {sc}/{len(seed)}")
            print(f"  Saved to: {out_path}")
            results[name] = sub
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--kg", type=str, default=None)
    args = p.parse_args()
    extract_all([args.kg] if args.kg else None)
