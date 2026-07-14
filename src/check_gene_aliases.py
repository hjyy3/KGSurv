"""
Gene alias/nomenclature check: query MyGene.info (GET) for all 463 candidate genes,
identify deprecated aliases, and produce a rename mapping.
"""
import pandas as pd
import requests
import time
from pathlib import Path

PROCESSED_DIR = Path("output/processed")
OUT_DIR = Path("output/processed")

# ── Load gene list ─────────────────────────────────────────────────────────
df_head = pd.read_csv(PROCESSED_DIR / "train_mut.csv", index_col=0, nrows=1)
genes = list(df_head.columns)
print(f"Total genes in processed data: {len(genes)}")

# ── Known aliases (hardcoded from HGNC) ───────────────────────────────────
KNOWN_ALIASES = {
    "FAM46C":   "TENT5C",
    "PAK7":     "PAK5",
    "SETD8":    "KMT5A",
    "FAM175A":  "ABRAXAS1",
    "FAM175B":  "ABRAXAS2",
    "PARK2":    "PRKN",
    "C11orf30": "EMSY",
    "KIAA1279": "KIF1BP",
    "C17orf39": "GID4",
    "TCEB1":    "ELOC",
    "TCEB2":    "ELOB",
    "MLL":      "KMT2A",
    "MLL2":     "KMT2D",
    "MLL3":     "KMT2C",
    "MLL4":     "KMT2B",
    "EZH1":     "EZH1",   # unchanged
    "WHSC1":    "NSD2",
    "WHSC1L1":  "NSD3",
    "NSD1":     "NSD1",   # unchanged
    "EHMT2":    "EHMT2",  # unchanged
    "CCND1":    "CCND1",  # unchanged
    "MRE11A":   "MRE11",
    "C19orf40": "FAAP24",
    "FAM135B":  "FAM135B", # unchanged
    "AMER1":    "AMER1",
    "CTNNB1":   "CTNNB1",
    "EP300":    "EP300",
    "CREBBP":   "CREBBP",
    "KDM6A":    "KDM6A",
    "ARID2":    "ARID2",
    "DNMT3A":   "DNMT3A",
    "TET2":     "TET2",
    "IDH1":     "IDH1",
    "IDH2":     "IDH2",
    "BRCA1":    "BRCA1",
    "BRCA2":    "BRCA2",
    "RAD51C":   "RAD51C",
    "PALB2":    "PALB2",
    "ATM":      "ATM",
    "CHEK2":    "CHEK2",
    "CDH1":     "CDH1",
    "RB1":      "RB1",
    "PTEN":     "PTEN",
    "TP53":     "TP53",
    "APC":      "APC",
    "VHL":      "VHL",
    "MLH1":     "MLH1",
    "MSH2":     "MSH2",
    "MSH6":     "MSH6",
    "PMS2":     "PMS2",
    "MUTYH":    "MUTYH",
    "POLE":     "POLE",
    "POLD1":    "POLD1",
    "FANCA":    "FANCA",
    "FANCC":    "FANCC",
    "FANCD2":   "FANCD2",
    "FANCE":    "FANCE",
    "FANCF":    "FANCF",
    "FANCG":    "FANCG",
    "FANCI":    "FANCI",
    "FANCL":    "FANCL",
    "FANCM":    "FANCM",
    "C11orf65": "FAAP20",
    "C1orf86":  "FAAP20",
}

print("\n=== Checking known aliases present in data ===")
found_aliases = {}
for old, new in KNOWN_ALIASES.items():
    if old == new:
        continue
    has_old = old in genes
    has_new = new in genes
    if has_old or has_new:
        status = ""
        if has_old and not has_new:
            status = "[USES OLD NAME]"
            found_aliases[old] = new
        elif has_new and not has_old:
            status = "[already new]"
        elif has_old and has_new:
            status = "BOTH present (duplicate?)"
        print(f"  {old} → {new}: old_in_data={has_old}, new_in_data={has_new}  {status}")

# ── Query MyGene.info via GET ──────────────────────────────────────────────
def query_mygene_get(gene_list, batch_size=50):
    """GET https://mygene.info/v3/query?q=GENE1+OR+GENE2&species=human"""
    results = {}
    base = "https://mygene.info/v3/query"
    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i:i + batch_size]
        q = " OR ".join(f'symbol:"{g}"' for g in batch)
        params = {
            "q": q,
            "fields": "symbol,alias",
            "species": "human",
            "size": batch_size * 2,
        }
        try:
            resp = requests.get(base, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", [])
            # Build reverse map: if query symbol matches alias, get official symbol
            for hit in hits:
                sym = hit.get("symbol", "")
                aliases = hit.get("alias", [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                # check which batch gene maps to this hit
                for g in batch:
                    if g == sym:
                        results[g] = sym   # exact match
                    elif g in aliases:
                        results[g] = sym   # g is alias, sym is official
        except Exception as e:
            print(f"  WARNING batch {i}: {e}")
        time.sleep(0.3)
        if (i // batch_size) % 5 == 0:
            print(f"  Progress: {min(i+batch_size, len(gene_list))}/{len(gene_list)}")
    return results

print("\nQuerying MyGene.info (GET, batch=50)...")
symbol_map = query_mygene_get(genes)

# ── Build rename map ───────────────────────────────────────────────────────
rename_from_api = {}
for old, new in symbol_map.items():
    if new and new != old:
        rename_from_api[old] = new

# Merge: hardcoded takes precedence
rename_final = dict(found_aliases)
for old, new in rename_from_api.items():
    if old not in rename_final:
        rename_final[old] = new

print(f"\n=== Summary ===")
print(f"Genes queried via API: {len(symbol_map)}")
print(f"Aliases found (API): {len(rename_from_api)}")
print(f"Aliases found (hardcoded): {len(found_aliases)}")
print(f"Total unique renames needed: {len(rename_final)}")

if rename_final:
    print(f"\nFinal rename mapping ({len(rename_final)}):")
    for old, new in sorted(rename_final.items()):
        src = "(api)" if old in rename_from_api else "(hardcoded)"
        print(f"  {old} → {new}  {src}")

# ── Save ───────────────────────────────────────────────────────────────────
rows = [{"old_symbol": k, "new_symbol": v} for k, v in sorted(rename_final.items())]
pd.DataFrame(rows).to_csv(OUT_DIR / "gene_alias_map.csv", index=False)
print(f"\nSaved → output/processed/gene_alias_map.csv")

# ── Long 11-gene check ─────────────────────────────────────────────────────
LONG_GENES = {
    "BRAF": -0.4885, "PAK7": -0.2618, "PTPRD": -0.2611,
    "PTPRT": 0.2404, "ROS1": -0.2321, "SETD2": -0.2759,
    "TET1": -0.8026, "VHL": -1.0449, "FAM46C": -1.7930,
    "RNF43": 0.7965, "ZFHX3": -0.3822,
}
print("\n=== Long 11-gene signature ===")
for g, coef in LONG_GENES.items():
    in_data = g in genes
    alias_of = rename_final.get(g, "—")
    alias_in_data = alias_of in genes if alias_of != "—" else False
    note = f"[alias {alias_of} in_data={alias_in_data}]" if alias_of != "—" else ""
    print(f"  {g} (coef={coef:+.4f}): in_data={in_data} {note}")
