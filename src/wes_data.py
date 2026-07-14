"""WES data preprocessing for pan-cancer ICI survival retraining.

Loads 11 cohorts (7 train + 4 holdout) of unified MAF/SNP data and produces
mut/mask/clin matrices over a unified WES gene set (intersection of WES union
and KG gene union).

Outputs (output/processed_wes/):
  wes_candidate_genes.csv      - one column "gene" with selected genes
  train_wes_mut.csv            - [n_train, n_genes] binary
  train_wes_mask.csv           - [n_train, n_genes] all 1 (WES coverage)
  train_wes_clin.csv           - [n_train, OS_MONTHS + event + COV_COLS]
  train_wes_meta.csv           - [n_train, sample_id + cohort + cancer_type]
  valid_<cohort>_wes_*.csv     - same per-holdout cohort
  wes_data_summary.json        - manifest

Usage:
  python src/wes_data.py --audit                # quick check, no writes
  python src/wes_data.py --build                # full pipeline write outputs
  python src/wes_data.py --build --skip-fmb     # skip KG intersection (use full WES union)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import NON_SYN, parse_valid_clin
from wes_audit import SPECS as AUDIT_SPECS

from kg_features import GENE_ALIAS

# --------------------------------------------------------------------------- #
# Gene name canonicalization
# --------------------------------------------------------------------------- #
#
# Three sources of inconsistency between MAF cohorts and KGs:
#   1. Excel auto-conversion. Genes named MARCH1-12, SEPT1-15, DEC1 get turned
#      into dates (e.g. MARCH1 -> "1-MAR" -> Excel serial 46082). The maf_unified
#      files have been cleaned to leave the raw integer (46082); we decode it
#      back to the gene via Excel epoch (1899-12-30).
#   2. ANNOVAR transcript suffix in some Pleasance/PUSH rows
#      (e.g. "ACSL5(UC001KZS.3:EXON8:C.712-1G>A)" -> "ACSL5").
#   3. HGNC renaming. KGs use a mix of pre-2020 names (FAM46C/MARCH1/SEPT1/DEC1)
#      and current names (TENT5C/MARCHF1/SEPTIN1/BHLHE40). normalize maps to the
#      *current* HGNC symbol so the same gene appears once in any union/intersect.

import re as _re
from datetime import date as _date, timedelta as _timedelta

_DATE_PAT = _re.compile(
    r"^\d{1,2}-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)$", _re.I
)
_DIGIT_PAT = _re.compile(r"^\d+$")
_EXCEL_EPOCH = _date(1899, 12, 30)

# HGNC since 2020: MARCH1-12 -> MARCHF1-12, SEPT1-15 -> SEPTIN1-15, DEC1 -> BHLHE40.
# Most KGs have migrated; we add reverse migration so old-named WES tokens map up.
HGNC_MIGRATION: dict[str, str] = {
    **{f"MARCH{i}": f"MARCHF{i}" for i in range(1, 13)},
    **{f"SEPT{i}": f"SEPTIN{i}" for i in range(1, 16)},
    "DEC1": "BHLHE40",
}


def _decode_excel_serial_to_gene(s: str) -> str | None:
    """Excel serial integer (e.g. 46082) -> 'MARCH1' / 'SEPT3' / 'DEC1' if it
    decodes to a valid Mar/Sep/Dec date matching a real gene index. Returns
    None for integers outside the gene-affecting range.
    """
    if not _DIGIT_PAT.match(s):
        return None
    try:
        n = int(s)
    except (ValueError, OverflowError):
        return None
    if not (40000 <= n <= 50000):
        return None
    try:
        d = _EXCEL_EPOCH + _timedelta(days=n)
    except OverflowError:
        return None
    m, day = d.month, d.day
    if m == 3 and 1 <= day <= 12:
        return f"MARCH{day}"
    if m == 9 and 1 <= day <= 15:
        return f"SEPT{day}"
    if m == 12 and day == 1:
        return "DEC1"
    return None


def normalize_gene_name(name) -> str | None:
    """Canonicalize a gene symbol to the current HGNC name.

    Steps:
      1. Strip + uppercase + drop NaN/empty/sentinel.
      2. Strip ANNOVAR transcript suffix (`GENE(UC...)` -> `GENE`,
         and colon form `GENE:NM_...:EXON...` -> `GENE`).
      3. Drop Excel-residual date tokens (`4-MAR`).
      4. Decode Excel serial integers (46082 -> MARCH1).
      5. Apply HGNC migration (MARCH1 -> MARCHF1, SEPT1 -> SEPTIN1, DEC1 -> BHLHE40).
      6. Apply panel alias map (FAM46C -> TENT5C etc.).

    Returns None when the input is not a usable gene symbol.
    """
    if name is None:
        return None
    s = str(name).strip().upper()
    if not s or s in {"NAN", "NONE", "."}:
        return None
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    # ANNOVAR colon form: GENE:NM_xxxx:EXONn:c.X:p.Y[,GENE:...]. HGNC symbols
    # never contain ':', so the gene is always the token before the first colon.
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    # Defensive: drop any trailing comma-joined remainder (multi-transcript rows).
    if "," in s:
        s = s.split(",", 1)[0].strip()
    if _DATE_PAT.match(s):
        return None
    decoded = _decode_excel_serial_to_gene(s)
    if decoded is not None:
        s = decoded
    if s in HGNC_MIGRATION:
        s = HGNC_MIGRATION[s]
    if s in GENE_ALIAS:
        s = GENE_ALIAS[s]
    return s if s else None


ROOT = Path(__file__).resolve().parents[1]
MAF_DIR = ROOT / "source" / "input_data" / "maf_unified"
CLIN_DIR = ROOT / "source" / "input_data" / "valid"
SUBKG_DIR = ROOT / "output" / "subkg"
OUT_DIR = ROOT / "output" / "processed_wes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_KGS = ["primekg", "hetionet", "drkg", "ibkh", "monarch", "ogb_biokg", "openbiolink"]
# Cohort split used for WES training and external evaluation.
TRAIN_COHORTS = ["Liu", "Riaz", "Miao", "Ravi", "JV101", "PUSH"]
HOLDOUT_COHORTS = ["Whijae", "Hugo", "SnyderUC", "Pleasance"]
HOLDOUT_RCC_COHORTS = ["CM214", "Braun"]
ALL_HOLDOUT_COHORTS = HOLDOUT_COHORTS + HOLDOUT_RCC_COHORTS

CLIN_FILE_MAP = {
    "Braun": "clin_Braun.csv",
    "CM214": "clin_CM214_JV101.csv",
    "JV101": "clin_CM214_JV101.csv",
    "Hugo": "clin_Hugo.csv",
    "Liu": "clin_Liu.csv",
    "Miao": "clin_Miao.csv",
    "PUSH": "clin_PUSH.csv",
    "Pleasance": "clin_Pleasance.csv",
    "Ravi": "clin_Ravi.csv",
    "Riaz": "clin_Riaz.csv",
    "SnyderUC": "clin_Snyder_UC.csv",
    "Whijae": "clin_Whijae.csv",
    "Hellmann": "clin_Hellmann.csv",
    "Jung": "clin_Jung.csv",
}

SPECS_BY_NAME = {s.name: s for s in AUDIT_SPECS}


# --------------------------------------------------------------------------- #
# Per-cohort variant loaders with fixup
# --------------------------------------------------------------------------- #


def _split_nucmut(value: str) -> tuple[str, str] | tuple[None, None]:
    """Split combined 'REF>ALT' field. Returns (ref, alt) or (None, None)."""
    if not isinstance(value, str) or ">" not in value:
        return (None, None)
    parts = value.split(">")
    if len(parts) != 2:
        return (None, None)
    return (parts[0].strip(), parts[1].strip())


def load_variants_normalized(cohort: str) -> pd.DataFrame:
    """Load and normalize a cohort's variants to uniform columns.

    Returns DataFrame with columns: sample_id, gene, chrom, pos, ref, alt, var_class
    (var_class is Variant_Classification when available, else None.)
    """
    spec = SPECS_BY_NAME[cohort]
    df = pd.read_csv(MAF_DIR / spec.file, dtype=str, low_memory=False)

    out = pd.DataFrame()
    out["sample_id"] = df[spec.sample_col].astype(str)
    out["gene"] = df[spec.gene_col].astype(str) if spec.gene_col and spec.gene_col in df.columns else ""
    out["chrom"] = df[spec.chr_col].astype(str) if spec.chr_col else None
    out["pos"] = df[spec.pos_col].astype(str) if spec.pos_col else None

    if spec.ref_col and spec.alt_col and spec.ref_col in df.columns:
        out["ref"] = df[spec.ref_col]
        out["alt"] = df[spec.alt_col]
    elif spec.nucmut_col and spec.nucmut_col in df.columns:
        ref_alt = df[spec.nucmut_col].apply(_split_nucmut)
        out["ref"] = [r for r, _ in ref_alt]
        out["alt"] = [a for _, a in ref_alt]
    elif spec.ref_alt_file:
        ext = pd.read_csv(MAF_DIR / spec.ref_alt_file, dtype=str, low_memory=False)
        ext_key = spec.ref_alt_sample_col
        merge_cols = ["chrom", "pos", "ref", "alt"]
        ext_norm = pd.DataFrame({
            "sample_id": ext[ext_key].astype(str),
            "chrom": ext["CHROM"].astype(str),
            "pos": ext["POS"].astype(str),
            "ref": ext["REF"].astype(str),
            "alt": ext["ALT"].astype(str),
        })
        # Liu/Riaz: SNP file is THE source for ref/alt - use it directly,
        # but also need to keep gene info from the main variants file.
        # Strategy: the SNP file is per-position, gene info comes from main file via chr/pos.
        if spec.gene_col and spec.gene_col in df.columns:
            main_pos = pd.DataFrame({
                "sample_id": out["sample_id"],
                "chrom": out["chrom"],
                "pos": out["pos"],
                "gene": out["gene"],
            })
            merged = ext_norm.merge(
                main_pos, on=["sample_id", "chrom", "pos"], how="left"
            )
            out = merged[["sample_id", "gene", "chrom", "pos", "ref", "alt"]].copy()
        else:
            out = ext_norm[["sample_id", "chrom", "pos", "ref", "alt"]].copy()
            out["gene"] = ""
    else:
        out["ref"] = None
        out["alt"] = None

    if "Variant_Classification" in df.columns and len(out) == len(df):
        out["var_class"] = df["Variant_Classification"]
    else:
        out["var_class"] = None

    if "ref" in out.columns:
        out = out.dropna(subset=["ref", "alt"], how="any")
    out = out[out["sample_id"].notna() & (out["sample_id"] != "nan")]

    # Subset filter: split a merged cohort file by clin column value
    if spec.subset_clin_col and spec.subset_clin_value:
        clin_path = CLIN_DIR / CLIN_FILE_MAP[cohort]
        clin_full = pd.read_csv(clin_path)
        keep_ids = set(
            clin_full.loc[
                clin_full[spec.subset_clin_col].astype(str) == spec.subset_clin_value,
                "Sample.ID",
            ].astype(str)
        )
        out = out[out["sample_id"].astype(str).isin(keep_ids)]
    return out


# --------------------------------------------------------------------------- #
# Gene set
# --------------------------------------------------------------------------- #


def collect_wes_genes(cohorts: list[str]) -> set[str]:
    """Union of unique Hugo symbols across cohorts (canonicalized)."""
    genes: set[str] = set()
    for c in cohorts:
        df = load_variants_normalized(c)
        for raw in df["gene"].dropna():
            n = normalize_gene_name(raw)
            if n:
                genes.add(n)
    return genes


def collect_kg_genes(kgs: list[str] | None = None) -> set[str]:
    """Union of gene/protein-typed nodes across KG subgraphs (canonicalized)."""
    kgs = kgs or ALL_KGS
    genes: set[str] = set()
    for kg in kgs:
        p = SUBKG_DIR / f"subkg_{kg}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if "x_type" not in df.columns or "y_type" not in df.columns:
            continue
        x_mask = df["x_type"].astype(str).str.contains("gene|protein", case=False, na=False)
        y_mask = df["y_type"].astype(str).str.contains("gene|protein", case=False, na=False)
        for raw in df.loc[x_mask, "x_name"].dropna():
            n = normalize_gene_name(raw)
            if n:
                genes.add(n)
        for raw in df.loc[y_mask, "y_name"].dropna():
            n = normalize_gene_name(raw)
            if n:
                genes.add(n)
    return genes


def build_wes_candidate_genes(train_cohorts: list[str], use_kg_intersect: bool = True) -> list[str]:
    """Build candidate WES gene list = WES union (optionally ∩ KG union)."""
    wes = collect_wes_genes(train_cohorts)
    if use_kg_intersect:
        kg = collect_kg_genes()
        chosen = wes & kg
        print(f"  WES union: {len(wes)} | KG union: {len(kg)} | intersect: {len(chosen)}")
    else:
        chosen = wes
        print(f"  WES union (no KG intersect): {len(chosen)}")
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Binary matrix construction
# --------------------------------------------------------------------------- #


def maf_to_binary_wes(maf: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Variants DataFrame -> binary [samples, genes] matrix.

    Applies normalize_gene_name (Excel + alias) on the gene column before
    matching candidate genes. Filters by NON_SYN if var_class is populated.
    """
    df = maf.copy()
    df["gene_norm"] = df["gene"].apply(normalize_gene_name)
    df = df.dropna(subset=["gene_norm"])
    df = df[df["gene_norm"].isin(set(genes))]
    if "var_class" in df.columns and df["var_class"].notna().any():
        df = df[df["var_class"].isna() | df["var_class"].isin(NON_SYN)]
    if len(df) == 0:
        return pd.DataFrame(0, index=[], columns=genes, dtype=int)
    df["mutated"] = 1
    mat = (
        df.groupby(["sample_id", "gene_norm"])["mutated"].max().unstack(fill_value=0)
    )
    return mat.reindex(columns=genes, fill_value=0)


def load_cohort_clin(cohort: str) -> pd.DataFrame:
    """Reuse preprocess.parse_valid_clin for OS/event + 15 cov cols.

    For split cohorts (CM214/JV101), filter by clin Cohort col after parsing.
    """
    clin_path = CLIN_DIR / CLIN_FILE_MAP[cohort]
    full = parse_valid_clin(clin_path)
    spec = SPECS_BY_NAME[cohort]
    if spec.subset_clin_col and spec.subset_clin_value:
        clin_raw = pd.read_csv(clin_path).set_index("Sample.ID")
        keep_ids = clin_raw.loc[
            clin_raw[spec.subset_clin_col].astype(str) == spec.subset_clin_value
        ].index
        full = full.loc[full.index.intersection(keep_ids)]
    return full


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def build_split(cohorts: list[str], genes: list[str], split_label: str) -> dict:
    """Build mut/mask/clin/meta for a list of cohorts (concatenated by row).

    Returns dict with binary mut DataFrame, all-ones mask, aligned clinical, and per-sample meta.
    """
    mut_parts, clin_parts, meta_parts = [], [], []
    cohort_stats = {}
    for c in cohorts:
        maf = load_variants_normalized(c)
        cohort_clin = load_cohort_clin(c)
        binary = maf_to_binary_wes(maf, genes)
        common = binary.index.intersection(cohort_clin.index)
        if len(common) == 0:
            cohort_stats[c] = {"n_variants": int(len(maf)), "n_aligned": 0}
            continue
        binary = binary.loc[common]
        cohort_clin = cohort_clin.loc[common]
        meta = pd.DataFrame({
            "sample_id": common,
            "cohort": c,
            "cancer_type": _resolve_cancer_type(c, cohort_clin),
        }, index=common)
        mut_parts.append(binary)
        clin_parts.append(cohort_clin)
        meta_parts.append(meta)
        cohort_stats[c] = {
            "n_variants": int(len(maf)),
            "n_aligned": int(len(common)),
            "events": int(cohort_clin["event"].sum()),
            "median_mut": float(binary.sum(axis=1).median()),
        }
        print(f"  [{split_label}/{c}] aligned={len(common)} events={cohort_clin['event'].sum()} median_mut={binary.sum(axis=1).median():.0f}")

    if not mut_parts:
        return {"mut": None, "clin": None, "meta": None, "stats": cohort_stats}

    mut = pd.concat(mut_parts, axis=0)
    clin = pd.concat(clin_parts, axis=0)
    meta = pd.concat(meta_parts, axis=0)
    mask = pd.DataFrame(1, index=mut.index, columns=mut.columns, dtype=int)
    return {"mut": mut, "mask": mask, "clin": clin, "meta": meta, "stats": cohort_stats}


def _resolve_cancer_type(cohort: str, clin: pd.DataFrame) -> pd.Series:
    """Pull per-sample cancer type from clinical (default Mixed for Miao/Pleasance)."""
    raw_path = CLIN_DIR / CLIN_FILE_MAP[cohort]
    raw = pd.read_csv(raw_path)
    raw = raw.set_index(raw.columns[0])
    if "Cancer_type" in raw.columns:
        return raw["Cancer_type"].reindex(clin.index).fillna("Unknown")
    return pd.Series(["Unknown"] * len(clin), index=clin.index)


def write_split(name: str, data: dict) -> None:
    if data["mut"] is None:
        print(f"  [skip] {name}: empty after alignment")
        return
    # int8 cast: binary 0/1 only, avoids pandas int64 word_len bug on wide matrices
    data["mut"].astype("int8").to_csv(OUT_DIR / f"{name}_wes_mut.csv")
    data["mask"].astype("int8").to_csv(OUT_DIR / f"{name}_wes_mask.csv")
    data["clin"].to_csv(OUT_DIR / f"{name}_wes_clin.csv")
    data["meta"].to_csv(OUT_DIR / f"{name}_wes_meta.csv")
    print(
        f"  [wrote] {name}: mut={data['mut'].shape} clin={data['clin'].shape}"
    )


def build_pipeline(use_kg_intersect: bool = True) -> dict:
    print(f"\n[1/3] Building WES candidate gene set (KG intersect={use_kg_intersect}) ...")
    genes = build_wes_candidate_genes(TRAIN_COHORTS, use_kg_intersect=use_kg_intersect)
    pd.DataFrame({"gene": genes}).to_csv(OUT_DIR / "wes_candidate_genes.csv", index=False)
    print(f"  saved {len(genes)} genes -> wes_candidate_genes.csv")

    print(f"\n[2/3] Building train pool ({len(TRAIN_COHORTS)} cohorts) ...")
    train = build_split(TRAIN_COHORTS, genes, "train")
    write_split("train", train)

    print(f"\n[3/3] Building holdout cohorts (primary={len(HOLDOUT_COHORTS)}, RCC arm={len(HOLDOUT_RCC_COHORTS)}) ...")
    holdout: dict[str, dict] = {}
    for c in ALL_HOLDOUT_COHORTS:
        h = build_split([c], genes, c)
        write_split(f"valid_{c}", h)
        holdout[c] = h["stats"].get(c, {})

    summary = {
        "n_genes": len(genes),
        "use_kg_intersect": use_kg_intersect,
        "train_cohorts": TRAIN_COHORTS,
        "holdout_cohorts": HOLDOUT_COHORTS,
        "holdout_rcc_cohorts": HOLDOUT_RCC_COHORTS,
        "train_stats": train["stats"],
        "holdout_stats": holdout,
        "train_total_n": int(train["mut"].shape[0]) if train["mut"] is not None else 0,
        "train_total_events": int(train["clin"]["event"].sum()) if train["clin"] is not None else 0,
    }
    (OUT_DIR / "wes_data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSummary -> {OUT_DIR / 'wes_data_summary.json'}")
    return summary


def quick_audit() -> None:
    """Sanity check: load each cohort, show variant + clin alignment."""
    print(f"\n=== WES quick audit ===\n")
    for c in TRAIN_COHORTS + ALL_HOLDOUT_COHORTS:
        try:
            maf = load_variants_normalized(c)
            clin = load_cohort_clin(c)
            samples = set(maf["sample_id"].astype(str))
            aligned = samples & set(clin.index.astype(str))
            print(f"  {c:13s} variants={len(maf):>7d} samples={len(samples):>4d} clin={len(clin):>4d} aligned={len(aligned):>4d}")
        except Exception as exc:
            print(f"  {c:13s} ERROR: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="WES preprocessing pipeline")
    parser.add_argument("--audit", action="store_true", help="quick alignment check, no writes")
    parser.add_argument("--build", action="store_true", help="full build")
    parser.add_argument("--skip-kg-intersect", action="store_true",
                        help="use full WES union without KG intersection")
    args = parser.parse_args()
    if args.audit:
        quick_audit()
    if args.build:
        build_pipeline(use_kg_intersect=not args.skip_kg_intersect)
    if not (args.audit or args.build):
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
