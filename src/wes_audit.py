"""WES MAF audit and clinical overall-survival alignment.

Per-cohort schema-aware loading covering 13 ICI cohorts in source/input_data/maf_unified/.
Outputs:
  output/processed_wes/audit.csv      - per-cohort schema, sample/gene count, NaN ratio
  output/processed_wes/aligned_n.csv  - variants x clinical sample alignment (% retention)
  output/processed_wes/audit_log.txt  - human-readable summary

Run:
  python src/wes_audit.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAF_DIR = ROOT / "source" / "input_data" / "maf_unified"
CLIN_DIR = ROOT / "source" / "input_data" / "valid"
OUT_DIR = ROOT / "output" / "processed_wes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CohortSpec:
    """Per-cohort variants file schema specification.

    subset_clin_col / subset_clin_value: when present, restrict the cohort to
    samples whose value in clin[subset_clin_col] matches subset_clin_value.
    Used to split a merged cohort file (e.g. CM214_JV101) into sub-cohorts.
    """

    name: str
    file: str
    sample_col: str
    chr_col: str | None
    pos_col: str | None
    ref_col: str | None
    alt_col: str | None
    gene_col: str
    plan_role: str
    plan_n: int
    note: str = ""
    ref_alt_file: str | None = None
    ref_alt_sample_col: str | None = None
    nucmut_col: str | None = None
    subset_clin_col: str | None = None
    subset_clin_value: str | None = None


SPECS: list[CohortSpec] = [
    CohortSpec(
        "Braun", "variants_Braun.csv",
        "Tumor_Sample_Barcode", "Chromosome", "Start_position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "train", 261,
    ),
    CohortSpec(
        "CM214", "variants_CM214_JV101.csv",
        "UUID", "Chromosome", "Start_Position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "holdout_rcc", 182,
        note="CheckMate-214 (Nivo+Ipi) RCC trial; split from CM214_JV101 by clin Cohort==CM-214",
        subset_clin_col="Cohort",
        subset_clin_value="CM-214",
    ),
    CohortSpec(
        "JV101", "variants_CM214_JV101.csv",
        "UUID", "Chromosome", "Start_Position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "train", 283,
        note="JAVELIN-101 (Avelumab+Axitinib, PD-L1+TKI) RCC trial; split from CM214_JV101 by clin Cohort==JAVELIN-101",
        subset_clin_col="Cohort",
        subset_clin_value="JAVELIN-101",
    ),
    CohortSpec(
        "Hugo", "variants_Hugo.csv",
        "Sample", "Chr", "Pos",
        None, None, "Gene",
        "holdout", 37,
        note="ref/alt derived from NucMut split ('REF>ALT')",
        nucmut_col="NucMut",
    ),
    CohortSpec(
        "Liu", "variants_Liu.csv",
        "Patient", "Chromosome", "Start_position",
        None, None, "Hugo_Symbol",
        "train", 144,
        note="ref/alt sourced from variants_Liu_SNP.csv (sample_id matches Patient 1:1)",
        ref_alt_file="variants_Liu_SNP.csv",
        ref_alt_sample_col="sample_id",
    ),
    CohortSpec(
        "Liu_SNP", "variants_Liu_SNP.csv",
        "sample_id", "CHROM", "POS",
        "REF", "ALT", "",
        "train_supp", 0,
        note="SNP coords supplement for Liu",
    ),
    CohortSpec(
        "Miao", "variants_Miao.csv",
        "Tumor_Sample_Barcode", "Chromosome", "Start_position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "train_mixed", 245,
        note="per-sample Cancer_type from clin",
    ),
    CohortSpec(
        "PUSH", "variants_PUSH.csv",
        "Tumor_Sample_Barcode", "Chr", "Start",
        "ref", "alt", "Gene.refGene",
        "train", 92,
        note="docs/data.md mislabeled as panel - actually WES (ANNOVAR-annotated)",
    ),
    CohortSpec(
        "Pleasance", "variants_Pleasance.csv",
        "Tumor_Sample_Barcode", "Chromosome", "Start_Position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "holdout", 76,
    ),
    CohortSpec(
        "Ravi", "variants_Ravi.csv",
        "individual_id", "Chromosome", "Start_position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "train", 306,
        note="individual_id matches clin Sample.ID; pair_id has -T1 suffix",
    ),
    CohortSpec(
        "Riaz", "variants_Riaz.csv",
        "Patient", "Chromosome", "Start",
        None, None, "Hugo.Symbol",
        "train", 68,
        note="ref/alt sourced from variants_Riaz_SNP.csv (external SNP file with Riaz_ prefix applied)",
        ref_alt_file="variants_Riaz_SNP.csv",
        ref_alt_sample_col="sample_id",
    ),
    CohortSpec(
        "SnyderUC", "variants_SnyderUC.csv",
        "Tumor_Sample_Barcode", "Chromosome", "Start_Position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "holdout", 25,
    ),
    CohortSpec(
        "Whijae", "variants_Whijae.csv",
        "Sample", "chrom", "start",
        "ref_allele", "alt_allele", "Hugo_Symbol",
        "holdout", 19,
    ),
    CohortSpec(
        "Gandara", "variants_Gandara.csv",
        "PtID", "chromosome", "position",
        "reference_sequence", "alternate_sequence", "gene_name",
        "excluded_panel", 427,
        note="EXCLUDED: real panel (FoundationOne)",
    ),
    CohortSpec(
        "Hellmann", "variants_Hellmann.csv",
        "Tumor_Sample_Barcode", "Chromosome", "Start_Position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Hugo_Symbol",
        "holdout", 75,
        note="NSCLC anti-PD-1/PD-L1+CTLA-4 combo; OS absent, PFS-only; ORR from RECIST BTR (CR/PR=1)",
    ),
    CohortSpec(
        "Jung", "variants_Jung.csv",
        "Sample.ID", "Chromosome", "Start",
        "Ref", "Alt", "Gene",
        "holdout", 60,
        note="NSCLC anti-PD-1/PD-L1; OS absent, PFS-only; ORR from DCB(=1)/NDB(=0) durable clinical benefit; gene col has ANNOVAR suffix",
    ),
]

CLIN_FILE_MAP = {
    "Braun": "clin_Braun.csv",
    "CM214": "clin_CM214_JV101.csv",
    "JV101": "clin_CM214_JV101.csv",
    "Hugo": "clin_Hugo.csv",
    "Liu": "clin_Liu.csv",
    "Liu_SNP": "clin_Liu.csv",
    "Miao": "clin_Miao.csv",
    "PUSH": "clin_PUSH.csv",
    "Pleasance": "clin_Pleasance.csv",
    "Ravi": "clin_Ravi.csv",
    "Riaz": "clin_Riaz.csv",
    "SnyderUC": "clin_Snyder_UC.csv",
    "Whijae": "clin_Whijae.csv",
    "Gandara": "clin_Gandara.csv",
    "Hellmann": "clin_Hellmann.csv",
    "Jung": "clin_Jung.csv",
}


def load_variants(spec: CohortSpec) -> pd.DataFrame:
    """Read variants file; tolerate dtype mixing."""
    path = MAF_DIR / spec.file
    return pd.read_csv(path, low_memory=False, dtype=str)


def audit_cohort(spec: CohortSpec) -> dict:
    """Return single-cohort audit row."""
    try:
        df = load_variants(spec)
    except Exception as exc:
        return {"cohort": spec.name, "load_error": str(exc)}

    n_rows = len(df)
    samples_in_var = df[spec.sample_col].dropna().unique()
    n_samples_var = len(samples_in_var)

    if spec.gene_col and spec.gene_col in df.columns:
        n_unique_genes = df[spec.gene_col].dropna().nunique()
        median_mut_per_sample = (
            df.groupby(spec.sample_col)[spec.gene_col].count().median()
        )
    else:
        n_unique_genes = 0
        median_mut_per_sample = float("nan")

    def nan_ratio(col: str | None) -> float:
        if col is None or col not in df.columns:
            return 1.0
        return float(df[col].isna().mean())

    chr_nan = nan_ratio(spec.chr_col)
    pos_nan = nan_ratio(spec.pos_col)
    ref_nan = nan_ratio(spec.ref_col)
    alt_nan = nan_ratio(spec.alt_col)

    has_chr_pos_ref_alt = all(
        c is not None and c in df.columns
        for c in [spec.chr_col, spec.pos_col, spec.ref_col, spec.alt_col]
    )

    fixup_status = ""
    if not has_chr_pos_ref_alt:
        if spec.nucmut_col and spec.nucmut_col in df.columns:
            valid = df[spec.nucmut_col].dropna().astype(str)
            split_ok = valid.str.contains(r"^[ACGT]+>[ACGT]+$", regex=True).mean()
            ref_nan = 1.0 - split_ok
            alt_nan = 1.0 - split_ok
            has_chr_pos_ref_alt = split_ok >= 0.95
            fixup_status = f"NucMut split ok ratio={split_ok:.3f}"
        elif spec.ref_alt_file:
            ext_path = MAF_DIR / spec.ref_alt_file
            if ext_path.exists():
                ext = pd.read_csv(ext_path, dtype=str, low_memory=False)
                ext_ids = set(ext[spec.ref_alt_sample_col].dropna().astype(str))
                var_ids = set(map(str, samples_in_var))
                cov = len(ext_ids & var_ids) / max(len(var_ids), 1)
                ref_nan = 1.0 - cov
                alt_nan = 1.0 - cov
                has_chr_pos_ref_alt = cov >= 0.95
                fixup_status = f"{spec.ref_alt_file} sample coverage={cov:.3f}"
            else:
                fixup_status = f"MISSING ref_alt_file {spec.ref_alt_file}"

    clin_path = CLIN_DIR / CLIN_FILE_MAP[spec.name]
    if clin_path.exists():
        clin = pd.read_csv(clin_path)
        clin_samples = set(clin["Sample.ID"].dropna().astype(str))
        var_samples = set(map(str, samples_in_var))
        intersect = var_samples & clin_samples
        n_aligned = len(intersect)
        n_clin = len(clin_samples)
        retention_var = n_aligned / max(n_samples_var, 1)
        retention_clin = n_aligned / max(n_clin, 1)
    else:
        n_aligned = -1
        n_clin = -1
        retention_var = float("nan")
        retention_clin = float("nan")

    plan_n = spec.plan_n if spec.plan_n > 0 else n_clin
    plan_retention = (
        n_aligned / plan_n if (plan_n > 0 and n_aligned >= 0) else float("nan")
    )
    pass_80pct = plan_retention >= 0.80 if plan_retention == plan_retention else False

    return {
        "cohort": spec.name,
        "role": spec.plan_role,
        "plan_n": spec.plan_n,
        "var_rows": n_rows,
        "var_samples": n_samples_var,
        "clin_samples": n_clin,
        "aligned_n": n_aligned,
        "retention_var": round(retention_var, 4) if retention_var == retention_var else None,
        "retention_clin": round(retention_clin, 4) if retention_clin == retention_clin else None,
        "plan_retention": round(plan_retention, 4) if plan_retention == plan_retention else None,
        "pass_80pct": pass_80pct,
        "unique_genes": n_unique_genes,
        "median_mut_per_sample": (
            round(median_mut_per_sample, 1)
            if median_mut_per_sample == median_mut_per_sample
            else None
        ),
        "has_chr_pos_ref_alt": has_chr_pos_ref_alt,
        "chr_nan": round(chr_nan, 3),
        "pos_nan": round(pos_nan, 3),
        "ref_nan": round(ref_nan, 3),
        "alt_nan": round(alt_nan, 3),
        "sample_col": spec.sample_col,
        "chr_col": spec.chr_col,
        "pos_col": spec.pos_col,
        "ref_col": spec.ref_col,
        "alt_col": spec.alt_col,
        "gene_col": spec.gene_col,
        "note": spec.note,
        "fixup_status": fixup_status,
    }


def main() -> int:
    rows = [audit_cohort(s) for s in SPECS]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "audit.csv", index=False)

    align_cols = [
        "cohort", "role", "plan_n", "var_samples", "clin_samples",
        "aligned_n", "plan_retention", "pass_80pct", "note",
    ]
    df[align_cols].to_csv(OUT_DIR / "aligned_n.csv", index=False)

    log = []
    log.append("=" * 78)
    log.append("WES MAF Audit -- Phase 0 of WES Retraining Plan")
    log.append("=" * 78)
    log.append("")
    log.append(f"Cohorts audited: {len(df)}")
    log.append(f"  - train pool:      {(df['role'] == 'train').sum()}")
    log.append(f"  - train (mixed):   {(df['role'] == 'train_mixed').sum()}")
    log.append(f"  - train supp:      {(df['role'] == 'train_supp').sum()}")
    log.append(f"  - holdout:         {(df['role'] == 'holdout').sum()}")
    log.append(f"  - excluded:        {(df['role'] == 'excluded_panel').sum()}")
    log.append("")
    log.append("Schema completeness (chr/pos/ref/alt all present):")
    for _, r in df.iterrows():
        ok = "OK" if r["has_chr_pos_ref_alt"] else "WARN"
        log.append(
            f"  [{ok:4s}] {r['cohort']:13s} "
            f"chr_nan={r['chr_nan']:.3f} pos_nan={r['pos_nan']:.3f} "
            f"ref_nan={r['ref_nan']:.3f} alt_nan={r['alt_nan']:.3f}"
        )
    log.append("")
    log.append("Sample alignment (variants x clinical):")
    for _, r in df.iterrows():
        ok = "OK" if r["pass_80pct"] else "STOP"
        log.append(
            f"  [{ok:4s}] {r['cohort']:13s} "
            f"plan={r['plan_n']:>4d} var={r['var_samples']:>4d} "
            f"clin={r['clin_samples']:>4d} aligned={r['aligned_n']:>4d} "
            f"retention={r['plan_retention']}"
        )
    log.append("")
    log.append("Mutation density (median mut/sample, unique genes):")
    for _, r in df.iterrows():
        log.append(
            f"  {r['cohort']:13s} "
            f"unique_genes={r['unique_genes']:>6d} "
            f"median_mut={r['median_mut_per_sample']}"
        )
    log.append("")
    failed = df[~df["pass_80pct"] & df["role"].isin(["train", "train_mixed", "holdout"])]
    if len(failed) > 0:
        log.append("STOP RULE TRIGGERED:")
        for _, r in failed.iterrows():
            log.append(
                f"  {r['cohort']}: aligned={r['aligned_n']} / plan={r['plan_n']} "
                f"= {r['plan_retention']} (< 0.80)"
            )
    else:
        log.append("Stop rule check: all train/holdout cohorts >= 80%.")
    log.append("")
    log.append("Outputs:")
    log.append(f"  {OUT_DIR / 'audit.csv'}")
    log.append(f"  {OUT_DIR / 'aligned_n.csv'}")
    log.append(f"  {OUT_DIR / 'audit_log.txt'}")

    log_text = "\n".join(log)
    (OUT_DIR / "audit_log.txt").write_text(log_text, encoding="utf-8")
    print(log_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
