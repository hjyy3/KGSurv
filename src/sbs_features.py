"""Panel-friendly mutational signature-like feature extraction.

Extract 10-D panel-validated features per sample for ICI survival analysis.

Two-tier strategy by cohort data quality:
    Tier-0 (10-D, full): MAF with chr/pos/ref/alt
        → APOBEC enrichment (Roberts 2013 TCW>T/G)
        → Smoking C>A outside CpG (SBS4)
        → Aging C>T at CpG (SBS1)
        → MMR_indel_burden (Davies 2017)
        → POLE hotspot (P286R / V411L; Rayner 2016)
        → HRD score (BRCA1/2 + ATM + PALB2 + RAD51X; Telli 2016)
        → DDR_burden (12-gene panel; Teo 2018)
        → TMB_log
        → dNdS_like (non-syn / silent; Martincorena 2017)

    Tier-2 (4-D, gene-only): Sample x gene wt/mut matrix (Mariathasan / Ravi / Gandara
        if no MAF available; Mariathasan in particular falls back here even though
        the unified file exists, because its iM210 file mixed in CNV calls).
        → POLE_hotspot (any POLE mut), HRD_score, DDR_burden, TMB_log

Output: output/sbs_features/{train, valid_<cohort>}_sigfeats.csv
        Index = processed cohort sample IDs (after intersection with raw)
        Columns = 10 fixed feature names (lower tiers fill gaps with 0.0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx

ROOT = Path(__file__).resolve().parents[1]
GENOME_PATH = ROOT / "reference" / "hg19.fa"
PROC_DIR = ROOT / "output" / "processed"
OUT_DIR = ROOT / "output" / "sbs_features"
SRC_VALID = ROOT / "source" / "input_data" / "valid"
SRC_TRAIN = ROOT / "source" / "input_data" / "train"

FEATURES = [
    "APOBEC_score", "APOBEC_count", "Smoking_CtoA", "Aging_CtoT_CpG",
    "MMR_indel_burden", "POLE_hotspot", "HRD_score", "DDR_burden",
    "TMB_log", "dNdS_like",
]
N_FEAT = len(FEATURES)

HRD_GENES = {"BRCA1", "BRCA2", "ATM", "PALB2", "RAD51B", "RAD51C", "RAD51D"}
DDR_GENES = {
    "BRCA1", "BRCA2", "ATM", "ATR", "PALB2", "CHEK1", "CHEK2",
    "RAD51", "RAD51B", "RAD51C", "RAD51D", "MRE11", "NBN",
    "FANCA", "FANCD2", "POLE", "POLD1",
}
NON_SYN = {
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Ins", "Frame_Shift_Del",
    "Splice_Site", "In_Frame_Ins", "In_Frame_Del",
    "Translation_Start_Site", "Nonstop_Mutation",
}
SILENT = {"Silent", "Synonymous", "synonymous_variant", "synonymous SNV"}

COMP = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def get_trinucleotide(genome, chrom, pos: int, ref: str, alt: str) -> str | None:
    """Return pyrimidine-centered trinucleotide context, e.g. 'A[C>T]G'.
    Returns None for indels or if context lookup fails / ref mismatches genome.
    """
    if not isinstance(ref, str) or not isinstance(alt, str):
        return None
    if len(ref) != 1 or len(alt) != 1 or ref == "-" or alt == "-":
        return None
    if ref not in "ACGT" or alt not in "ACGT" or ref == alt:
        return None
    chrom_s = str(chrom)
    if chrom_s.startswith("chr"):
        pass
    else:
        chrom_s = "chr" + chrom_s
    chrom_s = {"chr23": "chrX", "chr24": "chrY", "chrMT": "chrM"}.get(chrom_s, chrom_s)
    if chrom_s not in genome:
        return None
    try:
        seq = str(genome[chrom_s][pos - 2:pos + 1]).upper()
    except (KeyError, ValueError):
        return None
    if len(seq) != 3 or seq[1] != ref.upper():
        return None
    if ref in "AG":
        seq = revcomp(seq)
        ref = ref.translate(COMP)
        alt = alt.translate(COMP)
    return f"{seq[0]}[{ref}>{alt}]{seq[2]}"


def is_apobec(tri: str) -> bool:
    """TCW>T/G (W=A/T), pyrimidine-strand."""
    return tri[0] == "T" and tri[2] == "C" and tri[6] in "AT" and tri[4] in ("T", "G")


def is_tcw_context(tri: str) -> bool:
    return tri[0] == "T" and tri[2] == "C" and tri[6] in "AT"


def is_smoking_CtoA(tri: str) -> bool:
    """C>A outside CpG (SBS4 hallmark)."""
    return tri[2] == "C" and tri[4] == "A" and tri[6] != "G"


def is_aging_CtoT_CpG(tri: str) -> bool:
    """C>T at CpG (SBS1 hallmark)."""
    return tri[2] == "C" and tri[4] == "T" and tri[6] == "G"


# ── Feature computation ──────────────────────────────────────────────────────

def compute_full(df: pd.DataFrame, genome) -> pd.DataFrame:
    """Tier-0: 10-D from MAF with chr/pos/ref/alt."""
    # Some cohorts (Hugo / Riaz / PUSH) only annotate non-synonymous variants —
    # in those files Silent count is 0 and dNdS_like becomes a meaningless
    # cohort-level constant. Detect and disable dNdS_like for such cohorts.
    has_silent = False
    if "var_class" in df.columns:
        has_silent = bool((df["var_class"] == "Silent").any())
    feats = {}
    for sample, sub in df.groupby("sample_id"):
        n = len(sub)
        if "var_type" in sub.columns:
            n_snv = (sub["var_type"] == "SNP").sum()
            n_indel = sub["var_type"].isin(["INS", "DEL"]).sum()
        else:
            is_snv = sub["ref"].astype(str).str.len().eq(1) & sub["alt"].astype(str).str.len().eq(1)
            n_snv = int(is_snv.sum())
            n_indel = n - n_snv
        tris = []
        chroms = sub["chrom"].tolist()
        positions = sub["pos"].tolist()
        refs = sub["ref"].tolist()
        alts = sub["alt"].tolist()
        for chrom, pos, ref, alt in zip(chroms, positions, refs, alts):
            try:
                pos_i = int(pos)
            except (TypeError, ValueError):
                continue
            tri = get_trinucleotide(genome, chrom, pos_i, str(ref), str(alt))
            if tri:
                tris.append(tri)
        n_tri = max(len(tris), 1)
        n_tcw = sum(1 for t in tris if is_tcw_context(t))
        n_apobec = sum(1 for t in tris if is_apobec(t))
        n_ca_nonCpG = sum(1 for t in tris if is_smoking_CtoA(t))
        n_ct_CpG = sum(1 for t in tris if is_aging_CtoT_CpG(t))
        # APOBEC enrichment-like: log((n_apobec+0.5) / sqrt(n_tcw+1))
        apobec_score = float(np.log1p(n_apobec) - 0.5 * np.log1p(n_tcw))
        # Gene-level features
        genes_set = set(sub["gene"].dropna().astype(str)) if "gene" in sub else set()
        pole_mut = 0
        if "hgvsp" in sub.columns and "gene" in sub.columns:
            mask = (sub["gene"] == "POLE") & sub["hgvsp"].fillna("").astype(str).str.contains(
                r"P286R|V411L", regex=True)
            pole_mut = int(mask.any())
        # dNdS_like — only meaningful when both Silent and non-syn are
        # consistently annotated. Clip to [0, 10] to bound outliers.
        if has_silent and "var_class" in sub.columns:
            n_silent = int((sub["var_class"] == "Silent").sum())
            n_nonsyn = int(sub["var_class"].isin(NON_SYN).sum())
            dnds = float(n_nonsyn) / max(n_silent + 1, 1)
            dnds = float(np.clip(dnds, 0.0, 10.0))
        else:
            dnds = 0.0
        feats[sample] = {
            "APOBEC_score": apobec_score,
            "APOBEC_count": float(n_apobec),
            "Smoking_CtoA": float(n_ca_nonCpG) / n_tri,
            "Aging_CtoT_CpG": float(n_ct_CpG) / n_tri,
            "MMR_indel_burden": float(n_indel),
            "POLE_hotspot": float(pole_mut),
            "HRD_score": float(int(bool(genes_set & HRD_GENES))),
            "DDR_burden": float(len(genes_set & DDR_GENES)),
            "TMB_log": float(np.log1p(n_snv)),
            "dNdS_like": dnds,
        }
    return pd.DataFrame(feats).T.reindex(columns=FEATURES)


def compute_class_only(df: pd.DataFrame) -> pd.DataFrame:
    """Tier-1: 6-D from MAF with Variant_Class but no chr/pos."""
    feats = {}
    for sample, sub in df.groupby("sample_id"):
        n = len(sub)
        if "var_class" in sub.columns:
            n_silent = int(sub["var_class"].isin(SILENT).sum())
            n_nonsyn = int(sub["var_class"].isin(NON_SYN).sum())
        else:
            n_silent, n_nonsyn = 0, n
        if "var_type" in sub.columns:
            n_indel = int(sub["var_type"].isin(["INS", "DEL"]).sum())
        else:
            n_indel = 0
        genes_set = set(sub["gene"].dropna().astype(str)) if "gene" in sub else set()
        pole_mut = 0
        if "hgvsp" in sub.columns and "gene" in sub.columns:
            mask = (sub["gene"] == "POLE") & sub["hgvsp"].fillna("").astype(str).str.contains(
                r"P286R|V411L", regex=True)
            pole_mut = int(mask.any())
        feats[sample] = {
            "APOBEC_score": 0.0, "APOBEC_count": 0.0,
            "Smoking_CtoA": 0.0, "Aging_CtoT_CpG": 0.0,
            "MMR_indel_burden": float(n_indel),
            "POLE_hotspot": float(pole_mut),
            "HRD_score": float(int(bool(genes_set & HRD_GENES))),
            "DDR_burden": float(len(genes_set & DDR_GENES)),
            "TMB_log": float(np.log1p(n_nonsyn + n_silent)),
            "dNdS_like": float(n_nonsyn) / max(n_silent + 1, 1),
        }
    return pd.DataFrame(feats).T.reindex(columns=FEATURES)


def compute_matrix(mut_df: pd.DataFrame) -> pd.DataFrame:
    """Tier-2: 4-D from sample x gene wt/mut matrix.

    POLE_hotspot inferred only from gene-level POLE mut status (no codon info).
    dNdS_like / trinucleotide / indel features unavailable -> 0.0.
    """
    if mut_df.dtypes.iloc[0] == object:
        mat = (mut_df == "mut").astype(int)
    else:
        mat = mut_df.astype(int)
    available = set(mat.columns)
    hrd_cols = [g for g in HRD_GENES if g in available]
    ddr_cols = [g for g in DDR_GENES if g in available]
    feats = {}
    for sample, row in mat.iterrows():
        n_mut = int(row.sum())
        hrd = int(row.reindex(hrd_cols, fill_value=0).sum() > 0) if hrd_cols else 0
        ddr_burden = int(row.reindex(ddr_cols, fill_value=0).sum())
        pole = int(row.get("POLE", 0)) if "POLE" in available else 0
        feats[sample] = {
            "APOBEC_score": 0.0, "APOBEC_count": 0.0,
            "Smoking_CtoA": 0.0, "Aging_CtoT_CpG": 0.0,
            "MMR_indel_burden": 0.0,
            "POLE_hotspot": float(pole),
            "HRD_score": float(hrd),
            "DDR_burden": float(ddr_burden),
            "TMB_log": float(np.log1p(n_mut)),
            "dNdS_like": 0.0,
        }
    return pd.DataFrame(feats).T.reindex(columns=FEATURES)


# ── Cohort MAF loaders ───────────────────────────────────────────────────────
# Each cohort-specific loader returns a long-format DataFrame with the unified
# schema {sample_id, chrom, pos, ref, alt, var_type, var_class, gene, hgvsp}.
# Missing columns → NaN; downstream tiers handle gracefully.

UNIFIED_DIR = ROOT / "source" / "input_data" / "maf_unified"

# Variant_Type normalisation (raw label → SNP/INS/DEL or "")
VTYPE_MAP = {
    "SNP": "SNP", "SNV": "SNP", "snv": "SNP",
    "INS": "INS", "Insertion": "INS", "insertion": "INS",
    "DEL": "DEL", "Deletion": "DEL", "deletion": "DEL",
    "DNP": "DNP", "TNP": "TNP", "ONP": "ONP",
}

# Variant_Class normalisation (raw label → standard MAF classes used in NON_SYN/SILENT sets)
VCLASS_MAP = {
    # Standard MAF labels (pass through)
    "Missense_Mutation": "Missense_Mutation",
    "Nonsense_Mutation": "Nonsense_Mutation",
    "Silent": "Silent",
    "Splice_Site": "Splice_Site",
    "Frame_Shift_Ins": "Frame_Shift_Ins",
    "Frame_Shift_Del": "Frame_Shift_Del",
    "In_Frame_Ins": "In_Frame_Ins",
    "In_Frame_Del": "In_Frame_Del",
    "Translation_Start_Site": "Translation_Start_Site",
    "Nonstop_Mutation": "Nonstop_Mutation",
    # Whijae short labels
    "Missense": "Missense_Mutation",
    "Nonsense": "Nonsense_Mutation",
    # Riaz Ensembl-style labels
    "missense_variant": "Missense_Mutation",
    "stop_gained": "Nonsense_Mutation",
    "synonymous_variant": "Silent",
    "splice_acceptor_variant": "Splice_Site",
    "splice_donor_variant": "Splice_Site",
    "splice_region_variant": "Splice_Site",
    "frameshift_variant": "Frame_Shift_Ins",  # ambiguous ins/del; keep as nonsyn
    "inframe_insertion": "In_Frame_Ins",
    "inframe_deletion": "In_Frame_Del",
    "stop_lost": "Nonstop_Mutation",
    "start_lost": "Translation_Start_Site",
    # Snyder ANNOVAR labels
    "synonymous SNV": "Silent",
    "nonsynonymous SNV": "Missense_Mutation",
    "stopgain": "Nonsense_Mutation",
    # PUSH ANNOVAR labels
    "nonsynonymous_SNV": "Missense_Mutation",
    "synonymous_SNV": "Silent",
    "frameshift_insertion": "Frame_Shift_Ins",
    "frameshift_deletion": "Frame_Shift_Del",
    "nonframeshift_insertion": "In_Frame_Ins",
    "nonframeshift_deletion": "In_Frame_Del",
    # Gandara
    "missense": "Missense_Mutation",
    "nonsense": "Nonsense_Mutation",
    "synonymous": "Silent",
    "splice": "Splice_Site",
}


def _norm_chrom(s) -> str:
    s = str(s)
    s = s[3:] if s.lower().startswith("chr") else s
    return s


def _load_unified_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(UNIFIED_DIR / name, low_memory=False, **kwargs)


def _read_train_maf() -> pd.DataFrame:
    df = pd.read_csv(SRC_TRAIN / "data_mutations.txt", sep="\t", low_memory=False, usecols=[
        "Tumor_Sample_Barcode", "Chromosome", "Start_Position",
        "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Type",
        "Variant_Classification", "Hugo_Symbol", "HGVSp_Short",
    ])
    return pd.DataFrame({
        "sample_id": df["Tumor_Sample_Barcode"].astype(str),
        "chrom": df["Chromosome"].astype(str),
        "pos": df["Start_Position"],
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["Variant_Classification"].map(VCLASS_MAP).fillna(df["Variant_Classification"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["HGVSp_Short"],
    })


def _read_snyder() -> pd.DataFrame:
    df = _load_unified_csv("variants_SnyderUC.csv")
    return pd.DataFrame({
        "sample_id": df["Tumor_Sample_Barcode"].astype(str),
        "chrom": df["Chromosome"].map(_norm_chrom),
        "pos": df["Start_Position"],
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["ExonicFunc.refGene"].map(VCLASS_MAP).fillna(df["Variant_Classification"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["aaChange"],
    })


def _read_pleasance() -> pd.DataFrame:
    df = _load_unified_csv("variants_Pleasance.csv")
    return pd.DataFrame({
        "sample_id": df["Tumor_Sample_Barcode"].astype(str),
        "chrom": df["Chromosome"].map(_norm_chrom),
        "pos": df["Start_Position"],
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["Variant_Classification"].map(VCLASS_MAP).fillna(df["Variant_Classification"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["HGVSp_Short"],
    })


def _read_braun() -> pd.DataFrame:
    df = _load_unified_csv("variants_Braun.csv")
    return pd.DataFrame({
        "sample_id": df["Tumor_Sample_Barcode"].astype(str),
        "chrom": df["Chromosome"].map(_norm_chrom),
        "pos": df["Start_position"],
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["Variant_Classification"].map(VCLASS_MAP).fillna(df["Variant_Classification"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["Protein_Change"],
    })


def _read_cm214() -> pd.DataFrame:
    df = _load_unified_csv("variants_CM214_JV101.csv")
    return pd.DataFrame({
        "sample_id": df["UUID"].astype(str),
        "chrom": df["Chromosome"].map(_norm_chrom),
        "pos": df["Start_Position"],
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["Variant_Classification"].map(VCLASS_MAP).fillna(df["Variant_Classification"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["Protein_Change"],
    })


def _read_hugo() -> pd.DataFrame:
    df = _load_unified_csv("variants_Hugo.csv")
    nm = df["NucMut"].astype(str)  # e.g. "C>T"
    ref = nm.str.split(">").str[0]
    alt = nm.str.split(">").str[1]
    is_snv = ref.str.len().eq(1) & alt.str.len().eq(1) & ref.isin(list("ACGT")) & alt.isin(list("ACGT"))
    return pd.DataFrame({
        "sample_id": df["Sample"].astype(str),
        "chrom": df["Chr"].map(_norm_chrom),
        "pos": df["Pos"],
        "ref": ref,
        "alt": alt,
        "var_type": np.where(is_snv, "SNP", ""),
        "var_class": df["MutType"].map(VCLASS_MAP).fillna(df["MutType"]),
        "gene": df["Gene"],
        "hgvsp": df["Aamut"].apply(lambda s: f"p.{s}" if isinstance(s, str) and s and not s.startswith("p.") else s),
    })


def _read_liu() -> pd.DataFrame:
    """Liu cohort: combine Liu_SNP (with REF/ALT) for trinucleotide signatures with
    variants_Liu.csv (with Hugo_Symbol) for gene-level HRD/DDR features.
    """
    snp = _load_unified_csv("variants_Liu_SNP.csv")
    full = _load_unified_csv("variants_Liu.csv")
    # Use Liu_SNP rows (have REF/ALT). Bring in Hugo_Symbol via (sample, chrom, pos)
    # join with variants_Liu.
    full_keys = full.rename(columns={
        "Patient": "sample_id", "Chromosome": "chrom", "Start_position": "pos",
        "Hugo_Symbol": "gene",
    })[["sample_id", "chrom", "pos", "gene"]].copy()
    full_keys["chrom"] = full_keys["chrom"].map(_norm_chrom)
    full_keys["sample_id"] = full_keys["sample_id"].astype(str)
    snp = snp.copy()
    snp["chrom"] = snp["CHROM"].map(_norm_chrom)
    snp["sample_id"] = snp["sample_id"].astype(str)
    snp["pos"] = snp["POS"]
    snp = snp.merge(full_keys, on=["sample_id", "chrom", "pos"],
                     how="left", suffixes=("", "_full"))
    return pd.DataFrame({
        "sample_id": snp["sample_id"],
        "chrom": snp["chrom"],
        "pos": snp["pos"],
        "ref": snp["REF"],
        "alt": snp["ALT"],
        "var_type": snp["Variant_Type"].map(VTYPE_MAP).fillna(snp["Variant_Type"]),
        "var_class": snp["Variant_Classification"].map(VCLASS_MAP).fillna(snp["Variant_Classification"]),
        "gene": snp.get("gene", np.nan),
        "hgvsp": np.nan,
    })


def _read_riaz() -> pd.DataFrame:
    df = _load_unified_csv("variants_Riaz.csv")
    # Parse ref/alt from HGVS_c if possible (e.g. "c.2021G>T")
    hg = df["HGVS_c"].fillna("").astype(str)
    sub = hg.str.extract(r"([ACGT])>([ACGT])$")
    return pd.DataFrame({
        "sample_id": df["Patient"].astype(str),
        "chrom": df["Chromosome"].map(_norm_chrom),
        "pos": df["Start"],
        "ref": sub[0],
        "alt": sub[1],
        "var_type": np.where(sub[0].notna(), "SNP", ""),
        "var_class": df["Variant.Classification"].map(VCLASS_MAP).fillna(df["Variant.Classification"]),
        "gene": df["Hugo.Symbol"],
        "hgvsp": df["HGVS_p"],
    })


def _read_whijae() -> pd.DataFrame:
    df = _load_unified_csv("variants_Whijae.csv")
    return pd.DataFrame({
        "sample_id": df["Sample"].astype(str),
        "chrom": df["chrom"].map(_norm_chrom),
        "pos": df["start"],
        "ref": df["ref_allele"],
        "alt": df["alt_allele"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["Variant_Class"].map(VCLASS_MAP).fillna(df["Variant_Class"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["aaannotation"],
    })


def _read_miao() -> pd.DataFrame:
    df = _load_unified_csv("variants_Miao.csv")
    return pd.DataFrame({
        "sample_id": df["Tumor_Sample_Barcode"].astype(str),
        "chrom": df["Chromosome"].map(_norm_chrom),
        "pos": df["Start_position"],
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["Variant_Classification"].map(VCLASS_MAP).fillna(df["Variant_Classification"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["Protein_Change"],
    })


def _read_push() -> pd.DataFrame:
    df = _load_unified_csv("variants_PUSH.csv")
    # PUSH stores indels with multi-char ref/alt; keep as-is, downstream filter
    # restricts trinucleotide context to single-base SNV.
    return pd.DataFrame({
        "sample_id": df["Tumor_Sample_Barcode"].astype(str),
        "chrom": df["Chr"].map(_norm_chrom),
        "pos": df["pos"].fillna(df["Start"]),
        "ref": df["ref"],
        "alt": df["alt"],
        "var_type": df["type"].map(VTYPE_MAP).fillna(df["type"]),
        "var_class": df["ExonicFunc.refGene"].map(VCLASS_MAP).fillna(df["ExonicFunc.refGene"]),
        "gene": df["Gene.refGene"],
        "hgvsp": df["mutation_p_short"],
    })


def _read_ravi() -> pd.DataFrame:
    df = _load_unified_csv("variants_Ravi.csv")
    # tumor_id is something like SU2CLC-CLE-NIVO18-T1 but proc clin uses
    # SU2CLC-CLE-NIVO1 (without -T1 / -T2 suffix). Strip -T<digit>$.
    sid = df["tumor_id"].astype(str).str.replace(r"-T\d+$", "", regex=True)
    return pd.DataFrame({
        "sample_id": sid,
        "chrom": df["Chromosome"].map(_norm_chrom),
        "pos": df["Start_position"],
        "ref": df["Reference_Allele"],
        "alt": df["Tumor_Seq_Allele2"],
        "var_type": df["Variant_Type"].map(VTYPE_MAP).fillna(df["Variant_Type"]),
        "var_class": df["Variant_Classification"].map(VCLASS_MAP).fillna(df["Variant_Classification"]),
        "gene": df["Hugo_Symbol"],
        "hgvsp": df["Protein_Change"],
    })


def _read_gandara() -> pd.DataFrame:
    df = _load_unified_csv("variants_Gandara.csv")
    return pd.DataFrame({
        "sample_id": df["PtID"].astype(str),
        "chrom": df["chromosome"].map(_norm_chrom),
        "pos": df["start"],
        "ref": df["reference_sequence"],
        "alt": df["alternate_sequence"],
        "var_type": np.where(df["reference_sequence"].astype(str).str.len().eq(1)
                              & df["alternate_sequence"].astype(str).str.len().eq(1),
                              "SNP", ""),
        "var_class": df["effect"].map(VCLASS_MAP).fillna(df["effect"]),
        "gene": df["gene_name"],
        "hgvsp": df["protein_syntax"],
    })


# ── Cohort dispatch table ────────────────────────────────────────────────────
# Tier 0 = full MAF + chr/pos/ref/alt → 10-D
# Tier 2 = gene matrix only → 4-D (POLE/HRD/DDR/TMB)
# Mariathasan / iM210: per user note, the unified file mixed CNV calls in,
# so we fall back to the existing 0/1 mutation matrix from preprocess.py.

COHORT_REGISTRY = {
    "train":        (_read_train_maf, 0),
    "Snyder_UC":    (_read_snyder,    0),
    "Pleasance":    (_read_pleasance, 0),
    "Braun":        (_read_braun,     0),
    "CM214_JV101":  (_read_cm214,     0),
    "Hugo":         (_read_hugo,      0),
    "Liu":          (_read_liu,       0),
    "Riaz":         (_read_riaz,      0),
    "Whijae":       (_read_whijae,    0),
    "Miao":         (_read_miao,      0),
    "PUSH":         (_read_push,      0),
    "Ravi":         (_read_ravi,      0),
    "Gandara":      (_read_gandara,   0),
    # Mariathasan: use existing wt/mut matrix only (iM210 unified file mixed CNV)
    "Mariathasan":  (None, 2),
}


def process_cohort(label: str, processed_index: pd.Index, genome) -> tuple[pd.DataFrame, dict]:
    loader, tier = COHORT_REGISTRY[label]
    if tier == 0 and loader is not None:
        df = loader()
        feats = compute_full(df, genome)
        raw_n = feats.shape[0]
    else:
        prefix = "train" if label == "train" else f"valid_{label}"
        mut = pd.read_csv(PROC_DIR / f"{prefix}_mut.csv", index_col=0)
        feats = compute_matrix(mut)
        raw_n = feats.shape[0]
        tier = 2
    overlap = feats.index.intersection(processed_index)
    feats = feats.reindex(processed_index, fill_value=0.0)
    nz_mask = (feats != 0).any(axis=1)
    info = {"tier": tier, "raw_n": int(raw_n),
            "processed_n": int(len(processed_index)),
            "overlap": int(len(overlap)),
            "nonzero": int(nz_mask.sum())}
    return feats, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cohort", default=None,
                        help="single cohort label (e.g. Snyder_UC)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading hg19 ({GENOME_PATH})...")
    genome = pyfaidx.Fasta(str(GENOME_PATH))

    cohorts = list(COHORT_REGISTRY) if args.all else [args.cohort]
    summary = []
    for label in cohorts:
        if label is None:
            print("usage: --all  or  --cohort <name>")
            sys.exit(1)
        prefix = "train" if label == "train" else f"valid_{label}"
        clin_path = PROC_DIR / f"{prefix}_clin.csv"
        if not clin_path.exists():
            print(f"  [skip] {label}: no processed clin file")
            continue
        idx = pd.read_csv(clin_path, index_col=0).index
        feats, info = process_cohort(label, idx, genome)
        out = OUT_DIR / f"{prefix}_sigfeats.csv"
        feats.to_csv(out)
        msg = (f"  [{label}] tier={info['tier']}  raw={info['raw_n']}"
               f"  proc={info['processed_n']}  overlap={info['overlap']}"
               f"  nonzero={info['nonzero']}  saved={out.name}")
        print(msg)
        summary.append({"cohort": label, **info})
    if summary:
        sdf = pd.DataFrame(summary)
        print("\nSummary:")
        print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
