"""
Data preprocessing for MutaPath-Surv.

Outputs (saved to output/processed/):
  train_mut.csv     - binary mutation matrix (samples × candidate genes)
  train_mask.csv    - panel coverage mask (1=assayed, 0=not assayed)
  train_clin.csv    - clinical data (OS_MONTHS, event)
  valid_{cohort}_mut.csv   - validation mutation matrix (aligned to same genes)
  valid_{cohort}_mask.csv  - validation mask
  valid_{cohort}_clin.csv  - validation clinical data
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "source" / "input_data" / "train"
VALID_DIR = ROOT / "source" / "input_data" / "valid"
OUT_DIR = ROOT / "output" / "processed"

NON_SYN = {
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Ins", "Frame_Shift_Del",
    "Splice_Site", "In_Frame_Ins", "In_Frame_Del",
    "Translation_Start_Site", "Nonstop_Mutation",
}

# ── Clinical covariate encoding ─────────────────────────────────────────────

TRAIN_CANCER_TYPES = [
    "Non-Small Cell Lung Cancer", "Melanoma", "Bladder Cancer",
    "Renal Cell Carcinoma", "Head and Neck Cancer",
    "Esophagogastric Cancer", "Colorectal Cancer", "Glioma",
    "Cancer of Unknown Primary", "Hepatobiliary Cancer",
]
TRAIN_DRUG_TYPES = ["PD-1/PDL-1", "Combo", "CTLA4"]

N_CLIN = 15  # 1 sex + 1 age + 3 drug + 10 cancer

GENDER_MAP: dict[str, float] = {"Male": 1.0, "Female": 0.0}
AGE_GROUP_MAP: dict[str, float] = {
    "<30": 0.0, "31-50": 0.25, "50-60": 0.5, "61-70": 0.75, ">71": 1.0,
    "<=60": 0.375, ">60": 0.875,
}
DRUG_MAP: dict[str, str] = {
    "PD-1/PDL-1": "PD-1/PDL-1", "Combo": "Combo", "CTLA4": "CTLA4",
    "Anti-PD-1/PD-L1": "PD-1/PDL-1",
    "anti-PD-1/anti-PD-L1": "PD-1/PDL-1",
    "Avelumab + Axitinib": "Combo",
    "Nivo+Ipi": "Combo",
    "Anti-PD-1/PD-L1+CTLA-4": "Combo",
    "anti-CTLA-4 + anti-PD-1/PD-L1": "Combo",
    "Anti-CTLA-4": "CTLA4",
    "anti-CTLA-4": "CTLA4",
    "PD(L)1": "PD-1/PDL-1",
}
CANCER_MAP: dict[str, str] = {
    "Non-Small Cell Lung Cancer": "Non-Small Cell Lung Cancer",
    "Melanoma": "Melanoma",
    "Bladder Cancer": "Bladder Cancer",
    "Renal Cell Carcinoma": "Renal Cell Carcinoma",
    "Head and Neck Cancer": "Head and Neck Cancer",
    "Esophagogastric Cancer": "Esophagogastric Cancer",
    "Colorectal Cancer": "Colorectal Cancer",
    "Glioma": "Glioma",
    "Cancer of Unknown Primary": "Cancer of Unknown Primary",
    "Hepatobiliary Cancer": "Hepatobiliary Cancer",
    "Breast Cancer": "Cancer of Unknown Primary",
    "RCC": "Renal Cell Carcinoma",
    "Lung": "Non-Small Cell Lung Cancer",
    "Lung Adenocarcinoma": "Non-Small Cell Lung Cancer",
    "UC": "Bladder Cancer",
    "HNSCC": "Head and Neck Cancer",
    "GC": "Esophagogastric Cancer",
    "EC": "Esophagogastric Cancer",
    "CRC": "Colorectal Cancer",
    "Colorectal Adenocarcinoma": "Colorectal Cancer",
    "Breast Invasive Ductal Carcinoma": "Cancer of Unknown Primary",
    "Pancreatic Adenocarcinoma": "Cancer of Unknown Primary",
    "Non-small cell lung cancer": "Non-Small Cell Lung Cancer",
    "NSCLC": "Non-Small Cell Lung Cancer",
    "Urothelial cancer": "Bladder Cancer",
}

COV_COLS = (
    ["cov_sex", "cov_age"]
    + [f"cov_drug_{d}" for d in TRAIN_DRUG_TYPES]
    + [f"cov_cancer_{c}" for c in TRAIN_CANCER_TYPES]
)


def _encode_covariates(
    df: pd.DataFrame,
    sex_col: str = "SEX",
    age_col: str = "AGE_GROUP",
    drug_col: str = "DRUG_TYPE",
    cancer_col: str = "CANCER_TYPE",
) -> pd.DataFrame:
    """Encode clinical covariates into 15 numeric columns.

    Works for both training (exact match) and validation (via mapping dicts).
    Returns a DataFrame with COV_COLS columns, same index as input.
    """
    n = len(df)
    result = pd.DataFrame(0.0, index=df.index, columns=COV_COLS)

    # SEX: Male=1, Female=0, Unknown=0.5
    if sex_col in df.columns:
        result["cov_sex"] = df[sex_col].map(GENDER_MAP).fillna(0.5)

    # AGE_GROUP: ordinal
    if age_col in df.columns:
        result["cov_age"] = df[age_col].map(AGE_GROUP_MAP).fillna(0.5)

    # DRUG_TYPE: one-hot (3 cols)
    if drug_col in df.columns:
        mapped_drug = df[drug_col].map(DRUG_MAP)
        for i, d in enumerate(TRAIN_DRUG_TYPES):
            result[f"cov_drug_{d}"] = (mapped_drug == d).astype(float)

    # CANCER_TYPE: one-hot (10 cols), strip trailing whitespace before mapping
    if cancer_col in df.columns:
        mapped_cancer = df[cancer_col].str.strip().map(CANCER_MAP)
        for i, c in enumerate(TRAIN_CANCER_TYPES):
            result[f"cov_cancer_{c}"] = (mapped_cancer == c).astype(float)

    return result

# Panel gene coverage: which genes each IMPACT panel covers
PANEL_GENE_INFO = TRAIN_DIR / "panel_gene_info.xlsx"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_candidate_genes() -> list[str]:
    df = pd.read_csv(TRAIN_DIR / "gene_candidate.csv")
    return df.iloc[:, 0].dropna().str.strip().tolist()


WES_GENE_FILE = ROOT / "output" / "processed_wes" / "wes_candidate_genes.csv"


def load_wes_candidate_genes(path=None) -> list[str]:
    """Load WES candidate genes (canonical HGNC, intersect of WES union and KG union).

    Built by `python src/wes_data.py --build`.
    """
    p = Path(path) if path else WES_GENE_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"WES gene file not found at {p}. Run `python src/wes_data.py --build` first."
        )
    df = pd.read_csv(p)
    return df.iloc[:, 0].dropna().str.strip().tolist()


def load_panel_coverage() -> dict[str, set[str]]:
    """Return {panel_name: set_of_covered_genes}.

    IMPACT341 → panel_gene_info.xlsx (340 genes)
    IMPACT410 → data/IMPACT410.xlsx  (410 genes)
    IMPACT468 → inferred from MAF (all genes observed in IMPACT468 samples)
                supplemented with IMPACT410 genes (conservative superset)
    """
    p341 = set(pd.read_excel(PANEL_GENE_INFO)["Gene_Symbol"].dropna().str.strip())
    p410 = set(pd.read_excel(TRAIN_DIR / "data" / "IMPACT410.xlsx")["Gene Symbol"].dropna().str.strip())

    # Infer IMPACT468 from MAF
    maf = pd.read_csv(TRAIN_DIR / "data_mutations.txt", sep="\t", low_memory=False,
                      usecols=["Hugo_Symbol", "Tumor_Sample_Barcode"])
    clin = pd.read_csv(TRAIN_DIR / "clin.csv", index_col=0, on_bad_lines="skip")
    samples468 = clin[clin["GENE_PANEL"] == "IMPACT468"].index
    p468_maf = set(maf[maf["Tumor_Sample_Barcode"].isin(samples468)]["Hugo_Symbol"].dropna())
    p468 = p468_maf | p410  # MAF-inferred + IMPACT410 as conservative superset

    return {"IMPACT341": p341, "IMPACT410": p410, "IMPACT468": p468}


def maf_to_binary(maf_path: Path, genes: list[str]) -> pd.DataFrame:
    """MAF file → binary mutation matrix (samples × genes)."""
    maf = pd.read_csv(maf_path, sep="\t", low_memory=False,
                      usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"])
    maf = maf[maf["Variant_Classification"].isin(NON_SYN)]
    maf["mutated"] = 1
    mat = (maf.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"])["mutated"]
              .max().unstack(fill_value=0))
    # align to candidate genes, fill missing genes with 0
    mat = mat.reindex(columns=genes, fill_value=0)
    return mat


def build_panel_mask(clin: pd.DataFrame, panel_coverage: dict[str, set[str]],
                     genes: list[str]) -> pd.DataFrame:
    """Build coverage mask from GENE_PANEL column in clinical data."""
    # pre-build per-panel binary rows, then map by sample
    panel_rows = {
        panel: pd.Series([int(g in covered) for g in genes], index=genes)
        for panel, covered in panel_coverage.items()
    }
    rows = clin["GENE_PANEL"].map(panel_rows)
    mask = pd.DataFrame(list(rows), index=clin.index, columns=genes).fillna(0).astype(int)
    return mask


def parse_train_clin(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, on_bad_lines="skip")
    df["event"] = df["OS_STATUS"].str.startswith("1").astype(int)
    cov = _encode_covariates(df, "SEX", "AGE_GROUP", "DRUG_TYPE", "CANCER_TYPE")
    out = df[["OS_MONTHS", "event", "GENE_PANEL"]].dropna(subset=["OS_MONTHS"])
    return pd.concat([out, cov.loc[out.index]], axis=1)


def parse_valid_clin(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardise column names across cohorts
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(".", "_").replace(" ", "_")
        if "overall" in cl and "survival" in cl and "status" not in cl:
            col_map[c] = "OS_MONTHS"
        elif "survival_status" in cl or "os_status" in cl:
            col_map[c] = "OS_STATUS_RAW"
    df = df.rename(columns=col_map)
    # event: 1=death (already 0/1 in most valid cohorts)
    if "OS_STATUS_RAW" in df.columns:
        df["event"] = pd.to_numeric(df["OS_STATUS_RAW"], errors="coerce").fillna(0).astype(int)
    id_col = df.columns[0]
    df = df.set_index(id_col)

    # PFS fallback for OS-less cohorts (e.g. Hellmann, Jung): when Overall.survival
    # is absent or entirely missing, use progression-free survival as the
    # time-to-event so the cohort can still be aligned/evaluated. ORR labels are
    # read separately, so this only affects the survival columns.
    os_missing = ("OS_MONTHS" not in df.columns) or df["OS_MONTHS"].isna().all()
    if os_missing:
        pfs_col = next((c for c in df.columns
                        if "progression" in c.lower() and "free" in c.lower()
                        and "surviv" in c.lower()), None)
        pstatus_col = next((c for c in df.columns
                            if "progression" in c.lower() and "status" in c.lower()), None)
        if pfs_col is not None:
            df["OS_MONTHS"] = pd.to_numeric(df[pfs_col], errors="coerce")
            if pstatus_col is not None:
                df["event"] = pd.to_numeric(df[pstatus_col], errors="coerce").fillna(0).astype(int)
            elif "event" not in df.columns:
                df["event"] = 0

    if "event" not in df.columns:
        df["event"] = 0

    # Encode covariates (validation columns: Gender, Age_group, Drug_type, Cancer_type)
    cov = _encode_covariates(df, "Gender", "Age_group", "Drug_type", "Cancer_type")
    out = df[["OS_MONTHS", "event"]].dropna()
    return pd.concat([out, cov.loc[out.index]], axis=1)


def valid_wide_to_binary(mut_path: Path, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide mutation table (wt/mut/NA) → (binary_matrix, mask)."""
    df = pd.read_csv(mut_path, index_col=0, low_memory=False)
    df = df.reindex(columns=genes).fillna("NA")  # missing genes → "NA" (not assayed)
    mask = (df != "NA").astype(int)         # 1=assayed, 0=not assayed
    binary = (df == "mut").astype(int)      # 1=mutated, 0=wt or NA
    return binary, mask


# ── main ─────────────────────────────────────────────────────────────────────

def preprocess_train(genes: list[str]) -> None:
    print("Processing training data...")
    clin = parse_train_clin(TRAIN_DIR / "clin.csv")
    mut = maf_to_binary(TRAIN_DIR / "data_mutations.txt", genes)

    # align samples
    common = clin.index.intersection(mut.index)
    clin, mut = clin.loc[common], mut.loc[common]

    panel_coverage = load_panel_coverage()
    mask = build_panel_mask(clin, panel_coverage, genes)

    out_clin = clin[["OS_MONTHS", "event"] + COV_COLS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mut.to_csv(OUT_DIR / "train_mut.csv")
    mask.to_csv(OUT_DIR / "train_mask.csv")
    out_clin.to_csv(OUT_DIR / "train_clin.csv")
    print(f"  Train samples: {len(common)}, genes: {len(genes)}")


def preprocess_valid(genes: list[str]) -> None:
    print("Processing validation cohorts...")
    for mut_path in sorted(VALID_DIR.glob("mutation_*.csv")):
        cohort = mut_path.stem.replace("mutation_", "")
        clin_path = VALID_DIR / f"clin_{cohort}.csv"
        if not clin_path.exists():
            print(f"  [{cohort}] missing clin file, skipping")
            continue

        clin = parse_valid_clin(clin_path)
        binary, mask = valid_wide_to_binary(mut_path, genes)

        common = clin.index.intersection(binary.index)
        if len(common) == 0:
            print(f"  [{cohort}] no overlapping samples, skipping")
            continue

        clin, binary, mask = clin.loc[common], binary.loc[common], mask.loc[common]
        binary.to_csv(OUT_DIR / f"valid_{cohort}_mut.csv")
        mask.to_csv(OUT_DIR / f"valid_{cohort}_mask.csv")
        clin.to_csv(OUT_DIR / f"valid_{cohort}_clin.csv")
        print(f"  [{cohort}] samples: {len(common)}")


def main() -> None:
    genes = load_candidate_genes()
    print(f"Candidate genes: {len(genes)}")
    preprocess_train(genes)
    preprocess_valid(genes)
    print("Done. Outputs in output/processed/")


if __name__ == "__main__":
    main()
