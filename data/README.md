# Data access

## Included

- Final model configuration and state dictionary
- Public KG download utilities
- Schemas and code needed to preprocess legally obtained source data
- Synthetic examples used by the smoke tests

## Not included

- Patient-level mutation matrices or MAF files
- Patient-level clinical, treatment, response, or survival records
- Patient-level model risks, embeddings, or attribution arrays
- Controlled-access OAK/POPLAR or other governed study files
- Multi-gigabyte KG downloads and intermediate feature matrices

## Expected input layout

The preprocessing code expects a training directory and one or more validation-cohort directories containing mutation records, panel coverage information, and survival fields. The canonical processed outputs are binary mutation matrices, assay masks, clinical tables containing `OS_MONTHS` and `event`, and aligned KG-derived feature matrices.

## Source access

- MSK and cBioPortal-hosted cohorts: obtain from the corresponding study pages and original publications.
- OAK and POPLAR: obtain through the European Genome-phenome Archive or the source publication's permitted supplementary resources.
- PrimeKG, Hetionet, DRKG, iBKH, Monarch KG, OGB-BioKG, and OpenBioLink: obtain from their original public repositories or by using `src/download_kgs.py` where supported.

Users are responsible for complying with the terms, ethics approvals, and access conditions of every source dataset. No patient-level record is required for installation or integrity testing.
