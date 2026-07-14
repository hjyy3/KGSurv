# KGSurv4ICI

KGSurv4ICI is a knowledge-graph-guided deep survival model for overall-survival risk stratification in patients treated with immune checkpoint inhibitors.

## Repository contents

- `src/`: preprocessing, knowledge-graph features, model training, evaluation, ablation, interpretation, and manuscript statistics
- `experiments/`: experiment replay and aggregation utilities
- `artifacts/final_model/`: final model weights and configuration
- `examples/`: synthetic-data example
- `tests/`: lightweight integrity tests
- `data/README.md`: data sources and access conditions

Patient-level mutation, clinical, response, survival, and model-output data are not included.

## Installation

The model was developed with Python 3.10 and PyTorch 2.5.1.

```bash
conda env create -f environment.yml
conda activate kgsurv4ici
```

Install a hardware-appropriate PyTorch build if the CUDA version in `environment.yml` does not match the target system.

## Quick check

Run the synthetic example:

```bash
python -X utf8 examples/smoke_model.py
```

Run the tests:

```bash
python -X utf8 -m pytest tests/test_repository.py -q
```

These commands do not require patient data.

## Model

- Architecture: `PathAttnSurv`
- Knowledge graph: PrimeKG
- Final configuration: `PathAttnSurv-PrimeKG-mg2`
- Primary endpoint: overall survival
- Model-state SHA-256: `c68cda76b8dde9103a9aec5bcd2b3d4acc20fb950acff1dd62083ce3a1ff66aa`

The model is intended for research use only.

## Reproduction

Obtain the source cohorts under their original access conditions, organize them as described in `data/README.md`, and use the entry points in `src/` for preprocessing, feature generation, training, and evaluation. Public knowledge graphs can be downloaded with `src/download_kgs.py` where supported.
