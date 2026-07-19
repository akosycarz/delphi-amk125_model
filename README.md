# Delphi AMK125 Model

This repository contains the code used to run and train my (amk125) MSc thesis model, which adapts the **Delphi** disease-trajectory model to a UK Biobank cohort.

## Source / attribution

This project builds on **Delphi (Delphi-2M)**, developed by the Gerstung Lab:

- Original repository: https://github.com/gerstung-lab/Delphi
- Delphi-2M is a generative transformer model trained on ~400K patient health trajectories from UK Biobank data.
- Code license: MIT, Copyright (c) 2024 Gerstung Lab. Trained model weights (not included here) are licensed separately under CC BY-NC-ND 4.0.

## Repository contents

- `Delphi/` - core model code (`model.py`, `train.py`, `utils.py`, `plotting.py`, `configurator.py`), evaluation code (`evaluate_auc.py`, `shap-agg-eval.py`), the `config/` training configurations (including my `train_delphi_amk125.py` config), `requirements.txt`, the disease label/colour mapping file, and supplementary docs, adapted for my thesis cohort.
- `scripts/` - R scripts used to extract, recode, and preprocess UK Biobank data into the format Delphi expects.
- `shell_scripts/` - HPC (PBS/bash) job submission scripts used to run the preprocessing and training pipeline on Imperial College's HPC cluster.

## Incremental comparison matrix

`scripts/delphi_preprocess.R` creates five cumulative datasets: clinical ICD;
+ demographics; + UKB bulk; + blood biochemistry; and finally + self-reported
diagnoses. All datasets use the same clinical ICD cohort and the same seeded
60/20/20 participant split. The exact experiment
matrix is recorded in `Delphi/config/experiment_matrix.csv`, and preprocessing
writes `data/split_assignments.csv` so split membership can be audited.

## No data included

No UK Biobank data, trained model checkpoints, or derived result files are included in this repository. UK Biobank data is only available to researchers upon application (https://www.ukbiobank.ac.uk/), and per Delphi's own licensing, trained weights are not freely redistributable. To run this pipeline yourself you will need your own approved UK Biobank data extract.

## Running the model

1. Install dependencies: `pip install -r Delphi/requirements.txt`
2. Obtain and preprocess your own UK Biobank extract using the scripts in `scripts/` and the guidance in `Delphi/supplementary/ukb_rap_training.md`.
3. Train: `python Delphi/train.py Delphi/config/train_delphi_amk125.py --device=cuda --out_dir=<output_dir>`
4. Evaluate: `python Delphi/evaluate_auc.py`

See the original Delphi repository (https://github.com/gerstung-lab/Delphi) for full background on the model architecture and methodology.
