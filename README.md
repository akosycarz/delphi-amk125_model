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

Preprocessing also writes auditable cohort reports:

- `data/participant_flow_all.csv` combines participant and event retention for
  every dataset and processing stage; each dataset directory also contains its
  own `participant_flow.csv`.
- `data/healthy_participant_summary.csv` reports healthy counts at five-year
  index ages by split and sex, using both ICD-only and ICD-plus-self-report
  definitions.
- `data/healthy_participants_at_age_50.csv` provides participant-level audit
  detail at the default trajectory start age.
- `data/healthy_definition.txt` records the exact operational definition and
  its limitation: no recorded diagnosis is not proof that disease is absent.
- `data/age_split_summary.csv`, `data/age_split_5year_bins.csv`, and
  `data/age_split_pairwise_tests.csv` compare age at first ICD diagnosis across
  the full, training, validation, and test cohorts, including sex strata.
- `data/age_split_distribution.png` visualises the split distributions, while
  `data/age_split_verification.txt` records a PASS or WARNING based on practical
  differences in means, medians, and distribution shape.

## No data included

No UK Biobank data, trained model checkpoints, or derived result files are included in this repository. UK Biobank data is only available to researchers upon application (https://www.ukbiobank.ac.uk/), and per Delphi's own licensing, trained weights are not freely redistributable. To run this pipeline yourself you will need your own approved UK Biobank data extract.

## Running the five-model workflow

1. Install dependencies: `pip install -r Delphi/requirements.txt`
2. Obtain and preprocess your own UK Biobank extract using the scripts in `scripts/` and the guidance in `Delphi/supplementary/ukb_rap_training.md`.
3. Run `scripts/delphi_preprocess.R` to create all five cumulative datasets with the same 60/20/20 participant split.
4. On the HPC, submit `shell_scripts/delphi_dryrun.sh` to train a small test model for every dataset.
5. Submit `shell_scripts/delphi_evaluate_dryrun.pbs` to evaluate all five dry-run checkpoints using the same test workflow.
6. After the dry run succeeds, submit `shell_scripts/delphi_train.pbs` to train all five full models.
7. Evaluate full checkpoints with `Delphi/run_evaluation.py`, supplying the matching dataset, checkpoint, and output directory.

The complete dry-run sequence can be submitted with
`shell_scripts/run_pipeline_dryrun.sh`. It chains preprocessing, five-model
training, and five-model evaluation with PBS dependencies, so later jobs run
only when the previous stage succeeds.

The five configurations are:

1. Clinical ICD.
2. Clinical ICD + demographics.
3. Clinical ICD + demographics + UKB bulk.
4. Clinical ICD + demographics + UKB bulk + blood biochemistry.
5. Clinical ICD + demographics + UKB bulk + blood biochemistry + self-reported diagnoses.

See the original Delphi repository (https://github.com/gerstung-lab/Delphi) for full background on the model architecture and methodology.
