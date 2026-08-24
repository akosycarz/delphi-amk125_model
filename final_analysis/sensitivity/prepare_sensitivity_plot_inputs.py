#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


MIN_POSITIVE_EVENTS = 100
OUTPUT_DIRECTORY = Path("prepared_plot_inputs")

MODEL_DIRECTORIES = {
    "ukb_amk125_clinical_icd":
        "clinical_icd",

    "ukb_amk125_clinical_demographics_icd":
        "clinical_demographics_icd",

    "ukb_amk125_clinical_demographics_ukb_icd":
        "clinical_demographics_ukb_icd",

    "ukb_amk125_clinical_demographics_ukb_biochem_icd":
        "clinical_demographics_ukb_biochem_icd",

    "ukb_amk125_clinical_demographics_ukb_biochem_icd_self_reported":
        "clinical_demographics_ukb_biochem_icd_self_reported"
}

frames = []

for directory, model_name in MODEL_DIRECTORIES.items():

    path = (
        Path(directory)
        / "sex_gap_evaluation"
        / "auc_results_all_gaps.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing sensitivity result: {path}"
        )

    print(f"Reading {path}")

    frame = pd.read_csv(path)

    required = {
        "icd_code",
        "disease",
        "chapter",
        "sex",
        "age_group",
        "horizon_months",
        "n_positive",
        "auc"
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    # Replace long sensitivity model name with the same model
    # identifiers used by the primary-analysis plotting scripts.
    frame["model"] = model_name
    frame["model_directory"] = directory

    frames.append(frame)

combined = pd.concat(
    frames,
    ignore_index=True
)

for column in [
    "horizon_months",
    "prediction_gap_months",
    "n_positive",
    "n_negative",
    "n_total",
    "auc",
    "auc_lower",
    "auc_upper"
]:
    if column in combined.columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce"
        )

combined["sex"] = (
    combined["sex"]
    .astype(str)
    .str.strip()
)

combined["age_group"] = (
    combined["age_group"]
    .astype(str)
    .str.strip()
)

combined["icd_code"] = (
    combined["icd_code"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Collapse accidental duplicate evaluations
group_columns = [
    "model",
    "model_directory",
    "icd_code",
    "disease",
    "chapter",
    "sex",
    "age_group",
    "horizon_months"
]

combined = (
    combined
    .groupby(
        group_columns,
        as_index=False,
        dropna=False
    )
    .agg(
        n_positive=("n_positive", "max"),
        n_negative=("n_negative", "max"),
        n_total=("n_total", "max"),
        auc=("auc", "mean"),
        auc_lower=("auc_lower", "mean"),
        auc_upper=("auc_upper", "mean")
    )
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

combined_path = (
    OUTPUT_DIRECTORY
    / "auc_plot_data.csv"
)

combined.to_csv(
    combined_path,
    index=False
)

# Create the summary used by the combined horizon plot
horizon_data = combined[
    (combined["sex"].str.lower() == "full")
    & (combined["age_group"].str.lower() == "full")
    & (
        combined["n_positive"]
        >= MIN_POSITIVE_EVENTS
    )
    & combined["auc"].between(0, 1)
].copy()

horizon_summary = (
    horizon_data
    .groupby(
        ["model", "horizon_months"],
        as_index=False
    )
    .agg(
        n_diseases=("icd_code", "nunique"),
        auc_median=("auc", "median"),
        auc_q25=("auc", lambda x: x.quantile(0.25)),
        auc_q75=("auc", lambda x: x.quantile(0.75))
    )
)

summary_path = (
    OUTPUT_DIRECTORY
    / "auc_by_prediction_horizon_summary.csv"
)

horizon_summary.to_csv(
    summary_path,
    index=False
)

print("\nModels:")
print(sorted(combined["model"].unique()))

print("\nSex groups:")
print(sorted(combined["sex"].unique()))

print("\nHorizons:")
print(sorted(combined["horizon_months"].dropna().unique()))

print("\nRows in combined input:", len(combined))
print("Rows in horizon summary:", len(horizon_summary))

print("\nCreated:")
print(combined_path)
print(summary_path)
