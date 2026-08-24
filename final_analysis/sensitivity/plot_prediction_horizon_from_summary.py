#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Files and settings
# ============================================================

INPUT_FILE = Path(
    "auc_by_prediction_horizon_summary.csv"
)

OUTPUT_DIRECTORY = Path(
    "prediction_horizon_final"
)

MODEL_ORDER = [
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4",
    "Model 5"
]

MODEL_COLOURS = {
    "Model 1": "#0072B2",
    "Model 2": "#E69F00",
    "Model 3": "#009E73",
    "Model 4": "#D55E00",
    "Model 5": "#CC79A7"
}

# Possible model labels in the CSV
MODEL_MAPPING = {
    # Already formatted labels
    "Model 1": "Model 1",
    "Model 2": "Model 2",
    "Model 3": "Model 3",
    "Model 4": "Model 4",
    "Model 5": "Model 5",

    # Raw directory/configuration names
    "clinical_icd": "Model 1",
    "clinical_demographics_icd": "Model 2",
    "clinical_demographics_ukb_icd": "Model 3",
    "clinical_demographics_ukb_biochem_icd": "Model 4",
    "clinical_demographics_ukb_biochem_icd_self_reported":
        "Model 5",

    # Descriptive labels
    "Clinical + ICD": "Model 1",
    "Clinical ICD": "Model 1",

    "Clinical + demographics + ICD": "Model 2",

    "Clinical + demographics + UKB + ICD": "Model 3",

    "Clinical + demographics + UKB + biochemistry + ICD":
        "Model 4",

    "Clinical + demographics + UKB + biochemistry + ICD + self-report":
        "Model 5",

    "Clinical + demographics + UKB + biochemistry + ICD + self-reported":
        "Model 5"
}


# ============================================================
# Read and validate summary file
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)

required_columns = {
    "model",
    "horizon_months",
    "n_diseases",
    "auc_median",
    "auc_q25",
    "auc_q75"
}

missing_columns = required_columns.difference(
    data.columns
)

if missing_columns:
    raise ValueError(
        f"{INPUT_FILE} is missing columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# Clean model and numerical columns
# ============================================================

data["model"] = (
    data["model"]
    .astype(str)
    .str.strip()
)

data["display_model"] = data["model"].map(
    MODEL_MAPPING
)

unknown_models = (
    data.loc[
        data["display_model"].isna(),
        "model"
    ]
    .drop_duplicates()
    .tolist()
)

if unknown_models:
    raise ValueError(
        "The following model labels were not recognised:\n"
        + "\n".join(unknown_models)
    )

numeric_columns = [
    "horizon_months",
    "n_diseases",
    "auc_median",
    "auc_q25",
    "auc_q75"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data = data.dropna(
    subset=[
        "display_model",
        "horizon_months",
        "auc_median",
        "auc_q25",
        "auc_q75"
    ]
)

data = data[
    data["display_model"].isin(MODEL_ORDER)
    & data["auc_median"].between(0, 1)
    & data["auc_q25"].between(0, 1)
    & data["auc_q75"].between(0, 1)
].copy()

if data.empty:
    raise ValueError(
        "No valid rows remained after cleaning."
    )

data["display_model"] = pd.Categorical(
    data["display_model"],
    categories=MODEL_ORDER,
    ordered=True
)

data = data.sort_values(
    [
        "display_model",
        "horizon_months"
    ]
)

print("\nSummary used in the graph:")

print(
    data[
        [
            "display_model",
            "horizon_months",
            "n_diseases",
            "auc_median",
            "auc_q25",
            "auc_q75"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(20, 12)
)

legend_handles = []

for model_name in MODEL_ORDER:

    model_data = (
        data[
            data["display_model"] == model_name
        ]
        .sort_values("horizon_months")
    )

    if model_data.empty:
        print(
            f"Warning: no summary rows found for {model_name}"
        )
        continue

    x = model_data[
        "horizon_months"
    ].to_numpy()

    median = model_data[
        "auc_median"
    ].to_numpy()

    q25 = model_data[
        "auc_q25"
    ].to_numpy()

    q75 = model_data[
        "auc_q75"
    ].to_numpy()

    colour = MODEL_COLOURS[model_name]

    ax.fill_between(
        x,
        q25,
        q75,
        color=colour,
        alpha=0.13,
        linewidth=0,
        zorder=1
    )

    ax.plot(
        x,
        median,
        color=colour,
        marker="o",
        markersize=14,
        markerfacecolor=colour,
        markeredgecolor=colour,
        linewidth=4.5,
        zorder=3
    )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colour,
            marker="o",
            markersize=13,
            linewidth=4.5,
            label=model_name
        )
    )


# ============================================================
# Chance-level line
# ============================================================

ax.axhline(
    y=0.5,
    color="#444444",
    linestyle="--",
    linewidth=3,
    zorder=2
)

chance_handle = Line2D(
    [0],
    [0],
    color="#444444",
    linestyle="--",
    linewidth=3,
    label="Chance level (ROC AUC = 0.5)"
)


# ============================================================
# Axis formatting
# ============================================================

horizons = sorted(
    data["horizon_months"]
    .dropna()
    .unique()
)

ax.set_xticks(horizons)

ax.set_xlim(
    min(horizons) - 2,
    max(horizons) + 5
)

ax.set_ylim(
    0,
    1
)

ax.set_title(
    "AUC by post-baseline prediction horizon — full population",
    fontsize=38,
    fontweight="bold",
    pad=22
)

ax.set_xlabel(
    "Post-baseline prediction horizon (months)",
    fontsize=32,
    labelpad=17
)

ax.set_ylabel(
    "Median disease-level ROC AUC",
    fontsize=32,
    labelpad=20
)

ax.tick_params(
    axis="x",
    labelsize=27,
    pad=8
)

ax.tick_params(
    axis="y",
    labelsize=27
)

ax.grid(
    True,
    color="#D4D4D4",
    linewidth=1
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# Legend
# ============================================================

legend_handles.append(
    chance_handle
)

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=False,
    fontsize=25,
    handlelength=2.8,
    columnspacing=2.1,
    labelspacing=0.9
)

fig.subplots_adjust(
    left=0.15,
    right=0.98,
    top=0.86,
    bottom=0.34
)


# ============================================================
# Save output
# ============================================================

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

png_path = (
    OUTPUT_DIRECTORY
    / "auc_by_prediction_horizon_full_population_final.png"
)

pdf_path = (
    OUTPUT_DIRECTORY
    / "auc_by_prediction_horizon_full_population_final.pdf"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.35,
    facecolor="white"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.35
)

plt.close(fig)

# Save a cleaned copy of the summary used
data.to_csv(
    OUTPUT_DIRECTORY
    / "auc_by_prediction_horizon_summary_used.csv",
    index=False
)

print("\nCreated successfully:")
print(f"  {png_path}")
print(f"  {pdf_path}")
