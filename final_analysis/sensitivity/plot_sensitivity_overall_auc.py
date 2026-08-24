#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Settings
# ============================================================

INPUT_FILE = Path("auc_plot_data.csv")
OUTPUT_DIRECTORY = Path("overall_auc_combined")

MIN_POSITIVE_EVENTS = 100
HORIZON_MONTHS = 0.0

MODEL_NAMES = {
    "clinical_icd": "Model 1",
    "clinical_demographics_icd": "Model 2",
    "clinical_demographics_ukb_icd": "Model 3",
    "clinical_demographics_ukb_biochem_icd": "Model 4",
    "clinical_demographics_ukb_biochem_icd_self_reported":
        "Model 5"
}

MODEL_ORDER = [
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4",
    "Model 5"
]

BOX_BLUE = "#5B84CC"
POINT_BLUE = "#234E91"


# ============================================================
# Read and validate data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)

required_columns = {
    "model",
    "sex",
    "age_group",
    "icd_code",
    "horizon_months",
    "n_positive",
    "auc"
}

missing = required_columns.difference(data.columns)

if missing:
    raise ValueError(
        f"{INPUT_FILE} is missing columns: {sorted(missing)}"
    )

data["display_model"] = (
    data["model"]
    .map(MODEL_NAMES)
    .fillna(data["model"])
)

for column in [
    "horizon_months",
    "n_positive",
    "auc"
]:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data["sex"] = (
    data["sex"]
    .astype(str)
    .str.strip()
)

data["age_group"] = (
    data["age_group"]
    .astype(str)
    .str.strip()
)

data["icd_code"] = (
    data["icd_code"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# Select main sensitivity comparison
# ============================================================

plot_data = data[
    data["display_model"].isin(MODEL_ORDER)
    & (data["sex"].str.lower() == "full")
    & (data["age_group"].str.lower() == "full")
    & np.isclose(
        data["horizon_months"],
        HORIZON_MONTHS,
        equal_nan=False
    )
    & (
        data["n_positive"]
        >= MIN_POSITIVE_EVENTS
    )
    & data["auc"].between(0, 1)
].copy()

if plot_data.empty:
    raise ValueError(
        "No eligible complete-case results remained."
    )

# Collapse duplicate model/outcome estimates
plot_data = (
    plot_data
    .groupby(
        [
            "display_model",
            "icd_code"
        ],
        as_index=False
    )
    .agg(
        auc=("auc", "mean"),
        n_positive=("n_positive", "max"),
        chapter=("chapter", "first")
    )
)


# ============================================================
# Retain outcomes shared by all five models
# ============================================================

code_sets = []

for model_name in MODEL_ORDER:

    codes = set(
        plot_data.loc[
            plot_data["display_model"] == model_name,
            "icd_code"
        ]
    )

    print(
        f"{model_name}: "
        f"{len(codes):,} eligible outcomes before matching"
    )

    code_sets.append(codes)

common_codes = set.intersection(
    *code_sets
)

if not common_codes:
    raise ValueError(
        "No eligible outcomes were shared by all five models."
    )

plot_data = plot_data[
    plot_data["icd_code"].isin(common_codes)
].copy()

print(
    f"\nCommon outcomes used in the comparison: "
    f"{len(common_codes):,}"
)

plot_data["display_model"] = pd.Categorical(
    plot_data["display_model"],
    categories=MODEL_ORDER,
    ordered=True
)

plot_data = plot_data.sort_values(
    [
        "display_model",
        "icd_code"
    ]
)


# ============================================================
# Summary statistics
# ============================================================

summary = (
    plot_data
    .groupby(
        "display_model",
        observed=False
    )
    .agg(
        n_outcomes=("icd_code", "nunique"),
        median_auc=("auc", "median"),
        mean_auc=("auc", "mean"),
        q1_auc=("auc", lambda x: x.quantile(0.25)),
        q3_auc=("auc", lambda x: x.quantile(0.75)),
        minimum_auc=("auc", "min"),
        maximum_auc=("auc", "max")
    )
    .reindex(MODEL_ORDER)
    .reset_index()
)

print("\nSensitivity summary:")
print(
    summary.round(3).to_string(index=False)
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUTPUT_DIRECTORY
    / "overall_auc_summary.csv",
    index=False
)

plot_data.to_csv(
    OUTPUT_DIRECTORY
    / "overall_auc_plotting_data.csv",
    index=False
)


# ============================================================
# Prepare boxplot values
# ============================================================

auc_groups = [
    plot_data.loc[
        plot_data["display_model"] == model_name,
        "auc"
    ].to_numpy()
    for model_name in MODEL_ORDER
]

positions = np.arange(
    1,
    len(MODEL_ORDER) + 1
)


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(15.5, 9.5)
)

boxplot = ax.boxplot(
    auc_groups,
    positions=positions,
    tick_labels=MODEL_ORDER,
    widths=0.68,
    patch_artist=True,
    showfliers=False,

    medianprops={
        "color": "#111111",
        "linewidth": 3
    },

    boxprops={
        "facecolor": BOX_BLUE,
        "edgecolor": "#222222",
        "linewidth": 1.7
    },

    whiskerprops={
        "color": "#333333",
        "linewidth": 1.6
    },

    capprops={
        "color": "#333333",
        "linewidth": 1.6
    }
)

for box in boxplot["boxes"]:
    box.set_facecolor(BOX_BLUE)
    box.set_alpha(1)


# ============================================================
# Add subtle outcome points
# ============================================================

random_generator = np.random.default_rng(
    125
)

for position, model_name in zip(
    positions,
    MODEL_ORDER
):

    values = plot_data.loc[
        plot_data["display_model"] == model_name,
        "auc"
    ].to_numpy()

    jitter = random_generator.uniform(
        -0.20,
        0.20,
        size=len(values)
    )

    ax.scatter(
        np.full(
            len(values),
            position
        ) + jitter,
        values,
        s=11,
        alpha=0.10,
        color=POINT_BLUE,
        edgecolors="none",
        rasterized=True,
        zorder=2
    )


# ============================================================
# Chance-level line
# ============================================================

ax.axhline(
    y=0.5,
    color="#444444",
    linestyle="--",
    linewidth=2.5,
    zorder=1
)


# ============================================================
# Axes and title
# ============================================================

ax.set_xlim(
    0.50,
    5.50
)

ax.set_ylim(
    0,
    1
)

ax.set_title(
    "AUC comparison across models",
    fontsize=30,
    fontweight="bold",
    pad=24
)

ax.set_xlabel(
    "Model configuration",
    fontsize=25,
    labelpad=18
)

ax.set_ylabel(
    "Disease-level ROC AUC",
    fontsize=25,
    labelpad=22
)

ax.tick_params(
    axis="x",
    labelsize=21,
    pad=9
)

ax.tick_params(
    axis="y",
    labelsize=21
)

ax.yaxis.grid(
    True,
    color="#D8D8D8",
    linewidth=1
)

ax.xaxis.grid(False)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# Legend with model medians
# ============================================================

legend_handles = []

for row in summary.itertuples(index=False):

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=BOX_BLUE,
            linewidth=9,
            solid_capstyle="butt",
            label=(
                f"{row.display_model}: "
                f"median ROC AUC = {row.median_auc:.3f}"
            )
        )
    )

legend_handles.append(
    Line2D(
        [0],
        [0],
        color="#444444",
        linestyle="--",
        linewidth=2.5,
        label="Chance level (ROC AUC = 0.5)"
    )
)

ax.legend(
    handles=legend_handles,
    title=(
        f"Common outcomes "
        f"(n={len(common_codes):,}; "
        f"at least {MIN_POSITIVE_EVENTS} positive observations)"
    ),
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False,
    fontsize=17,
    title_fontsize=18,
    handlelength=2.3,
    columnspacing=2.5,
    labelspacing=1
)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    top=0.90,
    bottom=0.34
)


# ============================================================
# Save figure
# ============================================================

png_path = (
    OUTPUT_DIRECTORY
    / "overall_auc_comparison_sensitivity.png"
)

pdf_path = (
    OUTPUT_DIRECTORY
    / "overall_auc_comparison_sensitivity.pdf"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.40,
    facecolor="white"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.40
)

plt.close(fig)

print("\nCreated successfully:")
print(png_path)
print(pdf_path)
