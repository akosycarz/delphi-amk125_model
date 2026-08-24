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
OUTPUT_DIRECTORY = Path("bs230_auc_comparison_final")

PREDICTION_GAP = 12.0
MIN_POSITIVE_EVENTS = 100

MODEL_NAMES = {
    "clinical_icd": "Model 1",
    "clinical_demographics_icd": "Model 2",
    "clinical_demographics_ukb_icd": "Model 3",
    "clinical_demographics_ukb_biochem_icd": "Model 4",
    "clinical_demographics_ukb_biochem_icd_self_reported": "Model 5"
}

MODEL_ORDER = [
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4",
    "Model 5"
]

BOX_BLUE = "#5B84CC"
POINT_BLUE = "#9CBCE5"

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 16,
    "legend.title_fontsize": 17,
    "savefig.dpi": 300
})


# ============================================================
# Read data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)

required_columns = {
    "model",
    "sex",
    "icd_code",
    "horizon_months",
    "n_positive",
    "auc"
}

missing = required_columns.difference(
    data.columns
)

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

data["icd_code"] = (
    data["icd_code"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# Filter to the 12-month BS230 analysis
# ============================================================

data = data[
    data["display_model"].isin(MODEL_ORDER)
    & (data["sex"].str.lower() == "full")
    & np.isclose(
        data["horizon_months"],
        PREDICTION_GAP,
        equal_nan=False
    )
    & (
        data["n_positive"]
        >= MIN_POSITIVE_EVENTS
    )
    & data["auc"].between(0, 1)
].copy()

if data.empty:
    raise ValueError(
        "No 12-month BS230 results remained after filtering."
    )

# Collapse duplicate estimates
data = (
    data
    .groupby(
        [
            "display_model",
            "icd_code"
        ],
        as_index=False
    )
    .agg(
        auc=("auc", "mean"),
        n_positive=("n_positive", "max")
    )
)


# ============================================================
# Restrict to outcomes available in all five models
# ============================================================

code_sets = []

for model_name in MODEL_ORDER:

    model_codes = set(
        data.loc[
            data["display_model"] == model_name,
            "icd_code"
        ]
    )

    print(
        f"{model_name}: "
        f"{len(model_codes):,} eligible outcomes before matching"
    )

    code_sets.append(model_codes)

common_codes = set.intersection(
    *code_sets
)

if not common_codes:
    raise ValueError(
        "No common eligible outcomes were found."
    )

comparison_data = data[
    data["icd_code"].isin(common_codes)
].copy()

print(
    f"\nCommon outcomes used: {len(common_codes):,}"
)


# ============================================================
# Calculate model summaries
# ============================================================

summary = (
    comparison_data
    .groupby("display_model")
    .agg(
        n_outcomes=("icd_code", "nunique"),
        median_auc=("auc", "median"),
        mean_auc=("auc", "mean"),
        q1_auc=("auc", lambda values: values.quantile(0.25)),
        q3_auc=("auc", lambda values: values.quantile(0.75)),
        minimum_auc=("auc", "min"),
        maximum_auc=("auc", "max")
    )
    .reindex(MODEL_ORDER)
    .reset_index()
)

print("\nSummary:")
print(
    summary.round(3).to_string(index=False)
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    OUTPUT_DIRECTORY /
    "auc_comparison_summary.csv",
    index=False
)

comparison_data.to_csv(
    OUTPUT_DIRECTORY /
    "auc_comparison_plotting_data.csv",
    index=False
)


# ============================================================
# Prepare plot
# ============================================================

values_by_model = [
    comparison_data.loc[
        comparison_data["display_model"] == model_name,
        "auc"
    ].to_numpy()
    for model_name in MODEL_ORDER
]

positions = np.arange(1, 6)

fig, ax = plt.subplots(
    figsize=(14.5, 10)
)


# ============================================================
# Add subtle disease points
# ============================================================

random_generator = np.random.default_rng(
    230
)

for position, model_name in zip(
    positions,
    MODEL_ORDER
):

    values = comparison_data.loc[
        comparison_data["display_model"] == model_name,
        "auc"
    ].to_numpy()

    jitter = random_generator.uniform(
        -0.28,
        0.28,
        size=len(values)
    )

    ax.scatter(
        np.full(len(values), position) + jitter,
        values,
        s=18,
        color=POINT_BLUE,
        alpha=0.23,
        edgecolors="none",
        rasterized=True,
        zorder=1
    )


# ============================================================
# Add boxplots
# ============================================================

boxplot = ax.boxplot(
    values_by_model,
    positions=positions,
    widths=0.58,
    patch_artist=True,
    showfliers=False,

    medianprops={
        "color": "#111111",
        "linewidth": 2.5
    },

    boxprops={
        "facecolor": BOX_BLUE,
        "edgecolor": "#333333",
        "linewidth": 1.5,
        "alpha": 1
    },

    whiskerprops={
        "color": "#444444",
        "linewidth": 1.5
    },

    capprops={
        "color": "#444444",
        "linewidth": 1.5
    },

    zorder=3
)

for box in boxplot["boxes"]:
    box.set_facecolor(BOX_BLUE)
    box.set_alpha(1)


# ============================================================
# Chance-level reference
# ============================================================

ax.axhline(
    y=0.5,
    color="#555555",
    linestyle="--",
    linewidth=1.8,
    zorder=2
)


# ============================================================
# Labels and formatting
# ============================================================

ax.set_xlim(
    0.55,
    5.45
)

ax.set_ylim(
    0,
    1
)

ax.set_xticks(
    positions,
    MODEL_ORDER
)

ax.set_title(
    "AUC comparison across models",
    fontsize=28,
    fontweight="bold",
    pad=22
)

ax.set_xlabel(
    "Model configuration",
    fontsize=24,
    fontweight="normal",
    labelpad=18
)

ax.set_ylabel(
    "Disease-level ROC AUC",
    fontsize=24,
    fontweight="normal",
    labelpad=20
)

ax.tick_params(
    axis="both",
    labelsize=19
)

ax.yaxis.grid(
    True,
    color="#E1E1E1",
    linewidth=0.9
)

ax.xaxis.grid(False)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# Median legend
# ============================================================

legend_handles = []

for row in summary.itertuples(
    index=False
):

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=BOX_BLUE,
            linewidth=7,
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
        color="#555555",
        linestyle="--",
        linewidth=1.8,
        label="Chance level (ROC AUC = 0.5)"
    )
)

ax.legend(
    handles=legend_handles,
    title=(
        f"Common outcomes with at least "
        f"{MIN_POSITIVE_EVENTS:,} positive observations "
        f"(n={len(common_codes):,})"
    ),
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=2,
    frameon=False,
    fontsize=16,
    title_fontsize=17,
    columnspacing=2.2,
    labelspacing=0.9,
    handlelength=2.0
)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    top=0.91,
    bottom=0.30
)


# ============================================================
# Save
# ============================================================

png_path = (
    OUTPUT_DIRECTORY /
    "auc_comparison_across_models_bs230.png"
)

pdf_path = (
    OUTPUT_DIRECTORY /
    "auc_comparison_across_models_bs230.pdf"
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

print("\nCreated:")
print(png_path)
print(pdf_path)
