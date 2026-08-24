#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Create BS230 ICD-10 chapter and cross-model AUC figures."
    )
)

parser.add_argument(
    "--gap",
    type=float,
    default=0.0,
    help="Prediction gap in months. Default: 0."
)

parser.add_argument(
    "--min-events",
    type=int,
    default=100,
    help="Minimum positive events. Default: 100."
)

args = parser.parse_args()

PREDICTION_GAP = args.gap
MIN_POSITIVE_EVENTS = args.min_events


# ============================================================
# Settings
# ============================================================

INPUT_FILE = Path("auc_plot_data.csv")

OUTPUT_DIRECTORY = Path(
    f"bs230_auc_figures_gap_{PREDICTION_GAP:g}m"
)

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

MODEL_COLOURS = {
    "Model 1": "#0072B2",
    "Model 2": "#E69F00",
    "Model 3": "#009E73",
    "Model 4": "#D55E00",
    "Model 5": "#CC79A7"
}

CHAPTER_COLOURS = [
    "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D",
    "#59A14F", "#8CD17D", "#E15759", "#FF9D9A",
    "#B6992D", "#F1CE63", "#499894", "#86BCB6",
    "#79706E", "#BAB0AC", "#D37295", "#FABFD2",
    "#B07AA1", "#D4A6C8", "#9D7660", "#D7B5A6",
    "#5F9ED1", "#76B7B2", "#EDC948"
]

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 17,
    "savefig.dpi": 300
})


# ============================================================
# Read and validate combined BS230 results
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}\n"
        "Run prepare_combined_bs230_auc.py first."
    )

data = pd.read_csv(INPUT_FILE)

required_columns = {
    "model",
    "sex",
    "icd_code",
    "chapter",
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

data["chapter"] = (
    data["chapter"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)


# ============================================================
# Confirm requested prediction gap exists
# ============================================================

available_gaps = sorted(
    data["horizon_months"]
    .dropna()
    .unique()
)

if not any(
    np.isclose(
        available_gaps,
        PREDICTION_GAP
    )
):
    raise ValueError(
        f"The requested {PREDICTION_GAP:g}-month gap is not available.\n"
        f"Available gaps: {available_gaps}\n"
        "A valid 0-month figure requires evaluation outputs produced "
        "with prediction_gap_months = 0."
    )


# ============================================================
# Filter analysis data
# ============================================================

plot_data = data[
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

if plot_data.empty:
    raise ValueError(
        "No eligible results remained after filtering."
    )

# Collapse duplicate model/code estimates
plot_data = (
    plot_data
    .groupby(
        [
            "display_model",
            "icd_code",
            "chapter"
        ],
        as_index=False
    )
    .agg(
        auc=("auc", "mean"),
        n_positive=("n_positive", "max")
    )
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

plot_data.to_csv(
    OUTPUT_DIRECTORY / "bs230_auc_plotting_data.csv",
    index=False
)


# ============================================================
# Helpers
# ============================================================

def icd_sort_key(code):
    match = re.match(
        r"^([A-Z])(\d{2})",
        str(code)
    )

    if match:
        return (
            match.group(1),
            int(match.group(2))
        )

    if str(code).upper() == "DEATH":
        return ("ZZ", 999)

    return ("ZY", 998)


def roman_chapter_label(chapter):
    chapter_lower = str(chapter).lower()

    mappings = [
        (("infectious", "parasitic"), "I"),
        (("neoplasm",), "II"),
        (("blood", "immune"), "III"),
        (("endocrine", "metabolic"), "IV"),
        (("mental", "behaviour"), "V"),
        (("nervous",), "VI"),
        (("eye",), "VII"),
        (("ear",), "VIII"),
        (("circulatory",), "IX"),
        (("respiratory",), "X"),
        (("digestive",), "XI"),
        (("skin",), "XII"),
        (("musculoskeletal", "connective"), "XIII"),
        (("genitourinary",), "XIV"),
        (("pregnancy", "childbirth"), "XV"),
        (("perinatal",), "XVI"),
        (("congenital",), "XVII"),
        (("symptom", "abnormal"), "XVIII"),
        (("injury", "poison"), "XIX"),
        (("external cause",), "XX"),
        (("factors influencing", "health services"), "XXI"),
        (("special purpose",), "XXII"),
        (("death",), "Death")
    ]

    for keywords, label in mappings:
        if any(
            keyword in chapter_lower
            for keyword in keywords
        ):
            return label

    roman_match = re.match(
        r"^([IVXLCDM]+)\.",
        str(chapter)
    )

    if roman_match:
        return roman_match.group(1)

    return str(chapter)



def full_chapter_name(chapter):
    """Return the full standard display name for an ICD-10 chapter."""

    roman = roman_chapter_label(chapter)

    full_names = {
        "I": "I. Infectious Diseases",
        "II": "II. Neoplasms",
        "III": "III. Blood & Immune Disorders",
        "IV": "IV. Metabolic Diseases",
        "V": "V. Mental Disorders",
        "VI": "VI. Nervous System Diseases",
        "VII": "VII. Eye Diseases",
        "VIII": "VIII. Ear Diseases",
        "IX": "IX. Circulatory Diseases",
        "X": "X. Respiratory Diseases",
        "XI": "XI. Digestive Diseases",
        "XII": "XII. Skin Diseases",
        "XIII": "XIII. Musculoskeletal Diseases",
        "XIV": "XIV. Genitourinary Diseases",
        "XV": "XV. Pregnancy & Childbirth",
        "XVI": "XVI. Perinatal Conditions",
        "XVII": "XVII. Congenital Abnormalities",
        "XVIII": "XVIII. Symptoms, Signs & Abnormal Findings",
        "XIX": "XIX. Injury, Poisoning & Other External Consequences",
        "XX": "XX. External Causes of Morbidity & Mortality",
        "XXI": "XXI. Factors Influencing Health Status & Health Services",
        "XXII": "XXII. Codes for Special Purposes",
        "Death": "Death"
    }

    return full_names.get(
        roman,
        str(chapter)
    )


def save_figure(fig, filename):
    png_path = OUTPUT_DIRECTORY / f"{filename}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{filename}.pdf"

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

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# 1. Separate ICD-10 chapter boxplot for each model
# ============================================================

chapter_summaries = []

for model_name in MODEL_ORDER:

    model_data = plot_data[
        plot_data["display_model"] == model_name
    ].copy()

    if model_data.empty:
        print(
            f"Skipping {model_name}: no eligible outcomes"
        )
        continue

    # Order chapters using their earliest ICD-10 code
    chapter_order_table = (
        model_data[
            ["chapter", "icd_code"]
        ]
        .assign(
            sort_key=lambda frame:
                frame["icd_code"].map(icd_sort_key)
        )
        .sort_values("sort_key")
    )

    chapter_order = (
        chapter_order_table["chapter"]
        .drop_duplicates()
        .tolist()
    )

    chapter_palette = {
        chapter: CHAPTER_COLOURS[
            index % len(CHAPTER_COLOURS)
        ]
        for index, chapter in enumerate(chapter_order)
    }

    values_by_chapter = [
        model_data.loc[
            model_data["chapter"] == chapter,
            "auc"
        ].to_numpy()
        for chapter in chapter_order
    ]

    positions = np.arange(
        1,
        len(chapter_order) + 1
    )

    fig, ax = plt.subplots(
        figsize=(24, 17)
    )

    boxplot = ax.boxplot(
        values_by_chapter,
        positions=positions,
        widths=0.82,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#BDBDBD",
            "markeredgecolor": "none",
            "alpha": 0.30
        },
        medianprops={
            "color": "#111111",
            "linewidth": 2.5
        },
        boxprops={
            "edgecolor": "#333333",
            "linewidth": 1.5
        },
        whiskerprops={
            "color": "#333333",
            "linewidth": 1.5
        },
        capprops={
            "color": "#333333",
            "linewidth": 1.5
        }
    )

    for box, chapter in zip(
        boxplot["boxes"],
        chapter_order
    ):
        box.set_facecolor(
            chapter_palette[chapter]
        )

    # Chance-level reference line
    ax.axhline(
        0.5,
        color="#444444",
        linestyle="--",
        linewidth=2
    )

    ax.set_ylim(0, 1)

    ax.set_title(
        f"AUC by ICD-10 chapter — {model_name}",
        fontsize=30,
        fontweight="bold",
        pad=24
    )

    ax.set_xlabel(
        "ICD-10 chapter",
        fontsize=26,
        fontweight="normal",
        labelpad=20
    )

    ax.set_ylabel(
        "Disease-level ROC AUC",
        fontsize=26,
        fontweight="normal",
        labelpad=22
    )

    ax.set_xticks(positions)

    ax.set_xticklabels(
        [
            roman_chapter_label(chapter)
            for chapter in chapter_order
        ],
        rotation=45,
        ha="right",
        fontsize=21,
        fontweight="normal"
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

    legend_handles = [
        Patch(
            facecolor=chapter_palette[chapter],
            edgecolor="#444444",
            label=full_chapter_name(chapter)
        )
        for chapter in chapter_order
    ]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            linewidth=2,
            label="Chance level (ROC AUC = 0.5)"
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.37),
        ncol=3,
        frameon=False,
        fontsize=17,
        handlelength=1.8,
        columnspacing=2.0,
        labelspacing=0.9
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.90,
        bottom=0.55
    )

    save_figure(
        fig,
        (
            "auc_by_icd10_chapter_"
            + model_name.lower().replace(" ", "_")
        )
    )

    model_chapter_summary = (
        model_data.groupby("chapter")
        .agg(
            n_outcomes=("icd_code", "nunique"),
            median_auc=("auc", "median"),
            q1_auc=("auc", lambda x: x.quantile(0.25)),
            q3_auc=("auc", lambda x: x.quantile(0.75))
        )
        .reset_index()
    )

    model_chapter_summary[
        "display_model"
    ] = model_name

    chapter_summaries.append(
        model_chapter_summary
    )

if chapter_summaries:
    pd.concat(
        chapter_summaries,
        ignore_index=True
    ).to_csv(
        OUTPUT_DIRECTORY /
        "auc_by_icd10_chapter_summary.csv",
        index=False
    )


# ============================================================
# 2. Fair AUC comparison across Models 1–5
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
        f"{model_name}: {len(codes):,} eligible outcomes"
    )

    code_sets.append(codes)

common_codes = set.intersection(
    *code_sets
)

if not common_codes:
    raise ValueError(
        "No eligible outcomes were shared across all models."
    )

comparison_data = plot_data[
    plot_data["icd_code"].isin(common_codes)
].copy()

comparison_summary = (
    comparison_data
    .groupby("display_model")
    .agg(
        n_outcomes=("icd_code", "nunique"),
        median_auc=("auc", "median"),
        mean_auc=("auc", "mean"),
        q1_auc=("auc", lambda x: x.quantile(0.25)),
        q3_auc=("auc", lambda x: x.quantile(0.75))
    )
    .reindex(MODEL_ORDER)
    .reset_index()
)

comparison_summary.to_csv(
    OUTPUT_DIRECTORY /
    "auc_comparison_across_models_summary.csv",
    index=False
)

comparison_data.to_csv(
    OUTPUT_DIRECTORY /
    "auc_comparison_across_models_data.csv",
    index=False
)

values_by_model = [
    comparison_data.loc[
        comparison_data["display_model"] == model_name,
        "auc"
    ].to_numpy()
    for model_name in MODEL_ORDER
]

positions = np.arange(1, 6)

fig, ax = plt.subplots(
    figsize=(14, 10)
)

boxplot = ax.boxplot(
    values_by_model,
    positions=positions,
    widths=0.82,
    patch_artist=True,
    showfliers=False,
    medianprops={
        "color": "#111111",
        "linewidth": 2.6
    },
    boxprops={
        "edgecolor": "#333333",
        "linewidth": 1.5
    },
    whiskerprops={
        "color": "#333333",
        "linewidth": 1.5
    },
    capprops={
        "color": "#333333",
        "linewidth": 1.5
    }
)

for box, model_name in zip(
    boxplot["boxes"],
    MODEL_ORDER
):
    box.set_facecolor(
        MODEL_COLOURS[model_name]
    )

# Subtle matched disease points
random_generator = np.random.default_rng(230)

for position, model_name in zip(
    positions,
    MODEL_ORDER
):
    values = comparison_data.loc[
        comparison_data["display_model"] == model_name,
        "auc"
    ].to_numpy()

    jitter = random_generator.uniform(
        -0.16,
        0.16,
        len(values)
    )

    ax.scatter(
        np.full(len(values), position) + jitter,
        values,
        s=10,
        alpha=0.10,
        color="#333333",
        edgecolors="none"
    )

ax.axhline(
    0.5,
    color="#444444",
    linestyle="--",
    linewidth=2
)

ax.set_ylim(0, 1)

ax.set_xticks(
    positions,
    MODEL_ORDER
)

ax.set_title(
    (
        "Disease-level AUC comparison across models "
        f"(BS230; {PREDICTION_GAP:g}-month gap)"
    ),
    fontsize=27,
    fontweight="bold",
    pad=22
)

ax.set_xlabel(
    "Model configuration",
    fontsize=24,
    fontweight="normal",
    labelpad=17
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
    color="#D8D8D8",
    linewidth=1
)

ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Median values in the legend
legend_handles = []

for row in comparison_summary.itertuples(
    index=False
):
    legend_handles.append(
        Patch(
            facecolor=MODEL_COLOURS[row.display_model],
            edgecolor="#333333",
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
        linewidth=2,
        label="Chance level (ROC AUC = 0.5)"
    )
)

ax.legend(
    handles=legend_handles,
    title=(
        f"{len(common_codes):,} common outcomes; "
        f"at least {MIN_POSITIVE_EVENTS:,} positives"
    ),
    loc="upper center",
    bbox_to_anchor=(0.5, -0.17),
    ncol=2,
    frameon=False,
    fontsize=16,
    title_fontsize=17
)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    top=0.91,
    bottom=0.29
)

save_figure(
    fig,
    "auc_comparison_across_models"
)

print(
    f"\nCross-model comparison used "
    f"{len(common_codes):,} common eligible outcomes."
)

print("\nCreated all BS230 AUC figures.")
