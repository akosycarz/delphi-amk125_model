#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns


# ============================================================
# Settings
# ============================================================

INPUT_FILE = Path("auc_plot_data_canonical_chapters.csv")
OUTPUT_DIRECTORY = Path("separate_chapter_boxplots_final")

MIN_POSITIVE_EVENTS = 100
HORIZON_MONTHS = 0.0

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

CHAPTER_COLOURS = [
    "#4E79A7",
    "#A0CBE8",
    "#F28E2B",
    "#FFBE7D",
    "#59A14F",
    "#8CD17D",
    "#E15759",
    "#FF9D9A",
    "#B6992D",
    "#F1CE63",
    "#499894",
    "#86BCB6",
    "#79706E",
    "#BAB0AC",
    "#D37295",
    "#FABFD2",
    "#B07AA1",
    "#D4A6C8",
    "#9D7660",
    "#D7B5A6",
    "#5F9ED1",
    "#76B7B2",
    "#EDC948"
]

sns.set_theme(
    style="whitegrid"
)

plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 27,
    "axes.labelsize": 23,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
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
    "chapter",
    "horizon_months",
    "auc",
    "n_positive"
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
# Clean data
# ============================================================

data["display_model"] = (
    data["model"]
    .map(MODEL_NAMES)
    .fillna(data["model"])
)

data["auc"] = pd.to_numeric(
    data["auc"],
    errors="coerce"
)

data["n_positive"] = pd.to_numeric(
    data["n_positive"],
    errors="coerce"
)

data["horizon_months"] = pd.to_numeric(
    data["horizon_months"],
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

# Full population, horizon 0 and at least 100 positive events
data = data[
    data["display_model"].isin(MODEL_ORDER)
    & (data["sex"].str.lower() == "full")
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

if data.empty:
    raise ValueError(
        "No rows remained after filtering."
    )

# Collapse accidental duplicate model/code estimates
data = (
    data
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


# ============================================================
# Helper functions
# ============================================================

def icd_sort_key(code):
    """Order ICD-10 categories by letter and number."""

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


def short_chapter_label(chapter):
    """Extract Roman-numeral chapter label."""

    if str(chapter).lower() == "death":
        return "Death"

    match = re.match(
        r"^([IVXLCDM]+)\.",
        str(chapter)
    )

    if match:
        return match.group(1)

    return str(chapter)


def safe_model_name(model_name):
    return model_name.lower().replace(" ", "_")


def save_figure(fig, model_name):
    stem = (
        OUTPUT_DIRECTORY
        / f"auc_by_icd10_chapter_{safe_model_name(model_name)}"
    )

    fig.savefig(
        f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.40,
        facecolor="white"
    )

    fig.savefig(
        f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.40
    )

    print(f"Saved: {stem}.png")
    print(f"Saved: {stem}.pdf")


OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Create one independent figure for each model
# ============================================================

for model_name in MODEL_ORDER:

    model_data = data[
        data["display_model"] == model_name
    ].copy()

    if model_data.empty:
        print(
            f"Skipping {model_name}: no eligible data"
        )
        continue

    # Determine chapter order from ICD-10 code order
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

    model_data["chapter"] = pd.Categorical(
        model_data["chapter"],
        categories=chapter_order,
        ordered=True
    )

    chapter_palette = {
        chapter: CHAPTER_COLOURS[
            index % len(CHAPTER_COLOURS)
        ]
        for index, chapter in enumerate(chapter_order)
    }

    # Wide figure gives every chapter more horizontal space
    fig, ax = plt.subplots(
        figsize=(22, 13)
    )

    sns.boxplot(
        data=model_data,
        x="chapter",
        y="auc",
        order=chapter_order,

        # Each x-axis category receives its corresponding colour.
        # No hue is used because hue-dodging makes the boxes very narrow.
        palette=chapter_palette,

        # Width relative to the complete space assigned to each chapter.
        width=0.82,

        saturation=1,
        showfliers=True,
        linewidth=1.8,

        medianprops={
            "color": "#111111",
            "linewidth": 2.4
        },

        whiskerprops={
            "color": "#333333",
            "linewidth": 1.6
        },

        capprops={
            "color": "#333333",
            "linewidth": 1.6
        },

        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.16
        },

        ax=ax
    )

    # Chance-level line
    ax.axhline(
        y=0.5,
        color="#444444",
        linestyle="--",
        linewidth=2.2,
        zorder=1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        f"AUC by ICD-10 chapter — {model_name}",
        fontsize=27,
        fontweight="bold",
        pad=24
    )

    ax.set_xlabel(
        "ICD-10 chapter",
        fontsize=23,
        labelpad=20
    )

    ax.set_ylabel(
        "Disease-level ROC AUC",
        fontsize=23,
        labelpad=22
    )

    ax.set_xticks(
        range(len(chapter_order))
    )

    ax.set_xticklabels(
        [
            short_chapter_label(chapter)
            for chapter in chapter_order
        ],
        rotation=45,
        ha="right",
        fontsize=18
    )

    ax.tick_params(
        axis="y",
        labelsize=18
    )

    ax.tick_params(
        axis="x",
        length=0,
        pad=8
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

    # Legend contains chapter descriptions but has no duplicate title
    legend_handles = [
        Patch(
            facecolor=chapter_palette[chapter],
            edgecolor="#444444",
            linewidth=1.2,
            label=chapter
        )
        for chapter in chapter_order
    ]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            linewidth=2.2,
            label="Chance level (ROC AUC = 0.5)"
        )
    )

    ax.legend(
        handles=legend_handles,

        # No second "ICD-10 chapter" title
        title=None,

        loc="upper center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=3,
        frameon=False,

        fontsize=15,
        handlelength=2.0,
        handleheight=1.1,
        columnspacing=2.0,
        labelspacing=0.9
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.91,
        bottom=0.47
    )

    save_figure(
        fig,
        model_name
    )

    plt.close(fig)

print("\nFinished creating five separate chapter boxplots.")
