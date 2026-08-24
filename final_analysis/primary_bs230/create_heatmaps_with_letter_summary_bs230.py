#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# Settings
# ============================================================

INPUT_FILE = Path("auc_plot_data.csv")
OUTPUT_DIRECTORY = Path("heatmaps_with_letter_summary_bs230")

MIN_POSITIVE_EVENTS = 100
HORIZON_MONTHS = 12.0

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

# Lighter grey for outcomes below the event threshold
LOW_EVENT_GREY = "#E8E8E8"

# Blue for letter-level median summaries
SUMMARY_BLUE = "#5B84CC"

# Space allocated to each ICD-10 letter
ROW_SPACING = 1.35

# Height of coloured cells within that space
CELL_HEIGHT = 1.0

# Width of each numerical category
CELL_WIDTH = 1.0


# ============================================================
# Read and prepare data
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

data = data[
    data["display_model"].isin(MODEL_ORDER)
    & (data["sex"].str.lower() == "full")
    & np.isclose(
        data["horizon_months"],
        HORIZON_MONTHS,
        equal_nan=False
    )
    & data["auc"].between(0, 1)
].copy()

# Extract ICD letter and two-digit numerical component
code_parts = data["icd_code"].str.extract(
    r"^([A-Z])(\d{2})"
)

data["letter"] = code_parts[0]

data["number"] = pd.to_numeric(
    code_parts[1],
    errors="coerce"
)

data = data.dropna(
    subset=["letter", "number"]
)

data["number"] = data["number"].astype(int)

# Collapse accidental duplicate model/code rows
data = (
    data
    .groupby(
        [
            "display_model",
            "icd_code",
            "letter",
            "number"
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


# ============================================================
# Helper
# ============================================================

def safe_name(model_name):
    return model_name.lower().replace(" ", "_")


def save_figure(fig, model_name):
    stem = (
        OUTPUT_DIRECTORY
        / f"icd10_heatmap_and_letter_summary_{safe_name(model_name)}"
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

    plt.close(fig)

    print(f"Saved: {stem}.png")
    print(f"Saved: {stem}.pdf")


# ============================================================
# Create one combined heatmap + summary for each model
# ============================================================

cmap = plt.colormaps["viridis"]
normaliser = Normalize(vmin=0, vmax=1)

for model_name in MODEL_ORDER:

    model_data = data[
        data["display_model"] == model_name
    ].copy()

    if model_data.empty:
        print(f"Skipping {model_name}: no ICD-10 data")
        continue

    letters = sorted(
        model_data["letter"].unique()
    )

    numbers = list(range(100))

    # Centre of each letter row, with added spacing
    y_positions = {
        letter: index * ROW_SPACING
        for index, letter in enumerate(letters)
    }

    y_centres = np.array([
        y_positions[letter] + CELL_HEIGHT / 2
        for letter in letters
    ])

    figure_height = max(
        12,
        len(letters) * 0.70
    )

    fig = plt.figure(
        figsize=(60, 22)
    )

    grid = fig.add_gridspec(
        nrows=1,
        ncols=3,

        # Heatmap, summary, colour bar
        width_ratios=[8.5, 2.4, 0.20],

        left=0.07,
        right=0.96,
        top=0.88,
        bottom=0.21,
        wspace=0.10
    )

    heatmap_ax = fig.add_subplot(
        grid[0, 0]
    )

    summary_ax = fig.add_subplot(
        grid[0, 1],
        sharey=heatmap_ax
    )

    colourbar_ax = fig.add_subplot(
        grid[0, 2]
    )

    # White means no evaluated category/no result
    heatmap_ax.set_facecolor("white")

    # --------------------------------------------------------
    # Draw heatmap cells without grid lines
    # --------------------------------------------------------

    for row in model_data.itertuples(index=False):

        y = y_positions[row.letter]

        if row.n_positive < MIN_POSITIVE_EVENTS:
            facecolour = LOW_EVENT_GREY
        else:
            facecolour = cmap(
                normaliser(row.auc)
            )

        rectangle = patches.Rectangle(
            (row.number, y),
            CELL_WIDTH,
            CELL_HEIGHT,
            facecolor=facecolour,

            # No cell borders and therefore no heatmap grid
            edgecolor="none",
            linewidth=0
        )

        heatmap_ax.add_patch(rectangle)

    heatmap_ax.set_xlim(
        0,
        100
    )

    # Use equal data-unit scaling so each 1 x 1 cell appears square
    heatmap_ax.set_aspect(
        "equal",
        adjustable="box"
    )

    heatmap_ax.set_ylim(
        len(letters) * ROW_SPACING,
        -0.18
    )

    # Display every ICD-10 numerical component from 00 to 99
    displayed_numbers = np.arange(0, 100, 1)

    heatmap_ax.set_xticks(
        displayed_numbers + 0.5
    )

    heatmap_ax.set_xticklabels(
        [f"{number:02d}" for number in displayed_numbers],
        rotation=90,
        fontsize=24,
        fontweight="normal",
        ha="center"
    )

    heatmap_ax.set_yticks(
        y_centres
    )

    heatmap_ax.set_yticklabels(
        letters,
        fontsize=30,
        fontweight="normal"
    )

    heatmap_ax.set_xlabel(
        "ICD-10 numerical component (00–99)",
        fontsize=44,
        fontweight="normal",
        labelpad=38
    )

    heatmap_ax.set_ylabel(
        "ICD-10 letter",
        fontsize=44,
        fontweight="normal",
        labelpad=38
    )

    heatmap_ax.tick_params(
        axis="x",
        length=0,
        pad=16
    )

    heatmap_ax.tick_params(
        axis="y",
        length=0,
        pad=12
    )

    # Explicitly disable all grid lines
    heatmap_ax.grid(False)

    for spine in [
        "top",
        "right",
        "bottom",
        "left"
    ]:
        heatmap_ax.spines[spine].set_visible(False)

    # --------------------------------------------------------
    # Calculate median AUC by ICD-10 letter
    # --------------------------------------------------------

    eligible = model_data[
        model_data["n_positive"]
        >= MIN_POSITIVE_EVENTS
    ].copy()

    letter_summary = (
        eligible
        .groupby("letter")
        .agg(
            median_auc=("auc", "median"),
            n_categories=("icd_code", "nunique")
        )
        .reindex(letters)
    )

    median_values = (
        letter_summary["median_auc"]
        .to_numpy()
    )

    category_counts = (
        letter_summary["n_categories"]
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Aligned median summary panel
    # --------------------------------------------------------

    summary_ax.barh(
        y_centres,
        np.nan_to_num(
            median_values,
            nan=0
        ),
        height=CELL_HEIGHT,
        color=SUMMARY_BLUE,
        edgecolor="#333333",
        linewidth=1
    )

    summary_ax.axvline(
        0.5,
        color="#444444",
        linestyle="--",
        linewidth=1.8
    )

    # Print median and category count beside each bar
    for y, median, count in zip(
        y_centres,
        median_values,
        category_counts
    ):

        if np.isnan(median):
            label = "NA (n=0)"
            text_x = 0.03
            alignment = "left"
        else:
            label = f"{median:.3f} (n={count})"

            if median >= 0.82:
                text_x = median - 0.025
                alignment = "right"
            else:
                text_x = median + 0.025
                alignment = "left"

        summary_ax.text(
            text_x,
            y,
            label,
            va="center",
            ha=alignment,
            fontsize=24,
            fontweight="bold",
            color="#111111"
        )

    summary_ax.set_xlim(
        0,
        1
    )

    summary_ax.set_ylim(
        heatmap_ax.get_ylim()
    )

    summary_ax.set_yticks(
        y_centres
    )

    # The heatmap already supplies shared letter labels
    summary_ax.tick_params(
        axis="y",
        left=False,
        labelleft=False
    )

    summary_ax.tick_params(
        axis="x",
        labelsize=24
    )

    summary_ax.set_xlabel(
        "Median ROC AUC",
        fontsize=34,
        fontweight="normal",
        labelpad=21
    )

    summary_ax.set_title(
        "Letter-level summary",
        fontsize=34,
        fontweight="bold",
        pad=18
    )

    summary_ax.grid(
        axis="x",
        color="#E2E2E2",
        linewidth=0.8
    )

    summary_ax.set_axisbelow(True)

    summary_ax.spines["top"].set_visible(False)
    summary_ax.spines["right"].set_visible(False)
    summary_ax.spines["left"].set_visible(False)

    # --------------------------------------------------------
    # Colour bar
    # --------------------------------------------------------

    scalar_mappable = ScalarMappable(
        norm=normaliser,
        cmap=cmap
    )

    scalar_mappable.set_array([])

    colourbar = fig.colorbar(
        scalar_mappable,
        cax=colourbar_ax
    )

    colourbar.set_label(
        "ROC AUC",
        fontsize=32,
        labelpad=18
    )

    colourbar.ax.tick_params(
        labelsize=25
    )

    # --------------------------------------------------------
    # Overall title and legend
    # --------------------------------------------------------

    fig.suptitle(
        f"ICD-10 category-level performance - {model_name}",
        fontsize=40,
        fontweight="bold",
        y=0.965
    )

    legend_handles = [
        Patch(
            facecolor=LOW_EVENT_GREY,
            edgecolor="none",
            label=(
                f"Fewer than "
                f"{MIN_POSITIVE_EVENTS} positive events"
            )
        ),
        Patch(
            facecolor="white",
            edgecolor="#999999",
            label="No evaluated category or no AUC result"
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            linewidth=1.8,
            label="Chance level in summary panel (ROC AUC = 0.5)"
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
        fontsize=36,
        columnspacing=2.5,
        handletextpad=1.0
    )


    # --------------------------------------------------------
    # Force the summary panel to align exactly with heatmap rows
    # --------------------------------------------------------

    # Drawing calculates the final heatmap position after the equal-aspect
    # adjustment. The summary panel is then assigned the same lower position
    # and height, so each bar centre aligns with its ICD-letter row.
    fig.canvas.draw()

    heatmap_position = heatmap_ax.get_position()
    summary_position = summary_ax.get_position()

    summary_ax.set_position([
        summary_position.x0,
        heatmap_position.y0,
        summary_position.width,
        heatmap_position.height
    ])

    # Reapply the identical y limits and tick positions
    summary_ax.set_ylim(
        heatmap_ax.get_ylim()
    )

    summary_ax.set_yticks(
        heatmap_ax.get_yticks()
    )

    save_figure(
        fig,
        model_name
    )

print("\nCreated five heatmaps with aligned letter-level summaries.")
