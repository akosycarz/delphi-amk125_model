#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


# ============================================================
# Settings
# ============================================================

INPUT_FILE = Path("auc_plot_data.csv")
OUTPUT_DIRECTORY = Path("heatmaps_with_letter_summary")

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

SUMMARY_BLUE = "#5B84CC"
LOW_EVENT_GREY = "#E5E5E5"


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
# Select full-population horizon-0 results
# ============================================================

data = data[
    data["display_model"].isin(MODEL_ORDER)
    & (data["sex"].str.lower() == "full")
    & (data["age_group"].str.lower() == "full")
    & np.isclose(
        data["horizon_months"],
        HORIZON_MONTHS,
        equal_nan=False
    )
    & data["auc"].between(0, 1)
].copy()

if data.empty:
    raise ValueError(
        "No full-population horizon-0 results remained."
    )

# Search anywhere in the string, supporting both A00 and ICD10::A00
code_parts = data["icd_code"].str.extract(
    r"([A-Z])(\d{2})"
)

data["letter"] = code_parts[0]

data["number"] = pd.to_numeric(
    code_parts[1],
    errors="coerce"
)

recognised = data[
    data["letter"].notna()
    & data["number"].notna()
].copy()

print("Rows after population/horizon filtering:", len(data))
print("Rows recognised as ICD-10:", len(recognised))
print("Rows not recognised as ICD-10:", len(data) - len(recognised))

if recognised.empty:
    print("\nExample unrecognised codes:")
    print(
        data["icd_code"]
        .drop_duplicates()
        .head(30)
        .tolist()
    )

    raise ValueError(
        "No ICD-10 codes were recognised."
    )

data = recognised

data["number"] = (
    data["number"]
    .astype(int)
)

# Collapse duplicate estimates
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
# Create a separate heatmap for each model
# ============================================================

for model_name in MODEL_ORDER:

    model_data = data[
        data["display_model"] == model_name
    ].copy()

    if model_data.empty:
        print(f"Skipping {model_name}: no data")
        continue

    letters = sorted(
        model_data["letter"].unique()
    )

    numbers = list(range(100))

    auc_matrix = pd.DataFrame(
        np.nan,
        index=letters,
        columns=numbers
    )

    # 0 = unavailable; 1 = low event count; 2 = eligible
    status_matrix = pd.DataFrame(
        0,
        index=letters,
        columns=numbers,
        dtype=int
    )

    for row in model_data.itertuples(index=False):

        letter = row.letter
        number = int(row.number)

        if not 0 <= number <= 99:
            continue

        if row.n_positive < MIN_POSITIVE_EVENTS:
            status_matrix.loc[
                letter,
                number
            ] = 1
        else:
            status_matrix.loc[
                letter,
                number
            ] = 2

            auc_matrix.loc[
                letter,
                number
            ] = row.auc


    # ========================================================
    # Letter-level summary
    # ========================================================

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

    summary_path = (
        OUTPUT_DIRECTORY
        / (
            "letter_summary_"
            + model_name.lower().replace(" ", "_")
            + ".csv"
        )
    )

    letter_summary.to_csv(
        summary_path
    )


    # ========================================================
    # Figure layout
    # ========================================================

    figure_height = max(
        10.5,
        0.45 * len(letters)
    )

    fig = plt.figure(
        figsize=(26, figure_height)
    )

    grid = GridSpec(
        1,
        3,
        width_ratios=[
            5.8,
            1.65,
            0.10
        ],
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


    # ========================================================
    # Heatmap background states
    # ========================================================

    heatmap_ax.set_facecolor("white")

    # Low-event categories shown in grey, without borders
    for row_index, letter in enumerate(letters):

        for number in numbers:

            if status_matrix.loc[letter, number] == 1:

                heatmap_ax.add_patch(
                    patches.Rectangle(
                        (
                            number,
                            row_index
                        ),
                        1,
                        1,
                        facecolor=LOW_EVENT_GREY,
                        edgecolor="none",
                        linewidth=0
                    )
                )

    cmap = plt.colormaps["viridis"].copy()

    # Missing values are transparent
    cmap.set_bad(
        (1, 1, 1, 0)
    )

    masked_auc = np.ma.masked_invalid(
        auc_matrix.to_numpy(
            dtype=float
        )
    )

    image = heatmap_ax.imshow(
        masked_auc,
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="none",
        origin="upper",
        extent=(
            0,
            100,
            len(letters),
            0
        )
    )

    # No grid around heatmap cells
    heatmap_ax.grid(False)

    heatmap_ax.set_xlim(
        0,
        100
    )

    heatmap_ax.set_ylim(
        len(letters),
        0
    )

    heatmap_ax.set_xticks(
        np.arange(100) + 0.5
    )

    heatmap_ax.set_xticklabels(
        [
            f"{number:02d}"
            for number in numbers
        ],
        rotation=90,
        fontsize=10
    )

    heatmap_ax.set_yticks(
        np.arange(len(letters)) + 0.5
    )

    heatmap_ax.set_yticklabels(
        letters,
        fontsize=17
    )

    heatmap_ax.tick_params(
        axis="both",
        length=0
    )

    heatmap_ax.set_xlabel(
        "ICD-10 numerical component (00–99)",
        fontsize=23,
        labelpad=18
    )

    heatmap_ax.set_ylabel(
        "ICD-10 letter",
        fontsize=23,
        labelpad=18
    )

    for spine in heatmap_ax.spines.values():
        spine.set_visible(False)


    # ========================================================
    # Letter-level summary bars
    # ========================================================

    y_positions = (
        np.arange(len(letters))
        + 0.5
    )

    median_values = (
        letter_summary["median_auc"]
        .fillna(0)
        .to_numpy()
    )

    summary_ax.barh(
        y_positions,
        median_values,
        height=0.72,
        color=SUMMARY_BLUE,
        edgecolor="none"
    )

    summary_ax.axvline(
        0.5,
        color="#555555",
        linestyle="--",
        linewidth=1.8
    )

    for index, letter in enumerate(letters):

        median = letter_summary.loc[
            letter,
            "median_auc"
        ]

        count = letter_summary.loc[
            letter,
            "n_categories"
        ]

        if pd.isna(median):
            label = "NA (n=0)"
            x_position = 0.02
        else:
            label = (
                f"{median:.3f} "
                f"(n={int(count)})"
            )

            x_position = min(
                median + 0.025,
                0.83
            )

        summary_ax.text(
            x_position,
            index + 0.5,
            label,
            va="center",
            ha="left",
            fontsize=13.5,
            fontweight="bold",
            color="#222222"
        )

    summary_ax.set_xlim(
        0,
        1
    )

    summary_ax.set_ylim(
        len(letters),
        0
    )

    summary_ax.grid(False)

    summary_ax.tick_params(
        axis="y",
        left=False,
        labelleft=False
    )

    summary_ax.tick_params(
        axis="x",
        labelsize=15
    )

    summary_ax.set_xlabel(
        "Median ROC AUC",
        fontsize=20,
        labelpad=15
    )

    summary_ax.set_title(
        "Letter-level summary",
        fontsize=21,
        fontweight="bold",
        pad=16
    )

    summary_ax.spines["top"].set_visible(False)
    summary_ax.spines["right"].set_visible(False)
    summary_ax.spines["left"].set_visible(False)


    # ========================================================
    # Colour bar
    # ========================================================

    colourbar = fig.colorbar(
        image,
        cax=colourbar_ax
    )

    colourbar.set_label(
        "ROC AUC",
        fontsize=20,
        labelpad=14
    )

    colourbar.ax.tick_params(
        labelsize=15
    )


    # ========================================================
    # Title and legend
    # ========================================================

    fig.suptitle(
        f"ICD-10 category-level performance — {model_name}",
        fontsize=28,
        fontweight="bold",
        y=0.98
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
            label=(
                "No evaluated category or no AUC result"
            )
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            linewidth=1.8,
            label=(
                "Chance level in summary panel "
                "(ROC AUC = 0.5)"
            )
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        fontsize=16,
        handlelength=2.2,
        columnspacing=2.5
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.96,
        top=0.91,
        bottom=0.18
    )


    # ========================================================
    # Save
    # ========================================================

    safe_model = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    png_path = (
        OUTPUT_DIRECTORY
        / f"icd10_heatmap_with_summary_{safe_model}.png"
    )

    pdf_path = (
        OUTPUT_DIRECTORY
        / f"icd10_heatmap_with_summary_{safe_model}.pdf"
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

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

print("\nFinished all five sensitivity heatmaps.")
