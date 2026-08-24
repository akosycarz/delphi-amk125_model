#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import pandas as pd


INPUT_FILE = Path("auc_plot_data.csv")
OUTPUT_FILE = Path("auc_plot_data_canonical_chapters.csv")

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

data = pd.read_csv(INPUT_FILE)


def assign_chapter(icd_code):
    code = str(icd_code).strip().upper()

    if code == "DEATH" or "DEATH" in code:
        return 23, "Death"

    # Search anywhere so values such as ICD10::A00 are recognised
    match = re.search(
        r"([A-Z])(\d{2})",
        code
    )

    if not match:
        return 999, np.nan

    letter = match.group(1)
    number = int(match.group(2))

    if letter in {"A", "B"}:
        return 1, "I. Infectious Diseases"

    if letter == "C" or (
        letter == "D"
        and number <= 48
    ):
        return 2, "II. Neoplasms"

    if letter == "D" and number >= 50:
        return 3, "III. Blood & Immune Disorders"

    if letter == "E":
        return 4, "IV. Metabolic Diseases"

    if letter == "F":
        return 5, "V. Mental Disorders"

    if letter == "G":
        return 6, "VI. Nervous System Diseases"

    if letter == "H" and number <= 59:
        return 7, "VII. Eye Diseases"

    if letter == "H" and number >= 60:
        return 8, "VIII. Ear Diseases"

    if letter == "I":
        return 9, "IX. Circulatory Diseases"

    if letter == "J":
        return 10, "X. Respiratory Diseases"

    if letter == "K":
        return 11, "XI. Digestive Diseases"

    if letter == "L":
        return 12, "XII. Skin Diseases"

    if letter == "M":
        return 13, "XIII. Musculoskeletal Diseases"

    if letter == "N":
        return 14, "XIV. Genitourinary Diseases"

    if letter == "O":
        return 15, "XV. Pregnancy & Childbirth"

    if letter == "P":
        return (
            16,
            "XVI. Conditions Originating in the Perinatal Period"
        )

    if letter == "Q":
        return 17, "XVII. Congenital Abnormalities"

    if letter == "R":
        return (
            18,
            "XVIII. Symptoms, Signs & Abnormal Findings"
        )

    if letter in {"S", "T"}:
        return (
            19,
            "XIX. Injury, Poisoning & Other External Consequences"
        )

    if letter in {"V", "W", "X", "Y"}:
        return (
            20,
            "XX. External Causes of Morbidity & Mortality"
        )

    if letter == "Z":
        return (
            21,
            "XXI. Factors Influencing Health Status & Health Services"
        )

    if letter == "U":
        return 22, "XXII. Codes for Special Purposes"

    return 999, np.nan


assigned = data["icd_code"].map(
    assign_chapter
)

data["chapter_order"] = assigned.map(
    lambda value: value[0]
)

data["chapter"] = assigned.map(
    lambda value: value[1]
)

unmatched = data[
    data["chapter"].isna()
].copy()

matched = data[
    data["chapter"].notna()
].copy()

matched = matched.sort_values(
    [
        "chapter_order",
        "model",
        "icd_code"
    ]
)

matched.to_csv(
    OUTPUT_FILE,
    index=False
)

unmatched[
    [
        "icd_code",
        "chapter"
    ]
].drop_duplicates().to_csv(
    "unmatched_icd_codes.csv",
    index=False
)

print("Original rows:", len(data))
print("Matched ICD-10/death rows:", len(matched))
print("Unmatched or ICD-9 rows:", len(unmatched))

print("\nCanonical chapters:")
print(
    matched[
        [
            "chapter_order",
            "chapter"
        ]
    ]
    .drop_duplicates()
    .sort_values("chapter_order")
    .to_string(index=False)
)

print("\nCreated:")
print(OUTPUT_FILE)
print("unmatched_icd_codes.csv")
