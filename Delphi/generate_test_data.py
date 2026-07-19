#!/usr/bin/env python3
"""
generate_test_data.py — Creates synthetic data for all 4 dataset configurations.

Writes train.bin / val.bin / test.bin + token_dictionary.csv + labels.csv +
config_values.py to data/<config>/ so the dryrun pipeline can run locally
without real UKB data.

Usage:
    python generate_test_data.py
"""

import numpy as np
import pandas as pd
import struct
from pathlib import Path

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic vocabulary
# ---------------------------------------------------------------------------
# token_id convention (model-facing, after get_batch +1 shift):
#   0  = Padding   (never in .bin)
#   1  = No event  (bin_token_id = 0, not generated here)
#   2  = Female    (bin_token_id = 1)
#   3  = Male      (bin_token_id = 2)
#   4+ = events    (bin_token_id = 3+)

TOKEN_DICT = [
    # token_id, token_wording,                        source_type,        coding,                     value_bin
    (0,  "Padding",                                    "reserved",         "padding",                  ""),
    (1,  "No event",                                   "reserved",         "no_event",                 ""),
    (2,  "demographics::Female",                       "demographics",     "sex",                      ""),
    (3,  "demographics::Male",                         "demographics",     "sex",                      ""),
    # ICD10 diseases
    (4,  "ICD10::E11",                                 "ICD10",            "ICD10",                    ""),
    (5,  "ICD10::I21",                                 "ICD10",            "ICD10",                    ""),
    (6,  "ICD10::J44",                                 "ICD10",            "ICD10",                    ""),
    (7,  "ICD10::F32",                                 "ICD10",            "ICD10",                    ""),
    (8,  "ICD10::C34",                                 "ICD10",            "ICD10",                    ""),
    (9,  "ICD10::K57",                                 "ICD10",            "ICD10",                    ""),
    (10, "ICD10::M15",                                 "ICD10",            "ICD10",                    ""),
    # Self-reported
    (11, "self_reported_cancer::1026",                 "ICD10",            "self_reported_cancer",     ""),
    (12, "self_reported_non_cancer::1081",             "ICD10",            "self_reported_non_cancer", ""),
    # Demographics
    (13, "ukb_bulk::body_mass_index::Q1",              "ukb_bulk",         "body_mass_index",          "Q1"),
    (14, "ukb_bulk::body_mass_index::Q2",              "ukb_bulk",         "body_mass_index",          "Q2"),
    (15, "ukb_bulk::body_mass_index::Q3",              "ukb_bulk",         "body_mass_index",          "Q3"),
    (16, "ukb_bulk::body_mass_index::Q4",              "ukb_bulk",         "body_mass_index",          "Q4"),
    # Blood biochemistry
    (17, "blood_biochemistry::albumin::Q1",            "blood_biochemistry", "albumin",                "Q1"),
    (18, "blood_biochemistry::albumin::Q2",            "blood_biochemistry", "albumin",                "Q2"),
    (19, "blood_biochemistry::albumin::Q3",            "blood_biochemistry", "albumin",                "Q3"),
    (20, "blood_biochemistry::albumin::Q4",            "blood_biochemistry", "albumin",                "Q4"),
]

VOCAB_SIZE = len(TOKEN_DICT)

DISEASE_CODINGS = {"ICD10", "self_reported_cancer", "self_reported_non_cancer"}

# Tokens to ignore in loss (non-disease)
IGNORE_TOKENS = [
    row[0] for row in TOKEN_DICT
    if row[3] not in DISEASE_CODINGS
]

# ---------------------------------------------------------------------------
# Configuration definitions: which token_ids are available per config
# ---------------------------------------------------------------------------

ALL_TOKEN_IDS = [r[0] for r in TOKEN_DICT]

def token_ids_for_config(config_name):
    """Return the token_ids available in each dataset configuration."""
    always = {0, 1, 2, 3}  # padding, no_event, sex
    icd_ids     = {r[0] for r in TOKEN_DICT if r[3] == "ICD10"}
    sr_ids      = {r[0] for r in TOKEN_DICT if r[3] in ("self_reported_cancer", "self_reported_non_cancer")}
    demo_ids    = {r[0] for r in TOKEN_DICT if r[2] == "ukb_bulk"}
    biochem_ids = {r[0] for r in TOKEN_DICT if r[2] == "blood_biochemistry"}

    clinical = always | icd_ids | sr_ids
    if config_name == "ukb_amk125_clinical":
        return sorted(clinical)
    if config_name == "ukb_amk125_clinical_demographics":
        return sorted(clinical | demo_ids)
    if config_name == "ukb_amk125_clinical_demographics_ukb":
        return sorted(clinical | demo_ids)
    if config_name == "ukb_amk125_clinical_demographics_ukb_biochem":
        return sorted(clinical | demo_ids | biochem_ids)
    raise ValueError(f"Unknown config: {config_name}")


# ---------------------------------------------------------------------------
# Synthetic patient generator
# ---------------------------------------------------------------------------

def generate_patients(n_patients, available_token_ids, rng):
    """
    Generate synthetic event sequences for n_patients.

    Returns list of (patient_id, age_days, bin_token_id) rows sorted by
    patient_id then age_days, ready to write to .bin.
    """
    sex_tokens   = [t for t in [2, 3] if t in available_token_ids]
    event_tokens = [t for t in available_token_ids if t > 3]

    rows = []
    for pid in range(n_patients):
        # Sex token at birth (~age 0 + small noise)
        sex = rng.choice(sex_tokens)
        rows.append((pid, int(rng.integers(0, 365)), sex - 1))  # bin_token_id = token_id - 1

        # Random number of events
        n_events = int(rng.integers(5, 30))
        ages_days = sorted(rng.integers(40 * 365, 75 * 365, size=n_events).tolist())
        for age in ages_days:
            tok = int(rng.choice(event_tokens))
            rows.append((pid, int(age), tok - 1))  # bin_token_id = token_id - 1

    return rows


def write_bin(rows, path):
    """Write list of (patient_id, age_days, bin_token_id) as uint32 little-endian."""
    path = Path(path)
    with open(path, "wb") as f:
        for row in rows:
            f.write(struct.pack("<3I", *[max(0, v) for v in row]))
    print(f"  Written {path.name}  ({len(rows):,} rows)")


def write_metadata(out_dir, available_token_ids):
    """Write token_dictionary.csv, labels.csv, config_values.py."""
    out_dir = Path(out_dir)

    # Filter token dict to available tokens
    td_rows = [r for r in TOKEN_DICT if r[0] in set(available_token_ids)]
    td = pd.DataFrame(td_rows, columns=["token_id", "token_wording", "source_type", "coding", "value_bin"])
    td.to_csv(out_dir / "token_dictionary.csv", index=False)

    # labels.csv — one row per token_id (row index == token_id)
    max_id = max(r[0] for r in td_rows)
    id_to_name = {r[0]: r[1] for r in td_rows}
    labels = pd.DataFrame(
        {"event_name": [id_to_name.get(i, "") for i in range(max_id + 1)]}
    )
    labels.to_csv(out_dir / "labels.csv", index=False)

    # ignore_tokens: non-disease tokens that are in this config's vocab
    ignore = sorted({
        r[0] for r in td_rows if r[3] not in DISEASE_CODINGS
    })
    vocab_size = max_id + 1

    with open(out_dir / "config_values.py", "w") as f:
        f.write(f"vocab_size = {vocab_size}\n")
        f.write(f"ignore_tokens = {ignore}\n")

    print(f"  vocab_size={vocab_size}, ignore_tokens={ignore}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = [
    "ukb_amk125_clinical",
    "ukb_amk125_clinical_demographics",
    "ukb_amk125_clinical_demographics_ukb",
    "ukb_amk125_clinical_demographics_ukb_biochem",
]

N_PATIENTS = 500   # enough to exercise the pipeline quickly

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    data_root = Path("data")

    # Shared patient split (60/20/20)
    all_pids = np.arange(N_PATIENTS)
    rng.shuffle(all_pids)
    n_val  = int(0.20 * N_PATIENTS)
    n_test = int(0.20 * N_PATIENTS)
    val_pids   = set(all_pids[:n_val].tolist())
    test_pids  = set(all_pids[n_val:n_val + n_test].tolist())
    train_pids = set(all_pids[n_val + n_test:].tolist())

    for config_name in CONFIGS:
        print(f"\n=== {config_name} ===")
        out_dir = data_root / config_name
        out_dir.mkdir(parents=True, exist_ok=True)

        available = set(token_ids_for_config(config_name))
        all_rows = generate_patients(N_PATIENTS, available, rng)

        def split_and_reindex(pids_set):
            rows = [r for r in all_rows if r[0] in pids_set]
            # Re-index patient IDs consecutively
            pid_map = {p: i for i, p in enumerate(sorted({r[0] for r in rows}))}
            return [(pid_map[r[0]], r[1], r[2]) for r in rows]

        train_rows = split_and_reindex(train_pids)
        val_rows   = split_and_reindex(val_pids)
        test_rows  = split_and_reindex(test_pids)

        write_bin(train_rows, out_dir / "train.bin")
        write_bin(val_rows,   out_dir / "val.bin")
        write_bin(test_rows,  out_dir / "test.bin")
        write_metadata(out_dir, available)

    print("\nDone. Synthetic data written to data/")
