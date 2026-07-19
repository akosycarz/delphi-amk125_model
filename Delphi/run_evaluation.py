#!/usr/bin/env python3
"""
run_evaluation.py — Comprehensive Delphi model evaluation

Produces (all written to --output_path):
  auc_table.csv            Per-disease AUC + counts, overall and by sex,
                           with 95% CI and mean probability for cases vs controls
  auc_vs_count.png         AUC as a function of event count (identify threshold)
  auc_distribution.png     AUC histogram above threshold, female vs male
  auc_by_age.png           Mean AUC across 5-year age bins, female vs male
  auc_female_vs_male.png   Female vs male AUC scatter, coloured by ICD chapter

Design notes:
  - Only disease tokens (ICD10, self_reported) are evaluated — biochemistry and
    demographic tokens are excluded from predictions.
  - "Other *" catch-all tokens are excluded from results (kept in training).
  - Predictions are only counted when made at least --offset days before the event.
  - Sex is determined from the input sequence (token 2 = Female, 3 = Male).
  - Count refers to the number of patients who develop each disease in the split.
  - Probabilities are reported as mean softmax probability at prediction time.

Usage:
  python run_evaluation.py \\
    --input_path  data/ukb_amk125_clinical_demographics_ukb_biochem \\
    --model_ckpt_path out_clinical_demographics_ukb_biochem/ckpt.pt \\
    --output_path results/full_model \\
    --count_threshold 1000 \\
    --icd10_names icd10_codes.csv    # optional; columns: code, description
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from model import DelphiConfig, Delphi
from utils import get_batch, get_p2i
from evaluate_auc import fastDeLong, compute_ground_truth_statistics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISEASE_CODINGS = {"ICD10", "self_reported_cancer", "self_reported_non_cancer"}

# Model-facing token IDs (after get_batch adds +1)
SEX_FEMALE_TOKEN = 2
SEX_MALE_TOKEN   = 3

# Simplified ICD-10 chapter lookup by first letter of code
ICD10_CHAPTERS = {
    "A": "Infectious",        "B": "Infectious",
    "C": "Neoplasms",         "D": "Neoplasms",
    "E": "Metabolic",
    "F": "Mental",
    "G": "Nervous system",
    "H": "Eye & ear",
    "I": "Circulatory",
    "J": "Respiratory",
    "K": "Digestive",
    "L": "Skin",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
    "O": "Pregnancy",
    "P": "Perinatal",
    "Q": "Congenital",
    "R": "Symptoms & signs",
    "S": "Injury",             "T": "Injury",
    "Z": "Factors/Health",
}

CHAPTER_COLORS = {
    "Infectious":       "#4e79a7",
    "Neoplasms":        "#f28e2b",
    "Metabolic":        "#59a14f",
    "Mental":           "#e15759",
    "Nervous system":   "#76b7b2",
    "Eye & ear":        "#edc948",
    "Circulatory":      "#b07aa1",
    "Respiratory":      "#ff9da7",
    "Digestive":        "#9c755f",
    "Skin":             "#bab0ac",
    "Musculoskeletal":  "#d37295",
    "Genitourinary":    "#a0cbe8",
    "Injury":           "#8cd17d",
    "Pregnancy":        "#86bcb6",
    "Symptoms & signs": "#ffbe7d",
    "Factors/Health":   "#d4a6c8",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive Delphi model evaluation")
    p.add_argument("--input_path", required=True,
                   help="Dataset folder (e.g. data/ukb_amk125_clinical_demographics_ukb_biochem)")
    p.add_argument("--model_ckpt_path", required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--output_path", required=True,
                   help="Directory where all output files are written")
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="Data split to evaluate on. Use 'test' for final evaluation (default)")
    p.add_argument("--block_size", type=int, default=128,
                   help="Context window length (should match training)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Inference batch size")
    p.add_argument("--no_event_token_rate", type=int, default=5,
                   help="No-event token insertion rate (should match training)")
    p.add_argument("--count_threshold", type=int, default=1000,
                   help="Minimum number of cases to include a disease in summary plots "
                        "(default: 1000; suggested range 1000–2000)")
    p.add_argument("--offset", type=float, default=365.25,
                   help="Minimum days between last observation and event for a prediction "
                        "to be counted (default: 365.25 = 1 year)")
    p.add_argument("--age_min", type=int, default=40,
                   help="Minimum age (years) for age-group analysis")
    p.add_argument("--age_max", type=int, default=80,
                   help="Maximum age (years) for age-group analysis (exclusive)")
    p.add_argument("--age_step", type=int, default=5,
                   help="Width of each age bin in years (default: 5)")
    p.add_argument("--icd10_names", default=None,
                   help="Optional CSV with columns 'code' and 'description' mapping ICD-10 "
                        "codes to readable names. Download from NHS/WHO.")
    p.add_argument("--dataset_subset_size", type=int, default=-1,
                   help="Limit evaluation to N patients (-1 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    conf = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(conf)
    state_dict = checkpoint["model"]
    # Strip compile prefix if present
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(input_path, split, block_size, no_event_token_rate, subset_size):
    bin_path = Path(input_path) / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"{bin_path} not found. Run scripts/delphi_preprocess.R first."
        )
    print(f"Loading {split} split from {bin_path}")
    data = np.fromfile(bin_path, dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(data)
    n = len(p2i) if subset_size == -1 else min(subset_size, len(p2i))
    print(f"  {n:,} patients | {data.shape[0]:,} events")
    batch = get_batch(
        range(n), data, p2i,
        select="left",
        block_size=block_size,
        device="cpu",
        padding="random",
        no_event_token_rate=no_event_token_rate,
    )
    return batch  # (x, a, y, b) CPU tensors


# ---------------------------------------------------------------------------
# Token dictionary
# ---------------------------------------------------------------------------

def load_token_dict(input_path, icd10_names_path=None):
    """
    Load token_dictionary.csv from the dataset folder.
    Adds icd_code, chapter, and readable_name columns.
    """
    td = pd.read_csv(Path(input_path) / "token_dictionary.csv")

    def extract_icd_code(row):
        if row["coding"] == "ICD10" and isinstance(row.get("token_wording"), str):
            parts = row["token_wording"].split("::")
            return parts[-1] if len(parts) >= 2 else None
        return None

    td["icd_code"] = td.apply(extract_icd_code, axis=1)
    td["chapter"] = td["icd_code"].apply(
        lambda c: ICD10_CHAPTERS.get(c[0], "Unknown")
        if isinstance(c, str) and len(c) > 0 else None
    )
    td["readable_name"] = td["token_wording"]

    if icd10_names_path and Path(icd10_names_path).exists():
        names = pd.read_csv(icd10_names_path)
        names.columns = names.columns.str.lower().str.strip()
        if {"code", "description"}.issubset(names.columns):
            names = names[["code", "description"]].drop_duplicates("code")
            td = td.merge(names, left_on="icd_code", right_on="code", how="left")
            td["readable_name"] = td["description"].fillna(td["token_wording"])
            td.drop(columns=["code", "description"], inplace=True, errors="ignore")
        else:
            print("Warning: --icd10_names CSV must have columns 'code' and 'description'.")

    return td


def get_disease_token_df(token_dict):
    """
    Return rows for disease tokens only.
    Excludes: biochemistry, sex/demographic tokens, and 'other *' catch-alls.
    """
    mask = (
        token_dict["coding"].isin(DISEASE_CODINGS) &
        ~token_dict["token_wording"].str.lower().str.contains("other", na=False)
    )
    cols = ["token_id", "token_wording", "icd_code", "chapter", "readable_name"]
    return token_dict[mask][cols].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, batch, disease_token_ids, batch_size, device):
    """
    Forward pass over all patients, returning logits for disease tokens only.

    Returns
    -------
    logits : np.ndarray, shape (n_patients, seq_len, n_diseases), float16
    probs  : np.ndarray, shape (n_patients, seq_len, n_diseases), float16
        Softmax probabilities over the full vocabulary, then subset to disease tokens.
        Use these for reporting mean probability rather than raw logits.
    """
    x, a, y, b = batch
    n = x.shape[0]
    d_ids = disease_token_ids  # list of int

    logit_chunks = []
    prob_chunks  = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inference"):
            end = min(start + batch_size, n)
            bx = x[start:end].to(device)
            ba = a[start:end].to(device)
            by = y[start:end].to(device)
            bb = b[start:end].to(device)
            raw = model(bx, ba, by, bb)[0].cpu().float()  # (bs, seq, vocab)
            logit_chunks.append(raw[:, :, d_ids].numpy().astype(np.float16))
            prob_chunks.append(
                F.softmax(raw, dim=-1)[:, :, d_ids].numpy().astype(np.float16)
            )

    return (
        np.concatenate(logit_chunks, axis=0),
        np.concatenate(prob_chunks,  axis=0),
    )


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------

def _delong_auc(scores_case, scores_ctrl):
    """
    DeLong AUC with 95% CI.
    Returns (auc, ci_lower, ci_upper) or None if insufficient data.
    """
    if len(scores_case) < 2 or len(scores_ctrl) < 2:
        return None
    scores_case = scores_case.astype(np.float32)
    scores_ctrl = scores_ctrl.astype(np.float32)
    all_scores = np.concatenate([scores_case, scores_ctrl])
    labels     = np.array([1] * len(scores_case) + [0] * len(scores_ctrl))
    try:
        order, n_pos = compute_ground_truth_statistics(labels)
        auc_arr, var = fastDeLong(all_scores[np.newaxis, order], n_pos)
    except Exception:
        return None
    a   = float(auc_arr[0])
    ci  = 1.96 * float(np.sqrt(max(float(var), 0.0)))
    return a, max(0.0, a - ci), min(1.0, a + ci)


def compute_disease_metrics(j, k, logits, probs, batch_np, sex_mask, age_groups, age_step, offset):
    """
    Compute AUC, counts, and mean probabilities for disease token k
    (column index j in logits/probs) for the subset of patients in sex_mask.

    Cases    : patients whose target sequence contains token k, with a valid
               prediction at least `offset` days before the event.
    Controls : patients who never develop disease k; prediction is taken at
               their last valid sequence position.

    Returns a flat dict or None if insufficient data.
    """
    x_np, a_np, y_np, b_np = [d[sex_mask] for d in batch_np]
    lp   = logits[sex_mask]
    prob = probs[sex_mask]

    # ---- Cases ----
    # Find every (patient, time) position where target token == k
    wk = np.where(y_np == k)
    if len(wk[0]) < 2:
        return None

    # For each case event: last position at least `offset` days before the event
    case_pred_idx = (a_np[wk[0]] < b_np[wk].reshape(-1, 1) - offset).sum(1) - 1
    valid = case_pred_idx >= 0
    if valid.sum() < 2:
        return None

    c_pts = wk[0][valid]
    c_idx = case_pred_idx[valid]

    # One prediction per case patient (earliest valid event)
    _, first = np.unique(c_pts, return_index=True)
    c_pts = c_pts[first]
    c_idx = c_idx[first]

    case_logits = lp[c_pts, c_idx, j].astype(np.float32)
    case_probs  = prob[c_pts, c_idx, j].astype(np.float32)
    case_ages   = a_np[c_pts, c_idx] / 365.25  # years

    # ---- Controls ----
    has_disease = np.zeros(x_np.shape[0], dtype=bool)
    has_disease[wk[0]] = True
    ctrl_patients = np.where(~has_disease)[0]
    if len(ctrl_patients) < 2:
        return None

    # One prediction per control: last valid token position
    ctrl_last  = np.clip((a_np[ctrl_patients] > -1000).sum(1) - 1, 0, a_np.shape[1] - 1)
    ctrl_logits = lp[ctrl_patients, ctrl_last, j].astype(np.float32)
    ctrl_probs  = prob[ctrl_patients, ctrl_last, j].astype(np.float32)
    ctrl_ages   = a_np[ctrl_patients, ctrl_last] / 365.25

    # ---- Overall AUC ----
    overall = _delong_auc(case_logits, ctrl_logits)
    if overall is None:
        return None

    result = {
        "count":              int(len(c_pts)),
        "n_controls":         int(len(ctrl_patients)),
        "auc":                overall[0],
        "auc_ci_lower":       overall[1],
        "auc_ci_upper":       overall[2],
        "mean_prob_case":     float(np.mean(case_probs)),
        "mean_prob_control":  float(np.mean(ctrl_probs)),
    }

    # ---- Per-age-group AUC ----
    for ag in age_groups:
        col = f"auc_{ag}_{ag + age_step}"
        in_case = (case_ages >= ag) & (case_ages < ag + age_step)
        in_ctrl = (ctrl_ages >= ag) & (ctrl_ages < ag + age_step)
        ag_res = _delong_auc(case_logits[in_case], ctrl_logits[in_ctrl])
        result[col]                         = ag_res[0] if ag_res else np.nan
        result[f"count_{ag}_{ag + age_step}"] = int(in_case.sum())

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _chapter_color(chapter):
    return CHAPTER_COLORS.get(chapter, "#aaaaaa")


def _chapter_legend(ax, chapters_present):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CHAPTER_COLORS.get(ch, "#aaaaaa"),
                   markersize=7, label=ch)
        for ch in CHAPTER_COLORS
        if ch in chapters_present
    ]
    if handles:
        ax.legend(handles=handles, fontsize=6, loc="lower right", ncol=1,
                  framealpha=0.7)


def plot_auc_vs_count(auc_df, count_threshold, out):
    """
    Scatter: AUC vs event count for each disease, coloured by ICD chapter.
    Includes a vertical line at the count threshold and a horizontal reference at 0.5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, sex in zip(axes, ["female", "male"]):
        sub = auc_df[auc_df["sex"] == sex].dropna(subset=["auc"])
        colors = sub["chapter"].map(_chapter_color).fillna("#aaaaaa")
        ax.scatter(sub["count"], sub["auc"], c=colors, alpha=0.55, s=18, linewidths=0)
        ax.axvline(count_threshold, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Threshold = {count_threshold:,}")
        ax.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Event count (log scale)", fontsize=11)
        ax.set_ylabel("AUC", fontsize=11)
        ax.set_title(sex.capitalize(), fontsize=12)
        ax.set_ylim(0.3, 1.02)
        ax.legend(fontsize=9)

    _chapter_legend(axes[1], auc_df["chapter"].dropna().unique())
    fig.suptitle("AUC vs Event Count — use this to choose a count threshold", fontsize=13)
    fig.tight_layout()
    path = out / "auc_vs_count.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_auc_distribution(auc_df, count_threshold, out):
    """
    Histogram of AUC values for diseases above the count threshold, female vs male.
    """
    above = auc_df[auc_df["count"] >= count_threshold]
    fig, ax = plt.subplots(figsize=(8, 4))
    for sex, color in [("female", "#e15759"), ("male", "#4e79a7")]:
        vals = above[above["sex"] == sex]["auc"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=20, alpha=0.6, color=color,
                label=f"{sex.capitalize()} (n={len(vals)})", edgecolor="white")
        ax.axvline(vals.median(), color=color, linestyle="--", linewidth=1.5,
                   label=f"{sex.capitalize()} median = {vals.median():.3f}")
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=0.8, label="0.5 (chance)")
    ax.set_xlabel("AUC", fontsize=11)
    ax.set_ylabel("Number of diseases", fontsize=11)
    ax.set_title(f"AUC distribution (count ≥ {count_threshold:,})", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out / "auc_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_auc_by_age(auc_df, count_threshold, out, age_groups, age_step):
    """
    Mean AUC ± SEM across 5-year age groups for diseases above the count threshold.
    """
    age_cols = [f"auc_{ag}_{ag + age_step}" for ag in age_groups]
    x_labels = [f"{ag}–{ag + age_step}" for ag in age_groups]
    above = auc_df[auc_df["count"] >= count_threshold]

    fig, ax = plt.subplots(figsize=(10, 5))
    for sex, color in [("female", "#e15759"), ("male", "#4e79a7")]:
        sub = above[above["sex"] == sex]
        means = [sub[c].dropna().mean() for c in age_cols]
        sems  = [sub[c].dropna().sem()  for c in age_cols]
        ax.plot(x_labels, means, color=color, marker="o", linewidth=1.8,
                label=f"{sex.capitalize()} (n diseases = {len(sub)})")
        ax.fill_between(x_labels,
                        [m - s for m, s in zip(means, sems)],
                        [m + s for m, s in zip(means, sems)],
                        alpha=0.15, color=color)

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Age group (years)", fontsize=11)
    ax.set_ylabel("Mean AUC ± SEM", fontsize=11)
    ax.set_title(f"AUC by 5-year age group (count ≥ {count_threshold:,})", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = out / "auc_by_age.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_female_vs_male(auc_df, count_threshold, out):
    """
    Scatter of female AUC vs male AUC for diseases above the count threshold.
    Points above the diagonal = higher AUC in males; below = higher in females.
    """
    above = auc_df[auc_df["count"] >= count_threshold]
    female = above[above["sex"] == "female"][
        ["token_id", "auc", "chapter", "readable_name"]
    ].rename(columns={"auc": "auc_f"})
    male = above[above["sex"] == "male"][
        ["token_id", "auc"]
    ].rename(columns={"auc": "auc_m"})
    merged = female.merge(male, on="token_id").dropna(subset=["auc_f", "auc_m"])

    if len(merged) < 2:
        print("  Skipping auc_female_vs_male.png (insufficient data)")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = merged["chapter"].map(_chapter_color).fillna("#aaaaaa")
    ax.scatter(merged["auc_f"], merged["auc_m"], c=colors, alpha=0.7, s=22, linewidths=0)
    lo, hi = 0.4, 1.0
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("AUC (female)", fontsize=11)
    ax.set_ylabel("AUC (male)", fontsize=11)
    ax.set_title(f"Female vs male AUC (count ≥ {count_threshold:,})", fontsize=12)
    _chapter_legend(ax, merged["chapter"].dropna().unique())
    fig.tight_layout()
    path = out / "auc_female_vs_male.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    age_groups = list(range(args.age_min, args.age_max, args.age_step))

    # ---- Load model ---------------------------------------------------------
    print("Loading model ...")
    model = load_model(args.model_ckpt_path, args.device)

    # ---- Load data ----------------------------------------------------------
    batch = load_split(
        args.input_path, args.split,
        args.block_size, args.no_event_token_rate, args.dataset_subset_size,
    )
    x, a, y, b = batch
    batch_np = [t.numpy() for t in [x, a, y, b]]

    # ---- Load token dictionary ----------------------------------------------
    token_dict  = load_token_dict(args.input_path, args.icd10_names)
    disease_df  = get_disease_token_df(token_dict)
    disease_ids = disease_df["token_id"].tolist()
    tok_to_j    = {tok: j for j, tok in enumerate(disease_ids)}
    print(f"Evaluating {len(disease_ids)} disease tokens "
          f"(excluding biochemistry, sex, and 'other' catch-alls)")

    # ---- Run inference ------------------------------------------------------
    print("Running inference ...")
    logits, probs = run_inference(model, batch, disease_ids, args.batch_size, args.device)
    # logits/probs: (n_patients, seq_len, n_diseases)

    # ---- Sex masks ----------------------------------------------------------
    x_np = batch_np[0]
    sex_masks = {
        "female": (x_np == SEX_FEMALE_TOKEN).any(axis=1),
        "male":   (x_np == SEX_MALE_TOKEN).any(axis=1),
    }
    for sex, mask in sex_masks.items():
        print(f"  {sex}: {mask.sum():,} patients")

    # ---- Compute AUC per disease × sex --------------------------------------
    print("Computing AUCs ...")
    rows = []
    for _, row in tqdm(disease_df.iterrows(), total=len(disease_df), desc="Diseases"):
        k = int(row["token_id"])
        j = tok_to_j[k]
        for sex, mask in sex_masks.items():
            res = compute_disease_metrics(
                j, k, logits, probs, batch_np, mask,
                age_groups, args.age_step, args.offset,
            )
            if res is None:
                continue
            entry = {
                "token_id":      k,
                "token_wording": row["token_wording"],
                "icd_code":      row.get("icd_code"),
                "readable_name": row.get("readable_name", row["token_wording"]),
                "chapter":       row.get("chapter"),
                "sex":           sex,
            }
            entry.update(res)
            rows.append(entry)

    if not rows:
        print("No AUC results computed — check that the dataset contains disease events.")
        sys.exit(1)

    auc_df = pd.DataFrame(rows)

    # ---- Save results -------------------------------------------------------
    csv_path = out / "auc_table.csv"
    auc_df.to_csv(csv_path, index=False)
    print(f"\nSaved auc_table.csv  ({len(auc_df)} rows, "
          f"{auc_df['token_id'].nunique()} unique diseases)")

    # Summary stats
    above = auc_df[auc_df["count"] >= args.count_threshold]
    print(f"\nDiseases above count threshold ({args.count_threshold:,}):")
    for sex in ["female", "male"]:
        sub = above[above["sex"] == sex]
        if len(sub):
            print(f"  {sex}: n={len(sub)}  "
                  f"median AUC={sub['auc'].median():.3f}  "
                  f"(range {sub['auc'].min():.3f}–{sub['auc'].max():.3f})")

    # ---- Plots --------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_auc_vs_count(auc_df, args.count_threshold, out)
    plot_auc_distribution(auc_df, args.count_threshold, out)
    plot_auc_by_age(auc_df, args.count_threshold, out, age_groups, args.age_step)
    plot_female_vs_male(auc_df, args.count_threshold, out)

    print(f"\nDone. All outputs written to: {out}")


if __name__ == "__main__":
    main()
