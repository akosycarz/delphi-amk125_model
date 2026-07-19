#!/usr/bin/env python3
"""
run_trajectory_evaluation.py — Delphi trajectory generation evaluation

Checks:
  1. Autoregressive trajectory generation for 5, 10, and 20-year horizons
     (replicating Delphi paper projections)
  2. Death token integrity:
       - Death token must appear only at the end of generated trajectories
       - No events may be generated after a death token
       - For patients with a death token in ground truth, checks recovery rate
  3. Age distribution consistency across train / val / test splits
     (validates the 60/20/20 re-split)

Outputs (written to --output_path):
  trajectory_disease_counts.csv   Per-disease incidence at each horizon, per sex
  trajectory_disease_counts.png   Bar chart: disease incidence at 5 / 10 / 20 years
  trajectory_sample.csv           Raw generated sequences (first --n_sample_save patients)
  death_validation.csv            Per-patient death-token integrity checks
  death_summary.txt               Aggregated death-token statistics
  age_distributions.png           Age-at-first-event histograms for train / val / test
  age_stats.csv                   Mean / median / std / KS test for each split pair

Usage:
  python run_trajectory_evaluation.py \\
    --input_path  data/ukb_amk125_clinical_demographics_ukb_biochem \\
    --model_ckpt_path out_clinical_demographics_ukb_biochem/ckpt.pt \\
    --output_path results/trajectory \\
    --split test \\
    --n_patients 200 \\
    --device cuda
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from tqdm import tqdm

from model import DelphiConfig, Delphi
from utils import get_p2i

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAYS_PER_YEAR = 365.25

# Model-facing token IDs (bin_token_id + 1 shift applied by get_batch)
PAD_TOKEN    = 0
NO_EVENT_TOKEN = 1
SEX_FEMALE_TOKEN = 2
SEX_MALE_TOKEN   = 3

DISEASE_CODINGS = {"ICD10", "self_reported_cancer", "self_reported_non_cancer"}

HORIZONS_YEARS = [5, 10, 20]  # projection horizons to evaluate


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Delphi trajectory generation evaluation")
    p.add_argument("--input_path", required=True,
                   help="Dataset folder (e.g. data/ukb_amk125_clinical_demographics_ukb_biochem)")
    p.add_argument("--model_ckpt_path", required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--output_path", required=True,
                   help="Directory where all output files are written")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Split to evaluate trajectories on (default: test)")
    p.add_argument("--n_patients", type=int, default=200,
                   help="Number of patients for trajectory generation (default: 200)")
    p.add_argument("--projection_start_age", type=float, default=50.0,
                   help="Age (years) from which to start generating future events (default: 50)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature. 1.0 = no change, <1 = more deterministic (default: 1.0)")
    p.add_argument("--max_new_tokens", type=int, default=500,
                   help="Safety cap on generated tokens per patient per horizon (default: 500)")
    p.add_argument("--n_sample_save", type=int, default=20,
                   help="Number of patient trajectories to write to trajectory_sample.csv (default: 20)")
    p.add_argument("--block_size", type=int, default=128,
                   help="Context window length (should match training, default: 128)")
    p.add_argument("--t_min", type=float, default=1.0,
                   help="Minimum time step in days (should match training, default: 1.0)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    conf = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(conf)
    state_dict = checkpoint["model"]
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device), conf


# ---------------------------------------------------------------------------
# Data loading (raw, no get_batch augmentation — we need original ages)
# ---------------------------------------------------------------------------

def load_raw_split(input_path, split):
    """Load split.bin as raw (patient_id, age_days, bin_token_id) array."""
    bin_path = Path(input_path) / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"{bin_path} not found.")
    data = np.fromfile(bin_path, dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    # bin_token_id → model token_id: add 1
    data[:, 2] = data[:, 2] + 1
    return data


def get_patient_context(data, p2i, pid, start_age_days, block_size):
    """
    Extract the context tokens and ages for patient `pid` up to `start_age_days`.
    Returns (tokens, ages) as 1-D numpy arrays, clamped to block_size.
    """
    s, l = p2i[pid]
    rows = data[s:s + l]
    # Keep only events before the projection start age
    mask = rows[:, 1] < start_age_days
    rows = rows[mask]
    if len(rows) == 0:
        return None, None
    tokens = rows[:, 2].astype(np.int64)
    ages   = rows[:, 1].astype(np.float32)
    # Keep the last block_size events (right-aligned context)
    if len(tokens) > block_size:
        tokens = tokens[-block_size:]
        ages   = ages[-block_size:]
    return tokens, ages


# ---------------------------------------------------------------------------
# Token dictionary
# ---------------------------------------------------------------------------

def load_token_dict(input_path):
    td = pd.read_csv(Path(input_path) / "token_dictionary.csv")
    # Identify death tokens by name
    td["is_death"] = td["token_wording"].str.lower().str.contains("death|died|mortality", na=False)
    td["is_disease"] = td["coding"].isin(DISEASE_CODINGS)
    # model_token_id = bin column token_id + 1 (already stored as token_id in csv, which IS model_id)
    return td


def get_death_token_ids(token_dict):
    """Return set of model-facing token IDs that represent death."""
    return set(token_dict[token_dict["is_death"]]["token_id"].tolist())


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_trajectory(
    model,
    context_tokens,   # 1-D int array
    context_ages,     # 1-D float array (days)
    horizon_days,     # float: how far into the future to generate
    death_token_ids,  # set of ints
    max_new_tokens,
    temperature,
    t_min,
    device,
):
    """
    Autoregressively generate events starting from `context_ages[-1]` until
    `current_age >= context_ages[-1] + horizon_days` or death is encountered.

    Returns:
        generated_tokens : list[int]
        generated_ages   : list[float]   (days)
        death_found      : bool
        events_after_death : int  (should always be 0 if implementation is correct)
    """
    tokens = list(context_tokens)
    ages   = list(context_ages)

    start_age = float(ages[-1]) if ages else 0.0
    end_age   = start_age + horizon_days

    generated_tokens = []
    generated_ages   = []
    death_found      = False
    events_after_death = 0

    for _ in range(max_new_tokens):
        current_age = ages[-1]
        if current_age >= end_age:
            break

        # Build input tensors (last block_size tokens)
        seq_tokens = tokens[-model.config.block_size:]
        seq_ages   = ages[-len(seq_tokens):]

        x = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
        a = torch.tensor(seq_ages,   dtype=torch.float32, device=device).unsqueeze(0)

        # Forward pass (inference mode: no targets)
        logits, _, _ = model(x, a)   # logits: (1, seq, vocab)
        last_logits = logits[0, -1, :]  # (vocab,)

        # --- Sample next token ---
        if temperature != 1.0:
            last_logits = last_logits / temperature
        # Zero out padding and no-event tokens from sampling
        last_logits[PAD_TOKEN] = -float("inf")
        last_logits[NO_EVENT_TOKEN] = -float("inf")

        probs = TF.softmax(last_logits, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())

        # --- Sample next time using exponential distribution ---
        # Model trains with: lse = -log(exp(-logsumexp(logits)) + t_min)
        # and loss_dt = -(lse - exp(lse) * (dt + t_min))  [Exponential NLL]
        # So: E[dt + t_min] = 1 / exp(lse) → dt = sample from Exp(rate) - t_min
        lse_raw  = torch.logsumexp(last_logits, dim=-1).item()
        rate     = 1.0 / (np.exp(-lse_raw) + t_min)
        dt       = float(np.random.exponential(1.0 / rate)) - t_min
        dt       = max(dt, 1.0)  # enforce t_min

        next_age = current_age + dt

        # Stop generation if we've gone past the horizon
        if next_age > end_age:
            break

        # Record event (even if it's a death token — then stop after)
        if death_found:
            events_after_death += 1

        generated_tokens.append(next_token)
        generated_ages.append(next_age)
        tokens.append(next_token)
        ages.append(next_age)

        if next_token in death_token_ids:
            death_found = True
            break  # no events after death

    return generated_tokens, generated_ages, death_found, events_after_death


# ---------------------------------------------------------------------------
# Age distribution check (split consistency)
# ---------------------------------------------------------------------------

def age_at_first_event(data, p2i):
    """
    Return an array of first-event ages (in years) for each patient.
    Uses the first non-sex, non-padding event in the trajectory.
    """
    ages = []
    for s, l in p2i:
        rows = data[s:s + l]
        # skip sex tokens (model ids 2, 3) and padding (0)
        clinical = rows[(rows[:, 2] > SEX_MALE_TOKEN)]
        if len(clinical) > 0:
            ages.append(clinical[0, 1] / DAYS_PER_YEAR)
        else:
            ages.append(rows[0, 1] / DAYS_PER_YEAR)
    return np.array(ages)


def check_age_distributions(input_path, out):
    """
    Load all three splits, compute age-at-first-event, and compare distributions.
    Prints KS test p-values and writes age_distributions.png + age_stats.csv.
    """
    print("\n--- Age distribution consistency check ---")
    split_ages = {}
    for split in ["train", "val", "test"]:
        bin_path = Path(input_path) / f"{split}.bin"
        if not bin_path.exists():
            print(f"  {split}.bin not found — skipping")
            continue
        data = np.fromfile(bin_path, dtype=np.uint32).reshape(-1, 3).astype(np.int64)
        data[:, 2] = data[:, 2] + 1  # shift to model IDs
        p2i = get_p2i(data)
        ages = age_at_first_event(data, p2i)
        split_ages[split] = ages
        print(f"  {split}: {len(ages):,} patients  "
              f"age mean={ages.mean():.1f}  median={np.median(ages):.1f}  "
              f"std={ages.std():.1f}  range=[{ages.min():.0f}, {ages.max():.0f}]")

    # KS tests between every pair
    stats_rows = []
    splits = list(split_ages.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            ks_stat, ks_p = stats.ks_2samp(split_ages[s1], split_ages[s2])
            note = "CONSISTENT" if ks_p > 0.05 else "DIFFERENT (p<0.05)"
            print(f"  KS test {s1} vs {s2}: stat={ks_stat:.4f}  p={ks_p:.4f}  [{note}]")
            stats_rows.append({
                "split_A": s1, "split_B": s2,
                "n_A": len(split_ages[s1]), "n_B": len(split_ages[s2]),
                "mean_A": split_ages[s1].mean(), "mean_B": split_ages[s2].mean(),
                "median_A": float(np.median(split_ages[s1])),
                "median_B": float(np.median(split_ages[s2])),
                "std_A": split_ages[s1].std(), "std_B": split_ages[s2].std(),
                "ks_stat": ks_stat, "ks_p": ks_p,
                "consistent": ks_p > 0.05,
            })

    # Save stats
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out / "age_stats.csv", index=False)
    print(f"  Saved age_stats.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"train": "#4e79a7", "val": "#f28e2b", "test": "#59a14f"}
    for split, ages in split_ages.items():
        ax.hist(ages, bins=40, alpha=0.55, color=colors.get(split, "#999"),
                label=f"{split} (n={len(ages):,})", density=True, edgecolor="white")
    ax.set_xlabel("Age at first clinical event (years)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Age distribution at first event — train / val / test splits\n"
                 "(distributions should overlap to confirm consistent 60/20/20 split)", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = out / "age_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Death token validation
# ---------------------------------------------------------------------------

def validate_death_tokens(data, p2i, death_token_ids, n_patients=None):
    """
    For each patient in the raw ground-truth data:
      - Check if they have a death token
      - Verify it is the LAST event in their trajectory

    Returns DataFrame with per-patient results.
    """
    n = len(p2i) if n_patients is None else min(n_patients, len(p2i))
    rows = []
    for pid in range(n):
        s, l = p2i[pid]
        patient_tokens = data[s:s + l, 2]  # already model-facing IDs
        has_death = any(t in death_token_ids for t in patient_tokens)
        if not has_death:
            continue
        death_positions = [i for i, t in enumerate(patient_tokens) if t in death_token_ids]
        last_death_pos  = max(death_positions)
        last_pos        = len(patient_tokens) - 1
        events_after    = last_pos - last_death_pos  # should be 0
        rows.append({
            "patient_id":        pid,
            "n_events":          int(l),
            "n_death_tokens":    len(death_positions),
            "death_at_last_pos": last_death_pos == last_pos,
            "events_after_death": events_after,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trajectory generation and analysis
# ---------------------------------------------------------------------------

def run_trajectory_generation(
    model, conf, data, p2i, token_dict, death_token_ids,
    args, out
):
    """
    Generate trajectories for args.n_patients patients at 5, 10, 20-year horizons.
    Produces:
      - trajectory_disease_counts.csv / .png
      - trajectory_sample.csv
      - death_validation.csv / death_summary.txt  (generated trajectories)
    """
    n = min(args.n_patients, len(p2i))
    start_age_days = args.projection_start_age * DAYS_PER_YEAR

    # Determine sex for each patient
    def patient_sex(pid):
        s, l = p2i[pid]
        toks = data[s:s + l, 2]
        if SEX_FEMALE_TOKEN in toks:
            return "female"
        if SEX_MALE_TOKEN in toks:
            return "male"
        return "unknown"

    # Disease tokens (for counting incidence in generated trajectories)
    disease_mask = token_dict["is_disease"]
    disease_ids  = set(token_dict[disease_mask]["token_id"].tolist())
    id_to_name   = dict(zip(token_dict["token_id"], token_dict["token_wording"]))

    # ---- Generate trajectories -----------------------------------------------
    print(f"\n--- Generating trajectories for {n} patients ---")
    print(f"    Projection start age: {args.projection_start_age:.0f} years")
    print(f"    Horizons: {HORIZONS_YEARS} years")
    print(f"    Temperature: {args.temperature}")

    all_results   = []   # one dict per (patient, horizon)
    sample_rows   = []   # raw event log for the first n_sample_save patients
    death_val_rows = []  # death-token integrity in generated trajectories

    for pid in tqdm(range(n), desc="Generating"):
        context_tokens, context_ages = get_patient_context(
            data, p2i, pid, start_age_days, args.block_size
        )
        if context_tokens is None or len(context_tokens) == 0:
            continue

        sex = patient_sex(pid)

        for horizon_years in HORIZONS_YEARS:
            horizon_days = horizon_years * DAYS_PER_YEAR

            gen_tokens, gen_ages, death_found, events_after = generate_trajectory(
                model=model,
                context_tokens=context_tokens,
                context_ages=context_ages,
                horizon_days=horizon_days,
                death_token_ids=death_token_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                t_min=args.t_min,
                device=args.device,
            )

            # Count disease events at this horizon
            disease_events = [t for t in gen_tokens if t in disease_ids]
            disease_counts = {}
            for t in disease_events:
                name = id_to_name.get(t, str(t))
                disease_counts[name] = disease_counts.get(name, 0) + 1

            all_results.append({
                "patient_id":          pid,
                "sex":                 sex,
                "horizon_years":       horizon_years,
                "n_generated":         len(gen_tokens),
                "n_disease_events":    len(disease_events),
                "n_unique_diseases":   len(set(disease_events)),
                "death_found":         death_found,
                "events_after_death":  events_after,
            })

            # Integrity check on generated trajectory
            if horizon_years == max(HORIZONS_YEARS):  # check only on longest horizon
                death_val_rows.append({
                    "patient_id":            pid,
                    "horizon_years":         horizon_years,
                    "death_found":           death_found,
                    "events_after_death":    events_after,
                    "generation_correct":    events_after == 0,
                })

            # Save sample trajectories
            if pid < args.n_sample_save:
                for tok, age in zip(gen_tokens, gen_ages):
                    sample_rows.append({
                        "patient_id":   pid,
                        "sex":          sex,
                        "horizon_years": horizon_years,
                        "age_days":     age,
                        "age_years":    age / DAYS_PER_YEAR,
                        "token_id":     tok,
                        "token_wording": id_to_name.get(tok, str(tok)),
                        "is_disease":   tok in disease_ids,
                        "is_death":     tok in death_token_ids,
                    })

    # ---- Save raw results ----------------------------------------------------
    results_df = pd.DataFrame(all_results)
    sample_df  = pd.DataFrame(sample_rows)
    death_df   = pd.DataFrame(death_val_rows)

    sample_df.to_csv(out / "trajectory_sample.csv", index=False)
    print(f"  Saved trajectory_sample.csv ({len(sample_df)} rows)")

    if len(death_df) > 0:
        death_df.to_csv(out / "death_validation_generated.csv", index=False)
        total_correct = death_df["generation_correct"].sum()
        total = len(death_df)
        n_with_death = death_df["death_found"].sum()
        n_events_after = (death_df["events_after_death"] > 0).sum()
        summary = (
            f"Death token validation (generated trajectories at {max(HORIZONS_YEARS)}-year horizon)\n"
            f"  Patients generated: {total}\n"
            f"  Trajectories ending with death: {n_with_death} ({100*n_with_death/total:.1f}%)\n"
            f"  Trajectories with events AFTER death: {n_events_after} "
            f"({'ERROR' if n_events_after > 0 else 'OK — none'})\n"
            f"  Generation integrity: {total_correct}/{total} correct\n"
        )
        print("\n" + summary)
        with open(out / "death_summary.txt", "w") as f:
            f.write(summary)
        print(f"  Saved death_summary.txt")
    else:
        print("  No death tokens found in generated trajectories "
              "(either model has not learned them or death tokens absent from vocab)")
        with open(out / "death_summary.txt", "w") as f:
            f.write("No death tokens generated. Check if death tokens are present in the vocabulary.\n")

    # ---- Disease incidence counts per horizon --------------------------------
    if len(results_df) == 0:
        print("No trajectory results generated.")
        return

    count_rows = []
    for horizon in HORIZONS_YEARS:
        sub = results_df[results_df["horizon_years"] == horizon]
        for sex in ["female", "male", "all"]:
            if sex == "all":
                grp = sub
            else:
                grp = sub[sub["sex"] == sex]
            n_grp = len(grp)
            if n_grp == 0:
                continue
            count_rows.append({
                "horizon_years":         horizon,
                "sex":                   sex,
                "n_patients":            n_grp,
                "mean_disease_events":   grp["n_disease_events"].mean(),
                "median_disease_events": grp["n_disease_events"].median(),
                "mean_unique_diseases":  grp["n_unique_diseases"].mean(),
                "pct_any_disease":       (grp["n_disease_events"] > 0).mean() * 100,
                "pct_death_found":       grp["death_found"].mean() * 100,
            })

    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(out / "trajectory_disease_counts.csv", index=False)
    print(f"  Saved trajectory_disease_counts.csv")

    # Print summary table
    print("\n  Disease incidence by horizon:")
    for _, r in counts_df[counts_df["sex"] == "all"].iterrows():
        print(f"    {r['horizon_years']:2d} years | n={r['n_patients']:4d} | "
              f"mean disease events={r['mean_disease_events']:.2f} | "
              f"any disease={r['pct_any_disease']:.1f}% | "
              f"death generated={r['pct_death_found']:.1f}%")

    # ---- Plot: disease events by horizon and sex ----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Left: mean number of disease events per patient at each horizon
    ax = axes[0]
    for sex, color in [("female", "#e15759"), ("male", "#4e79a7")]:
        sub = counts_df[counts_df["sex"] == sex]
        if len(sub) == 0:
            continue
        ax.plot(sub["horizon_years"], sub["mean_disease_events"],
                marker="o", color=color, linewidth=2,
                label=f"{sex.capitalize()}")
        ax.fill_between(
            sub["horizon_years"],
            sub["mean_disease_events"] - sub["median_disease_events"],
            sub["mean_disease_events"] + sub["median_disease_events"],
            alpha=0.12, color=color,
        )
    ax.set_xlabel("Horizon (years)", fontsize=11)
    ax.set_ylabel("Mean disease events per patient", fontsize=11)
    ax.set_title("Predicted disease burden by projection horizon", fontsize=12)
    ax.set_xticks(HORIZONS_YEARS)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Right: % patients with ≥1 disease event at each horizon
    ax = axes[1]
    for sex, color in [("female", "#e15759"), ("male", "#4e79a7")]:
        sub = counts_df[counts_df["sex"] == sex]
        if len(sub) == 0:
            continue
        ax.plot(sub["horizon_years"], sub["pct_any_disease"],
                marker="o", color=color, linewidth=2,
                label=f"{sex.capitalize()}")
    ax.set_xlabel("Horizon (years)", fontsize=11)
    ax.set_ylabel("% patients with ≥1 predicted disease", fontsize=11)
    ax.set_title("Cumulative disease incidence by horizon", fontsize=12)
    ax.set_xticks(HORIZONS_YEARS)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Trajectory generation — {n} patients, starting age {args.projection_start_age:.0f}",
                 fontsize=13)
    fig.tight_layout()
    path = out / "trajectory_disease_counts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Delphi — Trajectory Generation Evaluation")
    print("=" * 60)

    # ---- Load token dictionary ----------------------------------------------
    print("\nLoading token dictionary ...")
    token_dict = load_token_dict(args.input_path)
    death_token_ids = get_death_token_ids(token_dict)

    if death_token_ids:
        names = token_dict[token_dict["token_id"].isin(death_token_ids)]["token_wording"].tolist()
        print(f"  Death tokens found: {death_token_ids}  ({names})")
    else:
        print("  WARNING: No death tokens found in token_dictionary.csv. "
              "Death validation will be skipped.")

    n_disease = token_dict["is_disease"].sum()
    print(f"  Disease tokens in vocabulary: {n_disease}")

    # ---- Load model ---------------------------------------------------------
    print("\nLoading model ...")
    model, conf = load_model(args.model_ckpt_path, args.device)
    print(f"  vocab_size={conf.vocab_size}  block_size={conf.block_size}")

    # ---- Load evaluation split (raw, for generation) ------------------------
    print(f"\nLoading {args.split} split ...")
    data = load_raw_split(args.input_path, args.split)
    p2i  = get_p2i(data)
    print(f"  {len(p2i):,} patients | {data.shape[0]:,} events")

    # ---- Ground-truth death validation (before generation) ------------------
    if death_token_ids:
        print("\n--- Ground-truth death token validation ---")
        gt_death_df = validate_death_tokens(data, p2i, death_token_ids)
        if len(gt_death_df) > 0:
            n_correct  = gt_death_df["death_at_last_pos"].sum()
            n_after    = (gt_death_df["events_after_death"] > 0).sum()
            print(f"  Patients with death token:          {len(gt_death_df):,}")
            print(f"  Death at last position (expected):  {n_correct:,} / {len(gt_death_df):,} "
                  f"({100*n_correct/len(gt_death_df):.1f}%)")
            print(f"  Events AFTER death token (errors):  {n_after:,} "
                  f"({'OK' if n_after == 0 else 'PROBLEM — check data'})")
            gt_death_df.to_csv(out / "death_validation_groundtruth.csv", index=False)
            print(f"  Saved death_validation_groundtruth.csv")
        else:
            print("  No patients with death tokens found in ground-truth data.")
    else:
        print("\n  Skipping ground-truth death validation (no death tokens in vocab).")

    # ---- Trajectory generation and analysis ---------------------------------
    run_trajectory_generation(
        model=model, conf=conf, data=data, p2i=p2i,
        token_dict=token_dict, death_token_ids=death_token_ids,
        args=args, out=out,
    )

    # ---- Age distribution consistency check ---------------------------------
    check_age_distributions(args.input_path, out)

    print(f"\n{'='*60}")
    print(f"  Done. All outputs written to: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
