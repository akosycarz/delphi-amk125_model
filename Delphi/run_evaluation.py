#!/usr/bin/env python3
"""Comprehensive Delphi model evaluation with ROC/PR plots and calibration metrics."""

import argparse
import gc
import sys
from pathlib import Path

# This script may be stored in Delphi/plots while the imported Delphi modules
# remain one directory above it.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "model.py").is_file() else SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_auc import auc as evaluate_auc_rank
from evaluate_auc import get_auc_delong_var
from model import Delphi, DelphiConfig
from utils import get_batch, get_p2i


# These values must match token_dictionary.csv exactly. In this dataset,
# historical/self-reported diagnoses are represented by ICD9 rather than by
# separate self_reported_* coding labels.
DISEASE_CODINGS = {"ICD10", "ICD9", "DEATH"}
ICD10_CHAPTERS = {
    **dict.fromkeys("AB", "Infectious"), **dict.fromkeys("CD", "Neoplasms"),
    "E": "Metabolic", "F": "Mental", "G": "Nervous system", "H": "Eye & ear",
    "I": "Circulatory", "J": "Respiratory", "K": "Digestive", "L": "Skin",
    "M": "Musculoskeletal", "N": "Genitourinary", "O": "Pregnancy",
    "P": "Perinatal", "Q": "Congenital", "R": "Symptoms & signs",
    **dict.fromkeys("ST", "Injury"), "Z": "Factors/Health",
}
CHAPTER_COLORS = {
    name: plt.get_cmap("tab20")(i)
    for i, name in enumerate(dict.fromkeys(ICD10_CHAPTERS.values()))
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_path", required=True)
    p.add_argument("--model_ckpt_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument(
        "--block_size", type=int, default=None,
        help="Evaluation context length; defaults to the checkpoint block_size",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no_event_token_rate", type=int, default=5)
    p.add_argument("--count_threshold", type=int, default=1000)
    p.add_argument("--offset", type=float, default=365.25)
    p.add_argument("--age_min", type=int, default=40)
    p.add_argument("--age_max", type=int, default=80)
    p.add_argument("--age_step", type=int, default=5)
    p.add_argument("--icd10_names")
    p.add_argument("--dataset_subset_size", type=int, default=-1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max_curve_plots", type=int, default=15,
                   help="Number of frequent diseases retained for ROC/PR export (default: 15)")
    p.add_argument("--model_name",
                   help="Label stored in pr_curves.csv; defaults to checkpoint directory")
    p.add_argument("--selected_icd_codes", nargs="*",
                   help="Evaluate only these ICD-10 codes (fast targeted PR rerun)")
    p.add_argument("--select_from_auc_table", type=Path,
                   help="Select frequent full-population ICD-10 codes from an existing auc_table.csv")
    p.add_argument("--selected_outcomes", type=int, default=15,
                   help="Number selected by --select_from_auc_table (default: 15)")
    p.add_argument("--extended_analysis", action=argparse.BooleanOptionalAction,
                   default=True, help="Run notebook-style calibration, incidence, "
                   "attention, and embedding analyses (default: enabled)")
    p.add_argument("--analysis_diseases", type=int, default=10,
                   help="Number of frequent diseases in detailed plots")
    p.add_argument("--analysis_patients", type=int, default=256,
                   help="Maximum patients used for expensive timing/attention analyses")
    p.add_argument("--disease_chunk_size", type=int, default=64,
                   help="Diseases retained per inference pass; reduce if memory is limited")
    return p.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Delphi(DelphiConfig(**checkpoint["model_args"]))
    state = checkpoint["model"]
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[10:]] = state.pop(key)
    model.load_state_dict(state)
    return model.eval().to(device)


def load_split(path, split, block_size, no_event_rate, subset_size, sex_tokens):
    bin_path = Path(path) / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"{bin_path} not found")
    data = np.fromfile(bin_path, dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(data)
    n = len(p2i) if subset_size == -1 else min(subset_size, len(p2i))
    print(f"Loading {n:,} patients and {len(data):,} events from {bin_path}")
    sex_masks = raw_patient_sex_masks(data, p2i, n, sex_tokens)
    batch = get_batch(range(n), data, p2i, select="left", block_size=block_size,
                      device="cpu", padding="random",
                      no_event_token_rate=no_event_rate)
    return batch, sex_masks


def raw_patient_sex_masks(data, p2i, n_patients, sex_tokens):
    """Identify sex from complete raw trajectories, before context trimming.

    ``get_batch`` adds random no-event tokens and then trims sequences to the
    model context length. Demographic events occur at the start of a trajectory
    and can therefore be absent from every returned model input. The binary
    files store model-facing dictionary IDs minus one, so inspect the complete
    source trajectory for each selected patient instead.
    """
    stored_sex_tokens = {
        sex: int(token) - 1 for sex, token in sex_tokens.items()
    }
    masks = {sex: np.zeros(n_patients, dtype=bool) for sex in ("female", "male")}
    for patient_index, (start, length) in enumerate(p2i[:n_patients]):
        patient_tokens = data[int(start):int(start + length), 2]
        for sex, token in stored_sex_tokens.items():
            masks[sex][patient_index] = np.any(patient_tokens == token)
    overlap = masks["female"] & masks["male"]
    if overlap.any():
        raise RuntimeError(
            f"{int(overlap.sum()):,} patients contain both Female and Male tokens"
        )
    return masks


def load_token_dict(path, names_path=None):
    td = pd.read_csv(Path(path) / "token_dictionary.csv")
    td["icd_code"] = td.apply(
        lambda r: r["token_wording"].split("::")[-1]
        if r["coding"] in DISEASE_CODINGS
        and isinstance(r["token_wording"], str) else None,
        axis=1,
    )
    td["chapter"] = td.apply(
        lambda r: (
            ICD10_CHAPTERS.get(r["icd_code"][0], "Unknown")
            if r["coding"] == "ICD10" and isinstance(r["icd_code"], str)
            else "ICD-9"
            if r["coding"] == "ICD9"
            else "Death"
            if r["coding"] == "DEATH"
            else None
        ),
        axis=1,
    )
    td["readable_name"] = td["token_wording"]
    if names_path:
        names = pd.read_csv(names_path)
        names.columns = names.columns.str.lower().str.strip()
        if not {"code", "description"}.issubset(names.columns):
            raise ValueError("--icd10_names needs columns: code, description")
        td = td.merge(names[["code", "description"]].drop_duplicates("code"),
                      left_on="icd_code", right_on="code", how="left")
        td["readable_name"] = td["description"].fillna(td["token_wording"])
    return td


def resolve_sex_tokens(token_dictionary):
    """Resolve Female/Male token IDs from the dataset dictionary.

    Token IDs are dataset-specific; silently assuming IDs 2 and 3 can produce
    empty sex strata after vocabulary transformations.
    """
    id_candidates = ["token_id", "token", "index", "id"]
    id_column = next((column for column in id_candidates
                      if column in token_dictionary.columns), None)
    if id_column is None:
        # A dictionary written in vocabulary order may use its row index.
        ids = pd.Series(token_dictionary.index, index=token_dictionary.index)
    else:
        ids = pd.to_numeric(token_dictionary[id_column], errors="coerce")
    text_columns = [column for column in
                    ("token_wording", "readable_name", "icd_code", "coding")
                    if column in token_dictionary.columns]
    text = token_dictionary[text_columns].fillna("").astype(str).agg(" ".join, axis=1)

    resolved = {}
    for sex in ("female", "male"):
        # Whole-word matching prevents "male" from also matching "female".
        matched = text.str.contains(rf"(?i)(?:^|[^a-z]){sex}(?:$|[^a-z])", regex=True)
        # Rich UKB dictionaries can contain unrelated field descriptions such
        # as female-specific measurements.  Prefer the canonical demographic
        # row written by build_delphi_binary_datasets.R.
        wording = token_dictionary.get(
            "token_wording", pd.Series("", index=token_dictionary.index)
        ).fillna("").astype(str).str.strip().str.lower()
        canonical = wording.eq(f"demographics::{sex}")
        if "source_type" in token_dictionary.columns:
            source = token_dictionary["source_type"].fillna("").astype(str).str.lower()
            demographic = matched & source.eq("demographics")
        else:
            demographic = pd.Series(False, index=token_dictionary.index)
        preferred = canonical if canonical.sum() == 1 else demographic
        selected = preferred if preferred.sum() == 1 else matched
        candidates = pd.to_numeric(
            ids[selected], errors="coerce"
        ).dropna().astype(int).unique()
        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not uniquely resolve the {sex} token from token_dictionary.csv; "
                f"candidate IDs: {candidates.tolist()}"
            )
        resolved[sex] = int(candidates[0])
    if resolved["female"] == resolved["male"]:
        raise RuntimeError("Female and Male resolved to the same token ID")
    return resolved


def disease_tokens(td):
    mask = (td["coding"].isin(DISEASE_CODINGS)
            & ~td["token_wording"].str.lower().str.contains("other", na=False))
    # New preprocessing marks output classes explicitly. This is essential
    # because self-reported context rows also use coding="ICD10".
    if "is_output_class" in td:
        output_flag = td["is_output_class"]
        if output_flag.dtype != bool:
            output_flag = output_flag.astype(str).str.lower().isin(
                {"true", "t", "1", "yes"}
            )
        mask &= output_flag
    elif "source_type" in td:
        mask &= td["source_type"].eq("clinical")
    else:
        raise ValueError(
            "token_dictionary.csv must contain is_output_class or source_type "
            "to distinguish clinical from self-reported ICD10 tokens"
        )
    columns = ["token_id", "token_wording", "icd_code", "chapter",
               "readable_name"]
    for optional in ("coding", "source_type", "token_type",
                     "is_ce_target", "is_output_class"):
        if optional in td and optional not in columns:
            columns.append(optional)
    return td.loc[mask, columns].reset_index(drop=True)


def run_inference(model, batch, disease_ids, batch_size, device):
    x, a, y, b = batch
    logits, probs = [], []
    with torch.inference_mode():
        for start in tqdm(range(0, len(x), batch_size), desc="Inference"):
            end = min(start + batch_size, len(x))
            raw = model(x[start:end].to(device), a[start:end].to(device),
                        y[start:end].to(device), b[start:end].to(device))[0].float()
            # Metrics are sensitive to score ordering.  Keep float32 here so
            # weak/rare-outcome scores are not collapsed into float16 ties.
            logits.append(raw[:, :, disease_ids].cpu().numpy().astype(np.float32))
            probs.append(F.softmax(raw, dim=-1)[:, :, disease_ids]
                         .cpu().numpy().astype(np.float32))
    return np.concatenate(logits), np.concatenate(probs)


def delong_auc(case, control):
    """Use evaluate_auc.py for the ROC AUC estimate and DeLong variance."""
    if len(case) < 2 or len(control) < 2:
        return None
    try:
        auc, covariance = get_auc_delong_var(
            np.asarray(control, dtype=np.float32),
            np.asarray(case, dtype=np.float32),
        )
        variance = float(np.asarray(covariance).squeeze())
    except Exception:
        return None
    auc = float(auc)
    # The independent rank implementation is a useful orientation check when
    # scores are unique. Its argsort-based ranks do not give tied scores their
    # midrank, whereas the DeLong implementation above handles ties correctly.
    combined = np.r_[case, control]
    if len(np.unique(combined)) == len(combined):
        rank_auc = float(evaluate_auc_rank(
            np.asarray(control, dtype=np.float32),
            np.asarray(case, dtype=np.float32),
        ))
        if not np.isclose(auc, rank_auc, atol=1e-6):
            raise RuntimeError(
                f"AUC implementations disagree: DeLong={auc:.8f}, rank={rank_auc:.8f}"
            )
    margin = 1.96 * np.sqrt(max(variance, 0))
    return auc, max(0, auc - margin), min(1, auc + margin)


def binary_curves(labels, scores):
    """Return ROC and PR coordinates using every distinct score as a threshold."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct].astype(float)
    fp = (distinct + 1).astype(float) - tp
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    tpr = np.r_[0, tp / positives]
    fpr = np.r_[0, fp / negatives]
    precision = np.r_[1, tp / np.maximum(tp + fp, 1)]
    recall = np.r_[0, tp / positives]
    thresholds = s[distinct]
    return fpr, tpr, precision, recall, thresholds


def average_precision(precision, recall):
    return float(np.sum(np.diff(recall) * precision[1:]))


def optimal_metrics(labels, scores, fpr, tpr, thresholds):
    """Threshold-dependent metrics at the maximum Youden J operating point."""
    idx = int(np.argmax(tpr[1:] - fpr[1:]))
    threshold = float(thresholds[idx])
    pred = scores >= threshold
    y = labels.astype(bool)
    tp, tn = np.sum(pred & y), np.sum(~pred & ~y)
    fp, fn = np.sum(pred & ~y), np.sum(~pred & y)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    accuracy = (tp + tn) / len(y)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-15)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else np.nan
    return {
        "optimal_threshold": threshold, "sensitivity": sensitivity,
        "specificity": specificity, "precision": precision, "npv": npv,
        "accuracy": accuracy, "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": f1, "mcc": mcc, "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def _one_candidate_per_patient(patient_ids, eligible, rng):
    """Select one eligible prediction point per patient, as Delphi does."""
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return candidates
    shuffled = candidates[rng.permutation(len(candidates))]
    _, first = np.unique(patient_ids[shuffled], return_index=True)
    return shuffled[first]


def compute_metrics(j, token, logits, probs, batch_np, mask, age_groups, age_step,
                    offset, seed=1337, stratum="full"):
    # Select the one disease column before applying the population mask.
    # Masking the full N x T x disease-chunk arrays here created multi-GB
    # temporary copies once per disease and could exceed a 64-GB PBS limit.
    ages = batch_np[1][mask]
    targets = batch_np[2][mask]
    target_ages = batch_np[3][mask]
    lp = logits[:, :, j][mask]
    pp = probs[:, :, j][mask]
    # Delphi evaluates candidate prediction points, not cases just before their
    # diagnosis against controls at the end of follow-up. Cases are disease
    # targets; controls are all target positions from patients who never have
    # that disease in the evaluated trajectory window.
    case_pos = np.where(targets == token)
    if len(case_pos[0]) < 2:
        return None
    has_disease = (targets == token).any(axis=1)
    control_pos = np.where((targets != token) & (~has_disease[:, None]))
    patient_ids = np.r_[case_pos[0], control_pos[0]]
    target_pos = np.r_[case_pos[1], control_pos[1]]
    labels_all = np.r_[np.ones(len(case_pos[0]), dtype=np.int8),
                       np.zeros(len(control_pos[0]), dtype=np.int8)]
    target_time = target_ages[patient_ids, target_pos]
    pred_idx = (ages[patient_ids] < target_time[:, None] - offset).sum(1) - 1
    valid = pred_idx >= 0
    if valid.sum() < 4:
        return None
    patient_ids, pred_idx, labels_all = (x[valid] for x in
                                         (patient_ids, pred_idx, labels_all))
    scores_all = lp[patient_ids, pred_idx].astype(float)
    next_event_probs_all = pp[patient_ids, pred_idx].astype(float)
    prediction_ages = ages[patient_ids, pred_idx] / 365.25

    # Reproduce the reference sampling rule independently within each age band:
    # randomly retain one candidate observation per patient. A local, stable RNG
    # makes results invariant to disease chunking and evaluation order.
    stratum_code = sum((i + 1) * ord(c) for i, c in enumerate(stratum))
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(token), int(stratum_code)])
    )
    band_samples = []
    band_auc_results = []
    valid_band_counts = []
    band_roc_curves = []
    age_results = {}
    for age in age_groups:
        in_band = (prediction_ages >= age) & (prediction_ages < age + age_step)
        chosen = _one_candidate_per_patient(patient_ids, in_band, rng)
        band_samples.append(chosen)
        case_scores = scores_all[chosen][labels_all[chosen] == 1]
        control_scores = scores_all[chosen][labels_all[chosen] == 0]
        age_auc = delong_auc(case_scores, control_scores)
        if age_auc:
            # Recover the variance from the normal-approximation interval. It
            # is stored explicitly below for the Delphi-style overall CI.
            _, age_variance = get_auc_delong_var(
                np.asarray(control_scores, dtype=np.float32),
                np.asarray(case_scores, dtype=np.float32),
            )
            band_auc_results.append((float(age_auc[0]),
                                     float(np.asarray(age_variance).squeeze())))
            valid_band_counts.append((int(np.sum(labels_all[chosen] == 1)),
                                      int(np.sum(labels_all[chosen] == 0))))
            band_labels = labels_all[chosen]
            band_scores = scores_all[chosen]
            band_fpr, band_tpr, _, _, _ = binary_curves(
                band_labels, band_scores)
            band_roc_curves.append((band_fpr, band_tpr))
        age_results[f"auc_{age}_{age + age_step}"] = (
            age_auc[0] if age_auc else np.nan)
        age_results[f"count_{age}_{age + age_step}"] = int(
            np.sum(labels_all[chosen] == 1))
        age_results[f"n_controls_{age}_{age + age_step}"] = int(
            np.sum(labels_all[chosen] == 0))

    # A pooled curve must not contain the same patient once for every age band.
    # Select one point per patient over the configured age range for Full ROC/PR;
    # the per-band AUCs above retain Delphi's original band-specific sampling.
    in_evaluation_range = ((prediction_ages >= min(age_groups))
                           & (prediction_ages < max(age_groups) + age_step))
    chosen = _one_candidate_per_patient(patient_ids, in_evaluation_range, rng)
    labels = labels_all[chosen]
    scores = scores_all[chosen]
    probabilities = next_event_probs_all[chosen]
    selected_ages = prediction_ages[chosen]
    case_logits = scores[labels == 1]
    ctrl_logits = scores[labels == 0]
    if len(case_logits) < 2 or len(ctrl_logits) < 2:
        return None
    pooled_auc_result = delong_auc(case_logits, ctrl_logits)
    if pooled_auc_result is None or not band_auc_results:
        return None
    fpr, tpr, precision, recall, thresholds = binary_curves(labels, scores)

    # Match evaluate_auc.py: the reported overall AUC is the unweighted mean
    # of the available age-band AUCs. For independent normal estimates, the
    # variance of their mean is sum(variance) / n_bands**2.
    band_aucs = np.asarray([x[0] for x in band_auc_results], dtype=float)
    band_variances = np.asarray([x[1] for x in band_auc_results], dtype=float)
    auc = float(band_aucs.mean())
    auc_variance = float(band_variances.sum() / len(band_variances) ** 2)
    margin = 1.96 * np.sqrt(max(auc_variance, 0.0))

    # A mean of band-level AUCs has no single empirical ROC curve. Export a
    # macro-average curve by interpolating each band equally on a common FPR
    # grid; pooled PR remains participant-level because Delphi has no PR logic.
    macro_fpr = np.unique(np.concatenate(
        [np.asarray(x[0], dtype=float) for x in band_roc_curves]))
    macro_tpr = np.mean([
        np.interp(macro_fpr, band_fpr, band_tpr)
        for band_fpr, band_tpr in band_roc_curves
    ], axis=0)
    result = {
        "count": int(sum(x[0] for x in valid_band_counts)),
        "n_controls": int(sum(x[1] for x in valid_band_counts)),
        "pooled_count": int(labels.sum()),
        "pooled_n_controls": int((labels == 0).sum()),
        "prevalence": float(labels.mean()), "auc": auc,
        "auc_variance_delong": auc_variance,
        "auc_ci_lower": max(0.0, auc - margin),
        "auc_ci_upper": min(1.0, auc + margin),
        "pooled_auc": float(pooled_auc_result[0]),
        "n_age_bands": int(len(band_auc_results)),
        "pr_auc": average_precision(precision, recall),
        # This softmax value is P(disease is the next event), not fixed-horizon
        # absolute risk, so a binary-outcome Brier score would be misleading.
        "brier_score": np.nan,
        "mean_prob_case": float(probabilities[labels == 1].mean()),
        "median_prob_case": float(np.median(probabilities[labels == 1])),
        "mean_prob_control": float(probabilities[labels == 0].mean()),
        "median_prob_control": float(np.median(probabilities[labels == 0])),
        "probability_ratio": float(
            probabilities[labels == 1].mean()
            / max(probabilities[labels == 0].mean(), 1e-15)),
    }
    result.update(optimal_metrics(labels, scores, fpr, tpr, thresholds))
    result.update(age_results)
    curves = {
        "fpr": macro_fpr, "tpr": macro_tpr,
        "pooled_fpr": fpr, "pooled_tpr": tpr,
        "precision": precision, "recall": recall,
        "labels": labels, "logits": scores, "probabilities": probabilities,
        "ages": selected_ages,
    }
    return result, curves


def style_axis(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, color="#d9d9d9", linewidth=.7, alpha=.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_curves(curves, table, threshold, max_plots, out):
    has_curve = table.apply(
        lambda row: (int(row.token_id), row.sex) in curves, axis=1)
    eligible = table[(table["count"] >= threshold) & has_curve].copy()
    if eligible.empty:
        eligible = table[has_curve].nlargest(max_plots, "count")
        print("No diseases met count threshold; curve plots use most frequent diseases.")
    for sex, color in [("female", "#c44e52"), ("male", "#4c72b0")]:
        sub = eligible[eligible.sex == sex].nlargest(max_plots, "count")
        if sub.empty:
            continue
        for kind in ("roc", "pr"):
            fig, ax = plt.subplots(figsize=(9, 7))
            for _, row in sub.iterrows():
                curve = curves[(int(row.token_id), sex)]
                name = str(row.readable_name)
                name = name if len(name) <= 38 else name[:35] + "..."
                if kind == "roc":
                    ax.plot(curve["fpr"], curve["tpr"], lw=1.7,
                            label=f"{name} (AUC {row.auc:.3f})")
                else:
                    ax.plot(curve["recall"], curve["precision"], lw=1.7,
                            label=f"{name} (AP {row.pr_auc:.3f})")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], "--", color="#777", lw=1, label="Chance")
                style_axis(ax, "False positive rate", "True positive rate",
                           f"ROC curves — {sex}")
            else:
                style_axis(ax, "Recall (sensitivity)", "Precision",
                           f"Precision–recall curves — {sex}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8,
                      frameon=False)
            fig.tight_layout()
            fig.savefig(out / f"{kind}_curves_{sex}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


def save_pr_curves(curves, table, model_name, prediction_gap_months, out):
    """Write tidy precision–recall coordinates already computed in memory."""
    metadata = table.drop_duplicates(["token_id", "sex"]).set_index(
        ["token_id", "sex"])
    frames = []
    for (token, sex), curve in curves.items():
        if (token, sex) not in metadata.index:
            continue
        row = metadata.loc[(token, sex)]
        precision = np.asarray(curve["precision"], dtype=float)
        recall = np.asarray(curve["recall"], dtype=float)
        if len(precision) != len(recall):
            raise RuntimeError(f"PR coordinate length mismatch for token {token}, {sex}")
        frames.append(pd.DataFrame({
            "model": model_name,
            "token_id": int(token),
            "icd_code": row.icd_code,
            "disease": row.readable_name,
            "sex": str(sex).title(),
            "age_group": "Full",
            "horizon_months": np.nan,
            "prediction_gap_months": float(prediction_gap_months),
            "recall": recall,
            "precision": precision,
            "average_precision": float(row.pr_auc),
            "prevalence": float(row.prevalence),
            "n_positive": int(np.sum(curve["labels"] == 1)),
            "n_negative": int(np.sum(curve["labels"] == 0)),
        }))
    columns = ["model", "token_id", "icd_code", "disease", "sex",
               "age_group", "horizon_months", "prediction_gap_months",
               "recall", "precision",
               "average_precision", "prevalence", "n_positive", "n_negative"]
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    output.to_csv(out / "pr_curves.csv", index=False)
    print(f"PR coordinates: {len(output):,} rows written to {out / 'pr_curves.csv'}")


def save_roc_curves(curves, table, model_name, prediction_gap_months, out):
    """Write tidy ROC coordinates for the outcomes retained in memory."""
    metadata = table.drop_duplicates(["token_id", "sex"]).set_index(
        ["token_id", "sex"])
    frames = []
    for (token, sex), curve in curves.items():
        if (token, sex) not in metadata.index:
            continue
        row = metadata.loc[(token, sex)]
        fpr = np.asarray(curve["fpr"], dtype=float)
        tpr = np.asarray(curve["tpr"], dtype=float)
        if len(fpr) != len(tpr):
            raise RuntimeError(f"ROC coordinate length mismatch for {token}, {sex}")
        frames.append(pd.DataFrame({
            "model": model_name, "token_id": int(token),
            "icd_code": row.icd_code, "disease": row.readable_name,
            "sex": str(sex).title(), "age_group": "Full",
            "horizon_months": np.nan,
            "prediction_gap_months": float(prediction_gap_months),
            "fpr": fpr, "tpr": tpr, "auc": float(row.auc),
            "n_positive": int(row["count"]),
            "n_negative": int(row.n_controls),
        }))
    columns = ["model", "token_id", "icd_code", "disease", "sex",
               "age_group", "horizon_months", "prediction_gap_months",
               "fpr", "tpr", "auc",
               "n_positive", "n_negative"]
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=columns)
    output.to_csv(out / "roc_curves.csv", index=False)
    print(f"ROC coordinates: {len(output):,} rows written to {out / 'roc_curves.csv'}")


def save_plotting_summaries(table, tidy, threshold, age_groups, age_step, out):
    """Export reusable aggregated tables so plot scripts need no recomputation."""
    full = tidy[tidy["age_group"].eq("Full")].copy()
    full["eligible"] = full["n_positive"].ge(threshold)
    count_columns = [
        "model", "token_id", "icd_code", "disease", "chapter", "sex",
        "n_positive", "n_negative", "n_total", "prevalence", "eligible",
    ]
    full[count_columns].to_csv(out / "outcome_counts.csv", index=False)

    eligible = full[full["eligible"] & full["auc"].notna()].copy()
    chapter_summary = (eligible.groupby(["model", "sex", "chapter"],
                                        dropna=False)
                       .agg(n_outcomes=("token_id", "nunique"),
                            auc_mean=("auc", "mean"),
                            auc_median=("auc", "median"),
                            auc_q25=("auc", lambda x: x.quantile(.25)),
                            auc_q75=("auc", lambda x: x.quantile(.75)))
                       .reset_index())
    chapter_summary.to_csv(out / "auc_by_chapter_summary.csv", index=False)

    sex_summary = (eligible.groupby(["model", "sex"], dropna=False)
                   .agg(n_outcomes=("token_id", "nunique"),
                        auc_mean=("auc", "mean"),
                        auc_median=("auc", "median"),
                        auc_q25=("auc", lambda x: x.quantile(.25)),
                        auc_q75=("auc", lambda x: x.quantile(.75)))
                   .reset_index())
    sex_summary.to_csv(out / "auc_by_sex_summary.csv", index=False)

    age = tidy[~tidy["age_group"].eq("Full")].copy()
    age["eligible"] = age["n_positive"].ge(threshold)
    age_summary = (age[age["eligible"] & age["auc"].notna()]
                   .groupby(["model", "sex", "age_group"], dropna=False)
                   .agg(n_outcomes=("token_id", "nunique"),
                        auc_mean=("auc", "mean"),
                        auc_median=("auc", "median"),
                        auc_sem=("auc", "sem"))
                   .reset_index())
    ordered = [f"{a}-{a + age_step}" for a in age_groups]
    age_summary["age_group"] = pd.Categorical(
        age_summary["age_group"], categories=ordered, ordered=True)
    age_summary.sort_values(["sex", "age_group"]).to_csv(
        out / "auc_by_age_summary.csv", index=False)


def tidy_auc_results(table, model_name, model_directory, age_groups, age_step,
                     prediction_gap_months):
    """Return the canonical plotting schema used by the figure scripts."""
    base = pd.DataFrame({
        "model": model_name,
        "model_directory": model_directory,
        # plot_auc_training_comparison.py uses this legacy column name.
        "training_data": model_name,
        "token_id": table["token_id"].astype(int),
        "icd_code": table["icd_code"],
        "disease": table["readable_name"],
        "chapter": table["chapter"],
        "sex": table["sex"].astype(str).str.title(),
        "age_group": "Full",
        "horizon_months": np.nan,
        "prediction_gap_months": float(prediction_gap_months),
        "horizon_definition": "minimum lead time before observed event",
        "n_positive": table["count"].astype(int),
        "n_negative": table["n_controls"].astype(int),
        "n_total": (table["count"] + table["n_controls"]).astype(int),
        "auc": table["auc"],
        "auc_lower": table["auc_ci_lower"],
        "auc_upper": table["auc_ci_upper"],
        "pr_auc": table["pr_auc"],
        "prevalence": table["prevalence"],
    })
    parts = [base]
    for age in age_groups:
        end = age + age_step
        auc_column = f"auc_{age}_{end}"
        count_column = f"count_{age}_{end}"
        control_column = f"n_controls_{age}_{end}"
        if auc_column not in table:
            continue
        age_rows = base.copy()
        age_rows["age_group"] = f"{age}-{end}"
        age_rows["auc"] = pd.to_numeric(table[auc_column], errors="coerce")
        age_rows["n_positive"] = pd.to_numeric(
            table.get(count_column), errors="coerce"
        )
        age_rows["n_negative"] = pd.to_numeric(
            table.get(control_column), errors="coerce"
        )
        age_rows["n_total"] = age_rows["n_positive"] + age_rows["n_negative"]
        age_rows[["auc_lower", "auc_upper", "pr_auc", "prevalence"]] = np.nan
        parts.append(age_rows)
    return pd.concat(parts, ignore_index=True, sort=False)


def save_prediction_rows(curves, table, model_name, prediction_gap_months, out):
    """Write row-level scores for plotting scripts that accept predictions.

    Only diseases retained for curve export are included. ``y_score`` is the
    disease logit used for ROC/PR ranking; ``next_event_probability`` is kept
    separate because it is not a fixed-horizon absolute risk.
    """
    metadata = table.drop_duplicates(["token_id", "sex"]).set_index(
        ["token_id", "sex"]
    )
    frames = []
    for (token, sex), curve in curves.items():
        if (token, sex) not in metadata.index:
            continue
        row = metadata.loc[(token, sex)]
        labels = np.asarray(curve["labels"], dtype=np.int8)
        scores = np.asarray(curve["logits"], dtype=np.float32)
        probabilities = np.asarray(curve["probabilities"], dtype=np.float32)
        ages = np.asarray(curve["ages"], dtype=np.float32)
        if not (len(labels) == len(scores) == len(probabilities) == len(ages)):
            raise RuntimeError(
                f"Prediction-array length mismatch for token {token}, {sex}"
            )
        frames.append(pd.DataFrame({
            "model": model_name,
            "training_data": model_name,
            "observation_id": np.arange(len(labels), dtype=np.int64),
            "token_id": int(token),
            "icd_code": row.icd_code,
            "disease": row.readable_name,
            "sex": str(sex).title(),
            "age_group": "Full",
            "horizon_months": np.nan,
            "prediction_gap_months": float(prediction_gap_months),
            "horizon_definition": "minimum lead time before observed event",
            "prediction_age": ages,
            "y_true": labels,
            "y_score": scores,
            "next_event_probability": probabilities,
        }))
    columns = [
        "model", "training_data", "observation_id", "token_id", "icd_code",
        "disease", "sex", "age_group", "horizon_months",
        "prediction_gap_months", "horizon_definition", "prediction_age",
        "y_true", "y_score", "next_event_probability",
    ]
    predictions = (pd.concat(frames, ignore_index=True)
                   if frames else pd.DataFrame(columns=columns))
    predictions.to_csv(out / "predictions.csv", index=False)
    print(f"Prediction rows: {len(predictions):,} written to {out / 'predictions.csv'}")


def save_csv_manifest(out):
    """Describe every CSV emitted by this evaluation run."""
    descriptions = {
        "age_incidence_table.csv": "Age-specific predicted and observed incidence",
        "attention_decay.csv": "Mean attention by token lag",
        "auc_results.csv": "Canonical tidy AUC table for plotting",
        "auc_results_filtered.csv": "Tidy AUC rows meeting the event threshold",
        "auc_table.csv": "Native wide disease/sex evaluation metrics",
        "calibration_table.csv": "Binned predicted and observed risks",
        "embedding_umap.csv": "Disease-token embedding coordinates",
        "evaluation_summary.csv": "Across-disease metric summary",
        "model_report.csv": "Eligible disease-level model report",
        "pr_curves.csv": "Tidy precision-recall curve coordinates",
        "roc_curves.csv": "Tidy receiver-operating-characteristic coordinates",
        "predictions.csv": "Row-level labels and ranking scores for retained diseases",
        "outcome_counts.csv": "Outcome case/control counts and threshold eligibility",
        "auc_by_age_summary.csv": "Plot-ready age-stratified AUC summary",
        "auc_by_chapter_summary.csv": "Plot-ready chapter-stratified AUC summary",
        "auc_by_sex_summary.csv": "Plot-ready sex-stratified AUC summary",
        "evaluation_run_config.csv": "Arguments and resolved checkpoint settings",
        "token_evaluation_registry.csv": "Clinical target tokens evaluated in this run",
    }
    rows = []
    for path in sorted(out.glob("*.csv")):
        try:
            row_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
        except UnicodeDecodeError:
            row_count = np.nan
        rows.append({
            "file": path.name,
            "rows": max(row_count, 0) if np.isfinite(row_count) else row_count,
            "description": descriptions.get(path.name, "Evaluation output table"),
        })
    pd.DataFrame(rows, columns=["file", "rows", "description"]).to_csv(
        out / "csv_manifest.csv", index=False
    )


def plot_summaries(df, threshold, age_groups, age_step, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, sex in zip(axes, ("female", "male")):
        sub = df[df.sex == sex]
        colors = sub.chapter.map(lambda x: CHAPTER_COLORS.get(x, "#999999"))
        ax.scatter(sub["count"], sub.auc, c=list(colors), alpha=.65, s=24)
        ax.axvline(threshold, ls="--", color="#c44e52")
        ax.axhline(.5, ls=":", color="#666")
        ax.set_xscale("log")
        ax.set_ylim(.3, 1.01)
        style_axis(ax, "Number of cases (log scale)", "ROC AUC", sex.capitalize())
    fig.suptitle("Disease-level discrimination and sample size")
    fig.tight_layout()
    fig.savefig(out / "auc_vs_count.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    above = df[df["count"] >= threshold]
    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, color in (("female", "#c44e52"), ("male", "#4c72b0")):
        vals = above.loc[above.sex == sex, "auc"].dropna()
        if len(vals):
            ax.hist(vals, bins=np.linspace(.4, 1, 25), alpha=.55, color=color,
                    edgecolor="white", label=f"{sex.capitalize()} (n={len(vals)})")
            ax.axvline(vals.median(), color=color, ls="--")
    ax.axvline(.5, color="#666", ls=":")
    style_axis(ax, "ROC AUC", "Number of diseases",
               f"AUC distribution (cases ≥ {threshold:,})")
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "auc_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{a}–{a + age_step}" for a in age_groups]
    for sex, color in (("female", "#c44e52"), ("male", "#4c72b0")):
        sub = above[above.sex == sex]
        values = [sub[f"auc_{a}_{a + age_step}"].dropna() for a in age_groups]
        means = np.array([v.mean() if len(v) else np.nan for v in values])
        sems = np.array([v.sem() if len(v) > 1 else np.nan for v in values])
        ax.plot(labels, means, "o-", color=color, label=sex.capitalize())
        ax.fill_between(labels, means - sems, means + sems, color=color, alpha=.15)
    ax.axhline(.5, color="#666", ls=":")
    style_axis(ax, "Age at prediction (years)", "Mean ROC AUC ± SEM", "AUC by age")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "auc_by_age.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summaries(df, threshold, out):
    selected = df[df["count"] >= threshold]
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity",
               "precision", "f1", "mcc", "brier_score"]
    rows = []
    for sex in ("full", "female", "male"):
        sub = selected[selected.sex == sex]
        row = {"sex": sex, "n_diseases": len(sub), "count_threshold": threshold}
        for metric in metrics:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_median"] = sub[metric].median()
            row[f"{metric}_q25"] = sub[metric].quantile(.25)
            row[f"{metric}_q75"] = sub[metric].quantile(.75)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "evaluation_summary.csv", index=False)
    columns = ["token_id", "readable_name", "sex", "count", "n_controls",
               "auc", "auc_ci_lower", "auc_ci_upper", "pr_auc",
               "balanced_accuracy", "sensitivity", "specificity", "precision",
               "f1", "mcc", "brier_score", "optimal_threshold"]
    selected.sort_values(["sex", "auc"], ascending=[True, False])[columns].to_csv(
        out / "model_report.csv", index=False)


def plot_selected_auc_ci(df, n_diseases, out):
    """Notebook-style bar chart of frequent diseases with DeLong intervals."""
    pooled = (df.groupby(["token_id", "readable_name"], as_index=False)
              .agg(count=("count", "sum"), auc=("auc", "mean"),
                   ci_lower=("auc_ci_lower", "mean"),
                   ci_upper=("auc_ci_upper", "mean"))
              .nlargest(n_diseases, "count").sort_values("auc"))
    if pooled.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, .42 * len(pooled))))
    y = np.arange(len(pooled))
    ax.barh(y, pooled.auc, color="#7eb6d8")
    ax.errorbar(pooled.auc, y,
                xerr=np.vstack([pooled.auc - pooled.ci_lower,
                                pooled.ci_upper - pooled.auc]),
                fmt="none", color="#222", lw=1)
    ax.set_yticks(y, [str(x)[:55] for x in pooled.readable_name])
    ax.axvline(.5, color="#666", ls="--")
    ax.set_xlim(0, 1.02)
    style_axis(ax, "Mean ROC AUC", "", "Frequent diseases: AUC with 95% CI")
    fig.tight_layout()
    fig.savefig(out / "auc_selected_diseases_ci.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_auc_group_comparisons(df, threshold, out):
    """Notebook-style AUC distributions grouped by ICD chapter and sex."""
    selected = df[df["count"] >= threshold].dropna(subset=["auc"])
    if selected.empty:
        selected = df.dropna(subset=["auc"])
    chapters = [c for c in selected.chapter.dropna().unique()
                if c not in {"Technical", "Sex"}]
    if chapters:
        data = [selected.loc[selected.chapter == c, "auc"].values for c in chapters]
        fig, ax = plt.subplots(figsize=(max(9, .65 * len(chapters)), 5))
        boxes = ax.boxplot(data, patch_artist=True,
                           whis=(2.5, 97.5), showfliers=True)
        ax.set_xticks(np.arange(1, len(chapters) + 1), chapters)
        for box, chapter in zip(boxes["boxes"], chapters):
            box.set_facecolor(CHAPTER_COLORS.get(chapter, "#aaaaaa"))
            box.set_alpha(.8)
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(.5, color="#666", ls="--")
        ax.set_ylim(0, 1.02)
        style_axis(ax, "ICD-10 chapter", "ROC AUC", "AUC grouped by ICD-10 chapter")
        fig.tight_layout()
        fig.savefig(out / "auc_by_chapter.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 5))
    sex_data = [selected.loc[selected.sex == s, "auc"].values
                for s in ("female", "male")]
    boxes = ax.boxplot(sex_data, patch_artist=True,
                       whis=(2.5, 97.5), showfliers=True)
    ax.set_xticks([1, 2], ["Female", "Male"])
    for box, color in zip(boxes["boxes"], ("#c44e52", "#4c72b0")):
        box.set_facecolor(color)
    ax.axhline(.5, color="#666", ls="--")
    ax.set_ylim(0, 1.02)
    style_axis(ax, "Sex", "ROC AUC", "AUC grouped by sex")
    fig.tight_layout()
    fig.savefig(out / "auc_by_sex_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_calibration(logits, batch_np, diseases, df, age_groups,
                              age_step, offset, n_diseases, sex_masks, out):
    """Observed versus modelled disease risk, stratified by age and sex."""
    available = set(diseases.token_id.astype(int))
    selected = (df[df.token_id.astype(int).isin(available)]
                .groupby("token_id", as_index=False)["count"].sum()
                .nlargest(n_diseases, "count").token_id.astype(int))
    cmap = plt.get_cmap("viridis")
    records = []
    for token in selected:
        name_rows = df[df.token_id == token]
        if name_rows.empty:
            continue
        name = str(name_rows.iloc[0].readable_name)
        disease_row = diseases.index[diseases.token_id.astype(int) == token]
        if len(disease_row) == 0:
            continue
        j = int(disease_row[0])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for ax, sex in zip(axes, ("female", "male")):
            sex_mask = sex_masks[sex]
            ages, targets, target_ages = (batch_np[1][sex_mask],
                                          batch_np[2][sex_mask],
                                          batch_np[3][sex_mask])
            disease_logits = logits[sex_mask, :, j].astype(float)
            patient_idx, target_idx = np.where(targets >= 0)
            if len(patient_idx) == 0:
                continue
            prediction_idx = (
                ages[patient_idx] < target_ages[patient_idx, target_idx, None] - offset
            ).sum(axis=1) - 1
            valid = prediction_idx >= 0
            patient_idx, target_idx = patient_idx[valid], target_idx[valid]
            prediction_idx = prediction_idx[valid]
            prediction_age = ages[patient_idx, prediction_idx] / 365.25
            horizon = target_ages[patient_idx, target_idx] - ages[patient_idx, prediction_idx]
            within_horizon = (horizon > 0) & (horizon <= 365.25 * age_step)
            labels = ((targets[patient_idx, target_idx] == token) & within_horizon).astype(int)
            selected_logits = disease_logits[patient_idx, prediction_idx]
            risks = -np.expm1(
                -np.exp(np.clip(selected_logits, -30, 10)) * 365.25 * age_step
            )
            for ai, age in enumerate(age_groups):
                keep = ((prediction_age >= age)
                        & (prediction_age < age + age_step))
                if keep.sum() < 20 or labels[keep].sum() < 2:
                    continue
                x, y = risks[keep], labels[keep]
                edges = np.unique(np.quantile(x, np.linspace(0, 1, 6)))
                if len(edges) < 3:
                    continue
                bins = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
                for b in np.unique(bins):
                    take = bins == b
                    if take.sum() < 5:
                        continue
                    pred, obs = float(x[take].mean()), float(y[take].mean())
                    records.append({"token_id": token, "sex": sex, "age_start": age,
                                    "predicted_risk": pred, "observed_risk": obs,
                                    "n": int(take.sum())})
                    ax.scatter(pred, obs, s=22, color=cmap(ai / max(len(age_groups)-1, 1)),
                               alpha=.85, label=f"{age}–{age+age_step}" if b == 0 else None)
            ax.plot([1e-6, 1], [1e-6, 1], "--", color="#666", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-5, 1)
            ax.set_ylim(1e-5, 1)
            style_axis(ax, f"Predicted {age_step}-year risk", "Observed case fraction",
                       sex.capitalize())
        handles, labels_legend = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels_legend, fontsize=7, frameon=False)
        fig.suptitle(f"Age- and sex-stratified calibration: {name}")
        fig.tight_layout()
        fig.savefig(out / f"calibration_token_{token}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
    calibration_df = pd.DataFrame(records, columns=[
        "token_id", "sex", "age_start", "predicted_risk",
        "observed_risk", "n",
    ])
    calibration_df.to_csv(out / "calibration_table.csv", index=False)
    return calibration_df


def plot_age_incidence(calibration_df, df, age_step, n_diseases, out):
    """Notebook-style modelled versus observed age-incidence summary."""
    selected = (df.groupby("token_id", as_index=False)["count"].sum()
                .nlargest(n_diseases, "count").token_id.astype(int).tolist())
    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    if not selected:
        return
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows),
                             sharex=True, squeeze=False)
    rows = []
    for ax, token in zip(axes.ravel(), selected):
        name = str(df[df.token_id == token].iloc[0].readable_name)
        for sex, color in (("female", "#c44e52"), ("male", "#4c72b0")):
            sub = calibration_df[
                (calibration_df.token_id == token) & (calibration_df.sex == sex)
            ]
            for age, group in sub.groupby("age_start"):
                weights = group["n"].to_numpy()
                predicted = float(np.average(group.predicted_risk, weights=weights))
                observed = float(np.average(group.observed_risk, weights=weights))
                midpoint = age + age_step / 2
                rows.append({"token_id": token, "sex": sex, "age_start": age,
                             "predicted_risk": predicted,
                             "observed_risk": observed, "n": int(weights.sum())})
                ax.plot(midpoint, predicted, "o", color=color, ms=4)
                ax.plot(midpoint, observed, "x", color=color, ms=5)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        style_axis(ax, "Age (years)", "Risk / case fraction", name[:60])
    for ax in axes.ravel()[len(selected):]:
        ax.set_visible(False)
    fig.suptitle("Age-specific modelled risk (circles) and observed cases (crosses)")
    fig.tight_layout()
    fig.savefig(out / "age_incidence_selected_diseases.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "age_incidence_table.csv", index=False)


def plot_next_event_timing(model, batch, n_patients, batch_size, device, out):
    """Compare expected and observed time to the next token."""
    x, a, y, b = [v[:n_patients] for v in batch]
    expected, observed = [], []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            logits = model(x[start:end].to(device), a[start:end].to(device),
                           y[start:end].to(device), b[start:end].to(device))[0]
            expected.append(torch.exp(-torch.logsumexp(logits.float(), dim=-1)).cpu().numpy())
            observed.append((b[start:end] - a[start:end]).numpy())
    expected, observed = np.concatenate(expected).ravel(), np.concatenate(observed).ravel()
    keep = np.isfinite(expected) & (expected > 0) & (observed > 0)
    expected, observed = expected[keep], observed[keep]
    if not len(expected):
        return
    edges = np.logspace(np.log10(max(expected.min(), 1)), np.log10(max(expected.max(), 2)), 20)
    bins = np.digitize(expected, edges)
    means_x, means_y = [], []
    for i in range(1, len(edges)):
        take = bins == i
        if take.sum() >= 10:
            means_x.append(np.exp(np.mean(np.log(expected[take]))))
            means_y.append(np.mean(observed[take]))
    sample = np.random.default_rng(1337).choice(
        len(expected), min(20000, len(expected)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected[sample], observed[sample], s=4, alpha=.12, color="#777")
    ax.plot(means_x, means_y, "o-", color="#c44e52", label="Binned mean")
    lo = max(1, min(expected.min(), observed.min()))
    hi = max(expected.max(), observed.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#222", label="Perfect calibration")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axis(ax, "Expected days to next token", "Observed days to next token",
               "Next-event timing calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "next_event_time_calibration.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def plot_attention_decay(model, batch, n_patients, device, out):
    """Aggregate attention by token lag; avoids exposing participant trajectories."""
    n = min(n_patients, len(batch[0]), 32)
    if n == 0:
        return
    d = [v[:n].to(device) for v in batch]
    with torch.inference_mode():
        attention = model(*d)[2]
    if attention is None:
        print("Attention output unavailable; skipping attention_decay.png")
        return
    att = attention.detach().float().cpu().numpy()
    # Common layouts: layers×batch×heads×query×key or batch×layers×heads×query×key.
    if att.ndim == 4:
        att = att[:, None, ...]
    if att.ndim != 5:
        print(f"Unexpected attention shape {att.shape}; skipping attention analysis")
        return
    if att.shape[1] != n and att.shape[0] == n:
        att = np.swapaxes(att, 0, 1)
    mean_att = att.mean(axis=(0, 1, 2))
    lags, values = [], []
    for lag in range(mean_att.shape[-1]):
        diagonal = np.diagonal(mean_att, offset=-lag)
        if diagonal.size:
            lags.append(lag)
            values.append(float(diagonal.mean()))
    pd.DataFrame({"token_lag": lags, "mean_attention": values}).to_csv(
        out / "attention_decay.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, values, "o-", color="#4c72b0")
    style_axis(ax, "Token lag", "Mean attention weight",
               "Aggregate attention decay across layers, heads, and patients")
    fig.tight_layout()
    fig.savefig(out / "attention_decay.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_umap(model, diseases, out, seed):
    """Notebook-style UMAP of learned token embeddings."""
    try:
        import umap
    except ImportError:
        print("umap-learn is unavailable; skipping embedding_umap.png")
        return
    weights = model.transformer.wte.weight.detach().float().cpu().numpy()
    coords = umap.UMAP(random_state=seed, n_neighbors=30, min_dist=.05,
                       metric="cosine").fit_transform(weights)
    points = diseases.copy()
    token_ids = points.token_id.astype(int).to_numpy()
    valid = (token_ids >= 0) & (token_ids < len(coords))
    points = points.loc[valid].copy()
    token_ids = token_ids[valid]
    points["UMAP1"], points["UMAP2"] = coords[token_ids, 0], coords[token_ids, 1]
    points.to_csv(out / "embedding_umap.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    for chapter, group in points.groupby("chapter", dropna=False):
        ax.scatter(group.UMAP1, group.UMAP2, s=18, alpha=.75,
                   color=CHAPTER_COLORS.get(chapter, "#999999"),
                   label=chapter if pd.notna(chapter) else "Other")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("UMAP of learned disease embeddings")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=7,
              frameon=False)
    fig.tight_layout()
    fig.savefig(out / "embedding_umap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_extended_analysis(model, batch, batch_np, logits, curves, results,
                          diseases, sex_masks, args, out):
    print("Running notebook-style extended evaluation ...")
    age_groups = list(range(args.age_min, args.age_max, args.age_step))
    plot_next_event_timing(model, batch, min(args.analysis_patients, len(batch[0])),
                           args.batch_size, args.device, out)
    calibration_df = plot_absolute_calibration(
        logits, batch_np, diseases, results, age_groups, args.age_step,
        args.offset, args.analysis_diseases, sex_masks, out)
    plot_age_incidence(calibration_df, results, args.age_step,
                       args.analysis_diseases, out)
    plot_selected_auc_ci(results, args.analysis_diseases, out)
    plot_auc_group_comparisons(results, args.count_threshold, out)
    plot_attention_decay(model, batch, args.analysis_patients, args.device, out)
    plot_embedding_umap(model, diseases, out, args.seed)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    age_groups = list(range(args.age_min, args.age_max, args.age_step))

    model = load_model(args.model_ckpt_path, args.device)
    checkpoint_block_size = int(model.config.block_size)
    evaluation_block_size = (
        checkpoint_block_size if args.block_size is None else int(args.block_size)
    )
    if evaluation_block_size <= 0:
        raise ValueError("--block_size must be positive")
    if evaluation_block_size != checkpoint_block_size:
        print(
            f"WARNING: evaluating with block_size={evaluation_block_size}, but "
            f"the checkpoint was trained with block_size={checkpoint_block_size}.",
            file=sys.stderr,
        )
    print(f"Evaluation block_size: {evaluation_block_size}")
    token_dictionary = load_token_dict(args.input_path, args.icd10_names)
    sex_tokens = resolve_sex_tokens(token_dictionary)
    print(f"Resolved sex tokens: Female={sex_tokens['female']}, "
          f"Male={sex_tokens['male']}")
    batch, raw_sex_masks = load_split(
        args.input_path, args.split, evaluation_block_size,
        args.no_event_token_rate, args.dataset_subset_size, sex_tokens,
    )
    batch_np = [tensor.numpy() for tensor in batch]
    diseases = disease_tokens(token_dictionary)
    selected = ({str(code).upper().strip() for code in args.selected_icd_codes}
                if args.selected_icd_codes else set())
    if args.select_from_auc_table:
        old = pd.read_csv(args.select_from_auc_table)
        if not {"icd_code", "count"}.issubset(old.columns):
            raise ValueError("--select_from_auc_table requires icd_code and count columns")
        if "sex" in old:
            old = old[old.sex.fillna("full").astype(str).str.lower().eq("full")]
        old["count"] = pd.to_numeric(old["count"], errors="coerce")
        chosen = (old.dropna(subset=["icd_code", "count"])
                  .sort_values("count", ascending=False)
                  .drop_duplicates("icd_code").head(args.selected_outcomes))
        selected.update(chosen.icd_code.astype(str).str.upper().str.strip())
        print("Selected from existing AUC table: " + ", ".join(chosen.icd_code.astype(str)))
    if selected:
        diseases = diseases[
            diseases.icd_code.fillna("").astype(str).str.upper().str.strip().isin(selected)
        ].reset_index(drop=True)
        found = set(diseases.icd_code.astype(str).str.upper())
        missing = sorted(selected - found)
        if missing:
            print(f"WARNING: requested ICD-10 codes not found: {', '.join(missing)}",
                  file=sys.stderr)
        print(f"Targeted evaluation: {len(diseases)} selected ICD-10 outcomes")
    ids = diseases.token_id.astype(int).tolist()
    if not ids:
        raise RuntimeError("No disease tokens found")
    print(f"Evaluating {len(ids)} disease tokens in chunks of "
          f"{args.disease_chunk_size}")
    sex_masks = {
        "full": np.ones(len(batch_np[0]), dtype=bool),
        **raw_sex_masks,
    }
    print("Sex-stratum patient counts: "
          + ", ".join(f"{sex}={int(mask.sum()):,}"
                      for sex, mask in sex_masks.items()))
    empty_sex_strata = [
        sex for sex in ("female", "male") if not sex_masks[sex].any()
    ]
    if empty_sex_strata:
        print(
            "WARNING: skipping empty sex strata: "
            + ", ".join(empty_sex_strata)
            + ". Full-population evaluation will still be produced.",
            file=sys.stderr,
        )
        sex_masks = {
            sex: mask for sex, mask in sex_masks.items() if mask.any()
        }

    # Only a small, frequent subset needs patient-level arrays for ROC/PR,
    # calibration, and incidence plots. All diseases still receive scalar
    # metrics in auc_table.csv.
    retain_n = max(args.max_curve_plots, args.analysis_diseases)
    retain_ids = set()
    for mask in [np.ones(len(batch_np[2]), dtype=bool), *sex_masks.values()]:
        target_values, target_counts = np.unique(batch_np[2][mask], return_counts=True)
        counts = dict(zip(target_values.astype(int), target_counts.astype(int)))
        retain_ids.update(sorted(ids, key=lambda token: counts.get(token, 0),
                                 reverse=True)[:retain_n])
    retained_diseases = diseases[diseases.token_id.astype(int).isin(retain_ids)]
    retained_diseases = retained_diseases.reset_index(drop=True)
    retained_column = {int(token): j for j, token in
                       enumerate(retained_diseases.token_id)}
    retained_logits = np.empty(
        (len(batch_np[0]), batch_np[0].shape[1], len(retained_diseases)),
        dtype=np.float32,
    )

    rows, curves = [], {}
    for start in tqdm(range(0, len(diseases), args.disease_chunk_size),
                      desc="Disease chunks"):
        chunk = diseases.iloc[start:start + args.disease_chunk_size]
        chunk_ids = chunk.token_id.astype(int).tolist()
        chunk_logits, chunk_probs = run_inference(
            model, batch, chunk_ids, args.batch_size, args.device)
        for j, (_, row) in enumerate(chunk.iterrows()):
            token = int(row.token_id)
            if token in retained_column:
                retained_logits[:, :, retained_column[token]] = chunk_logits[:, :, j]
            for sex, mask in sex_masks.items():
                computed = compute_metrics(
                    j, token, chunk_logits, chunk_probs, batch_np, mask,
                    age_groups, args.age_step, args.offset,
                    seed=args.seed, stratum=sex)
                if computed is None:
                    continue
                metrics, curve = computed
                rows.append({
                    "token_id": token, "token_wording": row.token_wording,
                    "icd_code": row.icd_code, "readable_name": row.readable_name,
                    "chapter": row.chapter, "sex": sex, **metrics,
                })
                if token in retain_ids:
                    curves[(token, sex)] = curve
        del chunk_logits, chunk_probs
        gc.collect()
    if not rows:
        print("No results computed; check disease events and offset.", file=sys.stderr)
        return 1
    results = pd.DataFrame(rows)
    results.to_csv(out / "auc_table.csv", index=False)
    model_name = args.model_name or Path(args.model_ckpt_path).resolve().parent.name
    model_directory = Path(args.model_ckpt_path).resolve().parent.name
    prediction_gap_months = args.offset / 365.25 * 12.0
    tidy_results = tidy_auc_results(
        results, model_name, model_directory, age_groups, args.age_step,
        prediction_gap_months,
    )
    tidy_results.to_csv(out / "auc_results.csv", index=False)
    tidy_results[
        pd.to_numeric(tidy_results["n_positive"], errors="coerce")
        .ge(args.count_threshold)
    ].to_csv(out / "auc_results_filtered.csv", index=False)
    print(f"Plotting-ready AUC rows: {len(tidy_results):,} written to "
          f"{out / 'auc_results.csv'}")
    registry = diseases.copy()
    registry["curve_arrays_retained"] = registry["token_id"].astype(int).isin(
        retain_ids)
    registry.to_csv(out / "token_evaluation_registry.csv", index=False)

    run_settings = {
        **{key: str(value) for key, value in vars(args).items()},
        "checkpoint_block_size": checkpoint_block_size,
        "resolved_evaluation_block_size": evaluation_block_size,
        "n_patients": len(batch_np[0]),
        "n_evaluated_tokens": len(diseases),
        "n_curve_retained_tokens": len(retained_diseases),
        "model_name_resolved": model_name,
        "model_directory_resolved": model_directory,
    }
    pd.DataFrame(
        [{"setting": key, "value": value} for key, value in run_settings.items()]
    ).to_csv(out / "evaluation_run_config.csv", index=False)

    save_pr_curves(curves, results, model_name, prediction_gap_months, out)
    save_roc_curves(curves, results, model_name, prediction_gap_months, out)
    save_prediction_rows(curves, results, model_name, prediction_gap_months, out)
    save_plotting_summaries(
        results, tidy_results, args.count_threshold, age_groups,
        args.age_step, out,
    )
    save_summaries(results, args.count_threshold, out)
    plot_curves(curves, results, args.count_threshold, args.max_curve_plots, out)
    plot_summaries(results, args.count_threshold, age_groups, args.age_step, out)
    if args.extended_analysis:
        run_extended_analysis(model, batch, batch_np, retained_logits, curves,
                              results, retained_diseases, raw_sex_masks, args, out)
    save_csv_manifest(out)
    print(f"Done: {len(results)} disease/sex results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
