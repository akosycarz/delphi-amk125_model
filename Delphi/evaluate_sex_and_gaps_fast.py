#!/usr/bin/env python3
"""Fast AUC-only evaluation for several gaps and sex strata in one inference pass.

Unlike run_evaluation.py, this script does not export ROC/PR curves,
predictions, age strata, calibration, attention, or diagnostic figures. Model
inference is performed once per disease chunk and reused for every requested
minimum prediction-to-event gap.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Support installation as either Delphi/evaluate_sex_and_gaps_fast.py or
# Delphi/scripts/evaluate_sex_and_gaps_fast.py.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR if (SCRIPT_DIR / "run_evaluation.py").is_file() else SCRIPT_DIR.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from run_evaluation import (  # noqa: E402
    delong_auc, disease_tokens, load_model, load_split, load_token_dict,
    resolve_sex_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--gaps-months", nargs="+", type=float,
                        default=[0, 6, 12, 60, 120])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--disease-chunk-size", type=int, default=64)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--dataset-subset-size", type=int, default=-1)
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def run_logits(model, batch, disease_ids, batch_size, device):
    x, ages, targets, target_ages = batch
    output = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(x), batch_size), desc="Patient batches",
                          leave=False):
            end = min(start + batch_size, len(x))
            raw = model(x[start:end].to(device), ages[start:end].to(device),
                        targets[start:end].to(device),
                        target_ages[start:end].to(device))[0].float()
            output.append(raw[:, :, disease_ids].cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def resolve_sex_tokens_from_batch(dictionary: pd.DataFrame,
                                  input_tokens: np.ndarray) -> dict[str, int]:
    """Resolve sex IDs using dictionary text and their presence in the batch.

    The raw binary vocabulary is shifted by +1 inside get_batch(). Some richer
    dictionaries also contain derived variables whose labels mention a sex.
    Selecting the candidate that identifies the largest patient cohort handles
    both cases without relying on global hard-coded token IDs.
    """
    id_column = next((column for column in ("token_id", "token", "index", "id")
                      if column in dictionary.columns), None)
    if id_column is None:
        raw_ids = pd.Series(dictionary.index, index=dictionary.index)
    else:
        raw_ids = pd.to_numeric(dictionary[id_column], errors="coerce")
    text_columns = [column for column in
                    ("token_wording", "readable_name", "icd_code", "coding")
                    if column in dictionary.columns]
    text = dictionary[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    resolved: dict[str, int] = {}
    diagnostics = []
    for sex in ("female", "male"):
        match = text.str.contains(
            rf"(?i)(?:^|[^a-z]){sex}(?:$|[^a-z])", regex=True)
        dictionary_ids = raw_ids[match].dropna().astype(int).unique().tolist()
        # Test both conventions because get_batch shifts raw binary IDs by one.
        candidate_ids = sorted(set(dictionary_ids + [value + 1 for value in dictionary_ids]))
        counts = {
            token: int((input_tokens == token).any(axis=1).sum())
            for token in candidate_ids
        }
        usable = {token: count for token, count in counts.items() if count > 0}
        diagnostics.append(f"{sex} candidates={counts}")
        if not usable:
            raise RuntimeError(
                f"No {sex} dictionary candidate occurs in the batched input. "
                + "; ".join(diagnostics))
        # The true demographic token should occur in roughly half the cohort;
        # derived female-/male-specific variables occur in fewer participants.
        resolved[sex] = max(usable, key=usable.get)
    if resolved["female"] == resolved["male"]:
        raise RuntimeError("Female and Male resolved to the same batched token ID")
    print("Sex-token resolution: " + "; ".join(diagnostics))
    return resolved


def auc_for_gap(token, scores, batch_np, patient_mask, offset_days):
    ages = batch_np[1][patient_mask]
    targets = batch_np[2][patient_mask]
    target_ages = batch_np[3][patient_mask]
    scores = scores[patient_mask]
    events = np.where(targets == token)
    if len(events[0]) < 2:
        return None
    prediction_index = (
        ages[events[0]] < target_ages[events].reshape(-1, 1) - offset_days
    ).sum(axis=1) - 1
    valid = prediction_index >= 0
    if valid.sum() < 2:
        return None
    case_patients = events[0][valid]
    case_index = prediction_index[valid]
    _, first = np.unique(case_patients, return_index=True)
    case_patients, case_index = case_patients[first], case_index[first]
    case_scores = scores[case_patients, case_index].astype(float)

    has_disease = np.zeros(len(targets), dtype=bool)
    has_disease[events[0]] = True
    control_patients = np.where(~has_disease)[0]
    if len(control_patients) < 2:
        return None
    valid_positions = ages[control_patients] > -1000
    control_index = np.where(
        valid_positions, np.arange(ages.shape[1])[None, :], -1
    ).max(axis=1)
    usable = control_index >= 0
    control_patients, control_index = control_patients[usable], control_index[usable]
    if len(control_patients) < 2:
        return None
    control_scores = scores[control_patients, control_index].astype(float)
    result = delong_auc(case_scores, control_scores)
    if result is None:
        return None
    return {
        "n_positive": len(case_patients),
        "n_negative": len(control_patients),
        "n_total": len(case_patients) + len(control_patients),
        "auc": result[0], "auc_lower": result[1], "auc_upper": result[2],
    }


def main() -> None:
    args = parse_args()
    if args.min_events < 1:
        raise ValueError("--min-events must be positive")
    if any(value < 0 for value in args.gaps_months):
        raise ValueError("Prediction gaps cannot be negative")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.checkpoint, args.data_dir / f"{args.split}.bin",
                 args.data_dir / "token_dictionary.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, args.device)
    dictionary = load_token_dict(args.data_dir)
    sex_tokens = resolve_sex_tokens(dictionary)
    batch, raw_sex_masks = load_split(
        args.data_dir, args.split, int(model.config.block_size),
        args.no_event_token_rate, args.dataset_subset_size, sex_tokens,
    )
    batch_np = [tensor.numpy() for tensor in batch]
    diseases = disease_tokens(dictionary)
    masks = {
        "Full": np.ones(len(batch_np[0]), dtype=bool),
        "Female": raw_sex_masks["female"],
        "Male": raw_sex_masks["male"],
    }
    print("Sex tokens: " + ", ".join(f"{key}={value}"
                                      for key, value in sex_tokens.items()))
    print("Patients: " + ", ".join(f"{key}={int(value.sum()):,}"
                                    for key, value in masks.items()))

    rows = []
    gaps = sorted(set(args.gaps_months))
    for start in tqdm(range(0, len(diseases), args.disease_chunk_size),
                      desc="Disease chunks"):
        chunk = diseases.iloc[start:start + args.disease_chunk_size]
        ids = chunk["token_id"].astype(int).tolist()
        logits = run_logits(model, batch, ids, args.batch_size, args.device)
        for column, (_, disease) in enumerate(chunk.iterrows()):
            token = int(disease.token_id)
            scores = logits[:, :, column]
            for months in gaps:
                offset_days = months / 12.0 * 365.25
                for sex, mask in masks.items():
                    metric = auc_for_gap(token, scores, batch_np, mask, offset_days)
                    if metric is None:
                        continue
                    rows.append({
                        "model": args.model_name,
                        "model_directory": args.checkpoint.parent.name,
                        "token_id": token,
                        "icd_code": disease.icd_code,
                        "disease": disease.readable_name,
                        "chapter": disease.chapter,
                        "sex": sex,
                        "age_group": "Full",
                        "horizon_months": float(months),
                        "prediction_gap_months": float(months),
                        "horizon_definition": "minimum lead time before observed event",
                        **metric,
                    })
        del logits
        gc.collect()

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No AUC results were computed")
    results.to_csv(args.output_dir / "auc_results_all_gaps.csv", index=False)
    # Conventional name lets plot_auc_summary.py consume this directory.
    results.to_csv(args.output_dir / "auc_results.csv", index=False)
    filtered = results[results["n_positive"].ge(args.min_events)]
    filtered.to_csv(args.output_dir / "auc_results_all_gaps_filtered.csv", index=False)
    coverage = (results.groupby(["horizon_months", "sex"])["icd_code"]
                .nunique().rename("n_diseases").reset_index())
    coverage.to_csv(args.output_dir / "evaluation_coverage_by_gap_and_sex.csv",
                    index=False)
    print(coverage.to_string(index=False))
    print(f"Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
