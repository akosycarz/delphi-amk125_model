#!/usr/bin/env python3
"""Evaluate Delphi AUC by age and sex at several prediction-event gaps.

The program calls ``run_evaluation.py`` once per gap, then combines its tidy
age- and sex-stratified AUC tables. A gap is a minimum lead time before an
observed event, not a fixed absolute-risk horizon.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def arguments() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--gaps-months", nargs="+", type=float,
                        default=[0, 6, 12, 60, 120])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block-size", type=int, default=230)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--disease-chunk-size", type=int, default=256)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--dataset-subset-size", type=int, default=-1)
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--age-min", type=int, default=40)
    parser.add_argument("--age-max", type=int, default=80)
    parser.add_argument("--age-step", type=int, default=5)
    parser.add_argument("--evaluator", type=Path, default=project / "run_evaluation.py")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    for name in ("checkpoint", "data_dir", "output_dir", "evaluator"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    required = [args.checkpoint, args.evaluator,
                args.data_dir / f"{args.split}.bin",
                args.data_dir / "token_dictionary.csv"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" +
                                "\n".join(f"  {path}" for path in missing))
    if any(gap < 0 for gap in args.gaps_months):
        raise ValueError("Prediction gaps cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for months in sorted(set(args.gaps_months)):
        label = f"{months:g}".replace(".", "p")
        gap_dir = args.output_dir / f"gap_{label}_months"
        auc_file = gap_dir / "auc_results.csv"
        if args.force or not auc_file.is_file() or not auc_file.stat().st_size:
            gap_dir.mkdir(parents=True, exist_ok=True)
            offset_days = months / 12.0 * 365.25
            command = [
                sys.executable, str(args.evaluator),
                "--input_path", str(args.data_dir),
                "--model_ckpt_path", str(args.checkpoint),
                "--output_path", str(gap_dir),
                "--model_name", args.model_name,
                "--split", args.split,
                "--device", args.device,
                "--block_size", str(args.block_size),
                "--batch_size", str(args.batch_size),
                "--disease_chunk_size", str(args.disease_chunk_size),
                "--no_event_token_rate", str(args.no_event_token_rate),
                "--count_threshold", str(args.min_events),
                "--dataset_subset_size", str(args.dataset_subset_size),
                "--age_min", str(args.age_min),
                "--age_max", str(args.age_max),
                "--age_step", str(args.age_step),
                "--offset", str(offset_days),
                "--no-extended_analysis",
            ]
            print(f"Evaluating {months:g}-month gap", flush=True)
            subprocess.run(command, check=True)
        frame = pd.read_csv(auc_file)
        needed = {"model", "icd_code", "age_group", "sex", "auc", "n_positive"}
        absent = needed - set(frame.columns)
        if absent:
            raise ValueError(f"{auc_file} is missing columns: {sorted(absent)}")
        frame["horizon_months"] = float(months)
        frame["prediction_gap_months"] = float(months)
        frame["horizon_definition"] = "minimum lead time before observed event"
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(args.output_dir / "auc_results_all_gaps.csv", index=False)
    combined.to_csv(args.output_dir / "auc_results.csv", index=False)
    filtered = combined[pd.to_numeric(combined["n_positive"], errors="coerce")
                        .ge(args.min_events)]
    filtered.to_csv(args.output_dir / "auc_results_all_gaps_filtered.csv", index=False)
    coverage = (combined.groupby(["horizon_months", "sex", "age_group"])
                ["icd_code"].nunique().rename("n_diseases").reset_index())
    coverage.to_csv(args.output_dir / "evaluation_coverage_by_gap_sex_and_age.csv",
                    index=False)
    print(f"Created {args.output_dir / 'auc_results_all_gaps.csv'}")


if __name__ == "__main__":
    main()
