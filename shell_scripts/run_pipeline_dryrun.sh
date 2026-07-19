#!/bin/bash
# run_pipeline_dryrun.sh
#
# Submits the full dryrun pipeline as three chained PBS jobs:
#   1. Preprocessing  (R)    — builds train/val/test.bin for all 4 configurations
#   2. Dryrun training (GPU) — trains a tiny model on each configuration
#   3. Dryrun evaluation (GPU) — runs run_evaluation.py on each checkpoint
#
# Each job only starts if the previous one succeeded (afterok dependency).
#
# Usage (from the cluster login node):
#   bash shell_scripts/run_pipeline_dryrun.sh
#
# To monitor:
#   qstat -u $USER
#
# Logs are written to:
#   /rds/general/project/hda_24-25/live/amk125_thesis/logs/

set -euo pipefail

REPO_DIR="/rds/general/project/hda_24-25/live/amk125_thesis/Delphi"
SCRIPTS_DIR="$REPO_DIR/shell_scripts"
LOG_DIR="/rds/general/project/hda_24-25/live/amk125_thesis/logs"

mkdir -p "$LOG_DIR"

echo "======================================================"
echo "  Delphi dryrun pipeline"
echo "  $(date)"
echo "======================================================"

# Pull latest code first
echo ""
echo "Pulling latest code..."
git -C "$REPO_DIR" pull

# 1. Preprocessing
echo ""
echo "Submitting preprocessing job..."
PREPROCESS_JOB=$(qsub "$SCRIPTS_DIR/delphi_preprocess.pbs")
echo "  Job ID: $PREPROCESS_JOB"

# 2. Dryrun training (depends on preprocessing completing successfully)
echo ""
echo "Submitting dryrun training job (depends on $PREPROCESS_JOB)..."
TRAIN_JOB=$(qsub -W depend=afterok:"$PREPROCESS_JOB" "$SCRIPTS_DIR/delphi_dryrun.sh")
echo "  Job ID: $TRAIN_JOB"

# 3. Dryrun evaluation (depends on training completing successfully)
echo ""
echo "Submitting dryrun evaluation job (depends on $TRAIN_JOB)..."
EVAL_JOB=$(qsub -W depend=afterok:"$TRAIN_JOB" "$SCRIPTS_DIR/delphi_evaluate_dryrun.pbs")
echo "  Job ID: $EVAL_JOB"

echo ""
echo "======================================================"
echo "  All jobs submitted."
echo ""
echo "  Pipeline:"
echo "    Preprocess  → $PREPROCESS_JOB"
echo "    Train       → $TRAIN_JOB       (starts after preprocess)"
echo "    Evaluate    → $EVAL_JOB        (starts after train)"
echo ""
echo "  Monitor with:  qstat -u $USER"
echo "  Logs in:       $LOG_DIR"
echo "======================================================"
