#!/bin/bash
#PBS -N build_delphi
#PBS -l select=1:ncpus=4:mem=128gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs/build_delphi.log

set -euo pipefail

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate r413

echo "Starting Delphi preprocessing at $(date)"

Rscript \
  "/rds/general/user/amk125/home/delphi-amk125_model/scripts/delphi_preproces.R"

echo "Preprocessing done at $(date)"
