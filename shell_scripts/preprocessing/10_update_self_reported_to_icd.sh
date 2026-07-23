#!/bin/bash
#PBS -N icd10_update
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs/update_with_chapter_files_with_icd10.log

set -euo pipefail

R_SCRIPT="${HOME}/delphi-amk125_model/scripts/10-update_with_chapter_files_with_icd10.R"
TEMP_DIR="/rds/general/user/amk125/ephemeral"
LOG_DIR="${TEMP_DIR}/logs"

echo "Starting ICD-10 update at $(date)"
echo "Compute node: $(hostname)"
echo "R script: ${R_SCRIPT}"

mkdir -p "${TEMP_DIR}" "${LOG_DIR}"

if [[ ! -f "${R_SCRIPT}" ]]; then
echo "ERROR: R script not found: ${R_SCRIPT}" >&2
exit 1
fi

# Initialise Conda and activate the R 4.1.3 environment
eval "$("${HOME}/anaconda3/bin/conda" shell.bash hook)"
conda activate r413

# Use ephemeral storage for any additional temporary files
export TMPDIR="${TEMP_DIR}"

echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Rscript location: $(command -v Rscript)"
echo "R version:"
Rscript --version

# Check that the required R package is available
Rscript -e '
if (!requireNamespace("data.table", quietly = TRUE)) {
    stop(
        "The data.table package is not installed in the active Conda environment."
    )
}
cat("data.table version:", as.character(packageVersion("data.table")), "\n")
'

echo "Running ICD-10 update..."

Rscript "${R_SCRIPT}"

echo "ICD-10 update completed successfully at $(date)"