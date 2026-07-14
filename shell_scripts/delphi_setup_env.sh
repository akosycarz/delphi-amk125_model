#!/bin/bash
# Run once to set up the Delphi conda environment on the HPC
# Usage: bash delphi_setup_env.sh

set -e

DELPHI_DIR="/rds/general/project/hda_24-25/live/amk125_thesis/Delphi"

echo "=== Setting up Delphi conda environment ==="
module load anaconda3/personal 2>/dev/null || true

# Create environment
conda create -n delphi python=3.11 -y

# Install requirements
conda run -n delphi pip install -r "$DELPHI_DIR/requirements.txt"

# Also install pyarrow for reading data
conda run -n delphi pip install pyarrow

echo "=== Environment setup complete ==="
conda run -n delphi python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
