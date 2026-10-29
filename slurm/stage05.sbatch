#!/usr/bin/env bash
# Stage 04 batch 5: GRU + Transformer on the A40, then blending with the batch-4 LightGBM ensemble.
# Submit from the project root after batch 4:
#   sbatch slurm/stage05.sbatch
# Safe to resubmit: finished runs are loaded from disk instead of retrained.

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=optiver_s05
#SBATCH --output=slurm_stage05_%j.out
#SBATCH --error=slurm_stage05_%j.err

set -eo pipefail
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate optiver
set -u

cd /home/msai/wang2496/AI6102-project

echo "Host: $(hostname)  Started: $(date)  CPUs: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# PyTorch with CUDA; installed once into the conda env if missing.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Installing PyTorch (CUDA build) ..."
    pip install --no-cache-dir torch
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

python -u scripts/run_stage04_batch5.py --skip-xgb --dl-device cuda

echo "Finished: $(date)"
