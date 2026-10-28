#!/usr/bin/env bash
# Stage 04 pipeline: LightGBM on the CPUs (main chain) + XGBoost on the GPU (comparison). One job per batch; each batch needs the previous one.
# Submit from the project root with the batch number as the only argument:
#   sbatch slurm/stage04.sbatch 1
#   sbatch slurm/stage04.sbatch 2
#   sbatch slurm/stage04.sbatch 3
# Safe to resubmit: finished runs are loaded from disk instead of retrained.

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=optiver_s04
#SBATCH --output=slurm_stage04_%j.out
#SBATCH --error=slurm_stage04_%j.err

set -eo pipefail
BATCH="${1:?usage: sbatch slurm/stage04.sbatch <1|2|3>}"
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate optiver
set -u

cd /home/msai/wang2496/AI6102-project

echo "Host: $(hostname)  Batch: ${BATCH}"
echo "Started: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# -u: unbuffered output so `tail -f` shows progress immediately.
python -u "scripts/run_stage04_batch${BATCH}.py" --xgb-device cuda

echo "Finished: $(date)"
