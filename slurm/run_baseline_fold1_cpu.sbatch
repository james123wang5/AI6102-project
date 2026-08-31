#!/usr/bin/env bash
# CPU-only smoke run for the first time-validation fold.
# Submit from the project root with:
#   sbatch slurm/run_baseline_fold1_cpu.sbatch

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=optiver_cpu_f1
#SBATCH --output=slurm_optiver_cpu_f1_%j.out
#SBATCH --error=slurm_optiver_cpu_f1_%j.err

set -euo pipefail

module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate optiver

cd /home/msai/wang2496/AI6102-project

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $(python --version)"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK:-unknown}"

# This code path uses LightGBM's CPU learner; no CUDA or PyTorch is involved.
python scripts/run_baselines.py --fold fold_1

echo "Finished: $(date)"
