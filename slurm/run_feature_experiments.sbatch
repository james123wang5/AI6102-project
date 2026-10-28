#!/usr/bin/env bash
# Full stage-03/04 feature-group experiments: 5 experiments x 3 dev folds = 15 runs.
# Submit from the project root with:
#   sbatch slurm/run_feature_experiments.sbatch
# Safe to resubmit: finished (experiment, fold) pairs are skipped.

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --job-name=optiver_feat
#SBATCH --output=slurm_optiver_feat_%j.out
#SBATCH --error=slurm_optiver_feat_%j.err

set -eo pipefail
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate optiver
set -u

cd /home/msai/wang2496/AI6102-project

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

# -u: unbuffered output so `tail -f` shows progress immediately.
python -u scripts/run_feature_experiments.py

echo "Finished: $(date)"
