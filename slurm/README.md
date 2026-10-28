# NTU / TC2 batch jobs

`run_baseline_fold1_cpu.sbatch` runs the first baseline validation fold with
LightGBM's CPU learner. It requests 8 CPUs and the account maximum of 30 GB
RAM, and writes scheduler output to the project root.

## Submit

```bash
sbatch slurm/run_baseline_fold1_cpu.sbatch
```

## Inspect

```bash
squeue -u wang2496
tail -f slurm_optiver_cpu_f1_<job_id>.out
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

When the job completes, baseline metrics and predictions are written beneath
`artifacts/02_baselines/` in the server-side project directory.

## Feature-group experiments (stage 03/04)

```bash
sbatch slurm/run_feature_experiments.sbatch
tail -f slurm_optiver_feat_<job_id>.out     # shows "[k/15]" progress lines
```

Results: `reports/03_feature_experiments/` (`feature_experiments.csv`,
`mae_summary.csv`, per-run feature importance). Resubmitting resumes unfinished runs.

## Stage 04: LightGBM main chain + XGBoost GPU comparison (batches 1-3)

One job per batch, each needs the previous batch's results. Requests 1 GPU,
10 CPUs and 30 GB RAM: LightGBM (main model) trains on the CPUs, XGBoost
(model-family comparison) on the GPU.

```bash
sbatch slurm/stage04.sbatch 1     # baselines, tuning, re-run ablation, XGBoost comparison, index weights
sbatch slurm/stage04.sbatch 2     # market / index / stock features, neutralization
sbatch slurm/stage04.sbatch 3     # recency, seeds, error analysis, XGBoost comparison, final holdout
tail -f slurm_stage04_<job_id>.out
```

Results: `reports/04_xgb_gpu/batch{1,2,3}/summary.md`; every run is logged in
`reports/04_xgb_gpu/runs.csv`. Resubmitting resumes unfinished runs.
