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
