# Stage 04 · Batch 5 summary (neural models and blending)

Features: `h_hist@drop`; recency weighting: True; seeds: [2026, 2027, 2028].

```json
{
  "version": "stage05_dl_v1",
  "feature_set": "h_hist@drop",
  "models": [
    "gru",
    "tfm"
  ],
  "seeds": [
    2026,
    2027,
    2028
  ],
  "hidden": 128,
  "gru_layers": 3,
  "tfm_layers": 3,
  "heads": 4,
  "dropout": 0.1,
  "lr": 0.001,
  "weight_decay": 0.01,
  "warmup_epochs": 1,
  "max_epochs": 30,
  "patience": 4,
  "grad_clip": 1.0,
  "days_per_batch_gru": 4,
  "days_per_batch_tfm": 4,
  "feature_clip": 5.0,
  "recency": true,
  "blend_step": 0.05
}
```

## Single runs

| model | seed | fold | mae | best_iteration | seconds |
|---|---|---|---|---|---|
| gru | 2026 | fold_1 | 5.92103 | 8 | 22.50000 |
| gru | 2027 | fold_1 | 5.92377 | 9 | 10.30000 |
| gru | 2028 | fold_1 | 5.92944 | 11 | 12.20000 |
| tfm | 2026 | fold_1 | 5.92077 | 11 | 42.00000 |
| tfm | 2027 | fold_1 | 5.92144 | 10 | 38.30000 |
| tfm | 2028 | fold_1 | 5.92164 | 10 | 38.00000 |
| gru | 2026 | fold_2 | 6.30855 | 7 | 10.20000 |
| gru | 2027 | fold_2 | 6.31139 | 13 | 16.40000 |
| gru | 2028 | fold_2 | 6.31546 | 8 | 11.40000 |
| tfm | 2026 | fold_2 | 6.29399 | 12 | 53.80000 |
| tfm | 2027 | fold_2 | 6.29933 | 13 | 57.80000 |
| tfm | 2028 | fold_2 | 6.30908 | 15 | 65.10000 |
| gru | 2026 | fold_3 | 6.29951 | 13 | 19.30000 |
| gru | 2027 | fold_3 | 6.29291 | 12 | 18.00000 |
| gru | 2028 | fold_3 | 6.30335 | 6 | 10.70000 |
| tfm | 2026 | fold_3 | 6.31048 | 6 | 36.50000 |
| tfm | 2027 | fold_3 | 6.29978 | 15 | 77.30000 |
| tfm | 2028 | fold_3 | 6.29333 | 11 | 58.80000 |
| gru | 2026 | holdout | 5.88501 | 11 | 19.70000 |
| gru | 2027 | holdout | 5.87457 | 9 | 16.90000 |
| gru | 2028 | holdout | 5.89003 | 12 | 21.20000 |
| tfm | 2026 | holdout | 5.89993 | 11 | 69.50000 |
| tfm | 2027 | holdout | 5.86865 | 12 | 74.70000 |
| tfm | 2028 | holdout | 5.87320 | 7 | 48.60000 |

## Dev folds (blend weights and alphas chosen here)

| model | fold_1 | fold_2 | fold_3 | mae | vs_zero_pct |
|---|---|---|---|---|---|
| Zero | 6.01730 | 6.43783 | 6.42367 | 6.30152 | 0.00000 |
| LightGBM (batch 4 final), 3-seed avg, index, alpha=1.0 | 5.89613 | 6.28155 | 6.26602 | 6.15566 | -2.31468 |
| GRU (time axis), 3-seed avg, index, alpha=0.95 | 5.91285 | 6.29780 | 6.28100 | 6.17158 | -2.06205 |
| Transformer (stock axis), 3-seed avg, index, alpha=0.9 | 5.90909 | 6.28713 | 6.27793 | 6.16592 | -2.15192 |
| Blend (lgb 0.55, gru 0.20, tfm 0.25), index, alpha=1.05 | 5.89383 | 6.27442 | 6.25874 | 6.14998 | -2.40483 |

Day-block bootstrap blend - LightGBM on dev: diff -0.00568, 95% CI [-0.00698, -0.00431].

Top blends:

| w_lgb | w_gru | w_tfm | alpha | pooled |
|---|---|---|---|---|
| 0.55000 | 0.20000 | 0.25000 | 1.05000 | 6.14998 |
| 0.55000 | 0.25000 | 0.20000 | 1.05000 | 6.15004 |
| 0.60000 | 0.20000 | 0.20000 | 1.05000 | 6.15004 |
| 0.50000 | 0.25000 | 0.25000 | 1.05000 | 6.15008 |
| 0.60000 | 0.15000 | 0.25000 | 1.05000 | 6.15010 |
| 0.50000 | 0.20000 | 0.30000 | 1.05000 | 6.15013 |
| 0.55000 | 0.15000 | 0.30000 | 1.05000 | 6.15014 |
| 0.60000 | 0.25000 | 0.15000 | 1.05000 | 6.15021 |
| 0.50000 | 0.30000 | 0.20000 | 1.05000 | 6.15026 |
| 0.65000 | 0.15000 | 0.20000 | 1.05000 | 6.15027 |

## Holdout (dates 400-480; third use, after batches 3 and 4; nothing selected on it)

| model | mae | vs_zero_pct | rows |
|---|---|---|---|
| Zero | 5.99364 | 0.00000 | 890999 |
| LightGBM (batch 4 final), 3-seed avg, index, alpha=1.0 | 5.84474 | -2.48425 | 890999 |
| GRU (time axis), 3-seed avg, index, alpha=0.95 | 5.86482 | -2.14915 | 890999 |
| Transformer (stock axis), 3-seed avg, index, alpha=0.9 | 5.85717 | -2.27691 | 890999 |
| Blend (lgb 0.55, gru 0.20, tfm 0.25), index, alpha=1.05 | 5.83933 | -2.57456 | 890999 |

Day-block bootstrap blend - LightGBM on holdout: diff -0.00541, 95% CI [-0.00978, -0.00261].
