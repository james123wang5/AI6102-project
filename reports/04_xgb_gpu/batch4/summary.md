# Stage 04 · Batch 4 summary

Reference: batch 3 frozen configuration `{"model": "lgb", "params_name": "l255", "feature_set": "s_stock", "recency": true, "seeds": [2026, 2027, 2028], "mode": "index", "alpha": 0.95}`.
Keep rule: pooled dev MAE improves and at least 2 of 3 folds improve.

## Step-by-step dev results (single seed, before post-processing)

| step | set | role | fold_1 | fold_2 | fold_3 | pooled | delta_pooled | folds_better | kept |
|---|---|---|---|---|---|---|---|---|---|
| A/B | s_stock | reference | 5.92067 | 6.30457 | 6.28888 | 6.17910 | 0.00000 |  |  |
| A/B | h_hist | candidate | 5.91562 | 6.30253 | 6.28576 | 6.17571 | -0.00338 | 3/3 | True |
| A/B | h_intra | candidate | 5.91973 | 6.30572 | 6.28917 | 6.17927 | 0.00017 | 1/3 | False |
| C stock_id | h_hist | reference | 5.91562 | 6.30253 | 6.28576 | 6.17571 | 0.00000 |  |  |
| C stock_id | h_hist@num | candidate | 5.90528 | 6.29179 | 6.27368 | 6.16459 | -0.01112 | 3/3 | True |
| C stock_id | h_hist@drop | candidate | 5.90582 | 6.29110 | 6.27417 | 6.16473 | -0.01098 | 3/3 | True |

Final feature set: **h_hist@num**

## Seeds

| seed | fold_1 | fold_2 | fold_3 | pooled |
|---|---|---|---|---|
| 2026 | 5.90528 | 6.29179 | 6.27368 | 6.16459 |
| 2027 | 5.90365 | 6.29132 | 6.27468 | 6.16431 |
| 2028 | 5.90595 | 6.29094 | 6.27470 | 6.16492 |
| ensemble | 5.90067 | 6.28645 | 6.26923 | 6.15981 |

## Dev comparison (after post-processing)

| model | fold_1 | fold_2 | fold_3 | pooled | vs_zero_pct |
|---|---|---|---|---|---|
| Zero | 6.01730 | 6.43783 | 6.42367 | 6.30152 | 0.00000 |
| Batch 3 final (s_stock, 3 seeds, index, alpha=0.95) | 5.91082 | 6.29693 | 6.28292 | 6.17140 | -2.06498 |
| Batch 4 h_hist@num, 3-seed avg (no post-processing) | 5.90067 | 6.28645 | 6.26923 | 6.15981 | -2.24880 |
| Batch 4 final (h_hist@num, index, alpha=1.0) | 5.89613 | 6.28155 | 6.26602 | 6.15566 | -2.31468 |

Day-block bootstrap batch 4 - batch 3 on dev: diff -0.01574, 95% CI [-0.01808, -0.01343].

## Frozen configuration

```json
{
  "model": "lgb",
  "params_name": "l255",
  "feature_set": "h_hist@num",
  "recency": true,
  "seeds": [
    2026,
    2027,
    2028
  ],
  "mode": "index",
  "alpha": 1.0
}
```

## Holdout (dates 400-480; second use, after batch 3; nothing selected on it)

| model | mae | vs_zero_pct | rows |
|---|---|---|---|
| Zero | 5.99364 | 0.00000 | 890999 |
| Batch 3 final | 5.86406 | -2.16192 | 890999 |
| Batch 4 final (h_hist@num, index, alpha=1.0) | 5.84474 | -2.48426 | 890999 |

Day-block bootstrap batch 4 - batch 3 on holdout: diff -0.01932, 95% CI [-0.02272, -0.01628].
