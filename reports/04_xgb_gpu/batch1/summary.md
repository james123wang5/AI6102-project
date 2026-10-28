# Stage 04 · Batch 1 summary

Development folds only (holdout dates 400-480 untouched). MAE, lower is better; `pooled` is weighted by rows.

## Constant baselines

| fold | eval_rows | zero | median | train_median |
|---|---|---|---|---|
| fold_1 | 547250 | 6.01730 | 6.01775 | -0.07987 |
| fold_2 | 549669 | 6.43783 | 6.43782 | -0.06974 |
| fold_3 | 659945 | 6.42367 | 6.42316 | -0.06020 |
| pooled | 1756864 | 6.30152 | 6.30146 |  |

## Tuning on the V1 full feature set (LightGBM main; XGBoost GPU as model comparison)

| model | fold_1 | fold_2 | fold_3 | pooled | mean_best_round |
|---|---|---|---|---|---|
| lgb l63 | 5.92661 | 6.31239 | 6.29478 | 6.18561 | 541.00000 |
| lgb l127 | 5.92263 | 6.31046 | 6.29598 | 6.18422 | 341.33333 |
| lgb l255 | 5.92116 | 6.31077 | 6.29428 | 6.18322 | 191.00000 |
| xgb d6 | 5.92125 | 6.30788 | 6.29464 | 6.18247 | 576.33333 |
| xgb d8 | 5.91898 | 6.30575 | 6.29266 | 6.18036 | 381.66667 |
| xgb d10 | 5.91738 | 6.30514 | 6.29085 | 6.17899 | 275.66667 |

Selected LightGBM params: **l255** (lowest pooled MAE).

## Cumulative feature-group ablation (tuned params)

| model | n_features | fold_1 | fold_2 | fold_3 | pooled | mean_best_round |
|---|---|---|---|---|---|---|
| E1 raw fields | 13 | 5.95636 | 6.35275 | 6.33099 | 6.22110 | 171.00000 |
| E2 +Optiver microstructure | 25 | 5.93411 | 6.33131 | 6.31114 | 6.20001 | 137.00000 |
| E3 +intraday temporal | 61 | 5.92403 | 6.31362 | 6.29755 | 6.18623 | 233.33333 |
| E4 +cross-section | 73 | 5.91958 | 6.30845 | 6.29348 | 6.18170 | 231.00000 |
| E5 +Alpha101-inspired | 78 | 5.92116 | 6.31077 | 6.29428 | 6.18322 | 191.00000 |

## Synthetic index fit

`r2`: share of the implied index return explained by the weighted stock returns. `implied_dispersion_bp`: median cross-stock spread of the implied index return; near 0 means the target definition is reproduced exactly.

| fit | train_end | r2 | n_times | implied_dispersion_bp | nonzero_weights | max_weight |
|---|---|---|---|---|---|---|
| fold_1 | 239 | 0.99996 | 11760 | 0.00502 | 199 | 0.04063 |
| fold_2 | 289 | 0.99996 | 14210 | 0.00502 | 199 | 0.04066 |
| fold_3 | 339 | 0.99996 | 16660 | 0.00502 | 200 | 0.04045 |
| holdout_train | 399 | 0.99996 | 19600 | 0.00502 | 200 | 0.04028 |

Weight correlation across folds:

| index | fold_1 | fold_2 | fold_3 | holdout_train |
|---|---|---|---|---|
| fold_1 | 1.0000 | 1.0000 | 0.9989 | 0.9985 |
| fold_2 | 1.0000 | 1.0000 | 0.9989 | 0.9986 |
| fold_3 | 0.9989 | 0.9989 | 1.0000 | 0.9999 |
| holdout_train | 0.9985 | 0.9986 | 0.9999 | 1.0000 |
