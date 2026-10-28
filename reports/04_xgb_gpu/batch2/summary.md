# Stage 04 · Batch 2 summary

Params from batch 1: `l255`. Development folds only. MAE, lower is better.

## New feature groups (cumulative on the V1 set)

| model | n_features | fold_1 | fold_2 | fold_3 | pooled | mean_best_round |
|---|---|---|---|---|---|---|
| E5 +Alpha101-inspired | 78 | 5.92116 | 6.31077 | 6.29428 | 6.18322 | 191.00000 |
| N1 +market aggregates | 85 | 5.92049 | 6.31027 | 6.29445 | 6.18291 | 255.33333 |
| N2 +synthetic-index features | 96 | 5.91867 | 6.30738 | 6.28826 | 6.17912 | 183.00000 |
| N3 +stock statistics | 103 | 5.92134 | 6.30744 | 6.28817 | 6.17993 | 221.66667 |

## Post-processing

`equal`: subtract the per-time equal-weight mean prediction; `index`: subtract the index-weighted mean (weights fitted on each fold's training dates). `best_alpha`: shrink factor chosen on the dev folds.

| feature_set | mode | pooled_alpha_1 | best_alpha | pooled_best_alpha |
|---|---|---|---|---|
| alpha101 | none | 6.18322 | 0.90000 | 6.18208 |
| alpha101 | equal | 6.18014 | 0.95000 | 6.17958 |
| alpha101 | index | 6.17759 | 0.95000 | 6.17710 |
| s_market | none | 6.18291 | 0.85000 | 6.18035 |
| s_market | equal | 6.18218 | 0.90000 | 6.18026 |
| s_market | index | 6.17976 | 0.90000 | 6.17788 |
| s_index | none | 6.17912 | 0.90000 | 6.17780 |
| s_index | equal | 6.17944 | 0.90000 | 6.17853 |
| s_index | index | 6.17705 | 0.90000 | 6.17619 |
| s_stock | none | 6.17993 | 0.90000 | 6.17785 |
| s_stock | equal | 6.18006 | 0.90000 | 6.17850 |
| s_stock | index | 6.17769 | 0.90000 | 6.17614 |

Selected for batch 3: feature set **s_stock**, mode **index**, alpha **0.9**, pooled dev MAE **6.17614**.
