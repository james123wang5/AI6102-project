# Stage 04 · Batch 3 summary

## Development folds

Recency weighting: pooled dev MAE 6.17910 with vs 6.17993 without -> used: **True**.

Seed averaging:

| seed | pooled |
|---|---|
| 2026 | 6.17910 |
| 2027 | 6.17987 |
| 2028 | 6.17765 |
| ensemble | 6.17478 |

| model | fold_1 | fold_2 | fold_3 | pooled | vs_zero_pct |
|---|---|---|---|---|---|
| Zero | 6.01730 | 6.43783 | 6.42367 | 6.30152 | 0.00000 |
| E1 raw fields (tuned LightGBM) | 5.95636 | 6.35275 | 6.33099 | 6.22110 | -1.27615 |
| E5 V1 all features | 5.92116 | 6.31077 | 6.29428 | 6.18322 | -1.87740 |
| Best set s_stock (1 seed) | 5.92134 | 6.30744 | 6.28817 | 6.17993 | -1.92951 |
| +recency=True, 3-seed avg | 5.91626 | 6.30005 | 6.28482 | 6.17478 | -2.01121 |
| Final: + index neutralization, alpha=0.95 | 5.91082 | 6.29693 | 6.28292 | 6.17140 | -2.06498 |
| [compare] XGBoost GPU d10, s_stock, 1 seed | 5.91422 | 6.29882 | 6.28418 | 6.17352 | -2.03121 |
| [compare] XGBoost + same post-processing | 5.90764 | 6.29685 | 6.28144 | 6.16982 | -2.08990 |
| [compare] 50/50 LightGBM+XGBoost + post-processing | 5.90634 | 6.29419 | 6.27932 | 6.16779 | -2.12217 |

Day-block bootstrap of the MAE difference (95% CI; negative = final better):

| comparison | diff | ci_low | ci_high | share_negative | n_days |
|---|---|---|---|---|---|
| final - E5 V1 | -0.01182 | -0.01501 | -0.00872 | 1.00000 | 160 |
| final - E1 raw | -0.04971 | -0.05378 | -0.04564 | 1.00000 | 160 |

## Error breakdown (dev folds pooled)

| phase | err_zero | err_v1 | err_final | final_vs_v1_pct |
|---|---|---|---|---|
| 000-120s | 6.17272 | 6.02367 | 6.02008 | -0.05956 |
| 130-290s | 7.35293 | 7.23356 | 7.22681 | -0.09341 |
| 300-410s | 5.85488 | 5.76296 | 5.74724 | -0.27279 |
| 420-540s | 5.46769 | 5.35716 | 5.33408 | -0.43069 |

| liquidity | err_zero | err_v1 | err_final | final_vs_v1_pct |
|---|---|---|---|---|
| Q1 least liquid | 8.82799 | 8.65590 | 8.64681 | -0.10507 |
| Q2 | 6.73182 | 6.61505 | 6.60601 | -0.13673 |
| Q3 | 5.96317 | 5.84458 | 5.83311 | -0.19618 |
| Q4 | 5.28727 | 5.18688 | 5.17073 | -0.31142 |
| Q5 most liquid | 4.64387 | 4.56049 | 4.54717 | -0.29215 |

## Frozen configuration

```json
{
  "model": "lgb",
  "params_name": "l255",
  "feature_set": "s_stock",
  "recency": true,
  "seeds": [
    2026,
    2027,
    2028
  ],
  "mode": "index",
  "alpha": 0.95
}
```

## Final holdout (dates 400-480, evaluated once after freezing)

| model | mae | vs_zero_pct | rows |
|---|---|---|---|
| Zero | 5.99364 | 0.00000 | 890999 |
| Median (train 0-399) | 5.99359 | -0.00074 | 890999 |
| E1 raw fields (tuned LightGBM) | 5.91036 | -1.38939 | 890999 |
| E5 V1 all features | 5.87109 | -2.04468 | 890999 |
| Final pipeline (s_stock, rec=True, 3 seeds, index, alpha=0.95) | 5.86406 | -2.16192 | 890999 |
| [compare] XGBoost GPU d10 + post-processing | 5.86054 | -2.22061 | 890999 |
| [compare] 50/50 LightGBM+XGBoost + post-processing | 5.85988 | -2.23160 | 890999 |

Bootstrap final - V1 on holdout: diff -0.00703, 95% CI [-0.01102, -0.00361].
