**English** | [中文](README.zh-CN.md)

# AI6102 Team Project — Optiver: Trading at the Close

Predicting 60-second stock returns relative to a synthetic index in the Nasdaq closing auction, using the completed Kaggle competition [Optiver – Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close).

This repository contains the code and every experimental output behind our AI6102 report. All numbers below are copied from files under [`reports/`](reports/).

## Final result

Offline holdout (dates 400–480, 890,999 rows; zero forecast MAE 5.9936):

| Model | Holdout MAE | vs. zero |
|---|---|---|
| LightGBM, 13 raw fields | 5.9104 | −1.39% |
| LightGBM, V1 full feature set | 5.8711 | −2.04% |
| Batch 3 pipeline (index features, centering, recency weighting, 3 seeds) | 5.8641 | −2.16% |
| Batch 4 LightGBM (+ historical targets, numeric stock ID) | 5.8447 | −2.48% |
| GRU (time axis) / Transformer (stock axis), standalone | 5.8648 / 5.8572 | −2.15% / −2.28% |
| **Final blend: 0.55 LightGBM + 0.20 GRU + 0.25 Transformer** | **5.8393** | **−2.57%** |

Blend minus LightGBM, paired day-bootstrap: dev −0.00568 (95% CI [−0.00698, −0.00431]); holdout −0.00541 (95% CI [−0.00978, −0.00261]).

**Leaderboard.** Late submission is disabled for this competition (the "Late Submission" button on Kaggle is greyed out), so we cannot obtain an official leaderboard score. Instead, our holdout uses exactly the same date range (400–480) as the first-place team's published offline validation, which allows a direct reference comparison:

| Solution (dates 400–480) | MAE | vs. zero |
|---|---|---|
| 1st place, single CatBoost (with post-processing) | 5.8240 | −2.83% |
| 1st place, single GRU / Transformer | 5.8481 / 5.8619 | −2.43% / −2.20% |
| 1st place, CatBoost + GRU + Transformer | 5.8117 | −3.03% |
| **Ours, final blend** | **5.8393** | **−2.57%** |

Source: [1st-place write-up](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution). The final private leaderboard (1st place 5.4030) is scored on post-competition market data and is not comparable to offline MAE.

## Task

For stock *i* at snapshot *t*, the label is

```text
target = 10,000 × [ WAP_i(t+60s) / WAP_i(t) − Index(t+60s) / Index(t) ]   (basis points)
```

where the index is a weighted basket of the 200 stocks with unpublished weights. The metric is MAE. Data: 481 trading dates × ~200 stocks × 55 ten-second snapshots (5,237,980 rows) of order-book and auction fields.

Main difficulties: very low signal-to-noise (good models beat the zero forecast by only 2–3%); non-stationarity (zero-forecast MAE ranges from 6.02 to 6.44 between folds); strict look-ahead control; unknown index weights.

## Validation protocol

| Split | Train dates | Eval dates | Rows |
|---|---|---|---|
| fold_1 | 0–239 | 240–289 | 547,250 |
| fold_2 | 0–289 | 290–339 | 549,669 |
| fold_3 | 0–339 | 340–399 | 659,945 |
| holdout | 0–399 | 400–480 | 890,999 |

- All design choices use the row-weighted **pooled** MAE over the three expanding-window dev folds.
- Each fit early-stops on the last 20 training dates, then refits on the full training window.
- Anything that is fitted (index weights, stock statistics, neural scalers) is fitted inside each split's training dates only.
- The holdout was evaluated after batches 3, 4 and 5 (three uses). Each batch froze its configuration on dev before looking at the holdout; this is disclosed in the report.

## Method

1. **Feature groups (LightGBM, L1 objective)**: raw fields → microstructure (spreads, order-book/auction imbalances) → intraday history (lags, diffs, rolling stats) → same-time cross-section → Alpha101-inspired operators.
2. **Recovering the synthetic index**: implied index return `r_i − target/1e4`, median across stocks, then NNLS regression on stock returns. R² = 0.99996 on every training window; weight correlation across windows ≥ 0.998.
3. **Index features and index-weighted centering**: stock returns relative to the recovered index; at each snapshot subtract the index-weighted mean prediction, then scale by α chosen on dev.
4. **Recency weighting** (120-day half-life) and **3-seed averaging**.
5. **Historical-target features**: rolling statistics of past dates' targets for the same (stock, snapshot time), using only dates ≤ d−1 (the competition API reveals previous-day targets). Stock ID switched from categorical to numeric.
6. **Neural models and blending**: a GRU over each stock's 55-step intraday sequence and a Transformer across stocks at each snapshot (no positional encoding, absent stocks masked, index-centered output); 3 layers, width 128, same L1 / recency / refit protocol. Blend weights and α are grid-searched on dev (step 0.05).

## Key experiments (pooled dev MAE; zero = 6.30152)

**Cumulative feature ablation** (tuned LightGBM, [`batch1/ablation.csv`](reports/04_xgb_gpu/batch1/ablation.csv)):

| Inputs | # features | Dev MAE |
|---|---|---|
| Raw fields | 13 | 6.22110 |
| + Microstructure | 25 | 6.20001 |
| + Intraday history | 61 | 6.18623 |
| + Cross-section | 73 | 6.18170 |
| + Alpha101-inspired | 78 | 6.18322 (no gain) |

**Index structure** ([`batch2/`](reports/04_xgb_gpu/batch2/)): adding synthetic-index features 6.18291 → 6.17912. With features fixed and α = 1, centering: none 6.17912, equal-weight 6.17944, **index-weighted 6.17705**.

**Pipeline steps** ([`batch3/dev_comparison.csv`](reports/04_xgb_gpu/batch3/)): single seed 6.17993 → + recency weighting & 3 seeds 6.17478 → + index centering (α = 0.95) **6.17140**. XGBoost (GPU) with the same post-processing: 6.16982 (model-family control).

**Historical targets and stock ID** ([`batch4/steps.csv`](reports/04_xgb_gpu/batch4/steps.csv)): reference 6.17910 → + historical targets 6.17571 (3/3 folds better) → numeric stock ID 6.16459 / dropped stock ID 6.16473 (3/3 folds better). Same-day relative-return rolling features: 6.17927 (rejected).

**Error breakdown** ([`batch3/`](reports/04_xgb_gpu/batch3/)): gains concentrate late in the auction (−0.43% at 420–540 s vs. −0.06% at 0–120 s) and in liquid stocks (≈ −0.3% in the top two liquidity quintiles vs. −0.11% in the least liquid).

**Final models per fold** ([`batch5/`](reports/04_xgb_gpu/batch5/)):

| Model | Fold 1 | Fold 2 | Fold 3 | Dev pooled | Holdout |
|---|---|---|---|---|---|
| Zero | 6.01730 | 6.43783 | 6.42367 | 6.30152 | 5.99364 |
| LightGBM | 5.89613 | 6.28155 | 6.26602 | 6.15566 | 5.84474 |
| GRU | 5.91285 | 6.29780 | 6.28100 | 6.17158 | 5.86482 |
| Transformer | 5.90909 | 6.28713 | 6.27793 | 6.16592 | 5.85717 |
| **Blend** | **5.89383** | **6.27442** | **6.25874** | **6.14998** | **5.83933** |

## Repository layout

```text
src/ai6102/
  features/            feature groups: optiver, temporal, cross_section, alpha_adapted, market
  index_model.py       index-weight recovery (NNLS) and index-weighted centering
  fold_features.py     per-fold fitted features (index-relative, stock statistics)
  history_features.py  historical-target features (batch 4)
  deep.py              GRU and Transformer (batch 5)
  stage04.py           shared training / caching / run registry / evaluation
scripts/               run_baselines.py, run_feature_experiments.py, run_stage04_batch{1..5}.py
configs/               data paths, fold definitions, parameter grids
slurm/                 cluster job scripts (NTU TC2)
reports/               all experimental outputs (CSV + summary.md per batch)
docs/                  detailed notes (in Chinese): PROJECT_STATUS, BATCH5_DL_REPORT, data dictionary, factor registry, original research plan
```

Each batch folder in `reports/04_xgb_gpu/batch*/` has an English `summary.md`.

## Reproducing

```bash
# data: download train.csv from Kaggle into optiver-trading-at-the-close/ (not committed)
pip install -e .
python scripts/run_data_audit.py
python scripts/run_baselines.py
python scripts/run_feature_experiments.py
# stage 04, on a Slurm cluster (each batch reads the previous one's outputs)
sbatch slurm/stage04.sbatch 1   # then 2, 3, 4
sbatch slurm/stage05.sbatch     # batch 5: GRU / Transformer / blend (1×A40)
```

Local smoke test: add `--smoke-stocks 12 --smoke-rounds 40 --xgb-device cpu` to a stage-04 script. Feature caches and prediction files (`artifacts/`, ~3 GB) are not committed.

## Limitations

- The holdout was used three times (after batches 3, 4, 5); every choice was frozen on dev first.
- Historical-target features assume previous-day labels are available at prediction time, as in the competition API.
- Alpha101-inspired features gave no gain after tuning; neural architectures were not tuned.
- No official leaderboard score because late submission is closed.
- MAE measures forecast accuracy only; no trading or cost evaluation.
