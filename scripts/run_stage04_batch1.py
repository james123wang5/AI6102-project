#!/usr/bin/env python3
"""Stage 04 batch 1: baselines, tuning, re-run ablation, index weights.

Steps (development folds only; dates >= 400 are dropped right after loading):
  1. Zero / Median baselines on all three folds.
  2. Tune LightGBM (CPU, main model) on the full V1 feature set (alpha101) over
     a small grid; the best entry by pooled dev MAE goes to `best_params.json`.
  3. Re-run the cumulative feature-group ablation (raw -> alpha101) with it.
  4. Model-family comparison: XGBoost (GPU) on the same V1 features with a
     grid of the same size (skipped with --skip-xgb).
  5. Fit synthetic-index weights per fold (and on 0..399 for batch 3), report R^2.

Outputs: reports/04_xgb_gpu/batch1/. Resubmitting skips finished runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai6102 import stage04 as s4  # noqa: E402
from ai6102.baselines import TARGET_COLUMN  # noqa: E402

ABLATION_SETS = ["raw", "optiver", "temporal", "cross_section", "alpha101"]


def main() -> None:
    args = s4.common_parser(__doc__).parse_args()
    ctx = s4.Context(args)
    out = ctx.report_dir / "batch1"
    out.mkdir(parents=True, exist_ok=True)
    s4.check_environment(ctx)

    features = s4.load_features(ctx)
    features = features.loc[features["date_id"] < ctx.holdout.validation_start].reset_index(drop=True)
    s4.log(f"Development rows (dates < {ctx.holdout.validation_start}): {len(features):,}")
    seed = int(ctx.stage["seed"])
    main = ctx.stage["main_model"]
    other = ctx.stage["comparison_model"]
    grid = list(ctx.stage[f"{main}_tuning_grid"])
    other_grid = list(ctx.stage[f"{other}_tuning_grid"]) if ctx.use_xgb else []
    tuning_set = ctx.stage["tuning_feature_set"]
    total = len(ctx.folds) * (len(grid) + len(ABLATION_SETS) - 1 + len(other_grid))
    step = 0
    started = time.perf_counter()

    def banner(text: str) -> None:
        nonlocal step
        step += 1
        s4.log(f"=== [{step}/{total}] {text}  (elapsed {(time.perf_counter() - started) / 60:.1f} min) ===")

    # 1. constant baselines
    s4.log("Step 1: Zero / Median baselines")
    baseline_rows = []
    for fold in ctx.folds:
        dates = features["date_id"]
        y_train = features.loc[fold.train_mask(dates), TARGET_COLUMN].to_numpy()
        y_eval = features.loc[fold.validation_mask(dates), TARGET_COLUMN].to_numpy()
        y_train, y_eval = y_train[np.isfinite(y_train)], y_eval[np.isfinite(y_eval)]
        median = float(np.median(y_train))
        baseline_rows.append({"fold": fold.name, "eval_rows": len(y_eval), "zero": float(np.abs(y_eval).mean()),
                              "median": float(np.abs(y_eval - median).mean()), "train_median": median})
    baselines = pd.DataFrame(baseline_rows)
    weights = baselines["eval_rows"] / baselines["eval_rows"].sum()
    baselines.loc[len(baselines)] = {"fold": "pooled", "eval_rows": int(baselines["eval_rows"].sum()),
                                     "zero": float((baselines["zero"] * weights).sum()),
                                     "median": float((baselines["median"] * weights).sum()), "train_median": np.nan}
    baselines.to_csv(out / "baselines.csv", index=False)
    s4.log("Baselines:\n" + baselines.round(5).to_string(index=False))

    # 2. tuning grid on the V1 full feature set (main model)
    s4.log(f"Step 2: {main} tuning grid {grid} on feature set '{tuning_set}'")
    for fold in ctx.folds:
        for name in grid:
            banner(f"tune {main} {name} @ {fold.name}")
            s4.execute_run(ctx, features, fold, main, tuning_set, name, seed=seed, importance_dir=out / "importance")
    runs = ctx.registry.table()
    tuning = s4.fold_table(runs, {f"{main} {n}": s4.run_key(main, tuning_set, n, False, seed) for n in grid}, ctx.folds)
    best_name = grid[int(tuning["pooled"].to_numpy().argmin())]
    best = {main: {"params_name": best_name, "params": ctx.params(main, best_name)}}
    s4.log(f"Best {main} params: {best_name}")

    # 3. cumulative ablation with the tuned params
    s4.log(f"Step 3: cumulative feature-group ablation with params '{best_name}'")
    for fold in ctx.folds:
        for set_name in ABLATION_SETS:
            if set_name == tuning_set:
                continue  # identical to the tuning run with the best params
            banner(f"ablation {main} {set_name} @ {fold.name}")
            s4.execute_run(ctx, features, fold, main, set_name, best_name, seed=seed, importance_dir=out / "importance")
    runs = ctx.registry.table()
    ablation = s4.fold_table(runs, {s4.SET_LABELS[n]: s4.run_key(main, n, best_name, False, seed) for n in ABLATION_SETS}, ctx.folds)
    ablation.insert(1, "n_features", [len(s4.feature_set_columns(features)[n]) for n in ABLATION_SETS])
    ablation.to_csv(out / "ablation.csv", index=False)
    s4.log("Ablation:\n" + ablation.round(5).to_string(index=False))

    # 4. model-family comparison on the same V1 features
    if other_grid:
        s4.log(f"Step 4: {other} ({ctx.xgb_device}) grid {other_grid} on feature set '{tuning_set}'")
        for fold in ctx.folds:
            for name in other_grid:
                banner(f"compare {other} {name} @ {fold.name}")
                s4.execute_run(ctx, features, fold, other, tuning_set, name, seed=seed, importance_dir=out / "importance")
        runs = ctx.registry.table()
        other_table = s4.fold_table(runs, {f"{other} {n}": s4.run_key(other, tuning_set, n, False, seed) for n in other_grid}, ctx.folds)
        other_best = other_grid[int(other_table["pooled"].to_numpy().argmin())]
        best[other] = {"params_name": other_best, "params": ctx.params(other, other_best)}
        tuning = pd.concat([tuning, other_table], ignore_index=True)
        seconds = runs.loc[runs["set"] == tuning_set].groupby("model")["seconds"].sum() / 60
        s4.log(f"Training minutes on '{tuning_set}' (all grid runs): {seconds.round(1).to_dict()}")
    else:
        s4.log("Step 4: XGBoost comparison skipped")
    tuning.to_csv(out / "tuning.csv", index=False)
    best["rule"] = "lowest row-weighted pooled MAE over the three dev folds, per model"
    (out / "best_params.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    s4.log("Tuning and model comparison (lower is better):\n" + tuning.round(5).to_string(index=False))

    # 5. synthetic index weights
    s4.log("Step 5: synthetic index weights (training dates only)")
    fit_rows, weight_columns = [], {}
    for name, train_end in [(f.name, f.train_end) for f in ctx.folds] + [("holdout_train", ctx.holdout.train_end)]:
        fit = ctx.index_fit(features, train_end)
        fit_rows.append({"fit": name, "train_end": train_end, "r2": fit.r2, "n_times": fit.n_times,
                         "implied_dispersion_bp": fit.implied_dispersion, "nonzero_weights": int((fit.weights > 0).sum()),
                         "max_weight": float(fit.weights.max())})
        weight_columns[name] = fit.weights
    index_fit = pd.DataFrame(fit_rows)
    index_fit.to_csv(out / "index_fit.csv", index=False)
    weights_table = pd.DataFrame(weight_columns)
    weights_table.index.name = "stock_id"
    weights_table.to_csv(out / "index_weights.csv")
    correlation = weights_table.corr().round(4)
    s4.log("Index fit:\n" + index_fit.round(5).to_string(index=False))
    s4.log("Weight correlation across fits:\n" + correlation.to_string())

    write_summary(out, baselines, tuning, best_name, ablation, index_fit, correlation)
    s4.log(f"Batch 1 finished in {(time.perf_counter() - started) / 60:.1f} min. Results: {out}")


def write_summary(out, baselines, tuning, best_name, ablation, index_fit, correlation) -> None:
    lines = [
        "# Stage 04 · Batch 1 summary", "",
        "Development folds only (holdout dates 400-480 untouched). MAE, lower is better; `pooled` is weighted by rows.", "",
        "## Constant baselines", "", s4.markdown_table(baselines), "",
        "## Tuning on the V1 full feature set (LightGBM main; XGBoost GPU as model comparison)", "",
        s4.markdown_table(tuning), "",
        f"Selected LightGBM params: **{best_name}** (lowest pooled MAE).", "",
        "## Cumulative feature-group ablation (tuned params)", "", s4.markdown_table(ablation), "",
        "## Synthetic index fit", "",
        "`r2`: share of the implied index return explained by the weighted stock returns. "
        "`implied_dispersion_bp`: median cross-stock spread of the implied index return; near 0 means the target "
        "definition is reproduced exactly.", "",
        s4.markdown_table(index_fit), "",
        "Weight correlation across folds:", "", s4.markdown_table(correlation.reset_index(), 4), "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
