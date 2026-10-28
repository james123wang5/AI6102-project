#!/usr/bin/env python3
"""Stage 04 batch 2: new feature groups and index-aware post-processing.

LightGBM main chain only, with the params chosen in batch 1. Cumulative on top of the V1 set (alpha101):
  N1 s_market  + equal-weight market aggregates
  N2 s_index   + synthetic-index features (weights fitted on fold training dates)
  N3 s_stock   + per-stock statistics (fitted on fold training dates)
Then, for every feature set, neutralization (none / equal / index weights)
times a shrink factor alpha is evaluated on the dev folds. The best
(feature set, mode, alpha) by pooled dev MAE is written to `choice.json`.

Outputs: reports/04_xgb_gpu/batch2/. Resubmitting skips finished runs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai6102 import stage04 as s4  # noqa: E402

NEW_SETS = ["s_market", "s_index", "s_stock"]
COMPARED_SETS = ["alpha101"] + NEW_SETS


def main() -> None:
    args = s4.common_parser(__doc__).parse_args()
    ctx = s4.Context(args)
    out = ctx.report_dir / "batch2"
    out.mkdir(parents=True, exist_ok=True)
    s4.check_environment(ctx)
    main = ctx.stage["main_model"]
    params_name = ctx.best_params_name(main)
    seed = int(ctx.stage["seed"])
    s4.log(f"Using params from batch 1: {params_name}")

    features = s4.load_features(ctx)
    features = features.loc[features["date_id"] < ctx.holdout.validation_start].reset_index(drop=True)
    total = len(ctx.folds) * len(COMPARED_SETS)
    step = 0
    started = time.perf_counter()

    preds: dict[str, dict[str, pd.DataFrame]] = {name: {} for name in COMPARED_SETS}
    for fold in ctx.folds:
        for set_name in COMPARED_SETS:
            step += 1
            s4.log(f"=== [{step}/{total}] {set_name} @ {fold.name}  (elapsed {(time.perf_counter() - started) / 60:.1f} min) ===")
            preds[set_name][fold.name] = s4.execute_run(
                ctx, features, fold, main, set_name, params_name, seed=seed, importance_dir=out / "importance"
            )

    runs = ctx.registry.table()
    features_table = s4.fold_table(
        runs, {s4.SET_LABELS[n]: s4.run_key(main, n, params_name, False, seed) for n in COMPARED_SETS}, ctx.folds
    )
    features_table.insert(1, "n_features", [len(s4.feature_set_columns(features)[n]) for n in COMPARED_SETS])
    features_table.to_csv(out / "feature_groups.csv", index=False)
    s4.log("New feature groups:\n" + features_table.round(5).to_string(index=False))

    s4.log("Post-processing grid (neutralization x shrink) on dev predictions")
    grid = pd.concat([s4.postprocess_grid(ctx, features, preds[n], n) for n in COMPARED_SETS], ignore_index=True)
    grid.to_csv(out / "postprocess_grid.csv", index=False)
    best = grid.sort_values("pooled").iloc[0]
    # The headline post-processing table: each mode at alpha=1 and at its best alpha, per feature set.
    headline = []
    for set_name in COMPARED_SETS:
        subset = grid.loc[grid["model"] == set_name]
        for mode in ("none", "equal", "index"):
            mode_rows = subset.loc[subset["mode"] == mode]
            at_one = mode_rows.loc[mode_rows["alpha"] == 1.0].iloc[0]
            tuned = mode_rows.sort_values("pooled").iloc[0]
            headline.append({"feature_set": set_name, "mode": mode, "pooled_alpha_1": at_one["pooled"],
                             "best_alpha": tuned["alpha"], "pooled_best_alpha": tuned["pooled"]})
    headline = pd.DataFrame(headline)
    headline.to_csv(out / "postprocess_summary.csv", index=False)
    s4.log("Post-processing summary:\n" + headline.round(5).to_string(index=False))

    choice = {"feature_set": str(best["model"]), "mode": str(best["mode"]), "alpha": float(best["alpha"]),
              "pooled_dev_mae": float(best["pooled"]), "params_name": params_name,
              "rule": "lowest pooled dev MAE over feature set x neutralization mode x shrink alpha"}
    (out / "choice.json").write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    s4.log(f"Choice for batch 3: {choice}")

    lines = [
        "# Stage 04 · Batch 2 summary", "",
        f"Params from batch 1: `{params_name}`. Development folds only. MAE, lower is better.", "",
        "## New feature groups (cumulative on the V1 set)", "", s4.markdown_table(features_table), "",
        "## Post-processing", "",
        "`equal`: subtract the per-time equal-weight mean prediction; `index`: subtract the index-weighted mean "
        "(weights fitted on each fold's training dates). `best_alpha`: shrink factor chosen on the dev folds.", "",
        s4.markdown_table(headline), "",
        f"Selected for batch 3: feature set **{choice['feature_set']}**, mode **{choice['mode']}**, "
        f"alpha **{choice['alpha']}**, pooled dev MAE **{choice['pooled_dev_mae']:.5f}**.", "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    s4.log(f"Batch 2 finished in {(time.perf_counter() - started) / 60:.1f} min. Results: {out}")


if __name__ == "__main__":
    main()
