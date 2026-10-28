#!/usr/bin/env python3
"""Stage 04 batch 3: final configuration, error analysis, one-off holdout evaluation.

Starting from batch 2's choice (feature set, neutralization mode, alpha):
  1. Recency weighting (half-life in configs/stage04.json) vs none, on dev folds.
  2. Seed averaging with the extra seeds, using the chosen recency setting.
  3. Re-select neutralization mode and alpha for the seed ensemble on dev folds.
  4. Dev comparison table, day-block bootstrap CIs, error breakdowns.
  5. Model-family comparison (report only, never selected): XGBoost on GPU
     with the final feature set, alone and averaged with the LightGBM ensemble.
  6. Final holdout (train 0..399, evaluate 400..480) for Zero, Median, raw,
     V1 (alpha101), the final LightGBM pipeline and the XGBoost comparison.
     Every choice above is frozen before this step; the holdout is not used
     to select anything.

Outputs: reports/04_xgb_gpu/batch3/. Resubmitting skips finished runs.
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

PHASES = [(0, 120, "000-120s"), (130, 290, "130-290s"), (300, 410, "300-410s"), (420, 540, "420-540s")]


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0].copy()
    for other in frames[1:]:
        if not np.array_equal(other[s4.KEY_COLUMNS].to_numpy(), base[s4.KEY_COLUMNS].to_numpy()):
            raise ValueError("Seed runs are not row-aligned; cannot average.")
    base["pred"] = np.mean([f["pred"].to_numpy(dtype="float64") for f in frames], axis=0).astype("float32")
    return base


def main() -> None:
    args = s4.common_parser(__doc__).parse_args()
    ctx = s4.Context(args)
    out = ctx.report_dir / "batch3"
    out.mkdir(parents=True, exist_ok=True)
    s4.check_environment(ctx)
    main = ctx.stage["main_model"]
    choice_path = ctx.report_dir / "batch2" / "choice.json"
    if not choice_path.exists():
        raise FileNotFoundError(f"{choice_path} not found: run batch 2 first.")
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    params_name, best_set = choice["params_name"], choice["feature_set"]
    seed = int(ctx.stage["seed"])
    seeds = [seed] + [int(s) for s in ctx.stage["extra_seeds"]]
    s4.log(f"Batch 2 choice: {choice}")

    full = s4.load_features(ctx)
    dev = full.loc[full["date_id"] < ctx.holdout.validation_start].reset_index(drop=True)
    xgb_name = ctx.best_params_name("xgb") if ctx.use_xgb else None
    total = len(ctx.folds) * (len(seeds) + (1 if ctx.use_xgb else 0)) + 3 + len(seeds) + (1 if ctx.use_xgb else 0)
    step = 0
    started = time.perf_counter()

    def banner(text: str) -> None:
        nonlocal step
        step += 1
        s4.log(f"=== [{step}/{total}] {text}  (elapsed {(time.perf_counter() - started) / 60:.1f} min) ===")

    # 1. recency weighting
    plain = {f.name: s4.execute_run(ctx, dev, f, main, best_set, params_name, False, seed) for f in ctx.folds}
    recent = {}
    for fold in ctx.folds:
        banner(f"recency weighting @ {fold.name}")
        recent[fold.name] = s4.execute_run(ctx, dev, fold, main, best_set, params_name, True, seed)
    plain_mae, recent_mae = s4.pooled_mae(list(plain.values())), s4.pooled_mae(list(recent.values()))
    use_recency = recent_mae < plain_mae
    s4.log(f"Recency: pooled dev MAE {recent_mae:.5f} vs {plain_mae:.5f} without -> use_recency={use_recency}")

    # 2. seed averaging
    by_seed = {seed: recent if use_recency else plain}
    for extra in seeds[1:]:
        by_seed[extra] = {}
        for fold in ctx.folds:
            banner(f"seed {extra} @ {fold.name}")
            by_seed[extra][fold.name] = s4.execute_run(ctx, dev, fold, main, best_set, params_name, use_recency, extra)
    final_dev = {f.name: ensemble([by_seed[s][f.name] for s in seeds]) for f in ctx.folds}
    seed_rows = [{"seed": s, "pooled": s4.pooled_mae(list(by_seed[s].values()))} for s in seeds]
    seed_rows.append({"seed": "ensemble", "pooled": s4.pooled_mae(list(final_dev.values()))})
    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(out / "seeds.csv", index=False)
    s4.log("Seeds:\n" + seed_table.round(5).to_string(index=False))

    # 3. post-processing for the ensemble
    grid = s4.postprocess_grid(ctx, dev, final_dev, "final_ensemble")
    grid.to_csv(out / "postprocess_grid_final.csv", index=False)
    best_pp = grid.sort_values("pooled").iloc[0]
    mode, alpha = str(best_pp["mode"]), float(best_pp["alpha"])
    s4.log(f"Final post-processing: mode={mode}, alpha={alpha}, pooled dev MAE {best_pp['pooled']:.5f}")
    fold_by_name = {f.name: f for f in ctx.folds}
    for name, frame in final_dev.items():
        weights = ctx.index_fit(dev, fold_by_name[name].train_end).weights
        frame["pred_final"] = s4.postprocess(frame, mode, alpha, weights).astype("float32")

    # model-family comparison, reported only
    xgb_dev = {}
    if ctx.use_xgb:
        for fold in ctx.folds:
            banner(f"compare xgb {xgb_name} on {best_set} @ {fold.name}")
            frame = s4.execute_run(ctx, dev, fold, "xgb", best_set, xgb_name, use_recency, seed).copy()
            if not np.array_equal(frame[s4.KEY_COLUMNS].to_numpy(), final_dev[fold.name][s4.KEY_COLUMNS].to_numpy()):
                raise ValueError("XGBoost and LightGBM runs are not row-aligned.")
            weights = ctx.index_fit(dev, fold.train_end).weights
            frame["pred_final"] = s4.postprocess(frame, mode, alpha, weights).astype("float32")
            blend = frame.copy()
            blend["pred"] = (0.5 * frame["pred"].to_numpy(dtype="float64")
                             + 0.5 * final_dev[fold.name]["pred"].to_numpy(dtype="float64"))
            frame["pred_blend"] = s4.postprocess(blend, mode, alpha, weights).astype("float32")
            xgb_dev[fold.name] = frame

    # 4. dev comparison, bootstrap, error analysis
    raw = {f.name: ctx.registry.predictions(s4.run_key(main, "raw", params_name, False, seed), f.name) for f in ctx.folds}
    v1 = {f.name: ctx.registry.predictions(s4.run_key(main, "alpha101", params_name, False, seed), f.name) for f in ctx.folds}
    dev_rows = []
    zero_by_fold = {n: float(np.abs(f[TARGET_COLUMN]).mean()) for n, f in final_dev.items()}
    for label, frames, column in [
        ("Zero", None, None), ("E1 raw fields (tuned LightGBM)", raw, "pred"), ("E5 V1 all features", v1, "pred"),
        (f"Best set {best_set} (1 seed)", plain, "pred"),
        (f"+recency={use_recency}, {len(seeds)}-seed avg", final_dev, "pred"),
        (f"Final: + {mode} neutralization, alpha={alpha}", final_dev, "pred_final"),
    ] + ([
        (f"[compare] XGBoost GPU {xgb_name}, {best_set}, 1 seed", xgb_dev, "pred"),
        ("[compare] XGBoost + same post-processing", xgb_dev, "pred_final"),
        ("[compare] 50/50 LightGBM+XGBoost + post-processing", xgb_dev, "pred_blend"),
    ] if xgb_dev else []):
        row = {"model": label}
        all_frames = list(final_dev.values()) if frames is None else list(frames.values())
        for f in ctx.folds:
            row[f.name] = zero_by_fold[f.name] if frames is None else s4.mae(frames[f.name], column)
        row["pooled"] = (float(np.abs(pd.concat(all_frames)[TARGET_COLUMN]).mean()) if frames is None
                         else s4.pooled_mae(all_frames, column))
        dev_rows.append(row)
    dev_table = pd.DataFrame(dev_rows)
    dev_table["vs_zero_pct"] = 100 * (dev_table["pooled"] / dev_table.loc[0, "pooled"] - 1)
    dev_table.to_csv(out / "dev_comparison.csv", index=False)
    s4.log("Dev comparison:\n" + dev_table.round(5).to_string(index=False))

    final_all = pd.concat(final_dev.values(), ignore_index=True)
    reps, boot_seed = int(ctx.stage["bootstrap_reps"]), seed
    boot_rows = []
    for label, frames in [("final - E5 V1", v1), ("final - E1 raw", raw)]:
        other = pd.concat(frames.values(), ignore_index=True)
        merged = final_all.merge(other[s4.KEY_COLUMNS + ["pred"]].rename(columns={"pred": "pred_other"}), on=s4.KEY_COLUMNS)
        result = s4.day_block_bootstrap(merged, merged, "pred_final", "pred_other", reps, boot_seed)
        boot_rows.append({"comparison": label, **result})
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(out / "bootstrap.csv", index=False)
    s4.log("Day-block bootstrap of MAE difference (negative = final better):\n" + bootstrap.round(5).to_string(index=False))

    v1_all = pd.concat(v1.values(), ignore_index=True)
    analysis = final_all.merge(v1_all[s4.KEY_COLUMNS + ["pred"]].rename(columns={"pred": "pred_v1"}), on=s4.KEY_COLUMNS)
    analysis["err_final"] = (analysis[TARGET_COLUMN] - analysis["pred_final"]).abs()
    analysis["err_v1"] = (analysis[TARGET_COLUMN] - analysis["pred_v1"]).abs()
    analysis["err_zero"] = analysis[TARGET_COLUMN].abs()
    phase = pd.Series("", index=analysis.index)
    for low, high, name in PHASES:
        phase[analysis["seconds_in_bucket"].between(low, high)] = name
    analysis["phase"] = phase
    liquidity = ctx.stock_stats(dev, ctx.folds[0].train_end)["matched_size"]
    quintile = pd.qcut(liquidity.rank(method="first"), 5, labels=["Q1 least liquid", "Q2", "Q3", "Q4", "Q5 most liquid"])
    analysis["liquidity"] = quintile.to_numpy()[analysis["stock_id"].to_numpy()]
    error_cols = ["err_zero", "err_v1", "err_final"]
    by_phase = analysis.groupby("phase")[error_cols].mean()
    by_liquidity = analysis.groupby("liquidity", observed=True)[error_cols].mean()
    by_day = analysis.groupby("date_id")[error_cols].mean()
    for table in (by_phase, by_liquidity):
        table["final_vs_v1_pct"] = 100 * (table["err_final"] / table["err_v1"] - 1)
    by_phase.to_csv(out / "error_by_phase.csv")
    by_liquidity.to_csv(out / "error_by_liquidity.csv")
    by_day.to_csv(out / "error_by_day.csv")
    s4.log("Error by closing phase:\n" + by_phase.round(5).to_string())
    s4.log("Error by stock liquidity (median matched size on fold_1 training dates):\n" + by_liquidity.round(5).to_string())

    # 5. one-off holdout evaluation with everything frozen
    frozen = {"model": main, "params_name": params_name, "feature_set": best_set, "recency": use_recency, "seeds": seeds,
              "mode": mode, "alpha": alpha}
    (out / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    s4.log(f"Frozen configuration for holdout: {frozen}")
    holdout = ctx.holdout
    dates = full["date_id"]
    y_train = full.loc[holdout.train_mask(dates), TARGET_COLUMN].to_numpy()
    median = float(np.median(y_train[np.isfinite(y_train)]))
    banner("holdout raw fields")
    h_raw = s4.execute_run(ctx, full, holdout, main, "raw", params_name, False, seed)
    banner("holdout V1 all features")
    h_v1 = s4.execute_run(ctx, full, holdout, main, "alpha101", params_name, False, seed)
    h_seeds = []
    for s in seeds:
        banner(f"holdout final, seed {s}")
        h_seeds.append(s4.execute_run(ctx, full, holdout, main, best_set, params_name, use_recency, s))
    h_weights = ctx.index_fit(full, holdout.train_end).weights
    h_xgb = None
    if ctx.use_xgb:
        banner("holdout xgb comparison")
        h_xgb = s4.execute_run(ctx, full, holdout, "xgb", best_set, xgb_name, use_recency, seed).copy()
    banner("holdout scoring")
    h_final = ensemble(h_seeds)
    h_final["pred_final"] = s4.postprocess(h_final, mode, alpha, h_weights)
    y = h_final[TARGET_COLUMN].to_numpy()
    holdout_rows = [
        {"model": "Zero", "mae": float(np.abs(y).mean())},
        {"model": "Median (train 0-399)", "mae": float(np.abs(y - median).mean())},
        {"model": "E1 raw fields (tuned LightGBM)", "mae": s4.mae(h_raw)},
        {"model": "E5 V1 all features", "mae": s4.mae(h_v1)},
        {"model": f"Final pipeline ({best_set}, rec={use_recency}, {len(seeds)} seeds, {mode}, alpha={alpha})",
         "mae": s4.mae(h_final, "pred_final")},
    ]
    if h_xgb is not None:
        h_xgb["pred_final"] = s4.postprocess(h_xgb, mode, alpha, h_weights)
        blend = h_xgb.copy()
        blend["pred"] = 0.5 * h_xgb["pred"].to_numpy(dtype="float64") + 0.5 * h_final["pred"].to_numpy(dtype="float64")
        h_xgb["pred_blend"] = s4.postprocess(blend, mode, alpha, h_weights)
        holdout_rows += [
            {"model": f"[compare] XGBoost GPU {xgb_name} + post-processing", "mae": s4.mae(h_xgb, "pred_final")},
            {"model": "[compare] 50/50 LightGBM+XGBoost + post-processing", "mae": s4.mae(h_xgb, "pred_blend")},
        ]
    holdout_table = pd.DataFrame(holdout_rows)
    holdout_table["vs_zero_pct"] = 100 * (holdout_table["mae"] / holdout_table.loc[0, "mae"] - 1)
    holdout_table["rows"] = len(y)
    holdout_table.to_csv(out / "holdout.csv", index=False)
    merged = h_final.merge(h_v1[s4.KEY_COLUMNS + ["pred"]].rename(columns={"pred": "pred_v1"}), on=s4.KEY_COLUMNS)
    holdout_boot = s4.day_block_bootstrap(merged, merged, "pred_final", "pred_v1", reps, boot_seed)
    pd.DataFrame([{"comparison": "holdout final - E5 V1", **holdout_boot}]).to_csv(out / "holdout_bootstrap.csv", index=False)
    s4.log("HOLDOUT (dates 400-480, evaluated once):\n" + holdout_table.round(5).to_string(index=False))
    s4.log(f"Holdout bootstrap final - V1: {holdout_boot}")

    lines = [
        "# Stage 04 · Batch 3 summary", "",
        "## Development folds", "",
        f"Recency weighting: pooled dev MAE {recent_mae:.5f} with vs {plain_mae:.5f} without -> used: **{use_recency}**.", "",
        "Seed averaging:", "", s4.markdown_table(seed_table), "",
        s4.markdown_table(dev_table), "",
        "Day-block bootstrap of the MAE difference (95% CI; negative = final better):", "",
        s4.markdown_table(bootstrap), "",
        "## Error breakdown (dev folds pooled)", "", s4.markdown_table(by_phase.reset_index()), "",
        s4.markdown_table(by_liquidity.reset_index()), "",
        "## Frozen configuration", "", "```json", json.dumps(frozen, indent=2), "```", "",
        "## Final holdout (dates 400-480, evaluated once after freezing)", "", s4.markdown_table(holdout_table), "",
        f"Bootstrap final - V1 on holdout: diff {holdout_boot['diff']:.5f}, "
        f"95% CI [{holdout_boot['ci_low']:.5f}, {holdout_boot['ci_high']:.5f}].", "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    s4.log(f"Batch 3 finished in {(time.perf_counter() - started) / 60:.1f} min. Results: {out}")


if __name__ == "__main__":
    main()
