#!/usr/bin/env python3
"""Stage 04 batch 5: neural models (GRU over time, Transformer over stocks) and blending with LightGBM.

Same folds, features (batch 4 set without stock_id), recency weighting and
protocol as the trees; settings in configs/stage05.json.
  1. For every dev fold and the holdout: GRU and Transformer, 3 seeds each
     (inner early stopping on the last 20 training days, then refit).
  2. Blend weights for LightGBM (batch 4 final ensemble) / GRU / Transformer
     and the shrink alpha are chosen on the pooled dev folds, after index
     neutralization of each model.
  3. Holdout (dates 400-480) with everything frozen. Third use of the holdout
     (after batches 3 and 4); nothing is selected on it.

Outputs: reports/04_xgb_gpu/batch5/. Resubmitting skips finished runs.
"""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai6102 import deep  # noqa: E402
from ai6102 import stage04 as s4  # noqa: E402
from ai6102.baselines import TARGET_COLUMN  # noqa: E402
from ai6102.history_features import add_history_features  # noqa: E402
from ai6102.index_model import neutralize  # noqa: E402

MIN_FREE_GB = 5.0
LABELS = {"lgb": "LightGBM (batch 4 final)", "gru": "GRU (time axis)", "tfm": "Transformer (stock axis)"}


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0].copy()
    for other in frames[1:]:
        if not np.array_equal(other[s4.KEY_COLUMNS].to_numpy(), base[s4.KEY_COLUMNS].to_numpy()):
            raise ValueError("Runs are not row-aligned; cannot average.")
    base["pred"] = np.mean([f["pred"].to_numpy(dtype="float64") for f in frames], axis=0).astype("float32")
    return base


def day_slice(grid: deep.Grid, mask: np.ndarray) -> deep.Grid:
    return deep.Grid(grid.days[mask], grid.x[mask], grid.y[mask], grid.present[mask], grid.labelled[mask], grid.index)


def main() -> None:
    parser = s4.common_parser(__doc__)
    parser.add_argument("--dl-config", type=Path, default=s4.PROJECT_ROOT / "configs" / "stage05.json")
    parser.add_argument("--dl-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--smoke-epochs", type=int, default=None, help="smoke test: cap epochs")
    args = parser.parse_args()
    ctx = s4.Context(args)
    ctx.use_xgb = False
    cfg = json.loads(args.dl_config.read_text(encoding="utf-8"))
    out = ctx.report_dir / "batch5"
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(ctx.artifact_dir).free / 1e9
    s4.log(f"Free disk space at {ctx.artifact_dir}: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"Only {free_gb:.1f} GB free (< {MIN_FREE_GB} GB). Run `conda clean -a -y; pip cache purge` first.")

    import torch

    if args.dl_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")
    s4.log(f"torch {torch.__version__}, device {args.dl_device}"
           + (f" ({torch.cuda.get_device_name(0)})" if args.dl_device == "cuda" else ""))
    s4.check_environment(ctx)

    frozen_path = ctx.report_dir / "batch4" / "frozen_config.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"{frozen_path} not found: run batch 4 first.")
    tree = json.loads(frozen_path.read_text(encoding="utf-8"))
    s4.log(f"Batch 4 frozen tree configuration: {tree}")
    s4.log(f"DL configuration: {cfg}")

    full = s4.load_features(ctx)
    s4.log("Adding cross-day target history features ...")
    full = add_history_features(full)
    dev = full.loc[full["date_id"] < ctx.holdout.validation_start].reset_index(drop=True)
    set_name, seeds, kinds = cfg["feature_set"], [int(s) for s in cfg["seeds"]], list(cfg["models"])
    recency = bool(cfg["recency"])
    max_epochs = int(args.smoke_epochs or cfg["max_epochs"])
    params_name = cfg["version"]
    started = time.perf_counter()

    def key(kind: str, seed: int) -> str:
        return s4.run_key(kind, set_name, params_name, recency, seed)

    # 1. neural runs per fold
    for fold in ctx.folds + [ctx.holdout]:
        pending = [(k, s) for k in kinds for s in seeds if not ctx.registry.done(key(k, s), fold.name)]
        if not pending:
            s4.log(f"{fold.name}: all neural runs already done")
            continue
        source = full if fold.name == "holdout" else dev
        s4.log(f"=== {fold.name}: building tensors  (elapsed {(time.perf_counter() - started) / 60:.1f} min) ===")
        train, evaluation, columns = s4.fold_frames(ctx, source, fold, set_name)
        scaler = deep.Scaler.fit(train[columns].to_numpy(), float(cfg["feature_clip"]), seed=0)
        train_grid = deep.to_grid(train, columns, scaler)
        eval_grid = deep.to_grid(evaluation, columns, scaler)
        eval_keys = evaluation.loc[:, s4.KEY_COLUMNS + [TARGET_COLUMN]].copy()
        eval_keys["stock_id"] = eval_keys["stock_id"].astype("int16")
        train_rows, n_features = len(train), len(columns)
        del train, evaluation
        s4.log(f"  train tensor {train_grid.x.shape}, eval tensor {eval_grid.x.shape}, "
               f"{train_grid.x.nbytes / 1e9:.2f} GB; signed-log on {int(scaler.log_mask.sum())} of {n_features} features")
        inner_start = fold.train_end - int(ctx.config["inner_early_stopping_days"]) + 1
        fit_grid = day_slice(train_grid, train_grid.days < inner_start)
        stop_grid = day_slice(train_grid, train_grid.days >= inner_start)
        half_life = float(ctx.stage["recency_half_life_days"])

        def day_weights(grid: deep.Grid) -> np.ndarray | None:
            return np.power(0.5, (fold.train_end - grid.days) / half_life) if recency else None

        weights = ctx.index_fit(source, fold.train_end).weights
        for kind, seed in pending:
            run_started = time.perf_counter()
            tag = f"{kind}/s{seed}/{fold.name}"
            s4.log(f"=== {tag}: step 1/2 early stopping (fit days < {inner_start}, inner days {inner_start}..{fold.train_end})"
                   f"  (elapsed {(time.perf_counter() - started) / 60:.1f} min) ===")
            trainer = deep.Trainer(kind, cfg, args.dl_device, weights, seed, max_epochs)
            _, best = trainer.fit(fit_grid, day_weights(fit_grid), stop_grid, max_epochs, tag)
            s4.log(f"  {tag}: best epoch {best}; step 2/2 refit on all training days")
            model, _ = trainer.fit(train_grid, day_weights(train_grid), None, max(best, 1), tag + "/refit")
            prediction = trainer.predict(model, eval_grid)
            predictions = eval_keys.copy()
            predictions["pred"] = prediction[eval_grid.index].astype("float32")
            mae = s4.mae(predictions)
            row = {
                "key": key(kind, seed), "model": kind, "set": set_name, "params": params_name, "recency": int(recency),
                "seed": seed, "fold": fold.name, "n_features": n_features, "best_iteration": best,
                "max_rounds": max_epochs, "mae": mae, "train_rows": train_rows, "eval_rows": len(predictions),
                "seconds": round(time.perf_counter() - run_started, 1), "top_features": "",
            }
            ctx.registry.record(row, predictions)
            s4.log(f"  DONE {key(kind, seed)} @ {fold.name}: MAE {mae:.5f}, best epoch {best}, {row['seconds']:.0f}s")
            del model, trainer
            if args.dl_device == "cuda":
                torch.cuda.empty_cache()
        del train_grid, eval_grid, fit_grid, stop_grid

    # 2. per-fold ensembles; neutralize each model once (neutralization is linear, so blends reuse it)
    def model_frames(fold, source) -> dict[str, pd.DataFrame]:
        frames = {"lgb": ensemble([s4.execute_run(ctx, source, fold, tree["model"], tree["feature_set"],
                                                  tree["params_name"], tree["recency"], s) for s in tree["seeds"]])}
        for kind in kinds:
            frames[kind] = ensemble([ctx.registry.predictions(key(kind, s), fold.name) for s in seeds])
            if not np.array_equal(frames[kind][s4.KEY_COLUMNS].to_numpy(), frames["lgb"][s4.KEY_COLUMNS].to_numpy()):
                raise ValueError(f"{kind} and LightGBM predictions are not row-aligned on {fold.name}")
        w = ctx.index_fit(source, fold.train_end).weights
        base = frames["lgb"].loc[:, s4.KEY_COLUMNS + [TARGET_COLUMN]].copy()
        for name, frame in frames.items():
            base[name] = frame["pred"].to_numpy(dtype="float64")
            base[f"{name}_nz"] = neutralize(base, base[name].to_numpy(), w)
        return base

    members = ["lgb"] + kinds
    dev_frames = {f.name: model_frames(f, dev) for f in ctx.folds}
    dev_all = pd.concat(dev_frames.values(), ignore_index=True)
    y_dev = dev_all[TARGET_COLUMN].to_numpy()
    nz_dev = {m: dev_all[f"{m}_nz"].to_numpy() for m in members}
    alphas = [float(a) for a in ctx.stage["shrink_alphas"]]

    def best_alpha(pred_nz: np.ndarray) -> tuple[float, float]:
        scores = [(float(np.abs(y_dev - a * pred_nz).mean()), a) for a in alphas]
        return min(scores)

    step = float(cfg["blend_step"])
    n = round(1 / step)
    blend_rows = []
    for combo in itertools.product(range(n + 1), repeat=len(members) - 1):
        if sum(combo) > n:
            continue
        w = dict(zip(members[1:], [c * step for c in combo]))
        w["lgb"] = 1.0 - sum(w.values())
        pred = sum(w[m] * nz_dev[m] for m in members)
        score, alpha = best_alpha(pred)
        blend_rows.append({**{f"w_{m}": round(w[m], 4) for m in members}, "alpha": alpha, "pooled": score})
    blend_grid = pd.DataFrame(blend_rows).sort_values("pooled").reset_index(drop=True)
    blend_grid.to_csv(out / "blend_grid.csv", index=False)
    best = blend_grid.iloc[0]
    blend_w = {m: float(best[f"w_{m}"]) for m in members}
    blend_alpha = float(best["alpha"])
    s4.log(f"Best dev blend: weights {blend_w}, alpha {blend_alpha}, pooled {best['pooled']:.5f}")

    # each model alone: LightGBM keeps its batch-4 alpha; neural models get their own dev-chosen alpha
    solo_alpha = {"lgb": float(tree["alpha"])}
    for kind in kinds:
        solo_alpha[kind] = best_alpha(nz_dev[kind])[1]
    if tree["mode"] != "index":
        raise ValueError("Batch 4 tree post-processing is not index neutralization; blending assumes it is.")

    def score_rows(frames: dict[str, pd.DataFrame], names: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        rows, columns = [], {}
        allf = pd.concat([frames[n] for n in names], ignore_index=True)
        y = allf[TARGET_COLUMN].to_numpy()
        candidates = [("Zero", np.zeros(len(allf)))]
        for m in members:
            label = f"{LABELS[m]}, {len(seeds) if m != 'lgb' else len(tree['seeds'])}-seed avg, index, alpha={solo_alpha[m]}"
            candidates.append((label, solo_alpha[m] * allf[f"{m}_nz"].to_numpy()))
        blend = blend_alpha * sum(blend_w[m] * allf[f"{m}_nz"].to_numpy() for m in members)
        weights_text = ", ".join(f"{m} {blend_w[m]:.2f}" for m in members)
        candidates.append((f"Blend ({weights_text}), index, alpha={blend_alpha}", blend))
        sizes = np.cumsum([0] + [len(frames[n]) for n in names])
        for label, pred in candidates:
            err = np.abs(y - pred)
            row = {"model": label}
            if len(names) > 1:
                for i, name in enumerate(names):
                    row[name] = float(err[sizes[i]:sizes[i + 1]].mean())
            row["mae"] = float(err.mean())
            rows.append(row)
            columns[label] = pred
        table = pd.DataFrame(rows)
        table["vs_zero_pct"] = 100 * (table["mae"] / table.loc[0, "mae"] - 1)
        return table, {"frame": allf, **columns}

    dev_table, dev_preds = score_rows(dev_frames, [f.name for f in ctx.folds])
    dev_table.to_csv(out / "dev_comparison.csv", index=False)
    s4.log("Dev comparison (pooled = mae):\n" + dev_table.round(5).to_string(index=False))
    reps, seed = int(ctx.stage["bootstrap_reps"]), int(ctx.stage["seed"])

    def bootstrap(preds: dict) -> dict:
        frame = preds["frame"].copy()
        labels = list(preds)[1:]
        frame["pred_blend"], frame["pred_lgb"] = preds[labels[-1]], preds[labels[1]]
        return s4.day_block_bootstrap(frame, frame, "pred_blend", "pred_lgb", reps, seed)

    dev_boot = bootstrap(dev_preds)
    s4.log(f"Dev bootstrap blend - LightGBM: {dev_boot}")

    frozen5 = {"tree": tree, "dl_config": cfg, "blend_weights": blend_w, "blend_alpha": blend_alpha,
               "solo_alpha": solo_alpha, "mode": "index"}
    (out / "frozen_config.json").write_text(json.dumps(frozen5, indent=2) + "\n", encoding="utf-8")

    # 3. holdout, third use, everything frozen
    holdout_frames = {"holdout": model_frames(ctx.holdout, full)}
    holdout_table, holdout_preds = score_rows(holdout_frames, ["holdout"])
    holdout_table["rows"] = len(holdout_preds["frame"])
    holdout_table.to_csv(out / "holdout.csv", index=False)
    holdout_boot = bootstrap(holdout_preds)
    s4.log("HOLDOUT (dates 400-480, third use):\n" + holdout_table.round(5).to_string(index=False))
    s4.log(f"Holdout bootstrap blend - LightGBM: {holdout_boot}")

    runs = ctx.registry.table()
    seed_table = runs.loc[runs["model"].isin(kinds) & (runs["set"] == set_name),
                          ["model", "seed", "fold", "mae", "best_iteration", "seconds"]].drop_duplicates(
        ["model", "seed", "fold"], keep="last")
    seed_table.to_csv(out / "neural_runs.csv", index=False)
    lines = [
        "# Stage 04 · Batch 5 summary (neural models and blending)", "",
        f"Features: `{set_name}`; recency weighting: {recency}; seeds: {seeds}.", "",
        "```json", json.dumps(cfg, indent=2), "```", "",
        "## Single runs", "", s4.markdown_table(seed_table), "",
        "## Dev folds (blend weights and alphas chosen here)", "", s4.markdown_table(dev_table), "",
        f"Day-block bootstrap blend - LightGBM on dev: diff {dev_boot['diff']:.5f}, "
        f"95% CI [{dev_boot['ci_low']:.5f}, {dev_boot['ci_high']:.5f}].", "",
        "Top blends:", "", s4.markdown_table(blend_grid.head(10)), "",
        "## Holdout (dates 400-480; third use, after batches 3 and 4; nothing selected on it)", "",
        s4.markdown_table(holdout_table), "",
        f"Day-block bootstrap blend - LightGBM on holdout: diff {holdout_boot['diff']:.5f}, "
        f"95% CI [{holdout_boot['ci_low']:.5f}, {holdout_boot['ci_high']:.5f}].", "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    s4.log(f"Batch 5 finished in {(time.perf_counter() - started) / 60:.1f} min. Results: {out}")


if __name__ == "__main__":
    main()
