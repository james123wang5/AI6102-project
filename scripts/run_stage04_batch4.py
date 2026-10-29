#!/usr/bin/env python3
"""Stage 04 batch 4: target-history features, intraday relative path, stock_id treatment.

Starts from batch 3's frozen configuration (LightGBM, s_stock, recency, index
neutralization) and tests three additions on the dev folds, one at a time:
  A. h_hist  = s_stock + previous-day target statistics per (stock, second)
  B. h_intra = s_stock + rolling realised stock-minus-index returns within the day
     (A+B together as h_both only if both pass on their own)
  C. stock_id as categorical (default) vs plain integer (@num) vs removed (@drop)
Keep rule for every step: pooled dev MAE improves AND at least 2 of 3 folds improve.
Then seed averaging, a fresh post-processing grid and a second holdout
evaluation (dates 400-480). The holdout was already used once in batch 3; this
second use is reported as such and nothing is selected on it.

LightGBM only (no XGBoost). Outputs: reports/04_xgb_gpu/batch4/. Resubmitting skips finished runs.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai6102 import stage04 as s4  # noqa: E402
from ai6102.baselines import TARGET_COLUMN  # noqa: E402
from ai6102.history_features import add_history_features  # noqa: E402

MIN_FREE_GB = 5.0


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
    ctx.use_xgb = False
    out = ctx.report_dir / "batch4"
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(ctx.artifact_dir).free / 1e9
    s4.log(f"Free disk space at {ctx.artifact_dir}: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        raise SystemExit(f"Only {free_gb:.1f} GB free (< {MIN_FREE_GB} GB). Run `conda clean -a -y; pip cache purge` first.")
    s4.check_environment(ctx)

    frozen_path = ctx.report_dir / "batch3" / "frozen_config.json"
    if not frozen_path.exists():
        raise FileNotFoundError(f"{frozen_path} not found: run batch 3 first.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    main_model, params_name, base_set = frozen["model"], frozen["params_name"], frozen["feature_set"]
    recency, seeds = bool(frozen["recency"]), [int(s) for s in frozen["seeds"]]
    seed = seeds[0]
    s4.log(f"Batch 3 frozen configuration: {frozen}")

    full = s4.load_features(ctx)
    s4.log("Adding cross-day target history features ...")
    full = add_history_features(full)
    dev = full.loc[full["date_id"] < ctx.holdout.validation_start].reset_index(drop=True)
    started = time.perf_counter()
    imp_dir = out / "importance"
    fold_names = [f.name for f in ctx.folds]

    def runs(set_name: str, run_seed: int = seed, label: str = "") -> dict[str, pd.DataFrame]:
        result = {}
        for fold in ctx.folds:
            s4.log(f"=== {label or set_name} seed {run_seed} @ {fold.name}  "
                   f"(elapsed {(time.perf_counter() - started) / 60:.1f} min) ===")
            result[fold.name] = s4.execute_run(ctx, dev, fold, main_model, set_name, params_name, recency, run_seed, imp_dir)
        return result

    def scores(preds: dict[str, pd.DataFrame], column: str = "pred") -> dict[str, float]:
        row = {name: s4.mae(preds[name], column) for name in fold_names}
        row["pooled"] = s4.pooled_mae([preds[n] for n in fold_names], column)
        return row

    step_rows = []

    def compare(step: str, reference_name: str, reference: dict, candidates: dict[str, dict]) -> str:
        """Apply the keep rule; return the winning set name (the reference if nothing passes)."""
        ref = scores(reference)
        step_rows.append({"step": step, "set": reference_name, "role": "reference", **ref,
                          "delta_pooled": 0.0, "folds_better": "", "kept": ""})
        winner, winner_pooled = reference_name, ref["pooled"]
        for name, preds in candidates.items():
            sc = scores(preds)
            better = sum(sc[f] < ref[f] for f in fold_names)
            passed = sc["pooled"] < ref["pooled"] and better >= 2
            step_rows.append({"step": step, "set": name, "role": "candidate", **sc,
                              "delta_pooled": sc["pooled"] - ref["pooled"], "folds_better": f"{better}/3", "kept": passed})
            s4.log(f"{step}: {name} pooled {sc['pooled']:.5f} vs {reference_name} {ref['pooled']:.5f} "
                   f"({sc['pooled'] - ref['pooled']:+.5f}), folds better {better}/3 -> {'KEEP' if passed else 'reject'}")
            if passed and sc["pooled"] < winner_pooled:
                winner, winner_pooled = name, sc["pooled"]
        pd.DataFrame(step_rows).to_csv(out / "steps.csv", index=False)
        return winner

    # A and B, each on top of the batch-3 set
    base = runs(base_set, label="reference (batch 3 set)")
    cand = {"h_hist": runs("h_hist", label="A history"), "h_intra": runs("h_intra", label="B intraday path")}
    s4.log("Dev MAE so far:\n" + pd.DataFrame({k: scores(v) for k, v in {base_set: base, **cand}.items()}).T.round(5).to_string())
    winner = compare("A/B", base_set, base, cand)
    kept = [row["set"] for row in step_rows if row["step"] == "A/B" and row["kept"] is True]
    if set(kept) == {"h_hist", "h_intra"}:
        both = runs("h_both", label="A+B")
        winner = compare("A+B", winner, {base_set: base, **cand}[winner], {"h_both": both})
        cand["h_both"] = both
    chosen_preds = {base_set: base, **cand}[winner]
    s4.log(f"Feature set after A/B: {winner}")

    # C: stock_id treatment on the chosen set
    c_cand = {f"{winner}@{mode}": runs(f"{winner}@{mode}", label=f"C stock_id {mode}") for mode in ("num", "drop")}
    final_set = compare("C stock_id", winner, chosen_preds, c_cand)
    s4.log(f"Final feature set: {final_set}")
    if final_set == base_set:
        s4.log("Nothing beat the batch 3 configuration; skipping seeds and holdout (batch 3 results stand).")
    pd.DataFrame(step_rows).to_csv(out / "steps.csv", index=False)

    # seeds, post-processing, dev comparison with batch 3 final
    fold_by_name = {f.name: f for f in ctx.folds}
    by_seed = {s: runs(final_set, s, label="final seeds") for s in seeds} if final_set != base_set else {}
    old_by_seed = {s: base if s == seed else runs(base_set, s, label="batch 3 seeds") for s in seeds}
    old_dev = {n: ensemble([old_by_seed[s][n] for s in seeds]) for n in fold_names}
    for name, frame in old_dev.items():
        frame["pred_final"] = s4.postprocess(frame, frozen["mode"], frozen["alpha"],
                                             ctx.index_fit(dev, fold_by_name[name].train_end).weights)
    lines = ["# Stage 04 · Batch 4 summary", "",
             f"Reference: batch 3 frozen configuration `{json.dumps(frozen)}`.",
             "Keep rule: pooled dev MAE improves and at least 2 of 3 folds improve.", "",
             "## Step-by-step dev results (single seed, before post-processing)", "",
             s4.markdown_table(pd.DataFrame(step_rows)), "", f"Final feature set: **{final_set}**", ""]
    if by_seed:
        new_dev = {n: ensemble([by_seed[s][n] for s in seeds]) for n in fold_names}
        grid = s4.postprocess_grid(ctx, dev, new_dev, "batch4_ensemble")
        grid.to_csv(out / "postprocess_grid.csv", index=False)
        best_pp = grid.sort_values("pooled").iloc[0]
        mode, alpha = str(best_pp["mode"]), float(best_pp["alpha"])
        s4.log(f"Post-processing: mode={mode}, alpha={alpha}, pooled dev MAE {best_pp['pooled']:.5f}")
        for name, frame in new_dev.items():
            frame["pred_final"] = s4.postprocess(frame, mode, alpha, ctx.index_fit(dev, fold_by_name[name].train_end).weights)
        zero = {n: float(np.abs(old_dev[n][TARGET_COLUMN]).mean()) for n in fold_names}
        zero["pooled"] = float(np.abs(pd.concat(old_dev.values())[TARGET_COLUMN]).mean())
        seed_rows = [{"seed": s, **scores(by_seed[s])} for s in seeds] + [{"seed": "ensemble", **scores(new_dev)}]
        dev_table = pd.DataFrame([
            {"model": "Zero", **zero},
            {"model": f"Batch 3 final ({base_set}, {len(seeds)} seeds, {frozen['mode']}, alpha={frozen['alpha']})",
             **scores(old_dev, "pred_final")},
            {"model": f"Batch 4 {final_set}, {len(seeds)}-seed avg (no post-processing)", **scores(new_dev)},
            {"model": f"Batch 4 final ({final_set}, {mode}, alpha={alpha})", **scores(new_dev, "pred_final")},
        ])
        dev_table["vs_zero_pct"] = 100 * (dev_table["pooled"] / dev_table.loc[0, "pooled"] - 1)
        dev_table.to_csv(out / "dev_comparison.csv", index=False)
        s4.log("Dev comparison:\n" + dev_table.round(5).to_string(index=False))
        reps = int(ctx.stage["bootstrap_reps"])
        new_all = pd.concat(new_dev.values(), ignore_index=True)
        old_all = pd.concat(old_dev.values(), ignore_index=True)
        merged = new_all.merge(old_all[s4.KEY_COLUMNS + ["pred_final"]].rename(columns={"pred_final": "pred_old"}),
                               on=s4.KEY_COLUMNS)
        dev_boot = s4.day_block_bootstrap(merged, merged, "pred_final", "pred_old", reps, seed)
        s4.log(f"Dev bootstrap batch4 - batch3: {dev_boot}")

        # second holdout use, configuration frozen above
        frozen4 = {**frozen, "feature_set": final_set, "mode": mode, "alpha": alpha}
        (out / "frozen_config.json").write_text(json.dumps(frozen4, indent=2) + "\n", encoding="utf-8")
        s4.log(f"Frozen configuration for holdout: {frozen4}")
        h = ctx.holdout
        h_new = ensemble([s4.execute_run(ctx, full, h, main_model, final_set, params_name, recency, s, imp_dir) for s in seeds])
        h_old = ensemble([s4.execute_run(ctx, full, h, main_model, base_set, params_name, recency, s) for s in seeds])
        h_weights = ctx.index_fit(full, h.train_end).weights
        h_new["pred_final"] = s4.postprocess(h_new, mode, alpha, h_weights)
        h_old["pred_final"] = s4.postprocess(h_old, frozen["mode"], frozen["alpha"], h_weights)
        y = h_new[TARGET_COLUMN].to_numpy()
        holdout_table = pd.DataFrame([
            {"model": "Zero", "mae": float(np.abs(y).mean())},
            {"model": "Batch 3 final", "mae": s4.mae(h_old, "pred_final")},
            {"model": f"Batch 4 final ({final_set}, {mode}, alpha={alpha})", "mae": s4.mae(h_new, "pred_final")},
        ])
        holdout_table["vs_zero_pct"] = 100 * (holdout_table["mae"] / holdout_table.loc[0, "mae"] - 1)
        holdout_table["rows"] = len(y)
        holdout_table.to_csv(out / "holdout.csv", index=False)
        merged = h_new.merge(h_old[s4.KEY_COLUMNS + ["pred_final"]].rename(columns={"pred_final": "pred_old"}),
                             on=s4.KEY_COLUMNS)
        h_boot = s4.day_block_bootstrap(merged, merged, "pred_final", "pred_old", reps, seed)
        s4.log("HOLDOUT (dates 400-480, second use):\n" + holdout_table.round(5).to_string(index=False))
        s4.log(f"Holdout bootstrap batch4 - batch3: {h_boot}")
        lines += [
            "## Seeds", "", s4.markdown_table(pd.DataFrame(seed_rows)), "",
            "## Dev comparison (after post-processing)", "", s4.markdown_table(dev_table), "",
            f"Day-block bootstrap batch 4 - batch 3 on dev: diff {dev_boot['diff']:.5f}, "
            f"95% CI [{dev_boot['ci_low']:.5f}, {dev_boot['ci_high']:.5f}].", "",
            "## Frozen configuration", "", "```json", json.dumps(frozen4, indent=2), "```", "",
            "## Holdout (dates 400-480; second use, after batch 3; nothing selected on it)", "",
            s4.markdown_table(holdout_table), "",
            f"Day-block bootstrap batch 4 - batch 3 on holdout: diff {h_boot['diff']:.5f}, "
            f"95% CI [{h_boot['ci_low']:.5f}, {h_boot['ci_high']:.5f}].", "",
        ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    s4.log(f"Batch 4 finished in {(time.perf_counter() - started) / 60:.1f} min. Results: {out}")


if __name__ == "__main__":
    main()
