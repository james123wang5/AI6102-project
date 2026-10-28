#!/usr/bin/env python3
"""Stage 03/04: cumulative feature-group LightGBM experiments on the dev folds.

Experiments (cumulative), each run on every development fold:
  E1 raw                 raw order-book fields only
  E2 +optiver            + Optiver microstructure features
  E3 +temporal           + exact-lag / rolling features
  E4 +cross_section      + same-time cross-section features
  E5 +alpha101           + Alpha101-inspired candidates (= all features)

The final holdout (dates 400-480) is never loaded into any fold. Finished
(experiment, fold) pairs are appended to results CSV immediately, so a
re-submitted job resumes where the previous one stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai6102.baselines import (  # noqa: E402
    CSV_DTYPES,
    IDENTIFIER_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    folds_from_config,
    load_config,
)
from ai6102.features import build_feature_frame  # noqa: E402

EXPERIMENTS = ["raw", "optiver", "temporal", "cross_section", "alpha101"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_data(train_path: Path, nrows: int | None) -> pd.DataFrame:
    columns = list(dict.fromkeys(IDENTIFIER_COLUMNS + RAW_FEATURE_COLUMNS + [TARGET_COLUMN]))
    frame = pd.read_csv(train_path, usecols=columns, dtype=CSV_DTYPES, nrows=nrows)
    return frame


def build_or_load_features(frame: pd.DataFrame, cache: Path | None, config_path: Path) -> pd.DataFrame:
    if cache is not None and cache.exists():
        log(f"Loading cached features: {cache}")
        return pd.read_pickle(cache)
    feature_config = json.loads(config_path.read_text(encoding="utf-8"))
    log("Building features (this takes several minutes on the full data) ...")
    started = time.perf_counter()
    # Features never read target; keep it aside and re-attach by position.
    features = build_feature_frame(
        frame.drop(columns=[TARGET_COLUMN]),
        include_alpha_adapted=True,
        exact_lag_seconds=tuple(feature_config["exact_lag_seconds"]),
        rolling_windows_rows=tuple(feature_config["rolling_windows_rows"]),
    )
    features[TARGET_COLUMN] = frame[TARGET_COLUMN].to_numpy()
    for column in features.columns:
        if features[column].dtype == "float64":
            features[column] = features[column].astype("float32")
    features["stock_id"] = pd.Categorical(features["stock_id"], categories=range(200))
    log(f"Features built in {time.perf_counter() - started:.0f}s: {features.shape[1]} columns, {len(features):,} rows")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_pickle(cache)
        log(f"Cached features to {cache}")
    return features


def group_columns(features: pd.DataFrame) -> dict[str, list[str]]:
    """Cumulative column lists per experiment, derived from feature names."""
    raw = [c for c in RAW_FEATURE_COLUMNS if c in features.columns]
    optiver = [
        "price_spread", "mid_price", "wap_mid_diff", "reference_wap_diff", "near_wap_diff", "far_wap_diff",
        "near_price_is_missing", "far_price_is_missing", "liquidity_imbalance", "auction_imbalance_ratio",
        "signed_imbalance", "market_urgency",
    ]
    alpha = [c for c in features.columns if c.startswith("alpha_")]
    cross = [c for c in features.columns if c.endswith(("_cs_rank", "_cs_mean_diff", "_cs_zscore"))]
    known = set(raw) | set(optiver) | set(alpha) | set(cross) | set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN}
    temporal = [c for c in features.columns if c not in known]
    cumulative = {}
    cumulative["raw"] = raw
    cumulative["optiver"] = cumulative["raw"] + optiver
    cumulative["temporal"] = cumulative["optiver"] + temporal
    cumulative["cross_section"] = cumulative["temporal"] + cross
    cumulative["alpha101"] = cumulative["cross_section"] + alpha
    return cumulative


def progress_callback(tag: str, every: int = 50):
    started = time.perf_counter()

    def _callback(env: lgb.callback.CallbackEnv) -> None:
        iteration = env.iteration + 1
        if iteration % every == 0 or iteration == env.end_iteration:
            score = env.evaluation_result_list[0][2] if env.evaluation_result_list else float("nan")
            log(f"    {tag} tree {iteration}/{env.end_iteration}  valid MAE {score:.5f}  ({time.perf_counter() - started:.0f}s)")

    return _callback


def run_experiment(features: pd.DataFrame, columns: list[str], fold, config: dict, tag: str, threads: int) -> dict:
    dates = features["date_id"]
    train = features.loc[fold.train_mask(dates)]
    valid = features.loc[fold.validation_mask(dates)]
    inner_start = fold.inner_validation_start(int(config["inner_early_stopping_days"]))
    fit = train.loc[train["date_id"] < inner_start]
    stop = train.loc[train["date_id"] >= inner_start]

    params = dict(config["lightgbm"])
    max_rounds = int(params.pop("n_estimators"))
    early_rounds = int(params.pop("early_stopping_rounds"))
    seed = int(config["seed"])
    params.update(
        objective="regression_l1", metric="mae", seed=seed, verbosity=-1, num_threads=threads,
        bagging_freq=1, bagging_fraction=params.pop("subsample", 0.8),
        feature_fraction=params.pop("colsample_bytree", 0.9),
        lambda_l2=params.pop("reg_lambda", 1.0), min_data_in_leaf=params.pop("min_child_samples", 100),
        learning_rate=params.pop("learning_rate"), num_leaves=params.pop("num_leaves"),
    )
    params.pop("n_jobs", None)

    def dataset(frame: pd.DataFrame, reference=None) -> lgb.Dataset:
        return lgb.Dataset(frame[columns], label=frame[TARGET_COLUMN], categorical_feature=["stock_id"], reference=reference, free_raw_data=False)

    log(f"  {tag}: step 1/2 early-stopping search ({len(columns)} features, {len(fit):,} train rows)")
    t0 = time.perf_counter()
    d_fit = dataset(fit)
    d_stop = dataset(stop, d_fit)
    selector = lgb.train(
        params, d_fit, num_boost_round=max_rounds, valid_sets=[d_stop], valid_names=["inner"],
        callbacks=[lgb.early_stopping(early_rounds, verbose=False), progress_callback(tag)],
    )
    best = int(selector.best_iteration or max_rounds)
    log(f"  {tag}: best round = {best}")

    log(f"  {tag}: step 2/2 refit on full training window, {best} trees")
    d_train = dataset(train)
    d_valid = dataset(valid, d_train)
    model = lgb.train(params, d_train, num_boost_round=best, valid_sets=[d_valid], valid_names=["val"], callbacks=[progress_callback(tag)])
    prediction = model.predict(valid[columns], num_iteration=best)
    y = valid[TARGET_COLUMN].to_numpy()
    mae = float(np.mean(np.abs(y - prediction)))
    importance = pd.Series(model.feature_importance("gain"), index=columns).sort_values(ascending=False)
    return {
        "experiment": tag.split("/")[0], "fold": fold.name, "n_features": len(columns), "best_iteration": best,
        "mae": mae, "train_rows": len(train), "valid_rows": len(valid), "seconds": round(time.perf_counter() - t0, 1),
        "top_features": ";".join(importance.index[:10]),
    }, importance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "optiver-trading-at-the-close" / "train.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.json")
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features_v1.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "03_feature_experiments")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "artifacts" / "features_v1.pkl")
    parser.add_argument("--fold", action="append", dest="folds", help="fold name; repeat; default all dev folds")
    parser.add_argument("--experiment", action="append", dest="experiments", choices=EXPERIMENTS, help="default all five")
    parser.add_argument("--threads", type=int, default=0, help="0 = use all allocated CPUs")
    parser.add_argument("--nrows", type=int, default=None, help="smoke test: read only the first N csv rows")
    args = parser.parse_args()

    config = load_config(args.config)
    all_folds = folds_from_config(config)
    folds = [f for f in all_folds if not args.folds or f.name in args.folds]
    experiments = args.experiments or EXPERIMENTS
    holdout_start = int(config["final_holdout"]["validation_start"])
    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)

    log(f"Threads: {threads}; folds: {[f.name for f in folds]}; experiments: {experiments}")
    frame = load_data(args.train_path, args.nrows)
    log(f"Loaded {len(frame):,} rows")
    # Hard guard: the final holdout never enters development.
    frame = frame.loc[frame["date_id"] < holdout_start].reset_index(drop=True)
    log(f"Dropped holdout dates >= {holdout_start}; {len(frame):,} rows remain")

    features = build_or_load_features(frame, None if args.nrows else args.cache, args.features_config)
    valid_target = np.isfinite(features[TARGET_COLUMN].to_numpy())
    log(f"Rows with non-finite target are excluded from training/eval: {int((~valid_target).sum()):,}")
    features = features.loc[valid_target].reset_index(drop=True)
    columns_by_experiment = group_columns(features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "feature_experiments.csv"
    done = set()
    if results_path.exists():
        previous = pd.read_csv(results_path)
        done = set(zip(previous["experiment"], previous["fold"]))

    total = len(folds) * len(experiments)
    counter = 0
    started = time.perf_counter()
    for fold in folds:
        for name in experiments:
            counter += 1
            if (name, fold.name) in done:
                log(f"=== [{counter}/{total}] {name} @ {fold.name}: already done, skipping")
                continue
            elapsed = time.perf_counter() - started
            log(f"=== [{counter}/{total}] {name} @ {fold.name}  (elapsed {elapsed / 60:.1f} min) ===")
            row, importance = run_experiment(features, columns_by_experiment[name], fold, config, f"{name}/{fold.name}", threads)
            pd.DataFrame([row]).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
            importance.to_csv(args.output_dir / f"importance_{name}_{fold.name}.csv", header=["gain"])
            log(f"=== [{counter}/{total}] DONE {name} @ {fold.name}: MAE {row['mae']:.5f}, {row['n_features']} features, {row['seconds']:.0f}s")

    summary = pd.read_csv(results_path)
    table = summary.pivot_table(index="experiment", columns="fold", values="mae").reindex(EXPERIMENTS)
    table["mean"] = table.mean(axis=1)
    table.to_csv(args.output_dir / "mae_summary.csv")
    log("MAE summary (lower is better):\n" + table.round(5).to_string())
    log(f"All done. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
