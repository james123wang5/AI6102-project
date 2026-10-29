"""Shared runner for the stage-04 pipeline, batches 1-4.

The main chain uses LightGBM on CPU; XGBoost on GPU is a model-family
comparison run on the same folds, features and protocol. Every model run is
identified by a key such as ``model=lgb|set=alpha101|params=l127|rec=0|seed=2026``
plus a fold name. Finished
runs are appended to ``reports/04_xgb_gpu/runs.csv`` and their out-of-sample
predictions saved under ``artifacts/04_xgb_gpu/preds/``; a resubmitted job
loads them instead of retraining, and later batches reuse earlier runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ai6102.baselines import CSV_DTYPES, IDENTIFIER_COLUMNS, RAW_FEATURE_COLUMNS, TARGET_COLUMN, folds_from_config, load_config
from ai6102.features import build_feature_frame
from ai6102.features.market import MARKET_COLUMNS, add_market_features
from ai6102.fold_features import INDEX_COLUMNS, STOCK_COLUMNS, add_index_features, add_stock_features, fit_stock_stats
from ai6102.history_features import HIST_COLUMNS, INTRA_COLUMNS, add_intraday_relative_features
from ai6102.index_model import IndexFit, fit_index_weights, neutralize
from ai6102.validation import DateFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_COLUMNS = ["date_id", "seconds_in_bucket", "stock_id"]

OPTIVER_COLUMNS = [
    "price_spread", "mid_price", "wap_mid_diff", "reference_wap_diff", "near_wap_diff", "far_wap_diff",
    "near_price_is_missing", "far_price_is_missing", "liquidity_imbalance", "auction_imbalance_ratio",
    "signed_imbalance", "market_urgency",
]
FEATURE_SETS = ["raw", "optiver", "temporal", "cross_section", "alpha101", "s_market", "s_index", "s_stock",
                "h_hist", "h_intra", "h_both"]
SET_LABELS = {
    "raw": "E1 raw fields",
    "optiver": "E2 +Optiver microstructure",
    "temporal": "E3 +intraday temporal",
    "cross_section": "E4 +cross-section",
    "alpha101": "E5 +Alpha101-inspired",
    "s_market": "N1 +market aggregates",
    "s_index": "N2 +synthetic-index features",
    "s_stock": "N3 +stock statistics",
    "h_hist": "H1 +cross-day target history",
    "h_intra": "H2 +intraday relative path",
    "h_both": "H3 +history and intraday path",
}
NEEDS_INDEX = {"s_index", "s_stock", "h_hist", "h_intra", "h_both"}
NEEDS_STOCK = {"s_stock", "h_hist", "h_intra", "h_both"}
NEEDS_INTRA = {"h_intra", "h_both"}
# A set name may carry a stock_id treatment suffix: "h_both@num" (plain integer) or "h_both@drop" (removed);
# no suffix keeps stock_id as a LightGBM categorical feature.
STOCK_ID_MODES = ("cat", "num", "drop")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- arguments


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "optiver-trading-at-the-close" / "train.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.json")
    parser.add_argument("--stage-config", type=Path, default=PROJECT_ROOT / "configs" / "stage04.json")
    parser.add_argument("--features-config", type=Path, default=PROJECT_ROOT / "configs" / "features_v1.json")
    parser.add_argument("--xgb-device", default="cuda", choices=["cuda", "cpu"], help="XGBoost device (cpu only for local smoke tests)")
    parser.add_argument("--skip-xgb", action="store_true", help="run only the LightGBM main chain")
    parser.add_argument("--threads", type=int, default=0, help="LightGBM threads; 0 = SLURM_CPUS_PER_TASK or all cores")
    parser.add_argument("--smoke-stocks", type=int, default=None, help="smoke test: keep stock_id < N only")
    parser.add_argument("--smoke-rounds", type=int, default=None, help="smoke test: cap boosting rounds")
    parser.add_argument("--output-root", type=Path, default=None, help="default reports/04_xgb_gpu (smoke: reports/04_smoke)")
    return parser


class Context:
    """Paths, configs and caches shared by one batch script."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config(args.config)
        self.stage = json.loads(args.stage_config.read_text(encoding="utf-8"))
        self.folds = folds_from_config(self.config)
        h = self.config["final_holdout"]
        self.holdout = DateFold("holdout", h["train_start"], h["train_end"], h["validation_start"], h["validation_end"])
        self.xgb_device = args.xgb_device
        self.use_xgb = not args.skip_xgb
        self.threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
        smoke = args.smoke_stocks is not None
        name = "04_smoke" if smoke else "04_xgb_gpu"
        self.report_dir = args.output_root or PROJECT_ROOT / "reports" / name
        self.artifact_dir = PROJECT_ROOT / "artifacts" / name
        self.cache_path = self.artifact_dir / ("features_v2_full.pkl" if not smoke else f"features_v2_s{args.smoke_stocks}.pkl")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "preds").mkdir(parents=True, exist_ok=True)
        self.registry = RunRegistry(self.report_dir / "runs.csv", self.artifact_dir / "preds")
        self._index_fits: dict[int, IndexFit] = {}
        self._stock_stats: dict[int, pd.DataFrame] = {}

    # ------------------------------------------------------------ params

    def params(self, model: str, params_name: str) -> dict:
        params = dict(self.stage[f"{model}_base_params"])
        params.update(self.stage[f"{model}_tuning_grid"][params_name])
        return params

    def max_rounds(self) -> int:
        return int(self.args.smoke_rounds or self.stage["max_rounds"])

    def best_params_name(self, model: str) -> str:
        path = self.report_dir / "batch1" / "best_params.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run batch 1 first.")
        best = json.loads(path.read_text(encoding="utf-8"))
        if model not in best:
            raise KeyError(f"No tuned params for model '{model}' in {path}; rerun batch 1 without --skip-xgb.")
        return best[model]["params_name"]

    # ------------------------------------------------------------ fitted state

    def index_fit(self, features: pd.DataFrame, train_end: int) -> IndexFit:
        if train_end not in self._index_fits:
            path = self.artifact_dir / f"index_weights_train_end_{train_end}.json"
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                fit = IndexFit(np.asarray(raw["weights"]), raw["r2"], raw["n_times"], raw["implied_dispersion"])
            else:
                log(f"  fitting index weights on dates 0..{train_end}")
                train = features.loc[features["date_id"] <= train_end]
                fit = fit_index_weights(train)
                path.write_text(json.dumps({
                    "train_end": train_end, "r2": fit.r2, "n_times": fit.n_times,
                    "implied_dispersion": fit.implied_dispersion, "weights": fit.weights.tolist(),
                }), encoding="utf-8")
            self._index_fits[train_end] = fit
        return self._index_fits[train_end]

    def stock_stats(self, features: pd.DataFrame, train_end: int) -> pd.DataFrame:
        if train_end not in self._stock_stats:
            self._stock_stats[train_end] = fit_stock_stats(features.loc[features["date_id"] <= train_end])
        return self._stock_stats[train_end]


# ---------------------------------------------------------------- data


def load_features(ctx: Context) -> pd.DataFrame:
    """Target-free causal features for all 481 days, built once and cached.

    Every feature only reads the same stock-day history or the same-time
    cross-section, so building on all dates gives identical values for the
    development rows; folds decide which dates are used for what.
    """
    if ctx.cache_path.exists():
        log(f"Loading cached features: {ctx.cache_path}")
        return _infinite_to_nan(pd.read_pickle(ctx.cache_path))
    columns = list(dict.fromkeys(IDENTIFIER_COLUMNS + RAW_FEATURE_COLUMNS + [TARGET_COLUMN]))
    log(f"Reading {ctx.args.train_path} ...")
    frame = pd.read_csv(ctx.args.train_path, usecols=columns, dtype=CSV_DTYPES)
    if ctx.args.smoke_stocks is not None:
        frame = frame.loc[frame["stock_id"] < ctx.args.smoke_stocks].reset_index(drop=True)
    log(f"Loaded {len(frame):,} rows. Building features (one-off, roughly 10-30 min on the full data) ...")
    started = time.perf_counter()
    feature_config = json.loads(ctx.args.features_config.read_text(encoding="utf-8"))
    features = build_feature_frame(
        frame.drop(columns=[TARGET_COLUMN]),
        include_alpha_adapted=True,
        exact_lag_seconds=tuple(feature_config["exact_lag_seconds"]),
        rolling_windows_rows=tuple(feature_config["rolling_windows_rows"]),
    )
    features = add_market_features(features)
    features[TARGET_COLUMN] = frame[TARGET_COLUMN].to_numpy()
    for column in features.columns:
        if features[column].dtype == "float64":
            features[column] = features[column].astype("float32")
    features["stock_id"] = pd.Categorical(features["stock_id"], categories=range(200))
    log(f"Features built in {time.perf_counter() - started:.0f}s: {features.shape[1]} columns")
    features = _infinite_to_nan(features)
    features.to_pickle(ctx.cache_path)
    log(f"Cached features to {ctx.cache_path}")
    return features


def _infinite_to_nan(features: pd.DataFrame) -> pd.DataFrame:
    """XGBoost rejects inf; treat it as missing (e.g. a ratio over a zero size), like the README rule."""
    counts = {}
    for column in features.columns:
        if column != TARGET_COLUMN and pd.api.types.is_float_dtype(features[column]):
            infinite = np.isinf(features[column].to_numpy())
            if infinite.any():
                counts[column] = int(infinite.sum())
                features[column] = features[column].mask(infinite)
    if counts:
        log(f"Replaced inf with NaN: {counts}")
    return features


def feature_set_columns(features: pd.DataFrame) -> dict[str, list[str]]:
    raw = [c for c in RAW_FEATURE_COLUMNS if c in features.columns]
    alpha = [c for c in features.columns if c.startswith("alpha_")]
    cross = [c for c in features.columns if c.endswith(("_cs_rank", "_cs_mean_diff", "_cs_zscore"))]
    known = set(raw) | set(OPTIVER_COLUMNS) | set(alpha) | set(cross) | set(MARKET_COLUMNS)
    known |= set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN} | set(INDEX_COLUMNS) | set(STOCK_COLUMNS)
    known |= set(HIST_COLUMNS) | set(INTRA_COLUMNS) | {"stock_id_num"}
    temporal = [c for c in features.columns if c not in known]
    sets = {"raw": raw}
    sets["optiver"] = sets["raw"] + OPTIVER_COLUMNS
    sets["temporal"] = sets["optiver"] + temporal
    sets["cross_section"] = sets["temporal"] + cross
    sets["alpha101"] = sets["cross_section"] + alpha
    sets["s_market"] = sets["alpha101"] + MARKET_COLUMNS
    sets["s_index"] = sets["s_market"] + INDEX_COLUMNS
    sets["s_stock"] = sets["s_index"] + STOCK_COLUMNS
    sets["h_hist"] = sets["s_stock"] + HIST_COLUMNS
    sets["h_intra"] = sets["s_stock"] + INTRA_COLUMNS
    sets["h_both"] = sets["s_stock"] + HIST_COLUMNS + INTRA_COLUMNS
    return sets


def split_set_name(set_name: str) -> tuple[str, str]:
    base, _, mode = set_name.partition("@")
    mode = mode or "cat"
    if mode not in STOCK_ID_MODES:
        raise ValueError(f"Unknown stock_id mode in {set_name!r}")
    return base, mode


def fold_frames(ctx: Context, features: pd.DataFrame, fold: DateFold, set_name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train/eval frames for one fold, adding fold-fitted features if needed."""
    base, stock_mode = split_set_name(set_name)
    scope = features.loc[features["date_id"].between(fold.train_start, fold.validation_end)]
    if base in NEEDS_INDEX:
        scope = add_index_features(scope, ctx.index_fit(features, fold.train_end).weights)
    if base in NEEDS_STOCK:
        scope = add_stock_features(scope, ctx.stock_stats(features, fold.train_end))
    if base in NEEDS_INTRA:
        scope = add_intraday_relative_features(scope)
    columns = list(feature_set_columns(features)[base])
    if stock_mode == "num":
        scope = scope.assign(stock_id_num=scope["stock_id"].astype("int16"))
        columns = ["stock_id_num" if c == "stock_id" else c for c in columns]
    elif stock_mode == "drop":
        columns = [c for c in columns if c != "stock_id"]
    finite = np.isfinite(scope[TARGET_COLUMN].to_numpy())
    dates = scope["date_id"]
    train = scope.loc[finite & fold.train_mask(dates).to_numpy()]
    evaluation = scope.loc[finite & fold.validation_mask(dates).to_numpy()]
    return train, evaluation, columns


# ---------------------------------------------------------------- registry


def run_key(model: str, set_name: str, params_name: str, recency: bool = False, seed: int = 2026) -> str:
    return f"model={model}|set={set_name}|params={params_name}|rec={int(recency)}|seed={seed}"


class RunRegistry:
    def __init__(self, path: Path, pred_dir: Path) -> None:
        self.path = path
        self.pred_dir = pred_dir

    def _pred_path(self, key: str, fold: str) -> Path:
        slug = key.replace("|", "__").replace("=", "-")
        return self.pred_dir / f"{slug}__{fold}.pkl.gz"

    def done(self, key: str, fold: str) -> bool:
        if not self.path.exists() or not self._pred_path(key, fold).exists():
            return False
        table = pd.read_csv(self.path)
        return bool(((table["key"] == key) & (table["fold"] == fold)).any())

    def record(self, row: dict, predictions: pd.DataFrame) -> None:
        predictions.to_pickle(self._pred_path(row["key"], row["fold"]), compression="gzip")
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=not self.path.exists(), index=False)

    def predictions(self, key: str, fold: str) -> pd.DataFrame:
        return pd.read_pickle(self._pred_path(key, fold), compression="gzip")

    def table(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()


# ---------------------------------------------------------------- training


def _xgb_progress(tag: str, every: int):
    import xgboost as xgb

    class Progress(xgb.callback.TrainingCallback):
        def __init__(self) -> None:
            self.started = time.perf_counter()

        def after_iteration(self, model, epoch: int, evals_log) -> bool:
            if (epoch + 1) % every == 0:
                score = ""
                for data_name, metrics in evals_log.items():
                    for metric_name, values in metrics.items():
                        score = f"  {data_name} {metric_name.upper()} {values[-1]:.5f}"
                log(f"    {tag} round {epoch + 1}{score}  ({time.perf_counter() - self.started:.0f}s)")
            return False

    return Progress()


def _lgb_progress(tag: str, every: int):
    started = time.perf_counter()

    def _callback(env) -> None:
        iteration = env.iteration + 1
        if iteration % every == 0:
            score = f"  inner MAE {env.evaluation_result_list[0][2]:.5f}" if env.evaluation_result_list else ""
            log(f"    {tag} round {iteration}{score}  ({time.perf_counter() - started:.0f}s)")

    return _callback


def train_and_predict(
    ctx: Context,
    model: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    params_name: str,
    recency: bool,
    seed: int,
    tag: str,
) -> tuple[np.ndarray, int, pd.Series]:
    """Inner early stopping on the last training days, then refit on all training days."""
    train_end = int(train["date_id"].max())
    inner_start = train_end - int(ctx.config["inner_early_stopping_days"]) + 1
    fit = train.loc[train["date_id"] < inner_start]
    stop = train.loc[train["date_id"] >= inner_start]

    def weights(frame: pd.DataFrame) -> np.ndarray | None:
        if not recency:
            return None
        half_life = float(ctx.stage["recency_half_life_days"])
        return np.power(0.5, (train_end - frame["date_id"].to_numpy()) / half_life)

    log(f"  {tag}: step 1/2 early stopping ({len(columns)} features, {len(fit):,} rows, inner days {inner_start}..{train_end})")
    trainer = _train_lgb if model == "lgb" else _train_xgb
    return trainer(ctx, fit, stop, train, evaluation, columns, params_name, seed, weights, tag)


def _train_lgb(ctx, fit, stop, train, evaluation, columns, params_name, seed, weights, tag):
    import lightgbm as lgb

    params = ctx.params("lgb", params_name)
    params.update(seed=seed, num_threads=ctx.threads)
    early = int(ctx.stage["early_stopping_rounds"])

    def dataset(frame: pd.DataFrame, reference=None):
        return lgb.Dataset(frame[columns], label=frame[TARGET_COLUMN], weight=weights(frame),
                           categorical_feature=[c for c in ["stock_id"] if c in columns], reference=reference,
                           free_raw_data=True)

    d_fit = dataset(fit)
    selector = lgb.train(
        params, d_fit, num_boost_round=ctx.max_rounds(), valid_sets=[dataset(stop, d_fit)], valid_names=["inner"],
        callbacks=[lgb.early_stopping(early, verbose=False), _lgb_progress(tag, 200)],
    )
    best = int(selector.best_iteration or ctx.max_rounds())
    del d_fit, selector
    log(f"  {tag}: best round = {best} (cap {ctx.max_rounds()})")
    log(f"  {tag}: step 2/2 refit on all {len(train):,} training rows, {best} rounds")
    booster = lgb.train(params, dataset(train), num_boost_round=best, callbacks=[_lgb_progress(tag, 200)])
    prediction = booster.predict(evaluation[columns], num_threads=ctx.threads)
    importance = pd.Series(booster.feature_importance("gain"), index=columns, dtype="float64")
    return prediction, best, importance.sort_values(ascending=False)


def _train_xgb(ctx, fit, stop, train, evaluation, columns, params_name, seed, weights, tag):
    import xgboost as xgb

    params = ctx.params("xgb", params_name)
    params.update(device=ctx.xgb_device, seed=seed)
    early = int(ctx.stage["early_stopping_rounds"])

    def matrix(frame: pd.DataFrame, ref=None):
        return xgb.QuantileDMatrix(
            frame[columns], label=frame[TARGET_COLUMN], weight=weights(frame), ref=ref,
            enable_categorical=True, max_bin=int(params.get("max_bin", 256)),
        )

    d_fit = matrix(fit)
    d_stop = matrix(stop, d_fit)
    selector = xgb.train(
        params, d_fit, num_boost_round=ctx.max_rounds(), evals=[(d_stop, "inner")],
        early_stopping_rounds=early, verbose_eval=False, callbacks=[_xgb_progress(tag, 200)],
    )
    best = int(selector.best_iteration) + 1
    del d_fit, d_stop, selector
    log(f"  {tag}: best round = {best} (cap {ctx.max_rounds()})")
    log(f"  {tag}: step 2/2 refit on all {len(train):,} training rows, {best} rounds")
    d_train = matrix(train)
    booster = xgb.train(params, d_train, num_boost_round=best, verbose_eval=False, callbacks=[_xgb_progress(tag, 200)])
    del d_train
    prediction = booster.predict(xgb.DMatrix(evaluation[columns], enable_categorical=True))
    importance = pd.Series(booster.get_score(importance_type="total_gain"), dtype="float64").reindex(columns).fillna(0.0)
    return prediction, best, importance.sort_values(ascending=False)


def execute_run(
    ctx: Context,
    features: pd.DataFrame,
    fold: DateFold,
    model: str,
    set_name: str,
    params_name: str,
    recency: bool = False,
    seed: int = 2026,
    importance_dir: Path | None = None,
) -> pd.DataFrame:
    """Train (or load) one keyed run on one fold; return its prediction frame."""
    key = run_key(model, set_name, params_name, recency, seed)
    if ctx.registry.done(key, fold.name):
        log(f"  {key} @ {fold.name}: already done, loading saved predictions")
        return ctx.registry.predictions(key, fold.name)
    started = time.perf_counter()
    train, evaluation, columns = fold_frames(ctx, features, fold, set_name)
    tag = f"{model}/{set_name}/{params_name}/rec{int(recency)}/s{seed}/{fold.name}"
    prediction, best, importance = train_and_predict(ctx, model, train, evaluation, columns, params_name, recency, seed, tag)
    predictions = evaluation.loc[:, KEY_COLUMNS + [TARGET_COLUMN]].copy()
    predictions["stock_id"] = predictions["stock_id"].astype("int16")
    predictions["pred"] = prediction.astype("float32")
    mae = float(np.mean(np.abs(predictions[TARGET_COLUMN] - predictions["pred"])))
    row = {
        "key": key, "model": model, "set": set_name, "params": params_name, "recency": int(recency), "seed": seed,
        "fold": fold.name, "n_features": len(columns), "best_iteration": best, "max_rounds": ctx.max_rounds(),
        "mae": mae, "train_rows": len(train), "eval_rows": len(evaluation),
        "seconds": round(time.perf_counter() - started, 1), "top_features": ";".join(importance.index[:10]),
    }
    ctx.registry.record(row, predictions)
    if importance_dir is not None:
        importance_dir.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_dir / f"importance_{model}_{set_name}_{params_name}_{fold.name}.csv", header=["gain"])
    log(f"  DONE {key} @ {fold.name}: MAE {mae:.5f}, best round {best}, {row['seconds']:.0f}s")
    return predictions


# ---------------------------------------------------------------- evaluation helpers


def mae(frame: pd.DataFrame, column: str = "pred") -> float:
    return float(np.mean(np.abs(frame[TARGET_COLUMN].to_numpy() - frame[column].to_numpy())))


def pooled_mae(frames: list[pd.DataFrame], column: str = "pred") -> float:
    errors = np.concatenate([np.abs(f[TARGET_COLUMN].to_numpy() - f[column].to_numpy()) for f in frames])
    return float(errors.mean())


def postprocess(frame: pd.DataFrame, mode: str, alpha: float, weights: np.ndarray | None) -> np.ndarray:
    prediction = frame["pred"].to_numpy(dtype="float64")
    if mode == "index":
        prediction = neutralize(frame, prediction, weights)
    elif mode == "equal":
        prediction = neutralize(frame, prediction, None)
    elif mode != "none":
        raise ValueError(mode)
    return alpha * prediction


def postprocess_grid(ctx: Context, features: pd.DataFrame, preds_by_fold: dict[str, pd.DataFrame], label: str) -> pd.DataFrame:
    """MAE for every (neutralization mode, shrink alpha) on the dev folds."""
    rows = []
    fold_by_name = {f.name: f for f in ctx.folds}
    for mode in ("none", "equal", "index"):
        processed = {}
        for name, frame in preds_by_fold.items():
            weights = ctx.index_fit(features, fold_by_name[name].train_end).weights if mode == "index" else None
            processed[name] = postprocess(frame, mode, 1.0, weights)
        for alpha in ctx.stage["shrink_alphas"]:
            row = {"model": label, "mode": mode, "alpha": alpha}
            errors = []
            for name, frame in preds_by_fold.items():
                error = np.abs(frame[TARGET_COLUMN].to_numpy() - alpha * processed[name])
                row[name] = float(error.mean())
                errors.append(error)
            row["pooled"] = float(np.concatenate(errors).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def day_block_bootstrap(a: pd.DataFrame, b: pd.DataFrame, column_a: str, column_b: str, reps: int, seed: int) -> dict:
    """Bootstrap over whole trading days of MAE(a) - MAE(b) on identical rows."""
    error_a = np.abs(a[TARGET_COLUMN].to_numpy() - a[column_a].to_numpy())
    error_b = np.abs(b[TARGET_COLUMN].to_numpy() - b[column_b].to_numpy())
    days = a["date_id"].to_numpy()
    unique, inverse = np.unique(days, return_inverse=True)
    sum_a = np.bincount(inverse, weights=error_a)
    sum_b = np.bincount(inverse, weights=error_b)
    count = np.bincount(inverse).astype("float64")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(reps, len(unique)))
    counts = np.stack([np.bincount(row, minlength=len(unique)) for row in draws])
    diff = (counts @ sum_a - counts @ sum_b) / (counts @ count)
    point = (sum_a.sum() - sum_b.sum()) / count.sum()
    return {
        "diff": float(point), "ci_low": float(np.quantile(diff, 0.025)), "ci_high": float(np.quantile(diff, 0.975)),
        "share_negative": float((diff < 0).mean()), "n_days": int(len(unique)),
    }


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + body)


def fold_table(runs: pd.DataFrame, keys: dict[str, str], folds: list[DateFold]) -> pd.DataFrame:
    """Rows = labelled run keys, columns = per-fold MAE and the row-weighted pooled mean."""
    rows = []
    for label, key in keys.items():
        subset = runs.loc[runs["key"] == key].drop_duplicates("fold", keep="last").set_index("fold")
        row = {"model": label}
        for fold in folds:
            row[fold.name] = subset["mae"].get(fold.name, np.nan)
        if all(f.name in subset.index for f in folds):
            weights = subset.loc[[f.name for f in folds], "eval_rows"]
            row["pooled"] = float((subset.loc[weights.index, "mae"] * weights).sum() / weights.sum())
            row["mean_best_round"] = float(subset["best_iteration"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def check_environment(ctx: Context) -> None:
    import lightgbm as lgb

    log(f"lightgbm {lgb.__version__} on CPU with {ctx.threads} threads")
    if not ctx.use_xgb:
        log("XGBoost comparison skipped (--skip-xgb)")
        return
    check_device(ctx.xgb_device)


def check_device(device: str) -> None:
    import xgboost as xgb

    log(f"xgboost {xgb.__version__}; requested device={device}; build USE_CUDA={xgb.build_info().get('USE_CUDA')}")
    if device == "cuda":
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2000, 5))
        booster = xgb.train({"device": "cuda", "tree_method": "hist", "objective": "reg:absoluteerror"},
                            xgb.DMatrix(x, label=x[:, 0]), num_boost_round=5)
        config = json.loads(booster.save_config())
        log(f"GPU check passed: booster device = {config['learner']['generic_param']['device']}")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        log(f"CUDA_VISIBLE_DEVICES={visible}")
