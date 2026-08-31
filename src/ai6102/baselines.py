"""Leakage-aware Zero, Median and raw-field LightGBM baselines."""

from __future__ import annotations

import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ai6102.validation import DateFold


RAW_FEATURE_COLUMNS = [
    "stock_id",
    "seconds_in_bucket",
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "reference_price",
    "matched_size",
    "far_price",
    "near_price",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
]
IDENTIFIER_COLUMNS = ["date_id", "stock_id", "seconds_in_bucket"]
TARGET_COLUMN = "target"

CSV_DTYPES = {
    "stock_id": "int16",
    "date_id": "int16",
    "seconds_in_bucket": "int16",
    "imbalance_size": "float32",
    "imbalance_buy_sell_flag": "int8",
    "reference_price": "float32",
    "matched_size": "float32",
    "far_price": "float32",
    "near_price": "float32",
    "bid_price": "float32",
    "bid_size": "float32",
    "ask_price": "float32",
    "ask_size": "float32",
    "wap": "float32",
    "target": "float32",
}


@dataclass(frozen=True)
class BaselineRun:
    fold: DateFold
    zero_mae: float
    median_mae: float
    raw_lightgbm_mae: float
    training_rows: int
    validation_rows: int
    best_iteration: int
    training_seconds: float
    peak_rss_mb: float

    def rows(self) -> list[dict[str, Any]]:
        metadata = {
            "fold": self.fold.name,
            "train_dates": f"{self.fold.train_start}-{self.fold.train_end}",
            "validation_dates": f"{self.fold.validation_start}-{self.fold.validation_end}",
            "training_rows": self.training_rows,
            "validation_rows": self.validation_rows,
            "best_iteration": self.best_iteration,
            "raw_lightgbm_training_seconds": self.training_seconds,
            "peak_rss_mb": self.peak_rss_mb,
        }
        return [
            {**metadata, "model": "zero", "mae": self.zero_mae},
            {**metadata, "model": "median", "mae": self.median_mae},
            {**metadata, "model": "raw_lightgbm", "mae": self.raw_lightgbm_mae},
        ]


def load_raw_training_data(train_path: Path) -> tuple[pd.DataFrame, int]:
    """Load only fields needed for stage-02 baselines and drop invalid labels."""
    use_columns = list(dict.fromkeys(IDENTIFIER_COLUMNS + RAW_FEATURE_COLUMNS + [TARGET_COLUMN]))
    frame = pd.read_csv(train_path, usecols=use_columns, dtype=CSV_DTYPES)
    valid_target = np.isfinite(frame[TARGET_COLUMN].to_numpy())
    dropped_rows = int((~valid_target).sum())
    frame = frame.loc[valid_target].copy()
    frame["stock_id"] = pd.Categorical(frame["stock_id"], categories=range(200))
    return frame, dropped_rows


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, RAW_FEATURE_COLUMNS]


def _mae(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - prediction)))


def _build_model(parameters: dict[str, Any], n_estimators: int, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        **parameters,
        n_estimators=n_estimators,
        random_state=seed,
        verbosity=-1,
    )


def run_fold(frame: pd.DataFrame, fold: DateFold, config: dict[str, Any]) -> tuple[BaselineRun, pd.DataFrame, lgb.LGBMRegressor]:
    """Evaluate all three baselines on one chronological development fold."""
    dates = frame["date_id"]
    train_frame = frame.loc[fold.train_mask(dates)]
    validation_frame = frame.loc[fold.validation_mask(dates)]
    if train_frame.empty or validation_frame.empty:
        raise ValueError(f"Fold {fold.name} is empty after target filtering.")

    y_train = train_frame[TARGET_COLUMN].to_numpy(copy=False)
    y_validation = validation_frame[TARGET_COLUMN].to_numpy(copy=False)
    zero_mae = _mae(y_validation, np.zeros_like(y_validation))
    median_prediction = float(np.median(y_train))
    median_mae = _mae(y_validation, np.full_like(y_validation, median_prediction))

    inner_start = fold.inner_validation_start(int(config["inner_early_stopping_days"]))
    fit_frame = train_frame.loc[train_frame["date_id"] < inner_start]
    early_stop_frame = train_frame.loc[train_frame["date_id"] >= inner_start]
    parameters = dict(config["lightgbm"])
    max_estimators = int(parameters.pop("n_estimators"))
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds"))
    seed = int(config["seed"])

    selection_model = _build_model(parameters, max_estimators, seed)
    selection_model.fit(
        _features(fit_frame),
        fit_frame[TARGET_COLUMN],
        categorical_feature=["stock_id"],
        eval_X=_features(early_stop_frame),
        eval_y=early_stop_frame[TARGET_COLUMN],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_iteration = int(selection_model.best_iteration_ or max_estimators)

    started = time.perf_counter()
    final_model = _build_model(parameters, best_iteration, seed)
    final_model.fit(_features(train_frame), train_frame[TARGET_COLUMN], categorical_feature=["stock_id"])
    prediction = final_model.predict(_features(validation_frame))
    elapsed_seconds = time.perf_counter() - started
    raw_mae = _mae(y_validation, prediction)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    prediction_frame = validation_frame.loc[:, IDENTIFIER_COLUMNS].copy()
    prediction_frame["target"] = y_validation
    prediction_frame["prediction_zero"] = 0.0
    prediction_frame["prediction_median"] = median_prediction
    prediction_frame["prediction_raw_lightgbm"] = prediction
    prediction_frame["absolute_error_raw_lightgbm"] = np.abs(y_validation - prediction)
    return (
        BaselineRun(
            fold=fold,
            zero_mae=zero_mae,
            median_mae=median_mae,
            raw_lightgbm_mae=raw_mae,
            training_rows=len(train_frame),
            validation_rows=len(validation_frame),
            best_iteration=best_iteration,
            training_seconds=elapsed_seconds,
            peak_rss_mb=peak_rss_mb,
        ),
        prediction_frame,
        final_model,
    )


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def folds_from_config(config: dict[str, Any]) -> list[DateFold]:
    return [DateFold(**fold) for fold in config["development_folds"]]


def write_baseline_outputs(
    results: list[BaselineRun],
    predictions: dict[str, pd.DataFrame],
    output_dir: Path,
    config: dict[str, Any],
    invalid_target_rows: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [record for result in results for record in result.rows()]
    pd.DataFrame(records).to_csv(output_dir / "baseline_summary.csv", index=False)
    pd.DataFrame(records).query("model == 'raw_lightgbm'").to_csv(output_dir / "raw_lightgbm_summary.csv", index=False)
    for fold_name, frame in predictions.items():
        frame.to_csv(output_dir / f"predictions_{fold_name}.csv.gz", index=False, compression="gzip")
        frame.groupby("date_id", observed=True)["absolute_error_raw_lightgbm"].mean().rename("mae").to_csv(
            output_dir / f"raw_lightgbm_daily_mae_{fold_name}.csv"
        )
    metadata = {"invalid_target_rows_filtered": invalid_target_rows, "config": config}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
