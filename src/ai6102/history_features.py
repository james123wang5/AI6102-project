"""Stage-04 batch-4 features: cross-day target history and intraday relative path.

A. History (``HIST_COLUMNS``): for each (stock, seconds_in_bucket), statistics of
   the target on the previous 1..60 trading days. Day d only reads days < d,
   i.e. targets that the competition API had already revealed before day d
   started, so the same values are available online. Computed once on all
   dates; folds still decide which rows are trained or evaluated.

B. Intraday relative path (``INTRA_COLUMNS``): rolling statistics of
   ``rel_return_60s`` (stock 60-second return minus synthetic-index return,
   which equals the target realised 60 seconds ago) and the stock-minus-index
   return since the auction opened. Needs the fold-fitted index weights, so it
   is added per fold after ``add_index_features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai6102.features.operators import TIME_KEYS, stock_day_rolling
from ai6102.index_model import N_STOCKS


HIST_WINDOWS = (1, 3, 5, 10, 20, 40, 60)
HIST_STD_WINDOWS = (20, 60)
HIST_DAY_WINDOWS = (5, 20)
HIST_COLUMNS = (
    [f"hist_tgt_mean_{w}d" for w in HIST_WINDOWS]
    + [f"hist_tgt_std_{w}d" for w in HIST_STD_WINDOWS]
    + ["hist_abs_tgt_mean_20d"]
    + [f"hist_day_tgt_mean_{w}d" for w in HIST_DAY_WINDOWS]
)
INTRA_WINDOWS = (3, 6, 12)
INTRA_COLUMNS = (
    [f"rel_ret60_mean_{k}r" for k in INTRA_WINDOWS]
    + ["rel_ret60_std_12r", "rel_return_since_open"]
)
N_SECONDS = 55


def _trailing(cumulative: np.ndarray, window: int) -> np.ndarray:
    """cumulative[d] = sum over days < d; return the sum over days [d-window, d)."""
    shifted = np.zeros_like(cumulative)
    shifted[window:] = cumulative[:-window]
    return cumulative - shifted


def add_history_features(frame: pd.DataFrame, target_column: str = "target") -> pd.DataFrame:
    """Previous-day target statistics per (stock, second); no same-day information."""
    output = frame.copy()
    date = output["date_id"].to_numpy().astype("int64")
    stock = output["stock_id"].astype("int16").to_numpy().astype("int64")
    second = (output["seconds_in_bucket"].to_numpy() // 10).astype("int64")
    n_days = int(date.max()) + 1

    target = np.full((n_days, N_STOCKS, N_SECONDS), np.nan)
    target[date, stock, second] = output[target_column].to_numpy(dtype="float64")
    valid = np.isfinite(target)
    values = np.where(valid, target, 0.0)

    def exclusive_cumsum(array: np.ndarray) -> np.ndarray:
        out = np.zeros((n_days + 1,) + array.shape[1:])
        out[1:] = np.cumsum(array, axis=0)
        return out[:-1]  # out[d] = sum over days < d

    c_sum = exclusive_cumsum(values)
    c_sq = exclusive_cumsum(values ** 2)
    c_abs = exclusive_cumsum(np.abs(values))
    c_n = exclusive_cumsum(valid.astype("float64"))

    def lookup(array: np.ndarray) -> np.ndarray:
        return array[date, stock, second].astype("float32")

    with np.errstate(invalid="ignore", divide="ignore"):
        for w in HIST_WINDOWS:
            n = _trailing(c_n, w)
            output[f"hist_tgt_mean_{w}d"] = lookup(np.where(n > 0, _trailing(c_sum, w) / n, np.nan))
        for w in HIST_STD_WINDOWS:
            n = _trailing(c_n, w)
            mean = _trailing(c_sum, w) / n
            var = _trailing(c_sq, w) / n - mean ** 2
            output[f"hist_tgt_std_{w}d"] = lookup(np.where(n > 1, np.sqrt(np.clip(var, 0, None)), np.nan))
        n = _trailing(c_n, 20)
        output["hist_abs_tgt_mean_20d"] = lookup(np.where(n > 0, _trailing(c_abs, 20) / n, np.nan))
        # whole-day level: average over all seconds of the stock's previous days
        day_sum, day_n = c_sum.sum(axis=2, keepdims=True), c_n.sum(axis=2, keepdims=True)
        for w in HIST_DAY_WINDOWS:
            n = _trailing(day_n, w)
            level = np.where(n > 0, _trailing(day_sum, w) / n, np.nan)
            output[f"hist_day_tgt_mean_{w}d"] = level[date, stock, 0].astype("float32")
    return output


def add_intraday_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rolling realised relative returns within the stock-day (needs index features)."""
    output = frame.copy()
    for k in INTRA_WINDOWS:
        output[f"rel_ret60_mean_{k}r"] = stock_day_rolling(output, "rel_return_60s", k, "mean")
    output["rel_ret60_std_12r"] = stock_day_rolling(output, "rel_return_60s", 12, "std")
    first_wap = output.groupby(["stock_id", "date_id"], observed=True, sort=False)["wap"].transform("first")
    first_index = output.groupby(TIME_KEYS[:1], sort=False)["index_wap"].transform("first")
    output["rel_return_since_open"] = (
        output["wap"] / first_wap - output["index_wap"] / first_index
    ).astype("float32")
    return output
