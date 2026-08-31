"""Small, explicit operators shared by causal feature modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


STOCK_DAY_KEYS = ["stock_id", "date_id"]
TIME_KEYS = ["date_id", "seconds_in_bucket"]
ROW_KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


def safe_ratio(numerator: pd.Series, denominator: pd.Series, minimum_denominator: float = 1e-12) -> pd.Series:
    """Divide only when a finite denominator has sufficient absolute magnitude."""
    valid = denominator.notna() & (denominator.abs() > minimum_denominator)
    result = pd.Series(np.nan, index=numerator.index, dtype="float32")
    result.loc[valid] = (numerator.loc[valid] / denominator.loc[valid]).astype("float32")
    return result


def exact_time_lag(frame: pd.DataFrame, column: str, lag_seconds: int) -> pd.Series:
    """Return an exact past-time value, never a previous-row approximation."""
    lookup = frame.loc[:, ROW_KEYS + [column]].copy()
    lookup["seconds_in_bucket"] = lookup["seconds_in_bucket"] + lag_seconds
    lookup = lookup.rename(columns={column: "_lag_value"})
    merged = frame.loc[:, ROW_KEYS].merge(lookup, on=ROW_KEYS, how="left", sort=False)
    return merged["_lag_value"].set_axis(frame.index).astype("float32")


def stock_day_rolling(frame: pd.DataFrame, column: str, window_rows: int, statistic: str) -> pd.Series:
    """Causal rolling statistic within a stock and a trading day.

    The window unit is explicitly rows, not seconds. The caller must provide a
    frame sorted by stock, date, and seconds.
    """
    grouped = frame.groupby(STOCK_DAY_KEYS, observed=True, sort=False)[column]
    rolling = grouped.rolling(window_rows, min_periods=1)
    if statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std(ddof=0)
    else:
        raise ValueError(f"Unsupported rolling statistic: {statistic}")
    return result.reset_index(level=STOCK_DAY_KEYS, drop=True).astype("float32")


def rolling_last_rank(frame: pd.DataFrame, column: str, window_rows: int) -> pd.Series:
    """Percentile rank of the newest value in its causal stock-day row window."""
    def last_rank(values: np.ndarray) -> float:
        current = values[-1]
        finite = values[np.isfinite(values)]
        if not np.isfinite(current) or len(finite) == 0:
            return np.nan
        return float((finite <= current).mean())

    result = frame.groupby(STOCK_DAY_KEYS, observed=True, sort=False)[column].rolling(
        window_rows, min_periods=1
    ).apply(last_rank, raw=True)
    return result.reset_index(level=STOCK_DAY_KEYS, drop=True).astype("float32")


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
