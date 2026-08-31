"""Causal within-day temporal features."""

from __future__ import annotations

import pandas as pd

from ai6102.features.operators import exact_time_lag, rolling_last_rank, safe_ratio, stock_day_rolling


def add_temporal_features(
    frame: pd.DataFrame,
    exact_lag_seconds: tuple[int, ...] = (10, 30, 60),
    rolling_windows_rows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Add exact-time changes and explicitly row-based causal rolling features."""
    output = frame.copy()
    for column in ("wap", "price_spread", "liquidity_imbalance", "signed_imbalance"):
        for lag_seconds in exact_lag_seconds:
            lagged = exact_time_lag(output, column, lag_seconds)
            output[f"{column}_lag_{lag_seconds}s"] = lagged
            output[f"{column}_diff_{lag_seconds}s"] = (output[column] - lagged).astype("float32")
            if column == "wap":
                output[f"wap_return_{lag_seconds}s"] = safe_ratio(output[column] - lagged, lagged)

    for window_rows in rolling_windows_rows:
        output[f"wap_rolling_rows_{window_rows}_mean"] = stock_day_rolling(output, "wap", window_rows, "mean")
        output[f"wap_rolling_rows_{window_rows}_std"] = stock_day_rolling(output, "wap", window_rows, "std")
        output[f"liquidity_imbalance_rolling_rows_{window_rows}_mean"] = stock_day_rolling(
            output, "liquidity_imbalance", window_rows, "mean"
        )
    return output
