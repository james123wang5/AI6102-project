"""Features whose state is fitted on a fold's training dates only.

- Index features: need the static index weights (see `index_model`).
- Stock statistics: per-stock medians of sizes, spread and |target|.

Both are fitted on the training part and then applied unchanged to the
evaluation dates, so no evaluation information enters the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai6102.features.operators import TIME_KEYS, safe_ratio
from ai6102.index_model import N_STOCKS, weighted_time_mean


INDEX_LAGS = (10, 30, 60)
INDEX_COLUMNS = (
    ["index_weight", "index_wap", "wap_minus_index", "index_liquidity_imbalance", "rel_liquidity_imbalance"]
    + [f"index_return_{k}s" for k in INDEX_LAGS]
    + [f"rel_return_{k}s" for k in INDEX_LAGS]
)
STOCK_STAT_SOURCES = ["bid_size", "ask_size", "matched_size", "imbalance_size", "price_spread"]
STOCK_COLUMNS = [f"{c}_rel_stock_median" for c in STOCK_STAT_SOURCES] + [
    "stock_median_price_spread",
    "stock_target_abs_median",
]


def add_index_features(frame: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Index level/returns from current and past WAPs, and stock-vs-index gaps."""
    output = frame.copy()
    stock = output["stock_id"].astype("int16").to_numpy()
    output["index_weight"] = weights[stock].astype("float32")
    output["index_wap"] = weighted_time_mean(output, output["wap"], weights).astype("float32")
    output["index_liquidity_imbalance"] = weighted_time_mean(
        output, output["liquidity_imbalance"], weights
    ).astype("float32")

    per_time = output.loc[:, TIME_KEYS + ["index_wap"]].drop_duplicates(TIME_KEYS)
    for lag in INDEX_LAGS:
        past = per_time.assign(seconds_in_bucket=per_time["seconds_in_bucket"] + lag)
        past = past.rename(columns={"index_wap": "_past_index_wap"})
        joined = output.loc[:, TIME_KEYS].merge(past, on=TIME_KEYS, how="left", sort=False)
        past_wap = joined["_past_index_wap"].to_numpy()
        output[f"index_return_{lag}s"] = safe_ratio(
            output["index_wap"] - past_wap, pd.Series(past_wap, index=output.index)
        )
        output[f"rel_return_{lag}s"] = (output[f"wap_return_{lag}s"] - output[f"index_return_{lag}s"]).astype("float32")

    output["wap_minus_index"] = (output["wap"] - output["index_wap"]).astype("float32")
    output["rel_liquidity_imbalance"] = (
        output["liquidity_imbalance"] - output["index_liquidity_imbalance"]
    ).astype("float32")
    return output


def fit_stock_stats(train: pd.DataFrame, target_column: str = "target") -> pd.DataFrame:
    """Per-stock medians computed from training dates only."""
    data = train.loc[:, ["stock_id", target_column] + STOCK_STAT_SOURCES].copy()
    data["stock_id"] = data["stock_id"].astype("int16")
    data["abs_target"] = data[target_column].abs()
    stats = data.groupby("stock_id")[STOCK_STAT_SOURCES + ["abs_target"]].median()
    return stats.reindex(range(N_STOCKS))


def add_stock_features(frame: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    stock = output["stock_id"].astype("int16").to_numpy()
    for column in STOCK_STAT_SOURCES:
        median = pd.Series(stats[column].to_numpy()[stock], index=output.index)
        output[f"{column}_rel_stock_median"] = safe_ratio(output[column], median)
    output["stock_median_price_spread"] = stats["price_spread"].to_numpy()[stock].astype("float32")
    output["stock_target_abs_median"] = stats["abs_target"].to_numpy()[stock].astype("float32")
    return output
