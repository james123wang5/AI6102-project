"""Whole-market aggregates over the stocks observed at one time instant.

These use only the same (date_id, seconds_in_bucket) cross-section, so they
need no fitted state and can be cached together with the other features.
"""

from __future__ import annotations

import pandas as pd

from ai6102.features.operators import TIME_KEYS, safe_ratio


MARKET_COLUMNS = [
    "mkt_wap_return_10s_mean",
    "mkt_wap_return_10s_std",
    "mkt_wap_return_60s_mean",
    "mkt_liquidity_imbalance_mean",
    "mkt_price_spread_mean",
    "mkt_auction_imbalance_ratio_mean",
    "mkt_signed_imbalance_ratio",
]


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add equal-weight market state features shared by every stock at a time."""
    output = frame.copy()
    group = output.groupby(TIME_KEYS, observed=True, sort=False)
    output["mkt_wap_return_10s_mean"] = group["wap_return_10s"].transform("mean").astype("float32")
    output["mkt_wap_return_10s_std"] = group["wap_return_10s"].transform("std").astype("float32")
    output["mkt_wap_return_60s_mean"] = group["wap_return_60s"].transform("mean").astype("float32")
    output["mkt_liquidity_imbalance_mean"] = group["liquidity_imbalance"].transform("mean").astype("float32")
    output["mkt_price_spread_mean"] = group["price_spread"].transform("mean").astype("float32")
    output["mkt_auction_imbalance_ratio_mean"] = group["auction_imbalance_ratio"].transform("mean").astype("float32")
    output["mkt_signed_imbalance_ratio"] = safe_ratio(
        group["signed_imbalance"].transform("sum"), group["matched_size"].transform("sum")
    )
    return output
