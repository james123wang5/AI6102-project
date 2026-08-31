"""Explicit order-book and closing-auction features."""

from __future__ import annotations

import pandas as pd

from ai6102.features.operators import require_columns, safe_ratio


REQUIRED_COLUMNS = {
    "ask_price", "ask_size", "bid_price", "bid_size", "wap", "reference_price",
    "matched_size", "imbalance_size", "imbalance_buy_sell_flag", "near_price", "far_price",
}


def add_optiver_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable row-level microstructure features without time leakage."""
    require_columns(frame, REQUIRED_COLUMNS)
    output = frame.copy()
    output["price_spread"] = (output["ask_price"] - output["bid_price"]).astype("float32")
    output["mid_price"] = ((output["ask_price"] + output["bid_price"]) / 2).astype("float32")
    output["wap_mid_diff"] = (output["wap"] - output["mid_price"]).astype("float32")
    output["reference_wap_diff"] = (output["reference_price"] - output["wap"]).astype("float32")
    output["near_wap_diff"] = (output["near_price"] - output["wap"]).astype("float32")
    output["far_wap_diff"] = (output["far_price"] - output["wap"]).astype("float32")
    output["near_price_is_missing"] = output["near_price"].isna().astype("int8")
    output["far_price_is_missing"] = output["far_price"].isna().astype("int8")

    size_total = output["bid_size"] + output["ask_size"]
    output["liquidity_imbalance"] = safe_ratio(output["bid_size"] - output["ask_size"], size_total)
    auction_total = output["matched_size"] + output["imbalance_size"]
    output["auction_imbalance_ratio"] = safe_ratio(output["imbalance_size"], auction_total)
    output["signed_imbalance"] = (
        output["imbalance_size"] * output["imbalance_buy_sell_flag"]
    ).astype("float32")
    output["market_urgency"] = (
        output["price_spread"] * output["liquidity_imbalance"]
    ).astype("float32")
    return output
