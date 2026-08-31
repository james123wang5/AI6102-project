"""Small Alpha101-inspired factors adapted to the Optiver feature interface.

These are new high-frequency hypotheses, not direct copies of daily Alpha101
formulas. Every time-window name states its unit explicitly.
"""

from __future__ import annotations

import pandas as pd

from ai6102.features.operators import rolling_last_rank, safe_ratio, stock_day_rolling


def add_alpha_adapted_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a limited, documented set of causal formula-inspired candidates."""
    output = frame.copy()
    output["alpha_wap_return_10s_ts_rank_rows_6"] = rolling_last_rank(output, "wap_return_10s", 6)
    output["alpha_liquidity_imbalance_decay_rows_3"] = (
        0.5 * output["liquidity_imbalance"]
        + 0.3 * output["liquidity_imbalance_lag_10s"]
        + 0.2 * output["liquidity_imbalance_lag_30s"]
    ).astype("float32")
    output["alpha_wap_momentum_vol_adjusted_rows_6"] = safe_ratio(
        output["wap_diff_30s"], output["wap_rolling_rows_6_std"]
    )

    rolling_corr = output.groupby(["stock_id", "date_id"], observed=True, sort=False).apply(
        lambda group: group["price_spread"].rolling(6, min_periods=3).corr(group["liquidity_imbalance"]),
        include_groups=False,
    )
    output["alpha_spread_imbalance_corr_rows_6"] = rolling_corr.reset_index(level=[0, 1], drop=True).astype("float32")
    output["alpha_signed_imbalance_mean_rows_6"] = stock_day_rolling(output, "signed_imbalance", 6, "mean")
    return output
