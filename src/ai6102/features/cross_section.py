"""Features using the stock cross-section available at one observed time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai6102.features.operators import TIME_KEYS


def add_cross_section_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add rank and relative-to-cross-section features for selected signals."""
    output = frame.copy()
    group = output.groupby(TIME_KEYS, observed=True, sort=False)
    for column in ("price_spread", "liquidity_imbalance", "market_urgency", "wap_return_10s"):
        if column not in output:
            continue
        output[f"{column}_cs_rank"] = group[column].rank(pct=True).astype("float32")
        mean = group[column].transform("mean")
        std = group[column].transform("std")
        output[f"{column}_cs_mean_diff"] = (output[column] - mean).astype("float32")
        valid_std = std.notna() & (std.abs() > 1e-12)
        zscore = pd.Series(np.nan, index=output.index, dtype="float32")
        zscore.loc[valid_std] = ((output.loc[valid_std, column] - mean.loc[valid_std]) / std.loc[valid_std]).astype("float32")
        output[f"{column}_cs_zscore"] = zscore
    return output
