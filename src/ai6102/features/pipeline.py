"""Feature-pipeline entry point shared by offline research and later replay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai6102.features.alpha_adapted import add_alpha_adapted_features
from ai6102.features.cross_section import add_cross_section_features
from ai6102.features.operators import ROW_KEYS, require_columns
from ai6102.features.optiver import add_optiver_features
from ai6102.features.temporal import add_temporal_features


def _validate_unique_time_stock_keys(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(ROW_KEYS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ROW_KEYS].head(5).to_dict("records")
        raise ValueError(f"Feature pipeline requires unique time-stock keys; examples: {examples}")


def build_feature_frame(
    frame: pd.DataFrame,
    include_alpha_adapted: bool = True,
    exact_lag_seconds: tuple[int, ...] = (10, 30, 60),
    rolling_windows_rows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Build causal candidate features while preserving the input row order.

    The caller supplies only information available at the prediction instant.
    No target is read by this function.
    """
    require_columns(frame, ROW_KEYS)
    _validate_unique_time_stock_keys(frame)
    output = frame.copy()
    output["_source_order"] = np.arange(len(output), dtype="int64")
    output = output.sort_values(ROW_KEYS, kind="stable").reset_index(drop=True)
    output = add_optiver_features(output)
    output = add_temporal_features(output, exact_lag_seconds, rolling_windows_rows)
    output = add_cross_section_features(output)
    if include_alpha_adapted:
        output = add_alpha_adapted_features(output)
    output = output.sort_values("_source_order", kind="stable").drop(columns="_source_order")
    return output.reset_index(drop=True)
