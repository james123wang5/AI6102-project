import unittest

import numpy as np
import pandas as pd

from ai6102.features.pipeline import build_feature_frame


def make_panel() -> pd.DataFrame:
    rows = []
    for date_id in (0, 1):
        for seconds in (0, 10, 20, 30):
            for stock_id in (0, 1):
                rows.append(
                    {
                        "stock_id": stock_id,
                        "date_id": date_id,
                        "seconds_in_bucket": seconds,
                        "bid_price": 1.0 + stock_id * 0.01 + seconds / 10000,
                        "ask_price": 1.001 + stock_id * 0.01 + seconds / 10000,
                        "bid_size": 100.0 + stock_id,
                        "ask_size": 80.0 + stock_id,
                        "wap": 1.0005 + stock_id * 0.01 + seconds / 10000,
                        "reference_price": 1.0004 + stock_id * 0.01,
                        "matched_size": 1000.0,
                        "imbalance_size": 100.0,
                        "imbalance_buy_sell_flag": 1,
                        "near_price": np.nan if seconds == 0 else 1.0007,
                        "far_price": np.nan if seconds == 0 else 1.0008,
                    }
                )
    return pd.DataFrame(rows)


class FeaturePipelineTests(unittest.TestCase):
    def test_exact_lag_resets_each_day(self) -> None:
        features = build_feature_frame(make_panel())
        first_row_day_two = features.query("date_id == 1 and seconds_in_bucket == 0 and stock_id == 0").iloc[0]
        self.assertTrue(np.isnan(first_row_day_two["wap_lag_10s"]))

    def test_future_change_does_not_affect_past_features(self) -> None:
        before = make_panel()
        after = before.copy()
        after.loc[(after["date_id"] == 0) & (after["seconds_in_bucket"] == 30) & (after["stock_id"] == 0), "wap"] = 9.0
        feature_before = build_feature_frame(before)
        feature_after = build_feature_frame(after)
        earlier = (feature_before["date_id"] == 0) & (feature_before["seconds_in_bucket"] == 20) & (feature_before["stock_id"] == 0)
        compare_columns = ["wap_lag_10s", "wap_return_10s", "wap_rolling_rows_3_mean"]
        for column in compare_columns:
            self.assertEqual(feature_before.loc[earlier, column].iloc[0], feature_after.loc[earlier, column].iloc[0])

    def test_cross_section_rank_uses_current_time_slice(self) -> None:
        features = build_feature_frame(make_panel())
        values = features.loc[(features["date_id"] == 0) & (features["seconds_in_bucket"] == 10), "price_spread_cs_rank"]
        self.assertEqual(sorted(values.tolist()), [0.75, 0.75])

    def test_duplicate_keys_are_rejected(self) -> None:
        duplicated = pd.concat([make_panel(), make_panel().iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_feature_frame(duplicated)
