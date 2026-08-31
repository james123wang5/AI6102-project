import tempfile
import unittest
from pathlib import Path

from ai6102.data_audit import audit_train_csv, render_markdown_report


class DataAuditTests(unittest.TestCase):
    def test_audit_summarises_a_small_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            train_path = Path(temporary_directory) / "train.csv"
            train_path.write_text(
                "stock_id,date_id,seconds_in_bucket,imbalance_size,imbalance_buy_sell_flag,reference_price,matched_size,far_price,near_price,bid_price,bid_size,ask_price,ask_size,wap,target,time_id,row_id\n"
                "0,0,0,1.0,1,1.0,2.0,,,1.0,3.0,1.1,4.0,1.05,0.5,0,0_0_0\n"
                "1,0,10,2.0,-1,1.0,2.0,1.2,1.1,1.0,3.0,1.1,4.0,1.05,nan,1,0_1_10\n",
                encoding="utf-8",
            )
            summary = audit_train_csv(train_path)

        self.assertEqual(summary["dataset"]["row_count"], 2)
        self.assertEqual(summary["dataset"]["unique_stock_count"], 2)
        self.assertEqual(summary["numeric_columns"]["target"]["non_finite_count"], 1)
        self.assertEqual(summary["numeric_columns"]["far_price"]["missing_count"], 1)
        self.assertEqual(summary["dataset"]["duplicate_time_stock_key_count"], 0)
        self.assertIn("数据审计报告", render_markdown_report(summary))
