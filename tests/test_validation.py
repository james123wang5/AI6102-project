import unittest

import pandas as pd

from ai6102.validation import DateFold


class ValidationTests(unittest.TestCase):
    def test_fold_masks_are_chronological(self) -> None:
        fold = DateFold("example", 0, 9, 10, 12)
        dates = pd.Series([0, 9, 10, 12, 13])
        self.assertEqual(fold.train_mask(dates).tolist(), [True, True, False, False, False])
        self.assertEqual(fold.validation_mask(dates).tolist(), [False, False, True, True, False])
        self.assertEqual(fold.inner_validation_start(2), 8)

    def test_fold_rejects_time_reversal(self) -> None:
        with self.assertRaises(ValueError):
            DateFold("bad", 0, 10, 10, 20)
