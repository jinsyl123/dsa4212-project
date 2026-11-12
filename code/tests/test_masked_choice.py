"""Masked-choice packer (report §5.5, check 2). Run: python -m unittest tests.test_masked_choice"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from obpevo import core  # noqa: E402
from masked_choice import best_fit, pack_masked  # noqa: E402


class TestMaskedChoice(unittest.TestCase):
    def setUp(self):
        self.items = np.random.default_rng(0).integers(1, 70, size=300)

    def test_best_fit_unchanged(self):
        # Best Fit never prefers an empty bin while a used one fits, so masking changes nothing
        self.assertEqual(pack_masked(best_fit, self.items, 100), core.pack_funsearch(best_fit, self.items, 100))

    def test_empty_bin_rule_becomes_anyfit(self):
        # Worst Fit always prefers an empty bin; masked, it picks the emptiest *used* bin, i.e. AnyFit Worst Fit
        worst_fit = lambda item, bins: bins
        self.assertEqual(pack_masked(worst_fit, self.items, 100), core.pack_anyfit(worst_fit, self.items, 100))
        self.assertGreater(core.pack_funsearch(worst_fit, self.items, 100), pack_masked(worst_fit, self.items, 100))

    def test_rule_still_sees_empty_bins(self):
        # a rule that reads the capacity from the largest bin gets the same input as in the search
        seen = []
        pack_masked(lambda item, bins: (seen.append(bins.max()), -bins)[1], self.items[:20], 100)
        self.assertTrue(all(m == 100 for m in seen))


if __name__ == "__main__":
    unittest.main()
