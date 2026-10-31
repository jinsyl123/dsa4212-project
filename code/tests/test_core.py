"""Sanity tests for the evaluator, baselines and instance generator. Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from obpevo import core, baselines, instances  # noqa: E402


class TestCore(unittest.TestCase):
    def setUp(self):
        spec = dict(n_instances=3, n_items=[200, 400], capacity=100, dist="weibull", shape=3, scale=45)
        self.insts = instances.generate_set("t", spec, base_seed=0)

    def test_ff_bf_identical_across_conventions(self):
        # Evolved rules run in "funsearch" mode, baselines in "anyfit": FF and BF must agree in both
        for fn in (baselines.first_fit, baselines.best_fit):
            for i in self.insts:
                self.assertEqual(core.pack_funsearch(fn, i["items"], 100), core.pack_anyfit(fn, i["items"], 100))

    def test_small_hand_checked_example(self):
        items = np.array([60, 50, 40, 30, 20])     # BF: [60,40] [50,30,20] -> 2 bins; NF: [60] [50,40] [30,20] -> 3
        self.assertEqual(core.pack_anyfit(baselines.best_fit, items, 100), 2)
        self.assertEqual(baselines.pack_next_fit(items, 100), 3)
        self.assertEqual(core.lower_bound(items, 100), 2)

    def test_sgall_opt7_example_lower_bound(self):
        # Sgall (2014) Opt = 7 instance: total size fits in 7 bins
        items = np.array([12] * 6 + [34] * 6 + [52, 1, 48] + [53] * 6)
        self.assertEqual(core.lower_bound(items, 100), 7)
        for name, pack in baselines.BASELINES.items():
            self.assertGreaterEqual(pack(items, 100), 7, name)

    def test_invalid_heuristics_rejected(self):
        items = self.insts[0]["items"]
        with self.assertRaises(core.InvalidHeuristic):
            core.pack_funsearch(lambda item, bins: np.sqrt(item - bins), items, 100)   # NaN
        with self.assertRaises(core.InvalidHeuristic):
            core.pack_funsearch(lambda item, bins: (item - bins)[:-1], items, 100)     # wrong shape

    def test_scores_and_rows(self):
        rows = core.evaluate(baselines.best_fit, self.insts)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["bins"] >= r["lb"] for r in rows))
        self.assertLessEqual(core.score(rows), 0)

    def test_generator_deterministic(self):
        spec = dict(n_instances=2, n_items=100, capacity=100)
        a, b = instances.generate_set("x", spec, 7), instances.generate_set("x", spec, 7)
        self.assertTrue(all((p["items"] == q["items"]).all() for p, q in zip(a, b)))
        c = instances.generate_set("y", spec, 7)                                      # different set name -> different data
        self.assertFalse((a[0]["items"] == c[0]["items"]).all())


if __name__ == "__main__":
    unittest.main()
