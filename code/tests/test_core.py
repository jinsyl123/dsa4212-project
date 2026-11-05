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


def _optimum(items, capacity) -> int:
    """Exact minimum number of bins by exhaustive search (tiny instances only)."""
    items = sorted(items, reverse=True)
    best = [len(items)]

    def place(k, bins):
        if len(bins) >= best[0]:
            return
        if k == len(items):
            best[0] = len(bins)
            return
        for b in range(len(bins)):
            if bins[b] >= items[k] and bins[b] not in bins[:b]:              # skip bins with the same space left
                bins[b] -= items[k]
                place(k + 1, bins)
                bins[b] += items[k]
        place(k + 1, bins + [capacity - items[k]])

    place(0, [])
    return best[0]


class TestLowerBoundL2(unittest.TestCase):
    def test_hand_checked_examples(self):
        self.assertEqual(core.lower_bound(np.array([60, 60, 60]), 100), 2)    # L1 misses that no two 60s share a bin
        self.assertEqual(core.lower_bound_l2(np.array([60, 60, 60]), 100), 3)
        # L2 is a bound, not the optimum: it treats the 3 x 40 gaps as 120 units of shared space, so the four 30s
        # seem to fit (L2 = 3), though each gap holds only one 30 (optimum = 4)
        big = np.array([60, 60, 60, 30, 30, 30, 30])
        self.assertEqual(core.lower_bound_l2(big, 100), 3)
        self.assertEqual(_optimum(big, 100), 4)
        self.assertEqual(core.lower_bound_l2(np.array([50, 50]), 100), 1)     # exactly half a bin may share

    def test_never_above_optimum_never_below_l1(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            cap = int(rng.choice([10, 20, 100]))
            items = rng.integers(1, cap + 1, size=int(rng.integers(1, 9)))
            l1, l2, opt = core.lower_bound(items, cap), core.lower_bound_l2(items, cap), _optimum(items, cap)
            self.assertLessEqual(l1, l2, (items, cap))
            self.assertLessEqual(l2, opt, (items, cap))

    def test_matches_martello_toth_definition(self):
        # Martello & Toth (1990), Knapsack Problems, §8.3.2: L2 = max of L(alpha) (Theorem 8.3, eq. 8.19) over every
        # integer 0 <= alpha <= c/2 (Corollary 8.3, eq. 8.20). lower_bound_l2 only tries alpha = 0 and item sizes
        # <= c/2 (their Theorem 8.4); this checks that shortcut against the full definition
        def book_l2(w, c):
            best = 0
            for a in range(c // 2 + 1):
                j1, j2, j3 = w[w > c - a], w[(w <= c - a) & (w > c / 2)], w[(w <= c / 2) & (w >= a)]
                best = max(best, len(j1) + len(j2) + max(0, int(np.ceil((j3.sum() - (len(j2) * c - j2.sum())) / c))))
            return best

        rng = np.random.default_rng(1)
        for _ in range(1000):
            cap = int(rng.choice([10, 50, 100, 150, 200]))
            items = rng.integers(1, cap + 1, size=int(rng.integers(1, 60)))
            self.assertEqual(core.lower_bound_l2(items, cap), book_l2(items, cap), (items, cap))
        for cap, lo, hi in [(100, 1, 101), (100, 51, 101), (150, 20, 101)]:          # our set sizes: 5000 items
            items = rng.integers(lo, hi, size=5000)
            self.assertEqual(core.lower_bound_l2(items, cap), book_l2(items, cap), cap)

    def test_large_items_much_tighter_than_l1(self):
        # Weibull scale 65: ~63% of items exceed half a bin, so L1 is far too low
        spec = dict(n_instances=2, n_items=2000, capacity=100, dist="weibull", shape=3, scale=65)
        for i in instances.generate_set("big", spec, base_seed=0):
            l1, l2 = core.lower_bound(i["items"], 100), core.lower_bound_l2(i["items"], 100)
            bf = core.pack_anyfit(baselines.best_fit, i["items"], 100)
            self.assertGreater(l2, 1.05 * l1)
            self.assertLessEqual(l2, bf)

    def test_equals_l1_on_training_distribution(self):
        # Weibull(3, 45): few large items, so L2 adds nothing and existing scores are unchanged
        for i in instances.generate_set("t", dict(n_instances=3, n_items=1000, capacity=100), base_seed=0):
            self.assertEqual(core.lower_bound_l2(i["items"], 100), core.lower_bound(i["items"], 100))


if __name__ == "__main__":
    unittest.main()
