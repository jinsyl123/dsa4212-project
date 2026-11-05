"""Paired comparison against reference rules. Run: python -m unittest tests.test_paired_stats"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import paired_stats as P  # noqa: E402


def _rows(label, kind, seed, excess, set_name="s", status="ok"):
    return [dict(kind=kind, label=label, seed=seed, set=set_name, instance=f"i{k}", excess=e, excess_l2=e / 2,
                 status=status) for k, e in enumerate(excess)]


class TestSignTest(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(P.sign_test(5, 0), 0.0625)          # best case with 5 instances: never < 0.05
        self.assertAlmostEqual(P.sign_test(10, 0), 2 / 1024)
        self.assertAlmostEqual(P.sign_test(0, 10), 2 / 1024)       # symmetric
        self.assertEqual(P.sign_test(3, 3), 1.0)
        self.assertEqual(P.sign_test(0, 0), 1.0)                   # all ties: no evidence either way
        self.assertAlmostEqual(P.sign_test(8, 2), 0.109375)        # 2 * (1 + 10 + 45) / 1024


class TestBootstrap(unittest.TestCase):
    def test_constant_and_deterministic(self):
        self.assertEqual(P.bootstrap_ci(np.full(5, -1.3)), (-1.3, -1.3))
        d = np.array([-2.0, -1.0, 0.5, -1.5, -0.5])
        self.assertEqual(P.bootstrap_ci(d), P.bootstrap_ci(d))
        lo, hi = P.bootstrap_ci(d)
        self.assertTrue(lo <= d.mean() <= hi)


class TestCompare(unittest.TestCase):
    def setUp(self):
        bf = [0.040, 0.042, 0.041, 0.043, 0.039, 0.044, 0.038, 0.041, 0.040, 0.042]   # 10 instances
        self.df = pd.DataFrame(
            _rows("best_fit", "baseline", -1, bf)
            + _rows("eoh", "evolved", 0, bf)                                     # Best Fit rewritten
            + _rows("eoh", "evolved", 1, [e - 0.013 for e in bf])                # better on every instance
            + _rows("eoh", "evolved", 2, [e + 0.02 for e in bf], status="error: crash")
            + _rows("eoh", "evolved", 3, [0.030, 0.050, 0.035, 0.050, 0.039] * 2)    # mixed
            + _rows("tuned_bf", "tuned", 0, [e - 0.02 for e in bf])
            + _rows("tuned_bf", "tuned", 1, [e - 0.03 for e in bf]))             # reference averaged: -2.5 pp

    def _row(self, res, seed, ref, label="eoh"):
        return res[(res.label == label) & (res.seed == seed) & (res.reference == ref)].iloc[0]

    def test_verdicts_vs_best_fit(self):
        res = P.compare(self.df, n_boot=2000)
        r = self._row(res, 0, "best_fit")
        self.assertEqual((r.verdict, r.wins, r.ties, r.losses, r.sign_p), ("tie", 0, 10, 0, 1.0))
        r = self._row(res, 1, "best_fit")
        self.assertEqual((r.verdict, r.wins, r.losses), ("better", 10, 0))
        self.assertAlmostEqual(r.mean_diff_pp, -1.3)
        self.assertAlmostEqual(r.sign_p, 2 / 1024)
        r = self._row(res, 2, "best_fit")
        self.assertEqual((r.verdict, r.failed), ("worse", 10))                   # failures stay in, as Next Fit would
        self.assertEqual(self._row(res, 3, "best_fit").verdict, "unclear")

    def test_multi_seed_reference_is_averaged(self):
        res = P.compare(self.df, n_boot=2000)
        r = self._row(res, 1, "tuned_bf")                                         # -1.3 vs -2.5 -> +1.2 pp
        self.assertAlmostEqual(r.mean_diff_pp, 1.2)
        self.assertEqual(r.verdict, "worse")
        self.assertFalse(((res.label == "tuned_bf") & (res.reference == "tuned_bf")).any())

    def test_five_instances_never_decisive(self):
        # 5-0 has p = 0.0625: the interval excludes 0, but a verdict also needs p < 0.05
        five = self.df[self.df.instance.isin([f"i{k}" for k in range(5)])]
        r = self._row(P.compare(five, n_boot=2000), 1, "best_fit")
        self.assertTrue(r.ci_high_pp < 0)
        self.assertEqual(r.verdict, "unclear")
        r = self._row(P.compare(five, n_boot=2000, alpha=0.1), 1, "best_fit")
        self.assertEqual(r.verdict, "better")

    def test_l2_metric(self):
        res = P.compare(self.df, n_boot=500, metric="excess_l2")
        self.assertAlmostEqual(self._row(res, 1, "best_fit").mean_diff_pp, -0.65)   # excess_l2 = excess / 2 here
        self.assertEqual(set(res.metric), {"excess_l2"})
        with self.assertRaises(KeyError):
            P.compare(self.df.drop(columns="excess_l2"), metric="excess_l2")

    def test_default_metric_is_l2_when_present(self):
        self.assertEqual(P.default_metric(self.df), "excess_l2")
        self.assertEqual(P.default_metric(self.df.drop(columns="excess_l2")), "excess")   # older results

    def test_missing_reference_skipped(self):
        res = P.compare(self.df[self.df.label != "tuned_bf"], n_boot=500)
        self.assertEqual(set(res.reference), {"best_fit"})

    def test_sets_kept_apart(self):
        df = pd.concat([self.df, pd.DataFrame(_rows("best_fit", "baseline", -1, [0.1] * 10, set_name="t")
                                              + _rows("eoh", "evolved", 0, [0.2] * 10, set_name="t"))])
        res = P.compare(df, references=["best_fit"], n_boot=500)
        self.assertEqual(self._row(res[res.set == "s"], 0, "best_fit").verdict, "tie")
        self.assertEqual(self._row(res[res.set == "t"], 0, "best_fit").verdict, "worse")


if __name__ == "__main__":
    unittest.main()
