"""Tests for the tuned Best Fit baseline. Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from obpevo import baselines, core, instances, tuned_bf  # noqa: E402


def compile_priority(src):
    ns = {}
    exec(src, ns)
    return ns["priority"]


class TestTunedBestFit(unittest.TestCase):
    def setUp(self):
        spec = dict(n_instances=3, n_items=[200, 400], capacity=100, dist="weibull", shape=3, scale=45)
        self.insts = instances.generate_set("t", spec, base_seed=0)

    def test_zero_knobs_is_best_fit(self):
        bf = compile_priority(tuned_bf.program(**tuned_bf.BEST_FIT))
        for pack in (core.pack_funsearch, core.pack_anyfit):
            for i in self.insts:
                self.assertEqual(pack(bf, i["items"], 100), pack(baselines.best_fit, i["items"], 100))

    def test_sliver_penalty_hand_checked(self):
        # item 30 into bins with 45 / 70 / 100 left: gaps 15 / 40 / 70. Best Fit takes the 15 gap;
        # with lo=2, hi=20, penalty=50 that gap is a sliver (score -65), so the 40 gap (score -40) wins.
        bins = np.array([45, 70, 100])
        self.assertEqual(np.argmax(compile_priority(tuned_bf.program(0, 0, 0))(30, bins)), 0)
        self.assertEqual(np.argmax(compile_priority(tuned_bf.program(2, 20, 50))(30, bins)), 1)
        # a gap of at most lo is a tight fit, not a sliver: 32 left -> gap 2 is kept
        self.assertEqual(np.argmax(compile_priority(tuned_bf.program(2, 20, 50))(30, np.array([32, 70]))), 0)

    def test_sample_starts_with_best_fit_and_is_deterministic(self):
        a, b, c = tuned_bf.sample(50, 0), tuned_bf.sample(50, 0), tuned_bf.sample(50, 1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a[0], tuned_bf.BEST_FIT)

    def test_sample_distinct_and_in_range(self):
        s, sp = tuned_bf.sample(300, 3), tuned_bf.SPACE
        rest = s[1:]
        self.assertEqual(len(s), 300)
        self.assertEqual(len({(k["lo"], k["hi"], k["penalty"]) for k in rest}), 299)
        for k in rest:
            self.assertTrue(sp["lo"][0] <= k["lo"] <= sp["lo"][1])
            self.assertTrue(k["lo"] + 2 <= k["hi"] <= sp["hi_max"])
            self.assertTrue(sp["penalty"][0] <= k["penalty"] <= sp["penalty"][1])

    def test_sample_larger_than_space_raises(self):
        tiny = dict(lo=(0, 0), hi_max=3, penalty=(1, 2))     # hi in {2, 3} x penalty in {1, 2} = 4 settings
        self.assertEqual(tuned_bf.space_size(tiny), 4)
        self.assertEqual(len(tuned_bf.sample(5, 0, tiny)), 5)
        with self.assertRaises(ValueError):
            tuned_bf.sample(6, 0, tiny)

    def test_tune_partial_space_override(self):
        recs = list(tuned_bf.tune(self.insts, seed=0, max_evals=20, space={"hi_max": 12, "penalty": [5, 5]}))
        self.assertTrue(all(r["params"]["hi"] <= 12 and r["params"]["penalty"] == 5 for r in recs[1:]))
        self.assertTrue(all(r["params"]["lo"] <= tuned_bf.SPACE["lo"][1] for r in recs[1:]))   # default kept

    def test_tune_bad_params_raise_before_scoring(self):
        # both raise on the call itself, without iterating: nothing is scored with a misread config
        with self.assertRaises(ValueError):
            tuned_bf.tune(self.insts, seed=0, max_evals=5, space={"hi_maxx": 30})     # typo inside space:
        with self.assertRaises(TypeError):
            tuned_bf.tune(self.insts, seed=0, max_evals=5, hi_max=30)                # range key outside space:

    def test_tune_scores_best_fit_first(self):
        recs = list(tuned_bf.tune(self.insts, seed=0, max_evals=5))
        self.assertEqual([r["eval"] for r in recs], [1, 2, 3, 4, 5])
        self.assertTrue(all(r["status"] == "ok" for r in recs))
        self.assertEqual(recs[0]["params"], tuned_bf.BEST_FIT)
        self.assertEqual(core.score(recs[0]["rows"]), core.score(core.evaluate(baselines.best_fit, self.insts)))


if __name__ == "__main__":
    unittest.main()
