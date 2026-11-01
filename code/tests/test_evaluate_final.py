"""Failed programs must count in the held-out means (as Next Fit), not drop out as NaN.
Run: python -m unittest tests.test_evaluate_final"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import evaluate_final  # noqa: E402
from obpevo import baselines, core, instances  # noqa: E402

HEADER = "import numpy as np\n\ndef priority(item, bins):\n"
PROGRAMS = {
    "ok": HEADER + "    return item - bins\n",                       # Best Fit
    "crash": HEADER + "    raise ValueError('boom')\n",
    "nan": HEADER + "    return np.full(len(bins), np.nan)\n",        # rejected by core._checked
    "hang": HEADER + "    while True:\n        pass\n",
}


class TestFailedPrograms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = dict(n_instances=3, n_items=200, capacity=100, dist="weibull", shape=3, scale=45)
        cls.sets = {"t": instances.generate_set("t", spec, base_seed=0)}
        jobs = [dict(kind="evolved", label=k, method="eoh", seed=0, set="t", program=p) for k, p in PROGRAMS.items()]
        rows = evaluate_final.run_baselines(cls.sets) + evaluate_final.run_programs(
            jobs, cls.sets, "funsearch", timeout=3, workers=4)
        import pandas as pd
        cls.df = pd.DataFrame(rows)
        cls.summ = evaluate_final.summarise(cls.df).set_index("label")

    def test_failures_scored_as_next_fit(self):
        nf = {i["id"]: baselines.pack_next_fit(i["items"], i["capacity"]) for i in self.sets["t"]}
        for label, status in (("crash", "error"), ("nan", "error"), ("hang", "timeout")):
            rows = self.df[self.df.label == label]
            with self.subTest(label=label):
                self.assertEqual(len(rows), 3)
                self.assertTrue(rows.status.str.startswith(status).all())
                self.assertFalse(rows.excess.isna().any())
                self.assertEqual(list(rows.bins), [nf[i] for i in rows.instance])

    def test_failures_counted_in_summary(self):
        for label in ("crash", "nan", "hang"):
            self.assertEqual(self.summ.loc[label, "failed_frac"], 1.0)
            self.assertAlmostEqual(self.summ.loc[label, "mean_excess"], self.summ.loc["next_fit", "mean_excess"], 5)
        self.assertEqual(self.summ.loc["ok", "failed_frac"], 0.0)
        self.assertAlmostEqual(self.summ.loc["ok", "mean_excess"], self.summ.loc["best_fit", "mean_excess"], 5)

    def test_next_fit_is_worse_than_best_fit(self):
        # A failure is never better than Best Fit (Next Fit is not a strict floor: see _failed_row)
        for i in self.sets["t"]:
            self.assertGreaterEqual(baselines.pack_next_fit(i["items"], 100),
                                    core.pack_anyfit(baselines.best_fit, i["items"], 100))


if __name__ == "__main__":
    unittest.main()
