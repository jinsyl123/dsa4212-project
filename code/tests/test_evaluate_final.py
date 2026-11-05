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

    def test_l2_columns(self):
        df = evaluate_final.add_l2(self.df, self.sets)
        self.assertFalse(df[["lb_l2", "excess_l2"]].isna().any().any())
        self.assertTrue((df.lb_l2 >= df.lb).all())                            # L2 is never below L1
        self.assertTrue((df.excess_l2 <= df.excess).all())
        self.assertIn("mean_excess_l2", evaluate_final.summarise(df).columns)
        self.assertNotIn("mean_excess_l2", self.summ.columns)                 # optional: old per-instance files still work

    def test_next_fit_is_worse_than_best_fit(self):
        # A failure is never better than Best Fit (Next Fit is not a strict floor: see _failed_row)
        for i in self.sets["t"]:
            self.assertGreaterEqual(baselines.pack_next_fit(i["items"], 100),
                                    core.pack_anyfit(baselines.best_fit, i["items"], 100))


class TestL2Table(unittest.TestCase):
    """summarise.py reports excess over L2 on every set, plus an L1 table only for sets where L1 differs."""

    def test_main_table_on_l2_and_l1_table_only_for_large_item_sets(self):
        import json
        import tempfile

        import pandas as pd
        import summarise

        base = dict(n_instances=2, n_items=500, capacity=100, dist="weibull", shape=3)
        sets = {"normal": instances.generate_set("normal", {**base, "scale": 45}, base_seed=0),
                "big": instances.generate_set("big", {**base, "scale": 65}, base_seed=0)}
        df = evaluate_final.add_l2(pd.DataFrame(evaluate_final.run_baselines(sets)), sets)
        cfg = dict(experiment="x", budget="b", methods=[{"label": "eoh"}], final_eval={"sets": ["normal", "big"]})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "eoh" / "seed0").mkdir(parents=True)
            (out / "eoh" / "seed0" / "summary.json").write_text(json.dumps(dict(
                label="eoh", prompt_tokens=1, completion_tokens=1, n_evals=1, n_ok=1, n_error=0, n_timeout=0,
                n_llm_calls=1, cost_usd=0, wall_minutes=0, best_train_excess=0.04)))
            evaluate_final.summarise(df).to_csv(out / "final_summary.csv", index=False)
            text = summarise.tables(out, cfg, df)
        main, l1 = text.split("excess over L2 lower bound")[1].split("excess over L1")
        self.assertIn("| normal | big |", main)        # main table: every set, on L2
        self.assertNotIn("normal", l1)                  # L1 table: only the set where L1 differs
        self.assertIn("| big |", l1)
        bf = df[(df.label == "best_fit") & (df.set == "big")]
        row = lambda t: t.split("best_fit")[1].split("\n")[0]  # noqa: E731
        self.assertIn(f"{100 * bf.excess_l2.mean():.2f}", row(main))
        self.assertIn(f"{100 * bf.excess.mean():.2f}", row(l1))


if __name__ == "__main__":
    unittest.main()
