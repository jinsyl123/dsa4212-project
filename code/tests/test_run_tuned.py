"""End-to-end check that a tuned Best Fit run writes the same files as an LLM run. Needs llm4ad installed
(run_experiment imports it). Run: python -m unittest discover tests"""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from obpevo import baselines, config as C, core, instances  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("llm4ad"), "llm4ad not installed")
class TestRunTuned(unittest.TestCase):
    def test_tuned_run_outputs(self):
        import run_experiment
        spec = dict(n_instances=3, n_items=[200, 400], capacity=100, dist="weibull", shape=3, scale=45)
        train = instances.generate_set("t", spec, base_seed=0)
        cfg = C.load("configs/smoke.yaml", budget={"max_evals": 6, "max_tokens": None, "max_minutes": 5})
        m = dict(name="tuned_bf", label="tuned_bf", params={})
        with tempfile.TemporaryDirectory() as tmp:
            s = run_experiment.run_tuned(cfg, m, 0, train, Path(tmp))
            run = Path(tmp) / "tuned_bf" / "seed0"
            ev, calls = pd.read_csv(run / "evals.csv"), pd.read_csv(run / "llm_calls.csv")
            self.assertEqual(list(ev["eval"]), [1, 2, 3, 4, 5, 6])
            with open(run / "evals.csv", newline="") as f:          # pandas would pad a short row with NaN
                header, *rows = list(csv.reader(f))
            self.assertEqual(header, run_experiment.EvalRecorder.FIELDS)
            self.assertTrue(all(len(r) == len(header) for r in rows))
            self.assertEqual((s["n_flagged"], s["best_integrity"]), (0, ""))  # the tuned rule breaks no rule
            self.assertTrue((ev.status == "ok").all())
            self.assertEqual(len(calls), 0)
            self.assertEqual(json.loads((run / "summary.json").read_text()), s)
            self.assertEqual((s["kind"], s["stop_reason"], s["n_evals"], s["n_llm_calls"], s["cost_usd"]),
                             ("tuned", "budget:evals", 6, 0, 0.0))
            bf = -core.score(core.evaluate(baselines.best_fit, train))
            self.assertAlmostEqual(ev.train_excess.iloc[0], bf)              # eval 1 is plain Best Fit
            self.assertLessEqual(s["best_train_excess"], bf)                  # so tuning can only match or beat it
            self.assertEqual(set(s["best_params"]), {"lo", "hi", "penalty"})
            ns = {}                                                           # best_program.py is runnable as saved
            exec((run / "best_program.py").read_text(), ns)
            self.assertAlmostEqual(-core.score(core.evaluate(ns["priority"], train)), s["best_train_excess"])

    def test_bad_params_raise_before_writing(self):
        import run_experiment
        spec = dict(n_instances=2, n_items=[200, 300], capacity=100, dist="weibull", shape=3, scale=45)
        train = instances.generate_set("t", spec, base_seed=0)
        cfg = C.load("configs/smoke.yaml", budget={"max_evals": 5, "max_tokens": None, "max_minutes": 5})
        for params, err in [({"hi_max": 30}, TypeError), ({"space": {"hi_maxx": 30}}, ValueError)]:
            with self.subTest(params=params), tempfile.TemporaryDirectory() as tmp:
                m = dict(name="tuned_bf", label="tuned_bf", params=params)
                with self.assertRaises(err):
                    run_experiment.run_tuned(cfg, m, 0, train, Path(tmp))
                self.assertFalse((Path(tmp) / "tuned_bf").exists())   # no summary.json to be skipped as done


if __name__ == "__main__":
    unittest.main()
