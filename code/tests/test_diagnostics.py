"""Tests for obpevo.diagnostics: same bins as core.py, and fresh_bin_rate detects what it should.
Run: python -m unittest tests.test_diagnostics"""
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from obpevo import core, baselines, diagnostics, instances  # noqa: E402


def train_instances():
    spec = yaml.safe_load((ROOT / "configs/instances/v0.yaml").read_text())
    return instances.generate_set("train", spec["sets"]["train"], spec["seed"])


def eoh_seed0():
    ns = {}
    exec((ROOT / "results/eoh_baseline/eoh/seed0/best_program.py").read_text(), ns)
    return ns["priority"]


class TestDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = train_instances()

    def test_same_bins_as_core(self):
        # The trace must be the same packing as the scorer, in both modes
        for fn in (baselines.best_fit, baselines.first_fit, baselines.worst_fit, eoh_seed0()):
            for mode, pack in core.PACKERS.items():
                for inst in self.train:
                    with self.subTest(rule=fn.__name__, mode=mode, instance=inst["id"]):
                        t = diagnostics.pack_trace(fn, inst["items"], inst["capacity"], mode)
                        self.assertEqual(t["bins"], pack(fn, inst["items"], inst["capacity"]))

    def test_bf_ff_never_open_fresh(self):
        for fn in (baselines.best_fit, baselines.first_fit):
            for inst in self.train:
                self.assertEqual(diagnostics.pack_trace(fn, inst["items"], inst["capacity"])["n_fresh"], 0)

    def test_worst_fit_always_opens_fresh(self):
        # In funsearch mode an empty bin is always the emptiest, so Worst Fit takes it whenever it has the choice
        for inst in self.train[:4]:
            t = diagnostics.pack_trace(baselines.worst_fit, inst["items"], inst["capacity"])
            self.assertGreater(t["n_room"], 0)
            self.assertEqual(t["n_fresh"], t["n_room"])

    def test_anyfit_never_opens_fresh(self):
        for inst in self.train[:4]:
            t = diagnostics.pack_trace(eoh_seed0(), inst["items"], inst["capacity"], "anyfit")
            self.assertEqual(t["n_fresh"], 0)

    def test_hand_checked(self):
        # cap 10, items [3, 8, 2, 6], funsearch mode
        # BF: 3->A, 8->B (A has 7, too small), 2->B (tightest), 6->A          => fresh 0 of 2 choices, 2 bins
        # WF: 3->A, 8->B (no choice), 2->C (empty beats A=7, B=2), 6->D (empty beats A=7, C=8) => fresh 2 of 2, 4 bins
        items = np.array([3, 8, 2, 6])
        self.assertEqual(diagnostics.pack_trace(baselines.best_fit, items, 10), dict(bins=2, n_room=2, n_fresh=0))
        self.assertEqual(diagnostics.pack_trace(baselines.worst_fit, items, 10), dict(bins=4, n_room=2, n_fresh=2))


if __name__ == "__main__":
    unittest.main()
