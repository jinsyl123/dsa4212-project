"""Independent check of core.pack_funsearch: a packer written from scratch must give the same bin counts.
Run: python -m unittest tests.test_independent_packer"""
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from obpevo import core, baselines, instances  # noqa: E402


def pack(priority, items, capacity) -> int:
    """Pack items online with `priority`, FunSearch convention. Returns the number of bins used.

    Rules:
      1. One bin per item exists from the start, all at full capacity (remaining = capacity).
      2. For each item, score every bin it fits into (remaining >= item), empty ones included.
      3. The item goes into the highest-scoring bin; ties go to the earliest bin.
      4. Bins used = bins whose remaining capacity is no longer `capacity`.
    """
    # Written from the rules above only, without reading core.pack_funsearch
    bins = np.full(len(items), capacity, dtype=float)
    for item in items:
        idx = np.nonzero(bins >= item)[0]               # candidate bins, in bin order
        scores = np.asarray(priority(item, bins[idx]))
        best = idx[np.argmax(scores)]                   # argmax picks the first max, so ties go to the earliest bin
        bins[best] -= item
    return int(np.sum(bins != capacity))


def train_instances():
    spec = yaml.safe_load((ROOT / "configs/instances/v0.yaml").read_text())
    return instances.generate_set("train", spec["sets"]["train"], spec["seed"])


class TestIndependentPacker(unittest.TestCase):
    def test_hand_checked(self):
        # BF and FF disagree here, so a packer that ignores the scores fails.
        # BF: 3->A(7), 8->B(2), 2->B(0) tightest, 6->A(1)  => 2 bins
        # FF: 3->A(7), 8->B(2), 2->A(5) earliest, 6->C(4)  => 3 bins
        items, cap = np.array([3, 8, 2, 6]), 10
        self.assertEqual(pack(baselines.best_fit, items, cap), 2)
        self.assertEqual(pack(baselines.first_fit, items, cap), 3)

    def test_matches_core_on_train(self):
        # BF and FF, plus the committed EoH best program (a real evolved rule, not just a baseline)
        ns = {"np": np}
        exec((ROOT / "results/eoh_baseline/eoh/seed0/best_program.py").read_text(), ns)
        for fn in (baselines.best_fit, baselines.first_fit, ns["priority"]):
            for inst in train_instances():
                with self.subTest(rule=fn.__name__, instance=inst["id"]):
                    self.assertEqual(pack(fn, inst["items"], inst["capacity"]),
                                     core.pack_funsearch(fn, inst["items"], inst["capacity"]))


if __name__ == "__main__":
    unittest.main()
