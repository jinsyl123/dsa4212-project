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
    # TODO: write this yourself (~10 lines, numpy only), without reading core.pack_funsearch
    raise NotImplementedError


def train_instances():
    spec = yaml.safe_load((ROOT / "configs/instances/v0.yaml").read_text())
    return instances.generate_set("train", spec["sets"]["train"], spec["seed"])


class TestIndependentPacker(unittest.TestCase):
    def test_hand_checked(self):
        # Work these out on paper first, then fill them in
        items, cap = np.array([6, 5, 4, 3, 2]), 10
        self.assertEqual(pack(baselines.best_fit, items, cap), ...)   # TODO: best fit, by hand
        self.assertEqual(pack(baselines.first_fit, items, cap), ...)  # TODO: first fit, by hand

    def test_matches_core_on_train(self):
        for fn in (baselines.best_fit, baselines.first_fit):
            for inst in train_instances():
                with self.subTest(rule=fn.__name__, instance=inst["id"]):
                    self.assertEqual(pack(fn, inst["items"], inst["capacity"]),
                                     core.pack_funsearch(fn, inst["items"], inst["capacity"]))


if __name__ == "__main__":
    unittest.main()
