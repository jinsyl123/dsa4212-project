"""Checks on the frozen v0 instance sets (#12). Run: python -m unittest tests.test_instance_sets"""
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from obpevo import baselines, core, instances  # noqa: E402

SPEC = yaml.safe_load((ROOT / "configs" / "instances" / "v0.yaml").read_text())


class TestControlSet(unittest.TestCase):
    """Every item is over half a bin, so every rule must use exactly one bin per item."""

    def setUp(self):
        self.insts = instances.generate_set("test_control", SPEC["sets"]["test_control"], SPEC["seed"])

    def test_every_item_over_half_a_bin(self):
        for i in self.insts:
            self.assertTrue((i["items"] > i["capacity"] / 2).all(), i["id"])

    def test_every_baseline_uses_one_bin_per_item(self):
        for name, pack in baselines.BASELINES.items():
            for i in self.insts:
                self.assertEqual(pack(i["items"], i["capacity"]), len(i["items"]), (name, i["id"]))

    def test_funsearch_mode_rules_use_one_bin_per_item(self):
        # in funsearch mode a rule may prefer empty bins; here it has no other choice anyway
        prefer_empty = lambda item, bins: bins.astype(float)  # noqa: E731
        for rule in (baselines.best_fit, baselines.first_fit, prefer_empty):
            for i in self.insts:
                self.assertEqual(core.pack_funsearch(rule, i["items"], i["capacity"]), len(i["items"]))


class TestTestSets(unittest.TestCase):
    def test_ten_instances_each(self):
        # 10 instances, so a rule that beats Best Fit on every instance can be significant on the sign test
        for name, spec in SPEC["sets"].items():
            if name.startswith("test_"):
                self.assertEqual(spec["n_instances"], 10, name)


if __name__ == "__main__":
    unittest.main()
