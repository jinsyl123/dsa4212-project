"""Integrity checks: one deliberately bad program per rule, clean programs stay clean, and the
determinism rescoring catches randomness and state carried across instances.
Run: python -m unittest tests.test_integrity"""
import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import check_integrity  # noqa: E402
from obpevo import instances, integrity, task  # noqa: E402

H = "import numpy as np\n\n"
DEF = "def priority(item, bins):\n"
BF = "    return item - bins\n"

# One bad program per rule -> the flag it must raise
BAD = {
    "syntax_error": DEF + "    return item - \n",
    "import:os": "import os\n" + H + DEF + BF,
    "import:time": H + DEF + "    import time\n" + BF,
    "import:random": "import random\n" + H + DEF + BF,
    "random": H + DEF + "    return item - bins + np.random.rand(len(bins))\n",
    "global": H + "n = 0\n" + DEF + "    global n\n    n += 1\n" + BF,
    "attr_assign": H + DEF + "    priority.calls = 1\n" + BF,
    "module_state": H + "seen = []\n" + DEF + "    seen.append(item)\n" + BF,
    "mutable_default": H + "def priority(item, bins, cache={}):\n" + BF,
    "call:exec": H + DEF + "    exec('x = 1')\n" + BF,
    "call:eval": H + DEF + "    return eval('item - bins')\n",
    "call:open": H + DEF + "    open('/tmp/x', 'w')\n" + BF,
    "dunder:__builtins__": H + DEF + "    __builtins__['print']('hi')\n" + BF,
}
EXTRA_BAD = {   # same rules, other spellings
    "random": [H + DEF + "    rng = np.random.default_rng()\n" + BF,
               "from numpy import random\n" + H + DEF + BF,
               "import numpy.random as npr\n" + H + DEF + BF],
    "attr_assign": [H + DEF + "    setattr(priority, 'n', 1)\n" + BF],
    "module_state": [H + "best = {}\n" + DEF + "    best[item] = 1\n" + BF],
    "dunder:__class__": [H + DEF + "    x = ().__class__\n" + BF],
}
CLEAN = [
    task.TEMPLATE_LLM4AD,
    H + DEF + "    r = bins - item\n    out = np.where(r >= 0, -r, -np.inf)\n    out[r == 0] += 10\n    return out\n",
    H + "W = np.array([1.0, 2.0])\n" + DEF + "    w = W * 2\n" + BF,                   # reads a module constant
    H + DEF + "    cache = []\n    cache.append(item)\n" + BF,                         # local list, not shared
    H + DEF + "    \"\"\"Docstring mentioning random and global.\"\"\"\n" + BF,
]
STATEFUL = H + "seen = []\n" + DEF + "    seen.append(item)\n" \
    "    return bins - item if len(seen) > 300 else item - bins\n"           # Best Fit, then Worst Fit after 300 items
RANDOM = H + DEF + "    return np.random.rand(len(bins))\n"     # random bin: bin counts vary run to run
# Small noise on Best Fit often leaves bin counts unchanged, so the repeat check can miss it; the static
# `random` flag is what catches that case.


class TestStaticRules(unittest.TestCase):
    def test_each_bad_program_raises_its_flag(self):
        for flag, src in BAD.items():
            with self.subTest(flag=flag):
                self.assertIn(flag, integrity.flags(src))
        for flag, srcs in EXTRA_BAD.items():
            for src in srcs:
                with self.subTest(flag=flag, src=src):
                    self.assertIn(flag, integrity.flags(src))

    def test_clean_programs_have_no_flags(self):
        for src in CLEAN:
            with self.subTest(src=src):
                self.assertEqual(integrity.flags(src), [])

    def test_committed_best_programs_are_clean(self):
        progs = sorted(ROOT.glob("results/*/*/seed*/best_program.py"))
        self.assertTrue(progs)
        for p in progs:
            with self.subTest(program=str(p.relative_to(ROOT))):
                self.assertEqual(integrity.flags(p.read_text()), [])

    def test_recorder_logs_flags(self):
        try:
            from obpevo.recorder import EvalRecorder
        except ImportError:                                   # llm4ad not installed
            self.skipTest("llm4ad not installed")
        with TemporaryDirectory() as d:
            rec = EvalRecorder(Path(d), max_evals=None, timeout_s=60)
            with open(rec.evals_csv, "w", newline="") as f:
                csv.writer(f).writerow(rec.FIELDS)
            rec._log(1, 0.0, 0.1, "h1", task.TEMPLATE_LLM4AD, -0.05)
            rec._log(2, 0.0, 0.1, "h2", BAD["random"], -0.04)
            rows = list(csv.DictReader(open(rec.evals_csv)))
        self.assertEqual([r["integrity"] for r in rows], ["", "random"])
        self.assertEqual(rows[1]["status"], "ok")             # flagged, not dropped


class TestDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = dict(n_instances=4, n_items=200, capacity=100, dist="weibull", shape=3, scale=45)
        cls.sets = {"t": instances.generate_set("t", spec, base_seed=0)}
        progs = [dict(label=k, seed=0, program=p)
                 for k, p in (("bf", task.TEMPLATE_LLM4AD), ("stateful", STATEFUL), ("random", RANDOM))]
        cls.df = check_integrity.check_programs(progs, cls.sets, "funsearch", timeout=30, workers=4).set_index("label")

    def test_best_fit_is_deterministic_and_stateless(self):
        r = self.df.loc["bf"]
        self.assertEqual((r.status, r["flags"], r.repeat_mismatch, r.shuffle_mismatch), ("ok", "", 0, 0))

    def test_state_across_instances_is_caught(self):
        r = self.df.loc["stateful"]
        self.assertEqual(r.repeat_mismatch, 0)               # same order -> same result
        self.assertGreater(r.shuffle_mismatch, 0)            # different order -> different bins
        self.assertIn("module_state", r["flags"])

    def test_randomness_is_caught(self):
        r = self.df.loc["random"]
        self.assertGreater(r.repeat_mismatch, 0)             # fresh RNG seed per process, so runs differ
        self.assertIn("random", r["flags"])

    def test_shuffle_never_identity(self):
        insts = self.sets["t"]
        self.assertNotEqual([i["id"] for i in check_integrity.shuffled(insts)], [i["id"] for i in insts])
        self.assertEqual(sorted(i["id"] for i in check_integrity.shuffled(insts)), sorted(i["id"] for i in insts))


if __name__ == "__main__":
    unittest.main()
