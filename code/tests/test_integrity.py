"""Integrity checks: one deliberately bad program per rule, clean programs stay clean, and the
determinism rescoring catches randomness, hash()-based choices and state carried across instances.
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

# One bad program per rule -> the flag it must raise. Most are body-level, since LLM4AD keeps only the body.
BAD = {
    "syntax_error": DEF + "    return item - \n",
    "parse_error": "x = " + "1+" * 200000 + "1\n",                      # RecursionError inside ast.parse
    "import:os": "import os\n" + H + DEF + BF,
    "import:time": H + DEF + "    import time\n" + BF,
    "import:random": "import random\n" + H + DEF + BF,
    "star_import": H + DEF + "    from numpy import *\n" + BF,
    "random": H + DEF + "    return item - bins + np.random.rand(len(bins))\n",
    "clock": H + DEF + "    t = np.datetime64('now')\n" + BF,
    "call:hash": H + DEF + "    k = hash(str(item))\n" + BF,
    "call:id": H + DEF + "    k = id(bins)\n" + BF,
    "call:exec": H + DEF + "    exec('x = 1')\n" + BF,
    "call:eval": H + DEF + "    return eval('item - bins')\n",
    "call:open": H + DEF + "    open('/tmp/x', 'w')\n" + BF,
    "dynamic_getattr": H + DEF + "    r = getattr(np, 'ran' + 'dom')\n" + BF,
    "dunder:__builtins__": H + DEF + "    __builtins__['print']('hi')\n" + BF,
    "file_io": H + DEF + "    np.save('/tmp/s.npy', bins)\n" + BF,
    "numpy_state": H + DEF + "    np.set_printoptions(precision=int(item))\n" + BF,
    "global": H + "n = 0\n" + DEF + "    global n\n    n += 1\n" + BF,
    "attr_assign": H + DEF + "    priority.calls = 1\n" + BF,
    "module_state": H + "seen = []\n" + DEF + "    seen.append(item)\n" + BF,
    "mutable_default": H + "def priority(item, bins, cache={}):\n" + BF,
}
EXTRA_BAD = {   # same rules, other spellings (several found by review)
    "random": [H + DEF + "    rng = np.random.default_rng()\n" + BF,
               "from numpy import random\n" + H + DEF + BF,
               "import numpy.random as npr\n" + H + DEF + BF,
               "import numpy\n" + DEF + "    return item - bins + numpy.random.rand(len(bins))\n",
               H + DEF + "    r = getattr(np, 'random')\n" + BF],
    "file_io": [H + DEF + "    s = np.load('/tmp/s.npy')\n" + BF,
                H + DEF + "    bins.tofile('/tmp/b')\n" + BF],
    "attr_assign": [H + DEF + "    setattr(priority, 'n', 1)\n" + BF],
    "dynamic_getattr": [H + DEF + "    d = getattr(priority, '__d' + 'ict__')\n" + BF],
    "dunder:__dict__": [H + DEF + "    d = getattr(priority, '__dict__')\n" + BF],
    "dunder:__class__": [H + DEF + "    x = ().__class__\n" + BF],
    "module_state": [H + "best = {}\n" + DEF + "    best[item] = 1\n" + BF,
                     H + "S = np.zeros(1)\n" + DEF + "    np.add(S, 1, out=S)\n" + BF,
                     H + "S = np.zeros(1)\n" + DEF + "    np.copyto(S, item)\n" + BF,
                     H + DEF + "    np.sctypeDict.setdefault('_c', []).append(item)\n" + BF,
                     H + "try:\n    S = []\nexcept Exception:\n    pass\n" + DEF + "    S.append(item)\n" + BF,
                     H + "if True:\n    S = []\n" + DEF + "    S.append(item)\n" + BF,
                     H + "class C:\n    seen = []\n" + DEF + "    C.seen.append(item)\n" + BF],
    "mutable_default": [H + "def priority(item, bins, s=np.zeros(1)):\n" + BF,
                        H + "def priority(item, bins, *, s=list()):\n" + BF],
}
CLEAN = [
    task.TEMPLATE_LLM4AD,
    H + DEF + "    r = bins - item\n    out = np.where(r >= 0, -r, -np.inf)\n    out[r == 0] += 10\n    return out\n",
    H + "W = np.array([1.0, 2.0])\n" + DEF + "    w = W * 2\n" + BF,                   # reads a module constant
    H + DEF + "    cache = []\n    cache.append(item)\n" + BF,                         # local list, not shared
    H + DEF + "    \"\"\"Docstring mentioning random, hash and global.\"\"\"\n" + BF,
    "from __future__ import annotations\n" + H + DEF + BF,
    H + DEF + "    v = np.__version__\n" + BF,
    H + DEF + "    with np.errstate(divide='ignore'):\n        r = 1.0 / (bins - item)\n    return r\n",
    H + DEF + "    s = np.char.add(['a'], ['b'])\n" + BF,                             # numpy submodule, not a store
    H + DEF + "    x = np.sort(bins)\n    np.clip(x, 0, 1, out=x)\n" + BF,               # out= on a local array
    H + "s = []\n" + DEF + "    def g(s):\n        s.append(1)\n" + BF,                 # nested param shadows s
]
STATEFUL = H + "seen = []\n" + DEF + "    seen.append(item)\n" \
    "    return bins - item if len(seen) > 300 else item - bins\n"           # Best Fit, then Worst Fit after 300 items
RANDOM = H + DEF + "    return np.random.rand(len(bins))\n"                  # random bin: bin counts vary run to run
# Small noise on Best Fit often leaves bin counts unchanged, so the repeat check can miss it; the static
# `random` flag is what catches that case.
HASHY = H + DEF + "    return np.array([hash(str(item * 7919 + j)) % 97 for j in range(len(bins))], dtype=float)\n"


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

    def test_never_raises(self):
        for src in ("x = '\udcff'\n", "(" * 5000 + ")" * 5000, "", "\x00", BAD["parse_error"]):
            with self.subTest(src=src[:20]):
                self.assertIsInstance(integrity.flags(src), list)

    def test_runs_on_committed_best_programs(self):
        # Whether they are clean is a result (results/*/integrity.csv), not a property of the code
        progs = sorted(ROOT.glob("results/*/*/seed*/best_program.py"))
        self.assertTrue(progs)
        for p in progs:
            self.assertIsInstance(integrity.flags(p.read_text()), list)

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
            with open(rec.evals_csv) as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["integrity"] for r in rows], ["", "random"])
        self.assertEqual(rows[1]["status"], "ok")             # flagged, not dropped


class TestDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = dict(n_instances=4, n_items=200, capacity=100, dist="weibull", shape=3, scale=45)
        cls.sets = {"t": instances.generate_set("t", spec, base_seed=0)}
        progs = [dict(label=k, seed=0, program=p) for k, p in
                 (("bf", task.TEMPLATE_LLM4AD), ("stateful", STATEFUL), ("random", RANDOM), ("hashy", HASHY))]
        cls.df = check_integrity.check_programs(progs, cls.sets, "funsearch", timeout=60, workers=4).set_index("label")

    def test_best_fit_is_deterministic_and_stateless(self):
        r = self.df.loc["bf"]
        self.assertEqual((r.status, r["flags"], r.repeat_mismatch, r.shuffle_mismatch, r.verdict),
                         ("ok", "", 0, 0, "clean"))

    def test_state_across_instances_is_caught(self):
        r = self.df.loc["stateful"]
        self.assertEqual(r.repeat_mismatch, 0)               # same order -> same result
        self.assertGreater(r.shuffle_mismatch, 0)            # different order -> different bins
        self.assertEqual(r.verdict, "stateful")
        self.assertIn("module_state", r["flags"])

    def test_randomness_is_caught(self):
        r = self.df.loc["random"]
        self.assertGreater(r.repeat_mismatch, 0)             # fresh RNG seed per process, so runs differ
        self.assertEqual(r.verdict, "nondeterministic")
        self.assertIn("random", r["flags"])

    def test_hash_based_choice_is_caught(self):
        r = self.df.loc["hashy"]
        self.assertGreater(r.repeat_mismatch + r.shuffle_mismatch, 0)   # fresh hash seed per process (spawn)
        self.assertEqual(r.verdict, "nondeterministic")
        self.assertIn("call:hash", r["flags"])

    def test_verdict_rules(self):
        v = check_integrity._verdict
        self.assertEqual(v(True, "", 0, 0), "clean")
        self.assertEqual(v(True, "import:math", 0, 0), "flagged")
        self.assertEqual(v(True, "module_state", 0, 3), "stateful")
        self.assertEqual(v(True, "call:hash", 0, 1), "nondeterministic")    # rare randomness seen only in S/T
        self.assertEqual(v(True, "", 2, 0), "nondeterministic")
        self.assertEqual(v(False, "", None, None), "failed")

    def test_shuffle_never_identity(self):
        insts = self.sets["t"]
        self.assertNotEqual([i["id"] for i in check_integrity.shuffled(insts)], [i["id"] for i in insts])
        self.assertEqual(sorted(i["id"] for i in check_integrity.shuffled(insts)), sorted(i["id"] for i in insts))


if __name__ == "__main__":
    unittest.main()
