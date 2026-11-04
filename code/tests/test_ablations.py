"""Ablation configs (#7) and the mixed training set. Run: python -m unittest tests.test_ablations"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from obpevo import config as C, instances  # noqa: E402

# config -> the only setting that may differ from the main baseline (besides experiment name, LLM and seeds)
ABLATIONS = {
    "abl_anyfit": ("evaluation", "mode", "anyfit"),
    "abl_informed_rules": ("evaluation", "prompt", "informed_rules"),
    "abl_informed_dist": ("evaluation", "prompt", "informed_dist"),
    "abl_informed": ("evaluation", "prompt", "informed"),
    "abl_mixed_shapes": ("evaluation", "train_set", "train_mixed"),
}


class TestAblationConfigs(unittest.TestCase):
    def setUp(self):
        self.base = C.load(ROOT / "configs" / "eoh_baseline.yaml", llm="openrouter-gemini-3.5-flash-lite")

    def _load(self, name):
        return C.load(ROOT / "configs" / f"{name}.yaml")

    def test_one_change_from_baseline(self):
        for name, (section, key, value) in ABLATIONS.items():
            with self.subTest(name):
                cfg = self._load(name)
                self.assertEqual(cfg[section][key], value)
                for k in ("llm", "budget", "methods", "final_eval", "instances"):
                    self.assertEqual(cfg[k], self.base[k], k)
                expected = {**self.base["evaluation"], key: value}
                self.assertEqual(cfg["evaluation"], expected)

    def test_shared_settings(self):
        names = list(ABLATIONS) + ["abl_no_m2", "control_randsample"]
        cfgs = {n: self._load(n) for n in names}
        self.assertEqual(len({c["experiment"] for c in cfgs.values()}), len(names))      # separate results folders
        for n, cfg in cfgs.items():
            with self.subTest(n):
                self.assertEqual(cfg["llm"]["model"], "google/gemini-3.5-flash-lite")
                self.assertEqual(cfg["llm"]["provider"], "openrouter")
                self.assertEqual(cfg["budget"]["max_evals"], 300)
                self.assertEqual(cfg["seeds"], [0, 1, 2, 3, 4])

    def test_method_ablations(self):
        no_m2 = self._load("abl_no_m2")["methods"]
        self.assertEqual([m["label"] for m in no_m2], ["eoh_no_m2"])
        self.assertFalse(no_m2[0]["params"]["use_m2_operator"])
        self.assertEqual([m["name"] for m in self._load("control_randsample")["methods"]], ["randsample"])


class TestMixedSet(unittest.TestCase):
    def test_alternates_components(self):
        spec = dict(n_instances=8, n_items=3000, capacity=100, dist="weibull", scale=45, mix=[{"shape": 2}, {"shape": 4}])
        insts = instances.generate_set("m", spec, base_seed=0)
        cv = [np.std(i["items"]) / np.mean(i["items"]) for i in insts]
        self.assertTrue(all(cv[k] > cv[k + 1] for k in range(0, 8, 2)))                 # shape 2 is more spread out
        again = instances.generate_set("m", spec, base_seed=0)
        self.assertTrue(all((a["items"] == b["items"]).all() for a, b in zip(insts, again)))

    def test_train_mixed_keeps_test_shapes_out(self):
        import yaml
        sets = yaml.safe_load((ROOT / "configs" / "instances" / "v0.yaml").read_text())["sets"]
        train_shapes = {m["shape"] for m in sets["train_mixed"]["mix"]}
        test_shapes = {s["shape"] for n, s in sets.items() if n.startswith("test_shape")}
        self.assertFalse(train_shapes & test_shapes)

    def test_missing_set_is_built(self):
        # a laptop with data/ already built must still get sets added to the spec later
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / "spec.yaml"
            spec.write_text("seed: 0\nsets:\n  a: {n_instances: 1, n_items: 50, capacity: 100}\n")
            cfg = {"instances": {"spec": str(spec), "dir": str(Path(d) / "data")}}
            out = C.ensure_instances(cfg)
            self.assertTrue((out / "a.npz").exists())
            spec.write_text(spec.read_text() + "  b: {n_instances: 1, n_items: 50, capacity: 100}\n")
            a_before = (out / "a.npz").read_bytes()
            C.ensure_instances(cfg)
            self.assertTrue((out / "b.npz").exists())
            self.assertEqual((out / "a.npz").read_bytes(), a_before)                     # existing sets untouched

    def test_changed_set_is_rebuilt(self):
        # if a set is redefined under the same name (e.g. #12), an existing data/ must not keep the old instances
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / "spec.yaml"
            spec.write_text("seed: 0\nsets:\n  a: {n_instances: 1, n_items: 50, capacity: 100}\n"
                            "  b: {n_instances: 1, n_items: 50, capacity: 100}\n")
            cfg = {"instances": {"spec": str(spec), "dir": str(Path(d) / "data")}}
            out = C.ensure_instances(cfg)
            b_before = (out / "b.npz").read_bytes()
            spec.write_text(spec.read_text().replace("a: {n_instances: 1", "a: {n_instances: 3"))
            C.ensure_instances(cfg)
            self.assertEqual(len(instances.load_set(out / "a.npz")), 3)                  # changed set rebuilt
            self.assertEqual((out / "b.npz").read_bytes(), b_before)                     # unchanged set untouched
            self.assertEqual(instances.load_set(out / "a.npz")[0]["items"].tolist(),
                             instances.generate_set("a", {"n_instances": 3, "n_items": 50, "capacity": 100}, 0)[0]["items"].tolist())


if __name__ == "__main__":
    unittest.main()
