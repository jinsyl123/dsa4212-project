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
        self.base = C.load(
            ROOT / "configs" / "eoh_baseline.yaml",
            llm="openrouter-gemini-3.5-flash-lite",
        )

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
        names = list(ABLATIONS) + ["abl_no_m2", "abl_mixed_capacities", "control_randsample"]
        cfgs = {n: self._load(n) for n in names}
        self.assertEqual(
            len({c["experiment"] for c in cfgs.values()}), len(names)
        )  # separate results folders
        for n, cfg in cfgs.items():
            with self.subTest(n):
                self.assertEqual(cfg["llm"]["model"], "google/gemini-3.5-flash-lite")
                self.assertEqual(cfg["llm"]["provider"], "openrouter")
                self.assertEqual(cfg["budget"]["max_evals"], 300)
                # Ablations 1 and 2 and the control run 10 seeds (docs/log.md, 7 and 9 Nov); the dropped rows keep 5
                expected = (
                    list(range(10))
                    if cfg["experiment"]
                    in ("abl_anyfit", "abl_informed_rules", "abl_mixed_shapes", "abl_mixed_capacities", "control_randsample")
                    else list(range(5))
                )
                self.assertEqual(cfg["seeds"], expected)

    def test_method_ablations(self):
        no_m2 = self._load("abl_no_m2")["methods"]
        self.assertEqual([m["label"] for m in no_m2], ["eoh_no_m2"])
        self.assertFalse(no_m2[0]["params"]["use_m2_operator"])
        # tuned_bf is the no-LLM reference, retuned in the control's own folder like the baseline's
        self.assertEqual(
            [m["name"] for m in self._load("control_randsample")["methods"]],
            ["randsample", "tuned_bf"],
        )

    def test_strong_model_changes_only_the_llm(self):
        # Optional A3 (#7): the full model of the same generation, same reasoning effort and token cap
        cfg = self._load("abl_strong_model")
        self.assertEqual(cfg["llm"]["model"], "google/gemini-3.5-flash")
        self.assertEqual(cfg["llm"]["provider"], self.base["llm"]["provider"])
        for k in ("extra", "max_tokens"):
            self.assertEqual(cfg["llm"].get(k), self.base["llm"].get(k), k)
        for k in ("budget", "methods", "evaluation", "final_eval", "instances"):
            self.assertEqual(cfg[k], self.base[k], k)
        self.assertEqual(cfg["seeds"], list(range(10)))


    def test_mixed_capacities_changes_only_the_training_set_from_a1b(self):
        # Ablation 2b (#7): A1b's setup; its comparison row is abl_informed_rules (claims, #15)
        cfg, a1b = self._load("abl_mixed_capacities"), self._load("abl_informed_rules")
        self.assertEqual(cfg["evaluation"], {**a1b["evaluation"], "train_set": "train_mixed_cap"})
        for k in ("llm", "budget", "final_eval", "instances", "seeds"):
            self.assertEqual(cfg[k], a1b[k], k)
        eoh, tbf = cfg["methods"]
        self.assertEqual(eoh, a1b["methods"][0])
        self.assertEqual(tbf["name"], "tuned_bf")
        # only the penalty range is widened, so the fresh-bin bet is reachable at capacity 300 (docs/log.md, 9 Nov)
        from obpevo import tuned_bf
        space = tuned_bf.check_space(tbf["params"]["space"])
        self.assertEqual({k: v for k, v in space.items() if k != "penalty"},
                         {k: v for k, v in tuned_bf.SPACE.items() if k != "penalty"})
        self.assertEqual(list(space["penalty"]), [1, 300])
        self.assertGreater(space["penalty"][1], 300 - 45 - space["hi_max"])  # empty bin r ~ 300 - item > used r + penalty


class TestMixedSet(unittest.TestCase):
    def test_alternates_components(self):
        spec = dict(
            n_instances=8,
            n_items=3000,
            capacity=100,
            dist="weibull",
            scale=45,
            mix=[{"shape": 2}, {"shape": 4}],
        )
        insts = instances.generate_set("m", spec, base_seed=0)
        cv = [np.std(i["items"]) / np.mean(i["items"]) for i in insts]
        self.assertTrue(
            all(cv[k] > cv[k + 1] for k in range(0, 8, 2))
        )  # shape 2 is more spread out
        again = instances.generate_set("m", spec, base_seed=0)
        self.assertTrue(
            all((a["items"] == b["items"]).all() for a, b in zip(insts, again))
        )

    def test_train_mixed_keeps_test_shapes_out(self):
        import yaml

        sets = yaml.safe_load((ROOT / "configs" / "instances" / "v0.yaml").read_text())[
            "sets"
        ]
        train_shapes = {m["shape"] for m in sets["train_mixed"]["mix"]}
        test_shapes = {
            s["shape"] for n, s in sets.items() if n.startswith("test_shape")
        }
        self.assertFalse(train_shapes & test_shapes)

    def test_train_mixed_cap_alternates_capacity_with_the_same_items(self):
        import yaml

        sets = yaml.safe_load((ROOT / "configs" / "instances" / "v0.yaml").read_text())["sets"]
        insts = instances.generate_set("train_mixed_cap", sets["train_mixed_cap"], base_seed=0)
        self.assertEqual(len(insts), 32)
        self.assertEqual([i["capacity"] for i in insts], [100, 300] * 16)
        for cap in (100, 300):
            items = np.concatenate([i["items"] for i in insts if i["capacity"] == cap])
            self.assertAlmostEqual(items.mean(), 40.2, delta=1.0)  # Weibull(3) * 45 in both halves
        # every other key matches train, so the capacity is the only change
        rest = {k: v for k, v in sets["train_mixed_cap"].items() if k != "mix"}
        self.assertEqual(rest, sets["train"])

    def test_missing_set_is_built(self):
        # a laptop with data/ already built must still get sets added to the spec later
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / "spec.yaml"
            spec.write_text(
                "seed: 0\nsets:\n  a: {n_instances: 1, n_items: 50, capacity: 100}\n"
            )
            cfg = {"instances": {"spec": str(spec), "dir": str(Path(d) / "data")}}
            out = C.ensure_instances(cfg)
            self.assertTrue((out / "a.npz").exists())
            spec.write_text(
                spec.read_text() + "  b: {n_instances: 1, n_items: 50, capacity: 100}\n"
            )
            a_before = (out / "a.npz").read_bytes()
            C.ensure_instances(cfg)
            self.assertTrue((out / "b.npz").exists())
            self.assertEqual(
                (out / "a.npz").read_bytes(), a_before
            )  # existing sets untouched

    def test_changed_set_is_rebuilt(self):
        # if a set is redefined under the same name (e.g. #12), an existing data/ must not keep the old instances
        with tempfile.TemporaryDirectory() as d:
            spec = Path(d) / "spec.yaml"
            spec.write_text(
                "seed: 0\nsets:\n  a: {n_instances: 1, n_items: 50, capacity: 100}\n"
                "  b: {n_instances: 1, n_items: 50, capacity: 100}\n"
            )
            cfg = {"instances": {"spec": str(spec), "dir": str(Path(d) / "data")}}
            out = C.ensure_instances(cfg)
            b_before = (out / "b.npz").read_bytes()
            spec.write_text(
                spec.read_text().replace("a: {n_instances: 1", "a: {n_instances: 3")
            )
            C.ensure_instances(cfg)
            self.assertEqual(
                len(instances.load_set(out / "a.npz")), 3
            )  # changed set rebuilt
            self.assertEqual(
                (out / "b.npz").read_bytes(), b_before
            )  # unchanged set untouched
            self.assertEqual(
                instances.load_set(out / "a.npz")[0]["items"].tolist(),
                instances.generate_set(
                    "a", {"n_instances": 3, "n_items": 50, "capacity": 100}, 0
                )[0]["items"].tolist(),
            )


if __name__ == "__main__":
    unittest.main()
