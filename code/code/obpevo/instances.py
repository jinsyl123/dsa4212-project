"""Instance generation, saving and loading.

Each named set (train, val, test_*) is generated from a spec in configs/instances/*.yaml
with its own deterministic seed, so every member of the team regenerates identical data.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def _set_seed(base_seed: int, set_name: str) -> int:
    """Stable per-set seed: changing one set's spec never changes another set's data."""
    return (base_seed * 1_000_003 + zlib.crc32(set_name.encode())) % (2 ** 32)


def _sample_items(rng: np.random.Generator, n: int, cap: int, spec: dict) -> np.ndarray:
    dist = spec.get("dist", "weibull")
    if dist == "weibull":
        # FunSearch/LLM4AD default: Weibull(shape=3) * 45, clipped to [1, capacity], rounded
        # scale_frac (relative to capacity) keeps the item distribution identical in relative terms
        scale = spec["scale_frac"] * cap if "scale_frac" in spec else spec.get("scale", 45)
        x = rng.weibull(spec.get("shape", 3.0), n) * scale
    elif dist == "uniform":
        lo = spec["low_frac"] * cap if "low_frac" in spec else spec.get("low", 20)
        hi = spec["high_frac"] * cap if "high_frac" in spec else spec.get("high", 100)
        x = rng.uniform(lo, hi, n)
    else:
        raise ValueError(f"unknown dist {dist!r}")
    return np.clip(np.round(x), 1, cap).astype(int)


def generate_set(set_name: str, spec: dict, base_seed: int) -> list[dict]:
    rng = np.random.default_rng(_set_seed(base_seed, set_name))
    n_spec, cap = spec["n_items"], int(spec["capacity"])
    out = []
    for k in range(int(spec["n_instances"])):
        n = int(rng.integers(n_spec[0], n_spec[1] + 1)) if isinstance(n_spec, list) else int(n_spec)
        out.append(dict(id=f"{set_name}_{k:03d}", capacity=cap, items=_sample_items(rng, n, cap, spec)))
    return out


def save_set(instances: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array([i["id"] for i in instances]),
                        capacities=np.array([i["capacity"] for i in instances]),
                        **{f"items_{k}": i["items"] for k, i in enumerate(instances)})


def load_set(path: Path) -> list[dict]:
    z = np.load(path)
    return [dict(id=str(i), capacity=int(c), items=z[f"items_{k}"])
            for k, (i, c) in enumerate(zip(z["ids"], z["capacities"]))]


def build_all(spec_file: Path, out_dir: Path, force: bool = False) -> pd.DataFrame:
    """Generate every set in a spec file into out_dir and write a manifest of basic statistics."""
    import yaml
    cfg = yaml.safe_load(Path(spec_file).read_text())
    rows = []
    for name, spec in cfg["sets"].items():
        path = out_dir / f"{name}.npz"
        insts = load_set(path) if path.exists() and not force else generate_set(name, spec, cfg.get("seed", 0))
        if force or not path.exists():
            save_set(insts, path)
        for i in insts:
            rows.append(dict(set=name, instance=i["id"], n_items=len(i["items"]), capacity=i["capacity"],
                             mean_size=round(float(i["items"].mean()), 2), mean_size_frac=round(float(i["items"].mean()) / i["capacity"], 3),
                             lb=int(np.ceil(i["items"].sum() / i["capacity"]))))
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "spec.json").write_text(json.dumps(cfg, indent=2))      # copy of the spec used
    return man
