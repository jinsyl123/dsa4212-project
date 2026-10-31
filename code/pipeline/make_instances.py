"""Generate the instance sets defined in a spec file. Usage:
    python code/make_instances.py --spec configs/instances/v0.yaml --out data/v0 [--force]
Writes one .npz per set plus manifest.csv (per-instance statistics) and a copy of the spec.
"""
import argparse
from pathlib import Path

from obpevo import instances
from obpevo.config import ROOT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/instances/v0.yaml")
    ap.add_argument("--out", default="data/v0")
    ap.add_argument("--force", action="store_true", help="regenerate even if files exist")
    a = ap.parse_args()
    man = instances.build_all(ROOT / a.spec, ROOT / a.out, force=a.force)
    print(man.groupby("set").agg(instances=("instance", "size"), n_items=("n_items", "mean"),
                                 capacity=("capacity", "first"), mean_size_frac=("mean_size_frac", "mean")).round(3))
