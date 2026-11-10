"""Experiment configuration: one YAML file fixes methods, seeds, LLM and the shared budget."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # repository root

DEFAULTS = dict(
    experiment="unnamed",
    instances=dict(spec="configs/instances/v0.yaml", dir="data/v0"),
    llm=dict(provider="fake", model="fake", temperature=1.0, max_tokens=4096,
             price_in_per_1m=0.0, price_out_per_1m=0.0),     # see PROVIDERS in llm.py
    budget=dict(max_evals=100, max_tokens=None, max_minutes=None),   # identical for every method
    parallel=dict(num_samplers=2, num_evaluators=2),
    evaluation=dict(mode="funsearch", timeout_seconds=60, train_set="train", prompt="llm4ad", format_hint=True),
    seeds=[0],
    methods=[dict(name="eoh")],
    final_eval=dict(sets=["val"], timeout_seconds=600),
)


def _merge(base: dict, override: dict) -> dict:
    """Deep merge. Exception: an `llm` section that names a `provider` or registry `name` replaces the old one
    entirely, so switching model never inherits the previous model's URL, key name or prices."""
    override = override or {}
    if isinstance(override.get("llm"), dict) and ({"provider", "name"} & set(override["llm"])):
        base = {**base, "llm": copy.deepcopy(DEFAULTS["llm"])}
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path                              # so notebooks in subfolders can pass "configs/x.yaml"
    return path


def _read(path: str | Path) -> dict:
    """Read a YAML config; `base: other.yaml` inherits everything from that file and overrides what is given here."""
    raw = yaml.safe_load(_resolve(path).read_text()) or {}
    base = raw.pop("base", None)
    return _merge(_read(base), raw) if base else raw


def _registry(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text()) or {}


def llm_registry() -> dict:
    """All entries in configs/llms.yaml: the single source of truth for models and prices."""
    return _registry("llms")


def budget_presets() -> dict:
    return _registry("budgets")


def _resolve_llm(spec) -> dict:
    """A registry name ("gemini-3.8-flash"), a dict naming one plus tweaks ({"name": ..., "temperature": 0.5}),
    or a full dict with provider/model (used as given)."""
    if isinstance(spec, str):
        spec = {"name": spec}
    if "name" in spec:
        reg = llm_registry()
        if spec["name"] not in reg:
            raise KeyError(f"LLM {spec['name']!r} is not in configs/llms.yaml; available: {sorted(reg)}")
        entry = {k: v for k, v in reg[spec["name"]].items() if k not in ("notes", "checked")}
        spec = {**entry, **spec}
    return _merge(DEFAULTS["llm"], spec)


def _resolve_budget(spec) -> dict:
    if isinstance(spec, str):
        presets = budget_presets()
        if spec not in presets:
            raise KeyError(f"Budget {spec!r} is not in configs/budgets.yaml; available: {sorted(presets)}")
        return {**presets[spec], "name": spec}
    return {**DEFAULTS["budget"], **spec}


def load(path: str | Path, overrides: dict | None = None, llm=None, budget=None, experiment: str | None = None) -> dict:
    """Load a config. Optional arguments take priority over the file:
    llm:        registry name or dict (see configs/llms.yaml)
    budget:     preset name or dict (see configs/budgets.yaml)
    experiment: results folder name. If omitted while llm/budget are given, it is derived from them
                (e.g. "eoh_baseline__gemini-3.8-flash__pilot"), so different choices never share a folder.
    overrides:  any other nested settings, applied last."""
    cfg = _merge(DEFAULTS, _read(path))
    base_name = cfg["experiment"]
    cfg["llm"] = _resolve_llm(llm if llm is not None else cfg["llm"])          # names -> full settings first,
    cfg["budget"] = _resolve_budget(budget if budget is not None else cfg["budget"])   # so overrides adjust them
    ov = dict(overrides or {})
    for key, resolve in (("llm", _resolve_llm), ("budget", _resolve_budget)):
        if isinstance(ov.get(key), str):                                       # a name in overrides replaces
            cfg[key] = resolve(ov.pop(key))
    cfg = _merge(cfg, ov)
    cfg["llm"] = _resolve_llm(cfg["llm"])
    if experiment:
        cfg["experiment"] = experiment
    elif (llm is not None or budget is not None) and "experiment" not in (overrides or {}):
        tags = [_tag(llm, "name"), _tag(budget, "name")]
        cfg["experiment"] = "__".join([base_name] + [t for t in tags if t])
    for m in cfg["methods"]:
        m.setdefault("label", m["name"])            # label lets the same method run with different params (ablations)
        m.setdefault("params", {})
    cfg["_config_path"] = str(_resolve(path))
    return cfg


def _tag(spec, key) -> str | None:
    """Folder-name tag for an llm/budget argument: its name, plus a short hash if it was tweaked."""
    import hashlib, json
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec
    extra = {k: v for k, v in spec.items() if k != key}
    h = hashlib.sha1(json.dumps(extra, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{spec[key]}-{h}" if key in spec and extra else spec.get(key) or f"custom-{h}"


def results_dir(cfg: dict) -> Path:
    return ROOT / "results" / cfg["experiment"]


def instance_dir(cfg: dict) -> Path:
    return ROOT / cfg["instances"]["dir"]


def ensure_instances(cfg: dict) -> Path:
    """data/ is not in git: build the instance sets from their spec (deterministic) the first time any step needs them."""
    d = instance_dir(cfg)
    spec = ROOT / cfg["instances"]["spec"]
    sets = (yaml.safe_load(spec.read_text()) or {}).get("sets", {})
    from obpevo import instances
    if (not (d / "manifest.csv").exists() or any(not (d / f"{s}.npz").exists() for s in sets)
            or instances.spec_changed(spec, d)):
        instances.build_all(spec, d)        # builds missing sets and sets whose spec changed; others are kept
    return d
