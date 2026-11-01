"""Registry of LLM4AD search methods. Adding a method = adding one entry to METHODS.

Every method receives the same LLM wrapper, the same task (training instances only) and the
same evaluation budget (`max_sample_nums`, which all LLM4AD methods accept). Parameters a
method does not accept are dropped, so shared settings such as `num_samplers` are safe.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

# name: (module, class, profiler class or None, method-specific defaults)
METHODS = {
    "eoh":        ("llm4ad.method.eoh", "EoH", "EoHProfiler", dict(max_generations=None, pop_size=10)),
    "reevo":      ("llm4ad.method.reevo", "ReEvo", "ReEvoProfiler", dict(pop_size=10)),
    "funsearch":  ("llm4ad.method.funsearch", "FunSearch", None, dict()),
    "hillclimb":  ("llm4ad.method.hillclimb", "HillClimb", "HillClimbProfiler", dict()),   # (1+1)-EPS, PPSN 2024
    "randsample": ("llm4ad.method.randsample", "RandSample", "RandSampleProfiler", dict()),  # no-search control
    "mcts_ahd":   ("llm4ad.method.mcts_ahd", "MCTS_AHD", "MAProfiler", dict(pop_size=10)),   # experimental in v0
    "hsevo":      ("llm4ad.method.hsevo", "HSEvo", "HSEvoProfiler", dict()),                 # experimental in v0
}


def build(name: str, llm, task, max_evals: int, log_dir: Path, params: dict | None = None,
          num_samplers: int = 2, num_evaluators: int = 2):
    module, cls_name, prof_name, defaults = METHODS[name]
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)

    # A profiler is required: EoH and MCTS-AHD only count evaluations towards the budget when one is attached
    from llm4ad.tools.profiler import ProfilerBase
    prof_cls = getattr(mod, prof_name) if prof_name else ProfilerBase
    profiler = prof_cls(log_dir=str(log_dir), log_style="simple", create_random_path=False)

    kwargs = {**defaults, **(params or {}), "max_sample_nums": max_evals,
              "num_samplers": num_samplers, "num_evaluators": num_evaluators}
    accepted = inspect.signature(cls.__init__).parameters
    dropped = sorted(k for k in kwargs if k not in accepted)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(llm=llm, evaluation=task, profiler=profiler, **kwargs), kwargs, dropped
