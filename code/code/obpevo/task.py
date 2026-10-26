"""LLM4AD task adapter: exposes the framework-free evaluator (core.py) to every LLM4AD method.

`evaluate_program` runs inside LLM4AD's sandboxed child process. It only ever sees the
*training* instances, writes the full per-instance results for the candidate to
<run_dir>/per_instance/<code_hash>.json, and returns the mean score to the search method.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from llm4ad.base import Evaluation

from . import core

# Same template and task description as LLM4AD's built-in OBP task (comparability with EoH/FunSearch).
# The template body is Best Fit, so every method starts from the same seed heuristic.
TEMPLATE_LLM4AD = '''
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    return item - bins
'''
DESCRIPTION_LLM4AD = "Implement a function that returns the priority with which we want to add an item to each bin."

# Appended to every prompt, for every method, by LoggedLLM (config: evaluation.format_hint). Chat models often
# reply with markdown, helper functions or multi-line signatures; LLM4AD's parser then silently drops the candidate
# or evolves the wrong function. Identical text for all methods, so the comparison stays fair.
FORMAT_HINT = """Formatting rules for your reply:
- If you are asked to describe the algorithm, write that description as one sentence on a single line inside braces {}, before the code.
- Write the code as ONE complete Python function in a single ```python code block, keeping the function name requested above.
- Put the whole `def` line, including all arguments, on a single line starting at the beginning of the line.
- Use only numpy (imported as np) and Python built-ins. Do not define helper functions; put all logic inside the one function.
- Do not add example usage, tests, or explanations after the code.
- If you are asked only for hints or reflections, reply in plain text without code."""

# Optional variant for the "information exploitation" ablation (BLADE's third capability):
# describes the online setting and the training distribution explicitly.
DESCRIPTION_INFORMED = (
    "Online 1D bin packing: items arrive one at a time and must be placed immediately; the goal is to minimise "
    "the number of bins used. Implement a function that scores the bins the current item fits into "
    "(remaining capacities, including empty bins); the item goes to the highest-scoring bin. "
    "Training items are integers drawn from a Weibull(shape={shape}, scale={scale}) distribution, bin capacity {capacity}."
)


def code_hash(program_str: str) -> str:
    return hashlib.sha1(program_str.encode()).hexdigest()[:12]


class OBPTask(Evaluation):
    def __init__(
        self,
        train_instances: list[dict],
        run_dir: Path,
        mode: str = "funsearch",
        timeout_seconds: float = 60,
        task_description: str = DESCRIPTION_LLM4AD,
        template_program: str = TEMPLATE_LLM4AD,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            timeout_seconds=timeout_seconds,
        )
        self.instances, self.mode = train_instances, mode
        self.out_dir = Path(run_dir) / "per_instance"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_program(
        self,
        program_str: str,
        callable_func: callable,
        code_hash_: str | None = None,
        **kwargs,
    ):
        h = code_hash_ or code_hash(program_str)
        try:
            rows = core.evaluate(callable_func, self.instances, self.mode)
            rec, result = dict(
                status="ok", score=core.score(rows), rows=rows
            ), core.score(rows)
        except Exception as e:  # heuristic crashed or returned NaN / wrong shape
            rec, result = (
                dict(
                    status="error", error=f"{type(e).__name__}: {str(e)[:300]}", rows=[]
                ),
                None,
            )
        tmp = self.out_dir / f".{h}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(rec, default=float))
        os.replace(
            tmp, self.out_dir / f"{h}.json"
        )  # atomic: the parent never reads a half-written file
        return result
