"""Method-independent evaluation log (BLADE principle).

Every LLM4AD method evaluates candidates through `llm4ad.base.SecureEvaluator.evaluate_program`.
`EvalRecorder` wraps that one method for the duration of a run, so each evaluation, by any
method, is counted against the shared budget and logged identically to evals.csv, including
timeouts and crashes, which a child process cannot report itself.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

from llm4ad.base import SecureEvaluator

from .llm import BudgetExhausted
from .task import code_hash


class EvalRecorder:
    FIELDS = [
        "eval",
        "t_start",
        "wall_s",
        "code_hash",
        "status",
        "score",
        "train_excess",
        "code_chars",
        "code_lines",
        "error",
    ]

    def __init__(self, run_dir: Path, max_evals: int | None, timeout_s: float):
        self.run_dir, self.max_evals, self.timeout_s = (
            Path(run_dir),
            max_evals,
            timeout_s,
        )
        self.evals_csv, self.cands = (
            self.run_dir / "evals.csv",
            self.run_dir / "candidates.jsonl",
        )
        self.lock, self.n, self.exhausted, self._orig = threading.Lock(), 0, None, None

    def __enter__(self):
        with open(self.evals_csv, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)
        self.cands.write_text("")
        self._orig = orig = SecureEvaluator.evaluate_program
        rec = self

        def logged_evaluate(sec_self, program, **kwargs):
            with rec.lock:
                if rec.max_evals is not None and rec.n >= rec.max_evals:
                    rec.exhausted = "evals"
                    raise BudgetExhausted("evals")
                idx = rec.n = rec.n + 1
            src = str(program)
            h, t0 = code_hash(src), time.time()
            result = orig(
                sec_self, program, code_hash_=h, **kwargs
            )  # forks LLM4AD's sandboxed child
            wall = time.time() - t0
            rec._log(idx, t0, wall, h, src, result)
            return result

        SecureEvaluator.evaluate_program = logged_evaluate
        return self

    def __exit__(self, *exc):
        SecureEvaluator.evaluate_program = self._orig  # always restore the original
        return False

    def _log(self, idx, t0, wall, h, src, result):
        child = self.run_dir / "per_instance" / f"{h}.json"
        info = json.loads(child.read_text()) if child.exists() else {}
        if result is not None:
            status, err = "ok", ""
        elif info.get("status") == "error":
            status, err = "error", info.get(
                "error", ""
            )  # heuristic raised / NaN / wrong shape
        elif wall >= 0.95 * self.timeout_s:
            status, err = (
                "timeout",
                f"exceeded {self.timeout_s}s",
            )  # child killed by LLM4AD
        else:
            status, err = (
                "crash",
                "no result from child process",
            )  # e.g. syntax error in exec
        score = "" if result is None else float(result)  # full precision: best-program
        with self.lock:  # ranking and re-scoring stay exact
            with open(self.evals_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        idx,
                        round(t0, 3),
                        round(wall, 3),
                        h,
                        status,
                        score,
                        "" if result is None else -float(result),
                        len(src),
                        src.count("\n") + 1,
                        err[:200],
                    ]
                )
            with open(self.cands, "a") as f:
                f.write(
                    json.dumps(
                        dict(
                            eval=idx,
                            code_hash=h,
                            status=status,
                            score=result,
                            program=src,
                        )
                    )
                    + "\n"
                )
